# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Image preprocessing utilities for LongCat-Next SFT training.

Dataset-side preprocessing only (CPU): load image, resize/normalize via
HuggingFace processor, and return pixel_values + grid_thw tensors.

ViT encoding and VQ quantization happen in the model's forward pass,
so the ViT is part of the computation graph and its weights are managed
by FSDP (auto-saved in checkpoints, optionally trainable in the future).
"""

import io
import logging
import os
import signal
import time
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# Timeout for image loading (seconds). If a single image takes longer than
# this to load, it's likely a hung filesystem read or a corrupt file.
# We raise TimeoutError instead of hanging indefinitely, which would cause
# NCCL deadlock across all ranks.
_IMAGE_LOAD_TIMEOUT = 120  # 2 minutes


class _ImageLoadTimeout:
    """Context manager for image loading timeout using SIGALRM.

    Only works in the main thread (which is where DataLoader num_workers=0 runs).
    Falls back to no timeout if SIGALRM is not available (Windows, non-main thread).
    """
    def __init__(self, seconds: int, image_path: str = ""):
        self.seconds = seconds
        self.image_path = image_path
        self._old_handler = None
        self._can_use_alarm = hasattr(signal, 'SIGALRM')

    def _handler(self, signum, frame):
        raise TimeoutError(
            f"Image loading timed out after {self.seconds}s: {self.image_path[:200]}"
        )

    def __enter__(self):
        if self._can_use_alarm:
            try:
                self._old_handler = signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self.seconds)
            except (ValueError, OSError):
                # Not in main thread or signal not available
                self._can_use_alarm = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._can_use_alarm:
            signal.alarm(0)  # Cancel the alarm
            if self._old_handler is not None:
                try:
                    signal.signal(signal.SIGALRM, self._old_handler)
                except (ValueError, OSError):
                    pass
        return False  # Don't suppress exceptions

from .tokenize_utils import IMG_PAD_ID, IMG_START_ID, IMG_END_ID, IMG_NEWLINE_ID

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """CPU-only image preprocessing using Qwen2VLImageProcessor.

    Performs: image loading -> resize/normalize/patchify -> pixel_values tensor.
    Does NOT run ViT or VQ encoding (those happen in model forward).

    Uses the image_processor sub-component of LongcatNextProcessor, NOT the
    full processor (which requires a `text` argument with embedded image paths).
    """

    def __init__(self, processor):
        """Initialize with a HuggingFace processor.

        Args:
            processor: AutoProcessor (LongcatNextProcessor) from the model
                checkpoint. We extract its .image_processor sub-component
                (Qwen2VLImageProcessor) for direct image preprocessing.
        """
        # Extract the Qwen2VLImageProcessor sub-component.
        # LongcatNextProcessor.__call__ requires `text` as first arg,
        # but we only need image preprocessing (pixel_values + grid_thw).
        if hasattr(processor, "image_processor"):
            self.image_processor = processor.image_processor
        else:
            # Fallback: assume processor IS the image processor
            self.image_processor = processor

    def preprocess(self, image_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Preprocess a single image for ViT input.

        Args:
            image_path: Local image file path.

        Returns:
            Dict containing:
            - "pixel_values": [num_patches, C, patch_H, patch_W] float tensor
            - "image_grid_thw": [1, 3] grid dimensions (t, h, w)
            Returns None if loading fails (warning logged).

        Raises:
            RuntimeError: If image loading or preprocessing fails.
        """
        # --- Load image (with timeout to prevent NCCL deadlock on hung reads) ---
        try:
            with _ImageLoadTimeout(_IMAGE_LOAD_TIMEOUT, image_path):
                if not os.path.exists(image_path):
                    logger.warning(f"[IMAGE_NOT_FOUND] Local image does not exist: "
                                   f"{image_path[:200]}")
                    raise FileNotFoundError(f"Image not found: {image_path}")
                with open(image_path, "rb") as f:
                    image = Image.open(io.BytesIO(f.read())).convert("RGB")
        except TimeoutError as e:
            logger.error(f"[IMAGE_TIMEOUT] {e}")
            raise RuntimeError(f"Image loading timed out: {image_path[:200]}")
        except Exception as e:
            logger.warning(f"[IMAGE_LOAD_FAIL] Failed to load image: "
                           f"path={image_path[:200]}, error={e}")
            raise RuntimeError(f"Failed to load image {image_path[:200]}: {e}")

        # --- Preprocess with Qwen2VLImageProcessor ---
        try:
            inputs = self.image_processor(images=[image], return_tensors="pt")
        except Exception as e:
            logger.warning(f"[IMAGE_PREPROCESS_FAIL] Processor failed: "
                           f"path={image_path[:200]}, size={image.size}, error={e}")
            raise RuntimeError(f"Failed to preprocess image {image_path[:200]}: {e}")

        return {
            "pixel_values": inputs["pixel_values"],       # [num_patches, C, pH, pW]
            "image_grid_thw": inputs["image_grid_thw"],    # [1, 3]
        }


def estimate_visual_tokens(grid_thw: torch.Tensor) -> Tuple[int, int, int]:
    """Estimate the number of visual tokens from grid dimensions.

    After ViT + VQ bridge, the spatial dimensions are typically compressed
    by the merge factor. This returns the token grid (t, h, w) and total count.

    Args:
        grid_thw: [1, 3] or [3] tensor of (t, h, w) grid dimensions.

    Returns:
        Tuple of (num_tokens, token_h, token_w).
    """
    if grid_thw.dim() == 2:
        grid_thw = grid_thw[0]
    t, h, w = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    # After ViT spatial merge (factor=2), token grid is (h/2, w/2)
    token_h = h // 2
    token_w = w // 2
    num_tokens = t * token_h * token_w
    return num_tokens, token_h, token_w


def build_image_placeholder_sequence(
    num_tokens: int,
    token_h: int = 0,
    token_w: int = 0,
    with_newline: bool = True,
    trainable: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build placeholder token sequence for an image region.

    Creates:
    - input_ids: [img_start, pad*N (with optional newlines), img_end]
    - loss_mask: matching binary mask

    This is used during tokenization to reserve positions in the sequence
    for image tokens. The actual visual embeddings are filled in during
    model forward.

    Args:
        num_tokens: Total number of visual tokens for this image.
        token_h: Height of the visual token grid.
        token_w: Width of the visual token grid.
        with_newline: Insert <img_newline> between rows.
        trainable: Whether image tokens should have loss computed.

    Returns:
        Dict with input_ids, loss_mask, and num_tokens.
    """
    ids_list = [IMG_START_ID]
    mask_list = [0.0]  # img_start: no loss

    if with_newline and token_h > 0 and token_w > 0:
        for row in range(token_h):
            ids_list.extend([IMG_PAD_ID] * token_w)
            mask_list.extend([1.0 if trainable else 0.0] * token_w)
            if row < token_h - 1:
                ids_list.append(IMG_NEWLINE_ID)
                mask_list.append(0.0)  # newline: no loss
    else:
        ids_list.extend([IMG_PAD_ID] * num_tokens)
        mask_list.extend([1.0 if trainable else 0.0] * num_tokens)

    ids_list.append(IMG_END_ID)
    mask_list.append(0.0)  # img_end: no loss

    input_ids = torch.tensor(ids_list, dtype=torch.long)
    loss_mask = torch.tensor(mask_list, dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "num_tokens": num_tokens,
    }
