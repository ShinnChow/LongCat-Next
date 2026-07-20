# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Tokenization and loss mask generation for LongCat-Next SFT training.

Converts trainable-marked text into token IDs and a per-token loss mask: text
inside the trainable markers gets loss_mask=1, everything else gets 0.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from .chat_template import TRAINABLE_START, TRAINABLE_END


# LongCat-Next special token IDs
EOD_TOKEN_ID = 2          # </longcat_s>
IMG_START_ID = 131106     # <longcat_img_start>
IMG_END_ID = 131107       # <longcat_img_end>
IMG_PAD_ID = 131108       # <longcat_img_pad>
IMG_NEWLINE_ID = 131109   # <longcat_img_newline>

# Image generation tags (used in raw text, NOT token IDs)
IMG_START_TAG = "<longcat_img_start>"
IMG_END_TAG = "<longcat_img_end>"
IMG_GEN_START_TAG = "<img_gen_token_start>"
IMG_GEN_END_TAG = "<img_gen_token_end>"
# Image token-size prefix — these names must match the added tokens in the
# longcat-next vocab (ids 131090 / 131091) so that tokenizer.encode maps each
# tag to a single token id.
IMG_TOKEN_SIZE_START = "<longcat_img_token_size>"
IMG_TOKEN_SIZE_END = "</longcat_img_token_size>"
IMG_TOKEN_SIZE_START_ID = 131090
IMG_TOKEN_SIZE_END_ID = 131091

# Special tokens that should NOT have loss computed (multimodal special tokens)
SPECIAL_NO_LOSS_START = 131085
SPECIAL_NO_LOSS_END = 131125  # exclusive


def tokenize_text_segment(tokenizer, text: str, trainable: bool) -> Dict[str, torch.Tensor]:
    """Tokenize a text segment and generate corresponding loss mask.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Text to tokenize.
        trainable: Whether this segment should participate in loss computation.

    Returns:
        Dict with "input_ids" and "loss_mask" tensors.
    """
    if not text:
        return {"input_ids": torch.tensor([], dtype=torch.long),
                "loss_mask": torch.tensor([], dtype=torch.float32)}

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    loss_mask = torch.ones(len(token_ids), dtype=torch.float32) if trainable else \
                torch.zeros(len(token_ids), dtype=torch.float32)

    return {"input_ids": input_ids, "loss_mask": loss_mask}


def create_image_placeholder_tokens(
    num_tokens: int,
    token_h: int = 0,
    token_w: int = 0,
    with_newline: bool = False,
    trainable: bool = False,
) -> Dict[str, torch.Tensor]:
    """Create placeholder token IDs for image positions in the sequence.

    For understanding: images are in input (non-trainable, loss_mask=0)
    For generation: images are in output (trainable, loss_mask=1)

    Args:
        num_tokens: Number of image tokens (H * W for VQ grid).
        token_h: Height of the VQ token grid (for newline insertion).
        token_w: Width of the VQ token grid.
        with_newline: Whether to insert <img_newline> between rows.
        trainable: Whether image tokens participate in loss.

    Returns:
        Dict with "input_ids" and "loss_mask".
    """
    ids_list = [IMG_START_ID]

    if with_newline and token_h > 0 and token_w > 0:
        # Insert img_newline at the end of each row except the last
        for row in range(token_h):
            ids_list.extend([IMG_PAD_ID] * token_w)
            if row < token_h - 1:
                ids_list.append(IMG_NEWLINE_ID)
    else:
        ids_list.extend([IMG_PAD_ID] * num_tokens)

    ids_list.append(IMG_END_ID)

    input_ids = torch.tensor(ids_list, dtype=torch.long)
    mask_val = 1.0 if trainable else 0.0
    loss_mask = torch.full((len(ids_list),), mask_val, dtype=torch.float32)

    # img_start and img_end tokens should not have loss
    loss_mask[0] = 0.0   # <longcat_img_start>
    loss_mask[-1] = 0.0  # <longcat_img_end>

    return {"input_ids": input_ids, "loss_mask": loss_mask}


def tokenize_with_trainable_markers(
    tokenizer,
    text: str,
    task: str = "understand",
) -> Dict[str, object]:
    """Tokenize text with <trainable_start>/<trainable_end> markers.

    Processes the marked text into input_ids and loss_mask, handling:
    - Text segments (trainable and non-trainable)
    - Image regions (<longcat_img_start>...<longcat_img_end>) as placeholders
    - EOD token at the end

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Text with trainable markers and optional image tags.
        task: "understand" or "generate".

    Returns:
        Dict containing:
        - "input_ids": token IDs (LongTensor)
        - "loss_mask": loss mask (FloatTensor, same length as input_ids)
        - "image_paths": list of image file paths found in the text
        - "image_positions": list of (start, end) tuples for image placeholder ranges
        - "image_info": list of dicts with image metadata (for generation task)
    """
    all_ids = []
    all_masks = []
    image_paths = []
    image_positions = []  # (start_idx, end_idx) in the token sequence
    image_info = []       # metadata for generation task

    # Split by trainable markers
    parts = re.split(
        rf"({re.escape(TRAINABLE_START)}|{re.escape(TRAINABLE_END)})",
        text
    )

    trainable = False
    for part in parts:
        if part == TRAINABLE_START:
            trainable = True
            continue
        elif part == TRAINABLE_END:
            trainable = False
            continue
        elif not part:
            continue

        # Process this segment: may contain image tags
        _process_segment(
            tokenizer, part, trainable, task,
            all_ids, all_masks, image_paths, image_positions, image_info
        )

    # Add EOD token at the end
    all_ids.append(torch.tensor([EOD_TOKEN_ID], dtype=torch.long))
    all_masks.append(torch.tensor([1.0], dtype=torch.float32))  # EOD participates in loss (predict end-of-document)

    # Concatenate
    input_ids = torch.cat(all_ids) if all_ids else torch.tensor([], dtype=torch.long)
    loss_mask = torch.cat(all_masks) if all_masks else torch.tensor([], dtype=torch.float32)

    # Mask special tokens (131085:131125) — these should not have loss
    special_mask = (input_ids >= SPECIAL_NO_LOSS_START) & (input_ids < SPECIAL_NO_LOSS_END)
    loss_mask[special_mask] = 0.0

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "image_paths": image_paths,
        "image_positions": image_positions,
        "image_info": image_info,
    }


def _process_segment(
    tokenizer,
    text: str,
    trainable: bool,
    task: str,
    all_ids: list,
    all_masks: list,
    image_paths: list,
    image_positions: list,
    image_info: list,
):
    """Process a text segment, splitting out image regions.

    Image regions are identified by <longcat_img_start>...<longcat_img_end> tags.
    """
    # Pattern to find image regions
    img_pattern = re.compile(
        rf"{re.escape(IMG_START_TAG)}(.*?){re.escape(IMG_END_TAG)}",
        re.DOTALL
    )

    last_end = 0
    for match in img_pattern.finditer(text):
        # Tokenize text before the image tag
        prefix = text[last_end:match.start()]
        if prefix:
            result = tokenize_text_segment(tokenizer, prefix, trainable)
            all_ids.append(result["input_ids"])
            all_masks.append(result["loss_mask"])

        # Record image path
        image_content = match.group(1).strip()
        image_paths.append(image_content)

        # Create placeholder tokens (actual count will be set later by image processing)
        # For now, store a marker; the dataset will replace this with actual VQ tokens
        start_pos = sum(len(t) for t in all_ids)
        placeholder = create_image_placeholder_tokens(
            num_tokens=0,  # will be updated by image processing
            trainable=(task == "generate" and trainable),
        )
        # Store image info for later processing
        image_info.append({
            "path": image_content,
            "token_start_idx": start_pos,
            "trainable": trainable,
        })

        # We don't add placeholder tokens here — the dataset will handle that
        # after ViT encoding, when we know the actual token count
        image_positions.append((start_pos, -1))  # end will be filled later

        last_end = match.end()

    # Tokenize remaining text after the last image tag
    suffix = text[last_end:]
    if suffix:
        result = tokenize_text_segment(tokenizer, suffix, trainable)
        all_ids.append(result["input_ids"])
        all_masks.append(result["loss_mask"])


def apply_causal_lm_shift(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply standard causal language model shift.

    input  = tokens[:-1]  (predict next token)
    labels = tokens[1:]
    loss_mask = loss_mask[1:]  (aligned with labels)

    Args:
        input_ids: Full token sequence [seq_len+1].
        loss_mask: Loss mask [seq_len+1].

    Returns:
        Tuple of (input_ids, labels, loss_mask), each of length seq_len.
    """
    shifted_input = input_ids[:-1].clone()
    labels = input_ids[1:].clone()
    shifted_mask = loss_mask[1:].clone()

    return shifted_input, labels, shifted_mask
