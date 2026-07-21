# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Understanding task dataset with online packing for LongCat-Next FSDP SFT.

Input: images + text -> Output: text only (standard CE loss)

Data flow:
1. Read raw JSONL (single pre-merged+shuffled file)
2. Shard by line_idx % world_size == rank (uniform distribution)
3. Apply chat template + trainable markers
4. Tokenize text segments, preprocess images on CPU (pixel_values)
5. Pack multiple samples into one sequence (varlen attention)
6. Apply causal LM shift
7. Model forward: ViT encode pixel_values -> VQ IDs -> embeddings

Data sharding:
  Previously used round-robin across N files + sample_idx % world_size.
  When gcd(world_size, N) > 1 this causes systematic bias (e.g. even ranks
  only see even-indexed files). Now uses a single pre-merged+shuffled JSONL
  with simple line_idx % world_size == rank sharding — perfectly uniform.
"""

import json
import re
import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import IterableDataset

from .chat_template import (
    TRAINABLE_START, TRAINABLE_END,
    unify_message_format, encode_with_trainable_markers,
)
from .tokenize_utils import (
    EOD_TOKEN_ID, IMG_PAD_ID, IMG_START_ID, IMG_END_ID, IMG_NEWLINE_ID,
    IMG_START_TAG, IMG_END_TAG,
    SPECIAL_NO_LOSS_START, SPECIAL_NO_LOSS_END,
    tokenize_text_segment, apply_causal_lm_shift,
)
from .image_processing import (
    ImagePreprocessor, estimate_visual_tokens, build_image_placeholder_sequence,
)


logger = logging.getLogger(__name__)

class UnderstandPackedDataset(IterableDataset):
    """Packed dataset for understanding task SFT training.

    Reads a single pre-merged+shuffled JSONL file, shards by line index,
    performs CPU-only image preprocessing, and packs multiple samples into
    fixed-length sequences with cu_seqlens for varlen attention. ViT
    encoding happens in the model's forward pass.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        image_processor: ImagePreprocessor,
        seq_length: int = 65536,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        num_epochs: int = 1,
        no_packing: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data_paths: List of JSONL file paths. If a single file is given,
                it's used directly. If multiple files are given, they are
                expected to have been pre-merged (the caller is responsible
                for calling merge_shuffle before constructing this dataset).
            tokenizer: HuggingFace tokenizer.
            image_processor: ImagePreprocessor for CPU-only preprocessing.
            seq_length: Maximum packed sequence length.
            seed: Random seed for shuffling.
            rank: Current process rank (for data sharding).
            world_size: Total number of processes.
            num_epochs: Number of training epochs.
            no_packing: If True, each sample occupies its own sequence (no packing).
        """
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_length = seq_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.no_packing = no_packing

        # ── Stateful resume support ──
        # Simplified for single-file sequential reading.
        self._state_epoch = 0              # current epoch index
        self._state_line_idx = 0           # global line index in merged file
        self._state_byte_offset = 0        # byte offset for seek-based resume
        self._state_pack_count = 0         # packs yielded so far (total across epochs)
        self._resume_state = None          # set by load_state_dict, consumed by __iter__

    # ── Stateful protocol for StatefulDataLoader ──

    def state_dict(self) -> Dict:
        """Return current iteration state for checkpoint resume."""
        return {
            "epoch": self._state_epoch,
            "line_idx": self._state_line_idx,
            "byte_offset": self._state_byte_offset,
            "pack_count": self._state_pack_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Restore iteration state from checkpoint."""
        self._resume_state = {
            "epoch": state_dict["epoch"],
            "line_idx": state_dict.get("line_idx", state_dict.get("sample_idx", 0)),
            "byte_offset": state_dict.get("byte_offset", 0),
            "pack_count": state_dict["pack_count"],
        }

    def __iter__(self):
        """Yield packed training samples.

        Iterates over the merged JSONL file sequentially. Each line is
        assigned to rank = line_idx % world_size. Maintains cumulative
        line_idx across epochs for accurate epoch tracking.
        """
        resume = self._resume_state
        self._resume_state = None  # consume once

        start_epoch = 0
        start_line_idx = 0
        start_byte_offset = 0
        if resume is not None:
            start_epoch = resume["epoch"]
            start_line_idx = resume["line_idx"]
            start_byte_offset = resume["byte_offset"]
            self._state_pack_count = resume["pack_count"]
            if self.rank == 0:
                logger.info(f"[STATEFUL_RESUME] Resuming from epoch={start_epoch}, "
                      f"line_idx={start_line_idx}, byte_offset={start_byte_offset}, "
                      f"pack_count={self._state_pack_count}")

        for epoch in range(start_epoch, self.num_epochs):
            byte_offset = start_byte_offset if epoch == start_epoch else 0
            line_idx_start = start_line_idx if epoch == start_epoch else 0

            for pack in self._iterate_epoch(epoch, line_idx_start, byte_offset):
                yield pack

    def _iterate_epoch(self, epoch: int, start_line_idx: int = 0,
                       start_byte_offset: int = 0):
        """Iterate through one epoch of data with packing."""
        if self.no_packing:
            yield from self._iterate_epoch_no_packing(
                epoch, start_line_idx, start_byte_offset)
            return

        # Current packing buffer
        packed_ids = []
        packed_masks = []
        packed_pixel_values_list = []     # list of pixel_values tensors per image
        packed_image_grid_thw_list = []   # list of grid_thw per image
        packed_image_positions = []       # (start, end) of img_pad tokens in packed seq
        sample_lengths = []
        current_length = 0

        pack_count = 0  # number of packed batches yielded so far
        processed_count = 0  # number of samples processed by this rank
        line_idx = start_line_idx  # track last seen line_idx for final pack

        # Update stateful epoch tracker
        self._state_epoch = epoch

        import time as _time
        for line_idx, data, pre_byte_offset, post_byte_offset in \
                self._sequential_sample_iter(start_line_idx, start_byte_offset):
            # Shard data across workers: simple modulo on line index
            if line_idx % self.world_size != self.rank:
                continue

            # Save pre-iteration state for checkpoint
            pre_line_idx = line_idx
            pre_offset = pre_byte_offset

            result = self._process_sample(data)
            processed_count += 1

            if result is None:
                continue

            sample_len = len(result["input_ids"])

            # Skip samples that are too long
            if sample_len > self.seq_length:
                continue

            # If adding this sample exceeds seq_length, yield current pack
            if current_length + sample_len > self.seq_length and current_length > 0:
                pack_count += 1
                self._state_pack_count += 1
                # Save state using PRE-read position. On resume, the
                # triggering sample will be re-read and re-processed.
                self._state_line_idx = pre_line_idx
                self._state_byte_offset = pre_offset

                yield self._finalize_pack(
                    packed_ids, packed_masks,
                    packed_pixel_values_list, packed_image_grid_thw_list,
                    packed_image_positions, sample_lengths,
                    data_iter_idx=line_idx + 1,
                )
                packed_ids = []
                packed_masks = []
                packed_pixel_values_list = []
                packed_image_grid_thw_list = []
                packed_image_positions = []
                sample_lengths = []
                current_length = 0

            # Track image positions with offset in the packed sequence
            offset = current_length
            for img_info in result.get("image_info", []):
                packed_pixel_values_list.append(img_info["pixel_values"])
                packed_image_grid_thw_list.append(img_info["image_grid_thw"])
                packed_image_positions.append((
                    offset + img_info["start"],
                    offset + img_info["end"],
                ))

            packed_ids.append(result["input_ids"])
            packed_masks.append(result["loss_mask"])
            sample_lengths.append(sample_len)
            current_length += sample_len

        # Yield remaining samples
        if current_length > 0:
            self._state_pack_count += 1
            self._state_line_idx = line_idx + 1
            self._state_byte_offset = 0  # epoch done
            yield self._finalize_pack(
                packed_ids, packed_masks,
                packed_pixel_values_list, packed_image_grid_thw_list,
                packed_image_positions, sample_lengths,
                data_iter_idx=line_idx + 1,
            )

    def _sequential_sample_iter(self, start_line_idx: int = 0,
                                 start_byte_offset: int = 0):
        """Yield (line_idx, data, pre_byte_offset, post_byte_offset) from all data files.

        Reads files sequentially (for a single merged file, this is just one
        file). For backward compatibility with multiple files, reads them
        in order (concatenated). Uses simple line_idx based sharding.

        When start_byte_offset > 0, seeks to that position for resume.
        """
        line_idx = start_line_idx

        for file_num, path in enumerate(self.data_paths):
            try:
                fh = open(path, "r", encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not open {path}: {e}")
                continue

            # Seek to saved position (only for the first file on resume)
            if file_num == 0 and start_byte_offset > 0:
                fh.seek(start_byte_offset)
                if self.rank == 0:
                    logger.info(f"[RESUME_SEEK] Seeked to byte_offset={start_byte_offset} "
                          f"in {path}, starting from line_idx={start_line_idx}")

            while True:
                pre_offset = fh.tell()
                line = fh.readline()
                if not line:
                    break  # EOF
                post_offset = fh.tell()

                line = line.strip()
                if not line:
                    line_idx += 1
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    line_idx += 1
                    continue

                yield line_idx, data, pre_offset, post_offset
                line_idx += 1

            fh.close()

    def _iterate_epoch_no_packing(self, epoch: int, start_line_idx: int = 0,
                                   start_byte_offset: int = 0):
        """Iterate through one epoch, yielding each sample as a standalone sequence."""
        # Update stateful epoch tracker
        self._state_epoch = epoch

        for line_idx, data, pre_offset, post_offset in \
                self._sequential_sample_iter(start_line_idx, start_byte_offset):
            # Shard data across workers
            if line_idx % self.world_size != self.rank:
                continue

            # Process single sample
            result = self._process_sample(data)
            if result is None:
                continue

            sample_len = len(result["input_ids"])

            # Skip samples that are too long
            if sample_len > self.seq_length:
                continue

            # For no_packing: save POST-read state (sample already yielded)
            self._state_pack_count += 1
            self._state_line_idx = line_idx + 1
            self._state_byte_offset = post_offset

            # Each sample is its own pack (no packing)
            yield self._finalize_pack(
                [result["input_ids"]],
                [result["loss_mask"]],
                [info["pixel_values"] for info in result.get("image_info", [])],
                [info["image_grid_thw"] for info in result.get("image_info", [])],
                [(info["start"], info["end"]) for info in result.get("image_info", [])],
                [sample_len],
                data_iter_idx=line_idx + 1,
            )

    def _process_sample(self, data: Dict) -> Optional[Dict]:
        """Process a single raw data sample.

        Args:
            data: Raw JSON data with "messages" key.

        Returns:
            Dict with input_ids, loss_mask, image_info; or None on failure.
        """
        try:
            messages = data.get("messages", [])
            if not messages:
                return None

            messages = unify_message_format(messages)

            # Apply chat template with trainable markers
            marked_text = encode_with_trainable_markers(self.tokenizer, messages)
            if not marked_text:
                return None

            # Process the marked text: tokenize text parts, preprocess images
            return self._tokenize_with_images(marked_text)

        except Exception as e:
            logger.warning(f"Failed to process sample: {e}")
            return None

    def _tokenize_with_images(self, text: str) -> Optional[Dict]:
        """Tokenize text with trainable markers and image placeholders.

        Images are preprocessed on CPU (pixel_values only). ViT encoding
        happens later in the model forward pass.
        """
        all_ids = []
        all_masks = []
        image_info = []

        # Split by trainable markers first
        parts = re.split(
            rf"({re.escape(TRAINABLE_START)}|{re.escape(TRAINABLE_END)})",
            text,
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

            # Process segment that may contain image tags
            self._process_text_with_images(
                part, trainable, all_ids, all_masks, image_info,
            )

        # Add EOD token (mask=1: model should predict sequence end)
        # After causal shift, this becomes label=EOD with loss_mask=1 at the last content position
        all_ids.append(torch.tensor([EOD_TOKEN_ID], dtype=torch.long))
        all_masks.append(torch.tensor([1.0], dtype=torch.float32))

        if not all_ids:
            return None

        input_ids = torch.cat(all_ids)
        loss_mask = torch.cat(all_masks)

        # Mask special tokens (131085:131125)
        special_mask = (input_ids >= SPECIAL_NO_LOSS_START) & (input_ids < SPECIAL_NO_LOSS_END)
        loss_mask[special_mask] = 0.0

        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "image_info": image_info,
        }

    def _process_text_with_images(
        self,
        text: str,
        trainable: bool,
        all_ids: list,
        all_masks: list,
        image_info: list,
    ):
        """Process a text segment, preprocessing images found within it."""
        img_pattern = re.compile(
            rf"{re.escape(IMG_START_TAG)}(.*?){re.escape(IMG_END_TAG)}", re.DOTALL
        )

        last_end = 0
        for match in img_pattern.finditer(text):
            # Tokenize text before image
            prefix = text[last_end:match.start()]
            if prefix:
                result = tokenize_text_segment(self.tokenizer, prefix, trainable)
                all_ids.append(result["input_ids"])
                all_masks.append(result["loss_mask"])

            # Preprocess image on CPU (no ViT encoding)
            image_path = match.group(1).strip()
            try:
                preprocess_result = self.image_processor.preprocess(image_path)
                pixel_values = preprocess_result["pixel_values"]
                image_grid_thw = preprocess_result["image_grid_thw"]

                # Estimate token count from grid dimensions
                num_tokens, token_h, token_w = estimate_visual_tokens(image_grid_thw)

                # Build placeholder sequence (img_start + img_pad*N + img_end)
                img_seq = build_image_placeholder_sequence(
                    num_tokens=num_tokens,
                    token_h=token_h,
                    token_w=token_w,
                    with_newline=False,  # understanding: no newline
                    trainable=False,     # understanding: images are input, no loss
                )

                # Record image position and pixel data for model forward
                start_pos = sum(len(t) for t in all_ids)
                end_pos = start_pos + len(img_seq["input_ids"])

                image_info.append({
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "start": start_pos,
                    "end": end_pos,
                })

                all_ids.append(img_seq["input_ids"])
                all_masks.append(img_seq["loss_mask"])

            except Exception as e:
                # Image preprocessing failed — skip this entire sample to avoid
                # corrupted data (e.g., tokenizing raw image paths as text).
                raise RuntimeError(f"Failed to preprocess image {image_path}: {e}")

            last_end = match.end()

        # Remaining text after last image
        suffix = text[last_end:]
        if suffix:
            result = tokenize_text_segment(self.tokenizer, suffix, trainable)
            all_ids.append(result["input_ids"])
            all_masks.append(result["loss_mask"])

    def _finalize_pack(
        self,
        packed_ids: List[torch.Tensor],
        packed_masks: List[torch.Tensor],
        packed_pixel_values_list: List[torch.Tensor],
        packed_image_grid_thw_list: List[torch.Tensor],
        packed_image_positions: List,
        sample_lengths: List[int],
        data_iter_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Finalize a packed sequence: pad, shift, and build cu_seqlens."""
        # Concatenate all samples
        concat_ids = torch.cat(packed_ids)
        concat_masks = torch.cat(packed_masks)
        total_len = len(concat_ids)

        # Pad to seq_length + 1 (need +1 for causal LM shift)
        pad_len = self.seq_length + 1 - total_len
        if pad_len > 0:
            concat_ids = torch.cat([
                concat_ids,
                torch.full((pad_len,), EOD_TOKEN_ID, dtype=torch.long)
            ])
            concat_masks = torch.cat([
                concat_masks,
                torch.zeros(pad_len, dtype=torch.float32)
            ])
        else:
            concat_ids = concat_ids[: self.seq_length + 1]
            concat_masks = concat_masks[: self.seq_length + 1]

        # Causal LM shift
        input_ids, labels, loss_mask = apply_causal_lm_shift(concat_ids, concat_masks)

        # Build cu_seqlens for varlen attention
        cu_seqlens = [0]
        cumulative = 0
        for sl in sample_lengths:
            cumulative += sl
            cu_seqlens.append(min(cumulative, self.seq_length))
        # Track real sample count before adding padding segment
        num_real_samples = len(sample_lengths)
        # Ensure last entry covers full seq_length (padding region)
        if cu_seqlens[-1] < self.seq_length:
            cu_seqlens.append(self.seq_length)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        # Build position_ids as a global arange(0, seq_length), NOT reset at
        # sub-sequence boundaries. With varlen attention (cu_seqlens) the attention
        # is already isolated per sample, so RoPE can safely use global positions.
        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        # Collect pixel_values and grid_thw for all images in this pack
        # These are passed to model forward for ViT encoding
        if packed_pixel_values_list:
            all_pixel_values = torch.cat(packed_pixel_values_list, dim=0)
            all_image_grid_thw = torch.cat(packed_image_grid_thw_list, dim=0)
        else:
            all_pixel_values = torch.empty(0, dtype=torch.float32)
            all_image_grid_thw = torch.empty(0, 3, dtype=torch.long)

        # Build visual position mask (where img_pad tokens are in input_ids)
        visual_mask = (input_ids == IMG_PAD_ID)

        # Understand task: no image-loss positions. Labels may contain
        # IMG_PAD_ID/IMG_END_ID from the placeholder sequence, but they must not
        # contribute to the loss — otherwise the loss would infer image positions
        # from labels and add a depth CE term for the understand task.
        loss_visual_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        img_end_mask = torch.zeros_like(loss_mask, dtype=torch.bool)

        # target_visual_mask: generation (trainable) image-pad positions. The
        # understand task has NO generation targets (all images are inputs), so
        # this is all-False. Emitted so the training forward can uniformly slice
        # visual_ids for the depth loss without branching on task.
        target_visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "cu_seqlens": cu_seqlens,
            "num_real_samples": num_real_samples,
            "pixel_values": all_pixel_values,
            "image_grid_thw": all_image_grid_thw,
            "visual_mask": visual_mask,
            "loss_visual_mask": loss_visual_mask,
            "img_end_mask": img_end_mask,
            "target_visual_mask": target_visual_mask,
            "data_iter_idx": data_iter_idx,
        }
