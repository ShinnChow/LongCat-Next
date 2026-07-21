# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Generation task dataset with online packing for LongCat-Next FSDP SFT.

Input: text only -> Output: image (VQ tokens with depth CE loss)

Data flow:
1. Read raw JSONL (messages format, assistant response contains <longcat_img_start>path<longcat_img_end>)
2. Apply chat template + trainable markers
3. Tokenize text segments, preprocess images on CPU (pixel_values)
4. Image regions are trainable (loss_mask=1)
5. Pack multiple samples, apply causal LM shift
6. Model forward: ViT encode pixel_values -> VQ IDs (used as both embeddings and labels)
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
    IMG_GEN_START_TAG, IMG_GEN_END_TAG,
    IMG_TOKEN_SIZE_START, IMG_TOKEN_SIZE_END,
    IMG_TOKEN_SIZE_START_ID, IMG_TOKEN_SIZE_END_ID,
    SPECIAL_NO_LOSS_START, SPECIAL_NO_LOSS_END,
    tokenize_text_segment, apply_causal_lm_shift,
)
from .image_processing import (
    ImagePreprocessor, estimate_visual_tokens, build_image_placeholder_sequence,
)


logger = logging.getLogger(__name__)

class GeneratePackedDataset(IterableDataset):
    """Packed dataset for generation task SFT training.

    Similar to UnderstandPackedDataset, but:
    - Images are in the assistant response (trainable region)
    - loss_mask=1 for image tokens (they are the generation target)
    - Model forward produces VQ labels via ViT encoding for depth CE loss
    - Supports <img_newline> between rows
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        image_processor: ImagePreprocessor,
        seq_length: int = 8192,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        num_epochs: int = 3,
        no_packing: bool = False,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.seq_length = seq_length
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.num_epochs = num_epochs
        self.no_packing = no_packing

    def __iter__(self):
        for epoch in range(self.num_epochs):
            yield from self._iterate_epoch(epoch)

    def _iterate_epoch(self, epoch: int):
        """Iterate through one epoch with packing."""
        packed_ids = []
        packed_masks = []
        packed_pixel_values_list = []
        packed_image_grid_thw_list = []
        packed_image_positions = []
        sample_lengths = []
        current_length = 0

        sample_idx = 0
        for data_path in self.data_paths:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if sample_idx % self.world_size != self.rank:
                        sample_idx += 1
                        continue
                    sample_idx += 1

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    result = self._process_sample(data)
                    if result is None:
                        continue

                    sample_len = len(result["input_ids"])
                    if sample_len > self.seq_length:
                        continue

                    # Flush condition: either no_packing (1 sample per sequence)
                    # or current pack is full
                    flush = False
                    if self.no_packing and current_length > 0:
                        flush = True
                    elif current_length + sample_len > self.seq_length and current_length > 0:
                        flush = True

                    if flush:
                        yield self._finalize_pack(
                            packed_ids, packed_masks,
                            packed_pixel_values_list, packed_image_grid_thw_list,
                            packed_image_positions, sample_lengths,
                            data_iter_idx=sample_idx,
                        )
                        packed_ids = []
                        packed_masks = []
                        packed_pixel_values_list = []
                        packed_image_grid_thw_list = []
                        packed_image_positions = []
                        sample_lengths = []
                        current_length = 0

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

        if current_length > 0:
            yield self._finalize_pack(
                packed_ids, packed_masks,
                packed_pixel_values_list, packed_image_grid_thw_list,
                packed_image_positions, sample_lengths,
                data_iter_idx=sample_idx,
            )

    @staticmethod
    def _convert_offline_to_raw_messages(data: Dict) -> List[Dict]:
        """Convert offline-tokenized jsonl messages to raw format.

        Offline-tokenized jsonl (*.mm-tokenized.*.jsonl) has assistant messages
        in the format:
            <img_gen_token_start><img_token_size>H W</img_token_size>KEY<img_gen_token_end>
        while FSDP expects:
            <longcat_img_start>IMAGE_PATH<longcat_img_end>

        The actual image paths are stored in data["img_path"] list.
        """
        messages = data.get("messages", [])
        img_paths = data.get("img_path", [])

        converted = []
        img_idx = 0
        gen_pattern = re.compile(
            re.escape(IMG_GEN_START_TAG) + r".*?" + re.escape(IMG_GEN_END_TAG),
            re.DOTALL,
        )

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and IMG_GEN_START_TAG in content:
                # Replace each <img_gen_token_start>...<img_gen_token_end>
                # with <longcat_img_start>PATH<longcat_img_end>
                def _replace_gen_tag(match):
                    nonlocal img_idx
                    if img_idx < len(img_paths):
                        path = img_paths[img_idx]
                        img_idx += 1
                        return f"{IMG_START_TAG}{path}{IMG_END_TAG}"
                    return match.group(0)

                new_content = gen_pattern.sub(_replace_gen_tag, content)
                converted.append({"role": msg.get("role", "assistant"), "content": new_content})
            else:
                converted.append(msg)

        return converted

    def _process_sample(self, data: Dict) -> Optional[Dict]:
        """Process a single raw data sample for generation task.

        Supports both raw jsonl format (img_start/img_end tags) and
        offline-tokenized jsonl format (img_gen_token_start/img_gen_token_end tags).
        """
        try:
            messages = data.get("messages", [])
            if not messages:
                return None

            # Detect offline-tokenized format and convert to raw format
            has_offline_format = any(
                IMG_GEN_START_TAG in str(m.get("content", ""))
                for m in messages
            )
            if has_offline_format:
                messages = self._convert_offline_to_raw_messages(data)

            messages = unify_message_format(messages)
            marked_text = encode_with_trainable_markers(self.tokenizer, messages)
            if not marked_text:
                return None

            return self._tokenize_with_images(marked_text)
        except Exception as e:
            logger.warning(f"Failed to process generation sample: {e}")
            return None

    def _tokenize_with_images(self, text: str) -> Optional[Dict]:
        """Tokenize text with images for generation task.

        Key difference from understanding: images in trainable regions have
        loss_mask=1, and model forward will produce VQ labels for depth CE loss.
        """
        all_ids = []
        all_masks = []
        image_info = []

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

            self._process_text_with_images(
                part, trainable, all_ids, all_masks, image_info,
            )

        all_ids.append(torch.tensor([EOD_TOKEN_ID], dtype=torch.long))
        # EOD gets loss_mask=1: after the causal shift this trains the model to
        # predict the end-of-document token.
        all_masks.append(torch.tensor([1.0], dtype=torch.float32))

        if not all_ids:
            return None

        input_ids = torch.cat(all_ids)
        loss_mask = torch.cat(all_masks)

        # Mask special tokens, but preserve loss_mask for image content tokens
        # (IMG_PAD_ID) in trainable regions — they are the generation target.
        # special_token_no_loss handling is only applied to z-loss,
        # not the main loss. Image tokens have loss_mask=1 from the data pipeline.
        special_mask = (input_ids >= SPECIAL_NO_LOSS_START) & (input_ids < SPECIAL_NO_LOSS_END)
        # Exclude IMG_PAD_ID from special masking (image content tokens keep their loss_mask)
        special_mask = special_mask & (input_ids != IMG_PAD_ID)
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
        """Process text segment with images for generation."""
        img_pattern = re.compile(
            rf"{re.escape(IMG_START_TAG)}(.*?){re.escape(IMG_END_TAG)}", re.DOTALL
        )

        last_end = 0
        for match in img_pattern.finditer(text):
            prefix = text[last_end:match.start()]
            if prefix:
                result = tokenize_text_segment(self.tokenizer, prefix, trainable)
                all_ids.append(result["input_ids"])
                all_masks.append(result["loss_mask"])

            image_path = match.group(1).strip()
            try:
                preprocess_result = self.image_processor.preprocess(image_path)
                pixel_values = preprocess_result["pixel_values"]
                image_grid_thw = preprocess_result["image_grid_thw"]

                num_tokens, token_h, token_w = estimate_visual_tokens(image_grid_thw)

                # For generation: images ARE the target, so trainable=True in
                # trainable regions
                img_seq = build_image_placeholder_sequence(
                    num_tokens=num_tokens,
                    token_h=token_h,
                    token_w=token_w,
                    with_newline=True,   # generation uses newlines
                    trainable=trainable, # in trainable region -> loss_mask=1
                )

                # Prepend the image-size prefix "<...img_token_size>H W</...>".
                # The tag names are single added tokens in this vocab (ids
                # 131090/131091), so encode() maps each to one token id.
                _token_size_str = f"{IMG_TOKEN_SIZE_START}{token_h} {token_w}{IMG_TOKEN_SIZE_END}"
                _prefix_ids = self.tokenizer.encode(_token_size_str, add_special_tokens=False)
                assert _prefix_ids[0] == IMG_TOKEN_SIZE_START_ID and _prefix_ids[-1] == IMG_TOKEN_SIZE_END_ID, \
                    f"token_size prefix not encoded as single added_tokens: {_prefix_ids}"
                _prefix_ids = torch.tensor(_prefix_ids, dtype=torch.long)
                _prefix_mask = torch.zeros(len(_prefix_ids), dtype=torch.float32)

                start_pos = sum(len(t) for t in all_ids)
                end_pos = start_pos + len(_prefix_ids) + len(img_seq["input_ids"])

                image_info.append({
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "start": start_pos + len(_prefix_ids),
                    "end": end_pos,
                })

                all_ids.append(_prefix_ids)
                all_masks.append(_prefix_mask)
                all_ids.append(img_seq["input_ids"])
                all_masks.append(img_seq["loss_mask"])

            except Exception as e:
                # Image preprocessing failed — skip this entire sample
                raise RuntimeError(f"Failed to preprocess image {image_path}: {e}")

            last_end = match.end()

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
        """Finalize a packed sequence for generation task."""
        concat_ids = torch.cat(packed_ids)
        concat_masks = torch.cat(packed_masks)
        total_len = len(concat_ids)

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

        input_ids, labels, loss_mask = apply_causal_lm_shift(concat_ids, concat_masks)

        # cu_seqlens
        cu_seqlens = [0]
        cumulative = 0
        for sl in sample_lengths:
            cumulative += sl
            cu_seqlens.append(min(cumulative, self.seq_length))
        # Track real sample count before adding padding segment
        num_real_samples = len(sample_lengths)
        if cu_seqlens[-1] < self.seq_length:
            cu_seqlens.append(self.seq_length)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        # Build position_ids as a global arange(0, seq_length), NOT reset at
        # sub-sequence boundaries. With varlen attention (cu_seqlens) the attention
        # is already isolated per sample, so RoPE can safely use global positions.
        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        # Pixel data for ViT encoding in model forward
        if packed_pixel_values_list:
            all_pixel_values = torch.cat(packed_pixel_values_list, dim=0)
            all_image_grid_thw = torch.cat(packed_image_grid_thw_list, dim=0)
        else:
            all_pixel_values = torch.empty(0, dtype=torch.float32)
            all_image_grid_thw = torch.empty(0, 3, dtype=torch.long)

        # visual_mask: where IMG_PAD tokens are in input_ids — used for
        # replacing input embeddings with visual embeddings.
        visual_mask = (input_ids == IMG_PAD_ID)

        # loss_visual_mask: IMG_PAD / IMG_END positions in labels, used to select
        # hidden states for the depth CE loss. Because of the causal shift
        # (input_ids = concat[:-1], labels = concat[1:]) this mask is offset by one
        # from visual_mask. img_end positions predict a special end-of-image label
        # (codebook_size) and use zero embeddings for depth conditioning.
        loss_visual_mask = (labels == IMG_PAD_ID) | (labels == IMG_END_ID)

        # img_end_mask: marks img_end positions in labels, used to distinguish
        # IMG_PAD vs img_end inside depth CE loss. img_end positions get special
        # treatment: VQ label = codebook_size, embedding = zeros.
        img_end_mask = (labels == IMG_END_ID)

        # target_visual_mask: generation (trainable) image-pad positions in
        # input_ids. For the generate task EVERY image is a target, so this
        # equals visual_mask. Emitted so the training forward can uniformly
        # slice visual_ids for the depth loss without branching on task.
        target_visual_mask = visual_mask.clone()

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
