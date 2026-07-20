# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unified (understand + generate) packed dataset for LongCat-Next FSDP SFT.

This dataset enables *mixed training*: a single packed sequence may contain both
understanding samples (image -> text, CE loss) and generation samples
(text -> image, depth CE loss). Documents of different modalities are packed into
the same sequence and routed per-token by the loss masks.

Design principles:
  * Per-image tokenisation is **reused verbatim** from
    ``UnderstandPackedDataset`` and ``GeneratePackedDataset``. We only borrow their
    ``_process_sample`` (and the generate offline->raw conversion), so the
    placeholder / loss_mask / newline / token_size logic stays consistent.
  * Sample type is decided **per sample**: a sample is a *generation* sample iff
    an image appears inside a trainable (assistant) region; otherwise it is an
    *understanding* (or pure-text) sample.
  * The mixed pack carries a per-image ``is_target`` flag so that
    ``loss_visual_mask`` / ``img_end_mask`` / ``target_visual_mask`` mark ONLY the
    generation images. Understanding images keep loss_mask=0 and never enter the
    depth-CE path, even though their labels contain IMG_PAD/IMG_END ids.

The output dict is identical in shape to the two single-task datasets, plus one
extra key ``target_visual_mask`` (trainable image-pad positions in input_ids),
consumed by the training forward to slice ``visual_ids`` for the depth loss.
"""

from typing import Dict, List

import torch
from torch.utils.data import IterableDataset

from .chat_template import unify_message_format
from .tokenize_utils import (
    EOD_TOKEN_ID, IMG_PAD_ID, IMG_END_ID,
    IMG_START_TAG,
    IMG_GEN_START_TAG,
    apply_causal_lm_shift,
)
from .image_processing import ImagePreprocessor
from .understand_dataset import UnderstandPackedDataset
from .generate_dataset import GeneratePackedDataset


class UnifyPackedDataset(IterableDataset):
    """Packed dataset interleaving understand and generate samples.

    Reads a single pre-merged+shuffled JSONL file (already containing both
    understand and generate samples), shards by line index, and packs samples of
    *either* type into fixed-length sequences. Each image's placeholder is built
    by the corresponding aligned single-task processor.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer,
        image_processor: ImagePreprocessor,
        seq_length: int = 16384,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        num_epochs: int = 1,
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

        # Sub-processors: we only use their `_process_sample` (per-image
        # tokenisation), never their iteration loops. This guarantees the
        # mixed dataset reuses the *exact* aligned per-image logic.
        self._und = UnderstandPackedDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=seq_length,
            seed=seed,
            rank=rank,
            world_size=world_size,
            num_epochs=1,
            no_packing=no_packing,
        )
        self._gen = GeneratePackedDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_processor=image_processor,
            seq_length=seq_length,
            seed=seed,
            rank=rank,
            world_size=world_size,
            num_epochs=1,
            no_packing=no_packing,
        )

        # ── Stateful resume support (mirrors UnderstandPackedDataset) ──
        self._state_epoch = 0
        self._state_line_idx = 0
        self._state_byte_offset = 0
        self._state_pack_count = 0
        self._resume_state = None

    # ── Stateful protocol for StatefulDataLoader ──

    def state_dict(self) -> Dict:
        return {
            "epoch": self._state_epoch,
            "line_idx": self._state_line_idx,
            "byte_offset": self._state_byte_offset,
            "pack_count": self._state_pack_count,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self._resume_state = {
            "epoch": state_dict["epoch"],
            "line_idx": state_dict.get("line_idx", state_dict.get("sample_idx", 0)),
            "byte_offset": state_dict.get("byte_offset", 0),
            "pack_count": state_dict["pack_count"],
        }

    # ── Sample type detection ──

    @staticmethod
    def _is_generate_sample(messages: List[Dict]) -> bool:
        """A sample is a generation sample iff an image appears in a trainable
        (assistant) region.

        Detects both formats:
          * offline-tokenized: ``<img_gen_token_start>`` anywhere (always generate).
          * raw: ``<longcat_img_start>`` inside an assistant message content.
        """
        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            if IMG_GEN_START_TAG in content:
                return True
            if m.get("role") == "assistant" and IMG_START_TAG in content:
                return True
        return False

    def _process_sample(self, data: Dict):
        """Route a raw sample to the matching aligned processor.

        Returns ``(result, is_target)`` where ``result`` is the
        ``{input_ids, loss_mask, image_info}`` dict from the chosen processor and
        ``is_target`` marks whether the sample's images are generation targets.
        Returns ``(None, False)`` on failure / empty.
        """
        messages = data.get("messages", [])
        if not messages:
            return None, False
        try:
            uni = unify_message_format(messages)
        except Exception:
            uni = messages
        is_gen = self._is_generate_sample(uni)
        if is_gen:
            result = self._gen._process_sample(data)
        else:
            result = self._und._process_sample(data)
        return result, is_gen

    # ── Iteration ──

    def __iter__(self):
        resume = self._resume_state
        self._resume_state = None

        start_epoch = 0
        start_line_idx = 0
        start_byte_offset = 0
        if resume is not None:
            start_epoch = resume["epoch"]
            start_line_idx = resume["line_idx"]
            start_byte_offset = resume["byte_offset"]
            self._state_pack_count = resume["pack_count"]

        for epoch in range(start_epoch, self.num_epochs):
            byte_offset = start_byte_offset if epoch == start_epoch else 0
            line_idx_start = start_line_idx if epoch == start_epoch else 0
            yield from self._iterate_epoch(epoch, line_idx_start, byte_offset)

    def _iterate_epoch(self, epoch: int, start_line_idx: int = 0,
                       start_byte_offset: int = 0):
        if self.no_packing:
            yield from self._iterate_epoch_no_packing(
                epoch, start_line_idx, start_byte_offset)
            return

        packed_ids = []
        packed_masks = []
        packed_pixel_values_list = []
        packed_image_grid_thw_list = []
        packed_image_is_target = []   # per-image bool, parallel to pixel_values_list
        packed_image_positions = []   # (start, end) of each image's pad span in pack
        sample_lengths = []
        current_length = 0

        pack_count = 0
        line_idx = start_line_idx
        self._state_epoch = epoch

        for line_idx, data, pre_byte_offset, post_byte_offset in \
                self._und._sequential_sample_iter(start_line_idx, start_byte_offset):
            if line_idx % self.world_size != self.rank:
                continue

            pre_line_idx = line_idx
            pre_offset = pre_byte_offset

            result, is_target = self._process_sample(data)
            if result is None:
                continue

            sample_len = len(result["input_ids"])
            if sample_len > self.seq_length:
                continue

            if current_length + sample_len > self.seq_length and current_length > 0:
                pack_count += 1
                self._state_pack_count += 1
                self._state_line_idx = pre_line_idx
                self._state_byte_offset = pre_offset
                yield self._finalize_pack(
                    packed_ids, packed_masks,
                    packed_pixel_values_list, packed_image_grid_thw_list,
                    packed_image_positions, packed_image_is_target,
                    sample_lengths, data_iter_idx=line_idx + 1,
                )
                packed_ids = []
                packed_masks = []
                packed_pixel_values_list = []
                packed_image_grid_thw_list = []
                packed_image_is_target = []
                packed_image_positions = []
                sample_lengths = []
                current_length = 0

            offset = current_length
            for img_info in result.get("image_info", []):
                packed_pixel_values_list.append(img_info["pixel_values"])
                packed_image_grid_thw_list.append(img_info["image_grid_thw"])
                packed_image_is_target.append(is_target)
                packed_image_positions.append((
                    offset + img_info["start"],
                    offset + img_info["end"],
                ))

            packed_ids.append(result["input_ids"])
            packed_masks.append(result["loss_mask"])
            sample_lengths.append(sample_len)
            current_length += sample_len

        if current_length > 0:
            self._state_pack_count += 1
            self._state_line_idx = line_idx + 1
            self._state_byte_offset = 0
            yield self._finalize_pack(
                packed_ids, packed_masks,
                packed_pixel_values_list, packed_image_grid_thw_list,
                packed_image_positions, packed_image_is_target,
                sample_lengths, data_iter_idx=line_idx + 1,
            )

    def _iterate_epoch_no_packing(self, epoch: int, start_line_idx: int = 0,
                                  start_byte_offset: int = 0):
        self._state_epoch = epoch
        for line_idx, data, pre_offset, post_offset in \
                self._und._sequential_sample_iter(start_line_idx, start_byte_offset):
            if line_idx % self.world_size != self.rank:
                continue
            result, is_target = self._process_sample(data)
            if result is None:
                continue
            sample_len = len(result["input_ids"])
            if sample_len > self.seq_length:
                continue
            self._state_pack_count += 1
            self._state_line_idx = line_idx + 1
            self._state_byte_offset = post_offset

            image_info = result.get("image_info", [])
            yield self._finalize_pack(
                [result["input_ids"]],
                [result["loss_mask"]],
                [info["pixel_values"] for info in image_info],
                [info["image_grid_thw"] for info in image_info],
                [(info["start"], info["end"]) for info in image_info],
                [is_target for _ in image_info],
                [sample_len],
                data_iter_idx=line_idx + 1,
            )

    # ── Pack finalisation ──

    def _finalize_pack(
        self,
        packed_ids: List[torch.Tensor],
        packed_masks: List[torch.Tensor],
        packed_pixel_values_list: List[torch.Tensor],
        packed_image_grid_thw_list: List[torch.Tensor],
        packed_image_positions: List,
        packed_image_is_target: List[bool],
        sample_lengths: List[int],
        data_iter_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Finalize a packed sequence: pad, shift, build cu_seqlens and the
        per-modality masks.

        The mixed-pack subtlety lives here. ``loss_visual_mask`` /
        ``img_end_mask`` / ``target_visual_mask`` must mark ONLY generation
        images. We build a concat-level boolean ``target_region`` over the pad
        spans of generation images (``is_target=True``), then derive the masks in
        the shifted (label/input) space. Understanding images are excluded even
        though their labels contain IMG_PAD/IMG_END ids.
        """
        concat_ids = torch.cat(packed_ids)
        concat_masks = torch.cat(packed_masks)
        total_len = len(concat_ids)

        # Build a concat-level (pre-pad, pre-shift) boolean marking generation
        # image spans. packed_image_positions are (start, end) offsets into the
        # concatenated (unpadded) sequence; understanding images get False.
        target_region_concat = torch.zeros(total_len, dtype=torch.bool)
        for (start, end), is_tgt in zip(packed_image_positions, packed_image_is_target):
            if is_tgt:
                s = max(0, min(start, total_len))
                e = max(0, min(end, total_len))
                if e > s:
                    target_region_concat[s:e] = True

        # Pad to seq_length + 1 (need +1 for causal LM shift)
        pad_len = self.seq_length + 1 - total_len
        if pad_len > 0:
            concat_ids = torch.cat([
                concat_ids,
                torch.full((pad_len,), EOD_TOKEN_ID, dtype=torch.long),
            ])
            concat_masks = torch.cat([
                concat_masks,
                torch.zeros(pad_len, dtype=torch.float32),
            ])
            target_region_concat = torch.cat([
                target_region_concat,
                torch.zeros(pad_len, dtype=torch.bool),
            ])
        else:
            concat_ids = concat_ids[: self.seq_length + 1]
            concat_masks = concat_masks[: self.seq_length + 1]
            target_region_concat = target_region_concat[: self.seq_length + 1]

        # Causal LM shift
        input_ids, labels, loss_mask = apply_causal_lm_shift(concat_ids, concat_masks)
        # target_region aligned to input_ids (concat[:-1]) and labels (concat[1:])
        target_region_input = target_region_concat[:-1]
        target_region_label = target_region_concat[1:]

        # cu_seqlens
        cu_seqlens = [0]
        cumulative = 0
        for sl in sample_lengths:
            cumulative += sl
            cu_seqlens.append(min(cumulative, self.seq_length))
        num_real_samples = len(sample_lengths)
        if cu_seqlens[-1] < self.seq_length:
            cu_seqlens.append(self.seq_length)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        # Global arange position ids: positions run continuously across the whole
        # pack and are NOT reset at sub-sequence boundaries.
        position_ids = torch.arange(self.seq_length, dtype=torch.long)

        # Pixel data (all images, both modalities) for ViT encoding in forward
        if packed_pixel_values_list:
            all_pixel_values = torch.cat(packed_pixel_values_list, dim=0)
            all_image_grid_thw = torch.cat(packed_image_grid_thw_list, dim=0)
        else:
            all_pixel_values = torch.empty(0, dtype=torch.float32)
            all_image_grid_thw = torch.empty(0, 3, dtype=torch.long)

        # visual_mask: ALL image pad positions in input_ids (both modalities).
        # Used to fill visual embeddings (every image, input or target).
        visual_mask = (input_ids == IMG_PAD_ID)

        # loss_visual_mask: depth-CE hidden-state selection. ONLY generation
        # images: label is IMG_PAD/IMG_END AND inside a target region.
        loss_visual_mask = ((labels == IMG_PAD_ID) | (labels == IMG_END_ID)) & target_region_label

        # img_end_mask: img_end label positions among target images only.
        img_end_mask = (labels == IMG_END_ID) & target_region_label

        # target_visual_mask: generation image pad positions in input_ids.
        # Used by forward to slice visual_ids (all-images) down to target images.
        target_visual_mask = visual_mask & target_region_input

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
