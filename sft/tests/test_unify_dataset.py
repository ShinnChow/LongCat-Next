# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for data/unify_dataset.py — mixed understand+generate packing.

These tests cover the parts that do NOT require a real tokenizer / image
processor:
  * sample-type routing (`_is_generate_sample`)
  * `_finalize_pack` mask construction (the core mixed-pack correctness):
      - visual_mask covers ALL image pads (both modalities)
      - loss_visual_mask / img_end_mask / target_visual_mask cover ONLY
        generation (target) images
      - pure-generate pack reproduces the generate-only mask semantics

The end-to-end bitwise equivalence vs the single-task datasets (which needs a
tokenizer) is exercised by the smoke script run on the GPU docker.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.unify_dataset import UnifyPackedDataset
from data.tokenize_utils import (
    IMG_PAD_ID, IMG_END_ID, IMG_START_ID, IMG_NEWLINE_ID, EOD_TOKEN_ID,
)


def _make_ds(seq_length):
    """Build an UnifyPackedDataset instance without running __init__ (which
    would construct sub-datasets needing real paths). We only need the
    `_finalize_pack` / `_is_generate_sample` methods + a couple attrs."""
    ds = UnifyPackedDataset.__new__(UnifyPackedDataset)
    ds.seq_length = seq_length
    ds.world_size = 1
    ds.rank = 0
    ds.no_packing = False
    return ds


def _img_placeholder(n_pad, trainable, with_newline=False, token_w=0):
    """Build a single image placeholder (ids, mask) like the datasets do.

    Layout: START, PAD*n (with optional NEWLINE between rows), END.
    """
    ids = [IMG_START_ID]
    mask = [0.0]
    if with_newline and token_w > 0:
        rows = n_pad // token_w
        for r in range(rows):
            ids.extend([IMG_PAD_ID] * token_w)
            mask.extend([1.0 if trainable else 0.0] * token_w)
            if r < rows - 1:
                ids.append(IMG_NEWLINE_ID)
                mask.append(0.0)
    else:
        ids.extend([IMG_PAD_ID] * n_pad)
        mask.extend([1.0 if trainable else 0.0] * n_pad)
    ids.append(IMG_END_ID)
    mask.append(0.0)
    return ids, mask


# ─────────────────────────── _is_generate_sample ───────────────────────────

class TestSampleTypeRouting:
    def test_understand_image_in_user(self):
        msgs = [
            {"role": "user", "content": "look <longcat_img_start>/p.png<longcat_img_end>"},
            {"role": "assistant", "content": "it is a cat"},
        ]
        assert UnifyPackedDataset._is_generate_sample(msgs) is False

    def test_generate_image_in_assistant(self):
        msgs = [
            {"role": "user", "content": "draw a cat"},
            {"role": "assistant", "content": "<longcat_img_start>/p.png<longcat_img_end>"},
        ]
        assert UnifyPackedDataset._is_generate_sample(msgs) is True

    def test_generate_offline_format(self):
        msgs = [
            {"role": "user", "content": "draw"},
            {"role": "assistant",
             "content": "<img_gen_token_start>...<img_gen_token_end>"},
        ]
        assert UnifyPackedDataset._is_generate_sample(msgs) is True

    def test_pure_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert UnifyPackedDataset._is_generate_sample(msgs) is False


# ─────────────────────────── _finalize_pack masks ───────────────────────────

class TestFinalizePackMasks:
    def _build_sample(self, text_pre, img_ids, img_mask, text_post=()):
        """Assemble one sample's (ids, mask, image span). Returns dict with
        input_ids, loss_mask tensors and the image (start,end) span within the
        *unpadded sample-local* sequence, plus EOD appended (mask=1)."""
        ids = list(text_pre)
        mask = [0.0] * len(text_pre)
        img_start = len(ids)
        ids.extend(img_ids)
        mask.extend(img_mask)
        img_end = len(ids)
        ids.extend(text_post)
        mask.extend([1.0] * len(text_post))
        ids.append(EOD_TOKEN_ID)
        mask.append(1.0)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.float32),
            "img_span": (img_start, img_end),
        }

    def test_understand_image_excluded_from_loss(self):
        """A pure-understand pack: image pads enter visual_mask but NOT
        loss_visual_mask / target_visual_mask."""
        seq = 64
        ds = _make_ds(seq)
        n_pad = 6
        img_ids, img_mask = _img_placeholder(n_pad, trainable=False)
        s = self._build_sample([10, 11], img_ids, img_mask, text_post=[12, 13])

        out = ds._finalize_pack(
            packed_ids=[s["input_ids"]],
            packed_masks=[s["loss_mask"]],
            packed_pixel_values_list=[torch.zeros(1)],
            packed_image_grid_thw_list=[torch.tensor([[1, 2, 2]])],
            packed_image_positions=[s["img_span"]],
            packed_image_is_target=[False],
            sample_lengths=[len(s["input_ids"])],
        )
        assert out["visual_mask"].sum().item() == n_pad      # pads visible for embedding
        assert out["loss_visual_mask"].sum().item() == 0     # NOT a depth target
        assert out["img_end_mask"].sum().item() == 0
        assert out["target_visual_mask"].sum().item() == 0

    def test_generate_image_in_loss(self):
        """A pure-generate pack: image pads enter loss_visual_mask and
        target_visual_mask; img_end label position included."""
        seq = 64
        ds = _make_ds(seq)
        n_pad = 6
        img_ids, img_mask = _img_placeholder(n_pad, trainable=True)
        s = self._build_sample([10, 11], img_ids, img_mask)

        out = ds._finalize_pack(
            packed_ids=[s["input_ids"]],
            packed_masks=[s["loss_mask"]],
            packed_pixel_values_list=[torch.zeros(1)],
            packed_image_grid_thw_list=[torch.tensor([[1, 2, 2]])],
            packed_image_positions=[s["img_span"]],
            packed_image_is_target=[True],
            sample_lengths=[len(s["input_ids"])],
        )
        assert out["target_visual_mask"].sum().item() == n_pad
        # loss_visual_mask = PAD labels (n_pad) + END label (1) within target
        assert out["img_end_mask"].sum().item() == 1
        assert out["loss_visual_mask"].sum().item() == n_pad + 1

    def test_mixed_pack(self):
        """Mixed pack: 1 understand image + 1 generate image. Only the generate
        image enters the depth-loss masks; both enter visual_mask."""
        seq = 128
        ds = _make_ds(seq)
        n_u, n_g = 4, 6
        u_ids, u_mask = _img_placeholder(n_u, trainable=False)
        g_ids, g_mask = _img_placeholder(n_g, trainable=True)
        u = self._build_sample([20, 21], u_ids, u_mask, text_post=[22])  # understand
        g = self._build_sample([30], g_ids, g_mask)                       # generate

        # pack: understand then generate; compute global spans with offsets
        len_u = len(u["input_ids"])
        u_span = u["img_span"]
        g_span = (g["img_span"][0] + len_u, g["img_span"][1] + len_u)

        out = ds._finalize_pack(
            packed_ids=[u["input_ids"], g["input_ids"]],
            packed_masks=[u["loss_mask"], g["loss_mask"]],
            packed_pixel_values_list=[torch.zeros(1), torch.zeros(1)],
            packed_image_grid_thw_list=[torch.tensor([[1, 2, 2]]),
                                        torch.tensor([[1, 2, 3]])],
            packed_image_positions=[u_span, g_span],
            packed_image_is_target=[False, True],
            sample_lengths=[len_u, len(g["input_ids"])],
        )
        # both images' pads visible for embedding
        assert out["visual_mask"].sum().item() == n_u + n_g
        # only generate image enters depth loss
        assert out["target_visual_mask"].sum().item() == n_g
        assert out["img_end_mask"].sum().item() == 1
        assert out["loss_visual_mask"].sum().item() == n_g + 1
        # cu_seqlens reflects 2 real samples
        assert out["num_real_samples"] == 2

    def test_target_visual_mask_aligns_visual_mask_order(self):
        """The selection `target_visual_mask[visual_mask]` (used in forward to
        slice visual_ids) must pick exactly the generate-image pad rows, in
        order. Understand image comes first so the selection must be
        [False*n_u, True*n_g]."""
        seq = 128
        ds = _make_ds(seq)
        n_u, n_g = 4, 6
        u_ids, u_mask = _img_placeholder(n_u, trainable=False)
        g_ids, g_mask = _img_placeholder(n_g, trainable=True)
        u = self._build_sample([20], u_ids, u_mask)
        g = self._build_sample([30], g_ids, g_mask)
        len_u = len(u["input_ids"])
        u_span = u["img_span"]
        g_span = (g["img_span"][0] + len_u, g["img_span"][1] + len_u)

        out = ds._finalize_pack(
            packed_ids=[u["input_ids"], g["input_ids"]],
            packed_masks=[u["loss_mask"], g["loss_mask"]],
            packed_pixel_values_list=[torch.zeros(1), torch.zeros(1)],
            packed_image_grid_thw_list=[torch.tensor([[1, 2, 2]]),
                                        torch.tensor([[1, 2, 3]])],
            packed_image_positions=[u_span, g_span],
            packed_image_is_target=[False, True],
            sample_lengths=[len_u, len(g["input_ids"])],
        )
        gen_sel = out["target_visual_mask"][out["visual_mask"]]
        assert gen_sel.shape[0] == n_u + n_g
        assert gen_sel[:n_u].sum().item() == 0      # understand rows -> not target
        assert gen_sel[n_u:].sum().item() == n_g    # generate rows -> target
