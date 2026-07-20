# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for data/image_processing.py — Image preprocessing, placeholder building."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.image_processing import (
    estimate_visual_tokens,
    build_image_placeholder_sequence,
)
from data.tokenize_utils import IMG_START_ID, IMG_END_ID, IMG_PAD_ID, IMG_NEWLINE_ID


class TestEstimateVisualTokens:
    """Test visual token count estimation from grid dimensions."""

    def test_basic_estimation(self):
        """ViT spatial merge factor=2: token_h = h//2, token_w = w//2."""
        grid_thw = torch.tensor([[1, 14, 14]])  # t=1, h=14, w=14
        num_tokens, token_h, token_w = estimate_visual_tokens(grid_thw)
        assert token_h == 7
        assert token_w == 7
        assert num_tokens == 1 * 7 * 7  # 49

    def test_non_square_grid(self):
        grid_thw = torch.tensor([[1, 28, 14]])
        num_tokens, token_h, token_w = estimate_visual_tokens(grid_thw)
        assert token_h == 14
        assert token_w == 7
        assert num_tokens == 14 * 7

    def test_temporal_dimension(self):
        """Temporal dim t > 1 (video case)."""
        grid_thw = torch.tensor([[4, 14, 14]])
        num_tokens, token_h, token_w = estimate_visual_tokens(grid_thw)
        assert token_h == 7
        assert token_w == 7
        assert num_tokens == 4 * 7 * 7  # 196

    def test_1d_input(self):
        """Should handle [3] shaped input (not batched)."""
        grid_thw = torch.tensor([1, 14, 14])
        num_tokens, token_h, token_w = estimate_visual_tokens(grid_thw)
        assert num_tokens == 49

    def test_small_grid(self):
        """Minimum grid: h=2, w=2 → 1x1 tokens."""
        grid_thw = torch.tensor([[1, 2, 2]])
        num_tokens, token_h, token_w = estimate_visual_tokens(grid_thw)
        assert token_h == 1
        assert token_w == 1
        assert num_tokens == 1


class TestBuildImagePlaceholderSequence:
    """Test placeholder token sequence generation."""

    def test_basic_sequence(self):
        """Without newlines: img_start + N pads + img_end."""
        result = build_image_placeholder_sequence(num_tokens=5)
        ids = result["input_ids"]
        mask = result["loss_mask"]

        assert ids[0].item() == IMG_START_ID
        assert ids[-1].item() == IMG_END_ID
        assert len(ids) == 7  # 1 + 5 + 1
        assert result["num_tokens"] == 5

        # All masks should be 0 (non-trainable by default)
        assert mask.sum().item() == 0.0

    def test_trainable_sequence(self):
        """Trainable=True should set loss_mask=1 on pad tokens."""
        result = build_image_placeholder_sequence(num_tokens=5, trainable=True)
        mask = result["loss_mask"]

        # start and end: 0
        assert mask[0].item() == 0.0
        assert mask[-1].item() == 0.0
        # pads: 1
        assert mask[1:-1].sum().item() == 5.0

    def test_with_newlines(self):
        """Grid-based with newlines between rows."""
        result = build_image_placeholder_sequence(
            num_tokens=12, token_h=3, token_w=4, with_newline=True
        )
        ids = result["input_ids"]

        # Expected structure: start + [4 pads + newline] * 2 + 4 pads + end
        assert ids[0].item() == IMG_START_ID
        assert ids[-1].item() == IMG_END_ID

        # Count newlines
        newlines = (ids == IMG_NEWLINE_ID).sum().item()
        assert newlines == 2  # 3 rows -> 2 newlines

        # Count pads
        pads = (ids == IMG_PAD_ID).sum().item()
        assert pads == 12  # 3 * 4 = 12

    def test_newline_masks_are_zero(self):
        """Newline tokens should have loss_mask=0 even when trainable."""
        result = build_image_placeholder_sequence(
            num_tokens=6, token_h=2, token_w=3, with_newline=True, trainable=True
        )
        ids = result["input_ids"]
        mask = result["loss_mask"]

        for i in range(len(ids)):
            if ids[i].item() in [IMG_START_ID, IMG_END_ID, IMG_NEWLINE_ID]:
                assert mask[i].item() == 0.0, \
                    f"Special token at pos {i} (id={ids[i].item()}) should have mask=0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
