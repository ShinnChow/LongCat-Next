# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for data/tokenize_utils.py — Tokenization, loss masks, special tokens."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.tokenize_utils import (
    tokenize_text_segment,
    create_image_placeholder_tokens,
    apply_causal_lm_shift,
    EOD_TOKEN_ID,
    IMG_PAD_ID,
    IMG_START_ID,
    IMG_END_ID,
    IMG_NEWLINE_ID,
    SPECIAL_NO_LOSS_START,
    SPECIAL_NO_LOSS_END,
)


class MockTokenizer:
    """Simple tokenizer that assigns ASCII codes as token IDs."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class TestTokenizeTextSegment:
    """Test text segment tokenization."""

    def test_trainable_text(self):
        tokenizer = MockTokenizer()
        result = tokenize_text_segment(tokenizer, "hi", trainable=True)
        assert len(result["input_ids"]) == 2
        assert result["loss_mask"].sum().item() == 2.0  # all trainable

    def test_non_trainable_text(self):
        tokenizer = MockTokenizer()
        result = tokenize_text_segment(tokenizer, "hi", trainable=False)
        assert len(result["input_ids"]) == 2
        assert result["loss_mask"].sum().item() == 0.0  # none trainable

    def test_empty_text(self):
        tokenizer = MockTokenizer()
        result = tokenize_text_segment(tokenizer, "", trainable=True)
        assert len(result["input_ids"]) == 0
        assert len(result["loss_mask"]) == 0


class TestCreateImagePlaceholderTokens:
    """Test image placeholder token generation."""

    def test_basic_placeholder(self):
        """img_start + N pads + img_end."""
        result = create_image_placeholder_tokens(num_tokens=10, trainable=False)
        ids = result["input_ids"]
        mask = result["loss_mask"]

        assert ids[0].item() == IMG_START_ID
        assert ids[-1].item() == IMG_END_ID
        # 10 pad tokens between start and end
        assert (ids[1:-1] == IMG_PAD_ID).all()
        assert len(ids) == 12  # 1 start + 10 pad + 1 end

        # loss_mask: start and end should be 0
        assert mask[0].item() == 0.0
        assert mask[-1].item() == 0.0
        # pads should be 0 (non-trainable)
        assert mask[1:-1].sum().item() == 0.0

    def test_trainable_placeholder(self):
        """Trainable image tokens should have loss_mask=1 on pads."""
        result = create_image_placeholder_tokens(num_tokens=5, trainable=True)
        mask = result["loss_mask"]

        # start/end should still be 0
        assert mask[0].item() == 0.0
        assert mask[-1].item() == 0.0
        # pads should be 1 (trainable)
        assert mask[1:-1].sum().item() == 5.0

    def test_with_newlines(self):
        """Grid-based placeholder with img_newline between rows."""
        # 3 rows x 4 cols = 12 tokens
        result = create_image_placeholder_tokens(
            num_tokens=12, token_h=3, token_w=4, with_newline=True, trainable=False
        )
        ids = result["input_ids"]

        # Expected: start + (4 pads + newline) * 2 + 4 pads + end
        # = 1 + (4+1)*2 + 4 + 1 = 1 + 10 + 4 + 1 = 16
        assert ids[0].item() == IMG_START_ID
        assert ids[-1].item() == IMG_END_ID

        # Check newlines between rows
        newline_positions = [i for i, x in enumerate(ids) if x.item() == IMG_NEWLINE_ID]
        assert len(newline_positions) == 2  # 3 rows -> 2 newlines

    def test_without_newlines_grid(self):
        """When with_newline=False, no newlines even with grid dims."""
        result = create_image_placeholder_tokens(
            num_tokens=12, token_h=3, token_w=4, with_newline=False, trainable=False
        )
        ids = result["input_ids"]
        assert (ids != IMG_NEWLINE_ID).all()
        assert len(ids) == 14  # 1 start + 12 pad + 1 end


class TestApplyCausalLMShift:
    """Test the causal LM shift operation."""

    def test_basic_shift(self):
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        loss_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])

        shifted_input, labels, shifted_mask = apply_causal_lm_shift(input_ids, loss_mask)

        assert len(shifted_input) == 4
        assert len(labels) == 4
        assert len(shifted_mask) == 4

        # input = [1,2,3,4], labels = [2,3,4,5]
        assert shifted_input.tolist() == [1, 2, 3, 4]
        assert labels.tolist() == [2, 3, 4, 5]
        # mask is shifted by 1: loss_mask[1:] = [0,1,1,0]
        assert shifted_mask.tolist() == [0.0, 1.0, 1.0, 0.0]

    def test_single_element(self):
        """Edge case: single element sequence."""
        input_ids = torch.tensor([42])
        loss_mask = torch.tensor([1.0])
        shifted_input, labels, shifted_mask = apply_causal_lm_shift(input_ids, loss_mask)
        assert len(shifted_input) == 0

    def test_preserves_dtypes(self):
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        loss_mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        shifted_input, labels, shifted_mask = apply_causal_lm_shift(input_ids, loss_mask)
        assert shifted_input.dtype == torch.long
        assert labels.dtype == torch.long
        assert shifted_mask.dtype == torch.float32


class TestSpecialTokenConstants:
    """Test special token ID constants are correct."""

    def test_token_ids(self):
        assert EOD_TOKEN_ID == 2
        assert IMG_START_ID == 131106
        assert IMG_END_ID == 131107
        assert IMG_PAD_ID == 131108
        assert IMG_NEWLINE_ID == 131109

    def test_special_no_loss_range(self):
        """Special multimodal tokens range is valid."""
        assert SPECIAL_NO_LOSS_START < SPECIAL_NO_LOSS_END
        # img_start, img_end, img_pad, img_newline are IN this range
        for tid in [IMG_START_ID, IMG_END_ID, IMG_PAD_ID, IMG_NEWLINE_ID]:
            assert SPECIAL_NO_LOSS_START <= tid < SPECIAL_NO_LOSS_END


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
