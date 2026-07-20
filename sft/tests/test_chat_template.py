# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for data/chat_template.py — Chat template and trainable markers."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.chat_template import (
    unify_message_format,
    encode_with_trainable_markers,
    TRAINABLE_START,
    TRAINABLE_END,
    ROLE_SYSTEM,
    ROLE_USER,
    ROLE_ASSISTANT,
    EOD_TOKEN,
)


class TestUnifyMessageFormat:
    """Test message format normalization."""

    def test_standard_format(self):
        """Standard {role, content} should pass through."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = unify_message_format(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Hi there"

    def test_legacy_format(self):
        """Legacy {from, value} should be converted."""
        msgs = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
        ]
        result = unify_message_format(msgs)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_unsupported_format_raises(self):
        """Unknown format should raise ValueError."""
        msgs = [{"text": "hello"}]
        with pytest.raises(ValueError, match="Unsupported message format"):
            unify_message_format(msgs)

    def test_assistant_masks_preserved(self):
        """assistant_masks should be copied to unified format."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "assistant_masks": 0},
        ]
        result = unify_message_format(msgs)
        assert result[1].get("assistant_masks") == 0


class TestEncodeWithTrainableMarkers:
    """Test trainable marker insertion.

    These tests require a tokenizer with apply_chat_template support.
    We use a mock tokenizer for unit testing.
    """

    class MockTokenizer:
        """Mock tokenizer that simulates LongCat-Next chat template."""

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            """Simple mock: join messages with role tokens."""
            result = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    result += f"<longcat_system>{content}</longcat_s>"
                elif role == "user":
                    result += f"<longcat_user>{content}</longcat_s>"
                elif role == "assistant":
                    result += f"<longcat_assistant>{content}</longcat_s>"
            return result

    def test_single_turn(self):
        """Single user-assistant turn should wrap assistant content."""
        tokenizer = self.MockTokenizer()
        messages = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
        result = encode_with_trainable_markers(tokenizer, messages)
        assert TRAINABLE_START in result
        assert TRAINABLE_END in result
        # User content should NOT be wrapped
        assert f"{TRAINABLE_START}What is 1+1?" not in result

    def test_multi_turn(self):
        """Multiple turns should only wrap assistant responses."""
        tokenizer = self.MockTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "Fine"},
        ]
        result = encode_with_trainable_markers(tokenizer, messages)
        # Should have exactly 2 trainable regions
        assert result.count(TRAINABLE_START) == 2
        assert result.count(TRAINABLE_END) == 2

    def test_empty_messages(self):
        """Empty message list should return empty string."""
        tokenizer = self.MockTokenizer()
        result = encode_with_trainable_markers(tokenizer, [])
        assert result == ""

    def test_consecutive_assistant_raises(self):
        """Consecutive assistant messages should raise ValueError."""
        tokenizer = self.MockTokenizer()
        messages = [
            {"role": "assistant", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        with pytest.raises(ValueError, match="Consecutive assistant"):
            encode_with_trainable_markers(tokenizer, messages)

    def test_assistant_masks_zero_skips_loss(self):
        """assistant_masks=0 should remove trainable markers for that turn."""
        tokenizer = self.MockTokenizer()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "assistant_masks": 0},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "World", "assistant_masks": 1},
        ]
        result = encode_with_trainable_markers(tokenizer, messages)
        # Only 1 trainable region (the second assistant turn)
        assert result.count(TRAINABLE_START) == 1
        assert "World" in result.split(TRAINABLE_START)[1].split(TRAINABLE_END)[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
