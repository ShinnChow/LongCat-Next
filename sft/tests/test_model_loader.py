# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for model/model_loader.py — Key mapping, freeze strategies."""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.model_loader import _fsdp_name_to_hf_key


class TestFsdpNameToHfKey:
    """Test FSDP parameter name → HF state dict key conversion.

    This is a critical function that must handle:
    1. SFTTrainingWrapper "model." prefix stripping
    2. CheckpointWrapper "_checkpoint_wrapped_module." removal
    3. embed_tokens ↔ ngram_embeddings.word_embeddings mapping
    4. codebooks._shared → codebooks.0 mapping
    """

    def test_strip_wrapper_prefix(self):
        """SFTTrainingWrapper adds 'model.' prefix → strip it."""
        assert _fsdp_name_to_hf_key("model.lm_head.weight") == "lm_head.weight"

    def test_strip_checkpoint_wrapper(self):
        """CheckpointWrapper inserts '_checkpoint_wrapped_module.' → remove it."""
        fsdp_name = "model.model.layers.0._checkpoint_wrapped_module.mlp.experts.0.gate_proj.weight"
        expected = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_multiple_checkpoint_wrappers(self):
        """Handle nested CheckpointWrapper segments."""
        fsdp_name = "model.model.layers.0._checkpoint_wrapped_module.self_attn._checkpoint_wrapped_module.q_proj.weight"
        expected = "model.layers.0.self_attn.q_proj.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_embed_tokens_to_word_embeddings(self):
        """After _break_parameter_sharing, embed_tokens appears as
        ngram_embeddings.word_embeddings in FSDP params, but HF uses embed_tokens."""
        fsdp_name = "model.model.ngram_embeddings.word_embeddings.weight"
        expected = "model.embed_tokens.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_codebook_shared_to_zero(self):
        """_SharedModuleList registers the shared codebook as '_shared',
        but HF state dict uses '0' index."""
        fsdp_name = "model.model.visual_tokenizer.visual_bridge_model.quantizer.quantize.codebooks._shared.embed"
        expected = "model.visual_tokenizer.visual_bridge_model.quantizer.quantize.codebooks.0.embed"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_regular_llm_layer(self):
        """Regular LLM layer without checkpointing."""
        fsdp_name = "model.model.layers.5.self_attn.q_proj.weight"
        expected = "model.layers.5.self_attn.q_proj.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_visual_model(self):
        """Visual model (ViT) path."""
        fsdp_name = "model.model.visual_tokenizer.visual_model.encoder.layers.0.self_attn.q_proj.weight"
        expected = "model.visual_tokenizer.visual_model.encoder.layers.0.self_attn.q_proj.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_visual_head(self):
        fsdp_name = "model.visual_head.heads.0.weight"
        expected = "visual_head.heads.0.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_audio_tokenizer(self):
        fsdp_name = "model.model.audio_tokenizer.audio_model.encoder.weight"
        expected = "model.audio_tokenizer.audio_model.encoder.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_ngram_embeddings_non_word(self):
        """ngram_embeddings paths that are NOT word_embeddings should NOT
        be mapped to embed_tokens."""
        fsdp_name = "model.model.ngram_embeddings.embedders.0.dense.weight"
        expected = "model.ngram_embeddings.embedders.0.dense.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_combined_checkpoint_and_sharing(self):
        """Both CheckpointWrapper and embed_tokens sharing in same path."""
        fsdp_name = "model.model._checkpoint_wrapped_module.ngram_embeddings.word_embeddings.weight"
        expected = "model.embed_tokens.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected

    def test_norm_layer(self):
        fsdp_name = "model.model.norm.weight"
        expected = "model.norm.weight"
        assert _fsdp_name_to_hf_key(fsdp_name) == expected


class TestFreezeStrategies:
    """Test freeze_for_understand and freeze_for_generate.

    Uses a simple mock model since we can't load the real model locally.
    """

    class SimpleMockModel(nn.Module):
        """Minimal model that mimics LongcatNextForCausalLM param names."""

        def __init__(self):
            super().__init__()
            # LLM layers
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])
            self.model.embed_tokens = nn.Embedding(100, 10)
            self.model.ngram_embeddings = nn.Module()
            self.model.ngram_embeddings.word_embeddings = self.model.embed_tokens
            self.model.ngram_embeddings.embedders = nn.ModuleList([nn.Linear(10, 10)])
            self.model.norm = nn.LayerNorm(10)

            # Heads
            self.lm_head = nn.Linear(10, 100)
            self.visual_head = nn.Linear(10, 50)
            self.audio_head = nn.Linear(10, 30)

            # Visual tokenizer
            self.model.visual_tokenizer = nn.Module()
            self.model.visual_tokenizer.visual_model = nn.Linear(10, 10)
            self.model.visual_tokenizer.visual_bridge_model = nn.Linear(10, 10)
            self.model.visual_tokenizer.visual_embedding_layer = nn.Linear(10, 10)

            # Audio tokenizer
            self.model.audio_tokenizer = nn.Linear(10, 10)

    def test_freeze_for_understand(self):
        """Understand: train LLM+embedding+lm_head+visual_embedding, freeze rest."""
        from model.model_loader import freeze_for_understand

        model = self.SimpleMockModel()

        # Mock dist to avoid distributed check
        import unittest.mock as mock
        with mock.patch("model.model_loader.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            mock_dist.get_rank.return_value = 0

            freeze_for_understand(model)

        # Check trainable
        for name, param in model.named_parameters():
            if "model.layers." in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "model.embed_tokens." in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "model.ngram_embeddings." in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "model.norm." in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "lm_head." in name:
                assert param.requires_grad, f"{name} should be trainable"
            elif "visual_tokenizer.visual_embedding_layer." in name:
                assert param.requires_grad, f"{name} should be trainable"

        # Check frozen
        for name, param in model.named_parameters():
            if "visual_model." in name or "visual_bridge_model." in name:
                assert not param.requires_grad, f"{name} should be frozen"
            elif "visual_head." in name:
                assert not param.requires_grad, f"{name} should be frozen"
            elif "audio_head." in name:
                assert not param.requires_grad, f"{name} should be frozen"
            elif "audio_tokenizer." in name:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_freeze_for_generate(self):
        """Generate: same as understand + visual_head is trainable."""
        from model.model_loader import freeze_for_generate

        model = self.SimpleMockModel()

        import unittest.mock as mock
        with mock.patch("model.model_loader.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            mock_dist.get_rank.return_value = 0

            freeze_for_generate(model)

        # visual_head should be trainable in generate mode
        for name, param in model.named_parameters():
            if "visual_head." in name:
                assert param.requires_grad, f"{name} should be trainable in generate mode"

        # audio_head should still be frozen
        for name, param in model.named_parameters():
            if "audio_head." in name:
                assert not param.requires_grad, f"{name} should be frozen"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
