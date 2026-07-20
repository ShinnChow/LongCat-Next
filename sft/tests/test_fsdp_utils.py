# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for model/fsdp_utils.py — Parameter sharing, SharedModuleList, FSDP setup.

These tests focus on the pure-Python logic that can be tested without a GPU or
distributed environment: parameter sharing detection/breaking, SharedModuleList
behavior, and module navigation.
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.fsdp_utils import _SharedModuleList, _break_parameter_sharing


class TestSharedModuleList:
    """Test _SharedModuleList drop-in replacement for nn.ModuleList."""

    def test_len(self):
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=8)
        assert len(sml) == 8

    def test_getitem_returns_shared(self):
        """All indices should return the same module."""
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=4)
        for i in range(4):
            assert sml[i] is module

    def test_getitem_out_of_range(self):
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=4)
        with pytest.raises(IndexError):
            sml[4]
        with pytest.raises(IndexError):
            sml[-1]

    def test_iteration(self):
        """Iteration should yield the shared module N times."""
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=5)
        items = list(sml)
        assert len(items) == 5
        for item in items:
            assert item is module

    def test_single_child_registration(self):
        """FSDP sees children via named_children(). Should only see 1 child."""
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=8)
        children = list(sml.named_children())
        assert len(children) == 1
        name, child = children[0]
        assert name == "_shared"
        assert child is module

    def test_parameters_counted_once(self):
        """Parameters from the shared module should only appear once."""
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=8)
        params = list(sml.parameters())
        # Linear has weight + bias = 2 parameters
        assert len(params) == 2
        # Verify it's actually the shared module's params
        assert params[0] is module.weight
        assert params[1] is module.bias

    def test_named_parameters(self):
        """named_parameters should show _shared.weight, _shared.bias."""
        module = nn.Linear(10, 10)
        sml = _SharedModuleList(module, length=8)
        param_names = [name for name, _ in sml.named_parameters()]
        assert "_shared.weight" in param_names
        assert "_shared.bias" in param_names
        # Should NOT have "0.weight", "1.weight" etc.
        for i in range(8):
            assert f"{i}.weight" not in param_names

    def test_forward_compatible(self):
        """Simulate a quantizer forward loop: for i in range(N): out = codebooks[i](x)."""
        module = nn.Embedding(100, 32)
        sml = _SharedModuleList(module, length=8)

        x = torch.randint(0, 100, (5,))
        # All levels should work and produce the same output
        outputs = [sml[i](x) for i in range(8)]
        for out in outputs:
            torch.testing.assert_close(out, outputs[0])


class TestBreakParameterSharing:
    """Test _break_parameter_sharing logic using mock models."""

    class MockInnerModel(nn.Module):
        """Mimics LongcatNextModel with embed_tokens ↔ ngram_embeddings.word_embeddings sharing."""

        def __init__(self, shared=True):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 32)
            self.ngram_embeddings = nn.Module()
            if shared:
                # Same object — the sharing pattern we need to break
                self.ngram_embeddings.word_embeddings = self.embed_tokens
            else:
                self.ngram_embeddings.word_embeddings = nn.Embedding(100, 32)

    def test_detects_sharing(self):
        """When embed_tokens IS word_embeddings, should break the sharing."""
        inner = self.MockInnerModel(shared=True)

        # Before fix: both in _modules
        assert "embed_tokens" in inner._modules
        assert inner.embed_tokens is inner.ngram_embeddings.word_embeddings

        # Create a dummy top-level model for the sweep
        class DummyModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner  # simulates causal_lm.model
        model = DummyModel(inner)

        _break_parameter_sharing(model, inner, rank=0)

        # After fix: embed_tokens should NOT be in _modules
        assert "embed_tokens" not in inner._modules
        # But should still be accessible as attribute
        assert hasattr(inner, "embed_tokens")
        # And should still be the same object (just not registered)
        assert inner.embed_tokens is inner.ngram_embeddings.word_embeddings

    def test_no_double_parameters(self):
        """After breaking, named_parameters should not show the same param twice."""
        inner = self.MockInnerModel(shared=True)

        class DummyModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner
        model = DummyModel(inner)

        _break_parameter_sharing(model, inner, rank=0)

        # Count parameter IDs
        param_ids = set()
        for name, param in model.named_parameters():
            pid = id(param)
            assert pid not in param_ids, \
                f"Parameter id {pid} appears multiple times (duplicate: {name})"
            param_ids.add(pid)

    def test_no_sharing_noop(self):
        """When embed_tokens and word_embeddings are separate, should be a noop."""
        inner = self.MockInnerModel(shared=False)

        class DummyModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner
        model = DummyModel(inner)

        _break_parameter_sharing(model, inner, rank=0)

        # embed_tokens should still be in _modules (no change needed)
        assert "embed_tokens" in inner._modules

    def test_embed_tokens_still_callable(self):
        """After breaking, inner_model.embed_tokens(ids) should still work."""
        inner = self.MockInnerModel(shared=True)

        class DummyModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner
        model = DummyModel(inner)

        _break_parameter_sharing(model, inner, rank=0)

        # Should still work as an embedding lookup
        ids = torch.randint(0, 100, (5,))
        output = inner.embed_tokens(ids)
        assert output.shape == (5, 32)


class TestCodebookDedup:
    """Test codebook de-duplication via _dedup_shared_codebooks."""

    def test_shared_codebook_replaced(self):
        """nn.ModuleList with all-same entries should become _SharedModuleList."""
        from model.fsdp_utils import _dedup_shared_codebooks

        shared_module = nn.Linear(10, 10)
        parent = nn.Module()
        parent.codebooks = nn.ModuleList([shared_module] * 8)

        # Before: 8 entries, all same object
        assert len(parent.codebooks) == 8
        assert isinstance(parent.codebooks, nn.ModuleList)

        _dedup_shared_codebooks(parent, "codebooks", rank=0, attr_chain=False)

        # After: _SharedModuleList with 8 indices, 1 real module
        assert isinstance(parent.codebooks, _SharedModuleList)
        assert len(parent.codebooks) == 8
        assert parent.codebooks[0] is shared_module
        assert parent.codebooks[7] is shared_module

    def test_non_shared_modulelist_unchanged(self):
        """ModuleList with different entries should NOT be replaced."""
        from model.fsdp_utils import _dedup_shared_codebooks

        parent = nn.Module()
        parent.codebooks = nn.ModuleList([nn.Linear(10, 10) for _ in range(4)])

        _dedup_shared_codebooks(parent, "codebooks", rank=0, attr_chain=False)

        # Should still be a regular ModuleList
        assert isinstance(parent.codebooks, nn.ModuleList)
        assert len(parent.codebooks) == 4

    def test_nested_attr_chain(self):
        """Attr chain like 'quantizer.quantize.codebooks' should be traversed."""
        from model.fsdp_utils import _dedup_shared_codebooks

        shared_module = nn.Linear(10, 10)
        parent = nn.Module()
        parent.quantizer = nn.Module()
        parent.quantizer.quantize = nn.Module()
        parent.quantizer.quantize.codebooks = nn.ModuleList([shared_module] * 4)

        _dedup_shared_codebooks(parent, "quantizer.quantize.codebooks", rank=0, attr_chain=True)

        assert isinstance(parent.quantizer.quantize.codebooks, _SharedModuleList)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
