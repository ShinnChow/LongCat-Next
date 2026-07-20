# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Verify LossFreeBalanceManager's per-step bias delta.

The per-expert bias target for the normal experts (with zero-experts present) is
(target_topk/moe_topk)/num_normal, NOT a uniform 1/n_experts. This test locks the
per-step bias delta to that formula via an independent reference implementation.
"""
import torch
from losses.loss_free_balance import LossFreeBalanceManager


class _FakeRouter:
    def __init__(self, n):
        self.e_score_correction_bias = torch.zeros(n)


def _reference_delta(topk_idx, N, ZERO, TT, MT, RATE, only_adapt_ffn=True):
    counter = torch.zeros(N, dtype=torch.long)
    counter.scatter_add_(0, topk_idx.reshape(-1), torch.ones_like(topk_idx.reshape(-1)))
    assigned = counter.float() / counter.float().sum()
    num_normal = N - ZERO
    r = TT / MT
    target = torch.empty(N)
    target[:num_normal] = r / num_normal
    target[num_normal:] = (1 - r) / ZERO
    offset = target - assigned
    if only_adapt_ffn:
        offset[-ZERO:] = 0.0
    return offset * RATE


def test_bias_delta_matches_formula_with_zero_experts():
    N, ZERO, TT, MT, RATE = 40, 8, 8, 12, 0.1
    torch.manual_seed(0)
    topk_idx = torch.randint(0, N, (500, MT))

    router = _FakeRouter(N)
    mgr = LossFreeBalanceManager([router], rate=RATE, dynamic_update=True,
                                 only_adapt_ffn_bias=True, zero_expert_num=ZERO,
                                 target_topk=TT, moe_topk=None)
    mgr.add_batch([(None, topk_idx, N, MT)])
    assert mgr.moe_topk == MT  # auto-captured
    before = router.e_score_correction_bias.clone()
    mgr.finalize_step(iteration=0)
    fsdp_delta = router.e_score_correction_bias - before

    ref_delta = _reference_delta(topk_idx, N, ZERO, TT, MT, RATE)
    assert torch.allclose(fsdp_delta, ref_delta, atol=1e-7), \
        f"max diff {(fsdp_delta - ref_delta).abs().max()}"
    # zero-expert bias untouched (only_adapt_ffn_bias)
    assert bool((fsdp_delta[-ZERO:] == 0).all())


def test_old_uniform_target_would_differ():
    """Guard: the pre-fix uniform 1/N target must NOT equal the correct formula."""
    N, ZERO, TT, MT, RATE = 40, 8, 8, 12, 0.1
    torch.manual_seed(0)
    topk_idx = torch.randint(0, N, (500, MT))
    counter = torch.zeros(N, dtype=torch.long)
    counter.scatter_add_(0, topk_idx.reshape(-1), torch.ones_like(topk_idx.reshape(-1)))
    assigned = counter.float() / counter.float().sum()
    old_offset = (1.0 / N) - assigned
    old_offset[-ZERO:] = 0.0
    old_delta = old_offset * RATE
    ref_delta = _reference_delta(topk_idx, N, ZERO, TT, MT, RATE)
    assert (old_delta - ref_delta).abs().max() > 1e-5
