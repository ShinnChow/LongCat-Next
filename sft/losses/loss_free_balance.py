# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Loss-free expert balance for LongCat-Next MoE routing.

Implements loss-free load balancing (arXiv:2408.15664): no gradients; updates the
router.e_score_correction_bias buffer in-place once per optimizer step, driven by
the observed expert utilization. The bias steers routing toward uniform expert
usage without adding a balance loss term.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


class LossFreeBalanceManager:
    """Per-step expert-bias updater for LFB.

    Usage per optimizer step:
        manager.add_batch(router_outputs)   # collect from forward (multiple micro-batches OK)
        ...
        loss.backward(); optimizer.step()
        manager.finalize_step(iteration)    # all-reduce, update each router's bias, decay rate
    """

    def __init__(
        self,
        router_modules: List[torch.nn.Module],
        rate: float,
        decay_rule: str = "",
        dynamic_update: bool = True,
        only_adapt_ffn_bias: bool = True,
        zero_expert_num: int = 0,
        target_topk: Optional[float] = None,
        moe_topk: Optional[int] = None,
    ):
        assert rate > 0, "LossFreeBalanceManager should not be instantiated with rate=0"
        self.routers = router_modules
        self.rate = float(rate)
        self.original_rate = float(rate)
        self.dynamic_update = dynamic_update
        self.only_adapt_ffn_bias = only_adapt_ffn_bias
        self.zero_expert_num = int(zero_expert_num)
        # target_topk / moe_topk set the per-expert bias target. When both are set
        # and there are zero-experts, the normal-expert target is
        # (target_topk/moe_topk)/num_normal, NOT the uniform 1/n_experts.
        self.target_topk = None if target_topk is None else float(target_topk)
        self.moe_topk = None if moe_topk is None else int(moe_topk)

        # Parse "interval ratio min" -> (int, float, float) or None
        self.decay_rule: Optional[Tuple[int, float, float]] = None
        if decay_rule:
            parts = decay_rule.strip().split()
            assert len(parts) == 3, f"decay_rule must be 'interval ratio min', got {decay_rule!r}"
            self.decay_rule = (int(parts[0]), float(parts[1]), float(parts[2]))

        # Sanity-check shapes & init counters: one counter per router, on bias's device.
        self._counters: List[torch.Tensor] = []
        for r in self.routers:
            bias = r.e_score_correction_bias
            assert bias.dim() == 1, f"expected 1-D bias, got {bias.shape}"
            self._counters.append(
                torch.zeros(bias.shape[0], dtype=torch.int64, device=bias.device)
            )

    # ------------------------------------------------------------------ collect

    def add_batch(self, router_outputs: List[Tuple[torch.Tensor, torch.Tensor, int, int]]):
        """Accumulate topk_indices from one forward pass.

        router_outputs: list of (scores, topk_indices, n_experts, topk),
                        one entry per MoE layer (the order MUST match `routers`).
        """
        if len(router_outputs) != len(self.routers):
            # forward may produce additional captures (e.g. dummy text-only batches) —
            # only consume the prefix that lines up with our router list.
            n = min(len(router_outputs), len(self.routers))
        else:
            n = len(self.routers)

        for i in range(n):
            _, topk_indices, n_experts, _topk = router_outputs[i]
            counter = self._counters[i]
            assert counter.numel() == n_experts, \
                f"router {i}: counter size {counter.numel()} != n_experts {n_experts}"
            flat = topk_indices.detach().reshape(-1).long()
            counter.scatter_add_(0, flat, torch.ones_like(flat))
            # Capture moe_topk from the router output (matches lb-loss which also
            # reads topk per layer). Used as the bias-target denominator so we
            # don't depend on a separately-passed moe_topk that could drift.
            if self.moe_topk is None and _topk:
                self.moe_topk = int(_topk)

    # ------------------------------------------------------------------ update

    @torch.no_grad()
    def finalize_step(self, iteration: int) -> None:
        """All-reduce counters, update each router's bias, reset counters, apply decay."""
        if not self._counters:
            return

        world_size = dist.get_world_size() if dist.is_initialized() else 1

        for router, counter in zip(self.routers, self._counters):
            if world_size > 1:
                dist.all_reduce(counter, op=dist.ReduceOp.SUM)

            total = counter.sum()
            if total.item() == 0:
                counter.zero_()
                continue

            n_experts = counter.numel()
            assigned_frac = counter.float() / total.float()

            # Per-expert bias target:
            #   if zero_expert_num and target_topk are set:
            #     num_normal = n_experts - zero_expert_num
            #     normal_target = (target_topk / moe_topk) / num_normal
            #     zero_target   = (1 - target_topk / moe_topk) / zero_expert_num
            #     target = [normal_target]*num_normal + [zero_target]*zero_expert_num
            #   else:
            #     target = 1 / n_experts   (uniform fallback)
            # (zero-expert entries are zeroed below when only_adapt_ffn_bias, so
            #  zero_target only matters when that flag is off.)
            if (self.zero_expert_num > 0 and self.target_topk is not None
                    and self.moe_topk is not None):
                num_normal = n_experts - self.zero_expert_num
                ratio_topk = self.target_topk / self.moe_topk
                normal_target = ratio_topk / num_normal
                zero_target = (1.0 - ratio_topk) / self.zero_expert_num
                target_frac = torch.empty(
                    n_experts, dtype=torch.float32, device=counter.device)
                target_frac[:num_normal] = normal_target
                target_frac[num_normal:] = zero_target
            else:
                target_frac = 1.0 / n_experts
            offset = target_frac - assigned_frac  # [n_experts]

            if self.only_adapt_ffn_bias and self.zero_expert_num > 0:
                # Zero-expert sit at the tail of the expert list.
                offset[-self.zero_expert_num:] = 0.0

            if self.dynamic_update:
                delta = offset * self.rate
            else:
                delta = torch.sign(offset) * self.rate

            router.e_score_correction_bias.add_(delta.to(router.e_score_correction_bias.dtype))
            counter.zero_()

        # Decay schedule: every <interval> steps, rate *= ratio (floor at min).
        if self.decay_rule is not None:
            interval, ratio, min_rate = self.decay_rule
            new_rate = max(
                self.original_rate * (ratio ** (iteration // interval)),
                min_rate,
            )
            self.rate = new_rate
