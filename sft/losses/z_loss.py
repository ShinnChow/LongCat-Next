# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Z-loss and load balance loss for LongCat-Next FSDP SFT training.

Custom autograd functions inject auxiliary-loss gradients WITHOUT modifying the
forward loss scalar, so the reported loss value stays equal to the pure
cross-entropy loss while the auxiliary gradients still flow in backward.

Three MoE auxiliary loss components:

1. Hidden z-loss:
   Applied to hidden states BEFORE the final layernorm.
   z_loss = coeff * mean(logsumexp(|hidden_states|, dim=-1)^2)
   Operates on the absolute values of the hidden states.

2. MoE Router z-loss:
   Applied to each MoE layer's router logits.
   z_loss = mean(logsumexp(router_logits, dim=-1)^2)
   Scaled by router_z_loss_coeff and moe_loss_coeff.

3. Expert load balance loss:
   Encourages uniform expert utilization.
   lb_loss = sum(mean_prob * mean_selection) * n_experts / topk
   Scaled by moe_loss_coeff.

Displayed metrics:
  "expert load balance loss" = sum_layers(lb_loss * moe_loss_coeff)
  "z loss"                    = sum_layers(z_loss * z_loss_coeff * moe_loss_coeff)
"""

import torch


class AddAuxLossToBackward(torch.autograd.Function):
    """Inject an auxiliary MoE loss gradient without changing the forward loss.
    Forward: identity (returns input unchanged).
    Backward: passes grad through + provides gradient=1 for aux_loss.
    """
    @staticmethod
    def forward(ctx, x, aux_loss):
        ctx.save_for_backward(aux_loss)
        return x

    @staticmethod
    def backward(ctx, grad):
        (aux_loss,) = ctx.saved_tensors
        return grad, torch.ones(1, dtype=aux_loss.dtype, device=aux_loss.device)


class AddHiddenZLossToBackward(torch.autograd.Function):
    """Inject a hidden z-loss gradient without changing the forward loss."""
    @staticmethod
    def forward(ctx, x, z_loss):
        ctx.save_for_backward(z_loss)
        return x

    @staticmethod
    def backward(ctx, grad):
        (z_loss,) = ctx.saved_tensors
        return grad, torch.ones(1, dtype=z_loss.dtype, device=z_loss.device)


def compute_hidden_z_loss(
    hidden_states: torch.Tensor,
    coeff: float = 1e-7,
) -> torch.Tensor:
    """Compute hidden z-loss on transformer hidden states.

    Hidden z-loss on absolute values:
        log_z = logsumexp(|hidden_states|, dim=-1)
        z_loss = (coeff * log_z^2).mean()

    Applied BEFORE the final layernorm; simply added to the total loss.

    In FSDP, the total loss is already divided by gradient_accumulation_steps
    in train_step, so we don't need to divide here.

    Args:
        hidden_states: [B, seq_len, hidden_size] LLM output before final norm.
        coeff: Hidden z-loss coefficient (e.g. 1e-7).

    Returns:
        Scalar z-loss tensor (with gradient).
    """
    # logsumexp on absolute values
    # Operates in the input dtype (bf16), no float32 upcast
    # hidden_states shape: [B, seq_len, hidden_size]
    log_z = torch.logsumexp(hidden_states.abs(), dim=-1)  # [B, seq_len]
    z_loss = (coeff * log_z ** 2).mean()
    return z_loss


def compute_router_z_loss(
    router_logits_list: list,
    z_loss_coeff: float = 0.2,
    moe_loss_coeff: float = 0.0005,
) -> torch.Tensor:
    """Compute MoE router z-loss from captured router logits.

    Router z-loss:
        log_z = logsumexp(router_logits, dim=-1)
        z_loss = sum(log_z^2) / (num_groups * tokens_per_group)

    Scaling: z_loss * only_z_loss_coeff * moe_loss_coeff (per layer),
    summed across all layers,
    which in FSDP is handled by train_step's gradient_accumulation division.

    Args:
        router_logits_list: List of router logit tensors, one per MoE layer.
            Each has shape [num_tokens, n_routed_experts].
        z_loss_coeff: Router z-loss coefficient.
            Default 0.2 for both understand and generate SFT tasks.
        moe_loss_coeff: MoE loss scaling coefficient.
            Default 0.0005.

    Returns:
        Scalar z-loss tensor (with gradient). Returns 0 if no logits provided.
    """
    if not router_logits_list:
        return torch.tensor(0.0)

    total_z_loss = torch.tensor(0.0, device=router_logits_list[0].device)

    for idx, logits in enumerate(router_logits_list):
        # logits shape: [num_tokens, n_routed_experts] (already truncated at capture)
        num_tokens = logits.shape[0]
        if num_tokens == 0:
            continue
        log_z = torch.logsumexp(logits.float(), dim=-1)  # [num_tokens]
        layer_z_loss = (log_z ** 2).sum() / num_tokens
        # Apply per-layer scaling: z_loss_coeff * moe_loss_coeff
        total_z_loss = total_z_loss + layer_z_loss * z_loss_coeff * moe_loss_coeff

    return total_z_loss


def compute_router_z_loss_display(
    router_logits_list: list,
    z_loss_coeff: float = 0.2,
    moe_loss_coeff: float = 0.0005,
) -> float:
    """Compute the display-only router z-loss (no gradient) shown in the "z loss" metric.

    Displayed "z loss" = sum_layers(z_loss_per_layer * z_coeff * moe_coeff).
    (The per-microbatch normalization cancels out in the aggregation.)

    This function computes the same value WITHOUT gradient, for logging only.

    Returns:
        Float display value.
    """
    if not router_logits_list:
        return 0.0

    # Accumulate on-device tensors (not python floats) to avoid fp32 rounding drift.
    total = torch.tensor(0.0, dtype=torch.float32, device=router_logits_list[0].device)
    with torch.no_grad():
        for logits in router_logits_list:
            num_tokens = logits.shape[0]
            if num_tokens == 0:
                continue
            log_z = torch.logsumexp(logits.float(), dim=-1)
            layer_z = (log_z ** 2).sum() / num_tokens
            total = total + layer_z * z_loss_coeff * moe_loss_coeff
    return total.item()


def compute_load_balance_loss(
    router_outputs_list: list,
    moe_loss_coeff: float = 0.0005,
    zero_expert_num: int = 128,
    target_topk: int = 8,
    only_adapt_ffn_bias: bool = True,
) -> tuple:
    """Compute expert load balance loss from captured router outputs.

    Dynamic expert load balance loss,
    which is used when zero_expert_num is set:

        p = mean(router_prob, dim=tokens)  # per-expert mean probability
        f = mean(expert_mask, dim=tokens)  # per-expert selection fraction

        ffn_coeff = f[:ffn_num] * ffn_num / target_topk
        zero_coeff = f[ffn_num:] * zero_num / (topk - target_topk)
        if only_adapt_ffn_bias:
            zero_coeff = mean(zero_coeff) repeated
        lb_loss = sum(cat(ffn_coeff, zero_coeff) * p)

    Config: zero-expert-num=128, target-topk=8, moe-topk=12, only-adapt-ffn-bias=True

    The loss is scaled by moe_loss_coeff and summed across all layers.

    Note: absolute load balance loss values depend on how the reference
    uses loss-free-balance-rate=0.1 with dynamic-update-loss-free-bias to
    rebalance routing during training. FSDP does not implement this, so
    routing may be more concentrated. The formula is matched; the numerical
    difference comes from different routing distributions.

    Args:
        router_outputs_list: List of (scores, topk_indices, n_experts, topk)
            tuples, one per MoE layer. scores and topk_indices are detached.
        moe_loss_coeff: MoE loss scaling coefficient.
        zero_expert_num: Number of zero/identity experts.
        target_topk: Target number of FFN experts per token.
        only_adapt_ffn_bias: If True, average zero-expert coefficients.

    Returns:
        Tuple of (loss_tensor, display_value):
        - loss_tensor: Scalar tensor for backward (None if no data)
        - display_value: Float for the "expert load balance loss" metric
    """
    if not router_outputs_list:
        return None, 0.0

    display_total = 0.0

    for scores, topk_indices, n_experts, topk in router_outputs_list:
        num_tokens = scores.shape[0]
        if num_tokens == 0:
            continue

        ffn_expert_num = n_experts - zero_expert_num

        # p: per-expert mean routing probability [n_experts]
        p = scores.mean(dim=0)  # [n_experts]

        # f: per-expert fraction of tokens selecting each expert [n_experts]
        # Build expert_mask via scatter (binary mask, no double counting)
        expert_mask = torch.zeros(
            num_tokens, n_experts, device=scores.device, dtype=torch.float32
        )
        expert_mask.scatter_(1, topk_indices.long(), 1.0)  # [N, E] binary
        f = expert_mask.mean(dim=0)  # [n_experts]

        # Dynamic load balance loss:
        # FFN experts: scale by ffn_num / target_topk
        ffn_coeff = f[:ffn_expert_num] * ffn_expert_num / target_topk
        # Zero experts: scale by zero_num / (topk - target_topk)
        zero_coeff = f[ffn_expert_num:] * zero_expert_num / (topk - target_topk)

        if only_adapt_ffn_bias:
            # All zero experts get the same averaged coefficient
            avg_zero_coeff = zero_coeff.mean()
            zero_coeff = torch.full_like(zero_coeff, avg_zero_coeff)

        f_coeff = torch.cat([ffn_coeff, zero_coeff])
        lb_loss_val = torch.sum(f_coeff * p)

        display_total += lb_loss_val.item() * moe_loss_coeff

    # Return None for backward (load balance loss cannot flow gradients through
    # the HF router's detached scores), and the display value.
    return None, display_total
