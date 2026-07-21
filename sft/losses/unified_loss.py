# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unified loss computation for LongCat-Next FSDP SFT training.

Merges ce_loss.py (understand) and depth_ce_loss.py (generate) into a single
module that supports understand, generate, and unify (joint) training.

Design: this function performs ONLY tensor math — no access to FSDP-sharded
modules (lm_head, visual_head, embed_tokens).  All FSDP-triggering ops are
done in SFTTrainingWrapper.forward() while all ranks are still in lockstep.
In unify mode different ranks may have different pack contents (one rank's
pack has generation images, another's does not), so any conditional FSDP
access inside this function would deadlock.  Passing in pre-computed tensors
avoids that entirely.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from data.tokenize_utils import IMG_PAD_ID, IMG_END_ID

PARALLEL_DECODE_LOSS_WEIGHT = 0.25


class _ChunkedSumCrossEntropy(torch.autograd.Function):
    """Cross-entropy with a chunked logsumexp reduction.

    The exp(logits) values are summed in a fixed number of chunks (CE_CHUNK_SUM_CNT)
    rather than a single reduction. Fixing the summation order this way makes the
    logsumexp reproducible across runs. Computed in fp32 with a max-subtraction for
    numerical stability.
    """
    @staticmethod
    def forward(ctx, logits_bf16, target):
        import os
        chunk_cnt = int(os.environ.get("CE_CHUNK_SUM_CNT", "32"))
        logits = logits_bf16.float()
        logits_max = torch.max(logits, dim=-1)[0]
        logits.sub_(logits_max.unsqueeze(-1))
        arange = torch.arange(logits.size(0), device=logits.device)
        predicted_logits = logits[arange, target].clone().contiguous()
        torch.exp(logits, out=logits)
        exp_logits = logits
        s, v = exp_logits.shape
        pad = (v // chunk_cnt + 1) * chunk_cnt - v if v % chunk_cnt != 0 else 0
        if pad > 0:
            exp_3d = torch.cat([exp_logits.unsqueeze(1),
                                torch.zeros(s, 1, pad, device=exp_logits.device, dtype=exp_logits.dtype)], dim=-1)
        else:
            exp_3d = exp_logits.unsqueeze(1)
        vp = v + pad
        sum_exp = exp_3d.view(s, 1, chunk_cnt, vp // chunk_cnt).sum(-1).sum(-1).squeeze(1)
        loss = torch.log(sum_exp) - predicted_logits
        exp_logits.div_(sum_exp.unsqueeze(-1))
        ctx.save_for_backward(exp_logits, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors
        grad_input = softmax
        grad_2d = grad_input.view(-1, softmax.size(-1))
        arange = torch.arange(grad_2d.size(0), device=grad_2d.device)
        grad_2d[arange, target] -= 1.0
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None


def compute_unified_loss(
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    hidden_states: Optional[torch.Tensor] = None,
    visual_mask: Optional[torch.Tensor] = None,
    visual_ids: Optional[torch.Tensor] = None,
    loss_visual_mask: Optional[torch.Tensor] = None,
    img_end_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    num_real_samples: Optional[int] = None,
    task: str = "understand",
    # Pre-computed by caller (SFTTrainingWrapper.forward) while all ranks in sync.
    text_logits: Optional[torch.Tensor] = None,
    depth_per_token: Optional[torch.Tensor] = None,
    has_image_loss: bool = False,
    img_end_mask_in_loss_space: Optional[torch.Tensor] = None,
    level_loss_sums: Optional[Dict] = None,
    depth_mask_for_scatter: Optional[torch.Tensor] = None,
) -> Dict:
    # --- Text CE (from pre-computed logits — no FSDP access) ---
    if text_logits is not None:
        flat_logits = text_logits.view(-1, text_logits.shape[-1])
    else:
        raise RuntimeError("text_logits must be provided (computed in caller)")
    vocab_size = flat_logits.shape[-1]

    if loss_visual_mask is None:
        loss_visual_mask = torch.zeros_like(labels, dtype=torch.bool) if task == "understand" else ((labels == IMG_PAD_ID) | (labels == IMG_END_ID))
    if img_end_mask is None:
        img_end_mask = torch.zeros_like(labels, dtype=torch.bool) if task == "understand" else (labels == IMG_END_ID)

    has_image_loss = loss_visual_mask.any().item()

    text_mask = loss_mask.clone()
    if has_image_loss:
        text_mask[loss_visual_mask] = 0.0

    flat_labels = labels.view(-1)
    flat_text_mask = text_mask.view(-1).float()
    flat_loss_mask = loss_mask.view(-1).float()

    per_token_text_ce = _ChunkedSumCrossEntropy.apply(flat_logits, flat_labels)

    # --- 2. Depth CE (pre-computed by caller, pure scatter here) ---
    per_token_depth_ce = torch.zeros_like(per_token_text_ce)
    level_loss_sums = level_loss_sums if level_loss_sums is not None else {}
    num_img_tokens = 0

    # NOTE: depth CE is fully computed by the caller (SFTTrainingWrapper.forward)
    # while all ranks are in lockstep — see train.py Step 5. Here we ONLY scatter
    # the pre-computed per-token depth CE back into the flat [seq] layout. No FSDP
    # module access (visual_head / embed_tokens / offsets) happens in this file:
    # doing so on only the ranks whose pack contains a generation image would
    # desync FSDP2 all-gather/reduce-scatter and hang in unify mode.
    if has_image_loss and depth_per_token is not None and depth_mask_for_scatter is not None:
        num_img_tokens = depth_mask_for_scatter.sum().item()
        if num_img_tokens > 0:
            per_token_depth_ce[depth_mask_for_scatter] = depth_per_token[:num_img_tokens].to(per_token_depth_ce.dtype)

    # --- 3+4. Aggregate ---
    if has_image_loss:
        per_token_loss = per_token_text_ce * flat_text_mask + per_token_depth_ce
        flat_lvm = loss_visual_mask.view(-1).float()
        effective_loss_mask = torch.clamp(flat_loss_mask + flat_lvm, max=1.0)
    else:
        per_token_loss = per_token_text_ce
        effective_loss_mask = flat_loss_mask

    masked_loss_sum = torch.sum(torch.masked_select(per_token_loss, effective_loss_mask.to(torch.bool)))
    token_count = effective_loss_mask.sum()
    total_loss = masked_loss_sum / (1e-10 + token_count)

    # --- 5. Metrics ---
    with torch.no_grad():
        text_loss_sum = (per_token_text_ce * flat_text_mask).sum().item()
        text_token_num = flat_text_mask.sum().item()
        image_loss_sum = per_token_depth_ce.sum().item()
        image_token_num = float(num_img_tokens)
        total_loss_sum = masked_loss_sum.item()
        total_token_count = token_count.item()
        if num_real_samples is not None:
            sample_count = num_real_samples
        elif cu_seqlens is not None and cu_seqlens.numel() > 1:
            sample_count = cu_seqlens.shape[0] - 1
        else:
            sample_count = 1

    metrics = {
        "loss_sum": total_loss_sum, "token_count": total_token_count,
        "text_loss_sum": text_loss_sum, "text_token_num": text_token_num,
        "image_loss_sum": image_loss_sum, "image_token_num": image_token_num,
        "sample_count": sample_count,
    }
    metrics.update(level_loss_sums)

    return {"loss": total_loss, "metrics": metrics}


def _compute_depth_loss(
    visual_head, hidden_states, vq_labels, embed_tokens,
    codebook_sizes, offset_vals, img_end_mask_in_img=None,
):
    num_levels = len(codebook_sizes)
    device = hidden_states.device
    n_img = hidden_states.shape[0]
    has_end = img_end_mask_in_img is not None and img_end_mask_in_img.any()

    if num_levels > 1:
        prev_ids = vq_labels[:, : num_levels - 1]
        if has_end:
            safe_ids = prev_ids.clone()
            safe_ids[img_end_mask_in_img] = 0
            level_emb_all = embed_tokens(safe_ids)
            level_emb_all[img_end_mask_in_img] = 0.0
        else:
            level_emb_all = embed_tokens(prev_ids)
        cumsum_stack = torch.cumsum(level_emb_all, dim=1)
    else:
        cumsum_stack = torch.empty(n_img, 0, hidden_states.shape[1], device=device, dtype=hidden_states.dtype)

    depth_input = torch.cat([hidden_states.unsqueeze(1), cumsum_stack], dim=1)

    if hasattr(visual_head, 'hidden_norm') and visual_head.transformer_ffn_scale > 0:
        depth_input = visual_head.hidden_norm(depth_input)
        depth_input = visual_head.hidden_proj(depth_input)
    for layer in visual_head.transformer_layers:
        depth_input = layer(depth_input)
    depth_input = visual_head.headnorm(depth_input)

    level_loss_sums = {}
    level_ce_list = []
    level_sum_list = []
    for level in range(num_levels):
        level_logits = visual_head.heads[level](depth_input[:, level].contiguous())
        local_labels = vq_labels[:, level] - offset_vals[level].to(device)
        local_labels = local_labels.long()
        level_ce = F.cross_entropy(level_logits.float(), local_labels, reduction="none")
        level_ce_list.append(level_ce)
        level_sum_list.append(level_ce.sum())

    _sums_cpu = torch.stack(level_sum_list).detach().cpu().tolist()
    for level, s in enumerate(_sums_cpu):
        level_loss_sums[f"paraDec_level-{level + 1}_loss_sum"] = s

    level_ce_stack = torch.stack(level_ce_list, dim=0)
    per_token_depth_ce = (level_ce_stack * PARALLEL_DECODE_LOSS_WEIGHT).sum(dim=0)
    return per_token_depth_ce, level_loss_sums


def aggregate_metrics(accumulated_metrics, device, gradient_accumulation_steps):
    import torch.distributed as dist
    sum_keys = ["loss_sum", "token_count", "text_loss_sum", "text_token_num",
                "image_loss_sum", "image_token_num", "sample_count"]
    for i in range(1, 9):
        sum_keys.append(f"paraDec_level-{i}_loss_sum")
    vals = [float(accumulated_metrics.get(k, 0)) for k in sum_keys]
    tensor = torch.tensor(vals, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_sums = {k: tensor[i].item() for i, k in enumerate(sum_keys)}

    aux_avg_keys = ["hidden_z_loss", "router_z_loss", "expert load balance loss", "z loss"]
    aux_vals = []; aux_present_keys = []
    for k in aux_avg_keys:
        if k in accumulated_metrics:
            aux_vals.append(accumulated_metrics[k] / gradient_accumulation_steps)
            aux_present_keys.append(k)
    if aux_vals:
        aux_tensor = torch.tensor(aux_vals, dtype=torch.float64, device=device)
        dist.all_reduce(aux_tensor, op=dist.ReduceOp.SUM)
        aux_tensor /= dist.get_world_size()
        aux_avgs = {k: aux_tensor[i].item() for i, k in enumerate(aux_present_keys)}
    else:
        aux_avgs = {}

    global_total_tokens = global_sums["token_count"]
    global_text_tokens = global_sums["text_token_num"]
    global_img_tokens = global_sums["image_token_num"]
    lm_loss = global_sums["loss_sum"] / max(global_total_tokens, 1e-10)
    text_loss = global_sums["text_loss_sum"] / max(global_text_tokens, 1e-10)
    image_loss = global_sums["image_loss_sum"] / max(global_img_tokens, 1e-10)

    result = {
        "total_loss": lm_loss, "text_loss": text_loss,
        "o_text_loss": text_loss, "image_token_loss": image_loss,
        "o_image_token_loss": image_loss,
        "total_token_num": int(global_total_tokens),
        "text_token_num": int(global_text_tokens),
        "image_token_num": int(global_img_tokens),
        "sample_count": int(global_sums["sample_count"]),
        "image_token_token_ratio": global_img_tokens / max(global_total_tokens, 1),
        "text_token_ratio": global_text_tokens / max(global_total_tokens, 1),
        "image_token_weight_ratio": global_img_tokens / max(global_total_tokens, 1),
        "text_weight_ratio": global_text_tokens / max(global_total_tokens, 1),
    }
    for i in range(1, 9):
        key = f"paraDec_level-{i}_loss_sum"
        if global_sums.get(key, 0) > 0 or global_img_tokens > 0:
            result[f"paraDec_level-{i}_loss"] = global_sums.get(key, 0) / max(global_img_tokens, 1e-10)
    for k in ["hidden_z_loss", "router_z_loss", "expert load balance loss", "z loss"]:
        if k in aux_avgs:
            result[k] = aux_avgs[k]
    return result
