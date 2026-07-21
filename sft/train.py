# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Main training entry point for LongCat-Next FSDP2 SFT.

Supports three tasks:
- understand: Image+Text -> Text (CE loss)
- generate: Text -> Image (depth CE loss)
- unify: mixed understanding + generation packed into the same sequences,
  with the loss routed per token to the right head

Uses PyTorch 2.5+ FSDP2 (fully_shard) for distributed training. The load path
follows the standard PyTorch FSDP2 pattern for large models:
1. Rank 0: load model on CPU via from_pretrained
2. Other ranks: create model on meta device
3. Apply freeze + FSDP2 sharding
4. Distribute weights via set_model_state_dict(broadcast_from_rank0=True)
5. Broadcast non-persistent buffers

Image processing architecture:
- Dataset: CPU-only preprocessing (load + resize/normalize -> pixel_values)
- Model forward: ViT encode -> VQ quantize -> embedding lookup -> LLM forward
This ensures ViT weights are managed by FSDP and auto-saved in checkpoints.

Usage:
    torchrun --nproc_per_node=8 train.py --task understand --model_path /path/to/model --data_path /path/to/data.jsonl
"""

import os
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

# Compat: torch.nn.RMSNorm added in PyTorch 2.4+
if not hasattr(torch.nn, 'RMSNorm'):
    class _RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, **kwargs):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
            self.eps = eps
        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return (self.weight * x).to(dtype)
    torch.nn.RMSNorm = _RMSNorm

try:
    from torch.distributed import DeviceMesh
except ImportError:
    from torch.distributed.device_mesh import DeviceMesh
from torchdata.stateful_dataloader import StatefulDataLoader

from config import TrainConfig
from data.image_processing import ImagePreprocessor
from data.understand_dataset import UnderstandPackedDataset
from data.generate_dataset import GeneratePackedDataset
from data.unify_dataset import UnifyPackedDataset
from model.model_loader import (
    load_model_meta,
    load_fsdp2_weights_manual,
    fix_non_persistent_buffers,
    freeze_for_understand, freeze_for_generate,
)
from model.fsdp_utils import break_parameter_sharing, setup_fsdp2, save_checkpoint, load_checkpoint
from losses.unified_loss import compute_unified_loss, aggregate_metrics, _compute_depth_loss
from train_utils import (
    create_optimizer, create_scheduler,
    TrainingLogger, TensorBoardLogger, compute_params_norm,
)

_log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _allreduce_metrics(
    accumulated_metrics: dict,
    accumulated_tokens: int,
    accumulated_samples: int,
    device: torch.device,
    gradient_accumulation_steps: int,
) -> dict:
    """All-reduce display metrics across DP ranks.

    Delegates to unified_loss.aggregate_metrics which correctly implements
    Sums metrics across DP ranks and divides by the scale.
    pipeline using raw sums and counts (not averages).
    """
    return aggregate_metrics(
        accumulated_metrics=accumulated_metrics,
        device=device,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


def setup_distributed():
    """Initialize distributed training environment."""
    import datetime
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=60),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


class SFTTrainingWrapper(nn.Module):
    """Wraps LongcatNextForCausalLM with a training-friendly forward method.

    The original HF model's forward() is inference-oriented. This wrapper
    provides a clean forward that:
    1. Runs ViT encoding on pixel_values (in the computation graph)
    2. Builds input embeddings with visual tokens
    3. Forwards through the LLM backbone
    4. Computes task-specific loss
    """

    def __init__(self, model, config=None):
        super().__init__()
        self.model = model  # LongcatNextForCausalLM
        self.config = config
        # Resolve module references once here, before FSDP2 sharding. In unify mode
        # different ranks take different loss paths, and a lazy FSDP2 property access
        # at runtime could deadlock. Store the modules only — do NOT read
        # .weight/.data here, since non-rank-0 models are still on the meta device
        # (any tensor op would raise "Cannot copy out of meta tensor"). Values are
        # read at runtime after FSDP2 materializes the parameters.
        self._lm_head_module = model.lm_head
        self._visual_head_module = model.visual_head
        _inner_model = model.model if hasattr(model, "model") else model
        self._embed_tokens_module = _inner_model.embed_tokens
        # visual_offset_vals is a buffer; resolve it lazily (may be meta now).
        self._visual_offset_vals_buffer = getattr(_inner_model, "visual_offset_vals", None)
        self._activation_checkpointing = getattr(config, 'activation_checkpointing', False)

        # Z-loss configuration (from TrainConfig)
        self.hidden_z_loss_coeff = getattr(config, 'hidden_z_loss_coeff', 0.0) if config else 0.0
        self.router_z_loss_coeff = getattr(config, 'router_z_loss_coeff', 0.0) if config else 0.0
        self.moe_loss_coeff = getattr(config, 'moe_loss_coeff', 0.0005) if config else 0.0005
        # load-balance-loss-type: "both" = z-loss + lb loss, "z_loss" = z-loss only
        # understand uses "z_loss", generation uses "both"
        self.load_balance_loss_type = getattr(config, 'load_balance_loss_type', 'both') if config else 'both'
        self._use_activation_checkpointing = getattr(config, 'activation_checkpointing', False) if config else False

        # Router logits/output capture for MoE z-loss and load balance loss.
        # Forward hooks on each LongcatFlashTopkRouter capture:
        # 1. router_logits (recomputed for gradient flow) → z-loss
        # 2. (topk_weights, topk_indices) → load balance loss
        self._router_logits = []
        self._router_outputs = []  # list of (scores, topk_indices, n_experts, topk)
        self._router_hooks = []
        # List of MoE routers, in the SAME order as router_outputs is appended.
        # Used by LossFreeBalanceManager for per-step bias update.
        self._routers: list = []
        # Always setup hooks: we need load balance loss even if z-loss is disabled.
        self._setup_router_hooks()

        # Patch LongcatFlashRMSNorm.forward to a form with a specific autograd graph.
        # Two RMSNorm formulations are numerically equal in the forward pass but build
        # different autograd graphs, so their backward gradients differ:
        #   HF naive : x.to(fp32) -> var -> x_fp32 * rsqrt -> .to(bf16) -> weight * out
        #   RMSNormCore: var = x.to(fp32).pow(2).mean(); x = x * rsqrt(var); x.to(bf16); weight * x
        # We use the RMSNormCore form so the backward gradients are the ones we want.
        try:
            import transformers.models.longcat_flash.modeling_longcat_flash as _lf_norm_core
            def _rmsnorm_core_forward(self, hidden_states):
                _eps = getattr(self, 'variance_epsilon', None) or getattr(self, 'eps', 1e-6)
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + _eps)
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                output = self.weight * hidden_states
                return output
            _lf_norm_core.LongcatFlashRMSNorm.forward = _rmsnorm_core_forward
        except Exception as _e:
            _log.warning(f"could not patch LongcatFlashRMSNorm.forward to RMSNormCore: {_e}")

        # Patch apply_rotary_pos_emb_interleave to run in fp32 (HF defaults to bf16).
        # This changes the numerics and must always be active. cos/sin are recomputed
        # in fp32 from inv_freq.
        try:
            import transformers.models.longcat_flash.modeling_longcat_flash as _lf_mod
            try:
                _wrapper_rot = self
                def _rotate_half_rot(x):
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat((-x2, x1), dim=-1)

                def _fp32_apply_rotary_uncond(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                    # Ignore the incoming bf16 cos/sin; recompute fp32 cos/sin from inv_freq.
                    b, h, s, d = q.shape
                    bk, hk, sk, dk = k.shape
                    _rot = _wrapper_rot.model.model.rotary_emb
                    # Recompute inv_freq on CPU (GPU pow() can differ by 1 ULP).
                    # theta is read from rotary_emb.config so the base stays in sync with the
                    # loaded model (longcat-next: 1e7) — don't hard-code.
                    _rope_dim = d
                    _theta = float(_rot.config.rope_theta)
                    _inv_freq = (1.0 / (_theta ** (torch.arange(0, _rope_dim, 2, dtype=torch.float32) / _rope_dim))).to(q.device)
                    _scaling = getattr(_rot, 'attention_scaling', 1.0)
                    _t = torch.arange(s, device=q.device, dtype=torch.float32)
                    _freqs = torch.einsum('i,j->ij', _t, _inv_freq)
                    _emb = torch.cat((_freqs, _freqs), dim=-1)
                    cos_4d = (_emb.cos() * _scaling).unsqueeze(0).unsqueeze(0)  # [1,1,s,d] fp32
                    sin_4d = (_emb.sin() * _scaling).unsqueeze(0).unsqueeze(0)
                    # interleave reordering
                    q_il = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
                    k_il = k.view(bk, hk, sk, dk // 2, 2).transpose(4, 3).reshape(bk, hk, sk, dk)
                    q_embed = (q_il.float() * cos_4d + _rotate_half_rot(q_il.float()) * sin_4d).to(q.dtype)
                    k_embed = (k_il.float() * cos_4d + _rotate_half_rot(k_il.float()) * sin_4d).to(k.dtype)
                    return q_embed, k_embed

                _lf_mod.apply_rotary_pos_emb_interleave = _fp32_apply_rotary_uncond
            except Exception as _e_rot:
                _log.warning(f"could not patch apply_rotary (uncond): {_e_rot}")

            # Patch LongcatFlashMoE.forward: delegate reshape to moe() (the deterministic
            # MoE replacement handles view(-1)/view(orig_shape) internally). Router logits
            # are captured separately by the router-forward patch below.
            try:
                _MoECls = _lf_mod.LongcatFlashMoE

                def _patched_moe_forward(moe_self, hidden_states):
                    orig_shape = hidden_states.shape
                    topk_indices, topk_weights = moe_self.router(hidden_states)
                    hidden_states = moe_self.moe(hidden_states, topk_indices, topk_weights)
                    return hidden_states

                _MoECls.forward = _patched_moe_forward
            except Exception as _e:
                _log.warning(f"[FSDP] could not patch LongcatFlashMoE.forward: {_e}")
        except Exception as _e:
            _log.warning(f"[FSDP] Could not patch modeling forwards (rotary/moe): {_e}")
        # === end

    def _setup_router_hooks(self):
        """Register forward hooks on MoE routers to capture router logits and outputs.

        The HF LongcatFlashTopkRouter computes:
            router_logits = F.linear(hidden_states.float(), classifier.weight.float())
            scores = router_logits.softmax(dim=-1)
            topk_indices = get_topk_indices(scores)
            topk_weights = scores.gather(1, topk_indices) * routed_scaling_factor
        but doesn't expose router_logits or scores in its output.

        We hook on each router module to:
        1. Recompute router_logits from (input, weight) for z-loss gradient flow
        2. Capture scores and topk_indices for load balance loss computation

        Hook is registered after FSDP setup, so it works with sharded parameters.
        """
        inner_model = self.model.model  # LongcatNextModel
        import types
        for layer_idx, layer in enumerate(inner_model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                router = layer.mlp.router
                _orig_router_forward = router.forward
                # Record router refs for LossFreeBalanceManager.
                self._routers.append(router)

                # Monkey-patch router forward to match the reference routing:
                # softmax(fp32) → to(bf16) → topk(bf16) → weights(bf16) * scale
                def _make_patched_router_fwd(orig_fwd, wrapper_self):
                    def _patched_fwd(hidden_states):
                        module = orig_fwd.__self__ if hasattr(orig_fwd, '__self__') else None

                        hidden_flat = hidden_states.view(-1, module.config.hidden_size) if module else hidden_states
                        _w = module.classifier.weight.float()
                        _b = module.classifier.bias.float() if module.classifier.bias is not None else None
                        router_logits = torch.nn.functional.linear(
                            hidden_flat.float(), _w, _b
                        )
                        wrapper_self._router_logits.append(router_logits)

                        # Greedy top-k with an fp32 correction bias:
                        # topk(bf16_score + fp32_bias), then gather weights from bf16_score
                        scores = router_logits.softmax(dim=-1)  # fp32
                        scores_bf16 = scores.to(torch.bfloat16)
                        scores_for_topk = scores_bf16 + module.e_score_correction_bias.float().unsqueeze(0)
                        topk_indices = torch.topk(scores_for_topk, k=module.top_k, dim=-1)[1]
                        topk_weights = scores_bf16.gather(1, topk_indices)
                        topk_weights = topk_weights * module.routed_scaling_factor


                        # Capture for load balance loss
                        with torch.no_grad():
                            wrapper_self._router_outputs.append(
                                (scores.detach(), topk_indices.detach(), module.n_routed_experts, module.top_k)
                            )

                        return topk_indices, topk_weights
                    return _patched_fwd

                router.forward = _make_patched_router_fwd(_orig_router_forward, self)
                self._router_hooks.append(None)  # placeholder

        if not getattr(self, '_logged_router_hooks', False):
            self._logged_router_hooks = True
            _log.info(f"Registered {len(self._router_hooks)} router hooks "
                  f"(z-loss + load balance loss)")

    def forward(
        self,
        input_ids,
        pixel_values,
        image_grid_thw,
        visual_mask,
        position_ids,
        labels,
        loss_mask,
        cu_seqlens=None,
        num_real_samples=None,
        loss_visual_mask=None,
        img_end_mask=None,
        target_visual_mask=None,
        task="understand",
    ):
        """Training forward pass with online ViT encoding.

        IMPORTANT: All sub-module calls go through normal __call__ (not direct
        .forward()) so that FSDP2's pre_forward/post_forward hooks fire correctly.
        Bypassing __call__ (e.g. Parent.forward(child, ...)) skips FSDP2 hooks
        and causes NCCL deadlocks.

        Args:
            input_ids: [B, seq_len] token IDs with IMG_PAD placeholders.
            pixel_values: [total_patches, C, pH, pW] raw image patches, or empty.
            image_grid_thw: [num_images, 3] grid dimensions per image.
            visual_mask: [B, seq_len] bool mask where IMG_PAD tokens are in
                input_ids. Used for embedding replacement.
            position_ids: [B, seq_len] position IDs.
            labels: [B, seq_len] target token IDs.
            loss_mask: [B, seq_len] float mask for loss computation.
            cu_seqlens: [num_samples+1] cumulative sequence lengths for packing.
            loss_visual_mask: [B, seq_len] bool mask where IMG_PAD and img_end
                tokens are in labels. Used for depth CE loss hidden state
                selection (generate task only).
            img_end_mask: [B, seq_len] bool mask where img_end tokens are in
                labels. Used to distinguish IMG_PAD vs img_end inside depth CE.
            task: "understand" or "generate".

        Returns:
            Dict with "loss" and "metrics".
        """
        causal_lm = self.model  # LongcatNextForCausalLM
        inner_model = causal_lm.model  # LongcatNextModel (-> LongcatFlashNgramModel)

        # Step 1: ViT encode pixel_values -> VQ token IDs.
        # FSDP2 requires every rank to invoke the sharded visual_model /
        # visual_bridge_model the same way, even on text-only batches — otherwise
        # the NCCL collectives desync and deadlock. On no-image ranks we therefore
        # run a dummy forward, and its output MUST be connected to the loss (as a
        # 0-weight term) so backward triggers the matching reduce-scatter. Simply
        # discarding the dummy output would desync the backward collectives.
        has_images = pixel_values is not None and pixel_values.numel() > 0
        _rank_fwd = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        visual_ids = None
        # `target_visual_mask` marks generation (trainable) image-pad positions.
        # All three datasets emit it, so the forward never branches on `task`:
        #   understand → all-False (no generation targets)
        #   generate   → == visual_mask (every image is a target)
        #   unify      → only the generation images' pads
        # If a caller omits it, treat it as "no generation targets".
        if target_visual_mask is None:
            target_visual_mask = torch.zeros_like(visual_mask)
        dummy_visual_loss = None  # Will hold 0-contribution loss from dummy forwards
        # NOTE: under autocast(bf16), LayerNorm/RMSNorm still run in fp32, whereas a
        # plain bf16 model runs them in bf16 — a ~1e-3 difference that the RQ cascade
        # amplifies into mismatched visual_ids. The ViT also downcasts its input on
        # the first line (`hidden_states = pixel_values.to(bfloat16)`), which would
        # force the norm layers to bf16. So we keep pixel_values in fp32 and rely on
        # autocast to upcast the norm layers back to fp32.
        if has_images and pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if has_images:
                visual_ids = inner_model.get_visual_ids(
                    pixel_values=pixel_values,
                    visual_grid_thw=image_grid_thw,
                    offset=True,
                )
            else:
                # No-image rank: run get_visual_ids on a dummy zero input so the
                # sharded ViT forward (and its collectives) stays identical on every
                # rank. In unify training some ranks have images while others don't
                # in the same step, so this keeps FSDP2 from hanging.
                device = next(inner_model.parameters()).device
                # Minimal valid input: t=1, h=2, w=2 → 4 patches (multiple of
                # spatial_merge_unit=4), patch_dim = 3*2*14*14 = 1176.
                dummy_pv = torch.zeros(4, 1176, dtype=torch.float32, device=device)
                dummy_thw = torch.tensor([[1, 2, 2]], dtype=torch.long, device=device)
                inner_model.get_visual_ids(
                    pixel_values=dummy_pv,
                    visual_grid_thw=dummy_thw,
                    offset=True,
                )

        # Step 2: Build text embeddings via ngram_embeddings
        # Zero out image placeholder positions to avoid embedding lookup on special tokens
        safe_input_ids = input_ids.clone()
        safe_input_ids[visual_mask] = 0
        # Call ngram_embeddings through __call__ — triggers FSDP2 hooks for
        # each embedder/word_embeddings inside NgramEmbedding.forward
        inputs_embeds = inner_model.ngram_embeddings(safe_input_ids)

        # Step 3: Replace image placeholder positions with visual embeddings.
        # visual_embedding_layer is FSDP2-sharded, so all ranks must call it even
        # with no images (same reason as Step 1). We inline the HF
        # `get_visual_embeddings` here (instead of a separate no_grad call) so the
        # per-level / sum-before-bridge / post-bridge tensors come from the real
        # forward path — a separate call would see reshard / autocast-off state and
        # produce different values.
        if visual_ids is not None and visual_mask.any():
            _per_level = inner_model.embed_tokens(visual_ids)                    # (N, 8, H)
            # Accumulate the 8 VQ levels with a chunked loop (bf16). The exact loop
            # form fixes the float summation order so the result is reproducible.
            _pl_unsqueezed = _per_level.unsqueeze(0)                              # (1, N, 8, H)
            _chunks = torch.chunk(_pl_unsqueezed, chunks=8, dim=-2)
            _sum_pre = torch.zeros_like(_chunks[0])
            for _i in range(8):
                _sum_pre += _chunks[_i]
            _sum_pre = _sum_pre.squeeze(-2).squeeze(0)                            # (N, H)
            # === end


            visual_embeddings = inner_model.visual_tokenizer.visual_embedding_layer(_sum_pre)
            n_vis_emb = visual_embeddings.shape[0]
            n_vis_mask = visual_mask.sum().item()
            if n_vis_emb != n_vis_mask:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                # Safety trim: handle any remaining mismatch (e.g. from edge
                # cases in packing/truncation).
                if n_vis_emb > n_vis_mask:
                    visual_embeddings = visual_embeddings[:n_vis_mask]
                    visual_ids = visual_ids[:n_vis_mask]
                else:
                    raise RuntimeError(
                        f"[rank={rank}] visual_embeddings({n_vis_emb}) < "
                        f"visual_mask({n_vis_mask}). "
                        f"grid_thw={image_grid_thw}")
            inputs_embeds[visual_mask] = visual_embeddings.to(inputs_embeds.dtype)
        else:
            # No-image rank: dummy pass through the same sharded path
            # (embed_tokens → sum → visual_embedding_layer) with a zero-length
            # visual_ids tensor, so the FSDP2 hooks fire identically to the real
            # branch above (see Step 1 for why this matters).
            _dummy_visual_ids = torch.zeros(0, 8, dtype=torch.long, device=inputs_embeds.device)
            _dummy_per_level = inner_model.embed_tokens(_dummy_visual_ids)  # [0, 8, H]
            _dummy_sum = _dummy_per_level.sum(dim=1)                        # [0, H]
            dummy_vel_out = inner_model.visual_tokenizer.visual_embedding_layer(_dummy_sum)  # [0, H]
            # Connect to computation graph for backward reduce-scatter
            dummy_vel_loss = 0.0 * dummy_vel_out.sum()
            if dummy_visual_loss is not None:
                dummy_visual_loss = dummy_visual_loss + dummy_vel_loss
            else:
                dummy_visual_loss = dummy_vel_loss


        # Step 4: Forward through the transformer backbone.
        # We call the decoder layers, rotary_emb, and norm directly via __call__
        # (which fires the FSDP2 all-gather/reshard hooks) instead of going through
        # LongcatNextModel.forward, which requires inference-only args like
        # multimodal_generation_status. This is equivalent to that forward.
        from transformers.masking_utils import create_causal_mask

        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )
        causal_mask = create_causal_mask(
            config=inner_model.config,
            input_embeds=inputs_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = inner_model.rotary_emb(hidden_states, position_ids)

        # Prepare flash-attention kwargs for varlen attention, so packed documents
        # don't attend across each other. We pass cu_seqlens explicitly via
        # FlashAttentionKwargs to force HF's varlen code path — position_ids is a
        # global arange (no per-sample reset), so HF's automatic packed-sequence
        # detection (which looks for position_ids==0 resets) would not trigger.
        flash_attn_kwargs = {}
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            # cu_seqlens: [num_samples+1], e.g. [0, 512, 1024, 8192]
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            flash_attn_kwargs = {
                "cu_seq_lens_q": cu_seqlens.to(dtype=torch.int32),
                "cu_seq_lens_k": cu_seqlens.to(dtype=torch.int32),
                "max_length_q": max_seqlen,
                "max_length_k": max_seqlen,
            }

        # Clear router captures from previous forward pass
        self._router_logits = []
        self._router_outputs = []


        for layer_idx, decoder_layer in enumerate(inner_model.layers):
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            # Unpack tuple return from DecoderLayer (HF convention: returns (hidden_states,) or (hidden_states, attn_weights))
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # Step 4b: Hidden z-loss, applied to hidden states BEFORE the final
        # layernorm. Injected via a custom autograd fn (identity in forward,
        # gradient in backward), so it does not change the reported loss scalar.
        hidden_z_loss = None
        if self.hidden_z_loss_coeff > 0:
            from losses.z_loss import compute_hidden_z_loss, AddHiddenZLossToBackward
            hidden_z_loss = compute_hidden_z_loss(hidden_states, self.hidden_z_loss_coeff)
            hidden_states = AddHiddenZLossToBackward.apply(hidden_states, hidden_z_loss)


        hidden_states = inner_model.norm(hidden_states)

        # Step 4c: Router z-loss (from captured router logits during layer forward)
        # z_loss = mean(logsumexp(router_logits)^2) per layer,
        #   scaled by only_z_loss_coeff * moe_loss_coeff, summed across layers.
        router_z_loss = None
        if self.router_z_loss_coeff > 0 and len(self._router_logits) > 0:
            from losses.z_loss import compute_router_z_loss
            router_z_loss = compute_router_z_loss(
                self._router_logits,
                z_loss_coeff=self.router_z_loss_coeff,
                moe_loss_coeff=self.moe_loss_coeff,
            )

        # Step 4d: Expert load balance loss (from captured router outputs).
        # lb_loss = sum(me*ce)*E/K per layer, scaled by moe_loss_coeff and summed
        # across layers; injected into backward without changing the loss scalar.
        # Enabled only when load_balance_loss_type == "both" (understand uses
        # "z_loss" = z-loss only, generate/unify use "both").
        load_balance_loss = None
        lb_loss_display = 0.0
        z_loss_display = 0.0
        if len(self._router_outputs) > 0 and self.load_balance_loss_type == "both":
            from losses.z_loss import compute_load_balance_loss
            load_balance_loss, lb_loss_display = compute_load_balance_loss(
                self._router_outputs,
                moe_loss_coeff=self.moe_loss_coeff,
                zero_expert_num=128,
                target_topk=8,
                only_adapt_ffn_bias=True,
            )
        # Compute z_loss display value (without gradient, just for logging)
        # Displayed: sum_layers(z_loss_per_layer * z_coeff * moe_coeff)
        if self.router_z_loss_coeff > 0 and len(self._router_logits) > 0:
            with torch.no_grad():
                from losses.z_loss import compute_router_z_loss_display
                z_loss_display = compute_router_z_loss_display(
                    self._router_logits,
                    z_loss_coeff=self.router_z_loss_coeff,
                    moe_loss_coeff=self.moe_loss_coeff,
                )

        # Step 4e: Slice visual_ids down to the *target* (generation) images.
        # `visual_ids` covers ALL images in the pack — understanding inputs AND
        # generation targets — in pad-token order, but the depth loss only trains
        # on generation images. For a pure-generate pack every image is a target;
        # for a MIXED pack the understanding-image rows come first and would be
        # mis-used as generation VQ labels. So we select the generation rows via
        # `target_visual_mask` (trainable image pads), whose order within
        # `visual_mask` matches visual_ids' row order. Task-agnostic:
        # understand→empty, generate→all, unify→generation rows.
        visual_ids_for_loss = visual_ids
        if visual_ids is not None and visual_mask.any():
            # Boolean selection in visual_mask space; both masks are over
            # input_ids so target_visual_mask[visual_mask] picks target rows.
            gen_sel = target_visual_mask[visual_mask]
            if gen_sel.numel() == visual_ids.shape[0]:
                visual_ids_for_loss = visual_ids[gen_sel]
            else:
                # Safety: mask/ids length mismatch (e.g. edge truncation).
                n = min(gen_sel.numel(), visual_ids.shape[0])
                visual_ids_for_loss = visual_ids[:n][gen_sel[:n]]

        # Step 5: text_logits + depth CE (all FSDP2 ops in this block run on
        # ALL ranks to keep all-gather/reduce-scatter aligned in unify mode).
        _text_logits = self._lm_head_module(hidden_states)

        # visual_offset_vals buffer may still be on meta device; resolve safely.
        _ov = self._visual_offset_vals_buffer
        if isinstance(_ov, torch.Tensor) and _ov.device.type == "meta":
            _ov = torch.tensor([150581, 166965, 183349, 199733, 216117, 232501, 248885, 265269])

        # Depth CE: dummy-path (1-token fake) keeps FSDP collectives aligned.
        _loss_vis = loss_visual_mask if loss_visual_mask is not None else torch.zeros_like(labels, dtype=torch.bool)
        _has_img_loss = _loss_vis.any().item()
        _img_end = img_end_mask if img_end_mask is not None else torch.zeros_like(labels, dtype=torch.bool)
        # Both branches (real / dummy) call the SAME FSDP2-sharded visual_head
        # submodules, so on no-image ranks we still run the dummy branch and graft
        # 0.0*sum() of its output onto the loss. Otherwise backward never fires the
        # reduce-scatter for those submodules on those ranks, the collective count
        # diverges from the image ranks, and backward deadlocks (all ranks reach
        # BWD_START, none reach BWD_DONE).
        _dummy_depth_loss = None

        _n_lvls = len(self._visual_head_module.codebook_sizes)
        # Depth-CE labels MUST come from visual_ids_for_loss (the generation rows
        # sliced in Step 4e), NOT the raw all-image visual_ids. In a mixed pack the
        # understanding images come first, so using raw visual_ids would feed
        # understanding-image VQ as the generation labels → wrong labels →
        # inflated image loss (2-3x per level). Pure-generate packs are unaffected.
        _vids_gen = visual_ids_for_loss
        if _has_img_loss and _vids_gen is not None and _vids_gen.shape[0] > 0:
            _img_h = hidden_states[_loss_vis]
            _end_in_img = _img_end[_loss_vis]
            _n_end = _end_in_img.sum().item()
            _n_pad = _img_h.shape[0] - _n_end
            if _n_end > 0:
                _full_vq = torch.zeros(_img_h.shape[0], _n_lvls, dtype=torch.long, device=hidden_states.device)
                _full_vq[~_end_in_img] = _vids_gen[:_n_pad].to(hidden_states.device)
                for _l in range(_n_lvls):
                    _full_vq[_end_in_img, _l] = _ov[_l].to(hidden_states.device).long() + self._visual_head_module.codebook_sizes[_l]
            else:
                _full_vq = _vids_gen.to(hidden_states.device)
            _depth_pt, _lvl_sums = _compute_depth_loss(
                visual_head=self._visual_head_module, hidden_states=_img_h, vq_labels=_full_vq,
                embed_tokens=self._embed_tokens_module, codebook_sizes=self._visual_head_module.codebook_sizes,
                offset_vals=_ov.to(hidden_states.device), img_end_mask_in_img=_end_in_img if _n_end > 0 else None,
            )
            _mask_for_scatter = _loss_vis.view(-1)
        else:
            _h = hidden_states.shape[-1]
            _f_h = torch.zeros(1, _h, device=hidden_states.device, dtype=hidden_states.dtype)
            _f_vq = torch.zeros(1, _n_lvls, dtype=torch.long, device=hidden_states.device)
            _pr = _f_vq[:, :_n_lvls - 1]
            _le = self._embed_tokens_module(_pr)
            _cs = torch.cumsum(_le, dim=1)
            _di = torch.cat([_f_h.unsqueeze(1), _cs], dim=1)  # [1, 8, H]
            vh = self._visual_head_module
            if hasattr(vh, 'hidden_norm') and vh.transformer_ffn_scale > 0:
                _di = vh.hidden_norm(_di); _di = vh.hidden_proj(_di)
            for _ly in vh.transformer_layers: _di = _ly(_di)
            _di = vh.headnorm(_di)
            # Accumulate every head's output (the real branch feeds heads[_l]
            # output into cross_entropy → loss, so heads must also see backward).
            _dummy_head_acc = 0.0 * _di.sum()
            for _l in range(_n_lvls):
                _hl = vh.heads[_l](_di[:, _l].contiguous())
                _dummy_head_acc = _dummy_head_acc + 0.0 * _hl.sum()
            _depth_pt = torch.zeros(0, device=hidden_states.device); _lvl_sums = {}
            _mask_for_scatter = _loss_vis.view(-1) if loss_visual_mask is not None else torch.zeros(_text_logits.shape[:-1], dtype=torch.bool, device=hidden_states.device).view(-1)
            # Connect dummy visual_head output to the graph so backward triggers
            # the SAME reduce-scatter set as the real branch (see note above).
            _dummy_depth_loss = _dummy_head_acc

        # Step 6: pure-math loss aggregation (no FSDP module access).
        result = compute_unified_loss(
            labels=labels, loss_mask=loss_mask, hidden_states=hidden_states,
            text_logits=_text_logits, visual_mask=visual_mask, visual_ids=visual_ids_for_loss,
            loss_visual_mask=loss_visual_mask, img_end_mask=img_end_mask,
            cu_seqlens=cu_seqlens, num_real_samples=num_real_samples, task=task,
            depth_per_token=_depth_pt, has_image_loss=_has_img_loss,
            level_loss_sums=_lvl_sums, depth_mask_for_scatter=_mask_for_scatter,
        )

        # Step 6: Inject auxiliary losses into backward via autograd.
        # Aux losses are NOT added to the loss scalar. Instead:
        # - moe load-balance loss: injected per MoE layer's output
        # - hidden z-loss: injected on the hidden states before the final norm
        # Both are identity in forward, gradient injection in backward.
        # This keeps the reported loss scalar = pure CE.
        #
        # hidden_z_loss was already injected above via AddHiddenZLossToBackward.
        # router_z_loss: inject via AddAuxLossToBackward on the loss tensor itself.
        # (This is equivalent to adding it to loss for backward purposes,
        # but we attach it to result["loss"] without changing its value.)
        if dummy_visual_loss is not None:
            result["loss"] = result["loss"] + dummy_visual_loss
        # Graft the depth-CE dummy-branch output (0-contribution) so no-image
        # ranks fire the same visual_head reduce-scatter as image ranks.
        if _dummy_depth_loss is not None:
            result["loss"] = result["loss"] + _dummy_depth_loss
        # router_z_loss: add to loss scalar for backward.
        if router_z_loss is not None:
            from losses.z_loss import AddAuxLossToBackward
            result["loss"] = AddAuxLossToBackward.apply(result["loss"], router_z_loss)
            result.setdefault("metrics", {})["router_z_loss"] = router_z_loss.item()
        if hidden_z_loss is not None:
            result.setdefault("metrics", {})["hidden_z_loss"] = hidden_z_loss.item()
        if load_balance_loss is not None:
            from losses.z_loss import AddAuxLossToBackward
            result["loss"] = AddAuxLossToBackward.apply(result["loss"], load_balance_loss)
            result.setdefault("metrics", {})["load_balance_loss"] = load_balance_loss.item() if load_balance_loss is not None else 0.0

        # Display metrics:
        # "expert load balance loss" and "z loss" are NOT the raw loss values,
        # they are already scaled by moe_loss_coeff (and z_coeff for z loss).
        result.setdefault("metrics", {})["expert load balance loss"] = lb_loss_display
        result.setdefault("metrics", {})["z loss"] = z_loss_display

        # Torch 2.3.x FSDP2 post_forward hook crashes on non-tensor values in output dict.
        # Stash non-tensor items separately so FSDP only sees tensors.
        _non_tensor_keys = [k for k, v in result.items() if not isinstance(v, torch.Tensor)]
        if _non_tensor_keys:
            self._last_forward_non_tensors = {k: result.pop(k) for k in _non_tensor_keys}
        return result


def count_jsonl_samples(data_paths, timeout=60):
    """Count total samples across JSONL data files (for epoch tracking).

    Uses `wc -l` for speed on large files / network filesystems.
    Falls back to 0 if counting fails or times out.
    """
    import subprocess
    total = 0
    for path in data_paths:
        try:
            result = subprocess.run(
                ["wc", "-l", path],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                # wc -l output: "  12345 /path/to/file"
                total += int(result.stdout.strip().split()[0])
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            _log.warning(f"Could not count lines in {path}: {e}")
    return total


def _maybe_merge_data(config: TrainConfig, data_paths, rank, world_size):
    """Pre-merge multiple JSONL files into one shuffled file if configured.

    When merged_data_dir is set and there are multiple data files, rank 0
    merges all files into a single shuffled JSONL. All ranks wait via barrier.
    Returns the list of data_paths to use (single merged file or original).
    """
    if not config.merged_data_dir or len(data_paths) <= 1:
        return data_paths

    from data.merge_shuffle import merge_and_shuffle
    import os

    merged_filename = f"{config.task}_seed{config.seed}_merged.jsonl"
    merged_path = os.path.join(config.merged_data_dir, merged_filename)

    # Fast path: when --skip_merge_check is set and the merged file already
    # exists, reuse it directly WITHOUT counting lines in the (full) input
    # files. This avoids re-reading all input data on every resubmit. We only
    # check the merged file's existence, not its line count.
    if getattr(config, "skip_merge_check", False):
        if os.path.exists(merged_path):
            if rank == 0:
                _log.info(f"[MERGE] skip_merge_check: reusing existing {merged_path} "
                      f"(NOT verifying line count against inputs).")
            dist.barrier()
            return [merged_path]
        elif rank == 0:
            _log.info(f"[MERGE] skip_merge_check set but {merged_path} does not exist; "
                  f"falling back to full merge.")

    if rank == 0:
        _log.info(f"[MERGE] Merging {len(data_paths)} files into {merged_path}...")
        merge_and_shuffle(data_paths, merged_path, seed=config.seed)
        _log.info(f"[MERGE] Done. Using merged file for training.")

    # All ranks wait for merge to complete
    dist.barrier()

    if rank == 0:
        _log.info(f"[MERGE] All ranks synced. Using {merged_path}")

    return [merged_path]


def build_dataloader(config: TrainConfig, tokenizer, image_preprocessor, rank, world_size):
    """Build data loader for the specified task."""
    data_paths = [p.strip() for p in config.data_path.split(",")]

    # Pre-merge multiple files into one shuffled file if configured.
    # This eliminates data distribution bias from round-robin + modulo sharding
    # when gcd(world_size, num_files) > 1.
    data_paths = _maybe_merge_data(config, data_paths, rank, world_size)

    if config.task == "understand":
        dataset = UnderstandPackedDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_processor=image_preprocessor,
            seq_length=config.seq_length,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
            num_epochs=config.num_epochs,
            no_packing=config.no_packing,
        )
    elif config.task == "generate":
        dataset = GeneratePackedDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_processor=image_preprocessor,
            seq_length=config.seq_length,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
            num_epochs=config.num_epochs,
            no_packing=config.no_packing,
        )
    elif config.task == "unify":
        # Mixed understand + generate training. Single pre-merged+shuffled JSONL
        # containing both sample types; per-image placeholder/loss logic is
        # reused from the aligned single-task datasets, packing is modality-
        # agnostic (task type is not needed to build the pack).
        dataset = UnifyPackedDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            image_processor=image_preprocessor,
            seq_length=config.seq_length,
            seed=config.seed,
            rank=rank,
            world_size=world_size,
            num_epochs=config.num_epochs,
            no_packing=config.no_packing,
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")

    # batch_size=1 because data is already packed into fixed-length sequences
    # StatefulDataLoader supports state_dict()/load_state_dict() for mid-epoch resume.
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader


def train_step(model, batch, config, device, gradient_accumulation_steps):
    """Execute one training step."""
    input_ids = batch["input_ids"].to(device)
    visual_mask = batch["visual_mask"].to(device)
    position_ids = batch["position_ids"].to(device)
    labels = batch["labels"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    cu_seqlens = batch["cu_seqlens"].squeeze(0).to(device)  # [num_samples+1]
    # Use the dataset's cu_seqlens directly. Packed samples are separated by EOD
    # and cu_seqlens drives varlen attention.

    # EOD is appended in the dataset with loss_mask=1, so the "predict EOD"
    # position is trained after the causal shift.

    # pixel_values and image_grid_thw are variable-length across samples.
    # DataLoader adds a batch dim (batch_size=1), so squeeze it:
    #   pixel_values: [1, num_patches, C, pH, pW] -> [num_patches, C, pH, pW]
    #   image_grid_thw: [1, N_images, 3] -> [N_images, 3]
    pixel_values = batch["pixel_values"].squeeze(0).to(device) if batch["pixel_values"].numel() > 0 else None
    image_grid_thw = batch["image_grid_thw"].squeeze(0).to(device) if batch["image_grid_thw"].numel() > 0 else None

    # loss_visual_mask: for generation task, selects hidden states for depth CE
    # based on labels (not input_ids).
    # Includes both IMG_PAD and img_end positions.
    loss_visual_mask = batch.get("loss_visual_mask")
    if loss_visual_mask is not None:
        loss_visual_mask = loss_visual_mask.to(device)

    # img_end_mask: marks img_end positions in labels, used to distinguish
    # IMG_PAD vs img_end inside depth CE loss.
    img_end_mask = batch.get("img_end_mask")
    if img_end_mask is not None:
        img_end_mask = img_end_mask.to(device)

    # target_visual_mask: generation (trainable) image-pad positions in input_ids.
    # Emitted by the unify dataset; used in forward to slice visual_ids for the
    # depth loss. Absent for single-task datasets (derived in forward by task).
    target_visual_mask = batch.get("target_visual_mask")
    if target_visual_mask is not None:
        target_visual_mask = target_visual_mask.to(device)

    # num_real_samples: actual doc count (excludes padding segments in cu_seqlens)
    num_real_samples = batch.get("num_real_samples")

    forward_kwargs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "visual_mask": visual_mask,
        "position_ids": position_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "cu_seqlens": cu_seqlens,
        "num_real_samples": num_real_samples,
        "loss_visual_mask": loss_visual_mask,
        "img_end_mask": img_end_mask,
        "target_visual_mask": target_visual_mask,
        "task": config.task,
    }

    result = model(**forward_kwargs)

    loss = result["loss"] / gradient_accumulation_steps
    loss.backward()

    # Restore non-tensor items that were stashed by forward() for FSDP2 compat
    _stashed = getattr(model, '_last_forward_non_tensors', None)
    if _stashed:
        result.update(_stashed)
        model._last_forward_non_tensors = None

    return result


def main():
    # Parse config
    config = TrainConfig.from_args()

    # Setup distributed
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # Configure logging: INFO to stdout, with rank prefix. Non-root ranks only
    # emit warnings/errors to avoid duplicating progress logs across the cluster.
    # Timestamps include the date and milliseconds (training can span days).
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[%(asctime)s.%(msecs)03d][rank{rank}][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    set_seed(config.seed + rank)

    tb_logger = TensorBoardLogger(
        log_dir=config.tensorboard_dir, rank=rank,
        enabled=bool(config.tensorboard_dir),
        global_batch_size=config.global_batch_size,
    )
    logger = TrainingLogger(
        rank=rank, log_interval=config.log_interval, tb_logger=tb_logger,
        global_batch_size=config.global_batch_size,
    )
    logger.log_message(f"Starting {config.task} SFT training (FSDP2)")
    logger.log_message(f"World size: {world_size}, Rank: {rank}")
    logger.log_message(f"Config: {config}")

    # =====================================================================
    # Model Loading (meta-device pattern for fast startup)
    # =====================================================================
    # All ranks create model on meta device (zero memory, no slow mmap reads).
    # Weights are loaded later via per-parameter broadcast from safetensors.
    logger.log_message(f"[Rank {rank}] Loading model on meta device...")
    t0 = time.time()
    base_model, tokenizer, processor = load_model_meta(
        config.model_path,
        rank=rank,
        rope_theta=config.rope_theta,
    )
    logger.log_message(f"[Rank {rank}] Model created in {time.time() - t0:.1f}s")

    # Step 2: Apply freeze strategy (sets requires_grad flags only)
    logger.log_message(f"[Rank {rank}] Applying freeze strategy for {config.task}...")
    if config.task == "understand":
        freeze_for_understand(base_model)
    elif config.task == "generate":
        freeze_for_generate(base_model)
    elif config.task == "unify":
        # Mixed training needs both modalities' trainable params. The generate
        # freeze set is a superset of understand (it additionally unfreezes
        # visual_head for depth-CE), so it covers both sample types.
        freeze_for_generate(base_model)

    # Step 3: Wrap in training wrapper
    training_model = SFTTrainingWrapper(base_model, config=config)

    # MoE determinism patch (AC-compatible, deterministic unpermute).
    from model.fsdp_utils import _patch_moe_for_checkpoint
    # _deterministic_moe (token-permute + index_copy/sum unpermute) replaces the
    # original HF moe():
    # - removes HF's data-dependent branch (if token_indices.numel()>0) so the tensor
    #   count is stable under AC recompute;
    # - replaces the non-deterministic index_add_ with a deterministic index_copy_+sum
    #   for reproducible training.
    # This patch is required when AC is enabled (HF's data-dependent branch conflicts
    # with recomputation).
    _use_patch_moe = config.activation_checkpointing
    if _use_patch_moe:
        _patch_moe_for_checkpoint(model=base_model, use_grouped_gemm=config.use_grouped_gemm)
        if rank == 0:
            _log.info("[FSDP] Applied deterministic MoE patch (AC-compatible)")

    # =====================================================================
    # FSDP2 Setup (break sharing → shard → materialize → load weights)
    # =====================================================================
    # Shard parameters across ranks and load weights from safetensors.
    #
    # HSDP support via FSDP_SHARD_SIZE env:
    #   unset or =world_size  → pure FSDP, 1D mesh ("dp",)         [default]
    #   <world_size           → HSDP, 2D mesh ("replicate", "shard")
    #                            params sharded inside each group of FSDP_SHARD_SIZE
    #                            ranks, replicated across world_size/FSDP_SHARD_SIZE groups.
    #
    # Per-rank shard cost vs default: shard ×= (world_size / FSDP_SHARD_SIZE).
    # Pick this only when (a) communication is the bottleneck and (b) the extra
    # per-rank memory fits.
    _shard_size_env = os.environ.get("FSDP_SHARD_SIZE", "").strip()
    if _shard_size_env and int(_shard_size_env) != world_size:
        from torch.distributed.device_mesh import init_device_mesh
        _shard_size = int(_shard_size_env)
        assert world_size % _shard_size == 0, (
            f"FSDP_SHARD_SIZE={_shard_size} must divide world_size={world_size}"
        )
        _replicate = world_size // _shard_size
        dp_mesh = init_device_mesh(
            "cuda",
            (_replicate, _shard_size),
            mesh_dim_names=("replicate", "shard"),
        )
        # Sub-mesh used for collectives that should run within a shard group only
        # (grad clip all-reduce, loss-display all-reduce). The 2D mesh as a whole
        # is what FSDP2 needs to enable HSDP.
        _fsdp_shard_pg = dp_mesh["shard"].get_group()
        if rank == 0:
            logger.log_message(
                f"[HSDP] replicate={_replicate} shard={_shard_size}; "
                f"per-rank param shard = P/{_shard_size} "
                f"(×{world_size // _shard_size} vs pure FSDP)"
            )
    else:
        dp_mesh = DeviceMesh("cuda", torch.arange(world_size), mesh_dim_names=("dp",))
        _fsdp_shard_pg = None  # None → use the global (WORLD) group, same as before
    logger.log_message(f"[Rank {rank}] Device mesh: {dp_mesh}")

    # Step 3b: Unify parameter dtypes for FSDP2 compatibility.
    # Some VQ codebook params are stored as float32 in the checkpoint. FSDP2 needs
    # a uniform dtype within each sharded module — except where a sub-module is
    # sharded separately with its own mp_policy (see setup_fsdp2, where
    # quantizer.quantize keeps fp32). That fp32 precision is required: the VQ
    # distance computation runs in fp32, and a bf16 codebook loses enough
    # precision that argmin can pick a different centroid (~35% of tokens drift).
    n_converted = 0
    n_kept_vq_fp32 = 0
    n_kept_router_fp32 = 0
    n_upcast_router_fp32 = 0
    for name, param in training_model.named_parameters():
        # Keep VQ codebook params as fp32
        if 'quantizer.quantize.codebooks' in name:
            n_kept_vq_fp32 += 1
            continue
        # Keep router classifier params in fp32.
        # Some model variants are instantiated directly in bf16 by
        # from_config(..., torch_dtype=bf16), so merely skipping the bf16 cast is
        # not enough. Explicitly upcast router classifier params to fp32 before
        # FSDP2 so nested fp32 router sharding loads exact fp32 checkpoint values.
        if '.mlp.router.classifier.' in name or 'mlp.router.classifier' in name:
            if param.dtype != torch.float32:
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=torch.float32, device=param.device),
                    requires_grad=param.requires_grad,
                )
                torch.utils.swap_tensors(param, new_param)
                n_upcast_router_fp32 += 1
            else:
                n_kept_router_fp32 += 1
            continue
        if param.dtype != torch.bfloat16:
            new_param = nn.Parameter(
                torch.empty(param.shape, dtype=torch.bfloat16, device=param.device),
                requires_grad=param.requires_grad,
            )
            torch.utils.swap_tensors(param, new_param)
            n_converted += 1
    if rank == 0:
        logger.log_message(f"[Rank {rank}] Converted {n_converted} non-bf16 params to bf16, "
                           f"kept {n_kept_vq_fp32} VQ codebook fp32, "
                           f"kept {n_kept_router_fp32} router fp32, "
                           f"upcast {n_upcast_router_fp32} router to fp32")

    # Step 4: Break parameter sharing BEFORE FSDP2 sharding.
    logger.log_message(f"[Rank {rank}] Breaking parameter sharing...")
    break_parameter_sharing(training_model, rank=rank)

    # Step 5: Apply FSDP2 (restructure ngram embedders + AC + shard layers + shard root)
    logger.log_message(f"[Rank {rank}] Applying FSDP2...")
    t0 = time.time()
    setup_fsdp2(
        training_model,
        dp_mesh=dp_mesh,
        activation_checkpointing=config.activation_checkpointing,
        rank=rank,
        task=config.task,
        offload_to_cpu=config.offload_to_cpu,
    )
    logger.log_message(f"[Rank {rank}] FSDP2 applied in {time.time() - t0:.1f}s")

    # Step 6: Materialize and load weights via streaming per-parameter broadcast
    # When CPU offload is enabled, FSDP2 requires parameters on CPU at lazy_init time.
    materialize_device = torch.device("cpu") if config.offload_to_cpu else device
    logger.log_message(f"[Rank {rank}] Loading weights (materialize_device={materialize_device})...")
    t0 = time.time()
    training_model.to_empty(device=materialize_device)
    load_fsdp2_weights_manual(training_model, config.model_path, rank=rank,
                               target_device=materialize_device)

    logger.log_message(f"[Rank {rank}] Weights loaded in {time.time() - t0:.1f}s")

    # Fix router e_score_correction_bias: the buffer may have been loaded as bf16
    # (from_config(torch_dtype=bf16) creates buffer in bf16, then load casts safetensors
    # fp32 value to tensor.dtype=bf16, losing precision). Reload from safetensors as fp32.
    _n_bias_fixed = 0
    from safetensors import safe_open as _safe_open_bias_fix
    _sf_index_path = os.path.join(config.model_path, "model.safetensors.index.json")
    if os.path.exists(_sf_index_path):
        import json as _json_bf
        with open(_sf_index_path) as _f_bf:
            _weight_map_bf = _json_bf.load(_f_bf).get("weight_map", {})
        for _name, _buf in training_model.named_buffers():
            if "e_score_correction_bias" not in _name:
                continue
            # SFTTrainingWrapper wraps as model.model.layers.X... → HF key is model.layers.X...
            _hf_key = _name
            if _hf_key.startswith("model.model."):
                _hf_key = _hf_key[len("model."):]  # strip first "model." → "model.layers.X..."
            elif _hf_key.startswith("model."):
                pass  # already correct
            _hf_key = _hf_key.replace("_checkpoint_wrapped_module.", "")
            _sf_file = _weight_map_bf.get(_hf_key)
            if _sf_file is None:
                if rank == 0:
                    _log.warning(f"router e_score_correction_bias key not found in weight_map: {_hf_key} (from {_name})")
                continue
            _sf_full_path = os.path.join(config.model_path, _sf_file)
            if not os.path.exists(_sf_full_path):
                continue
            with _safe_open_bias_fix(_sf_full_path, framework="pt", device=str(device)) as _sf_h:
                if _hf_key in _sf_h.keys():
                    _fp32_val = _sf_h.get_tensor(_hf_key).to(dtype=torch.float32, device=device)
                    with torch.no_grad():
                        _buf.data = _fp32_val
                    _n_bias_fixed += 1
    if rank == 0:
        logger.log_message(f"[Rank {rank}] Fixed {_n_bias_fixed} router e_score_correction_bias to fp32 from safetensors")

    # Step 6: Fix non-persistent buffers
    # Always compute on CUDA device — FSDP2 CPU offload only manages parameters,
    # buffers need to be on GPU for forward pass.
    logger.log_message(f"[Rank {rank}] Fixing non-persistent buffers...")
    fix_non_persistent_buffers(training_model, config.model_path, device, rank=rank,
                               rope_theta=config.rope_theta)

    # When CPU offload is enabled, ensure ALL buffers are on GPU.
    # FSDP2 CPUOffloadPolicy only offloads parameters (CPU↔GPU), not buffers.
    # Buffers stay wherever they are, and forward pass needs them on GPU.
    if config.offload_to_cpu:
        buf_count = 0
        for name, buf in training_model.named_buffers():
            if buf.device != device:
                buf.data = buf.data.to(device)
                buf_count += 1
        if rank == 0:
            logger.log_message(f"[Rank {rank}] Moved {buf_count} buffers to {device} for CPU offload")

    gpu_mem = torch.cuda.memory_allocated(device) / 1e9
    gpu_reserved = torch.cuda.memory_reserved(device) / 1e9
    logger.log_message(f"[Rank {rank}] GPU memory: allocated={gpu_mem:.2f}GB, "
                       f"reserved={gpu_reserved:.2f}GB")

    # =====================================================================
    # Data, Optimizer, Training Loop
    # =====================================================================
    image_preprocessor = ImagePreprocessor(processor)
    dataloader = build_dataloader(config, tokenizer, image_preprocessor, rank, world_size)

    # Count total samples for epoch tracking (used in stdout "epoch X.XX").
    # Only rank 0 counts to avoid redundant I/O, then broadcast to all ranks.
    data_paths = [p.strip() for p in config.data_path.split(",")]
    if rank == 0:
        total_samples_per_epoch = count_jsonl_samples(data_paths)
    else:
        total_samples_per_epoch = 0
    count_tensor = torch.tensor([total_samples_per_epoch], dtype=torch.long, device=device)
    dist.broadcast(count_tensor, src=0)
    total_samples_per_epoch = count_tensor.item()
    logger.total_samples_per_epoch = total_samples_per_epoch
    logger.log_message(f"Total samples per epoch: {total_samples_per_epoch}")

    # Gradient accumulation
    gradient_accumulation_steps = config.global_batch_size // (1 * world_size)
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    logger.log_message(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    # Optimizer and scheduler
    optimizer = create_optimizer(training_model, config)

    # Loss-free expert balance (DeepSeek LFB). Built after router hooks ran so
    # training_model._routers is populated. rate=0 keeps it disabled and the
    # manager stays None — train loop skips both collect and update.
    lfb_manager = None
    if getattr(config, "loss_free_balance_rate", 0.0) > 0:
        from losses.loss_free_balance import LossFreeBalanceManager
        _hf_cfg = base_model.config
        _zero_expert_num = int(getattr(_hf_cfg, "zero_expert_num", 0) or 0)
        if not training_model._routers:
            logger.log_message("[LFB] WARNING: no routers captured — LFB disabled.")
        else:
            lfb_manager = LossFreeBalanceManager(
                router_modules=training_model._routers,
                rate=config.loss_free_balance_rate,
                decay_rule=config.loss_free_decay_rule,
                dynamic_update=config.dynamic_update_loss_free_bias,
                only_adapt_ffn_bias=config.only_adapt_ffn_bias,
                zero_expert_num=_zero_expert_num,
                target_topk=8,
                moe_topk=None,        # auto-captured from router outputs in add_batch
            )
            logger.log_message(
                f"[LFB] enabled: rate={config.loss_free_balance_rate} "
                f"decay_rule={config.loss_free_decay_rule!r} "
                f"dynamic={config.dynamic_update_loss_free_bias} "
                f"only_adapt_ffn_bias={config.only_adapt_ffn_bias} "
                f"zero_expert_num={_zero_expert_num} "
                f"n_routers={len(training_model._routers)}"
            )

    # Estimate total optimizer steps from data count.
    # NOTE: This estimate is inaccurate when packing is enabled because
    # total_samples_per_epoch counts raw JSONL lines while each GBS step
    # processes multiple packed samples. The estimate is only used for the
    # learning rate scheduler (which just needs a rough upper bound) and is
    # NOT displayed in training logs to avoid confusion.
    if total_samples_per_epoch > 0:
        estimated_total_steps = (total_samples_per_epoch * config.num_epochs) // max(config.global_batch_size, 1)
    else:
        estimated_total_steps = 10000
    logger.log_message(f"Estimated total steps (for scheduler only, inaccurate with packing): {estimated_total_steps}")
    scheduler = create_scheduler(optimizer, config, estimated_total_steps)

    # Resume from checkpoint if specified
    resume_step = 0
    resume_dataloader_state = None
    if config.resume_from:
        resume_step, resume_dataloader_state = load_checkpoint(
            training_model, optimizer, scheduler, config.resume_from, device
        )
        logger.log_message(f"Resumed from step {resume_step}")
        if resume_dataloader_state is not None:
            logger.log_message("Restoring dataloader state via StatefulDataLoader...")

    # Training loop
    logger.log_message("Starting training loop...")
    global_step = resume_step
    micro_step = 0
    accumulated_loss = 0.0
    # Accumulate metrics across gradient accumulation micro-steps, then all-reduce
    # token counts across DP ranks. We do this by:
    # 1. Summing metric values across micro-steps (for additive metrics)
    # 2. All-reducing token counts across DP ranks after accumulation
    accumulated_metrics = {}
    accumulated_tokens = 0
    accumulated_samples = 0

    optimizer.zero_grad()
    last_data_iter_idx = 0  # tracks dataloader progress for epoch calculation

    # Track micro-steps consumed for resume: when resuming, we need to skip
    # the micro-batches that were already processed before the checkpoint.
    # When StatefulDataLoader state is available, the dataloader resumes
    # from the saved position — no fast-forward needed for checkpoint resume.
    # skip_steps (manual fast-forward) is still honored.
    if resume_dataloader_state is not None:
        # StatefulDataLoader handles resume — only honor explicit skip_steps
        dataloader.load_state_dict(resume_dataloader_state)
        skip_micro_steps = config.skip_steps * gradient_accumulation_steps
        if rank == 0:
            _log.info(f"[STATEFUL_RESUME] Dataloader state restored, "
                  f"no fast-forward needed for checkpoint resume")
    else:
        # Fallback: fast-forward by consuming batches (legacy behavior)
        skip_micro_steps = resume_step * gradient_accumulation_steps
        skip_micro_steps += config.skip_steps * gradient_accumulation_steps
    skipped = 0
    if skip_micro_steps > 0 and rank == 0:
        _log.info(f"[SKIP] Will skip {skip_micro_steps} micro-steps "
              f"({skip_micro_steps // gradient_accumulation_steps} optimizer steps)")

    first_batch_synced = False
    # Use manual iterator + prefetch to detect data exhaustion BEFORE entering
    # NCCL collectives. Without this, some ranks may exhaust data while others
    # are still in train_step → NCCL collective mismatch → timeout.
    data_iter = iter(dataloader)

    def _prefetch_batch():
        """Fetch next batch from iterator. Returns (batch, exhausted)."""
        try:
            return next(data_iter), False
        except StopIteration:
            return None, True

    # Prefetch first batch
    batch, data_exhausted = _prefetch_batch()
    if rank == 0:
        _log.warning(f"[TRAIN_LOOP] First prefetch: batch={'OK' if batch is not None else 'None'}, "
              f"data_exhausted={data_exhausted}, max_steps={config.max_steps}")

    while batch is not None:
        # Skip batches already processed before checkpoint / skip_steps
        if skipped < skip_micro_steps:
            skipped += 1
            if skipped % 100 == 0 and rank == 0:
                _log.info(f"  Skipping micro-step {skipped}/{skip_micro_steps}...")
            batch, data_exhausted = _prefetch_batch()
            continue

        # Barrier after each rank's first dataloader iteration. The first iteration
        # can take minutes (loading large JSONLs + preprocessing images); without
        # this, fast ranks enter the FSDP2 forward all-gather while slow ranks are
        # still loading, triggering the NCCL watchdog timeout.
        if not first_batch_synced:
            first_batch_synced = True
            dist.barrier()
            logger.log_message("All ranks completed first batch, starting forward...")

        result = train_step(training_model, batch, config, device, gradient_accumulation_steps)
        micro_step += 1
        accumulated_loss += result["loss"].item()

        # LFB: snapshot router topk_indices from this micro-batch (before the
        # next forward clears _router_outputs).
        if lfb_manager is not None and training_model._router_outputs:
            lfb_manager.add_batch(training_model._router_outputs)


        # Accumulate metrics across micro-steps
        step_metrics = result.get("metrics", {})
        accumulated_tokens += int(batch["loss_mask"].sum().item())
        if "num_real_samples" in batch and batch["num_real_samples"] is not None:
            accumulated_samples += int(batch["num_real_samples"])
        else:
            accumulated_samples += int(batch["cu_seqlens"].shape[-1]) - 1
        # Track data iterator progress for accurate epoch calculation
        _di = batch.get("data_iter_idx", 0)
        last_data_iter_idx = int(_di.item()) if isinstance(_di, torch.Tensor) else int(_di)
        for k, v in step_metrics.items():
            if isinstance(v, (int, float)):
                accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v

        if micro_step % gradient_accumulation_steps == 0:

            # Release fragmented memory before grad clipping to avoid OOM in
            # clip_grad_norm_'s _foreach_mul_ (DTensor dispatch needs contiguous space).
            torch.cuda.empty_cache()

            if config.offload_to_cpu:
                # Manual grad clipping for CPU offload: DTensor clip_grad_norm_
                # requires a distributed backend for the parameter device (CPU),
                # but NCCL only supports CUDA. Compute norm manually.
                #
                # HSDP: same shard-group-only reduce as the manual GPU path below.
                total_norm_sq = torch.tensor(0.0, dtype=torch.float32)
                for p in training_model.parameters():
                    if p.grad is not None:
                        g = p.grad
                        if hasattr(g, '_local_tensor'):
                            g = g._local_tensor
                        total_norm_sq += (g.float() ** 2).sum()
                # All-reduce via NCCL (needs CUDA tensor)
                total_norm_sq_cuda = total_norm_sq.to(device)
                dist.all_reduce(total_norm_sq_cuda, op=dist.ReduceOp.SUM, group=_fsdp_shard_pg)
                grad_norm = total_norm_sq_cuda.sqrt()
                # Clip
                clip_coeff = config.max_grad_norm / (grad_norm + 1e-6)
                if clip_coeff < 1.0:
                    for p in training_model.parameters():
                        if p.grad is not None:
                            g = p.grad
                            if hasattr(g, '_local_tensor'):
                                g = g._local_tensor
                            g.mul_(clip_coeff.to(g.device))
            elif os.environ.get("TORCH_GRAD_CLIP", "0") != "1":
                # DEFAULT: manual clip. Set TORCH_GRAD_CLIP=1 to use
                # torch.nn.utils.clip_grad_norm_ instead.
                # Manual clip avoids clip_grad_norm_'s foreach DTensor path, which
                # can OOM on large shards near the memory limit. Doing per-param
                # in-place ops on the local shard bypasses the DTensor dispatcher;
                # the result is identical (global L2 norm via all-reduce of local
                # sq-sums, then in-place scale).
                #
                # HSDP note: FSDP2 has already DP-all-reduced grads across the
                # `replicate` dim during backward, so the local grad value is the
                # same on every replicate of the same shard. The norm SUM must
                # therefore only happen inside the shard group; reducing across
                # the global group would multiply norm² by the replicate factor.
                _total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
                for n, p in training_model.named_parameters():
                    if p.grad is None or "ngram_embedding" in n:
                        continue
                    g = p.grad
                    if hasattr(g, '_local_tensor'):
                        g = g._local_tensor
                    _total_norm_sq += (g.float() ** 2).sum()
                dist.all_reduce(_total_norm_sq, op=dist.ReduceOp.SUM, group=_fsdp_shard_pg)
                grad_norm = _total_norm_sq.sqrt()
                clip_coeff = config.max_grad_norm / (grad_norm + 1e-6)
                if clip_coeff < 1.0:
                    _cc = clip_coeff.to(device)
                    for n, p in training_model.named_parameters():
                        if p.grad is None or "ngram_embedding" in n:
                            continue
                        g = p.grad
                        if hasattr(g, '_local_tensor'):
                            g = g._local_tensor
                        g.mul_(_cc)
            else:
                # Default: torch.nn.utils.clip_grad_norm_ (foreach DTensor path).
                # Exclude ngram_embeddings from grad norm (its bf16 grad sq_sum
                # overflows fp32).
                _clip_params = [p for n, p in training_model.named_parameters()
                                if p.grad is not None and "ngram_embedding" not in n]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    _clip_params, config.max_grad_norm
                )
                # clip_grad_norm_ returns a DTensor with _NormPartial placement
                # (local shard's norm). Convert to global norm via full_tensor().
                if hasattr(grad_norm, 'full_tensor'):
                    grad_norm = grad_norm.full_tensor()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            # LFB: per-step bias update. Runs once per optimizer step, after all
            # micro-batches in this accumulation window have contributed counts.
            if lfb_manager is not None:
                lfb_manager.finalize_step(global_step)

            current_lr = scheduler.get_last_lr()[0]

            # All-reduce display metrics across DP ranks: SUM the token counts and
            # loss sums, then recompute averages and ratios from the global sums.
            global_metrics = _allreduce_metrics(
                accumulated_metrics, accumulated_tokens, accumulated_samples,
                device, gradient_accumulation_steps,
            )

            # Compute parameter norm (params-norm metric)
            p_norm = 0.0
            if rank == 0:
                p_norm = compute_params_norm(training_model)

            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            # Display average loss across gradient accumulation steps.
            # All-reduce to average losses across the data-parallel group
            avg_loss = accumulated_loss / gradient_accumulation_steps
            if dist.is_initialized():
                avg_loss_tensor = torch.tensor([avg_loss], dtype=torch.float64, device=device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (avg_loss_tensor / dist.get_world_size()).item()
            logger.log_step(
                step=global_step,
                loss=avg_loss,
                lr=current_lr,
                grad_norm=gn,
                metrics=global_metrics,
                tokens_in_batch=accumulated_tokens,
                samples_in_batch=accumulated_samples,
                params_norm=p_norm,
                world_size=world_size,
                task=config.task,
                total_steps=0,  # Don't display — inaccurate with packing
                data_iter_idx=last_data_iter_idx,
            )
            accumulated_loss = 0.0
            accumulated_metrics = {}
            accumulated_tokens = 0
            accumulated_samples = 0

            if config.save_interval > 0 and global_step % config.save_interval == 0:
                # Release GPU memory before checkpoint save (full_state_dict all-gather needs headroom)
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                dl_state = dataloader.state_dict()
                save_checkpoint(training_model, optimizer, scheduler, global_step,
                                config.save_dir, config.model_path,
                                dataloader_state=dl_state)

            # Check max_steps limit
            if config.max_steps > 0 and global_step >= config.max_steps:
                if rank == 0:
                    _log.info(f"Reached max_steps={config.max_steps}, stopping training.")
                break

            # ── Coordinated data exhaustion check ──
            # At each global step boundary (a natural sync point after all-reduce),
            # prefetch the next batch and check if ANY rank has exhausted its data.
            # If so, ALL ranks exit together to prevent NCCL collective mismatch.
            # This is critical for 128+ GPU training where packing differences
            # cause ranks to produce slightly different numbers of batches.
            batch, data_exhausted = _prefetch_batch()
            # Exchange exhaustion status: 1 = exhausted, 0 = still have data
            exhausted_flag = torch.tensor(
                [1 if data_exhausted else 0], dtype=torch.int32, device=device
            )
            dist.all_reduce(exhausted_flag, op=dist.ReduceOp.MAX)
            if exhausted_flag.item() > 0:
                if rank == 0:
                    local_status = "exhausted" if data_exhausted else "still has data"
                    _log.warning(f"[DATA_EXHAUSTED] Coordinated exit at step {global_step}. "
                          f"Rank 0 status: {local_status}. "
                          f"Some rank(s) exhausted data.")
                break

        else:
            # Not at global step boundary (gradient accumulation in progress).
            # Prefetch next micro-batch without coordinated check — all ranks
            # must stay in lockstep for the remaining micro-steps of this
            # gradient accumulation group.
            batch, data_exhausted = _prefetch_batch()
            if data_exhausted:
                # Rank exhausted mid-accumulation. This is rare but possible.
                # Log warning and break — the partial accumulated gradients
                # will be discarded (not applied).
                _log.warning(f"[DATA_EXHAUSTED] Rank {rank} exhausted mid-accumulation "
                      f"at micro_step={micro_step}, global_step={global_step}. "
                      f"Partial accumulation discarded.")
                break

    # Log exit reason
    if rank == 0:
        _log.warning(f"[TRAIN_LOOP_EXIT] Exited at global_step={global_step}, "
              f"micro_step={micro_step}, batch_is_none={batch is None}, "
              f"data_exhausted={data_exhausted}")

    # ── Barrier before final save to ensure all ranks exit together ──
    dist.barrier()

    # Final save — always save at end of training.
    # Skip if the last global step was already saved by save_interval to avoid
    # writing a duplicate checkpoint.
    last_saved_at_interval = (
        config.save_interval > 0 and global_step > 0
        and global_step % config.save_interval == 0
    )
    if not last_saved_at_interval and global_step > 0 and config.save_interval > 0:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if rank == 0:
            _log.info(f"[FINAL_SAVE] Saving final checkpoint at step {global_step}...")
        dl_state = dataloader.state_dict()
        save_checkpoint(training_model, optimizer, scheduler, global_step,
                        config.save_dir, config.model_path,
                        dataloader_state=dl_state)
    elif rank == 0 and last_saved_at_interval:
        _log.info(f"[FINAL_SAVE] Step {global_step} already saved by save_interval, skipping duplicate.")
    elif rank == 0:
        _log.info(f"[FINAL_SAVE] save_interval=0, skipping final checkpoint.")

    logger.log_message(f"Training completed! Total steps: {global_step}")
    logger.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
