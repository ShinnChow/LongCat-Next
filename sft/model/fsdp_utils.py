# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""FSDP2 utilities for LongCat-Next distributed training.

Sharding strategy:
1. Break parameter sharing (LongCat-Next specific)
2. Shard transformer decoder layers individually (14 layers, ~1.35B each)
3. Shard large non-layer sub-modules individually:
   - 12 ngram embedder units (embedder + projection, ~2.62B each = 4.88GB bf16)
   - visual_tokenizer sub-modules (ViT, VQ bridge, embedding layer)
   - audio_tokenizer sub-modules
   - lm_head, visual_head, audio_head
4. Shard root model (collects only small remaining params)

The key insight: without step 3, the root FSDP group would hold ~54B params.
During forward pass, FSDP2 all-gathers the root group's params onto each GPU,
requiring ~109GB — more than a single H800 (140GB) can hold alongside the
model shards (~18.5GB) and optimizer states (~70GB).

IMPORTANT: FSDP2 backward reshard issue with integer-input modules:
FSDP2's `post_backward` is triggered via `RegisterPostBackwardFunction` which
is applied to the module's forward INPUT tensors (in `pre_forward`). If no
input tensor requires grad (e.g., embedding modules that receive integer indices),
`RegisterPostBackwardFunction` is NOT registered, and `post_backward` (which
reshards parameters) is NOT called during backward. Parameters stay unsharded
until `root_post_backward_final_callback` at the very end of backward.

For 12 ngram embedders (~4.88GB each), this means ALL 12 stay simultaneously
unsharded during backward (~58.5GB extra memory), causing OOM on 140GB GPUs.

Solution: Wrap each embedder + projection into `_NgramEmbedderUnit` that receives
a grad-requiring tensor (the accumulator `x`) as input. This ensures
`RegisterPostBackwardFunction` is registered and `post_backward` fires promptly,
resharding each embedder after its gradient is computed.

LongCat-Next has two parameter sharing patterns that conflict with FSDP2:
1. embed_tokens <-> ngram_embeddings.word_embeddings (same nn.Embedding object)
2. visual_bridge_model codebooks[0..7] (8 entries all pointing to one VQEmbedding)

FSDP2's fully_shard walks the module tree and will encounter shared parameters
multiple times, causing "Invalid mesh_dim_names ('dp','dp')" errors.
We break these sharing patterns before applying FSDP2.
"""

import os
import time
import functools
import logging
from typing import List, Optional, Set, Type

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    from torch.distributed import DeviceMesh
except ImportError:
    from torch.distributed.device_mesh import DeviceMesh
try:
    from torch.distributed._composable.fsdp import fully_shard as _fully_shard_orig, MixedPrecisionPolicy, CPUOffloadPolicy
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard as _fully_shard_orig, MixedPrecisionPolicy
    CPUOffloadPolicy = None

# Compat wrapper: torch 2.3.x fully_shard doesn't accept offload_policy kwarg
import inspect
_fully_shard_params = inspect.signature(_fully_shard_orig).parameters
if 'offload_policy' in _fully_shard_params:
    fully_shard = _fully_shard_orig
else:
    def fully_shard(module, **kwargs):
        kwargs.pop('offload_policy', None)
        return _fully_shard_orig(module, **kwargs)


class _NgramEmbedderUnit(nn.Module):
    """Wraps a single ngram embedder + projection as one FSDP-shardable unit.

    FSDP2's `post_backward` requires at least one input tensor with
    `requires_grad=True` to register `RegisterPostBackwardFunction`.
    Embedding modules receive integer indices (no grad), so their
    `post_backward` never fires during backward, causing parameters to stay
    unsharded and accumulate on GPU (12 × 4.88GB = 58.5GB → OOM).

    This wrapper takes the accumulator tensor `x_accum` (which requires grad)
    as an additional input, ensuring FSDP2 properly registers the backward
    hook. The wrapper's forward computes the embedder + projection and adds
    the result to `x_accum`, returning the updated accumulator.

    The zero-multiply trick `x_accum * 0` is NOT needed — we simply add the
    projection output to the accumulator, which naturally creates a gradient
    path from the output back through this module's input.
    """

    def __init__(self, embedder: nn.Module, post_proj: nn.Module):
        super().__init__()
        self.embedder = embedder
        self.post_proj = post_proj

    def forward(self, new_ids: torch.Tensor, text_mask: torch.Tensor,
                x_accum: torch.Tensor) -> torch.Tensor:
        """Compute ngram embedding + projection and add to accumulator.

        Args:
            new_ids: [B, seq_len] hashed ngram indices (integer tensor).
            text_mask: [B, seq_len] boolean mask for valid positions.
            x_accum: [B, seq_len, hidden_size] accumulated embedding tensor.
                     Must require grad (ensures FSDP2 post_backward fires).

        Returns:
            Updated accumulator: x_accum + projection(embedding(new_ids)).
        """
        x_ngram = self.embedder(new_ids, text_mask)
        x_proj = self.post_proj(x_ngram)
        return x_accum + x_proj


def break_parameter_sharing(model: nn.Module, rank: int = 0):
    """Break parameter sharing before FSDP2 sharding (public API).

    MUST be called BEFORE extract_state_dict() so that the state dict keys
    match the post-sharing-break model structure. The sequence is:
        break_parameter_sharing(model)  # modifies model structure
        restructure_ngram_for_fsdp(model)  # restructure ngram embedders
        state_dict = extract_state_dict(model)  # keys match new structure
        setup_fsdp2(model)  # applies FSDP2 sharding
        load_fsdp2_weights_manual(model, model_path)  # keys match, loading succeeds

    Args:
        model: PyTorch model (SFTTrainingWrapper).
        rank: Current process rank (for logging).
    """
    causal_lm = _get_causal_lm(model)
    inner_model = causal_lm.model if hasattr(causal_lm, "model") else None
    _break_parameter_sharing(model, inner_model, rank)


def restructure_ngram_for_fsdp(model: nn.Module, rank: int = 0):
    """Restructure NgramEmbedding embedders for FSDP2 compatibility (public API).

    Wraps each embedder+projection pair into _NgramEmbedderUnit modules and
    monkey-patches the NgramEmbedding.forward to use the new units.

    MUST be called AFTER break_parameter_sharing() and BEFORE setup_fsdp2()
    so that:
    1. The state dict keys reflect the new structure (embedder_units.N.embedder.weight)
    2. setup_fsdp2 can detect already-restructured ngram and skip re-restructuring

    Args:
        model: PyTorch model (SFTTrainingWrapper).
        rank: Current process rank (for logging).
    """
    causal_lm = _get_causal_lm(model)
    inner_model = causal_lm.model if hasattr(causal_lm, "model") else None
    if inner_model is None:
        return

    if hasattr(inner_model, "ngram_embeddings"):
        ngram = inner_model.ngram_embeddings
        if (hasattr(ngram, "embedders") and isinstance(ngram.embedders, nn.ModuleList)
                and hasattr(ngram, "post_projs") and isinstance(ngram.post_projs, nn.ModuleList)
                and len(ngram.embedders) > 0):
            _restructure_ngram_embedders(ngram, rank)
        elif hasattr(ngram, "embedder_units"):
            if rank == 0:
                logger.info(f"NgramEmbedding already restructured, skipping")


def setup_fsdp2(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    activation_checkpointing: bool = True,
    rank: int = 0,
    task: str = "understand",
    offload_to_cpu: bool = False,
) -> nn.Module:
    """Apply FSDP2 (fully_shard) to model for distributed training.

    Strategy: shard every large sub-module individually so that FSDP2 can
    all-gather and release each one independently during forward/backward.

    Sharding order (bottom-up — children before parents):
    1. Apply activation checkpointing to decoder layers
    2. Shard each transformer decoder layer (14 layers, ~1.35B each)
    3. Shard each ngram sub-embedder (12 embedders, ~2.62B each = 4.88GB bf16)
    4. Shard visual/audio tokenizer sub-modules
    5. Shard lm_head, visual_head, audio_head
    6. Shard root model (collects only remaining small params: norm, rotary_emb, etc.)

    All large sub-modules are sharded individually to prevent OOM from root
    all-gather. Without step 3-5, the root group would hold ~54B params, and
    its all-gather would need ~109GB on each GPU, causing OOM on H800 (140GB).

    FSDP2 correctly handles sharded modules that are not called during forward:
    they stay in sharded state and their backward hooks are not triggered.

    NOTE: break_parameter_sharing() MUST be called before this function.

    Args:
        model: PyTorch model (SFTTrainingWrapper). Can be on meta device.
        dp_mesh: DeviceMesh for data parallelism.
        activation_checkpointing: Whether to enable activation checkpointing.
        rank: Current process rank (for logging).
        task: "understand" or "generate" — determines task-specific behavior.

    Returns:
        The same model, modified in-place with FSDP2 sharding.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    offload_policy = CPUOffloadPolicy(pin_memory=True) if offload_to_cpu else None

    # Navigate to inner models
    causal_lm = _get_causal_lm(model)
    inner_model = causal_lm.model if hasattr(causal_lm, "model") else None

    # Verify parameter sharing has been broken
    if inner_model and hasattr(inner_model, "embed_tokens") and "embed_tokens" in inner_model._modules:
        if rank == 0:
            logger.warning("break_parameter_sharing() was not called before setup_fsdp2()! "
                  "Calling it now, but state dict keys may not match.")
        _break_parameter_sharing(model, inner_model, rank)

    # Step 1: Find transformer decoder layer class
    decoder_layer_cls = _find_decoder_layer_class(model)
    if decoder_layer_cls and rank == 0:
        logger.info(f"Found decoder layer class: {decoder_layer_cls.__name__}")

    t_start = time.time()
    shard_count = 0

    # Step 2: Apply FSDP + activation checkpointing to each decoder layer.
    #
    # AC strategy: wrap each layer's forward with torch.utils.checkpoint
    # (use_reentrant=False) AFTER FSDP2 sharding, so FSDP2's all-gather/reshard
    # hooks run normally (outside the checkpoint scope) and only the layer's
    # internal computation is recomputed. The composable checkpoint API
    # (torch.distributed._composable.checkpoint_activation) is NOT used: it
    # mismatches FSDP2's autograd node count between forward and recomputation,
    # raising CheckpointError.
    #
    # Router fp32: shard the router as a nested FSDP unit with an fp32 policy
    # BEFORE sharding the parent layer, so its classifier weight stays fp32
    # instead of being cast to bf16 with the rest of the layer.
    fp32_router_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
    )
    if inner_model and hasattr(inner_model, "layers"):
        for layer in inner_model.layers:
            if decoder_layer_cls and not isinstance(layer, decoder_layer_cls):
                continue
            # Shard router with fp32 policy BEFORE sharding the layer (nested FSDP)
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'router'):
                fully_shard(layer.mlp.router, mesh=dp_mesh, mp_policy=fp32_router_policy, offload_policy=offload_policy)
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            if activation_checkpointing:
                _apply_ac_to_layer(layer)
            shard_count += 1

    layer_count = shard_count
    if rank == 0:
        logger.info(f"Sharded {layer_count} transformer layers"
              f" (AC={'ON' if activation_checkpointing else 'OFF'})")

    # Step 4: Shard large non-layer sub-modules to prevent OOM from root all-gather.
    # Without this, root group would hold ~54B params and all-gather would need ~109GB.
    shard_count += _shard_large_submodules(model, causal_lm, inner_model, dp_mesh, mp_policy, rank, task, offload_policy=offload_policy)

    # Step 5: Shard root — collects only remaining small params (norm, rotary_emb, etc.)
    fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    elapsed = time.time() - t_start
    if rank == 0:
        logger.info(f"Root model sharded. Total sharded modules: {shard_count + 1} "
              f"({layer_count} layers + {shard_count - layer_count} submodules + 1 root). "
              f"Time: {elapsed:.1f}s")
        logger.info(f"Configuration: param=bf16, reduce=fp32, "
              f"AC={activation_checkpointing}, mesh={dp_mesh}")

    return model


def _apply_ac_to_layer(layer: nn.Module):
    """Apply activation checkpointing to a decoder layer by wrapping its forward.

    This replaces the layer's forward method with a version that uses
    torch.utils.checkpoint.checkpoint(use_reentrant=False). Unlike the
    composable checkpoint API, this approach:

    1. Applies AFTER fully_shard — FSDP2's hooks are already registered
    2. Only checkpoints the layer's internal computation (not FSDP2 hooks)
    3. Avoids the tensor-count mismatch bug in composable AC + FSDP2

    The FSDP2 hook execution order for a checkpointed layer:
      Forward:  FSDP2 pre_forward → [checkpoint scope] layer.forward() → FSDP2 post_forward
      Backward: FSDP2 pre_backward → [checkpoint recompute] layer.forward() → FSDP2 post_backward

    Since checkpoint wraps only the forward method (inside FSDP2's hooks),
    the tensor counts during forward and recomputation are identical.

    NOTE: MoE layers have data-dependent control flow (expert routing skips
    empty experts via `if token_indices.numel() > 0`), which creates different
    numbers of intermediate tensors between forward and recomputation when
    FSDP2 is active — causing CheckpointError with use_reentrant=False.

    We fix this by monkey-patching the MoE.moe() method to remove the
    data-dependent branch. Instead, all experts always execute (with empty
    tensor inputs for inactive experts), producing a deterministic tensor
    count. This is safe because empty-tensor operations produce zero-element
    results that don't affect the output.
    """
    _patch_moe_for_checkpoint(model=layer)
    original_forward = layer.forward

    @functools.wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        # Only checkpoint during training (not eval/inference)
        if not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            **kwargs,
        )

    layer.forward = checkpointed_forward


_moe_patched = False


def _find_moe_class(model=None):
    """Find the MoE class from the model instance.

    Avoids importing from transformers package since the actual class comes
    from the model directory (trust_remote_code=True).
    """
    if model is not None:
        for name, module in model.named_modules():
            if type(module).__name__ == "LongcatFlashMoE":
                return type(module)
    return None


def _patch_moe_for_checkpoint(model=None, use_grouped_gemm=False):
    """Patch MoE to remove data-dependent branching for checkpoint compatibility.

    The original MoE.moe() has:
        if token_indices.numel() > 0:
            expert_output = expert(expert_input)
            ...

    This creates different numbers of tensors depending on the input data
    (different expert routing → different number of active experts → different
    tensor count in the autograd graph). While this is fine normally, it
    breaks `use_reentrant=False` checkpointing because the checkpoint
    mechanism requires the exact same number of tensors during forward and
    recomputation.

    When FSDP2 is active, the autograd context differs between forward and
    recomputation (FSDP2's RegisterPostBackwardFunction creates extra nodes
    during forward but not during recomputation), which can cause the MoE
    routing to produce different tensor counts.

    Fix: Remove the `if token_indices.numel() > 0` guard. All experts always
    execute, even with empty inputs. Empty-tensor operations (matmul on 0×N
    tensors) are essentially free and produce correct zero-element outputs.
    This makes the tensor count deterministic regardless of routing.
    """
    global _moe_patched
    if _moe_patched:
        return
    _moe_patched = True

    try:
        LongcatFlashMoE = _find_moe_class(model)
        if LongcatFlashMoE is None:
            logger.warning("Could not find LongcatFlashMoE class from model, skipping MoE patch")
            return

        def _deterministic_moe(self, hidden_states, topk_indices, topk_weights):
            """MoE forward with a deterministic token dispatch order.

            Two modes:
            - use_grouped_gemm=True: uses the grouped_gemm library + unpermute pattern
            - use_grouped_gemm=False (default): Uses module calls + index_add_ pattern
            """
            topk_weights = topk_weights.to(hidden_states.dtype)
            orig_shape = hidden_states.shape
            hidden_states_for_dispatch = hidden_states.view(-1, hidden_states.shape[-1])
            _hs_orig_for_identity = hidden_states.view(-1, hidden_states.shape[-1])
            hidden_states = hidden_states_for_dispatch
            num_tokens, hidden_size = hidden_states.shape
            topk = topk_indices.shape[1]

            n_total = len(self.experts)
            n_routed = 256
            if hasattr(self, 'config') and hasattr(self.config, 'n_routed_experts'):
                n_routed = self.config.n_routed_experts

            # === Step 1: Permute tokens by expert ===
            # Full argsort over [num_tokens*topk], then truncate to [:num_out_tokens].
            # A full stable argsort (rather than filtering routed tokens first) keeps the
            # sort tie-breaking independent of input size, so the backward reduction order
            # is deterministic across runs.
            flat_indices = topk_indices.view(-1)             # [N*topk], includes identity ids

            # Full stable argsort over all [token, expert] pairs.
            sorted_indices_full = torch.argsort(flat_indices, stable=True)  # length N*topk
            # number of routed tokens (identity ids sort to the end since id >= n_routed)
            num_out_tokens = (flat_indices < n_routed).sum().item()
            sorted_indices = sorted_indices_full[:num_out_tokens]  # keep only routed tokens

            sorted_token_ids = sorted_indices // topk  # token id
            sorted_expert_ids = flat_indices[sorted_indices]

            tokens_per_expert = torch.zeros(n_routed, dtype=torch.long, device=hidden_states.device)
            if sorted_expert_ids.numel() > 0:
                tokens_per_expert.scatter_add_(0, sorted_expert_ids.long(),
                                               torch.ones_like(sorted_expert_ids, dtype=torch.long))

            # Use index_select (not fancy indexing) for a deterministic gather.
            permuted_input = hidden_states.index_select(0, sorted_token_ids)

            if use_grouped_gemm:
                # === grouped_gemm path ===
                from grouped_gemm.backend import gmm as gg_gmm

                # Expert weight layout for grouped GEMM:
                #   w1 = [E, 2*ffn, H] from concat([gate, up], dim=0)
                #   w2 = [E, H, ffn]
                # Stacked weights must be contiguous with the same dtype as the input so the
                # gmm kernel's access pattern gives a deterministic backward reduction order.
                w1 = torch.stack([
                    torch.cat([self.experts[e].gate_proj.weight,
                               self.experts[e].up_proj.weight], dim=0)
                    for e in range(n_routed)
                ]).contiguous()
                w2 = torch.stack([
                    self.experts[e].down_proj.weight
                    for e in range(n_routed)
                ]).contiguous()
                # sanity-check the layout
                assert w1.is_contiguous(), f"w1 not contiguous after stack"
                assert w2.is_contiguous(), f"w2 not contiguous after stack"
                assert w1.shape[1] % 2 == 0, f"w1[1] should be even (gate||up), got {w1.shape}"
                batch_sizes = tokens_per_expert.cpu().to(torch.long)

                class _GroupedGemmFunc(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, x, weight, batch_sizes):
                        ctx.save_for_backward(x, weight, batch_sizes)
                        return gg_gmm(x, weight, batch_sizes, trans_b=True)
                    @staticmethod
                    def backward(ctx, grad_output):
                        x, weight, batch_sizes = ctx.saved_tensors
                        grad_input = gg_gmm(grad_output, weight, batch_sizes, trans_b=False)
                        grad_weight = gg_gmm(grad_output, x, batch_sizes, trans_a=True)
                        return grad_input, grad_weight, None

                fc1_out = _GroupedGemmFunc.apply(permuted_input, w1, batch_sizes)
                gate_out, up_out = fc1_out.chunk(2, dim=-1)
                act_out = torch.nn.functional.silu(gate_out) * up_out
                permuted_output = _GroupedGemmFunc.apply(act_out, w2, batch_sizes)

                # Unpermute: index_copy_ -> view -> mul -> sum
                flat_output = torch.zeros(num_tokens * topk, hidden_size,
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
                flat_output.index_copy_(0, sorted_indices, permuted_output)
                flat_output = flat_output.view(num_tokens, topk, hidden_size)
                flat_output = flat_output * topk_weights.unsqueeze(-1)
                final_hidden_states = flat_output.sum(dim=1)
            else:
                # === Per-expert matmul + unpermute (memory-efficient) ===
                # Instead of stacking one big [E, 2*ffn, H] weight, iterate per expert and
                # use each expert's own gate/up/down weights directly.
                #
                # Memory optimization: avoid stacking one big [E, 2*ffn, H] tensor
                # (~3GB bf16 + 6GB fp32 grad for 256 experts, OOM per layer). Instead
                # iterate per expert and use
                # self.experts[e].gate_proj.weight / up_proj.weight / down_proj.weight
                # directly. This is numerically identical to the stacked version, since
                # per-expert cat([gate_e, up_e]) produces the same [2*ffn, H] tensor,
                # so the matmul result is identical.
                input_splits = list(torch.split(permuted_input, tokens_per_expert.tolist(), dim=0))

                expert_outputs = []
                for e in range(n_routed):
                    expert_input = input_splits[e]
                    # Don't fuse gate/up: two separate matmuls avoid autograd retaining a
                    # fused weight tensor per expert (which would OOM at 256 experts).
                    gate_out = torch.matmul(expert_input, self.experts[e].gate_proj.weight.t())
                    up_out = torch.matmul(expert_input, self.experts[e].up_proj.weight.t())
                    act_out = torch.nn.functional.silu(gate_out) * up_out
                    expert_output = torch.matmul(act_out, self.experts[e].down_proj.weight.t())
                    expert_outputs.append(expert_output)

                # Unpermute: index_copy_ -> view -> mul -> sum
                if len(expert_outputs) > 0:
                    permuted_output = torch.cat(expert_outputs, dim=0)

                flat_output = torch.zeros(num_tokens * topk, hidden_size,
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
                flat_output.index_copy_(0, sorted_indices, permuted_output)
                flat_output = flat_output.view(num_tokens, topk, hidden_size)
                flat_output = flat_output * topk_weights.unsqueeze(-1)
                final_hidden_states = flat_output.sum(dim=1)

            # === Identity experts (always compute for AC determinism) ===
            if n_routed < n_total:
                identity_mask = (topk_indices >= n_routed) & (topk_indices < n_total)
                iw_sum = (topk_weights * identity_mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
                final_hidden_states = final_hidden_states + _hs_orig_for_identity * iw_sum

            return final_hidden_states.view(orig_shape)

        LongcatFlashMoE.moe = _deterministic_moe

    except (ImportError, AttributeError):
        pass


def _shard_large_submodules(model, causal_lm, inner_model, dp_mesh, mp_policy, rank, task="understand", offload_policy=None):
    """Shard large sub-modules that would otherwise bloat the root FSDP group.

    IMPORTANT: All large sub-modules MUST be sharded individually to prevent OOM
    from root all-gather. Without individual sharding, the root FSDP group would
    hold ~54B params, and its all-gather would need ~109GB per GPU.

    FSDP2 correctly handles sharded modules that are not called during forward:
    - Their params stay sharded (no all-gather in forward)
    - No backward hooks fire for them
    - Root post_backward only processes groups that are in unsharded state
    So it is SAFE to shard modules even if they are not called in forward.

    Returns:
        Number of modules sharded.
    """
    if inner_model is None:
        return 0

    count = 0

    # --- Shard ngram embedder units (embedder + proj, the biggest items: 12 × 4.88GB) ---
    # We wrap each embedder+proj pair into _NgramEmbedderUnit so that FSDP2's
    # post_backward fires properly (see _NgramEmbedderUnit docstring for details).
    if hasattr(inner_model, "ngram_embeddings"):
        ngram = inner_model.ngram_embeddings
        # Check if already restructured (by restructure_ngram_for_fsdp called before setup_fsdp2)
        if hasattr(ngram, "embedder_units") and isinstance(ngram.embedder_units, nn.ModuleList):
            # Already restructured — just shard the units
            for i, unit in enumerate(ngram.embedder_units):
                fully_shard(unit, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
                count += 1
            if rank == 0:
                logger.info(f"Sharded {len(ngram.embedder_units)} ngram embedder units "
                      f"(pre-restructured)")
        elif (hasattr(ngram, "embedders") and isinstance(ngram.embedders, nn.ModuleList)
                and hasattr(ngram, "post_projs") and isinstance(ngram.post_projs, nn.ModuleList)):
            # Not yet restructured — restructure now and shard
            _restructure_ngram_embedders(ngram, rank)
            if hasattr(ngram, "embedder_units") and isinstance(ngram.embedder_units, nn.ModuleList):
                for i, unit in enumerate(ngram.embedder_units):
                    fully_shard(unit, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
                    count += 1
                if rank == 0:
                    logger.info(f"Sharded {len(ngram.embedder_units)} ngram embedder units")
        elif hasattr(ngram, "embedders") and isinstance(ngram.embedders, nn.ModuleList):
            # Fallback: shard individual embedders (may cause backward OOM)
            for i, emb in enumerate(ngram.embedders):
                fully_shard(emb, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
                count += 1
            if rank == 0:
                logger.warning(f"Sharded {len(ngram.embedders)} ngram embedders individually "
                      f"(no post_projs found, may cause backward OOM)")
        # Shard word_embeddings (shared embed_tokens, ~0.87B)
        if hasattr(ngram, "word_embeddings"):
            fully_shard(ngram.word_embeddings, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1
            if rank == 0:
                logger.info(f"Sharded ngram_embeddings.word_embeddings")

    # --- Shard visual_tokenizer sub-modules ---
    if hasattr(inner_model, "visual_tokenizer"):
        vt = inner_model.visual_tokenizer
        # visual_model (ViT encoder)
        if hasattr(vt, "visual_model"):
            fully_shard(vt.visual_model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1
            if rank == 0:
                logger.info(f"Sharded visual_tokenizer.visual_model")
        # visual_bridge_model (VQ bridge with codebooks)
        if hasattr(vt, "visual_bridge_model"):
            # VQ quantizer codebooks must stay fp32 to match the reference encoder
            # from_pretrained(dtype=bf16) behavior. We shard the `quantize`
            # submodule (which holds codebooks) with an fp32 mp_policy FIRST
            # (inner shard takes precedence), then shard the whole
            # visual_bridge_model with the default bf16 policy.
            fp32_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            )
            vbm = vt.visual_bridge_model
            if hasattr(vbm, "quantizer") and hasattr(vbm.quantizer, "quantize"):
                fully_shard(vbm.quantizer.quantize, mesh=dp_mesh,
                            mp_policy=fp32_mp_policy, offload_policy=offload_policy)
                count += 1
                if rank == 0:
                    logger.info(f"Sharded visual_tokenizer.visual_bridge_model.quantizer.quantize (fp32)")
            fully_shard(vt.visual_bridge_model, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1
            if rank == 0:
                logger.info(f"Sharded visual_tokenizer.visual_bridge_model")
        # visual_embedding_layer
        if hasattr(vt, "visual_embedding_layer"):
            fully_shard(vt.visual_embedding_layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1
            if rank == 0:
                logger.info(f"Sharded visual_tokenizer.visual_embedding_layer")

    # --- Shard audio_tokenizer sub-modules ---
    if hasattr(inner_model, "audio_tokenizer"):
        at = inner_model.audio_tokenizer
        # Shard each direct child module with parameters
        audio_sharded = 0
        for sub_name, sub_module in at.named_children():
            if sum(p.numel() for p in sub_module.parameters()) > 0:
                fully_shard(sub_module, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
                count += 1
                audio_sharded += 1
                if rank == 0:
                    logger.info(f"Sharded audio_tokenizer.{sub_name}")
        if audio_sharded == 0 and sum(p.numel() for p in at.parameters()) > 0:
            fully_shard(at, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1
            if rank == 0:
                logger.info(f"Sharded audio_tokenizer (whole)")

    # --- Shard head modules ---
    # For visual_head and audio_head (CasualDepthTransformerHead), we shard
    # their children individually rather than the parent as a whole.
    # This is necessary because training code calls sub-modules (hidden_norm,
    # transformer_layers, headnorm, heads) directly in _compute_depth_loss.
    # If the parent is sharded as a whole, calling children directly bypasses
    # FSDP2's pre_forward hooks, leaving params as DTensors and causing
    # "mixed torch.Tensor and DTensor" errors with apex fused kernels.
    for head_name in ["lm_head", "visual_head", "audio_head"]:
        if hasattr(causal_lm, head_name):
            head = getattr(causal_lm, head_name)
            if sum(p.numel() for p in head.parameters()) == 0:
                continue
            if head_name in ("visual_head", "audio_head"):
                # Shard children individually for depth transformer heads
                head_count = _shard_depth_head_children(head, head_name, dp_mesh, mp_policy, rank, offload_policy=offload_policy)
                count += head_count
            else:
                fully_shard(head, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
                count += 1
                if rank == 0:
                    logger.info(f"Sharded {head_name}")

    if rank == 0:
        logger.info(f"Total non-layer submodules sharded: {count}")

    return count


def _shard_depth_head_children(head: nn.Module, head_name: str, dp_mesh, mp_policy, rank: int, offload_policy=None) -> int:
    """Shard CasualDepthTransformerHead's children individually.

    The depth transformer head has sub-modules (hidden_norm, hidden_proj,
    transformer_layers, headnorm, heads) that are called individually during
    training loss computation. If the parent is sharded as a whole, these
    direct sub-module calls bypass FSDP2's pre_forward hooks, leaving params
    as DTensors. Apex fused RMSNorm cannot handle mixed DTensor/Tensor inputs,
    causing RuntimeError.

    By sharding children individually, each sub-module's __call__ triggers
    FSDP2's hooks, properly unsharding params before computation.

    Args:
        head: CasualDepthTransformerHead module.
        head_name: Name for logging (e.g., "visual_head").
        dp_mesh: DeviceMesh for data parallelism.
        mp_policy: Mixed precision policy.
        rank: Current process rank.

    Returns:
        Number of modules sharded.
    """
    count = 0

    # hidden_norm (RMSNorm on hidden_size)
    if hasattr(head, "hidden_norm"):
        fully_shard(head.hidden_norm, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        count += 1

    # hidden_proj (Linear: hidden_size -> transformer_dim)
    if hasattr(head, "hidden_proj"):
        fully_shard(head.hidden_proj, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        count += 1

    # transformer_layers (each CasualDepthTransformerLayer)
    if hasattr(head, "transformer_layers"):
        for layer in head.transformer_layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1

    # headnorm (RMSNorm on transformer_dim)
    if hasattr(head, "headnorm"):
        fully_shard(head.headnorm, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
        count += 1

    # heads (ModuleList of Linear layers)
    if hasattr(head, "heads"):
        for i, h in enumerate(head.heads):
            fully_shard(h, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy)
            count += 1

    if rank == 0:
        logger.info(f"Sharded {head_name} children: {count} sub-modules")

    return count


def _restructure_ngram_embedders(ngram_module: nn.Module, rank: int = 0):
    """Restructure NgramEmbedding to use _NgramEmbedderUnit wrapper modules.

    Replaces separate `embedders` and `post_projs` ModuleLists with a single
    `embedder_units` ModuleList of _NgramEmbedderUnit modules. Each unit wraps
    one embedder + one projection and receives a grad-carrying accumulator tensor,
    ensuring FSDP2's post_backward fires properly during backward.

    This modifies the NgramEmbedding in-place. The original `embedders` and
    `post_projs` ModuleLists are removed from `_modules` to avoid FSDP2
    encountering duplicate parameters.

    The NgramEmbedding.forward() method is also monkey-patched to use the
    new `embedder_units` instead of the old separate lists.

    Args:
        ngram_module: The NgramEmbedding module to restructure.
        rank: Current process rank (for logging).
    """
    embedders = ngram_module.embedders
    post_projs = ngram_module.post_projs
    n_units = len(embedders)

    if n_units != len(post_projs):
        if rank == 0:
            logger.warning(f"embedders ({n_units}) and post_projs ({len(post_projs)}) "
                  f"have different lengths, skipping restructure")
        return

    # Create units
    units = nn.ModuleList([
        _NgramEmbedderUnit(embedders[i], post_projs[i])
        for i in range(n_units)
    ])

    # Replace the module structure
    # Remove old ModuleLists from _modules registry
    del ngram_module._modules["embedders"]
    del ngram_module._modules["post_projs"]

    # Register the new ModuleList
    ngram_module.embedder_units = units

    # Keep plain attribute references for compatibility (non-registered)
    object.__setattr__(ngram_module, "embedders", embedders)
    object.__setattr__(ngram_module, "post_projs", post_projs)

    # Monkey-patch forward to use the new units
    _monkey_patch_ngram_forward(ngram_module)

    if rank == 0:
        logger.info(f"Restructured NgramEmbedding: {n_units} embedder+proj pairs → "
              f"{n_units} _NgramEmbedderUnit modules")


def _monkey_patch_ngram_forward(ngram_module: nn.Module):
    """Monkey-patch NgramEmbedding.forward to use _NgramEmbedderUnit.

    The patched forward replaces:
        for i in range(2, n+1):
            for j in range(k):
                x_ngram = self.embedders[index](new_ids, mask)
                x_proj = self.post_projs[index](x_ngram)
                x = x + x_proj

    With:
        for i in range(2, n+1):
            for j in range(k):
                x = self.embedder_units[index](new_ids, mask, x)

    The key difference: `x` (which requires grad) is passed as input to each
    unit, so FSDP2's RegisterPostBackwardFunction is registered, and
    post_backward (reshard + reduce-scatter) fires after each unit's gradient
    computation. Without this, all embedders stay unsharded during backward.
    """
    original_forward = ngram_module.forward

    @functools.wraps(original_forward)
    def patched_forward(input_ids, ngram_context=None):
        seq_len = input_ids.size(-1)
        config = ngram_module.config
        k = ngram_module.k
        n = ngram_module.n
        m = ngram_module.m

        # Determine complete context
        if ngram_context is not None:
            context = torch.cat([ngram_context[..., -(n-1):], input_ids], dim=-1)
        else:
            context = input_ids.clone()

        # Skip N-gram look-up for oe_ignored_token_ids
        oe_ignored_mask = torch.isin(
            input_ids,
            ngram_module.oe_ignored_token_ids.to(device=input_ids.device)
        )
        context[torch.isin(
            context,
            ngram_module.oe_ignored_token_ids.to(device=context.device)
        )] = 0

        # Base word embeddings
        # Note: with CPU offload, weight might be on CPU before FSDP2 unshard.
        # Don't move input_ids to weight's device — keep on CUDA.
        # FSDP2's forward hook will unshard weight to CUDA when word_embeddings() runs.
        x = ngram_module.word_embeddings(input_ids).clone()

        vocab_mods = ngram_module._precompute_vocab_mods()

        shifted_ids = {}
        for i in range(2, n + 1):
            shifted_ids[i] = ngram_module._shift_right_ignore_eos(
                context, i - 1, eos_token_id=config.eos_token_id
            )

        # Add N-gram embeddings using _NgramEmbedderUnit
        for i in range(2, n + 1):
            for j in range(k):
                index = (i - 2) * k + j
                emb_vocab_dim = int(m + index * 2 + 1)

                ngram_ids = ngram_module._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
                text_mask = new_ids > 0

                # Call the unit with x_accum (requires grad) — ensures FSDP2
                # post_backward fires after this unit's gradient is computed
                x = ngram_module.embedder_units[index](
                    new_ids.to(x.device), text_mask, x
                )

        # Normalize
        x[~oe_ignored_mask] /= (1 + k * (n - 1))

        return x

    ngram_module.forward = patched_forward


def _break_parameter_sharing(model, inner_model, rank):
    """Break module-level parameter sharing that causes FSDP2 double-sharding.

    LongCat-Next has two sharing patterns:
    1. embed_tokens <-> ngram_embeddings.word_embeddings: same nn.Embedding
    2. visual_bridge codebooks[0..7]: 8 ModuleList entries = 1 VQEmbedding

    For (1): Remove embed_tokens from _modules registry but keep it as a plain
    Python attribute. FSDP won't walk it (not in _modules), but model code
    like self.embed_tokens(x) still works.

    For (2): Replace nn.ModuleList with _SharedModuleList that registers the
    shared VQEmbedding only once.
    """
    if inner_model is None:
        if rank == 0:
            logger.info(f"_break_parameter_sharing: inner_model is None, skipping")
        return

    if rank == 0:
        logger.info(f"Breaking parameter sharing for {type(inner_model).__name__}...")

    # --- Fix (1): embed_tokens <-> ngram_embeddings.word_embeddings ---
    if (hasattr(inner_model, "embed_tokens")
            and hasattr(inner_model, "ngram_embeddings")
            and hasattr(inner_model.ngram_embeddings, "word_embeddings")):

        et = inner_model.embed_tokens
        we = inner_model.ngram_embeddings.word_embeddings

        if et is we or (hasattr(et, "weight") and hasattr(we, "weight") and et.weight is we.weight):
            # Same object or same weight — remove from _modules, keep as plain attribute
            embed_ref = inner_model.embed_tokens
            del inner_model._modules["embed_tokens"]
            object.__setattr__(inner_model, "embed_tokens", embed_ref)
            if rank == 0:
                logger.info(f"Broke embed_tokens <-> word_embeddings sharing")
        else:
            if rank == 0:
                logger.info(f"embed_tokens and word_embeddings are separate — no sharing")

    # --- Fix (2): visual codebook sharing ---
    if hasattr(inner_model, "visual_tokenizer"):
        vt = inner_model.visual_tokenizer
        if hasattr(vt, "visual_bridge_model"):
            _dedup_shared_codebooks(vt.visual_bridge_model, "quantizer.quantize.codebooks", rank)

    # Also check audio_tokenizer (defensively)
    if hasattr(inner_model, "audio_tokenizer"):
        at = inner_model.audio_tokenizer
        if hasattr(at, "audio_bridge_model"):
            _dedup_shared_codebooks(at.audio_bridge_model, "vq_list", rank, attr_chain=False)

    # --- Safety check: warn if any parameters are still shared (would break
    # FSDP2 sharding) ---
    if rank == 0:
        seen = {}
        shared_found = False
        for name, param in model.named_parameters():
            pid = id(param)
            if pid in seen:
                logger.warning(f"shared param: '{name}' <-> '{seen[pid]}'")
                shared_found = True
            else:
                seen[pid] = name
        if not shared_found:
            logger.info(f"No shared parameters remaining")


class _SharedModuleList(nn.Module):
    """Drop-in replacement for nn.ModuleList when all entries share one module.

    Registers the shared module only once as a child (avoiding FSDP
    double-sharding), but returns it for any index, preserving iteration
    semantics for code like: codebooks[i](x) for i in range(N).
    """

    def __init__(self, shared_module: nn.Module, length: int):
        super().__init__()
        self._shared = shared_module
        self._length = length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._length:
            raise IndexError(f"index {idx} out of range [0, {self._length})")
        return self._shared

    def __len__(self):
        return self._length

    def __iter__(self):
        for _ in range(self._length):
            yield self._shared

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self


def _dedup_shared_codebooks(parent_module, attr_path, rank, attr_chain=True):
    """De-duplicate a ModuleList where all entries point to the same object.

    Replaces nn.ModuleList with _SharedModuleList that registers the shared
    module only once for FSDP but still returns it for all indices.
    """
    container = parent_module
    path_parts = attr_path.split(".") if attr_chain else [attr_path]

    for part in path_parts[:-1]:
        if hasattr(container, part):
            container = getattr(container, part)
        else:
            return

    last_attr = path_parts[-1]
    if not hasattr(container, last_attr):
        return

    module_list = getattr(container, last_attr)
    if not isinstance(module_list, nn.ModuleList) or len(module_list) < 2:
        return

    # Check if all entries are the same object
    first = module_list[0]
    all_same = all(module_list[i] is first for i in range(1, len(module_list)))
    if not all_same:
        return

    shared_list = _SharedModuleList(first, len(module_list))
    setattr(container, last_attr, shared_list)

    if rank == 0:
        logger.info(f"Replaced shared codebook ModuleList({len(module_list)}) "
              f"with _SharedModuleList (1 module, {len(module_list)} indices)")


def _find_decoder_layer_class(model):
    """Find the transformer decoder layer class from the model.

    Dynamically discovers the class from the model instance rather than
    importing from transformers, because the actual class comes from
    the model directory (trust_remote_code=True) and may differ from
    the transformers package version.
    """
    for name, module in model.named_modules():
        if type(module).__name__ == "LongcatFlashDecoderLayer":
            return type(module)

    logger.warning("Could not find LongcatFlashDecoderLayer class.")
    return None


def _get_causal_lm(model):
    """Navigate to LongcatNextForCausalLM inside SFTTrainingWrapper."""
    if hasattr(model, "model"):
        return model.model
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    step: int,
    save_dir: str,
    model_path: str = "",
    dataloader_state: dict = None,
):
    """Save checkpoint in HuggingFace format + optimizer states for resume.

    Strategy: all-gather via get_model_state_dict (NCCL, fast over NVLink/RDMA),
    then rank 0 writes HF shard files using multi-threaded I/O.

      Phase 1: get_model_state_dict() → all-gather full params to every rank.
               Fast: ~150GB over NVLink/IB takes only seconds.
      Phase 2: Rank 0 writes HF shard files using a thread pool (8 threads).
               Each thread writes one shard file (~10GB) in parallel.
      Phase 3: rank 0 handles codebook replication, missing key supplement,
               index.json, config copy.

    Also saves:
      - optimizer_rankN.pt: per-rank optimizer state (for training resume)
      - scheduler.pt: LR scheduler state (rank 0 only)
      - train_state.pt: training metadata like step number (rank 0 only)
    """
    import shutil
    import json
    import time as _time
    from safetensors.torch import save_file
    from safetensors import safe_open
    from collections import defaultdict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
    from model.model_loader import _fsdp_name_to_hf_key

    save_path = os.path.join(save_dir, f"step_{step:07d}")
    hf_path = os.path.join(save_path, "huggingface")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        logger.info(f"Starting checkpoint save at step {step} to {save_path} ...")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(hf_path, exist_ok=True)
    dist.barrier()

    t0 = _time.time()

    # ── Phase 1: all-gather full state dict (NCCL, fast) ──
    if rank == 0:
        logger.info(f"Phase 1: all-gather model state via NCCL ...")

    # get_model_state_dict does FSDP all-gather internally.
    # cpu_offload=True: each FSDP group is all-gathered then immediately moved to CPU,
    # so GPU never holds the full 150GB model at once (avoids OOM).
    state_dict = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    # Convert FSDP param names to HF key names (tensors already on CPU from cpu_offload)
    hf_state = {}
    for fsdp_name, tensor in state_dict.items():
        hf_key = _fsdp_name_to_hf_key(fsdp_name)
        hf_state[hf_key] = tensor.contiguous()
    del state_dict

    # Also gather named_buffers not in state_dict (e.g., codebook buffers)
    for buf_name, buf in model.named_buffers():
        hf_key = _fsdp_name_to_hf_key(buf_name)
        if hf_key not in hf_state:
            hf_state[hf_key] = buf.detach().cpu().contiguous()

    if rank == 0:
        total_bytes = sum(t.numel() * t.element_size() for t in hf_state.values())
        logger.info(f"Phase 1 done: {len(hf_state)} keys, {total_bytes/1e9:.1f} GB "
              f"in {_time.time()-t0:.1f}s")

    # ── Phase 2: rank 0 writes HF shard files (multi-threaded) ──
    t2 = _time.time()

    orig_index_path = os.path.join(model_path, "model.safetensors.index.json") if model_path else ""
    weight_map = None
    if orig_index_path and os.path.exists(orig_index_path):
        with open(orig_index_path, "r") as f:
            orig_index = json.load(f)
        weight_map = orig_index["weight_map"]

    if weight_map is None:
        if rank == 0:
            logger.warning("no original index found, falling back to single-file save")
            save_file(hf_state, os.path.join(hf_path, "model.safetensors"))
        del hf_state
        dist.barrier()
        return

    # Group original keys by HF shard file (needed by Phase 2 + 3)
    filename_to_keys = defaultdict(list)
    for key, filename in weight_map.items():
        filename_to_keys[filename].append(key)
    shard_names = sorted(filename_to_keys.keys())
    total_shards = len(shard_names)

    if rank == 0:
        from concurrent.futures import ThreadPoolExecutor

        logger.info(f"Phase 2: writing {total_shards} HF shards with 8 threads ...")

        def _write_shard(shard_name):
            """Write one HF shard file."""
            ts = _time.time()
            keys_for_shard = filename_to_keys[shard_name]
            shard_data = {}
            for key in keys_for_shard:
                if key in hf_state:
                    shard_data[key] = hf_state[key]
            if shard_data:
                save_file(shard_data, os.path.join(hf_path, shard_name))
            elapsed = _time.time() - ts
            shard_bytes = sum(t.numel() * t.element_size() for t in shard_data.values())
            return shard_name, len(shard_data), shard_bytes, elapsed

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_write_shard, sn) for sn in shard_names]
            for i, future in enumerate(futures):
                sn, nkeys, sbytes, elapsed = future.result()
                logger.info(f"Shard {i+1}/{total_shards} ({sn}): "
                      f"{nkeys} keys, {sbytes/1e9:.1f} GB in {elapsed:.1f}s")

        logger.info(f"Phase 2 done: {total_shards} shards written in {_time.time()-t2:.1f}s")

    del hf_state
    dist.barrier()

    # ── Phase 3: rank 0 post-processing ──
    if rank == 0:
        t3 = _time.time()
        logger.info(f"Phase 3: Post-processing (codebook, missing keys, index) ...")

        # Count saved keys
        saved_keys = set()
        for sn in shard_names:
            sp = os.path.join(hf_path, sn)
            if os.path.exists(sp):
                with safe_open(sp, framework="pt", device="cpu") as f:
                    saved_keys.update(f.keys())
        logger.info(f"Phase 2 saved {len(saved_keys)} keys")

        # Codebook replication: codebooks.0 → codebooks.1~7
        codebook_prefix = "model.visual_tokenizer.visual_bridge_model.quantizer.quantize.codebooks.0."
        codebook_suffixes = ["cluster_size_ema", "embed", "embed_ema"]
        patches = defaultdict(dict)  # shard_name -> {key: tensor}

        for suffix in codebook_suffixes:
            src_key = codebook_prefix + suffix
            if src_key in saved_keys:
                src_shard = weight_map[src_key]
                with safe_open(os.path.join(hf_path, src_shard), framework="pt", device="cpu") as f:
                    src_tensor = f.get_tensor(src_key)
                for i in range(1, 8):
                    dst_key = src_key.replace("codebooks.0.", f"codebooks.{i}.")
                    if dst_key not in saved_keys:
                        dst_shard = weight_map.get(dst_key, src_shard)
                        patches[dst_shard][dst_key] = src_tensor.clone()

        # Supplement missing keys from original model
        all_orig_keys = set(weight_map.keys())
        patched_keys = set(k for p in patches.values() for k in p)
        still_missing = all_orig_keys - saved_keys - patched_keys
        if still_missing:
            logger.info(f"Supplementing {len(still_missing)} missing keys from original model ...")
            miss_by_shard = defaultdict(list)
            for mk in still_missing:
                miss_by_shard[weight_map[mk]].append(mk)
            for sf, keys in miss_by_shard.items():
                with safe_open(os.path.join(model_path, sf), framework="pt", device="cpu") as f:
                    for k in keys:
                        patches[sf][k] = f.get_tensor(k)

        # Apply patches to shard files
        for shard_name, additions in patches.items():
            shard_path = os.path.join(hf_path, shard_name)
            existing = {}
            if os.path.exists(shard_path):
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        existing[key] = f.get_tensor(key)
            existing.update(additions)
            save_file(existing, shard_path)
            logger.info(f"Patched {shard_name}: +{len(additions)} keys")

        # Verify and write index
        final_keys = set()
        total_size = 0
        for sn in shard_names:
            sp = os.path.join(hf_path, sn)
            if os.path.exists(sp):
                with safe_open(sp, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        final_keys.add(key)
                        shape = f.get_slice(key).get_shape()
                        numel = 1
                        for s in shape:
                            numel *= s
                        total_size += numel * 2  # bf16

        logger.info(f"Final: {len(final_keys)} keys (original: {len(all_orig_keys)})")
        if final_keys != all_orig_keys:
            missing_f = sorted(all_orig_keys - final_keys)
            extra_f = sorted(final_keys - all_orig_keys)
            if missing_f:
                logger.warning(f"{len(missing_f)} still missing: {missing_f[:5]}")
            if extra_f:
                logger.warning(f"{len(extra_f)} extra keys: {extra_f[:5]}")

        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(hf_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        logger.info(f"Index: {total_shards} shards, ~{total_size/1e9:.1f} GB")

        # Copy config/code files
        if model_path and os.path.isdir(model_path):
            copy_exts = {".json", ".py"}
            skip_files = {"model.safetensors.index.json"}
            copied = []
            for fname in os.listdir(model_path):
                if fname == "__pycache__" or fname in skip_files:
                    continue
                if fname.endswith(".safetensors") or fname.endswith(".bin"):
                    continue
                fpath = os.path.join(model_path, fname)
                if os.path.isfile(fpath):
                    _, ext = os.path.splitext(fname)
                    if ext in copy_exts:
                        shutil.copy2(fpath, os.path.join(hf_path, fname))
                        copied.append(fname)
            logger.info(f"Copied {len(copied)} config/code files")

        logger.info(f"Phase 3 done in {_time.time()-t3:.1f}s")

    # ── Save optimizer (each rank saves its own shard) ──
    if optimizer is not None:
        if rank == 0:
            logger.info(f"Saving optimizer state ...")
        torch.save(optimizer.state_dict(), os.path.join(save_path, f"optimizer_rank{rank}.pt"))

    # ── Save scheduler + train state (rank 0 only) ──
    if rank == 0:
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        train_state = {"step": step}
        # Save dataloader state if provided (for StatefulDataLoader resume)
        if dataloader_state is not None:
            train_state["dataloader_state"] = dataloader_state
        torch.save(train_state, os.path.join(save_path, "train_state.pt"))

    dist.barrier()
    if rank == 0:
        logger.info(f"Checkpoint save completed at step {step} ({_time.time()-t0:.1f}s total)")


def load_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    resume_from: str,
    device: torch.device,
):
    """Load checkpoint for resuming training.

    Args:
        model: FSDP-wrapped model (already initialized with weights)
        optimizer: FP32AdamW optimizer instance
        scheduler: LR scheduler instance
        resume_from: path to checkpoint directory (e.g. /path/checkpoints/step_0000005)
        device: current device

    Returns:
        step: the global step number to resume from
    """
    rank = dist.get_rank()

    if rank == 0:
        logger.info(f"Resuming from checkpoint: {resume_from}")

    # 1. Load training state to get the step number
    train_state_path = os.path.join(resume_from, "train_state.pt")
    if not os.path.exists(train_state_path):
        raise FileNotFoundError(f"train_state.pt not found in {resume_from}")
    train_state = torch.load(train_state_path, map_location="cpu", weights_only=False)
    step = train_state["step"]
    dataloader_state = train_state.get("dataloader_state", None)
    if rank == 0:
        logger.info(f"  Resuming from step {step}")
        if dataloader_state is not None:
            logger.info(f"  Found dataloader state for StatefulDataLoader resume")
        else:
            logger.info(f"  No dataloader state found, will use fast-forward fallback")

    # 2. Load model weights
    #    Checkpoint is saved in HF safetensors format (with HF key names).
    #    Use load_fsdp2_weights_manual which handles HF key → FSDP key mapping.
    #    New layout: huggingface/ subdirectory; fallback to flat layout for compat.
    from model.model_loader import load_fsdp2_weights_manual

    hf_subdir = os.path.join(resume_from, "huggingface")
    if os.path.isdir(hf_subdir):
        model_dir = hf_subdir
    else:
        model_dir = resume_from  # fallback: flat layout (old checkpoints)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        load_fsdp2_weights_manual(model, model_dir, rank=rank)
        if rank == 0:
            logger.info(f"  Loaded model weights from {model_dir}")
    else:
        if rank == 0:
            logger.warning(f"  No model weights found in {resume_from}, "
                  f"using current model weights")

    # 3. Load optimizer state (rank-local shard)
    if optimizer is not None:
        opt_path = os.path.join(resume_from, f"optimizer_rank{rank}.pt")
        if os.path.exists(opt_path):
            opt_state = torch.load(opt_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(opt_state)
            if rank == 0:
                logger.info(f"  Loaded optimizer state (per-rank shards)")
        else:
            if rank == 0:
                logger.warning(f"  optimizer_rank{rank}.pt not found, "
                      f"starting optimizer from scratch")

    # 4. Load scheduler state
    if scheduler is not None:
        sched_path = os.path.join(resume_from, "scheduler.pt")
        if os.path.exists(sched_path):
            sched_state = torch.load(sched_path, map_location="cpu", weights_only=True)
            scheduler.load_state_dict(sched_state)
            if rank == 0:
                logger.info(f"  Loaded scheduler state")

    dist.barrier()
    return step, dataloader_state
