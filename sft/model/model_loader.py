# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Model loading and freeze control for LongCat-Next FSDP2 SFT training.

Standard PyTorch FSDP2 weight-loading pattern for large models:
- Rank 0: load model on CPU via from_pretrained (holds full weights)
- Other ranks: create model on meta device (zero memory)
- After FSDP2 sharding: use set_model_state_dict(broadcast_from_rank0=True)
  to distribute sharded weights to all ranks
- Broadcast non-persistent buffers manually (they are not in state_dict)
"""

import os
import time
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16, rank: int = 0):
    """Load LongCat-Next model on CPU (rank 0) / meta device (other ranks).

    Rank 0 loads full model on CPU (holds all weights).
    Other ranks create model on meta device (zero memory).

    Args:
        model_path: Path to the HuggingFace model directory.
        dtype: Model parameter dtype.
        rank: Current process rank (for logging).

    Returns:
        Tuple of (model, tokenizer, processor).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

    logger.info(f"[Rank {rank}] Loading tokenizer and processor...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except (AttributeError, ImportError) as e:
        logger.warning(f"[Rank {rank}] AutoProcessor failed ({e}), "
              f"falling back to Qwen2VLImageProcessor")
        from transformers import Qwen2VLImageProcessor
        processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        processor.image_processor = processor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"

    if rank == 0:
        # Rank 0: load full model on CPU with real weights
        logger.info(f"[Rank {rank}] Loading model on CPU (rank 0 only)...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        elapsed = time.time() - t0
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"[Rank {rank}] Model loaded on CPU: {total_params:.2f}B params, "
              f"{elapsed:.1f}s")
    else:
        # Other ranks: create model on meta device (zero memory)
        logger.info(f"[Rank {rank}] Creating model on meta device...")
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, torch_dtype=dtype,
            )
        total_params = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"[Rank {rank}] Model created (meta device). {total_params:.2f}B params")

    return model, tokenizer, processor


def load_model_meta(model_path: str, dtype: torch.dtype = torch.bfloat16, rank: int = 0,
                    rope_theta: float = 0.0):
    """Load LongCat-Next model structure on meta device for ALL ranks.

    This avoids the slow mmap-based weight loading from from_pretrained.
    All ranks create the model on meta device (zero memory), and weights
    are loaded later via load_fsdp2_weights_manual which reads safetensors
    files directly and broadcasts per-parameter.

    Args:
        model_path: Path to the HuggingFace model directory.
        dtype: Model parameter dtype.
        rank: Current process rank (for logging).
        rope_theta: If > 0, override the model config's rope_theta (e.g. some
                    generation setups use 1e6).

    Returns:
        Tuple of (model, tokenizer, processor).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

    logger.info(f"[Rank {rank}] Loading tokenizer and processor...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
    except (AttributeError, ImportError) as e:
        # The model checkpoint's processing_longcat_next.py may be outdated
        # and missing LongcatNextProcessor. Fall back to loading just the
        # Qwen2VLImageProcessor which is what ImagePreprocessor actually uses.
        logger.warning(f"[Rank {rank}] AutoProcessor failed ({e}), "
              f"falling back to Qwen2VLImageProcessor")
        from transformers import Qwen2VLImageProcessor
        processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        # Wrap it so that processor.image_processor returns itself
        # (ImagePreprocessor checks hasattr(processor, 'image_processor'))
        processor.image_processor = processor

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"

    # Override rope_theta if requested (e.g. 1e6 for the generation task)
    if rope_theta > 0:
        original_rope_theta = getattr(config, "rope_theta", None)
        config.rope_theta = rope_theta
        if rank == 0:
            logger.info(f"[Rank {rank}] rope_theta overridden: {original_rope_theta} -> {rope_theta}")

    logger.info(f"[Rank {rank}] Creating model on meta device...")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=dtype,
        )
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"[Rank {rank}] Model created (meta device). {total_params:.2f}B params")

    # Patch the ViT attention class to a flash-attn implementation.
    _patch_vit_attn_forward(model, rank=rank)

    return model, tokenizer, processor


def extract_state_dict(model, rank: int = 0):
    """Extract state dict from model BEFORE FSDP2 sharding.

    Must be called before setup_fsdp2(). Only rank 0 has real weights
    (from from_pretrained), so only rank 0 extracts the state dict.

    After extraction, rank 0's model parameters are moved to meta device
    to avoid slow mmap page-in during FSDP2's _move_states_to_device.
    The state dict retains references to the original mmap storage, which
    remains valid until the state dict is freed.

    Args:
        model: Model (not yet FSDP2-sharded). Rank 0 has CPU weights,
               others have meta tensors.
        rank: Current process rank.

    Returns:
        State dict (non-empty only on rank 0).
    """
    if rank == 0:
        t0 = time.time()
        logger.info(f"[Rank {rank}] Extracting state dict before FSDP2...")
        # NOTE: We intentionally do NOT clone the tensors here. The model was
        # loaded with low_cpu_mem_usage=True (mmap), so state_dict values are
        # mmap views into the safetensor files. Cloning would force page-in of
        # the entire 147GB model, which takes >30 min on slow DFS and causes
        # NCCL timeouts. The mmap views remain valid as long as the files exist.
        full_state = dict(model.state_dict())
        elapsed = time.time() - t0
        logger.info(f"[Rank {rank}] State dict extracted: {len(full_state)} keys, "
              f"{elapsed:.1f}s")

        # Move rank 0's model to meta device. This frees the model's references
        # to the mmap storage (the state dict still holds its own references).
        # Without this, fully_shard would trigger _move_states_to_device which
        # moves each CPU param to CUDA one-by-one, causing extremely slow mmap
        # page-in from DFS (>70 min for 73B model) while other ranks wait,
        # leading to NCCL timeouts.
        _move_model_to_meta(model)
        logger.info(f"[Rank {rank}] Model moved to meta device after state dict extraction")

        return full_state
    else:
        return {}


def _move_model_to_meta(model: nn.Module):
    """Move all parameters and buffers of a model to meta device.

    This avoids the slow mmap page-in that occurs when FSDP2's
    _move_states_to_device tries to move CPU (mmap) parameters to CUDA.
    FSDP2 skips meta tensors in _move_states_to_device.

    Uses torch.utils.swap_tensors for safe parameter replacement
    (param.data = ... raises incompatible tensor type errors).
    """
    for param in model.parameters():
        meta_param = nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype, device="meta"),
            requires_grad=param.requires_grad,
        )
        torch.utils.swap_tensors(param, meta_param)
    for buf_name, buf in model.named_buffers():
        if buf.device.type != "meta":
            meta_buf = torch.empty(buf.shape, dtype=buf.dtype, device="meta")
            torch.utils.swap_tensors(buf, meta_buf)


def load_fsdp2_weights_manual(model, model_path: str, rank: int = 0,
                               target_device: torch.device = None):
    """Load weights into FSDP2-sharded model via streaming per-parameter broadcast.

    This is the primary weight loading method. It avoids the slow mmap page-in
    issues of set_model_state_dict by loading safetensors files one at a time
    and broadcasting each parameter directly to all ranks.

    Flow:
    1. Build mapping from FSDP parameter names to HuggingFace checkpoint keys
    2. Rank 0 loads safetensors files one at a time
    3. For each parameter, rank 0 loads the full weight to GPU, all ranks
       broadcast, then each rank extracts its local shard

    The model MUST already be materialized via to_empty(device=target_device) before
    calling this function.

    Args:
        model: FSDP2-sharded model (already materialized via to_empty).
        model_path: Path to HuggingFace checkpoint directory.
        rank: Current process rank.
        target_device: Device where parameters should end up. Default is CUDA.
                      Use CPU when FSDP2 CPU offload is enabled.
    """
    import json
    import glob
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor

    world_size = dist.get_world_size()
    gpu_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    # target_device is where the final parameters live (CPU for offload, GPU otherwise)
    if target_device is None:
        target_device = gpu_device
    t0 = time.time()

    # Step 1: Build FSDP name -> HF key mapping
    fsdp_to_hf = {}
    for fsdp_name, _ in model.named_parameters():
        fsdp_to_hf[fsdp_name] = _fsdp_name_to_hf_key(fsdp_name)
    for buf_name, _ in model.named_buffers():
        fsdp_to_hf[buf_name] = _fsdp_name_to_hf_key(buf_name)

    # Step 2: Build HF key -> safetensor file mapping (from model.safetensors.index.json)
    hf_key_to_file = {}
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        hf_key_to_file = index.get("weight_map", {})
    else:
        # Single file model
        sf_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if sf_files:
            from safetensors import safe_open
            for sf in sf_files:
                with safe_open(sf, framework="pt") as f:
                    for key in f.keys():
                        hf_key_to_file[key] = os.path.basename(sf)

    # Step 3: Group parameters by safetensor file for efficient loading
    file_to_params = {}  # filename -> [(fsdp_name, hf_key, is_buffer)]
    unmatched_keys = []

    for fsdp_name, param in model.named_parameters():
        hf_key = fsdp_to_hf[fsdp_name]
        sf_file = hf_key_to_file.get(hf_key)
        if sf_file:
            file_to_params.setdefault(sf_file, []).append((fsdp_name, hf_key, False))
        else:
            unmatched_keys.append((fsdp_name, hf_key))

    for buf_name, buf in model.named_buffers():
        hf_key = fsdp_to_hf[buf_name]
        sf_file = hf_key_to_file.get(hf_key)
        if sf_file:
            file_to_params.setdefault(sf_file, []).append((buf_name, hf_key, True))

    if rank == 0:
        total_params = sum(len(v) for v in file_to_params.values())
        logger.info(f"[Rank {rank}] Weight loading: {total_params} tensors across "
              f"{len(file_to_params)} safetensor files")
        if unmatched_keys:
            logger.warning(f"[Rank {rank}] {len(unmatched_keys)} params not in checkpoint: "
                  f"{[k[1] for k in unmatched_keys[:5]]}")

    # Step 4: Pre-build name→tensor lookup tables (avoid O(n²) dict rebuilds)
    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())

    # Step 5: Load each safetensor file and broadcast its parameters
    loaded_count = 0
    skipped_keys = []

    for file_idx, (sf_filename, param_list) in enumerate(sorted(file_to_params.items())):
        sf_path = os.path.join(model_path, sf_filename)

        # Rank 0 loads this safetensor file
        if rank == 0:
            from safetensors.torch import load_file
            t_load = time.time()
            shard_data = load_file(sf_path, device="cpu")
            if rank == 0:
                logger.info(f"[Rank {rank}] Loaded {sf_filename} ({len(shard_data)} tensors, "
                      f"{time.time() - t_load:.1f}s)")
        else:
            shard_data = {}

        # Broadcast each parameter from this file
        for fsdp_name, hf_key, is_buffer in param_list:
            tensor = buffer_dict[fsdp_name] if is_buffer else param_dict[fsdp_name]

            _broadcast_single_tensor(
                tensor, hf_key, shard_data, rank, world_size, gpu_device, skipped_keys,
                target_device=target_device,
            )
            loaded_count += 1

        # Free this shard's data to keep CPU memory low
        del shard_data

        if rank == 0:
            elapsed = time.time() - t0
            logger.info(f"[Rank {rank}] Progress: {file_idx + 1}/{len(file_to_params)} files, "
                  f"{loaded_count} tensors, {elapsed:.1f}s")

    elapsed = time.time() - t0
    if rank == 0:
        logger.info(f"[Rank {rank}] Manual weight loading complete: {loaded_count} tensors, "
              f"{elapsed:.1f}s")
        if skipped_keys:
            logger.warning(f"[Rank {rank}] {len(skipped_keys)} keys not in checkpoint: "
                  f"{skipped_keys[:10]}")

    dist.barrier()


def _broadcast_single_tensor(
    tensor, hf_key: str, shard_data: dict, rank: int,
    world_size: int, device: torch.device, skipped_keys: list,
    target_device: torch.device = None,
):
    """Broadcast a single parameter or buffer from rank 0 to all ranks.

    For DTensor parameters (FSDP2-sharded), broadcasts the full tensor and
    each rank extracts its local shard.

    Args:
        device: GPU device used for broadcast (NCCL requires CUDA tensors).
        target_device: Final device for parameters. If CPU (for offload),
                      the shard is moved to CPU after broadcast.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor
    if target_device is None:
        target_device = device

    if isinstance(tensor, DTensor):
        full_shape = tensor.shape
        local_tensor = tensor._local_tensor
        local_shape = local_tensor.shape

        # Always broadcast on GPU (NCCL requirement)
        if rank == 0:
            if hf_key in shard_data:
                full_weight = shard_data[hf_key].to(dtype=tensor.dtype, device=device)
                if full_weight.shape != full_shape:
                    full_weight = full_weight.reshape(full_shape)
            else:
                skipped_keys.append(hf_key)
                full_weight = torch.zeros(full_shape, dtype=tensor.dtype, device=device)
        else:
            full_weight = torch.empty(full_shape, dtype=tensor.dtype, device=device)

        dist.broadcast(full_weight, src=0)

        # Extract the local shard using DTensor's placement info for the correct
        # offset. FSDP2 Shard(0) splits along dim 0; when the dim-0 size is not
        # evenly divisible by world_size, the first (size % world_size) ranks get
        # one extra row, so `rank * local_shape[0]` computes the wrong offset.
        # compute_local_shape_and_global_offset gives the correct per-rank offset.
        from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
        _, global_offset = compute_local_shape_and_global_offset(
            full_shape, tensor.device_mesh, tensor.placements
        )
        start = global_offset[0]  # offset along dim 0
        end = start + local_shape[0]

        if end <= full_weight.shape[0]:
            my_shard = full_weight[start:end]
        else:
            # Handle edge case: shard extends beyond tensor (shouldn't happen
            # with correct offset, but keep as safety net)
            available = full_weight.shape[0] - start
            my_shard = torch.zeros(local_shape, dtype=tensor.dtype, device=device)
            if available > 0:
                my_shard[:available] = full_weight[start:start + available]

        with torch.no_grad():
            # If target is CPU (offload), move shard to CPU before copy
            if target_device.type == "cpu":
                local_tensor.copy_(my_shard.to(target_device))
            else:
                local_tensor.copy_(my_shard)

        del full_weight
    else:
        # Non-DTensor parameter or buffer
        # Always broadcast on GPU (NCCL requirement)
        if rank == 0:
            if hf_key in shard_data:
                data = shard_data[hf_key].to(dtype=tensor.dtype, device=device)
            else:
                skipped_keys.append(hf_key)
                data = torch.zeros(tensor.shape, dtype=tensor.dtype, device=device)
        else:
            data = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
        dist.broadcast(data, src=0)
        with torch.no_grad():
            if target_device.type == "cpu":
                tensor.data.copy_(data.to(target_device))
            else:
                tensor.data.copy_(data)
        del data


def _fsdp_name_to_hf_key(fsdp_name: str) -> str:
    """Convert FSDP parameter name to HuggingFace state dict key.

    Handles:
    1. Strip "model." prefix from SFTTrainingWrapper wrapping
    2. Strip "_checkpoint_wrapped_module." from activation checkpointing
    3. Map ngram_embeddings.word_embeddings -> embed_tokens (broken sharing)
    4. Map codebooks._shared -> codebooks.0 (broken sharing)
    5. Map embedder_units.N.embedder -> embedders.N (restructured ngram units)
    6. Map embedder_units.N.post_proj -> post_projs.N (restructured ngram units)

    Examples:
        "model.model.layers.0._checkpoint_wrapped_module.mlp.gate_proj.weight"
        -> "model.layers.0.mlp.gate_proj.weight"

        "model.model.ngram_embeddings.word_embeddings.weight"
        -> "model.embed_tokens.weight"

        "model.model.ngram_embeddings.embedder_units.3.embedder.weight"
        -> "model.ngram_embeddings.embedders.3.weight"
    """
    import re
    hf_key = fsdp_name
    # Strip SFTTrainingWrapper prefix
    if hf_key.startswith("model."):
        hf_key = hf_key[len("model."):]
    # Strip activation checkpointing wrapper segments
    hf_key = hf_key.replace("_checkpoint_wrapped_module.", "")
    # Map shared embedding alias
    hf_key = hf_key.replace(
        "model.ngram_embeddings.word_embeddings.",
        "model.embed_tokens.",
    )
    # Map restructured ngram embedder units back to original HF keys
    # embedder_units.N.embedder.X -> embedders.N.X
    hf_key = re.sub(
        r"ngram_embeddings\.embedder_units\.(\d+)\.embedder\.",
        r"ngram_embeddings.embedders.\1.",
        hf_key,
    )
    # embedder_units.N.post_proj.X -> post_projs.N.X
    hf_key = re.sub(
        r"ngram_embeddings\.embedder_units\.(\d+)\.post_proj\.",
        r"ngram_embeddings.post_projs.\1.",
        hf_key,
    )
    # Map shared codebook alias
    hf_key = hf_key.replace(".codebooks._shared.", ".codebooks.0.")
    return hf_key


def fix_non_persistent_buffers(model, model_path: str, device: torch.device, rank: int = 0,
                               rope_theta: float = 0.0):
    """Recompute non-persistent buffers lost during meta-device init.

    LongcatNextModel registers buffers with persistent=False:
    - visual_offset_vals, audio_offset_vals
    - img_start_token_id, img_end_token_id, etc.
    These are computed from config in __init__, lost on meta device.

    NOTE: rank 0 loads from_pretrained and already has these buffers, but
    non-rank-0 processes (meta device) don't. After FSDP2 sharding and
    set_model_state_dict, non-persistent buffers are NOT distributed. We recompute
    them on all ranks.

    Args:
        rope_theta: If > 0, override config's rope_theta for LLM rotary embedding.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Navigate to inner model
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "model") and hasattr(inner.model, "model"):
        inner_model = inner.model.model
    elif hasattr(inner, "model"):
        inner_model = inner.model
    else:
        if rank == 0:
            logger.warning(f"[Rank {rank}] Cannot find inner model for buffer fix")
        return

    # Recompute visual_offset_vals
    if hasattr(config, "visual_config") and hasattr(config.visual_config, "vq_config"):
        visual_offset_list = [config.visual_offset] + config.visual_config.vq_config.codebook_sizes[:-1]
        visual_offset_vals = torch.cumsum(torch.tensor(visual_offset_list, dtype=torch.long), dim=0)
        inner_model.visual_offset_vals = visual_offset_vals.to(device)

    # Recompute audio_offset_vals
    if hasattr(config, "audio_config") and hasattr(config.audio_config, "vq_config"):
        audio_offset_list = [config.audio_offset] + config.audio_config.vq_config.codebook_sizes[:-1]
        audio_offset_vals = torch.cumsum(torch.tensor(audio_offset_list, dtype=torch.long), dim=0)
        inner_model.audio_offset_vals = audio_offset_vals.to(device)

    # Recompute special token id buffers
    name2id_dict = {}
    if hasattr(config, "visual_config"):
        vc = config.visual_config
        for attr in ["img_start_token_id", "img_end_token_id", "imgtext_pad_token_id",
                      "imggen_end_token_id", "img_pad_token_id"]:
            if hasattr(vc, attr):
                name2id_dict[attr] = getattr(vc, attr)
    if hasattr(config, "audio_config"):
        ac = config.audio_config
        for attr in ["audio_start_token_id", "audio_end_token_id", "audiotext_pad_token_id",
                      "audiogen_end_token_id", "audio_pad_token_id"]:
            if hasattr(ac, attr):
                name2id_dict[attr] = getattr(ac, attr)

    for k, v in name2id_dict.items():
        if hasattr(inner_model, k):
            setattr(inner_model, k, torch.tensor([v], dtype=torch.long, device=device))

    # Recompute plain tensor attributes that are NOT parameters or buffers.
    # These are created from config in __init__, and stay on meta device for
    # non-rank-0 processes (since `with torch.device("meta")` affects all
    # tensor creation). They are not handled by to_empty or set_model_state_dict.

    # NgramEmbedding.oe_ignored_token_ids — used in forward for masking special tokens
    if hasattr(inner_model, "ngram_embeddings"):
        ngram = inner_model.ngram_embeddings
        if hasattr(config, "oe_ignored_token_ids"):
            ngram.oe_ignored_token_ids = torch.tensor(
                config.oe_ignored_token_ids, dtype=torch.long, device=device
            )
            if rank == 0:
                logger.info(f"  ngram_embeddings.oe_ignored_token_ids recomputed "
                      f"({len(config.oe_ignored_token_ids)} tokens)")

    # LongcatNextModel also registers non-persistent buffers from _init_multimodal_constants;
    # handle additional names from the visual/audio config
    extra_name2id = {}
    if hasattr(config, "visual_config"):
        vc = config.visual_config
        for attr in ["image_newline_token_id", "image_end_token_id", "image_pad_token_id"]:
            if hasattr(vc, attr):
                extra_name2id[attr] = getattr(vc, attr)
    if hasattr(config, "audio_config"):
        ac = config.audio_config
        for attr in ["audiotext_start_token_id", "audiotext_pad_token_id",
                      "audiogen_end_token_id", "audio_pad_token_id"]:
            if hasattr(ac, attr):
                extra_name2id[attr] = getattr(ac, attr)
    for k, v in extra_name2id.items():
        if hasattr(inner_model, k):
            setattr(inner_model, k, torch.tensor([v], dtype=torch.long, device=device))

    # Recompute ViT rotary embedding's inv_freq.
    # Qwen2_5_VisionRotaryEmbedding_Modified stores inv_freq as a plain tensor
    # attribute (not a parameter or buffer), so it stays on meta device after
    # to_empty() and load_fsdp2_weights_manual(). Must be recomputed from config.
    _fix_vit_rotary_embeddings(model, device, rank)

    # Fix the LLM's rotary embedding inv_freq buffer.
    # LongcatFlashRotaryEmbedding uses register_buffer("inv_freq", ..., persistent=False).
    # After meta init + to_empty(), the buffer becomes all-zeros on CUDA.
    # Without this fix, cos=1/sin=0 for all positions → no positional encoding → ~3x loss.
    _fix_llm_rotary_embedding(model, model_path, device, rank, rope_theta=rope_theta)

    if rank == 0:
        logger.info(f"[Rank {rank}] Non-persistent buffers recomputed on {device}")


def freeze_for_understand(model) -> None:
    """Apply freeze strategy for understanding task.

    Trainable: LLM backbone, embedding, ngram_embeddings, lm_head, visual_embedding_layer
    Frozen: ViT, visual_bridge_model, visual_head, audio_head, audio_tokenizer
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "model.layers." in name:
            param.requires_grad = True
        elif "model.embed_tokens." in name:
            param.requires_grad = True
        elif "model.ngram_embeddings." in name:
            param.requires_grad = True
        elif "model.norm." in name:
            param.requires_grad = True
        elif "lm_head." in name:
            param.requires_grad = True
        elif "visual_tokenizer.visual_embedding_layer." in name:
            param.requires_grad = True

    _print_trainable_summary(model, "understand")


def freeze_for_generate(model) -> None:
    """Apply freeze strategy for generation task.

    Trainable: LLM backbone, embedding, ngram_embeddings, lm_head,
               visual_embedding_layer, visual_head
    Frozen: ViT, visual_bridge_model, audio_head, audio_tokenizer
    """
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "model.layers." in name:
            param.requires_grad = True
        elif "model.embed_tokens." in name:
            param.requires_grad = True
        elif "model.ngram_embeddings." in name:
            param.requires_grad = True
        elif "model.norm." in name:
            param.requires_grad = True
        elif "lm_head." in name:
            param.requires_grad = True
        elif "visual_tokenizer.visual_embedding_layer." in name:
            param.requires_grad = True
        elif "visual_head." in name:
            param.requires_grad = True

    _print_trainable_summary(model, "generate")


def _print_trainable_summary(model, task: str) -> None:
    """Print summary of trainable vs frozen parameters (rank 0 only)."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(f"\n=== Freeze Summary ({task} task) ===")
    logger.info(f"Total parameters:     {total / 1e9:.2f}B")
    logger.info(f"Trainable parameters: {trainable / 1e9:.2f}B ({100 * trainable / total:.1f}%)")
    logger.info(f"Frozen parameters:    {frozen / 1e9:.2f}B ({100 * frozen / total:.1f}%)")

    logger.info("\nModule-level breakdown:")
    for name, module in model.named_children():
        total_m = sum(p.numel() for p in module.parameters())
        train_m = sum(p.numel() for p in module.parameters() if p.requires_grad)
        status = "TRAIN" if train_m > 0 else "FROZEN"
        if total_m > 0:
            logger.info(f"  {name:40s} {total_m / 1e6:10.1f}M  [{status}] "
                  f"({100 * train_m / total_m:.0f}% trainable)")
    logger.info("=" * 50)


def _patch_vit_attn_forward(model, rank: int = 0):
    """Patch ViT Qwen2_5_VLVisionAttention.forward to use FlashAttention.

    transformers silently falls back to the eager Qwen2_5_VLVisionAttention even
    when config._attn_implementation='flash_attention_2', because the transformers
    version used here doesn't ship Qwen2_5_VLVisionFlashAttention2 as an importable
    class. We rewrite the attention forward to use the flash path directly:
      rotary:    flash_attn.layers.rotary.apply_rotary_emb
      attention: flash_attn_varlen_func
    q/k are cast to fp32 for the rotary application (cos/sin are already fp32 from
    emb.cos()/emb.sin(), so no extra cast is needed there).

    Patching the CLASS method (not the instance) makes this apply to every block
    in every ViT in this process — call it once per process.
    """
    try:
        vit = model.model.visual_tokenizer.visual_model
        attn_cls = type(vit.blocks[0].attn)
        # Idempotency: don't double-patch.
        if getattr(attn_cls, "_fa_forward_patched", False):
            return
        from flash_attn import flash_attn_varlen_func as _fa_varlen
        from flash_attn.layers.rotary import apply_rotary_emb as _fa_rotary_emb

        def _vit_attn_forward(self, hidden_states, cu_seqlens=None,
                              rotary_pos_emb=None, position_embeddings=None,
                              **kwargs):
            seq_length = hidden_states.shape[0]
            qkv = self.qkv(hidden_states).reshape(
                seq_length, 3, self.num_heads, -1
            ).permute(1, 0, 2, 3)
            q, k, v = qkv.unbind(0)

            if position_embeddings is None:
                if rotary_pos_emb is None:
                    raise ValueError("Need rotary_pos_emb or position_embeddings")
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            else:
                cos, sin = position_embeddings

            half = cos.shape[-1] // 2
            cos_h = cos[..., :half]
            sin_h = sin[..., :half]
            q4 = q.unsqueeze(0).float()
            k4 = k.unsqueeze(0).float()
            q = _fa_rotary_emb(q4, cos_h, sin_h, interleaved=False,
                               inplace=False).squeeze(0).type_as(hidden_states)
            k = _fa_rotary_emb(k4, cos_h, sin_h, interleaved=False,
                               inplace=False).squeeze(0).type_as(hidden_states)

            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            out = _fa_varlen(
                q, k, v, cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                dropout_p=0.0, softmax_scale=None, causal=False,
            ).reshape(seq_length, -1)
            return self.proj(out)

        attn_cls.forward = _vit_attn_forward
        attn_cls._fa_forward_patched = True
    except Exception as _e:
        if rank == 0:
            logger.info(f"Could not patch ViT attn forward: {_e}")


def _fix_vit_rotary_embeddings(model, device: torch.device, rank: int = 0):
    """Fix ViT rotary embedding inv_freq that stays on meta device.

    Qwen2_5_VisionRotaryEmbedding_Modified stores inv_freq as a plain tensor
    attribute (not a parameter or buffer). When the model is created on meta
    device, inv_freq becomes a meta tensor. Since it's not a parameter or buffer,
    to_empty() and load_fsdp2_weights_manual() don't touch it.

    We find all such modules and recompute inv_freq on the target device.
    """
    fixed_count = 0
    for name, module in model.named_modules():
        # Match by class name to avoid importing the model-specific class
        cls_name = type(module).__name__
        if "RotaryEmbedding" in cls_name and hasattr(module, "inv_freq"):
            inv_freq = module.inv_freq
            # ALWAYS recompute: inv_freq is a plain tensor attribute, not in state_dict.
            # After meta init or to_empty(), it contains garbage/meta/zeros.
            needs_fix = True
            if needs_fix:
                # Recompute inv_freq from the module's config.
                # Compute inv_freq on CPU (not CUDA): fp32 pow differs by ~1 ULP
                # between CPU libm and CUDA math kernels, and those sub-ULP diffs
                # propagate through the 32 ViT blocks and amplify into ~35%
                # visual_ids drift. The original also computes on CPU then moves
                # to device, so this keeps the result identical.
                dim = inv_freq.shape[0] * 2  # inv_freq has shape [dim/2]
                # theta source priority: module.theta → module.base → 10000.0
                # (LongCat-Next passes only `dim`, so it uses the 10000.0 default).
                if hasattr(module, "theta") and isinstance(getattr(module, "theta"), (int, float)):
                    theta = float(module.theta)
                    theta_src = "module.theta"
                elif hasattr(module, "base") and isinstance(getattr(module, "base"), (int, float)):
                    theta = float(module.base)
                    theta_src = "module.base"
                else:
                    theta = 10000.0
                    theta_src = "default=10000.0"
                new_inv_freq = 1.0 / (
                    theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
                )
                new_inv_freq = new_inv_freq.to(device)
                module.inv_freq = new_inv_freq
                fixed_count += 1

    if rank == 0 and fixed_count > 0:
        logger.info(f"  Fixed {fixed_count} ViT rotary embedding(s)")


def _fix_llm_rotary_embedding(model, model_path: str, device: torch.device, rank: int = 0,
                              rope_theta: float = 0.0):
    """Fix the LLM's rotary embedding inv_freq buffer.

    LongcatFlashRotaryEmbedding registers inv_freq as a non-persistent buffer
    via register_buffer("inv_freq", ..., persistent=False). When the model is
    created on meta device and then materialized via to_empty(device=cuda),
    the buffer becomes a zero tensor on CUDA (not meta). Since:
    1. load_fsdp2_weights_manual only loads parameters, not buffers
    2. inv_freq is non-persistent so it's not in state_dict
    3. _fix_vit_rotary_embeddings only checks for meta tensors (is_meta)

    The buffer stays as all-zeros, which makes cos=1, sin=0 for all positions,
    completely destroying positional information and causing ~3x loss increase.

    Fix: Recompute inv_freq using the model's rope config.

    Args:
        rope_theta: If > 0, override the config's rope_theta (e.g., 1e6 for generate).
    """
    # Navigate to inner model
    inner = model
    if hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "model") and hasattr(inner.model, "model"):
        inner_model = inner.model.model
    elif hasattr(inner, "model"):
        inner_model = inner.model
    else:
        if rank == 0:
            logger.warning(f"[Rank {rank}] Cannot find inner model for LLM rotary fix")
        return

    if not hasattr(inner_model, "rotary_emb"):
        return

    rotary = inner_model.rotary_emb
    if not hasattr(rotary, "inv_freq"):
        return

    inv_freq = rotary.inv_freq

    # ALWAYS recompute inv_freq. It is a non-persistent buffer — never in state_dict,
    # never loaded by weight loading. After to_empty() it contains garbage (could be
    # zeros on CUDA, NaN/garbage on CPU, or meta). No point trying to validate it.
    if rank == 0:
        logger.info(f"  LLM rotary_emb.inv_freq: recomputing (was device={inv_freq.device})")

    # Recompute using the ROPE_INIT_FUNCTIONS from the model's config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Override rope_theta if requested (e.g. 1e6 for the generation task)
    if rope_theta > 0:
        original_rope_theta = getattr(config, "rope_theta", None)
        config.rope_theta = rope_theta
        if rank == 0:
            logger.info(f"  LLM rotary fix: rope_theta overridden {original_rope_theta} -> {rope_theta}")

    # Use the same initialization as LongcatFlashRotaryEmbedding.__init__
    from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashRotaryEmbedding
    ref_rotary = LongcatFlashRotaryEmbedding(config, device=str(device))
    new_inv_freq = ref_rotary.inv_freq

    # Replace the buffer
    rotary.inv_freq = new_inv_freq
    # Also update original_inv_freq if it exists
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = new_inv_freq
    # Copy attention_scaling if it was computed during init
    if hasattr(ref_rotary, "attention_scaling"):
        rotary.attention_scaling = ref_rotary.attention_scaling

    if rank == 0:
        logger.info(f"  Fixed LLM rotary_emb.inv_freq: shape={list(new_inv_freq.shape)}, "
              f"device={new_inv_freq.device}, first3={new_inv_freq[:3].tolist()}")
