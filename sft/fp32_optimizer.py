# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""FP32 AdamW optimizer wrapper for FSDP2 bf16 parameters.

Problem:
    FSDP2 with MixedPrecisionPolicy(param_dtype=bf16) stores parameters in bf16.
    PyTorch AdamW creates optimizer states (exp_avg, exp_avg_sq) in the same dtype
    as the parameter → bf16. bf16 optimizer states lose precision in the Adam
    moment accumulation, which drifts the loss by ~0.12 over thousands of steps.
    Keeping fp32 master copies and running Adam in fp32 avoids this.

Solution:
    Create fp32 master copies of all bf16 parameters,
    run AdamW on them (fp32 states), and copy back to bf16 after each step.

    The fp32 master copies and optimizer states live on CPU to avoid GPU OOM with
    large models (73B params + seq_length=8192 activations). The optimizer step
    runs on CPU which is slower but allows training with 4 nodes (32 GPUs) instead
    of requiring 8 nodes.
"""

import logging

import torch
import torch.distributed as dist
from torch.optim import AdamW


logger = logging.getLogger(__name__)

class FP32AdamW:
    """AdamW that runs in fp32 on CPU even when model parameters are bf16 on GPU.

    fp32 master-weight optimizer:
    1. Maintains fp32 master copies on CPU (saves ~18GB GPU memory per rank)
    2. Copies bf16 gradients → fp32 CPU before optimizer step
    3. Runs AdamW on CPU fp32 master copies
    4. Copies fp32 CPU master → bf16 GPU back to model after step

    For FSDP2 DTensor parameters, we extract the local tensor shard to avoid
    DTensor serialization complexities on CPU.
    """

    def __init__(self, param_groups, lr=1e-5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.1):
        self._bf16_params = []      # Original bf16 model params (GPU DTensor)
        self._fp32_masters = []     # fp32 master copies (CPU plain tensor)
        fp32_param_groups = []

        for group in param_groups:
            fp32_params_in_group = []
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                if p.dtype in (torch.bfloat16, torch.float16):
                    # Extract local shard from DTensor (if applicable) and move to CPU fp32.
                    local_data = p.data
                    if hasattr(local_data, '_local_tensor'):
                        local_data = local_data._local_tensor
                    fp32_cpu = local_data.detach().float().cpu()
                    fp32_p = torch.nn.Parameter(fp32_cpu, requires_grad=True)
                    self._bf16_params.append(p)
                    self._fp32_masters.append(fp32_p)
                    fp32_params_in_group.append(fp32_p)
                else:
                    fp32_params_in_group.append(p)

            fp32_group = {k: v for k, v in group.items() if k != "params"}
            fp32_group["params"] = fp32_params_in_group
            fp32_param_groups.append(fp32_group)

        self._inner_optimizer = AdamW(
            fp32_param_groups, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )

        if dist.is_initialized() and dist.get_rank() == 0:
            n_fp32 = len(self._fp32_masters)
            mem_mb = sum(p.numel() * 4 for p in self._fp32_masters) / 1e6
            logger.info(f"[FP32AdamW] Created {n_fp32} fp32 master params on CPU, "
                  f"CPU memory: {mem_mb:.1f} MB per rank (0 GPU overhead)")
            if n_fp32 == 0:
                logger.warning("[FP32AdamW] WARNING: No bf16 params found! "
                      "All params may already be fp32. FP32AdamW is a no-op.")

    def zero_grad(self, set_to_none=True):
        """Zero gradients on both bf16 model params and fp32 masters."""
        self._inner_optimizer.zero_grad(set_to_none=set_to_none)
        for p in self._bf16_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self):
        """Copy grads bf16 GPU → fp32 CPU, run Adam on CPU, copy fp32 CPU → bf16 GPU."""
        # 1) Copy gradients from bf16 GPU model params to fp32 CPU masters
        for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_masters):
            if bf16_p.grad is not None:
                grad = bf16_p.grad
                if hasattr(grad, '_local_tensor'):
                    grad = grad._local_tensor
                fp32_p.grad = grad.detach().float().cpu()
            else:
                fp32_p.grad = None

        # 2) Optimizer step on CPU fp32 masters
        self._inner_optimizer.step()

        # 3) Copy updated fp32 CPU masters back to bf16 GPU model params
        with torch.no_grad():
            for bf16_p, fp32_p in zip(self._bf16_params, self._fp32_masters):
                # Get the local shard to copy into
                local_data = bf16_p.data
                if hasattr(local_data, '_local_tensor'):
                    local_data = local_data._local_tensor
                local_data.copy_(fp32_p.data.to(device=local_data.device))

    @property
    def param_groups(self):
        return self._inner_optimizer.param_groups

    def state_dict(self):
        return self._inner_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._inner_optimizer.load_state_dict(state_dict)
