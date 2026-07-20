# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Training utilities for LongCat-Next FSDP SFT.

Provides optimizer/scheduler creation, stdout logging, and TensorBoard logging.
TensorBoard panel layout:

Panel structure:
  Primary loss:
    - lm-loss-training/lm loss                   (main backward loss)
    - training-loss/loss                          (same value, alternate panel)
    - multi-dimension-loss/training-loss vs steps/samples/tokens

  Multimodal loss (via _LOSS_MAPPING_DICT):
    - training-loss/image_token_loss              (per-token image CE avg)
    - training-loss/o_image_token_loss
    - training-loss/text_loss
    - training-loss/o_text_loss
    - training-data/image_token_token_ratio
    - training-data/image_token_weight_ratio
    - training-data/text_token_ratio
    - training-data/text_weight_ratio
    - training-data/efficency_token_num           (total trainable tokens, global)
    - training-loss/paraDec_level-{N}_loss        (per-level depth CE, N=1..8)

  Standard metrics:
    - learning-rate/learning-rate
    - grad-norm/grad-norm
    - params-norm/params-norm
    - iteration-time/iteration-time
    - iteration-time/samples per second
    - iteration-time/tokens per second
    - batch-size/batch-size
"""

import math
import os
import time
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from fp32_optimizer import FP32AdamW

from config import TrainConfig

logger = logging.getLogger(__name__)


def _is_emb_lr_scaled(name: str, param: torch.Tensor, ln_scale_lr: bool = False) -> bool:
    """Check if parameter should use the scaled LR (emb-lr-scale).

    - 2D params with 'embedding' in the name → scaled
    - With ln_scale_lr: 1D params (norm/bias) → scaled
    - 2D params without 'embedding' (attention, FFN, MoE experts) → NOT scaled

    NOTE: the visual bridge MLP module here is named
    `visual_embedding_layer.pre_buffer.mlp`, which contains the substring
    'embedding' but is a projection MLP, not an embedding table. It must use the
    base LR, so it is excluded explicitly.
    """
    if len(param.shape) == 2:
        if 'pre_buffer.mlp' in name:
            return False  # visual bridge MLP → base LR
        if 'embedding' in name:
            return True
        return False  # attention, FFN, MoE expert weights → base LR
    if ln_scale_lr and len(param.shape) == 1:
        # 1D params are norm weights and biases
        return True
    return False


def create_optimizer(model, config: TrainConfig) -> AdamW:
    """Create AdamW optimizer with proper weight decay and LR scaling.

    Weight decay is NOT applied to bias, LayerNorm/RMSNorm weights.

    When emb_lr_scale_base > 0:
    - Embedding params: lr × (hidden_size / emb_lr_scale_base)
    - Norm/bias params (if ln_scale_lr): lr × (hidden_size / emb_lr_scale_base)
    - Other 2D params (attention, FFN, MoE): lr × 1.0
    """
    # Determine LR multiplier
    hidden_size = 3072  # LongCat-Next hidden_size
    emb_lr_scale_base = getattr(config, 'emb_lr_scale_base', 0)
    ln_scale_lr = getattr(config, 'ln_scale_lr', False)
    lr_mult = hidden_size / emb_lr_scale_base if emb_lr_scale_base > 0 else 1.0

    if emb_lr_scale_base > 0:
        # 4 groups: (decay + scaled), (decay + base), (no_decay + scaled), (no_decay + base)
        decay_scaled = []       # embedding weights with weight decay + scaled LR
        decay_base = []         # attention/FFN weights with weight decay + base LR
        no_decay_scaled = []    # norm/bias with no weight decay + scaled LR
        no_decay_base = []      # other no-decay params + base LR

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_scaled = _is_emb_lr_scaled(name, param, ln_scale_lr)
            is_no_decay = "bias" in name or "norm" in name or "layernorm" in name

            if is_no_decay:
                if is_scaled:
                    no_decay_scaled.append(param)
                else:
                    no_decay_base.append(param)
            else:
                if is_scaled:
                    decay_scaled.append(param)
                else:
                    decay_base.append(param)

        param_groups = []
        if decay_scaled:
            param_groups.append({
                "params": decay_scaled,
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate * lr_mult,
                "name": "decay_scaled",
            })
        if decay_base:
            param_groups.append({
                "params": decay_base,
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate,
                "name": "decay_base",
            })
        if no_decay_scaled:
            param_groups.append({
                "params": no_decay_scaled,
                "weight_decay": 0.0,
                "lr": config.learning_rate * lr_mult,
                "name": "no_decay_scaled",
            })
        if no_decay_base:
            param_groups.append({
                "params": no_decay_base,
                "weight_decay": 0.0,
                "lr": config.learning_rate,
                "name": "no_decay_base",
            })

        # Log parameter group stats
        if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
            for pg in param_groups:
                n_params = sum(p.numel() for p in pg["params"])
                logger.info(f"  [Optimizer] {pg.get('name', '?')}: {len(pg['params'])} params, "
                      f"{n_params:,} elements, lr={pg['lr']:.6f}, wd={pg['weight_decay']}")
            logger.info(f"  [Optimizer] emb_lr_scale: base={emb_lr_scale_base}, "
                  f"lr_mult={lr_mult:.4f}, ln_scale_lr={ln_scale_lr}")
    else:
        # No LR scaling — standard 2 groups
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    # Use FP32AdamW: keep fp32 master copies of the bf16 parameters and run
    # Adam in fp32.
    # Without this, FSDP2 bf16 params create bf16 optimizer states, causing ~0.12
    # loss drift over many training steps.
    #
    # Set USE_BF16_OPTIMIZER=1 env var to fall back to standard AdamW (saves memory).
    import os
    if os.environ.get("USE_BF16_OPTIMIZER", "0") == "1":
        optimizer = AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info("[Optimizer] Using standard AdamW (bf16 states)")
    else:
        optimizer = FP32AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )
    return optimizer


def create_scheduler(optimizer, config: TrainConfig, total_steps: int) -> LambdaLR:
    """Create learning rate scheduler.

    Supports:
    - "cosine": Cosine decay with optional warmup
    - "constant": Constant LR with optional warmup
    """
    # LambdaLR requires torch.optim.Optimizer. FP32AdamW wraps an inner AdamW.
    sched_optimizer = getattr(optimizer, '_inner_optimizer', optimizer)
    warmup_steps = config.warmup_steps

    if config.lr_schedule == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            min_ratio = config.min_learning_rate / config.learning_rate
            return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    elif config.lr_schedule == "constant":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return 1.0

    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    return LambdaLR(sched_optimizer, lr_lambda)


def compute_params_norm(model) -> float:
    """Compute L2 norm of all trainable parameters.

    Parameter L2 norm metric.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total_norm_sq += p.data.float().norm().item() ** 2
    return total_norm_sq ** 0.5


# ============================================================================
# TensorBoard Logger
# ============================================================================

class TensorBoardLogger:
    """TensorBoard logger with a fixed panel structure.

    Only rank 0 writes to TensorBoard. Other ranks hold a no-op instance.
    Tracks consumed samples and tokens for x-axis alternatives.

    Metrics are written at TWO locations:

    Location 1 — mapped panels:
      Uses _LOSS_MAPPING_DICT to write metrics with the "training-" prefix.
      Tags like: training-loss/image_token_loss, training-data/text_token_ratio,
      training-loss/paraDec_level-N_loss.  X-axis: iteration only.

    Location 2 — per-key panels:
      Iterates over loss_dict keys and writes each with a panel title.
      Tags like: lm-loss-training/lm loss, loss-text_loss/text_loss,
      loss-paraDec_level-1_loss/paraDec_level-1_loss.
      X-axis: iteration, samples, tokens.

    Both locations are needed for full TensorBoard panel parity.
    """

    # Loss-name -> panel mapping.
    # Maps metric_key -> "loss/xxx" or "data/xxx", prefixed with "training-".
    _GENERATE_LOSS_MAPPING = {
        # Loss metrics → training-loss/xxx
        "image_token_loss": "loss/image_token_loss",
        "o_image_token_loss": "loss/o_image_token_loss",
        "text_loss": "loss/text_loss",
        "o_text_loss": "loss/o_text_loss",
        # Data metrics → training-data/xxx
        "total_token_num": "data/efficency_token_num",
        "image_token_token_ratio": "data/image_token_token_ratio",
        "image_token_weight_ratio": "data/image_token_weight_ratio",
        "text_token_ratio": "data/text_token_ratio",
        "text_weight_ratio": "data/text_weight_ratio",
        # Per-level depth losses → training-loss/paraDec_level-N_loss
        **{f"paraDec_level-{i}_loss": f"loss/paraDec_level-{i}_loss"
           for i in range(1, 9)},
        # MoE auxiliary losses
        "expert load balance loss": "loss/expert load balance loss",
        "z loss": "loss/z loss",
    }

    _UNDERSTAND_LOSS_MAPPING = {
        # Unified loss metrics for understand task
        "text_loss": "loss/text_loss",
        "o_text_loss": "loss/o_text_loss",
        "total_token_num": "data/efficency_token_num",
        "text_token_ratio": "data/text_token_ratio",
        "text_weight_ratio": "data/text_weight_ratio",
        # MoE auxiliary losses
        "expert load balance loss": "loss/expert load balance loss",
        "z loss": "loss/z loss",
    }

    # Panel title assignment.
    # Keys in this set get special panel titles; everything else → "loss-{key}".
    _SPECIAL_PANEL_KEYS = {
        "lm loss": "lm-loss-training",
        "total loss": "total-loss-training",
        "z loss": "loss-z loss",
        "expert load balance loss": "loss-expert load balance loss",
        "hidden_z_loss": "loss-hidden_z_loss",
        "router_z_loss": "loss-router_z_loss",
        "load_balance_loss": "loss-load_balance_loss",
    }

    def __init__(self, log_dir: str, rank: int = 0, enabled: bool = True,
                 global_batch_size: int = 0):
        self.rank = rank
        self.enabled = enabled and rank == 0 and bool(log_dir)
        self.writer = None
        self.consumed_samples = 0
        self.consumed_tokens = 0
        self.global_batch_size = global_batch_size

        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir, flush_secs=30)
                logger.info(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed, disabling TB logging.")
                self.enabled = False

    def log_training_step(
        self,
        step: int,
        lm_loss: float,
        metrics: Dict[str, float],
        lr: float,
        grad_norm: float,
        params_norm: float,
        step_time: float,
        tokens_in_batch: int,
        samples_in_batch: int,
        world_size: int,
        task: str,
        epoch: float = 0.0,
    ):
        """Log one optimizer step to TensorBoard.

        Writes to both TB locations:
        1. mapped panels: training-loss/xxx, training-data/xxx
        2. per-key panels: loss-{key}/{key} with 3 x-axes
        """
        if not self.enabled:
            return

        w = self.writer
        self.consumed_samples += samples_in_batch * world_size
        self.consumed_tokens += tokens_in_batch * world_size
        cs = self.consumed_samples
        ct = self.consumed_tokens

        # ================================================================
        # Location 1: primary loss panels
        # ================================================================

        # 1a. Main loss panel: training-loss/loss
        w.add_scalar("training-loss/loss", lm_loss, step)

        # 1b. Multi-dimension loss
        w.add_scalar("multi-dimension-loss/training-loss vs steps", lm_loss, step)
        w.add_scalar("multi-dimension-loss/training-loss vs samples", lm_loss, cs)
        w.add_scalar("multi-dimension-loss/training-loss vs tokens", lm_loss, ct)

        # 1c. Per-metric panels from _LOSS_MAPPING_DICT
        if metrics:
            # unify uses the generate mapping (it reports both text and image metrics).
            mapping = (self._GENERATE_LOSS_MAPPING if task in ("generate", "unify")
                       else self._UNDERSTAND_LOSS_MAPPING)
            for metric_key, tb_suffix in mapping.items():
                if metric_key in metrics:
                    value = metrics[metric_key]
                    if isinstance(value, (int, float)):
                        w.add_scalar(f"training-{tb_suffix}", float(value), step)

        # ================================================================
        # Location 2: per-key loss panels
        # Write each loss_dict key to its own panel, with 3 x-axes
        # (iteration, samples, tokens).
        # ================================================================

        # 2a. "lm loss" — always present
        self._write_loss_key(w, "lm loss", lm_loss, step, cs, ct)

        # 2b. All metric keys — written with panel "loss-{key}/{key}"
        if metrics:
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                # Skip internal keys not meant for logging
                if key in ("total_loss", "image_token_num", "text_token_num",
                           "sample_count"):
                    continue
                panel = self._SPECIAL_PANEL_KEYS.get(key, f"loss-{key}")
                w.add_scalar(f"{panel}/{key}", float(value), step)
                w.add_scalar(f"{panel}/{key} vs samples", float(value), cs)
                w.add_scalar(f"{panel}/{key} vs tokens", float(value), ct)

        # ================================================================
        # Standard training metrics
        # ================================================================

        # Epoch
        w.add_scalar("training-data/epoch", epoch, step)

        # Learning rate
        w.add_scalar("learning-rate/learning-rate", lr, step)
        w.add_scalar("learning-rate/learning-rate vs samples", lr, cs)
        w.add_scalar("learning-rate/learning-rate vs tokens", lr, ct)

        # Gradient norm
        w.add_scalar("grad-norm/grad-norm", grad_norm, step)
        w.add_scalar("grad-norm/grad-norm vs samples", grad_norm, cs)
        w.add_scalar("grad-norm/grad-norm vs tokens", grad_norm, ct)

        # Parameter norm
        if params_norm > 0:
            w.add_scalar("params-norm/params-norm", params_norm, step)
            w.add_scalar("params-norm/params-norm vs samples", params_norm, cs)
            w.add_scalar("params-norm/params-norm vs tokens", params_norm, ct)

        # Throughput
        if step_time > 0:
            tps = tokens_in_batch * world_size / max(step_time, 1e-6)
            tps_per_gpu = tps / max(world_size, 1)
            sps = samples_in_batch * world_size / max(step_time, 1e-6)
            sps_per_gpu = sps / max(world_size, 1)

            w.add_scalar("iteration-time/iteration-time", step_time, step)
            w.add_scalar("iteration-time/iteration-time vs samples", step_time, cs)
            w.add_scalar("iteration-time/iteration-time vs tokens", step_time, ct)
            w.add_scalar("iteration-time/samples per second", sps, step)
            w.add_scalar("iteration-time/samples per second per replica", sps_per_gpu, step)
            w.add_scalar("iteration-time/tokens per second", tps, step)
            w.add_scalar("iteration-time/tokens per second per replica", tps_per_gpu, step)

        # Batch size — fixed global_batch_size, not per-step sample count
        w.add_scalar("batch-size/batch-size", self.global_batch_size, step)
        w.add_scalar("batch-size/batch-size vs samples", self.global_batch_size, cs)

        # World size
        w.add_scalar("world-size", world_size, step)
        w.add_scalar("world-size vs samples", world_size, cs)

        # GPU memory (in GB for readability)
        device = torch.cuda.current_device()
        w.add_scalar("memory/allocated-gb",
                      torch.cuda.memory_allocated(device) / 1e9, step)
        w.add_scalar("memory/reserved-gb",
                      torch.cuda.memory_reserved(device) / 1e9, step)

    @staticmethod
    def _write_loss_key(w, key: str, value: float, step: int,
                        consumed_samples: int, consumed_tokens: int):
        """Write a loss_dict key using the panel title convention."""
        if key == "lm loss":
            panel = "lm-loss-training"
        elif key == "total loss":
            panel = "total-loss-training"
        elif key in ("z loss", "expert load balance loss", "hidden_z_loss",
                      "output_z_loss", "gen loss"):
            panel = f"loss-{key}"
        else:
            panel = f"loss-{key}"
        w.add_scalar(f"{panel}/{key}", value, step)
        w.add_scalar(f"{panel}/{key} vs samples", value, consumed_samples)
        w.add_scalar(f"{panel}/{key} vs tokens", value, consumed_tokens)

    def flush(self):
        """Flush pending writes to disk."""
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        if self.enabled and self.writer is not None:
            self.writer.close()


# ============================================================================
# Stdout Logger
# ============================================================================

class TrainingLogger:
    """Training logger that prints to stdout and writes to TensorBoard.

    Stdout format:
      iteration N/M | lm loss: X | ... | consumed samples: Y | elapsed ...
    """

    def __init__(
        self,
        rank: int = 0,
        log_interval: int = 1,
        tb_logger: Optional[TensorBoardLogger] = None,
        global_batch_size: int = 0,
        total_samples_per_epoch: int = 0,
    ):
        self.rank = rank
        self.log_interval = log_interval
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.total_tokens = 0
        self.consumed_samples = 0
        self.tb_logger = tb_logger
        self.global_batch_size = global_batch_size
        self.total_samples_per_epoch = total_samples_per_epoch
        # Counters for cumulative metrics
        self.nan_iterations = 0
        self.skipped_iterations = 0

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        metrics: Optional[Dict] = None,
        tokens_in_batch: int = 0,
        samples_in_batch: int = 0,
        params_norm: float = 0.0,
        world_size: int = 1,
        task: str = "understand",
        total_steps: int = 0,
        data_iter_idx: int = 0,
    ):
        """Log a training step (rank 0 only).

        Stdout format uses a fixed field order.
        """
        if self.rank != 0:
            return

        step_time = time.time() - self.step_start_time
        self.step_start_time = time.time()
        self.total_tokens += tokens_in_batch * world_size
        self.consumed_samples += samples_in_batch * world_size

        # Track NaN iterations
        if math.isnan(loss) or math.isinf(loss):
            self.nan_iterations += 1

        # Compute epoch from data-iterator progress. data_iter_idx counts every
        # line read across all files (including skipped/failed samples and lines
        # for other ranks), so it tracks dataset traversal more accurately than
        # consumed_samples, which only counts successfully processed documents.
        epoch = 0.0
        if data_iter_idx > 0 and self.total_samples_per_epoch > 0:
            epoch = data_iter_idx / self.total_samples_per_epoch
        elif self.total_samples_per_epoch > 0:
            # Fallback for datasets that don't report data_iter_idx
            epoch = self.consumed_samples / self.total_samples_per_epoch

        # ---- TensorBoard ----
        if self.tb_logger is not None and metrics:
            self.tb_logger.log_training_step(
                step=step,
                lm_loss=loss,
                metrics=metrics,
                lr=lr,
                grad_norm=grad_norm,
                params_norm=params_norm,
                step_time=step_time,
                tokens_in_batch=tokens_in_batch,
                samples_in_batch=samples_in_batch,
                world_size=world_size,
                task=task,
                epoch=epoch,
            )

        # ---- Stdout ----
        if step % self.log_interval != 0:
            return

        tokens_per_sec = tokens_in_batch * world_size / max(step_time, 1e-6)
        samples_per_sec = samples_in_batch * world_size / max(step_time, 1e-6)

        # Build the stdout line in a fixed field order:
        #   iteration N/M | {loss metrics} | epoch X.XX | consumed samples: N |
        #   elapsed time per iteration (ms): X | tokens per second: X |
        #   learning rate: X | global batch size: N | loss scale: 1.0 |
        #   grad norm: X | num zeros: 0.0 | number of skipped iterations: 0 |
        #   number of nan iterations: 0 | samples per second: X |
        #   tokens per second: X |

        if total_steps > 0:
            parts = [f" iteration {step:8d}/{total_steps}"]
        else:
            parts = [f" iteration {step:8d}"]

        if metrics and task in ("generate", "unify"):
            # Field order: text_loss, o_text_loss, text_token_ratio, text_weight_ratio,
            # image_token_loss, o_image_token_loss, image_token_token_ratio,
            # image_token_weight_ratio, total_token_num, sample_count, paraDec levels, lm loss.
            # unify mixes both modalities, so it prints text AND image metrics plus
            # sample_count; image-only fields are simply absent when a pack has no
            # generation images.
            _fmt = lambda k, m=metrics: f"{k}: {m[k]:.3E}" if k in m else None
            for key in [
                "text_loss", "o_text_loss",
                "text_token_ratio", "text_weight_ratio",
                "image_token_loss", "o_image_token_loss",
                "image_token_token_ratio", "image_token_weight_ratio",
                "total_token_num",
            ]:
                s = _fmt(key)
                if s:
                    parts.append(s)
            # sample_count (integer) — reported for unify (and any task that provides it)
            if "sample_count" in metrics:
                parts.append(f"sample_count: {metrics['sample_count']}")
            # Per-level depth losses
            for i in range(1, 9):
                k = f"paraDec_level-{i}_loss"
                if k in metrics:
                    parts.append(f"{k}: {metrics[k]:.3f}")
            # lm loss
            parts.append(f"lm loss: {loss:.3E}")
            # MoE auxiliary losses
            if "expert load balance loss" in metrics:
                parts.append(f"expert load balance loss: {metrics['expert load balance loss']:.3E}")
            if "z loss" in metrics:
                parts.append(f"z loss: {metrics['z loss']:.3E}")

        elif metrics and task == "understand":
            # Unified loss metrics: text_loss, total_token_num, sample_count
            for key in ["text_loss", "total_token_num", "sample_count"]:
                if key in metrics:
                    v = metrics[key]
                    if isinstance(v, float):
                        parts.append(f"{key}: {v:.6E}")
                    else:
                        parts.append(f"{key}: {v}")
            parts.append(f"lm loss: {loss:.6E}")
            # MoE auxiliary losses
            if "expert load balance loss" in metrics:
                parts.append(f"expert load balance loss: {metrics['expert load balance loss']:.3E}")
            if "z loss" in metrics:
                parts.append(f"z loss: {metrics['z loss']:.3E}")

        else:
            parts.append(f"lm loss: {loss:.6E}")

        # Epoch, consumed samples
        parts.append(f"epoch {epoch:.2f}")
        parts.append(f"consumed samples: {self.consumed_samples:12d}")
        # Timing and throughput
        parts.append(f"elapsed time per iteration (ms): {step_time * 1000.0:.1f}")
        parts.append(f"tokens per second: {tokens_per_sec:.3f}")
        # Learning rate
        parts.append(f"learning rate: {lr:.3E}")
        # Global batch size and loss scale
        parts.append(f"global batch size: {self.global_batch_size}")
        parts.append("loss scale: 1.0")
        # Gradient and parameter norms
        parts.append(f"grad norm: {grad_norm:.3f}")
        parts.append("num zeros: 0.0")
        # Skipped and NaN iteration counts
        parts.append(f"number of skipped iterations: {self.skipped_iterations}")
        parts.append(f"number of nan iterations: {self.nan_iterations}")
        # Throughput
        parts.append(f"samples per second: {samples_per_sec:.3f}")
        parts.append(f"tokens per second: {tokens_per_sec:.3f}")

        log_string = " | ".join(parts)
        # Prefix with a timestamp (date + ms) so multi-day runs stay readable.
        _now = time.time()
        _ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_now))
        _ms = int((_now % 1) * 1000)
        print(f"[{_ts}.{_ms:03d}] {log_string}", flush=True)

    def log_message(self, message: str):
        """Log a message (rank 0 only)."""
        if self.rank == 0:
            print(f"[INFO] {message}", flush=True)

    def close(self):
        """Close underlying TensorBoard writer."""
        if self.tb_logger is not None:
            self.tb_logger.flush()
            self.tb_logger.close()
