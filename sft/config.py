# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""SFT training configuration for LongCat-Next FSDP training."""

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Training configuration for FSDP SFT."""

    # Task type
    task: str = "understand"  # "understand", "generate", or "unify"

    # Model
    model_path: str = ""
    trust_remote_code: bool = True

    # Data
    data_path: str = ""  # path to JSONL data file(s), comma-separated for multiple
    seq_length: int = 8192
    max_samples_per_pack: int = 64  # maximum number of samples packed into one sequence
    no_packing: bool = False  # if True, each sample occupies its own sequence (no packing)
    merged_data_dir: str = ""  # directory for pre-merged+shuffled data files (empty = no merge, use data_path as-is)
    skip_merge_check: bool = False  # if True and merged file exists, reuse it WITHOUT counting input lines (skips reading full data)

    # Training hyperparameters
    global_batch_size: int = 64
    micro_batch_size: int = 1
    learning_rate: float = 1e-5
    min_learning_rate: float = 0.0
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-16
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    lr_schedule: str = "constant"  # "cosine" or "constant"
    num_epochs: int = 1
    seed: int = 42

    # FSDP2
    # Activation checkpointing reduces memory by recomputing activations during
    # backward. Uses forward-wrapping approach (not composable AC) for FSDP2
    # compatibility. MoE branching is patched for deterministic tensor counts.
    activation_checkpointing: bool = True
    offload_to_cpu: bool = False  # FSDP2 CPU offload: offload parameters to CPU when not in use

    # Training control
    max_steps: int = 0  # 0 = train full epochs; >0 = stop after this many optimizer steps
    skip_steps: int = 0  # skip this many optimizer steps at start (consume data without forward/backward)
    resume_from: str = ""  # path to checkpoint dir (e.g. /path/checkpoints/step_0000005) to resume training

    # Logging
    log_interval: int = 1
    save_interval: int = 1000
    save_dir: str = "./checkpoints"
    tensorboard_dir: str = ""  # empty = disabled

    # Precision
    dtype: str = "bf16"

    # Z-loss coefficients.
    # Hidden z-loss: applied to hidden states before the final layernorm.
    hidden_z_loss_coeff: float = 0.0  # 0 = disabled; 1e-7 is a good starting value

    # MoE router z-loss: applied to router logits in each MoE layer.
    router_z_loss_coeff: float = 0.0  # 0 = disabled; 0.2 is a good starting value

    # MoE loss scaling coefficient: multiplies the total MoE auxiliary loss.
    # Suggested: 0.0005 for the generation task, 0.0001 for understanding.
    moe_loss_coeff: float = 0.0005

    # Load balance loss type: controls which MoE auxiliary losses are enabled.
    #   "both":   router z-loss + load balance loss
    #   "z_loss": router z-loss only, no load balance loss
    load_balance_loss_type: str = "both"

    # Loss-free expert balance (DeepSeek LFB, arXiv:2408.15664).
    # Per-step update to the router's e_score_correction_bias, driving expert
    # utilization toward uniform. 0 = disabled. A typical setup uses rate 0.1
    # with decay rule "1000 0.5 0.01", dynamic update, and ffn-bias-only adaptation.
    loss_free_balance_rate: float = 0.0
    loss_free_decay_rule: str = ""  # "interval ratio min" e.g. "1000 0.5 0.01"
    dynamic_update_loss_free_bias: bool = True
    only_adapt_ffn_bias: bool = True

    # Use the grouped_gemm library for fused MoE expert compute.
    # Default False: uses a per-expert matmul loop that is compatible with FSDP2
    # gradient tracking (and avoids the extra grouped_gemm dependency).
    use_grouped_gemm: bool = False

    # RoPE theta override: if > 0, overrides the model config's rope_theta.
    # Set 0 to use the model default; some generation setups use 1e6.
    rope_theta: float = 0.0

    # Embedding LR scaling: multiply the LR of embedding params by
    # hidden_size / emb_lr_scale_base (e.g. 3072 / 576 = 5.333).
    # Applied to embedding weights (2D params with 'embedding' in the name).
    # With ln_scale_lr=True, norm weights and biases (1D params) are scaled too;
    # all other 2D params (attention, FFN, MoE experts) keep lr x 1.0.
    emb_lr_scale_base: int = 0  # 0 = disabled; e.g. 576
    ln_scale_lr: bool = False  # If True, scale LR for LayerNorm/bias too

    @classmethod
    def from_args(cls) -> "TrainConfig":
        """Parse command-line arguments and return a TrainConfig instance."""
        parser = argparse.ArgumentParser(description="LongCat-Next FSDP SFT Training")

        # Task
        parser.add_argument("--task", type=str, default="understand",
                            choices=["understand", "generate", "unify"],
                            help="Task type: understand (image→text), generate (text→image), or unify (mixed)")

        # Model
        parser.add_argument("--model_path", type=str, required=True,
                            help="Path to HuggingFace model checkpoint")

        # Data
        parser.add_argument("--data_path", type=str, required=True,
                            help="Path to JSONL data file(s), comma-separated for multiple files")
        parser.add_argument("--seq_length", type=int, default=None,
                            help="Maximum sequence length (default: 65536 for understand, 8192 for generate)")
        parser.add_argument("--no_packing", action="store_true", default=False,
                            help="Disable packing: each sample gets its own sequence")
        parser.add_argument("--merged_data_dir", type=str, default="",
                            help="Directory for pre-merged+shuffled data file. "
                                 "When set with multiple data files, rank 0 merges all files "
                                 "into one shuffled JSONL before training starts. "
                                 "Empty = no merge, use data_path as-is.")
        parser.add_argument("--skip_merge_check", action="store_true", default=False,
                            help="If the merged file already exists, reuse it directly "
                                 "WITHOUT counting lines in the (full) input files. "
                                 "Speeds up resubmits when the merged data is unchanged. "
                                 "WARNING: assumes the existing merged file is correct; "
                                 "does NOT verify line count against inputs.")

        # Training
        parser.add_argument("--global_batch_size", type=int, default=None)
        parser.add_argument("--micro_batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--weight_decay", type=float, default=None)
        parser.add_argument("--adam_beta1", type=float, default=None)
        parser.add_argument("--adam_beta2", type=float, default=None)
        parser.add_argument("--adam_eps", type=float, default=1e-16)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--warmup_steps", type=int, default=None)
        parser.add_argument("--lr_schedule", type=str, default=None,
                            choices=["cosine", "constant"])
        parser.add_argument("--num_epochs", type=int, default=None)
        parser.add_argument("--seed", type=int, default=42)

        # FSDP2
        # Activation checkpointing is ON by default; pass this flag to disable it.
        parser.add_argument("--no_activation_checkpointing", action="store_true", default=False,
                            help="Disable activation checkpointing (enabled by default)")
        parser.add_argument("--offload_to_cpu", action="store_true", default=False,
                            help="Enable FSDP2 CPU offload to reduce GPU memory")

        # Training control
        parser.add_argument("--max_steps", type=int, default=0,
                            help="Stop after this many optimizer steps (0 = full epochs)")
        parser.add_argument("--skip_steps", type=int, default=0,
                            help="Skip this many optimizer steps at start (consume data without forward/backward, for fast debug)")
        parser.add_argument("--resume_from", type=str, default="",
                            help="Path to checkpoint dir to resume training from")

        # Logging & Saving
        parser.add_argument("--log_interval", type=int, default=1)
        parser.add_argument("--save_interval", type=int, default=1000)
        parser.add_argument("--save_dir", type=str, default="./checkpoints")
        parser.add_argument("--tensorboard_dir", type=str, default="",
                            help="TensorBoard log directory (empty = disabled)")

        # Z-loss
        parser.add_argument("--hidden_z_loss_coeff", type=float, default=0.0,
                            help="Hidden z-loss coefficient (e.g. 1e-7)")
        parser.add_argument("--router_z_loss_coeff", type=float, default=0.0,
                            help="Router z-loss coefficient (e.g. 0.2)")
        parser.add_argument("--moe_loss_coeff", type=float, default=0.0005,
                            help="MoE loss scaling coefficient (e.g. 0.0005 for generate, 0.0001 for understand)")
        parser.add_argument("--load_balance_loss_type", type=str, default=None,
                            choices=["both", "z_loss"],
                            help="MoE loss type: 'both' (z-loss + lb) or 'z_loss' (z-loss only)")
        # Loss-free expert balance (DeepSeek LFB)
        parser.add_argument("--loss_free_balance_rate", type=float, default=0.0,
                            help="Per-step bias update rate. 0 = disabled. e.g. 0.1.")
        parser.add_argument("--loss_free_decay_rule", type=str, default="",
                            help='"interval ratio min" e.g. "1000 0.5 0.01" — every <interval> steps, '
                                 "rate *= ratio (floor at min). Empty = no decay.")
        parser.add_argument("--no_dynamic_update_loss_free_bias", action="store_true", default=False,
                            help="If set: use sign(offset)*rate (binary) instead of offset*rate (proportional).")
        parser.add_argument("--no_only_adapt_ffn_bias", action="store_true", default=False,
                            help="If set: also update bias for zero-experts. Default: only adapt FFN experts.")
        parser.add_argument("--use_grouped_gemm", action="store_true", default=False,
                            help="Use grouped_gemm library for fused MoE. Default: per-expert matmul loop.")

        # RoPE theta override
        parser.add_argument("--rope_theta", type=float, default=0.0,
                            help="Override model's rope_theta. 0 = use model default. "
                                 "e.g. 1000000 for the generation task")

        # Embedding LR scaling
        parser.add_argument("--emb_lr_scale_base", type=int, default=0,
                            help="Embedding LR scale base (e.g. 576). lr_mult = hidden_size / base. 0 = disabled.")
        parser.add_argument("--ln_scale_lr", action="store_true", default=False,
                            help="Also scale LR for LayerNorm/bias params")

        args = parser.parse_args()

        # Apply task-specific defaults
        config = cls(task=args.task, model_path=args.model_path, data_path=args.data_path)

        if args.task == "understand":
            config.seq_length = args.seq_length if args.seq_length is not None else 8192
            config.global_batch_size = args.global_batch_size if args.global_batch_size is not None else 64
            config.weight_decay = args.weight_decay if args.weight_decay is not None else 0.1
            config.adam_beta1 = args.adam_beta1 if args.adam_beta1 is not None else 0.9
            config.adam_beta2 = args.adam_beta2 if args.adam_beta2 is not None else 0.95
            config.warmup_steps = args.warmup_steps if args.warmup_steps is not None else 0
            config.lr_schedule = args.lr_schedule if args.lr_schedule is not None else "constant"
            config.num_epochs = args.num_epochs if args.num_epochs is not None else 1
        elif args.task == "generate":
            config.seq_length = args.seq_length if args.seq_length is not None else 8192
            config.global_batch_size = args.global_batch_size if args.global_batch_size is not None else 32
            config.weight_decay = args.weight_decay if args.weight_decay is not None else 0.0
            config.adam_beta1 = args.adam_beta1 if args.adam_beta1 is not None else 0.9
            config.adam_beta2 = args.adam_beta2 if args.adam_beta2 is not None else 0.99
            config.warmup_steps = args.warmup_steps if args.warmup_steps is not None else 150
            config.lr_schedule = args.lr_schedule if args.lr_schedule is not None else "constant"
            config.num_epochs = args.num_epochs if args.num_epochs is not None else 3
        elif args.task == "unify":
            # Mixed (understand + generate) training.
            config.seq_length = args.seq_length if args.seq_length is not None else 8192
            config.global_batch_size = args.global_batch_size if args.global_batch_size is not None else 128
            config.weight_decay = args.weight_decay if args.weight_decay is not None else 0.0
            config.adam_beta1 = args.adam_beta1 if args.adam_beta1 is not None else 0.9
            config.adam_beta2 = args.adam_beta2 if args.adam_beta2 is not None else 0.95
            config.warmup_steps = args.warmup_steps if args.warmup_steps is not None else 0
            config.lr_schedule = args.lr_schedule if args.lr_schedule is not None else "constant"
            config.num_epochs = args.num_epochs if args.num_epochs is not None else 1

        # Override with explicit args
        config.micro_batch_size = args.micro_batch_size
        config.learning_rate = args.learning_rate
        config.adam_eps = args.adam_eps
        config.max_grad_norm = args.max_grad_norm
        config.seed = args.seed
        config.activation_checkpointing = not args.no_activation_checkpointing
        config.offload_to_cpu = args.offload_to_cpu
        config.max_steps = args.max_steps
        config.skip_steps = args.skip_steps
        config.resume_from = args.resume_from
        config.log_interval = args.log_interval
        config.save_interval = args.save_interval
        config.save_dir = args.save_dir
        config.tensorboard_dir = args.tensorboard_dir

        # Z-loss and MoE loss
        config.hidden_z_loss_coeff = args.hidden_z_loss_coeff
        config.router_z_loss_coeff = args.router_z_loss_coeff
        config.moe_loss_coeff = args.moe_loss_coeff

        # Load balance loss type: default based on task
        # understand: z_loss only; generate: both
        if args.load_balance_loss_type is not None:
            config.load_balance_loss_type = args.load_balance_loss_type
        elif args.task == "understand":
            config.load_balance_loss_type = "z_loss"
        else:
            config.load_balance_loss_type = "both"

        # Loss-free expert balance
        config.loss_free_balance_rate = args.loss_free_balance_rate
        config.loss_free_decay_rule = args.loss_free_decay_rule
        config.dynamic_update_loss_free_bias = not args.no_dynamic_update_loss_free_bias
        config.only_adapt_ffn_bias = not args.no_only_adapt_ffn_bias

        config.use_grouped_gemm = args.use_grouped_gemm

        # RoPE theta override
        config.rope_theta = args.rope_theta

        # Embedding LR scaling
        config.emb_lr_scale_base = args.emb_lr_scale_base
        config.ln_scale_lr = args.ln_scale_lr

        # Packing
        config.no_packing = args.no_packing
        config.merged_data_dir = args.merged_data_dir
        config.skip_merge_check = args.skip_merge_check
        return config
