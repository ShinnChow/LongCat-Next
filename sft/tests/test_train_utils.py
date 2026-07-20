# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for train_utils.py — Optimizer, scheduler, logger."""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TrainConfig
from train_utils import create_optimizer, create_scheduler, TrainingLogger


class TestCreateOptimizer:
    """Test optimizer creation with weight decay grouping."""

    def test_basic_creation(self):
        """Optimizer should be created successfully."""
        model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        config = TrainConfig(learning_rate=1e-4, weight_decay=0.1)
        optimizer = create_optimizer(model, config)
        # Default optimizer is FP32AdamW (fp32 master weights); it exposes a
        # standard param_groups interface and a step() method.
        assert hasattr(optimizer, "param_groups")
        assert hasattr(optimizer, "step")

    def test_weight_decay_groups(self):
        """Bias and norm params should have weight_decay=0."""
        model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10))
        config = TrainConfig(learning_rate=1e-4, weight_decay=0.1)
        optimizer = create_optimizer(model, config)

        # Should have 2 param groups
        assert len(optimizer.param_groups) == 2
        # Group 0: decay params (weight_decay=0.1)
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        # Group 1: no_decay params (weight_decay=0.0)
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_only_trainable_params(self):
        """Frozen params should not be in the optimizer."""
        model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 5))
        model[0].requires_grad_(False)  # freeze first layer
        config = TrainConfig(learning_rate=1e-4, weight_decay=0.1)
        optimizer = create_optimizer(model, config)

        # Count total params in optimizer
        total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
        # Only the second linear should be included (weight + bias)
        assert total_opt_params == 2

    def test_adam_hyperparams(self):
        """Adam hyperparameters should match config."""
        model = nn.Linear(10, 10)
        config = TrainConfig(
            learning_rate=3e-4, adam_beta1=0.85, adam_beta2=0.98, adam_eps=1e-8
        )
        optimizer = create_optimizer(model, config)
        pg = optimizer.param_groups[0]
        assert pg["lr"] == 3e-4
        assert pg["betas"] == (0.85, 0.98)
        assert pg["eps"] == 1e-8


class TestCreateScheduler:
    """Test learning rate scheduler."""

    def _make_optimizer(self, lr=1e-3):
        model = nn.Linear(10, 10)
        return torch.optim.AdamW(model.parameters(), lr=lr)

    def test_cosine_schedule_shape(self):
        """Cosine schedule should decay from 1 to min_ratio."""
        config = TrainConfig(
            lr_schedule="cosine", warmup_steps=10,
            learning_rate=1e-3, min_learning_rate=0.0,
        )
        optimizer = self._make_optimizer()
        scheduler = create_scheduler(optimizer, config, total_steps=100)

        # Collect LR at each step
        lrs = []
        for step in range(100):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # Warmup: LR should increase
        assert lrs[0] < lrs[5] < lrs[9]
        # After warmup: LR should decrease
        assert lrs[10] > lrs[50] > lrs[99]

    def test_constant_schedule(self):
        """Constant schedule should be flat after warmup."""
        config = TrainConfig(
            lr_schedule="constant", warmup_steps=5,
            learning_rate=1e-3, min_learning_rate=0.0,
        )
        optimizer = self._make_optimizer()
        scheduler = create_scheduler(optimizer, config, total_steps=50)

        lrs = []
        for step in range(50):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        # After warmup, LR should be constant
        for lr in lrs[5:]:
            assert abs(lr - 1e-3) < 1e-8

    def test_warmup_linear(self):
        """Warmup should be linear from 0 to base_lr."""
        config = TrainConfig(
            lr_schedule="cosine", warmup_steps=10,
            learning_rate=1e-3, min_learning_rate=0.0,
        )
        optimizer = self._make_optimizer()
        scheduler = create_scheduler(optimizer, config, total_steps=100)

        # At step 0, LR should be 0
        assert scheduler.get_last_lr()[0] == 0.0

        # After 5 warmup steps, LR should be ~0.5 * base_lr
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr_at_5 = scheduler.get_last_lr()[0]
        assert abs(lr_at_5 - 0.5e-3) < 1e-6

    def test_unknown_schedule_raises(self):
        config = TrainConfig(lr_schedule="unknown")
        optimizer = self._make_optimizer()
        with pytest.raises(ValueError, match="Unknown lr_schedule"):
            create_scheduler(optimizer, config, total_steps=100)

    def test_zero_warmup(self):
        """Zero warmup should start at full LR."""
        config = TrainConfig(
            lr_schedule="cosine", warmup_steps=0,
            learning_rate=1e-3, min_learning_rate=0.0,
        )
        optimizer = self._make_optimizer()
        scheduler = create_scheduler(optimizer, config, total_steps=100)
        # At step 0 with 0 warmup, LR should be full base_lr
        # lr_lambda(0) with warmup_steps=0 → 0.5 * (1 + cos(0)) = 1.0
        assert abs(scheduler.get_last_lr()[0] - 1e-3) < 1e-8


class TestTrainingLogger:
    """Test training logger output."""

    def test_rank_zero_only(self, capsys):
        """Non-rank-0 loggers should not print."""
        logger = TrainingLogger(rank=1)
        logger.log_message("This should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_rank_zero_prints(self, capsys):
        """Rank 0 logger should print."""
        logger = TrainingLogger(rank=0)
        logger.log_message("Hello world")
        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_log_interval(self, capsys):
        """Steps not divisible by log_interval should be skipped."""
        logger = TrainingLogger(rank=0, log_interval=5)
        logger.log_step(step=1, loss=1.0, lr=1e-3, grad_norm=0.5)
        captured = capsys.readouterr()
        assert captured.out == ""  # step 1 not divisible by 5

        logger.log_step(step=5, loss=1.0, lr=1e-3, grad_norm=0.5)
        captured = capsys.readouterr()
        assert "iteration" in captured.out  # step 5 is divisible

    def test_log_step_content(self, capsys):
        """Log step should emit the iteration line with loss."""
        logger = TrainingLogger(rank=0, log_interval=1)
        logger.log_step(step=10, loss=2.5, lr=1e-4, grad_norm=0.3)
        captured = capsys.readouterr()
        assert "iteration" in captured.out
        assert "lm loss: 2.500000E+00" in captured.out

    def test_log_metrics(self, capsys):
        """Extra metrics should be printed."""
        logger = TrainingLogger(rank=0, log_interval=1)
        logger.log_step(
            step=1, loss=1.0, lr=1e-3, grad_norm=0.5,
            metrics={"ce_loss": 1.23, "tokens": 100}
        )
        captured = capsys.readouterr()
        assert "lm loss: 1.000000E+00" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
