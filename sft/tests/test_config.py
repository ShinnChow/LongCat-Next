# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Unit tests for config.py — TrainConfig parsing and defaults."""

import sys
import os
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TrainConfig


class TestTrainConfigDefaults:
    """Test TrainConfig default values."""

    def test_default_task(self):
        config = TrainConfig()
        assert config.task == "understand"

    def test_default_seq_length(self):
        config = TrainConfig()
        assert config.seq_length == 8192

    def test_default_learning_rate(self):
        config = TrainConfig()
        assert config.learning_rate == 1e-5

    def test_default_activation_checkpointing(self):
        config = TrainConfig()
        assert config.activation_checkpointing is True

    def test_default_adam_params(self):
        config = TrainConfig()
        assert config.adam_beta1 == 0.9
        assert config.adam_beta2 == 0.95
        assert config.adam_eps == 1e-16

    def test_default_grad_norm(self):
        config = TrainConfig()
        assert config.max_grad_norm == 1.0

    def test_default_tensorboard_dir(self):
        config = TrainConfig()
        assert config.tensorboard_dir == ""


class TestTrainConfigFromArgs:
    """Test TrainConfig.from_args() with simulated CLI arguments."""

    def test_understand_defaults(self, monkeypatch):
        """Understand task should set specific defaults."""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--task", "understand", "--model_path", "/tmp/model", "--data_path", "/tmp/data.jsonl"]
        )
        config = TrainConfig.from_args()
        assert config.task == "understand"
        assert config.seq_length == 8192  # default for 8-GPU FSDP
        assert config.global_batch_size == 64
        assert config.weight_decay == 0.1
        assert config.adam_beta2 == 0.95
        assert config.warmup_steps == 0
        assert config.lr_schedule == "constant"
        assert config.num_epochs == 1

    def test_generate_defaults(self, monkeypatch):
        """Generate task should set different defaults."""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--task", "generate", "--model_path", "/tmp/model", "--data_path", "/tmp/data.jsonl"]
        )
        config = TrainConfig.from_args()
        assert config.task == "generate"
        assert config.seq_length == 8192
        assert config.global_batch_size == 32
        assert config.weight_decay == 0.0
        assert config.adam_beta2 == 0.99
        assert config.warmup_steps == 150
        assert config.lr_schedule == "constant"
        assert config.num_epochs == 3

    def test_explicit_override(self, monkeypatch):
        """Explicit CLI args should override task defaults."""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--task", "understand", "--model_path", "/tmp/model",
             "--data_path", "/tmp/data.jsonl", "--learning_rate", "3e-4",
             "--seq_length", "4096", "--seed", "123"]
        )
        config = TrainConfig.from_args()
        assert config.learning_rate == 3e-4
        assert config.seq_length == 4096
        assert config.seed == 123

    def test_no_activation_checkpointing(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--task", "understand", "--model_path", "/tmp/model",
             "--data_path", "/tmp/data.jsonl", "--no_activation_checkpointing"]
        )
        config = TrainConfig.from_args()
        assert config.activation_checkpointing is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
