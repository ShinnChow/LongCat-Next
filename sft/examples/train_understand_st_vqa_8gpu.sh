#!/bin/bash
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
# Single-node, 8-GPU understanding-task SFT demo on the ST-VQA dataset.
#
# Prerequisites:
#   1. Download the LongCat-Next checkpoint and set MODEL_PATH:
#        https://huggingface.co/meituan-longcat/LongCat-Next
#   2. Prepare the data (from the repo root, so relative image paths resolve):
#        python examples/prepare_st_vqa.py \
#            --parquet_dir /path/to/st-vqa/data \
#            --output_jsonl dataset/st-vqa/st-vqa.jsonl \
#            --image_dir dataset/st-vqa/images \
#            --image_path_prefix dataset/st-vqa/images
#
# Run (from the repo root):
#   bash examples/train_understand_st_vqa_8gpu.sh
#
# Tested on 8x H800 (80GB). The ~75B model shards to ~24GB/GPU after loading;
# activation checkpointing keeps the forward/backward peak within 80GB.
set -e

# ---- Edit these ----
#MODEL_PATH="/path/to/longcat-next-checkpoint"   # HF-format LongCat-Next model
DATA_PATH="dataset/st-vqa/st-vqa.jsonl"          # produced by prepare_st_vqa.py
# --------------------

SEQ_LENGTH=2048
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

SAVE_DIR="checkpoints/st-vqa-understand"
TENSORBOARD_DIR="tensorboards/st-vqa-understand"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run from the repo root so that relative image paths in the JSONL resolve.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train.py \
    --task understand \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --seq_length ${SEQ_LENGTH} \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --seed 48 \
    --log_interval 1 \
    --save_interval 200 \
    --save_dir "${SAVE_DIR}" \
    --tensorboard_dir "${TENSORBOARD_DIR}"
