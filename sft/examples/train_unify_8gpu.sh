#!/bin/bash
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
# Single-node, 8-GPU UNIFIED (mixed understanding + generation) SFT demo.
#
# Mixed training packs understanding samples (image -> text) and generation
# samples (text -> image) into the same sequences and routes the loss per token.
# This demo mixes the ST-VQA (understanding) and Pokémon (generation) datasets.
#
# Prerequisites:
#   1. Download the LongCat-Next checkpoint and set MODEL_PATH:
#        https://huggingface.co/meituan-longcat/LongCat-Next
#   2. Prepare BOTH datasets (from the repo root):
#        python examples/prepare_st_vqa.py  --parquet_dir ... \
#            --output_jsonl dataset/st-vqa/st-vqa.jsonl \
#            --image_dir dataset/st-vqa/images --image_path_prefix dataset/st-vqa/images
#        python examples/prepare_pokemon.py --parquet_dir ... \
#            --output_jsonl dataset/pokemon/pokemon.jsonl \
#            --image_dir dataset/pokemon/images --image_path_prefix dataset/pokemon/images
#
# Run (from the repo root):
#   bash examples/train_unify_8gpu.sh
set -e

# ---- Edit these ----
MODEL_PATH="/path/to/longcat-next-checkpoint"   # HF-format LongCat-Next model
# Comma-separated list of the understanding + generation JSONL files.
DATA_PATH="dataset/st-vqa/st-vqa.jsonl,dataset/pokemon/pokemon.jsonl"
# rank 0 merges + shuffles the inputs into one JSONL here (must be writable).
MERGED_DATA_DIR="dataset/unify-merged"
# --------------------

# Generation samples dominate the sequence length (VQ tokens), so size seq_length
# for those. Raise it if your images produce more visual tokens.
SEQ_LENGTH=4096
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

SAVE_DIR="checkpoints/unify"
TENSORBOARD_DIR="tensorboards/unify"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run from the repo root so relative image paths in the JSONL resolve.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train.py \
    --task unify \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --merged_data_dir "${MERGED_DATA_DIR}" \
    --seq_length ${SEQ_LENGTH} \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --learning_rate 1e-5 \
    --lr_schedule constant \
    --warmup_steps 20 \
    --num_epochs 1 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --hidden_z_loss_coeff 1e-7 \
    --router_z_loss_coeff 0.2 \
    --moe_loss_coeff 0.0005 \
    --load_balance_loss_type both \
    --loss_free_balance_rate 0.1 \
    --loss_free_decay_rule "1000 0.5 0.01" \
    --emb_lr_scale_base 576 \
    --ln_scale_lr \
    --seed 48 \
    --log_interval 1 \
    --save_interval 200 \
    --save_dir "${SAVE_DIR}" \
    --tensorboard_dir "${TENSORBOARD_DIR}"
