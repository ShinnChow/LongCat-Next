#!/bin/bash
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
# Single-node, 8-GPU generation-task (text -> image) SFT demo on the
# Pokémon BLIP-captions dataset.
#
# Prerequisites:
#   1. Download the LongCat-Next checkpoint and set MODEL_PATH:
#        https://huggingface.co/meituan-longcat/LongCat-Next
#   2. Prepare the data (from the repo root, so relative image paths resolve):
#        python examples/prepare_pokemon.py \
#            --parquet_dir /path/to/pokemon-blip-captions/data \
#            --output_jsonl dataset/pokemon/pokemon.jsonl \
#            --image_dir dataset/pokemon/images \
#            --image_path_prefix dataset/pokemon/images
#
# Run (from the repo root):
#   bash examples/train_generate_pokemon_8gpu.sh
#
# The generation task trains the model to produce an image's VQ tokens from a text
# prompt, so the image lives in the (trainable) assistant turn and a depth
# cross-entropy loss is applied over its VQ levels.
set -e

# ---- Edit these ----
#MODEL_PATH="/path/to/longcat-next-checkpoint"   # HF-format LongCat-Next model
DATA_PATH="dataset/pokemon/pokemon.jsonl"        # produced by prepare_pokemon.py
# --------------------

# One image's VQ tokens + a short caption fit well under 4096. Increase this if
# your images produce more visual tokens (larger images) or captions are long.
SEQ_LENGTH=4096
NPROC_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29500}

SAVE_DIR="checkpoints/pokemon-generate"
TENSORBOARD_DIR="tensorboards/pokemon-generate"

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
    --task generate \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --seq_length ${SEQ_LENGTH} \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --learning_rate 1e-5 \
    --lr_schedule constant \
    --warmup_steps 20 \
    --num_epochs 3 \
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
