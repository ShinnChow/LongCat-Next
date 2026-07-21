#!/bin/bash
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
# Multi-node generation-task (text -> image) SFT demo on the Pokémon
# BLIP-captions dataset.
#
# Prerequisites:
#   1. Download the LongCat-Next checkpoint and set MODEL_PATH:
#        https://huggingface.co/meituan-longcat/LongCat-Next
#   2. Prepare the data once (see examples/prepare_pokemon.py). The checkpoint and
#      the dataset must live on a SHARED filesystem visible to every node, at the
#      SAME absolute path, and every node must launch from the same repo root.
#
# How to run:
#   Launch this script ON EACH NODE, passing the cluster topology via environment
#   variables. For an N-node job, NODE_RANK goes 0..N-1; node 0 is the rendezvous
#   host (MASTER_ADDR must point to it and be reachable from all nodes).
#
#   # on node 0 (the master):
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash examples/train_generate_pokemon_multinode.sh
#   # on each other node, with NODE_RANK=1..N-1 and the same MASTER_ADDR/PORT.
#
# A workload manager (Slurm, MPI, k8s, etc.) normally fills these in for you.
set -e

# ---- Edit these (must be identical + shared-path on every node) ----
MODEL_PATH="/path/to/longcat-next-checkpoint"   # HF-format LongCat-Next model
DATA_PATH="dataset/pokemon/pokemon.jsonl"        # produced by prepare_pokemon.py
# -------------------------------------------------------------------

# Cluster topology — provided per node via env vars (defaults = single node).
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}   # GPUs per node

# One image's VQ tokens + a short caption fit well under 4096. Increase this if
# your images produce more visual tokens (larger images) or captions are long.
SEQ_LENGTH=4096
# Scale the global batch size with the number of GPUs (1 per GPU here).
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE))

SAVE_DIR="checkpoints/pokemon-generate"
TENSORBOARD_DIR="tensorboards/pokemon-generate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run from the repo root so relative image paths in the JSONL resolve.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "node ${NODE_RANK}/${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}, " \
     "gpus/node=${NPROC_PER_NODE}, global_batch_size=${GLOBAL_BATCH_SIZE}"

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --task generate \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --seq_length ${SEQ_LENGTH} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
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
