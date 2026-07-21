#!/bin/bash
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
# Multi-node understanding-task SFT demo on the ST-VQA dataset.
#
# Prerequisites:
#   1. Download the LongCat-Next checkpoint and set MODEL_PATH:
#        https://huggingface.co/meituan-longcat/LongCat-Next
#   2. Prepare the data once (see examples/prepare_st_vqa.py). The checkpoint and
#      the dataset must live on a SHARED filesystem visible to every node, at the
#      SAME absolute path (image paths in the JSONL are relative to the repo root,
#      so every node must launch from the same repo root too).
#
# How to run:
#   Launch this script ON EACH NODE, passing the cluster topology via environment
#   variables. For an N-node job, NODE_RANK goes 0..N-1; node 0 is the rendezvous
#   host (MASTER_ADDR must point to it and be reachable from all nodes).
#
#   # on node 0 (the master):
#   NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash examples/train_understand_st_vqa_multinode.sh
#   # on node 1:
#   NNODES=4 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
#       bash examples/train_understand_st_vqa_multinode.sh
#   # ... likewise for the remaining nodes.
#
# A workload manager (Slurm, MPI, k8s, etc.) normally fills these in for you.
set -e

# ---- Edit these (must be identical + shared-path on every node) ----
MODEL_PATH="/path/to/longcat-next-checkpoint"   # HF-format LongCat-Next model
DATA_PATH="dataset/st-vqa/st-vqa.jsonl"          # produced by prepare_st_vqa.py
# -------------------------------------------------------------------

# Cluster topology — provided per node via env vars (defaults = single node).
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}   # GPUs per node

SEQ_LENGTH=2048
# Scale the global batch size with the number of GPUs (1 per GPU here).
GLOBAL_BATCH_SIZE=$((NNODES * NPROC_PER_NODE))

SAVE_DIR="checkpoints/st-vqa-understand"
TENSORBOARD_DIR="tensorboards/st-vqa-understand"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional: tune NCCL for your interconnect (e.g. export NCCL_IB_DISABLE=0).

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
    --task understand \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --seq_length ${SEQ_LENGTH} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --micro_batch_size 1 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --seed 48 \
    --log_interval 1 \
    --save_interval 200 \
    --save_dir "${SAVE_DIR}" \
    --tensorboard_dir "${TENSORBOARD_DIR}"
