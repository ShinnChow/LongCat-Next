# LongCat-Next FSDP2 SFT

Supervised fine-tuning (SFT) for the **LongCat-Next** omni-modal model, built on **PyTorch FSDP2** (`fully_shard`). This codebase trains a Mixture-of-Experts (MoE) LLM backbone together with a vision encoder (ViT + VQ bridge) and a visual generation head, and supports mixing understanding and generation samples in a single training run.

## Features

- **FSDP2 sharding** (`torch.distributed._composable.fsdp`) with no tensor/pipeline parallelism. Optional HSDP (Hybrid Sharded Data Parallel) via the `FSDP_SHARD_SIZE` environment variable.
- **Three task modes**, selected by `--task`:
  - `understand` — image + text in, text out (standard cross-entropy loss).
  - `generate` — text in, image out (VQ tokens with a depth cross-entropy loss).
  - `unify` — mixed understanding and generation samples packed into the same sequences. Losses are computed per-sample and combined.
- **Online packing** of variable-length samples with `cu_seqlens` for varlen attention.
- **fp32 master-weight optimizer** (`FP32AdamW`) that keeps optimizer state in fp32 on CPU while the model runs in bf16 on GPU.
- **Activation checkpointing** with a determinism-preserving MoE path, so recomputation matches the original forward bitwise.
- **Resumable training** via `torchdata` StatefulDataLoader (data order is restored on resume).

## Repository layout

```plain
sft/
├── train.py              # entry point (torchrun target)
├── config.py             # TrainConfig + CLI argument parsing
├── train_utils.py        # optimizer/scheduler, TrainingLogger, TensorBoard
├── fp32_optimizer.py     # FP32AdamW (fp32 master weights on CPU)
├── data/                 # datasets, packing, tokenization, image preprocessing
│   ├── understand_dataset.py
│   ├── generate_dataset.py
│   ├── unify_dataset.py
│   ├── image_processing.py
│   ├── tokenize_utils.py
│   ├── chat_template.py
│   └── merge_shuffle.py
├── model/
│   ├── model_loader.py   # meta-device load, weight sharding, freeze control
│   └── fsdp_utils.py     # FSDP2 setup, MoE patch, checkpoint save/load
├── losses/
│   ├── unified_loss.py   # text CE + depth CE, per-sample aggregation
│   ├── z_loss.py         # router z-loss / load-balance loss
│   └── loss_free_balance.py
├── examples/             # end-to-end data-prep + training scripts (see below)
└── tests/                # unit tests (run with pytest)
```

## Installation

Requires **Python > 3.11** (3.12 recommended). Start from a clean environment:

```bash
conda create -n py312 python=3.12 -y
conda activate py312
pip install -r requirements.txt
```

For `flash-attn`, prefer a prebuilt wheel — pip picks one automatically when it matches your torch/CUDA/Python, avoiding a slow and error-prone source build. If none matches, grab the right wheel (by torch/CUDA/Python/ABI) from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases).

To build from source instead, remove `flash-attn` from `requirements.txt` first, then `pip install "flash-attn>=2.8.1" --no-build-isolation`.

`grouped_gemm` is **optional**. By default the MoE experts run through a per-expert matmul loop, which needs no extra dependency. Installing `grouped_gemm` and passing `--use_grouped_gemm` fuses the expert GEMMs and can speed up MoE compute, but it materializes the stacked expert weights and uses noticeably more GPU memory — enable it only if you have headroom.

The LongCat-Next checkpoint is available at [meituan-longcat/LongCat-Next](https://huggingface.co/meituan-longcat/LongCat-Next). It is loaded with `trust_remote_code=True`.

## Quick start

Launch with `torchrun`. Minimal single-node, 8-GPU example for the unify (mixed) task:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --master_port=29517 \
    train.py \
    --task unify \
    --model_path /path/to/longcat-next-checkpoint \
    --data_path /path/to/data.jsonl \
    --seq_length 1600 \
    --global_batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --save_dir /path/to/save/checkpoints \
    --log_interval 1
```

Switch `--task` to `understand` or `generate` to train a single modality. For ready-to-run, end-to-end scripts (data preparation + launch) on open datasets, see the [Examples](#examples) section below.

### Key arguments

| Argument | Description |
| --- | --- |
| `--task` | `understand`, `generate`, or `unify`. |
| `--model_path` | HuggingFace-format LongCat-Next checkpoint directory. |
| `--data_path` | JSONL data file(s); comma-separated for multiple. |
| `--seq_length` | Packed sequence length. |
| `--global_batch_size` / `--micro_batch_size` | Global and per-step batch sizes. |
| `--no_activation_checkpointing` | Disable activation checkpointing (on by default). |
| `--save_interval` / `--save_dir` | Checkpoint cadence and output directory. |
| `--resume_from` | Resume from a saved checkpoint (weights + data order). |

Run `python train.py --help` for the full list.

### Distributed / sharding

- `FSDP_SHARD_SIZE` — set below the world size to enable HSDP (shard within groups of this size, replicate across groups). Unset or equal to the world size gives pure FSDP.

## Data format

Each training sample is one JSONL line. The only required field is `messages`, a list of chat turns. Each turn must have a `role` (`user` / `assistant`) and a `content` string. Images are referenced inline in the content, wrapped in `<longcat_img_start>` / `<longcat_img_end>` tags; the path is resolved relative to the repo root (training is launched from there).

The loss is computed only on the `assistant` turns. **Where an image sits decides its role**: an image in a `user` turn is an *input* (understanding), while an image in an `assistant` turn is a *generation target* (its VQ tokens are predicted under a depth loss). The same `messages` schema is used for all three tasks — the `--task` flag only selects the loss/packing behaviour.

**Understanding** (image + text -> text): the image is in the user turn.

```json
{"messages": [
  {"role": "user", "content": "<longcat_img_start>dataset/st-vqa/images/0.jpg<longcat_img_end>What is the book author's first name?"},
  {"role": "assistant", "content": "Susan"}
]}
```

**Generation** (text -> image): the image is in the assistant turn.

```json
{"messages": [
  {"role": "user", "content": "a drawing of a green pokemon with red eyes"},
  {"role": "assistant", "content": "<longcat_img_start>dataset/pokemon/images/0.jpg<longcat_img_end>"}
]}
```

**Unify** (mixed): use the exact same understanding and generation lines together. You do not write a special format — just pass both JSONL files (see the unify example below); each line is classified per sample by where its image sits.

Other fields (e.g. `source`, `img-size`) are ignored by the trainer and may be kept for bookkeeping.

## Examples

The `examples/` directory contains complete, runnable data-prep + training scripts for all three tasks on open datasets:

| Script | Task | Scale |
| --- | --- | --- |
| `prepare_st_vqa.py` | understanding data prep | — |
| `prepare_pokemon.py` | generation data prep | — |
| `train_understand_st_vqa_8gpu.sh` | understand | single node, 8 GPUs |
| `train_understand_st_vqa_multinode.sh` | understand | multi-node |
| `train_generate_pokemon_8gpu.sh` | generate | single node, 8 GPUs |
| `train_generate_pokemon_multinode.sh` | generate | multi-node |
| `train_unify_8gpu.sh` | unify (mixed) | single node, 8 GPUs |
| `train_unify_multinode.sh` | unify (mixed) | multi-node |

Set `MODEL_PATH` at the top of each training script to your LongCat-Next checkpoint. Run everything from the repo root so the relative image paths in the JSONL resolve. The generated data lands under `dataset/` (git-ignored). Tested on 8x H800 (80GB): the ~75B model shards to ~24GB/GPU after loading, and activation checkpointing keeps the forward/backward peak within 80GB.

### Understanding (image + text -> text)

Uses the open [ST-VQA](https://huggingface.co/datasets/vikhyatk/st-vqa) dataset.

```bash
# 1. Convert the parquet shards to training JSONL (decodes images, expands each
#    (image, question) pair into one example).
python examples/prepare_st_vqa.py \
    --parquet_dir /path/to/st-vqa/data \
    --output_jsonl dataset/st-vqa/st-vqa.jsonl \
    --image_dir dataset/st-vqa/images \
    --image_path_prefix dataset/st-vqa/images

# 2. Train (single node).
bash examples/train_understand_st_vqa_8gpu.sh
```

### Generation (text -> image)

Uses the open [Pokémon BLIP-captions](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) dataset. The image is placed in the (trainable) assistant turn, so the model learns to produce its VQ tokens.

```bash
python examples/prepare_pokemon.py \
    --parquet_dir /path/to/pokemon-blip-captions/data \
    --output_jsonl dataset/pokemon/pokemon.jsonl \
    --image_dir dataset/pokemon/images \
    --image_path_prefix dataset/pokemon/images

bash examples/train_generate_pokemon_8gpu.sh
```

### Unified (mixed understanding + generation)

Mixed training reuses the two datasets above — there is no separate prep step. Run both `prepare_st_vqa.py` and `prepare_pokemon.py` first, then launch the unify script. It passes both JSONL files as a comma-separated `--data_path` and a `--merged_data_dir`; rank 0 merges and shuffles them into one file once before training (the merge directory is created automatically).

```bash
# after preparing both dataset/st-vqa/st-vqa.jsonl and dataset/pokemon/pokemon.jsonl:
bash examples/train_unify_8gpu.sh
```

The relevant lines inside the script are:

```bash
DATA_PATH="dataset/st-vqa/st-vqa.jsonl,dataset/pokemon/pokemon.jsonl"
MERGED_DATA_DIR="dataset/unify-merged"
```

The stdout log then reports both `text_loss` (understanding) and `image_token_loss` / `paraDec_level-*` (generation) in each step.

### Multi-node

Each task has a `*_multinode.sh` variant. Launch it on every node, passing the cluster topology via environment variables (a job scheduler usually fills these in). The checkpoint, datasets, and merged-data directory must all live on a shared filesystem at the same path on every node.

```bash
# on the master node (rank 0):
NNODES=4 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29500 \
    bash examples/train_understand_st_vqa_multinode.sh
# on each other node, with NODE_RANK=1..N-1 and the same MASTER_ADDR/PORT.
```

The global batch size scales automatically with the total number of GPUs.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
