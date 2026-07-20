#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Convert the Pokémon BLIP-captions dataset (HuggingFace parquet) into the JSONL
format used by this SFT trainer's *generation* task (text -> image).

Source (https://huggingface.co/datasets/reach-vb/pokemon-blip-captions) is a
parquet file with columns:
    image : {bytes, path}   # the image encoded as bytes
    text  : str             # a caption describing the image

This script:
  1. decodes each image and writes it as a .jpg under --image_dir,
  2. writes one JSONL line per (caption, image) pair, referencing the image by a
     RELATIVE path (so the dataset is portable — training is launched from the
     repo root).

Output line format (generation task: text -> image):
    {
      "source": "pokemon-blip-captions",
      "messages": [
        {"role": "user", "content": "CAPTION"},
        {"role": "assistant", "content": "<longcat_img_start>REL/PATH.jpg<longcat_img_end>"}
      ]
    }

The image is placed in the ASSISTANT turn (the trainable region), so it becomes
the generation target — the model learns to produce its VQ tokens.

Example:
    python examples/prepare_pokemon.py \
        --parquet_dir /path/to/pokemon-blip-captions/data \
        --output_jsonl dataset/pokemon/pokemon.jsonl \
        --image_dir dataset/pokemon/images \
        --image_path_prefix dataset/pokemon/images
"""

import argparse
import glob
import io
import json
import os

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Convert Pokémon captions parquet to generation JSONL")
    p.add_argument("--parquet_dir", required=True,
                   help="Directory containing the pokemon *.parquet file(s)")
    p.add_argument("--output_jsonl", required=True,
                   help="Output JSONL path")
    p.add_argument("--image_dir", required=True,
                   help="Directory to write decoded .jpg files into")
    p.add_argument("--image_path_prefix", default="dataset/pokemon/images",
                   help="Relative path prefix written into the JSONL (relative to "
                        "the repo root, from which training is launched). "
                        "Default: dataset/pokemon/images")
    p.add_argument("--source_tag", default="pokemon-blip-captions",
                   help='Value for the "source" field.')
    p.add_argument("--max_samples", type=int, default=0,
                   help="If > 0, stop after writing this many examples (for a quick smoke test)")
    return p.parse_args()


def main():
    args = parse_args()

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No *.parquet found in {args.parquet_dir}")
    print(f"Found {len(shards)} parquet shard(s).")

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    import pyarrow.parquet as pq

    img_idx = 0
    n_examples = 0
    n_skipped = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for shard in shards:
            table = pq.read_table(shard, columns=["image", "text"])
            rows = table.to_pylist()
            print(f"[{os.path.basename(shard)}] {len(rows)} rows")

            for row in rows:
                image_field = row.get("image")
                caption = (row.get("text") or "").strip()
                if not image_field or not caption:
                    n_skipped += 1
                    continue

                image_bytes = image_field.get("bytes")
                if not image_bytes:
                    n_skipped += 1
                    continue

                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    print(f"  skip image {img_idx}: decode failed ({e})")
                    n_skipped += 1
                    continue

                jpg_name = f"{img_idx}.jpg"
                image.save(os.path.join(args.image_dir, jpg_name), "JPEG")
                rel_path = f"{args.image_path_prefix.rstrip('/')}/{jpg_name}"
                img_idx += 1

                record = {
                    "source": args.source_tag,
                    "messages": [
                        {"role": "user", "content": caption},
                        {"role": "assistant",
                         "content": f"<longcat_img_start>{rel_path}<longcat_img_end>"},
                    ],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_examples += 1

                if args.max_samples and n_examples >= args.max_samples:
                    print(f"Reached --max_samples={args.max_samples}, stopping.")
                    print(f"Wrote {n_examples} examples, {img_idx} images, "
                          f"skipped {n_skipped}.")
                    return

    print(f"Done. Wrote {n_examples} examples, {img_idx} images to "
          f"{args.image_dir}, skipped {n_skipped}.")
    print(f"JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
