#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Convert the ST-VQA dataset (HuggingFace parquet) into the JSONL format used by
this SFT trainer's *understanding* task.

Source (https://huggingface.co/datasets/vikhyatk/st-vqa) is stored as parquet
shards with columns:
    image : {bytes, path}         # the image encoded as bytes
    qas   : [{question, answers}] # one image may have several Q/A pairs

This script:
  1. decodes each image and writes it as a .jpg under --image_dir,
  2. expands every (image, question) pair into one training example,
  3. writes a JSONL line per example, referencing the image by a RELATIVE path
     (so the dataset is portable — training is launched from the repo root).

Output line format (understanding task: image + text -> text):
    {
      "source": "st-vqa",
      "messages": [
        {"role": "user", "content": "<longcat_img_start>REL/PATH.jpg<longcat_img_end>QUESTION"},
        {"role": "assistant", "content": "ANSWER"}
      ],
      "img-size": [[W, H]]
    }

Example:
    python examples/prepare_st_vqa.py \
        --parquet_dir /path/to/st-vqa/data \
        --output_jsonl dataset/st-vqa/st-vqa.jsonl \
        --image_dir dataset/st-vqa/images \
        --image_path_prefix dataset/st-vqa/images

The <longcat_img_start>/<longcat_img_end> tags wrap the image path; the trainer's understanding
dataset replaces that span with the encoded image tokens at load time.
"""

import argparse
import glob
import io
import json
import os

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Convert ST-VQA parquet to training JSONL")
    p.add_argument("--parquet_dir", required=True,
                   help="Directory containing the ST-VQA train-*.parquet shards")
    p.add_argument("--output_jsonl", required=True,
                   help="Output JSONL path")
    p.add_argument("--image_dir", required=True,
                   help="Directory to write decoded .jpg files into")
    p.add_argument("--image_path_prefix", default="dataset/st-vqa/images",
                   help="Relative path prefix written into the JSONL (relative to "
                        "the repo root, from which training is launched). "
                        "Default: dataset/st-vqa/images")
    p.add_argument("--source_tag", default="st-vqa",
                   help='Value for the "source" field. Default: st-vqa')
    p.add_argument("--max_samples", type=int, default=0,
                   help="If > 0, stop after writing this many examples (for a quick smoke test)")
    return p.parse_args()


def main():
    args = parse_args()

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "train-*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No train-*.parquet found in {args.parquet_dir}")
    print(f"Found {len(shards)} parquet shard(s).")

    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    # pyarrow is a transitive dependency of the HF datasets ecosystem; import lazily
    # so the rest of the repo doesn't require it.
    import pyarrow.parquet as pq

    img_idx = 0        # global image counter → N.jpg
    n_examples = 0     # written JSONL lines
    n_skipped = 0      # images/QAs skipped due to errors

    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for shard in shards:
            table = pq.read_table(shard, columns=["image", "qas"])
            rows = table.to_pylist()
            print(f"[{os.path.basename(shard)}] {len(rows)} rows")

            for row in rows:
                image_field = row.get("image")
                qas = row.get("qas") or []
                if not image_field or not qas:
                    n_skipped += 1
                    continue

                image_bytes = image_field.get("bytes")
                if not image_bytes:
                    n_skipped += 1
                    continue

                # Decode + save the image once, re-used by all its QA pairs.
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    print(f"  skip image {img_idx}: decode failed ({e})")
                    n_skipped += 1
                    continue

                w, h = image.size
                jpg_name = f"{img_idx}.jpg"
                image.save(os.path.join(args.image_dir, jpg_name), "JPEG")
                rel_path = f"{args.image_path_prefix.rstrip('/')}/{jpg_name}"
                img_idx += 1

                for qa in qas:
                    question = (qa.get("question") or "").strip()
                    answers = qa.get("answers") or []
                    if not question or not answers:
                        n_skipped += 1
                        continue
                    answer = str(answers[0]).strip()  # take the first reference answer
                    if not answer:
                        n_skipped += 1
                        continue

                    record = {
                        "source": args.source_tag,
                        "messages": [
                            {"role": "user",
                             "content": f"<longcat_img_start>{rel_path}<longcat_img_end>{question}"},
                            {"role": "assistant", "content": answer},
                        ],
                        "img-size": [[w, h]],
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
