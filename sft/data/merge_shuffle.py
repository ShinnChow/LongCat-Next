#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Merge multiple JSONL files into one deterministically-shuffled file.

Used to eliminate data distribution bias when sharding across many GPUs.
Without merging, round-robin + modulo sharding over N files where
gcd(world_size, N) > 1 causes systematic bias: some ranks only see
even-indexed files, others only odd-indexed files.

Solution: pre-merge all files into one shuffled JSONL, then use simple
line_idx % world_size == rank sharding.

Usage:
    python merge_shuffle.py \
        --input file1.jsonl file2.jsonl ... \
        --output /path/to/merged.jsonl \
        --seed 42

    # Or from Python:
    from data.merge_shuffle import merge_and_shuffle
    merge_and_shuffle(input_files, output_path, seed=42)
"""

import argparse
import logging
import os
import random
import sys
import time



logger = logging.getLogger(__name__)

def count_lines(filepath):
    """Count lines in a file efficiently."""
    count = 0
    with open(filepath, "rb") as f:
        # Read in chunks for speed
        buf_size = 1024 * 1024  # 1MB
        buf = f.raw.read(buf_size)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(buf_size)
    return count


def merge_and_shuffle(input_files, output_path, seed=42):
    """Merge multiple JSONL files into one deterministically-shuffled file.

    Args:
        input_files: List of JSONL file paths to merge.
        output_path: Output path for the merged+shuffled JSONL file.
        seed: Random seed for deterministic shuffling.

    Returns:
        output_path on success.

    Idempotent: if output_path already exists with the correct line count,
    skips re-creation.
    """
    # Compute expected total lines
    t0 = time.time()
    logger.info(f"Counting lines in {len(input_files)} input files...")
    expected_total = 0
    for f in input_files:
        n = count_lines(f)
        logger.info(f"  {os.path.basename(f)}: {n} lines")
        expected_total += n
    logger.info(f"Expected total: {expected_total} lines "
          f"(counted in {time.time() - t0:.1f}s)")

    # Idempotency check
    if os.path.exists(output_path):
        actual = count_lines(output_path)
        if actual == expected_total:
            logger.info(f"Output already exists with {actual} lines, skipping.")
            return output_path
        else:
            logger.info(f"Output exists but has {actual} lines "
                  f"(expected {expected_total}), re-creating.")

    # Read all lines
    t1 = time.time()
    logger.info(f"Reading all {expected_total} lines into memory...")
    all_lines = []
    for f in input_files:
        with open(f, "r", encoding="utf-8") as fh:
            all_lines.extend(fh)  # keeps newline chars
    logger.info(f"Read {len(all_lines)} lines in {time.time() - t1:.1f}s")

    # Deterministic shuffle
    t2 = time.time()
    logger.info(f"Shuffling with seed={seed}...")
    rng = random.Random(seed)
    rng.shuffle(all_lines)
    logger.info(f"Shuffled in {time.time() - t2:.1f}s")

    # Write output
    t3 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Write to temp file first, then atomic rename to prevent partial files
    tmp_path = output_path + ".tmp"
    logger.info(f"Writing to {tmp_path}...")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.writelines(all_lines)
    os.rename(tmp_path, output_path)
    logger.info(f"Written {len(all_lines)} lines to {output_path} "
          f"in {time.time() - t3:.1f}s")
    logger.info(f"Total time: {time.time() - t0:.1f}s")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Merge and deterministically shuffle multiple JSONL files."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input JSONL file paths."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output merged+shuffled JSONL file path."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic shuffling (default: 42)."
    )
    args = parser.parse_args()

    # Validate inputs exist
    for f in args.input:
        if not os.path.isfile(f):
            print(f"[ERROR] Input file not found: {f}", file=sys.stderr)
            sys.exit(1)

    merge_and_shuffle(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
