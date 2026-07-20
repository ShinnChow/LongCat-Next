# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Tests for understanding task dataset processing pipeline.

Run on the cluster (needs model tokenizer/processor and real data):
    cd /path/to/sft && python tests/test_understand_dataset.py

These tests validate:
1. Single sample tokenization (chat template + trainable markers + image placeholders)
2. Image preprocessing (pixel_values shape and dtype)
3. Visual token count consistency (placeholder count == pixel_values patch count after ViT merge)
4. Packing correctness (padding, causal shift, position_ids reset)
5. Loss mask correctness (only assistant response tokens have loss=1)
6. visual_mask correctness (IMG_PAD positions match pixel_values count)
7. Visual token count formula (matches grid_h * grid_w // merge // merge)

Usage:
    # Run all tests
    python tests/test_understand_dataset.py

    # Run specific test
    python tests/test_understand_dataset.py TestUnderstandDataset.test_single_sample_tokenize
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

# Model and data paths — provide via environment variables to run this test.
# These require a real tokenizer/processor and dataset, so the test is skipped
# when the paths are not configured.
MODEL_PATH = os.environ.get("MODEL_PATH", "")
DATA_PATH = os.environ.get("DATA_PATH", "")


def _load_tokenizer_and_processor():
    """Load tokenizer and processor from model path."""
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer, processor


def _load_first_n_samples(data_path, n=5):
    """Load first N samples from a JSONL file."""
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


class TestUnderstandDataset(unittest.TestCase):
    """Test the understanding task dataset pipeline."""

    @classmethod
    def setUpClass(cls):
        """Load tokenizer, processor, and sample data once."""
        if not os.path.exists(MODEL_PATH):
            raise unittest.SkipTest(f"Model not found: {MODEL_PATH}")
        if not os.path.exists(DATA_PATH):
            raise unittest.SkipTest(f"Data not found: {DATA_PATH}")

        print(f"Loading tokenizer from {MODEL_PATH}...")
        cls.tokenizer, cls.processor = _load_tokenizer_and_processor()
        print(f"  Vocab size: {len(cls.tokenizer)}")

        from data.image_processing import ImagePreprocessor
        cls.image_preprocessor = ImagePreprocessor(cls.processor)

        print(f"Loading samples from {DATA_PATH}...")
        cls.raw_samples = _load_first_n_samples(DATA_PATH, n=10)
        print(f"  Loaded {len(cls.raw_samples)} samples")

        # Print first sample structure for debugging
        if cls.raw_samples:
            s = cls.raw_samples[0]
            print(f"  Sample keys: {list(s.keys())}")
            if "messages" in s:
                print(f"  Messages count: {len(s['messages'])}")
                for i, m in enumerate(s["messages"][:3]):
                    role = m.get("role", m.get("from", "?"))
                    content = m.get("content", m.get("value", ""))
                    print(f"    [{i}] role={role}, content_len={len(content)}, "
                          f"has_img={'<longcat_img_start>' in content}")

    # ===================================================================
    # Test 1: Raw data format validation
    # ===================================================================
    def test_01_raw_data_format(self):
        """Validate raw data has expected format."""
        for i, sample in enumerate(self.raw_samples[:5]):
            self.assertIn("messages", sample, f"Sample {i} missing 'messages' key")
            messages = sample["messages"]
            self.assertGreater(len(messages), 0, f"Sample {i} has empty messages")

            for j, msg in enumerate(messages):
                # Either {role, content} or {from, value} format
                has_standard = "role" in msg and "content" in msg
                has_legacy = "from" in msg and "value" in msg
                self.assertTrue(has_standard or has_legacy,
                    f"Sample {i}, msg {j} has neither standard nor legacy format: {list(msg.keys())}")
        print("test_01_raw_data_format PASSED")

    # ===================================================================
    # Test 2: Chat template + trainable markers
    # ===================================================================
    def test_02_chat_template(self):
        """Test chat template produces correct trainable markers."""
        from data.chat_template import (
            unify_message_format, encode_with_trainable_markers,
            TRAINABLE_START, TRAINABLE_END, ROLE_ASSISTANT,
        )

        for i, sample in enumerate(self.raw_samples[:3]):
            messages = unify_message_format(sample["messages"])
            marked_text = encode_with_trainable_markers(self.tokenizer, messages)

            # Should have trainable markers
            self.assertIn(TRAINABLE_START, marked_text,
                f"Sample {i}: no TRAINABLE_START found")
            self.assertIn(TRAINABLE_END, marked_text,
                f"Sample {i}: no TRAINABLE_END found")

            # Trainable regions should be balanced
            starts = marked_text.count(TRAINABLE_START)
            ends = marked_text.count(TRAINABLE_END)
            self.assertEqual(starts, ends,
                f"Sample {i}: unbalanced markers ({starts} starts, {ends} ends)")

            # Number of trainable regions = number of assistant messages (with mask=1)
            assistant_count = sum(1 for m in messages if m["role"] == "assistant"
                                  and m.get("assistant_masks", 1) != 0)
            self.assertEqual(starts, assistant_count,
                f"Sample {i}: {starts} trainable regions but {assistant_count} assistant messages")

            print(f"  Sample {i}: {starts} trainable regions, text_len={len(marked_text)}")

        print("test_02_chat_template PASSED")

    # ===================================================================
    # Test 3: Image preprocessing
    # ===================================================================
    def test_03_image_preprocessing(self):
        """Test image preprocessing returns correct pixel_values and grid_thw."""
        import re

        # Find a sample with images
        image_paths = []
        for sample in self.raw_samples:
            for msg in sample.get("messages", []):
                content = msg.get("content", msg.get("value", ""))
                for match in re.finditer(r"<longcat_img_start>(.*?)<longcat_img_end>", content):
                    path = match.group(1).strip()
                    image_paths.append(path)
                    if len(image_paths) >= 3:
                        break
            if len(image_paths) >= 3:
                break

        if not image_paths:
            self.skipTest("No images found in sample data")

        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"  Skipping non-existent image: {img_path}")
                continue

            result = self.image_preprocessor.preprocess(img_path)

            pv = result["pixel_values"]
            gthw = result["image_grid_thw"]

            # pixel_values: [num_patches, C*t*pH*pW]
            self.assertEqual(pv.dim(), 2, f"pixel_values should be 2D, got {pv.dim()}D")
            self.assertEqual(pv.shape[1], 1176,
                f"pixel_values dim1 should be 1176 (3*2*14*14), got {pv.shape[1]}")

            # image_grid_thw: [1, 3]
            self.assertEqual(gthw.shape, (1, 3),
                f"image_grid_thw should be [1, 3], got {gthw.shape}")

            # Verify consistency: num_patches should equal t * grid_h * grid_w
            t, gh, gw = gthw[0].tolist()
            expected_patches = t * gh * gw
            self.assertEqual(pv.shape[0], expected_patches,
                f"pixel_values has {pv.shape[0]} patches but grid_thw says {expected_patches}")

            print(f"  Image: {os.path.basename(img_path)}, "
                  f"pixel_values={pv.shape}, grid_thw=[{t},{gh},{gw}]")

        print("test_03_image_preprocessing PASSED")

    # ===================================================================
    # Test 4: Visual token count formula
    # ===================================================================
    def test_04_visual_token_count(self):
        """Verify the visual token count formula.

        Expected: num_tokens = grid_h * grid_w // spatial_merge_size // spatial_merge_size
        Ours:     num_tokens = t * (grid_h // 2) * (grid_w // 2)
        These should be equal when t=1 and grid dims are even.
        """
        from data.image_processing import estimate_visual_tokens
        import re

        image_paths = []
        for sample in self.raw_samples:
            for msg in sample.get("messages", []):
                content = msg.get("content", msg.get("value", ""))
                for match in re.finditer(r"<longcat_img_start>(.*?)<longcat_img_end>", content):
                    path = match.group(1).strip()
                    if os.path.exists(path):
                        image_paths.append(path)
                    if len(image_paths) >= 3:
                        break
            if len(image_paths) >= 3:
                break

        if not image_paths:
            self.skipTest("No accessible images found")

        for img_path in image_paths:
            result = self.image_preprocessor.preprocess(img_path)
            gthw = result["image_grid_thw"]
            t, gh, gw = gthw[0].tolist()

            # Our formula
            our_tokens, our_h, our_w = estimate_visual_tokens(gthw)

            # Expected formula: grid_h * grid_w // 2 // 2 (spatial_merge_size=2)
            expected_tokens = gh * gw // 2 // 2

            self.assertEqual(our_tokens, expected_tokens,
                f"Token count mismatch for {img_path}: ours={our_tokens}, "
                f"expected={expected_tokens} (t={t}, gh={gh}, gw={gw})")

            # Verify grid dims are even (required for integer division)
            self.assertEqual(gh % 2, 0, f"grid_h={gh} is not even")
            self.assertEqual(gw % 2, 0, f"grid_w={gw} is not even")

            print(f"  Image: grid=[{t},{gh},{gw}], tokens={our_tokens}")

        print("test_04_visual_token_count PASSED")

    # ===================================================================
    # Test 5: Single sample processing
    # ===================================================================
    def test_05_single_sample_processing(self):
        """Test _process_sample returns correct structure."""
        from data.understand_dataset import UnderstandPackedDataset

        ds = UnderstandPackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=65536,
            rank=0,
            world_size=1,
        )

        for i, sample in enumerate(self.raw_samples[:3]):
            result = ds._process_sample(sample)
            if result is None:
                print(f"  Sample {i}: skipped (processing returned None)")
                continue

            input_ids = result["input_ids"]
            loss_mask = result["loss_mask"]
            image_info = result["image_info"]

            # Shape checks
            self.assertEqual(input_ids.dim(), 1, "input_ids should be 1D")
            self.assertEqual(loss_mask.dim(), 1, "loss_mask should be 1D")
            self.assertEqual(len(input_ids), len(loss_mask),
                f"input_ids len ({len(input_ids)}) != loss_mask len ({len(loss_mask)})")

            # input_ids should end with EOD
            from data.tokenize_utils import EOD_TOKEN_ID
            self.assertEqual(input_ids[-1].item(), EOD_TOKEN_ID,
                f"Sample {i}: last token should be EOD ({EOD_TOKEN_ID}), "
                f"got {input_ids[-1].item()}")

            # Loss mask values should be 0.0 or 1.0
            unique_mask = loss_mask.unique()
            for v in unique_mask:
                self.assertIn(v.item(), [0.0, 1.0],
                    f"Sample {i}: loss_mask has value {v.item()}")

            # Loss mask should have some trainable tokens (assistant response)
            trainable_count = int(loss_mask.sum().item())
            self.assertGreater(trainable_count, 0,
                f"Sample {i}: no trainable tokens (loss_mask all zeros)")

            # Image info
            for j, img in enumerate(image_info):
                self.assertIn("pixel_values", img)
                self.assertIn("image_grid_thw", img)
                self.assertIn("start", img)
                self.assertIn("end", img)
                self.assertGreater(img["end"], img["start"],
                    f"Sample {i}, image {j}: end <= start")

            print(f"  Sample {i}: ids_len={len(input_ids)}, "
                  f"trainable={trainable_count}, images={len(image_info)}")

        print("test_05_single_sample_processing PASSED")

    # ===================================================================
    # Test 6: visual_mask consistency with pixel_values
    # ===================================================================
    def test_06_visual_mask_pixel_consistency(self):
        """Verify visual_mask IMG_PAD count matches pixel_values patches after ViT merge.

        Key relationship:
        - pixel_values has shape [total_patches, 1176]
        - total_patches = sum(t * grid_h * grid_w) for all images
        - visual_mask has IMG_PAD_ID at positions = sum(t * (grid_h/2) * (grid_w/2)) tokens
        - After ViT spatial merge (2x2), each image produces t*(grid_h/2)*(grid_w/2) tokens
        - So: visual_mask_count = total_patches / merge_size^2 = total_patches / 4
        """
        from data.understand_dataset import UnderstandPackedDataset
        from data.tokenize_utils import IMG_PAD_ID

        ds = UnderstandPackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=65536,
            rank=0,
            world_size=1,
        )

        for i, sample in enumerate(self.raw_samples[:3]):
            result = ds._process_sample(sample)
            if result is None:
                continue
            if not result["image_info"]:
                continue

            # Count IMG_PAD tokens in input_ids
            input_ids = result["input_ids"]
            img_pad_count = (input_ids == IMG_PAD_ID).sum().item()

            # Count expected visual tokens from image_info
            from data.image_processing import estimate_visual_tokens
            expected_tokens = 0
            total_patches = 0
            for img in result["image_info"]:
                gthw = img["image_grid_thw"]
                num_tokens, _, _ = estimate_visual_tokens(gthw)
                expected_tokens += num_tokens
                t, gh, gw = gthw[0].tolist() if gthw.dim() == 2 else gthw.tolist()
                total_patches += t * gh * gw

            self.assertEqual(img_pad_count, expected_tokens,
                f"Sample {i}: IMG_PAD count ({img_pad_count}) != "
                f"expected visual tokens ({expected_tokens})")

            # Verify: total_patches / 4 == expected_tokens
            self.assertEqual(total_patches // 4, expected_tokens,
                f"Sample {i}: patches/4 ({total_patches//4}) != tokens ({expected_tokens})")

            print(f"  Sample {i}: {img_pad_count} IMG_PAD tokens, "
                  f"{total_patches} patches (patches/4 == tokens ✓)")

        print("test_06_visual_mask_pixel_consistency PASSED")

    # ===================================================================
    # Test 7: Packing correctness
    # ===================================================================
    def test_07_packing(self):
        """Test packed output has correct structure."""
        from data.understand_dataset import UnderstandPackedDataset
        from data.tokenize_utils import EOD_TOKEN_ID, IMG_PAD_ID

        seq_length = 65536
        ds = UnderstandPackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=seq_length,
            rank=0,
            world_size=1,
        )

        # Get first packed batch
        batch = None
        for b in ds:
            batch = b
            break

        if batch is None:
            self.skipTest("No batch produced from dataset")

        # Check all expected keys
        expected_keys = {
            "input_ids", "labels", "loss_mask", "position_ids",
            "cu_seqlens", "pixel_values", "image_grid_thw", "visual_mask",
        }
        self.assertEqual(set(batch.keys()), expected_keys,
            f"Missing/extra keys: {set(batch.keys()) ^ expected_keys}")

        # Shape checks (all should be seq_length after causal shift)
        self.assertEqual(batch["input_ids"].shape, (seq_length,),
            f"input_ids shape {batch['input_ids'].shape} != ({seq_length},)")
        self.assertEqual(batch["labels"].shape, (seq_length,))
        self.assertEqual(batch["loss_mask"].shape, (seq_length,))
        self.assertEqual(batch["position_ids"].shape, (seq_length,))
        self.assertEqual(batch["visual_mask"].shape, (seq_length,))

        # visual_mask should be boolean
        self.assertEqual(batch["visual_mask"].dtype, torch.bool)

        # visual_mask positions should correspond to IMG_PAD_ID in input_ids
        mask_from_ids = (batch["input_ids"] == IMG_PAD_ID)
        self.assertTrue(torch.equal(batch["visual_mask"], mask_from_ids),
            "visual_mask doesn't match IMG_PAD positions in input_ids")

        # pixel_values: [total_patches, 1176] or empty
        pv = batch["pixel_values"]
        if pv.numel() > 0:
            self.assertEqual(pv.dim(), 2, f"pixel_values should be 2D, got {pv.dim()}D")
            self.assertEqual(pv.shape[1], 1176,
                f"pixel_values dim1={pv.shape[1]}, expected 1176")

            # image_grid_thw should be [num_images, 3]
            gthw = batch["image_grid_thw"]
            self.assertEqual(gthw.dim(), 2)
            self.assertEqual(gthw.shape[1], 3)

            # Verify: total patches in pixel_values matches sum from grid_thw
            total_expected = 0
            for row in gthw:
                t, gh, gw = row.tolist()
                total_expected += t * gh * gw
            self.assertEqual(pv.shape[0], total_expected,
                f"pixel_values patches ({pv.shape[0]}) != grid_thw sum ({total_expected})")

            # Verify: visual_mask True count matches total visual tokens
            from data.image_processing import estimate_visual_tokens
            total_visual_tokens = 0
            for row in gthw:
                row_3d = row.unsqueeze(0)  # [1, 3]
                nt, _, _ = estimate_visual_tokens(row_3d)
                total_visual_tokens += nt
            mask_count = batch["visual_mask"].sum().item()
            self.assertEqual(mask_count, total_visual_tokens,
                f"visual_mask True count ({mask_count}) != visual tokens ({total_visual_tokens})")

        # Causal shift: labels should be shifted by 1 relative to a hypothetical full sequence
        # The padding region should have loss_mask = 0
        pad_region_mask = batch["loss_mask"][batch["loss_mask"] == 0]
        self.assertGreater(len(pad_region_mask), 0,
            "Should have some zero loss_mask (at least padding)")

        # cu_seqlens should be monotonically increasing, starting at 0
        cu = batch["cu_seqlens"]
        self.assertEqual(cu[0].item(), 0)
        for j in range(1, len(cu)):
            self.assertGreaterEqual(cu[j].item(), cu[j-1].item(),
                f"cu_seqlens not monotonic at index {j}")
        self.assertLessEqual(cu[-1].item(), seq_length)

        # position_ids: should reset per sample in the pack
        # First position in each sample should be 0
        for j in range(len(cu) - 1):
            start = cu[j].item()
            end = cu[j+1].item()
            if end > start:
                self.assertEqual(batch["position_ids"][start].item(), 0,
                    f"position_ids[{start}] should be 0 (start of sample {j})")

        print(f"  Packed batch: input_ids={batch['input_ids'].shape}, "
              f"pixel_values={pv.shape if pv.numel() > 0 else 'empty'}, "
              f"num_images={batch['image_grid_thw'].shape[0] if pv.numel() > 0 else 0}, "
              f"visual_mask_count={batch['visual_mask'].sum().item()}, "
              f"trainable_tokens={int(batch['loss_mask'].sum().item())}, "
              f"cu_seqlens_len={len(cu)}")

        print("test_07_packing PASSED")

    # ===================================================================
    # Test 8: Loss mask excludes special tokens
    # ===================================================================
    def test_08_loss_mask_special_tokens(self):
        """Verify special tokens (131085:131125) have loss_mask=0."""
        from data.understand_dataset import UnderstandPackedDataset
        from data.tokenize_utils import SPECIAL_NO_LOSS_START, SPECIAL_NO_LOSS_END

        ds = UnderstandPackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=65536,
            rank=0,
            world_size=1,
        )

        for batch in ds:
            input_ids = batch["input_ids"]
            loss_mask = batch["loss_mask"]

            # Check that special tokens have zero loss mask
            special_mask = (input_ids >= SPECIAL_NO_LOSS_START) & (input_ids < SPECIAL_NO_LOSS_END)
            if special_mask.any():
                special_loss = loss_mask[special_mask]
                self.assertTrue((special_loss == 0).all(),
                    f"Some special tokens have non-zero loss mask! "
                    f"Non-zero count: {(special_loss != 0).sum().item()}")
                print(f"  {special_mask.sum().item()} special tokens, all have loss_mask=0 ✓")
            break  # Only check first batch

        print("test_08_loss_mask_special_tokens PASSED")

    # ===================================================================
    # Test 9: DataLoader batch dimension
    # ===================================================================
    def test_09_dataloader_batch_dim(self):
        """Verify DataLoader(batch_size=1) adds correct batch dimension.

        This is the bug that caused 'too many values to unpack' — DataLoader
        adds a batch dim to all tensors, including variable-length pixel_values
        and image_grid_thw. The train_step must squeeze(0) them.
        """
        from data.understand_dataset import UnderstandPackedDataset
        from torch.utils.data import DataLoader

        ds = UnderstandPackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=65536,
            rank=0,
            world_size=1,
        )

        dl = DataLoader(ds, batch_size=1, num_workers=0)

        for batch in dl:
            # Fixed-shape tensors get batch dim
            self.assertEqual(batch["input_ids"].dim(), 2)  # [1, seq_len]
            self.assertEqual(batch["input_ids"].shape[0], 1)

            # Variable-shape tensors also get batch dim
            if batch["pixel_values"].numel() > 0:
                self.assertEqual(batch["pixel_values"].dim(), 3,
                    f"pixel_values should be 3D with DataLoader, got {batch['pixel_values'].dim()}D")
                self.assertEqual(batch["pixel_values"].shape[0], 1,
                    "pixel_values batch dim should be 1")

                self.assertEqual(batch["image_grid_thw"].dim(), 3,
                    f"image_grid_thw should be 3D with DataLoader, got {batch['image_grid_thw'].dim()}D")
                self.assertEqual(batch["image_grid_thw"].shape[0], 1)

                # After squeeze(0): should match raw dataset output
                pv = batch["pixel_values"].squeeze(0)
                gthw = batch["image_grid_thw"].squeeze(0)
                self.assertEqual(pv.dim(), 2)
                self.assertEqual(gthw.dim(), 2)
                self.assertEqual(gthw.shape[1], 3)

                # grid_thw should have correct format for ViT: each row is [t, h, w]
                for row in gthw:
                    t, h, w = row.tolist()
                    self.assertGreater(t, 0)
                    self.assertGreater(h, 0)
                    self.assertGreater(w, 0)

                print(f"  pixel_values: raw={pv.shape}, "
                      f"grid_thw: raw={gthw.shape}")

            break  # Only check first batch

        print("test_09_dataloader_batch_dim PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
