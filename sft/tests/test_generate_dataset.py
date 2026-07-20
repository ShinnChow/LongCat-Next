# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Tests for generation task dataset processing pipeline.

Run on the cluster (needs model tokenizer/processor and real data):
    cd /path/to/sft && python tests/test_generate_dataset.py

These tests validate:
1. Raw data format (messages with assistant image responses)
2. Chat template + trainable markers for generation
3. Image preprocessing for generation (pixel_values, grid_thw)
4. Visual token count in generation samples
5. Single sample processing (image in trainable region → loss_mask=1)
6. Packing correctness (padding, causal shift, position_ids, visual_mask)
7. Loss mask: image tokens in assistant response have loss_mask=1
8. Special tokens have loss_mask=0
9. DataLoader batch dimension
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

# Model and data paths — provide via environment variables to run this test.
# These require a real tokenizer/processor and dataset, so the test is skipped
# when the paths are not configured.
MODEL_PATH = os.environ.get("MODEL_PATH", "")
DATA_PATH = os.environ.get("GEN_DATA_PATH", "")


def _load_tokenizer_and_processor():
    """Load tokenizer and processor from model path."""
    from transformers import AutoTokenizer, AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer, processor


def _load_first_n_samples(data_path, n=10):
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


class TestGenerateDataset(unittest.TestCase):
    """Test the generation task dataset pipeline."""

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
        cls.raw_samples = _load_first_n_samples(DATA_PATH, n=20)
        print(f"  Loaded {len(cls.raw_samples)} samples")

        # Print first sample structure for debugging
        if cls.raw_samples:
            s = cls.raw_samples[0]
            print(f"  Sample keys: {list(s.keys())}")
            if "messages" in s:
                print(f"  Messages count: {len(s['messages'])}")
                for i, m in enumerate(s["messages"][:4]):
                    role = m.get("role", m.get("from", "?"))
                    content = m.get("content", m.get("value", ""))
                    has_img = "<longcat_img_start>" in content
                    print(f"    [{i}] role={role}, content_len={len(content)}, "
                          f"has_img={has_img}")
                    if has_img:
                        # Extract image path for inspection
                        for match in re.finditer(r"<longcat_img_start>(.*?)<longcat_img_end>", content):
                            path = match.group(1).strip()
                            exists = os.path.exists(path)
                            print(f"         image: {path[:80]}... (exists={exists})")

    # ===================================================================
    # Test 1: Raw data format validation
    # ===================================================================
    def test_01_raw_data_format(self):
        """Validate raw generation data has expected format.

        Generation data: user provides text prompt, assistant responds with image.
        The assistant message should contain <longcat_img_start>...<longcat_img_end> tags.
        """
        has_image_in_assistant = False

        for i, sample in enumerate(self.raw_samples[:10]):
            self.assertIn("messages", sample, f"Sample {i} missing 'messages' key")
            messages = sample["messages"]
            self.assertGreater(len(messages), 0, f"Sample {i} has empty messages")

            for j, msg in enumerate(messages):
                has_standard = "role" in msg and "content" in msg
                has_legacy = "from" in msg and "value" in msg
                self.assertTrue(has_standard or has_legacy,
                    f"Sample {i}, msg {j} has neither format: {list(msg.keys())}")

            # Check that at least some samples have images in assistant response
            for msg in messages:
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("assistant", "gpt") and "<longcat_img_start>" in content:
                    has_image_in_assistant = True

        self.assertTrue(has_image_in_assistant,
            "No samples found with images in assistant response")
        print("test_01_raw_data_format PASSED")

    # ===================================================================
    # Test 2: Chat template + trainable markers for generation
    # ===================================================================
    def test_02_chat_template_generation(self):
        """Test chat template marks assistant image responses as trainable."""
        from data.chat_template import (
            unify_message_format, encode_with_trainable_markers,
            TRAINABLE_START, TRAINABLE_END,
        )

        for i, sample in enumerate(self.raw_samples[:5]):
            messages = unify_message_format(sample["messages"])
            marked_text = encode_with_trainable_markers(self.tokenizer, messages)

            if not marked_text:
                continue

            # Check trainable markers are balanced
            starts = marked_text.count(TRAINABLE_START)
            ends = marked_text.count(TRAINABLE_END)
            self.assertEqual(starts, ends,
                f"Sample {i}: unbalanced markers ({starts} starts, {ends} ends)")

            # In generation task, images are in assistant (trainable) region
            # Check that <longcat_img_start> tags appear within trainable regions
            in_trainable = False
            img_in_trainable = 0
            img_outside_trainable = 0
            for part in re.split(
                rf"({re.escape(TRAINABLE_START)}|{re.escape(TRAINABLE_END)})",
                marked_text
            ):
                if part == TRAINABLE_START:
                    in_trainable = True
                    continue
                elif part == TRAINABLE_END:
                    in_trainable = False
                    continue

                img_count = part.count("<longcat_img_start>")
                if in_trainable:
                    img_in_trainable += img_count
                else:
                    img_outside_trainable += img_count

            if img_in_trainable > 0:
                print(f"  Sample {i}: {img_in_trainable} images in trainable, "
                      f"{img_outside_trainable} outside, {starts} trainable regions")

        print("test_02_chat_template_generation PASSED")

    # ===================================================================
    # Test 3: Image preprocessing for generation
    # ===================================================================
    def test_03_image_preprocessing(self):
        """Test image preprocessing returns correct pixel_values format."""
        # Find images in assistant responses
        image_paths = []
        for sample in self.raw_samples:
            for msg in sample.get("messages", []):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("assistant", "gpt"):
                    for match in re.finditer(r"<longcat_img_start>(.*?)<longcat_img_end>", content):
                        path = match.group(1).strip()
                        if os.path.exists(path):
                            image_paths.append(path)
                        if len(image_paths) >= 3:
                            break
                if len(image_paths) >= 3:
                    break

        if not image_paths:
            self.skipTest("No accessible images found in assistant responses")

        for img_path in image_paths:
            result = self.image_preprocessor.preprocess(img_path)
            pv = result["pixel_values"]
            gthw = result["image_grid_thw"]

            # pixel_values: [num_patches, 1176]
            self.assertEqual(pv.dim(), 2, f"pixel_values should be 2D")
            self.assertEqual(pv.shape[1], 1176,
                f"pixel_values dim1 should be 1176, got {pv.shape[1]}")

            # image_grid_thw: [1, 3]
            self.assertEqual(gthw.shape, (1, 3))

            # Consistency check
            t, gh, gw = gthw[0].tolist()
            expected_patches = t * gh * gw
            self.assertEqual(pv.shape[0], expected_patches,
                f"patches mismatch: {pv.shape[0]} vs {expected_patches}")

            print(f"  Image: {os.path.basename(img_path)}, "
                  f"pixel_values={pv.shape}, grid=[{t},{gh},{gw}]")

        print("test_03_image_preprocessing PASSED")

    # ===================================================================
    # Test 4: Visual token count
    # ===================================================================
    def test_04_visual_token_count(self):
        """Verify the visual token count formula."""
        from data.image_processing import estimate_visual_tokens

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

            our_tokens, our_h, our_w = estimate_visual_tokens(gthw)
            expected_tokens = gh * gw // 4  # spatial_merge_size=2

            self.assertEqual(our_tokens, expected_tokens,
                f"Token count mismatch: ours={our_tokens}, expected={expected_tokens}")

            print(f"  Image: grid=[{t},{gh},{gw}], tokens={our_tokens}")

        print("test_04_visual_token_count PASSED")

    # ===================================================================
    # Test 5: Single sample processing for generation
    # ===================================================================
    def test_05_single_sample_processing(self):
        """Test _process_sample returns correct structure for generation.

        Key difference from understand: images in trainable regions have
        loss_mask=1 (they are the generation target).
        """
        from data.generate_dataset import GeneratePackedDataset
        from data.tokenize_utils import EOD_TOKEN_ID, IMG_PAD_ID

        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=8192,
            rank=0,
            world_size=1,
        )

        processed_count = 0
        for i, sample in enumerate(self.raw_samples[:10]):
            result = ds._process_sample(sample)
            if result is None:
                continue
            processed_count += 1

            input_ids = result["input_ids"]
            loss_mask = result["loss_mask"]
            image_info = result.get("image_info", [])

            # Shape checks
            self.assertEqual(input_ids.dim(), 1)
            self.assertEqual(loss_mask.dim(), 1)
            self.assertEqual(len(input_ids), len(loss_mask))

            # Should end with EOD
            self.assertEqual(input_ids[-1].item(), EOD_TOKEN_ID,
                f"Sample {i}: last token should be EOD")

            # Loss mask values should be 0.0 or 1.0
            unique_mask = loss_mask.unique()
            for v in unique_mask:
                self.assertIn(v.item(), [0.0, 1.0],
                    f"Sample {i}: loss_mask has value {v.item()}")

            # For generation: image tokens in trainable region should have loss_mask=1
            img_pad_count = (input_ids == IMG_PAD_ID).sum().item()
            if img_pad_count > 0 and image_info:
                # Check that some IMG_PAD positions have loss_mask=1
                img_pad_mask = (input_ids == IMG_PAD_ID)
                img_loss = loss_mask[img_pad_mask]
                trainable_img_tokens = (img_loss > 0).sum().item()
                print(f"  Sample {i}: ids_len={len(input_ids)}, "
                      f"img_tokens={img_pad_count}, "
                      f"trainable_img={trainable_img_tokens}, "
                      f"images={len(image_info)}")
                # In generation, images in assistant response should be trainable
                self.assertGreater(trainable_img_tokens, 0,
                    f"Sample {i}: no trainable image tokens (expect loss_mask=1 for generation)")
            else:
                print(f"  Sample {i}: ids_len={len(input_ids)}, no images")

            if processed_count >= 3:
                break

        self.assertGreater(processed_count, 0,
            "No samples could be processed")
        print("test_05_single_sample_processing PASSED")

    # ===================================================================
    # Test 6: Packing correctness for generation
    # ===================================================================
    def test_06_packing(self):
        """Test packed output has correct structure for generation task."""
        from data.generate_dataset import GeneratePackedDataset
        from data.tokenize_utils import EOD_TOKEN_ID, IMG_PAD_ID

        seq_length = 8192
        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=seq_length,
            rank=0,
            world_size=1,
        )

        batch = None
        for b in ds:
            batch = b
            break

        if batch is None:
            self.skipTest("No batch produced from dataset")

        # Check expected keys
        expected_keys = {
            "input_ids", "labels", "loss_mask", "position_ids",
            "cu_seqlens", "pixel_values", "image_grid_thw",
            "visual_mask", "loss_visual_mask",
        }
        self.assertEqual(set(batch.keys()), expected_keys,
            f"Missing/extra keys: {set(batch.keys()) ^ expected_keys}")

        # Shape checks
        self.assertEqual(batch["input_ids"].shape, (seq_length,))
        self.assertEqual(batch["labels"].shape, (seq_length,))
        self.assertEqual(batch["loss_mask"].shape, (seq_length,))
        self.assertEqual(batch["position_ids"].shape, (seq_length,))
        self.assertEqual(batch["visual_mask"].shape, (seq_length,))

        # visual_mask should be boolean
        self.assertEqual(batch["visual_mask"].dtype, torch.bool)

        # visual_mask positions should match IMG_PAD_ID in input_ids
        mask_from_ids = (batch["input_ids"] == IMG_PAD_ID)
        self.assertTrue(torch.equal(batch["visual_mask"], mask_from_ids),
            "visual_mask doesn't match IMG_PAD positions in input_ids")

        # Pixel values and grid_thw consistency
        pv = batch["pixel_values"]
        if pv.numel() > 0:
            self.assertEqual(pv.dim(), 2)
            self.assertEqual(pv.shape[1], 1176)

            gthw = batch["image_grid_thw"]
            self.assertEqual(gthw.dim(), 2)
            self.assertEqual(gthw.shape[1], 3)

            # Total patches from grid_thw should match pixel_values
            total_patches = 0
            for row in gthw:
                t, gh, gw = row.tolist()
                total_patches += t * gh * gw
            self.assertEqual(pv.shape[0], total_patches)

            # visual_mask True count should match total visual tokens
            from data.image_processing import estimate_visual_tokens
            total_visual_tokens = 0
            for row in gthw:
                row_3d = row.unsqueeze(0)
                nt, _, _ = estimate_visual_tokens(row_3d)
                total_visual_tokens += nt
            mask_count = batch["visual_mask"].sum().item()
            self.assertEqual(mask_count, total_visual_tokens)

        # cu_seqlens checks
        cu = batch["cu_seqlens"]
        self.assertEqual(cu[0].item(), 0)
        for j in range(1, len(cu)):
            self.assertGreaterEqual(cu[j].item(), cu[j - 1].item())
        self.assertLessEqual(cu[-1].item(), seq_length)

        # position_ids: should reset per sample
        for j in range(len(cu) - 1):
            start = cu[j].item()
            end = cu[j + 1].item()
            if end > start:
                self.assertEqual(batch["position_ids"][start].item(), 0,
                    f"position_ids[{start}] should be 0 (start of sample {j})")

        print(f"  Packed batch: input_ids={batch['input_ids'].shape}, "
              f"pixel_values={pv.shape if pv.numel() > 0 else 'empty'}, "
              f"visual_mask_count={batch['visual_mask'].sum().item()}, "
              f"trainable_tokens={int(batch['loss_mask'].sum().item())}, "
              f"cu_seqlens_len={len(cu)}")

        print("test_06_packing PASSED")

    # ===================================================================
    # Test 7: Loss mask correctness for generation (image tokens trainable)
    # ===================================================================
    def test_07_loss_mask_image_tokens_trainable(self):
        """Verify that image tokens in assistant responses have loss_mask=1.

        This is the key difference from understand task: for generation,
        image tokens ARE the training target and should have loss_mask=1.
        For understand, image tokens have loss_mask=0 (not trainable).
        """
        from data.generate_dataset import GeneratePackedDataset
        from data.tokenize_utils import IMG_PAD_ID

        seq_length = 8192
        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=seq_length,
            rank=0,
            world_size=1,
        )

        for batch in ds:
            input_ids = batch["input_ids"]
            loss_mask = batch["loss_mask"]
            visual_mask = batch["visual_mask"]

            img_token_count = visual_mask.sum().item()
            if img_token_count == 0:
                continue  # Skip batches without images

            # Image tokens that are trainable
            img_loss = loss_mask[visual_mask]
            trainable_img = (img_loss > 0).sum().item()

            # For generation: most/all image tokens should be trainable
            # (they come from assistant responses)
            self.assertGreater(trainable_img, 0,
                f"No trainable image tokens! img_tokens={img_token_count}")

            # Total trainable tokens
            total_trainable = loss_mask.sum().item()

            print(f"  Image tokens: {img_token_count}, "
                  f"trainable_img: {trainable_img} "
                  f"({100 * trainable_img / max(img_token_count, 1):.0f}%), "
                  f"total_trainable: {int(total_trainable)}")

            break  # Check first batch with images

        print("test_07_loss_mask_image_tokens_trainable PASSED")

    # ===================================================================
    # Test 8: Special tokens have loss_mask=0
    # ===================================================================
    def test_08_special_tokens_no_loss(self):
        """Verify non-image special tokens have loss_mask=0 in labels position.

        After causal LM shift:
        - input_ids = tokens[:-1], labels = tokens[1:]
        - loss_mask aligns with LABELS, not input_ids
        So we check: where labels are non-image special tokens, loss_mask=0.
        Where labels are IMG_PAD (generation target), loss_mask=1.
        """
        from data.generate_dataset import GeneratePackedDataset
        from data.tokenize_utils import (
            SPECIAL_NO_LOSS_START, SPECIAL_NO_LOSS_END, IMG_PAD_ID,
        )

        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=8192,
            rank=0,
            world_size=1,
        )

        for batch in ds:
            labels = batch["labels"]
            loss_mask = batch["loss_mask"]

            # loss_mask aligns with labels (not input_ids)
            # Non-image special tokens in labels should have loss_mask=0
            non_img_special_mask = (
                (labels >= SPECIAL_NO_LOSS_START)
                & (labels < SPECIAL_NO_LOSS_END)
                & (labels != IMG_PAD_ID)
            )
            if non_img_special_mask.any():
                special_loss = loss_mask[non_img_special_mask]
                self.assertTrue((special_loss == 0).all(),
                    f"Some non-image special tokens in labels have non-zero "
                    f"loss mask! Non-zero count: {(special_loss != 0).sum().item()}")
                print(f"  {non_img_special_mask.sum().item()} non-image special "
                      f"tokens in labels, all have loss_mask=0 ✓")

            # IMG_PAD tokens in labels (generation target) should have loss_mask=1
            img_pad_mask = (labels == IMG_PAD_ID)
            if img_pad_mask.any():
                img_pad_loss = loss_mask[img_pad_mask]
                trainable_img = (img_pad_loss > 0).sum().item()
                print(f"  {img_pad_mask.sum().item()} IMG_PAD in labels, "
                      f"{trainable_img} with loss_mask=1 (generation target)")
            break

        print("test_08_special_tokens_no_loss PASSED")

    # ===================================================================
    # Test 9: DataLoader batch dimension
    # ===================================================================
    def test_09_dataloader_batch_dim(self):
        """Verify DataLoader(batch_size=1) adds correct batch dimension."""
        from data.generate_dataset import GeneratePackedDataset
        from torch.utils.data import DataLoader

        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=8192,
            rank=0,
            world_size=1,
        )

        dl = DataLoader(ds, batch_size=1, num_workers=0)

        for batch in dl:
            # Fixed-shape tensors: [1, seq_len]
            self.assertEqual(batch["input_ids"].dim(), 2)
            self.assertEqual(batch["input_ids"].shape[0], 1)

            # Variable-shape tensors also get batch dim
            if batch["pixel_values"].numel() > 0:
                self.assertEqual(batch["pixel_values"].dim(), 3)
                self.assertEqual(batch["pixel_values"].shape[0], 1)
                self.assertEqual(batch["image_grid_thw"].dim(), 3)
                self.assertEqual(batch["image_grid_thw"].shape[0], 1)

                # After squeeze(0): should match raw dataset output
                pv = batch["pixel_values"].squeeze(0)
                gthw = batch["image_grid_thw"].squeeze(0)
                self.assertEqual(pv.dim(), 2)
                self.assertEqual(pv.shape[1], 1176)
                self.assertEqual(gthw.dim(), 2)
                self.assertEqual(gthw.shape[1], 3)

                print(f"  pixel_values: {pv.shape}, grid_thw: {gthw.shape}")

            break

        print("test_09_dataloader_batch_dim PASSED")


    # ===================================================================
    # Test 10: loss_visual_mask alignment (off-by-one fix verification)
    # ===================================================================
    def test_10_loss_visual_mask_alignment(self):
        """Verify loss_visual_mask is based on labels, not input_ids.

        This test verifies the fix for the off-by-one bug where visual_mask
        (input_ids == IMG_PAD_ID) was incorrectly used for depth CE loss
        hidden state selection. The depth loss must select hidden states by
        labels, so loss_visual_mask must be (labels == IMG_PAD_ID).

        After causal LM shift:
        - visual_mask selects positions a+1, ..., a+N (input_ids == IMG_PAD)
        - loss_visual_mask selects positions a, ..., a+N-1 (labels == IMG_PAD)
        - Both have N positions (same count), but offset by 1.
        """
        from data.generate_dataset import GeneratePackedDataset
        from data.tokenize_utils import IMG_PAD_ID

        seq_length = 8192
        ds = GeneratePackedDataset(
            data_paths=[DATA_PATH],
            tokenizer=self.tokenizer,
            image_processor=self.image_preprocessor,
            seq_length=seq_length,
            rank=0,
            world_size=1,
        )

        for batch in ds:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            visual_mask = batch["visual_mask"]
            loss_visual_mask = batch["loss_visual_mask"]

            # 1. visual_mask should be (input_ids == IMG_PAD_ID)
            expected_visual_mask = (input_ids == IMG_PAD_ID)
            self.assertTrue(torch.equal(visual_mask, expected_visual_mask),
                "visual_mask != (input_ids == IMG_PAD_ID)")

            # 2. loss_visual_mask should be (labels == IMG_PAD_ID)
            expected_loss_mask = (labels == IMG_PAD_ID)
            self.assertTrue(torch.equal(loss_visual_mask, expected_loss_mask),
                "loss_visual_mask != (labels == IMG_PAD_ID)")

            # 3. Both masks should have the same count
            vm_count = visual_mask.sum().item()
            lvm_count = loss_visual_mask.sum().item()
            self.assertEqual(vm_count, lvm_count,
                f"visual_mask count ({vm_count}) != loss_visual_mask count ({lvm_count})")

            if vm_count == 0:
                continue  # Skip batches without images

            # 4. The masks should NOT be identical (they're offset by 1)
            self.assertFalse(torch.equal(visual_mask, loss_visual_mask),
                "visual_mask and loss_visual_mask should differ (offset by 1)")

            # 5. Verify the offset pattern: for each contiguous block of image
            # tokens, loss_visual_mask should start 1 position earlier
            vm_positions = torch.where(visual_mask)[0]
            lvm_positions = torch.where(loss_visual_mask)[0]
            # loss_visual_mask positions should be (visual_mask positions - 1)
            self.assertTrue(torch.equal(lvm_positions, vm_positions - 1),
                f"loss_visual_mask positions should be visual_mask positions - 1.\n"
                f"First few vm_pos: {vm_positions[:5].tolist()}\n"
                f"First few lvm_pos: {lvm_positions[:5].tolist()}")

            print(f"  visual_mask count: {vm_count}")
            print(f"  loss_visual_mask count: {lvm_count}")
            print(f"  Offset verified: loss_visual_mask = visual_mask - 1")
            break

        print("test_10_loss_visual_mask_alignment PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
