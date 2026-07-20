# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the LICENSE file.
"""Chat template processing for LongCat-Next SFT training.

Converts raw SFT messages into a single text string, wrapping each assistant
response with <trainable_start>/<trainable_end> markers so the tokenizer step can
build the loss mask (only the assistant text is trained on).
"""

import re
from typing import Dict, List


# LongCat-Next role tokens
ROLE_SYSTEM = "<longcat_system>"
ROLE_USER = "<longcat_user>"
ROLE_ASSISTANT = "<longcat_assistant>"
EOD_TOKEN = "</longcat_s>"

# All role tokens for pattern matching
ALL_ROLE_TOKENS = [ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT]

# Trainable region markers (virtual tokens, not in vocab)
TRAINABLE_START = "<trainable_start>"
TRAINABLE_END = "<trainable_end>"


def unify_message_format(messages: List[Dict]) -> List[Dict]:
    """Unify different message formats into standard {role, content} format.

    Supports:
    - Standard: {"role": "user", "content": "..."}
    - Legacy: {"from": "human", "value": "..."}
    """
    unified = []
    role_map = {"human": "user", "gpt": "assistant"}

    for msg in messages:
        if "role" in msg and "content" in msg:
            unified.append({"role": msg["role"], "content": msg["content"]})
        elif "from" in msg and "value" in msg:
            role = role_map.get(msg["from"], msg["from"])
            unified.append({"role": role, "content": msg["value"]})
        else:
            raise ValueError(f"Unsupported message format: {msg}")

    # Copy assistant_masks if present
    for orig, new in zip(messages, unified):
        if "assistant_masks" in orig:
            new["assistant_masks"] = orig["assistant_masks"]

    return unified


def encode_with_trainable_markers(
    tokenizer,
    messages: List[Dict],
    assistant_prefix: str = ROLE_ASSISTANT,
) -> str:
    """Apply chat template and wrap assistant responses with trainable markers.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template support.
        messages: List of message dicts with "role" and "content" keys.
        assistant_prefix: The role token prefix for assistant messages.

    Returns:
        Text string with <trainable_start>/<trainable_end> markers around
        assistant responses.
    """
    if not messages:
        return ""

    # Validate: no consecutive assistant messages
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "assistant" and messages[i + 1]["role"] == "assistant":
            raise ValueError(f"Consecutive assistant messages at index {i} and {i+1}")

    # Apply chat template to get formatted text (without tokenizing)
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Split by role tokens, keeping them as segment starts
    pattern_str = "|".join(re.escape(t) for t in ALL_ROLE_TOKENS)
    role_pattern = re.compile(rf"(?={pattern_str})", re.DOTALL)
    segments = role_pattern.split(input_text)

    # Identify which segments are assistant responses
    is_assistant_list = []
    for seg in segments:
        is_assistant_list.append(1 if seg.startswith(assistant_prefix) else 0)

    # Handle assistant_masks (per-message mask, 0 means skip loss for that turn)
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    num_assistant_segments = sum(1 for x in is_assistant_list if x == 1)

    if num_assistant_segments == len(assistant_msgs):
        j = 0
        for i in range(len(is_assistant_list)):
            if is_assistant_list[i] == 1:
                mask_val = assistant_msgs[j].get("assistant_masks", 1)
                if mask_val == 0:
                    is_assistant_list[i] = 0
                j += 1

    # Wrap assistant segments with trainable markers
    for i in range(len(segments)):
        if is_assistant_list[i] == 1:
            seg = segments[i]
            # Assistant segment should start with role token and end with EOD
            if seg.startswith(assistant_prefix) and seg.endswith(EOD_TOKEN):
                # Insert trainable markers: keep role token outside, wrap content+eod
                segments[i] = (
                    assistant_prefix
                    + TRAINABLE_START
                    + seg[len(assistant_prefix):]
                    + TRAINABLE_END
                )

    result = "".join(segments)

    # Remove trailing EOD from the last trainable region
    # (it will be added back during sequence packing)
    if result.endswith(TRAINABLE_END):
        suffix = EOD_TOKEN + TRAINABLE_END
        if result.endswith(suffix):
            result = result[: -len(suffix)] + TRAINABLE_END

    return result


def process_sample_messages(tokenizer, data: Dict) -> str:
    """Process a raw data sample's messages into trainable-marked text.

    Args:
        tokenizer: HuggingFace tokenizer.
        data: Raw data dict with "messages" key.

    Returns:
        Text with trainable markers.
    """
    messages = data.get("messages", [])
    if not messages:
        return ""

    messages = unify_message_format(messages)
    return encode_with_trainable_markers(tokenizer, messages)
