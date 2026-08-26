#!/usr/bin/env python3
"""Compare training-time vs predict_labels.py-time prompt construction for
gemma4_unified, CPU-only (no model load), to check whether the two code
paths actually feed different image conditioning into the processor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, "/workspace")
from data.schema import render_policy_prompt  # noqa: E402

HF_ID = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

IMAGE_A = "/workspace/saie-data/raw/benchmark_v1_3/warehouse_s196665/frames/bench_seed09001/spot_cam/rgb_0000.png"
IMAGE_B = "/workspace/saie-data/raw/benchmark_v1_3/warehouse_s196665/frames/bench_seed09001/spot_cam/rgb_1785.png"
INSTRUCTION = "Start by walking ahead to the cement bag stack on the pallet and count the bags."


def main() -> int:
    processor = AutoProcessor.from_pretrained(HF_ID, revision=REVISION, trust_remote_code=False)
    prompt = render_policy_prompt(INSTRUCTION, 1)

    img_a = Image.open(IMAGE_A).convert("RGB")
    img_b = Image.open(IMAGE_B).convert("RGB")

    def training_style(image):
        conversation = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        prefix_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = processor(text=[prefix_text], images=[[image]], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
        return encoded

    def predict_labels_style(image):
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, enable_thinking=False
        )
        return inputs

    ta = training_style(img_a)
    tb = training_style(img_b)
    pa = predict_labels_style(img_a)
    pb = predict_labels_style(img_b)

    print("=== training-style ===")
    print("A input_ids shape:", ta["input_ids"].shape, "pixel_values" in ta, ta.get("pixel_values", torch.tensor([])).shape if "pixel_values" in ta else None)
    print("B input_ids shape:", tb["input_ids"].shape, "pixel_values" in tb, tb.get("pixel_values", torch.tensor([])).shape if "pixel_values" in tb else None)
    print("A vs B input_ids identical:", torch.equal(ta["input_ids"], tb["input_ids"]) if ta["input_ids"].shape == tb["input_ids"].shape else "shape differs (expected if diff image)")
    if "pixel_values" in ta and "pixel_values" in tb and ta["pixel_values"].shape == tb["pixel_values"].shape:
        print("A vs B pixel_values identical:", torch.equal(ta["pixel_values"], tb["pixel_values"]))

    print("\n=== predict_labels-style ===")
    print("A input_ids shape:", pa["input_ids"].shape, "pixel_values" in pa, pa.get("pixel_values", torch.tensor([])).shape if "pixel_values" in pa else None)
    print("B input_ids shape:", pb["input_ids"].shape, "pixel_values" in pb, pb.get("pixel_values", torch.tensor([])).shape if "pixel_values" in pb else None)
    print("A vs B input_ids identical:", torch.equal(pa["input_ids"], pb["input_ids"]) if pa["input_ids"].shape == pb["input_ids"].shape else "shape differs (expected if diff image)")
    if "pixel_values" in pa and "pixel_values" in pb and pa["pixel_values"].shape == pb["pixel_values"].shape:
        print("A vs B pixel_values identical:", torch.equal(pa["pixel_values"], pb["pixel_values"]))

    print("\n=== cross-check: training-style A vs predict_labels-style A (same image, should match) ===")
    if ta["input_ids"].shape == pa["input_ids"].shape:
        print("input_ids identical:", torch.equal(ta["input_ids"], pa["input_ids"]))
    else:
        print("input_ids SHAPE DIFFERS:", ta["input_ids"].shape, "vs", pa["input_ids"].shape)
    for key in ("pixel_values", "pixel_values_videos", "token_type_ids", "mm_token_type_ids"):
        in_t, in_p = key in ta, key in pa
        print(f"key={key}: in training-style={in_t}, in predict_labels-style={in_p}", end="")
        if in_t and in_p:
            same_shape = ta[key].shape == pa[key].shape
            print(f", shapes {'match' if same_shape else 'DIFFER: ' + str(ta[key].shape) + ' vs ' + str(pa[key].shape)}", end="")
            if same_shape:
                print(f", identical={torch.equal(ta[key], pa[key])}", end="")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
