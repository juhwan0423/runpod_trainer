#!/usr/bin/env python3
"""Reproduce predict_labels.py's exact generate() call for a couple of real
examples, but print the FULL output (before slicing at input_length) so we
can see whether the corruption ("1navigate]...", "1] stop", etc. seen in
test_predictions.jsonl) comes from a slicing offset bug or from the model
itself generating a malformed continuation.
"""
from __future__ import annotations

import sys

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

sys.path.insert(0, "/workspace/gemma4_diag")
from schema import render_policy_prompt  # noqa: E402

CHECKPOINT = "/workspace/gemma4_diag/final"

EXAMPLES = [
    {
        "image": "/workspace/gemma4_diag/images/rgb_0000.png",
        "instruction": "Start by walking ahead to the cement bag stack on the pallet and count the bags.",
        "gold": "[navigate] turn right 15 degree",
    },
    {
        "image": "/workspace/gemma4_diag/images/rgb_1785.png",
        "instruction": "Start by walking ahead to the pressure gauge mounted on the pump and read its dial.",
        "gold": "[navigate] turn left 15 degree",
    },
]


def main() -> int:
    processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        CHECKPOINT, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False
    ).cuda().eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    for example in EXAMPLES:
        image = Image.open(example["image"]).convert("RGB")
        prompt = render_policy_prompt(example["instruction"], 1)
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False,
        ).to(model.device)
        input_length = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)

        print("=" * 70)
        print("gold:", example["gold"])
        print("input_length (prefix token count):", input_length)
        print("full output shape:", output.shape)
        print("output[0] length:", output.shape[-1])
        print()

        tokenizer = getattr(processor, "tokenizer", processor)

        # Decode the FULL output (prompt + generation), with and without
        # special tokens, so we can see exactly where the real generated
        # continuation begins relative to input_length.
        full_with_specials = tokenizer.decode(output[0], skip_special_tokens=False)
        print("--- full output, last 200 chars (prompt tail + generation), WITH specials ---")
        print(repr(full_with_specials[-200:]))
        print()

        # Token-by-token around the boundary: print the last 5 prompt tokens
        # and first 10 generated tokens individually.
        print("--- tokens around input_length boundary ---")
        for i in range(max(0, input_length - 5), min(output.shape[-1], input_length + 10)):
            tok_id = int(output[0, i])
            tok_str = tokenizer.decode([tok_id], skip_special_tokens=False)
            marker = "  <-- input_length" if i == input_length else ""
            print(f"  pos={i:4d} id={tok_id:8d} tok={tok_str!r}{marker}")

        # What predict_labels.py actually slices:
        sliced = tokenizer.decode(output[0, input_length:], skip_special_tokens=False)
        print()
        print("predict_labels.py-style slice (output[0, input_length:]):", repr(sliced))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
