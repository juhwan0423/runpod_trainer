#!/usr/bin/env python3
"""Check whether naive prefix/full string tokenization is exact-prefix-safe
for InternVLA-N1's Qwen2.5-VL tokenizer, across all 14 canonical SAIE labels.

Run with:
  PYTHONPATH=/home/oem6/InternNav \
  /data/oem6/SAIE_runtime_envs/internnav-venv/bin/python3 \
    scripts/verify_internvla_n1_exact_prefix.py
"""

from __future__ import annotations

from transformers import AutoProcessor

CHECKPOINT = "/data/oem6/InternVLA-N1-DualVLN"

LABELS = [
    "[navigate] move forward 25 cm",
    "[navigate] move forward 50 cm",
    "[navigate] move forward 75 cm",
    "[navigate] turn left 15 degree",
    "[navigate] turn left 30 degree",
    "[navigate] turn left 45 degree",
    "[navigate] turn right 15 degree",
    "[navigate] turn right 30 degree",
    "[navigate] turn right 45 degree",
    "[navigate] stop",
    "[gauge] stop",
    "[count] stop",
    "[state] stop",
    "[alert] stop",
]

INSTRUCTIONS = [
    "Start by walking ahead to the cement bag stack on the pallet and count the bags.",
    "Walk to your right to the charging station and identify its state.",
]


def main() -> int:
    processor = AutoProcessor.from_pretrained(CHECKPOINT)
    tokenizer = processor.tokenizer

    passed = 0
    failed = []
    for instruction in INSTRUCTIONS:
        for label in LABELS:
            messages = [{"role": "user", "content": instruction}]
            prefix_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prefix_text + label + "<|im_end|>\n"

            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

            ok = full_ids[: len(prefix_ids)] == prefix_ids
            if ok:
                passed += 1
            else:
                failed.append((instruction[:30], label))

    total = len(INSTRUCTIONS) * len(LABELS)
    print(f"{passed}/{total} exact-prefix checks passed")
    if failed:
        print("FAILED cases:")
        for inst, label in failed:
            print(f"  instruction={inst!r} label={label!r}")

    # Also test the ID-level concat strategy (should trivially always work by
    # construction) as a control, and confirm the resulting full sequence
    # decodes back to something sane.
    messages = [{"role": "user", "content": INSTRUCTIONS[0]}]
    prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(LABELS[0] + "<|im_end|>\n", add_special_tokens=False).input_ids
    full_ids_concat = prefix_ids + answer_ids
    decoded_answer = tokenizer.decode(full_ids_concat[len(prefix_ids):], skip_special_tokens=True)
    print(f"\nID-concat control: decoded answer region = {decoded_answer!r} (expect {LABELS[0]!r})")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
