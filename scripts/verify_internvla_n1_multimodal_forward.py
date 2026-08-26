#!/usr/bin/env python3
"""Real multimodal forward(labels=None) sanity check for InternVLA-N1.

Run with:
  cd /home/oem6/InternNav && PYTHONPATH=/home/oem6/InternNav \
  CUDA_VISIBLE_DEVICES=1 \
  /data/oem6/SAIE_runtime_envs/internnav-venv/bin/python3 \
    /data/oem4/runpod_trainer/scripts/verify_internvla_n1_multimodal_forward.py
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor

CHECKPOINT = "/data/oem6/InternVLA-N1-DualVLN"
SAMPLE_IMAGE = (
    "/data/oem4/IndustSimBenchmark/benchmark_v1_3/warehouse_s26355/"
    "frames/bench_seed09001/spot_cam/rgb_0000.png"
)


def main() -> int:
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    model = InternVLAN1ForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    processor = AutoProcessor.from_pretrained(CHECKPOINT)

    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What action should the robot take?"},
            ],
        }
    ]
    prefix_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prefix_text + "[navigate] turn left 15 degree" + "<|im_end|>\n"

    common = dict(return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    full_inputs = processor(text=[full_text], images=[image], **common).to(model.device)
    prefix_inputs = processor(text=[prefix_text], images=[image], **common).to(model.device)

    print("full input_ids shape:", full_inputs["input_ids"].shape)
    print("prefix input_ids shape:", prefix_inputs["input_ids"].shape)
    print("pixel_values shape:", full_inputs.get("pixel_values", "MISSING").shape if "pixel_values" in full_inputs else "MISSING")
    print("image_grid_thw:", full_inputs.get("image_grid_thw", "MISSING"))
    print(
        "full pixel_values == prefix pixel_values:",
        torch.equal(full_inputs["pixel_values"], prefix_inputs["pixel_values"]),
    )

    full_ids = full_inputs["input_ids"][0].tolist()
    prefix_ids = prefix_inputs["input_ids"][0].tolist()
    print("exact-prefix holds with real image tokens:", full_ids[: len(prefix_ids)] == prefix_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=full_inputs["input_ids"],
            attention_mask=full_inputs["attention_mask"],
            pixel_values=full_inputs["pixel_values"],
            image_grid_thw=full_inputs["image_grid_thw"],
            labels=None,
        )
    print("forward OK. logits shape:", outputs.logits.shape, "loss:", outputs.loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
