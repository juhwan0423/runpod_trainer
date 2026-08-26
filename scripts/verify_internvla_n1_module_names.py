#!/usr/bin/env python3
"""Empirically dump InternVLA-N1's real named_parameters() prefixes.

Per this project's established lesson (config/doc claims about module names
have been wrong for both Gemma4 and, partially, would-have-been for NaVILA):
never trust a plan document's module-name claims without a live dump against
the actual pinned checkpoint. Run with:

  PYTHONPATH=/home/oem6/InternNav \
  CUDA_VISIBLE_DEVICES=1 \
  /data/oem6/SAIE_runtime_envs/internnav-venv/bin/python3 \
    scripts/verify_internvla_n1_module_names.py
"""

from __future__ import annotations

import sys

import torch

CHECKPOINT = "/data/oem6/InternVLA-N1-DualVLN"


def main() -> int:
    from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM

    print(f"loading {CHECKPOINT} ...", file=sys.stderr)
    model = InternVLAN1ForCausalLM.from_pretrained(
        CHECKPOINT,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print("loaded OK, class:", type(model).__name__)
    print("model.model class:", type(model.model).__name__ if hasattr(model, "model") else "NO .model ATTR")

    names = [n for n, _ in model.named_parameters()]
    print("total parameter tensors:", len(names))

    top2 = sorted(set(".".join(n.split(".")[:2]) for n in names))
    print("\n--- top-2-level prefixes ---")
    for p in top2:
        print(" ", p)

    top3 = sorted(set(".".join(n.split(".")[:3]) for n in names))
    print("\n--- top-3-level prefixes (first 60) ---")
    for p in top3[:60]:
        print(" ", p)

    from collections import defaultdict

    sizes: dict[str, int] = defaultdict(int)
    param_map = dict(model.named_parameters())
    for n in names:
        top = ".".join(n.split(".")[:2])
        sizes[top] += param_map[n].numel()
    total = sum(sizes.values())
    print("\n--- numel by top-2-level prefix ---")
    for t, s in sorted(sizes.items()):
        print(f"  {t}: {s:,} ({100 * s / total:.2f}%)")
    print(f"  TOTAL: {total:,}")

    # Config fields the plan doc flagged as gating whether System-1 modules
    # actually get constructed at all.
    print("\n--- config.system1 ---", getattr(model.config, "system1", "MISSING"))
    print("--- config.use_cache ---", getattr(model.config, "use_cache", "MISSING"))
    print("--- config.tie_word_embeddings ---", getattr(model.config, "tie_word_embeddings", "MISSING"))

    # Does forward() actually crash on labels=None the way the plan expects?
    print("\n--- forward(labels=None) sanity: does .logits exist and require_grad flow work ---")
    dummy_ids = torch.zeros((1, 8), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=dummy_ids.to(model.device), labels=None)
    has_logits = hasattr(out, "logits") and out.logits is not None
    has_loss = getattr(out, "loss", None) is not None
    print(f"  has logits: {has_logits}, shape={tuple(out.logits.shape) if has_logits else None}")
    print(f"  has loss (should be None/absent since labels=None): {has_loss}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
