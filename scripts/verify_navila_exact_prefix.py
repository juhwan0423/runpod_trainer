#!/usr/bin/env python3
"""Exact-prefix check for NaVILA's llama_3 template against the pinned tokenizer.

Deliberately does NOT import the `llava` package -- reimplements just the
two functions under test (the llama_3_chat string builder from
llava/conversation.py, and tokenizer_image_token's chunk-and-splice
tokenization from llava/mm_utils.py) so this can run with nothing but
`transformers` + `huggingface_hub` installed, before the full patched
NaVILA venv (scripts/setup_navila_venv.sh) exists. Only the tokenizer
subfolder is downloaded (a few MB), not the ~16 GB of model weights.

This does not gate models/registry.py's NaVILAUnifiedAdapter.encode_batch,
which asserts the same invariant itself, every batch, using the *real*
llava.mm_utils.tokenizer_image_token -- this script is a fast standalone
sanity check to run once per pinned revision, from a laptop or CI, without a
GPU or the full venv.

Usage:
    python3 scripts/verify_navila_exact_prefix.py [path/to/navila.yaml]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ACTION_LABELS = (
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
)

SYSTEM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful language and vision assistant. You are able to understand the visual "
    "content that the user provides, and assist the user with a variety of tasks using natural "
    "language."
)
ROLE_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
ROLE_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
SEP = "<|eot_id|>"


def render_policy_prompt(mission: str, num_frames: int) -> str:
    sys.path.insert(0, str(ROOT))
    from data.schema import render_policy_prompt as _render

    return _render(mission, num_frames)


def build_prefix_text(mission: str, num_frames: int) -> str:
    image_tags = "<image>\n" * num_frames
    user_text = image_tags + render_policy_prompt(mission, num_frames)
    return SYSTEM + SEP + ROLE_USER + user_text + SEP + ROLE_ASSISTANT


def tokenizer_image_token(prompt: str, tokenizer, image_token_index: int = -200) -> list[int]:
    prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(x, sep):
        return [ele for sublist in zip(x, [sep] * len(x)) for ele in sublist][:-1]

    input_ids: list[int] = []
    for chunk in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(chunk)
    return input_ids


def load_model_cfg(config_path: Path) -> dict:
    import yaml

    return yaml.safe_load(config_path.read_text())


def main() -> int:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs/models/navila.yaml"
    model_cfg = load_model_cfg(config_path)
    hf_id = model_cfg["hf_id"]
    revision = model_cfg.get("revision")
    if not revision:
        raise SystemExit(f"{config_path} does not pin model.revision")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    local_dir = snapshot_download(repo_id=hf_id, revision=revision, allow_patterns=["llm/tokenizer*", "llm/special_tokens_map.json", "llm/config.json"])
    tokenizer = AutoTokenizer.from_pretrained(str(Path(local_dir) / "llm"), use_fast=True)
    print(f"hf_id={hf_id} revision={revision}")
    print(f"tokenizer class={type(tokenizer).__name__} vocab_size={tokenizer.vocab_size}")
    print(f"bos={tokenizer.bos_token!r} eos={tokenizer.eos_token!r} pad={tokenizer.pad_token!r}")
    if tokenizer.pad_token_id is None:
        print("FAIL: tokenizer has no pad_token_id -- navila_unified.load() will refuse to start")
        return 1
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    print(f"<|eot_id|> id={eot_id}")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        print("FAIL: <|eot_id|> is not a recognized token for this tokenizer")
        return 1

    missions = [
        "Patrol the west corridor and report any fallen workers.",
        "Inspect gauge G-14 and continue the patrol route.",
    ]

    failures = 0
    total = 0
    for num_frames in (1, 8):
        for mission in missions:
            for label in ACTION_LABELS:
                total += 1
                prefix_text = build_prefix_text(mission, num_frames)
                full_text = prefix_text + label + SEP

                prefix_ids = tokenizer_image_token(prefix_text, tokenizer)
                full_ids = tokenizer_image_token(full_text, tokenizer)

                if full_ids[: len(prefix_ids)] != prefix_ids:
                    failures += 1
                    mismatch_at = next(
                        i for i in range(min(len(prefix_ids), len(full_ids))) if prefix_ids[i] != full_ids[i]
                    )
                    print(
                        f"MISMATCH frames={num_frames} label={label!r} first_mismatch_index={mismatch_at} "
                        f"prefix_tail={prefix_ids[max(0, mismatch_at - 3):mismatch_at + 1]} "
                        f"full_tail={full_ids[max(0, mismatch_at - 3):mismatch_at + 1]}"
                    )

    print(f"\n{total - failures}/{total} exact-prefix matches")
    if failures:
        print(
            "FAIL: naive string-concat is not prefix-exact for this tokenizer/template. "
            "models/registry.py's NaVILAUnifiedAdapter.encode_batch already asserts this per "
            "batch and will raise RuntimeError at training time -- switch its implementation to "
            "ID-level concatenation (tokenize the answer separately and splice the id list onto "
            "prefix_ids) before training instead of relying on this string-concat path."
        )
        return 1
    print("PASS: naive prefix+answer string-concat is safe for this tokenizer/template/label set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
