#!/usr/bin/env python3
"""Generate free-text labels from a trained checkpoint into evaluation JSONL."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import select_history  # noqa: E402
from data.schema import ACTION_LABELS, PROMPT_VERSION, read_jsonl, render_policy_prompt  # noqa: E402
from models.registry import get_adapter  # noqa: E402


WRAPPER_VERSION = "saie_gemma4_predict_v2"


def canonical_from_raw(raw: str | None) -> tuple[str | None, str]:
    text = str(raw or "").strip()
    if text in ACTION_LABELS:
        return text, "exact"
    matches = [label for label in ACTION_LABELS if label in text]
    return (matches[0], "extracted") if len(matches) == 1 else (None, "invalid")


def _parsed_content(value: Any) -> str:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"parse_response returned {type(value).__name__}, expected dict")
    content = value.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [block.get("text") for block in content if isinstance(block, dict) and "text" in block]
        if texts and len(texts) == len(content):
            return "".join(str(text) for text in texts).strip()
    raise ValueError(f"parse_response content is not plain text: {content!r}")


def parse_native_response(processor, generated_ids: torch.Tensor, prefix_ids: torch.Tensor):
    """Preserve raw special tokens, then parse only the assistant content."""

    tokenizer = getattr(processor, "tokenizer", processor)
    raw_with_specials = tokenizer.decode(generated_ids, skip_special_tokens=False)
    parser = getattr(processor, "parse_response", None)
    if parser is None:
        return raw_with_specials, None, "processor has no parse_response"
    try:
        parameters = inspect.signature(parser).parameters
        if "prefix" in parameters:
            parsed = parser(raw_with_specials, prefix=prefix_ids)
        else:
            parsed = parser(raw_with_specials)
        return raw_with_specials, _parsed_content(parsed), None
    except Exception as exc:
        return raw_with_specials, None, f"{type(exc).__name__}: {exc}"


def checkpoint_sha256(checkpoint: Path) -> str:
    """Hash inference-relevant weights/configs, excluding optimizer state."""

    files = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and (path.suffix in {".json", ".safetensors", ".model"})
    )
    if not files:
        raise FileNotFoundError(f"no inference checkpoint files found under {checkpoint}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(checkpoint).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--vlm-output-dir",
        type=Path,
        help="also export <warehouse>/<episode>.json lists accepted by the SAIE judge",
    )
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--allow-extracted-label",
        action="store_true",
        help="diagnostic only: accept one canonical substring surrounded by extra text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = checkpoint_sha256(checkpoint)
    is_lora = (checkpoint / "adapter_config.json").is_file()
    if is_lora:
        adapter = get_adapter(model_cfg["adapter"], model_cfg)
        model, processor = adapter.load()
        from peft import PeftModel
        from transformers import AutoProcessor

        model = PeftModel.from_pretrained(model, checkpoint)
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False)
    else:
        load_cfg = dict(model_cfg)
        load_cfg["hf_id"] = str(checkpoint)
        adapter = get_adapter(load_cfg["adapter"], load_cfg)
        model, processor = adapter.load()
    if not torch.cuda.is_available():
        raise RuntimeError("predict_labels.py requires a CUDA GPU")
    model.eval().cuda()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    rows = read_jsonl(args.jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("prediction input is empty after applying --limit")
    normalization_rule = (
        "single_canonical_substring_v1"
        if args.allow_extracted_label
        else "strict_exact_canonical_v1"
    )
    image_root = args.image_root.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    episode_outputs: dict[tuple[str, str], list[dict]] = {}
    with temporary.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            refs = row.get("images") or row.get("image_history") or [row["image"]]
            refs = select_history(list(refs), args.num_frames)
            images = []
            for ref in refs:
                path = Path(ref)
                path = path if path.is_absolute() else image_root / path
                with Image.open(path) as image:
                    images.append(image.convert("RGB").copy())
            prompt = render_policy_prompt(row["instruction"], len(images))
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": prompt})
            conversation = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(model.device)
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            raw_model_output, native_action, parse_error = parse_native_response(
                processor, output[0, input_length:], inputs["input_ids"][0]
            )
            normalized, parse_status = canonical_from_raw(native_action)
            if parse_error:
                parse_status = "parse_error"
            accepted = normalized if (
                parse_status == "exact"
                or (args.allow_extracted_label and parse_status == "extracted")
            ) else None
            selected_frame_index = row.get("frame_id")
            if not isinstance(selected_frame_index, int):
                selected_frame_index = row.get("frame_index")
            if not isinstance(selected_frame_index, int):
                raise ValueError(f"row has no numeric frame id: {row.get('sample_id') or row.get('frame_id')}")
            source_frame_index = row.get("source_frame_id")
            if not isinstance(source_frame_index, int):
                source_frame_index = row.get("source_frame_index")
            if not isinstance(source_frame_index, int):
                raise ValueError(
                    "row has no numeric source annotation anchor: "
                    f"{row.get('sample_id') or row.get('frame_id')}"
                )
            warehouse_id = row.get("warehouse_id")
            source_episode_id = row.get("source_episode_id")
            if not warehouse_id or not source_episode_id:
                raise ValueError("prediction export requires warehouse_id and source_episode_id")
            record = {
                "sample_id": row.get("sample_id") or row.get("frame_id"),
                "episode_id": row.get("episode_id"),
                "source_episode_id": source_episode_id,
                "warehouse_id": warehouse_id,
                "frame_id": selected_frame_index,
                "source_frame_id": source_frame_index,
                "gold": row["action_label"],
                "pred": accepted,
                "normalized_action_label": normalized,
                "native_action": native_action,
                "raw": native_action,
                "raw_model_output": raw_model_output,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "wrapper_version": WRAPPER_VERSION,
                "normalization_rule": normalization_rule,
                "used_ground_truth_context": False,
                "prompt_version": PROMPT_VERSION,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "model_requested_revision": model_cfg.get("revision"),
                "model_resolved_commit": getattr(model.config, "_commit_hash", None),
                "decoding_config": {
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                    "use_cache": True,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            episode_outputs.setdefault((warehouse_id, source_episode_id), []).append(
                {
                    # The judge aligns against annotation video_id anchors,
                    # not the nearest saved PNG selected for model input.
                    "frame_id": source_frame_index,
                    "observation_frame_id": selected_frame_index,
                    "t": source_frame_index / 30.0,
                    "label": accepted or "[error] stop",
                    "raw": raw_model_output,
                    "native_action": native_action,
                    "parse_status": parse_status,
                    "normalization_rule": normalization_rule,
                    "used_ground_truth_context": False,
                }
            )
            if (index + 1) % 100 == 0:
                print(f"{index + 1}/{len(rows)}")
    temporary.replace(args.out)
    if args.vlm_output_dir:
        for (warehouse_id, source_episode_id), episode_rows in episode_outputs.items():
            episode_rows.sort(key=lambda value: value["frame_id"])
            destination = args.vlm_output_dir / warehouse_id / f"{source_episode_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(episode_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        export_manifest = {
            "schema_version": "saie_vlm_export_v1",
            "wrapper_version": WRAPPER_VERSION,
            "normalization_rule": normalization_rule,
            "allow_extracted_label": args.allow_extracted_label,
            "used_ground_truth_context": False,
            "prompt_version": PROMPT_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "source_jsonl": str(args.jsonl.resolve()),
            "source_jsonl_sha256": file_sha256(args.jsonl),
            "image_root": str(image_root),
            "num_frames": args.num_frames,
            "prediction_jsonl": str(args.out.resolve()),
            "episodes": len(episode_outputs),
            "rows": len(rows),
        }
        (args.vlm_output_dir / "prediction_export_manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote SAIE judge inputs under {args.vlm_output_dir}")
    status_counts = Counter(
        record["parse_status"]
        for episode_rows in episode_outputs.values()
        for record in episode_rows
    )
    inference_manifest = {
        "schema_version": "saie_prediction_manifest_v1",
        "wrapper_version": WRAPPER_VERSION,
        "normalization_rule": normalization_rule,
        "allow_extracted_label": args.allow_extracted_label,
        "prompt_version": PROMPT_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "model_requested_revision": model_cfg.get("revision"),
        "model_resolved_commit": getattr(model.config, "_commit_hash", None),
        "input_jsonl": str(args.jsonl.resolve()),
        "input_jsonl_sha256": file_sha256(args.jsonl),
        "image_root": str(image_root),
        "num_frames": args.num_frames,
        "output_jsonl": str(args.out.resolve()),
        "rows": len(rows),
        "parse_status_counts": dict(sorted(status_counts.items())),
        "decoding_config": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
    }
    manifest_path = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(inference_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(rows)} rows)")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
