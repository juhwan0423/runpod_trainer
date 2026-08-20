#!/usr/bin/env python3
"""Reproducible SAIE SFT entry point (Gemma 4 Unified first).

The script is location-independent: every relative config/data/output path is
resolved against the cloned repository directory, not the caller's shell cwd.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(base: Path, model: Path, method: Path) -> dict[str, Any]:
    cfg = read_yaml(base)
    cfg = deep_merge(cfg, {"model": read_yaml(model)})
    cfg = deep_merge(cfg, read_yaml(method))
    return cfg


def find_todos(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_todos(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_todos(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and "TODO" in value:
        found.append(prefix)
    return found


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "accelerate",
        "deepspeed",
        "peft",
        "bitsandbytes",
        "Pillow",
        "PyYAML",
    )
    out: dict[str, str | None] = {}
    for package in packages:
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = None
    return out


def git_commit(directory: Path, override_env: str) -> str | None:
    """Return one repository's commit without borrowing a sibling repo SHA."""

    explicit = os.environ.get(override_env)
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_resume(output_dir: Path, request: str) -> str | bool | None:
    from transformers.trainer_utils import get_last_checkpoint

    if request == "none":
        return None
    if request == "auto":
        checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
        return checkpoint or None
    checkpoint = project_path(request)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--deepspeed", default=None, help="path override, or 'none'")
    parser.add_argument("--report_to", default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto (latest checkpoint), none, or an explicit checkpoint directory",
    )
    parser.add_argument(
        "--require-paper-ready",
        action="store_true",
        help="require benchmark_ready (legacy flag name retained for compatibility)",
    )
    parser.add_argument(
        "--require-icra-ready",
        action="store_true",
        help="require the stronger hazard/distractor/risk/gaze scientific gate",
    )
    parser.add_argument("--save_total_limit", type=int)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="smoke-test-only cap; full paper launch leaves validation complete",
    )
    parser.add_argument(
        "--allow-resume-config-mismatch",
        action="store_true",
        help="dangerous escape hatch for an intentional checkpoint/config mismatch",
    )
    parser.add_argument("--dry-run", action="store_true", help="resolve and validate paths without loading a model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_path = project_path(args.base)
    model_path = project_path(args.config)
    method_path = project_path(args.method)
    cfg = load_config(base_path, model_path, method_path)
    todos = find_todos(cfg)
    if todos:
        raise ValueError(f"unfilled TODO fields: {', '.join(todos)}")
    if args.seed not in cfg["run"].get("seeds", [args.seed]):
        raise ValueError(f"seed {args.seed} is outside declared run.seeds={cfg['run'].get('seeds')}")

    data_root = project_path(args.data_root or cfg["data"]["root"])
    image_root = project_path(args.image_root or cfg["data"]["image_root"])
    output_root = project_path(args.output_root or cfg["run"]["output_root"])
    ablation = args.ablation or cfg["data"].get("ablation", "default")
    split_root = data_root / ablation
    manifest_path = split_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset manifest is mandatory: {manifest_path}; run data/build_sft_dataset.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_ready = manifest.get("benchmark_ready", manifest.get("paper_ready", False))
    benchmark_reasons = manifest.get(
        "benchmark_ready_reasons", manifest.get("paper_ready_reasons", [])
    )
    if args.require_paper_ready and not benchmark_ready:
        raise ValueError(
            "dataset_manifest.json is not benchmark-ready: "
            + "; ".join(benchmark_reasons or ["reason not recorded"])
        )
    if args.require_icra_ready and not manifest.get("icra_scientific_ready", False):
        raise ValueError(
            "dataset_manifest.json is not ICRA-scientific-ready: "
            + "; ".join(
                manifest.get("icra_scientific_ready_reasons") or ["reason not recorded"]
            )
        )
    if args.save_total_limit is not None and args.save_total_limit < 1:
        raise ValueError("--save_total_limit must be >= 1")
    if args.max_eval_samples is not None and args.max_eval_samples < 1:
        raise ValueError("--max_eval_samples must be >= 1")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    for split in ("train", "val", "test"):
        if not (split_root / f"{split}.jsonl").is_file():
            raise FileNotFoundError(f"missing split JSONL: {split_root / f'{split}.jsonl'}")

    model_cfg = cfg["model"]
    method = str(cfg["method"])
    freeze_prefixes = set(model_cfg.get("freeze_prefixes", []))
    if method == "full_text_backbone":
        required = {
            "model.vision_tower",
            "model.embed_vision",
            "model.audio_tower",
            "model.embed_audio",
        }
        missing = sorted(required - freeze_prefixes)
        unexpected = sorted(freeze_prefixes - required)
        if missing or unexpected:
            raise ValueError(
                "full_text_backbone requires frozen modality encoders/projectors; "
                f"missing={missing} unexpected={unexpected}"
            )
    elif method == "full_all_active":
        unexpected_vision_freeze = sorted(
            {"model.vision_tower", "model.embed_vision"} & freeze_prefixes
        )
        if unexpected_vision_freeze:
            raise ValueError(
                "full_all_active must train the complete vision path; frozen prefixes="
                f"{unexpected_vision_freeze}; use configs/models/gemma4_all_active.yaml"
            )
        missing_audio_freeze = sorted(
            {"model.audio_tower", "model.embed_audio"} - freeze_prefixes
        )
        if missing_audio_freeze:
            raise ValueError(
                "full_all_active requires the complete unused audio path to be frozen; "
                f"missing freeze_prefixes={missing_audio_freeze}"
            )
        unexpected_freeze = sorted(
            freeze_prefixes - {"model.audio_tower", "model.embed_audio"}
        )
        if unexpected_freeze:
            raise ValueError(
                "full_all_active has unexpected frozen prefixes: "
                f"{unexpected_freeze}"
            )
    elif method == "lora":
        if not isinstance(model_cfg.get("target_modules"), str):
            raise ValueError(
                "Gemma 4 LoRA target_modules must be a full-path regex, not suffix names"
            )
        if model_cfg.get("lora_trainable_path_fragment") != ".language_model.":
            raise ValueError("Gemma 4 LoRA must be restricted to the language backbone")

    tag = f"{model_cfg['name']}__{cfg['method']}__{ablation}__seed{args.seed}"
    output_dir = project_path(args.output_dir) if args.output_dir else output_root / tag
    deepspeed = cfg.get("deepspeed")
    if args.deepspeed is not None:
        deepspeed = None if args.deepspeed.lower() == "none" else args.deepspeed
    deepspeed_path = project_path(deepspeed) if deepspeed else None
    if deepspeed_path and not deepspeed_path.is_file():
        raise FileNotFoundError(f"DeepSpeed config does not exist: {deepspeed_path}")

    cfg["resolved"] = {
        "project_root": str(PROJECT_ROOT),
        "base_config": str(base_path),
        "model_config": str(model_path),
        "method_config": str(method_path),
        "data_root": str(data_root),
        "split_root": str(split_root),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "deepspeed": str(deepspeed_path) if deepspeed_path else None,
        "deepspeed_sha256": file_sha256(deepspeed_path) if deepspeed_path else None,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "ablation": ablation,
        "seed": args.seed,
    }
    config_bytes = json.dumps(cfg, sort_keys=True, default=str).encode()
    cfg["resolved"]["sha256"] = hashlib.sha256(config_bytes).hexdigest()

    prior_config_path = output_dir / "resolved_config.json"
    if prior_config_path.is_file():
        try:
            prior_config = json.loads(prior_config_path.read_text(encoding="utf-8"))
            prior_sha = prior_config.get("resolved", {}).get("sha256")
        except (OSError, json.JSONDecodeError):
            prior_sha = None
        if args.resume == "none":
            raise ValueError(
                f"refusing to overwrite an existing run with --resume none: {output_dir}; "
                "choose a new --output_dir"
            )
        if prior_sha != cfg["resolved"]["sha256"] and not args.allow_resume_config_mismatch:
            raise ValueError(
                "existing run config does not match this launch; use a new output directory "
                "(or explicitly pass --allow-resume-config-mismatch after auditing the diff)"
            )
    prior_run_manifest_path = output_dir / "run_manifest.json"
    if not args.dry_run and args.resume != "none" and prior_run_manifest_path.is_file():
        prior_run_manifest = json.loads(prior_run_manifest_path.read_text(encoding="utf-8"))
        prior_world_size = prior_run_manifest.get("world_size")
        current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if prior_world_size is not None and int(prior_world_size) != current_world_size:
            raise ValueError(
                "refusing to resume a ZeRO checkpoint with a different world size: "
                f"previous={prior_world_size} current={current_world_size}"
            )

    print(json.dumps(cfg["resolved"], indent=2))
    if args.dry_run:
        print("[dry-run] configuration and dataset paths are valid; model was not loaded")
        return 0

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    from data.dataset import InspectionSFTDataset, MultimodalSFTCollator
    from data.schema import PROMPT_VERSION
    from models.registry import get_adapter, parameter_numel

    set_seed(args.seed)
    train_cfg = cfg["train"]
    save_steps = args.save_steps or int(train_cfg["save_steps"])
    eval_steps = args.eval_steps or int(train_cfg["eval_steps"])
    if args.max_steps > 0:
        save_steps = min(save_steps, args.max_steps)
        eval_steps = min(eval_steps, save_steps)
    if save_steps % eval_steps:
        raise ValueError("save_steps must be a multiple of eval_steps when load_best_model_at_end=true")
    report_to = args.report_to if args.report_to is not None else cfg["run"].get("report_to", "none")

    # Construct TrainingArguments before from_pretrained. Its official
    # HfTrainerDeepSpeedConfig path resolves every DeepSpeed "auto" field,
    # establishes the distributed weakref, and activates zero.Init before the
    # 12B model is materialized. Keeping training_args alive keeps that state.
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        max_steps=args.max_steps,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        max_grad_norm=float(train_cfg["max_grad_norm"]),
        optim=train_cfg.get("optim", "adamw_torch"),
        bf16=bool(train_cfg.get("bf16", True)),
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(train_cfg["logging_steps"]),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=eval_steps,
        save_strategy=train_cfg["save_strategy"],
        save_steps=save_steps,
        save_total_limit=(
            args.save_total_limit
            if args.save_total_limit is not None
            else int(train_cfg["save_total_limit"])
        ),
        save_only_model=False,
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=bool(train_cfg.get("greater_is_better", False)),
        dataloader_num_workers=int(cfg["data"]["num_workers"]),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=int(cfg["data"]["num_workers"]) > 0,
        deepspeed=str(deepspeed_path) if deepspeed_path else None,
        report_to=report_to,
        run_name=tag,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        restore_callback_states_from_checkpoint=True,
    )
    resume_checkpoint = resolve_resume(output_dir, args.resume)
    if isinstance(resume_checkpoint, str):
        checkpoint_parent = Path(resume_checkpoint).resolve().parent
        if checkpoint_parent != output_dir.resolve() and not args.allow_resume_config_mismatch:
            raise ValueError(
                "external checkpoint resume is fail-closed because its config/data/world-size "
                "provenance cannot be guaranteed; resume in the original output directory or "
                "explicitly pass --allow-resume-config-mismatch after auditing it"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rank = training_args.process_index
    if rank == 0:
        atomic_json(output_dir / "resolved_config.json", cfg)

    adapter = get_adapter(model_cfg["adapter"], model_cfg)
    model, processor = adapter.load()
    # The collator needs only the processor methods on the adapter. Do not keep
    # a second model reference in an object handed to a DataLoader.
    adapter.model = None
    frozen_prefix_counts: dict[str, int] = {}
    if method in {"full_text_backbone", "full_all_active"}:
        if model_cfg.get("quantization", {}).get("load_in_4bit"):
            raise ValueError(f"{method} cannot use a 4-bit loaded base model")
        frozen_prefix_counts = adapter.freeze_input_modules(model)
        if rank == 0:
            print(
                "[freeze] "
                + ", ".join(
                    f"{key}={value:,}" for key, value in frozen_prefix_counts.items()
                )
            )
    elif method == "lora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if model_cfg.get("quantization", {}).get("load_in_4bit"):
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["train"].get("gradient_checkpointing", True)
            )
        lora = cfg["lora"]
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                bias=lora["bias"],
                task_type=lora["task_type"],
                target_modules=model_cfg["target_modules"],
            ),
        )
        allowed_fragment = model_cfg.get("lora_trainable_path_fragment")
        if allowed_fragment:
            misplaced_adapters = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and "lora_" in name
                and allowed_fragment not in name
            ]
            if misplaced_adapters:
                raise RuntimeError(
                    "LoRA adapters were inserted outside the declared language backbone: "
                    f"{misplaced_adapters[:10]}"
                )
        if rank == 0:
            model.print_trainable_parameters()
    else:
        raise ValueError(f"unknown training method {method!r}")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable = sum(
        parameter_numel(parameter) for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter_numel(parameter) for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("zero trainable parameters")
    frozen_parameters = total - trainable
    if method in {"full_text_backbone", "full_all_active"}:
        expected_counts = model_cfg.get("expected_parameter_counts") or {}
        expected_total = expected_counts.get("total")
        expected_trainable = expected_counts.get("trainable")
        expected_frozen_prefixes = expected_counts.get("frozen_by_prefix") or {}
        mismatches = []
        if expected_total is not None and int(expected_total) != total:
            mismatches.append(f"total expected={expected_total} actual={total}")
        if expected_trainable is not None and int(expected_trainable) != trainable:
            mismatches.append(
                f"trainable expected={expected_trainable} actual={trainable}"
            )
        for prefix, expected in expected_frozen_prefixes.items():
            actual = frozen_prefix_counts.get(prefix)
            if actual != int(expected):
                mismatches.append(f"{prefix} expected={expected} actual={actual}")
        if mismatches:
            raise RuntimeError(
                "pinned Gemma parameter/freeze contract changed: " + "; ".join(mismatches)
            )
    if rank == 0:
        print(
            f"[params] trainable={trainable:,} frozen={frozen_parameters:,} total={total:,} "
            f"({100 * trainable / total:.3f}% trainable)"
        )

    train_dataset = InspectionSFTDataset(split_root / "train.jsonl")
    val_dataset = InspectionSFTDataset(split_root / "val.jsonl")
    if args.max_eval_samples is not None and len(val_dataset) > args.max_eval_samples:
        from torch.utils.data import Subset

        val_dataset = Subset(val_dataset, range(args.max_eval_samples))
    collator = MultimodalSFTCollator(
        adapter=adapter,
        image_root=image_root,
        max_seq_len=int(cfg["data"]["max_seq_len"]),
        num_frames=int(cfg["data"]["num_frames"]),
    )
    if rank == 0:
        print(f"[data] train={len(train_dataset):,} val={len(val_dataset):,} images={image_root}")

    finetuning_commit = git_commit(PROJECT_ROOT, "SAIE_FINETUNING_GIT_COMMIT")
    saie_repo_commit = git_commit(PROJECT_ROOT.parent / "SAIE", "SAIE_GIT_COMMIT")
    run_manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        # Keep the legacy key, but never fill it from the nested simulator
        # repository. Code bundles that are not a Git checkout should provide
        # SAIE_BUNDLE_SHA256 (or SAIE_FINETUNING_GIT_COMMIT) explicitly.
        "git_commit": finetuning_commit,
        "finetuning_code_git_commit": finetuning_commit,
        "saie_repo_git_commit": saie_repo_commit,
        "finetuning_code_bundle_sha256": os.environ.get("SAIE_BUNDLE_SHA256"),
        "dataset_source_sha256": manifest.get("source_sha256"),
        "dataset_selected_images_sha256": manifest.get("selected_images_sha256"),
        "dataset_selected_image_count": manifest.get("selected_image_count"),
        "dataset_split_jsonl_sha256": manifest.get("split_jsonl_sha256"),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "dataset_benchmark_ready": benchmark_ready,
        "dataset_icra_scientific_ready": manifest.get("icra_scientific_ready", False),
        "training_scope": method,
        "parameter_counts": {
            "trainable": trainable,
            "frozen": frozen_parameters,
            "total": total,
        },
        "frozen_prefix_parameter_counts": frozen_prefix_counts,
        "model_id": model_cfg.get("hf_id"),
        "model_requested_revision": model_cfg.get("revision"),
        "model_resolved_commit": getattr(model.config, "_commit_hash", None),
        "resolved_config_sha256": cfg["resolved"]["sha256"],
        "prompt_version": PROMPT_VERSION,
        "resume_from_checkpoint": resume_checkpoint,
        "requested_max_steps": args.max_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "save_total_limit": training_args.save_total_limit,
        "eval_samples": len(val_dataset),
        "hostname": platform.node(),
        "world_size": training_args.world_size,
        "python": sys.version,
        "packages": package_versions(),
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "final": str(output_dir / "final"),
            "resolved_config": str(output_dir / "resolved_config.json"),
        },
    }
    if rank == 0:
        atomic_json(output_dir / "run_manifest.json", run_manifest)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    try:
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(str(output_dir / "final"))
        trainer.save_state()
        if trainer.is_world_process_zero():
            processor.save_pretrained(output_dir / "final")
            atomic_json(output_dir / "train_metrics.json", result.metrics)
            run_manifest["status"] = "complete"
            run_manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            run_manifest["train_metrics"] = result.metrics
            atomic_json(output_dir / "run_manifest.json", run_manifest)
            print(f"[done] final={output_dir / 'final'}")
    except BaseException as exc:
        if rank == 0:
            run_manifest["status"] = "failed"
            run_manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
            run_manifest["error"] = f"{type(exc).__name__}: {exc}"
            atomic_json(output_dir / "run_manifest.json", run_manifest)
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
