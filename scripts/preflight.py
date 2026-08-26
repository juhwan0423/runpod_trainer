#!/usr/bin/env python3
"""Fail-fast RunPod/data/config checks before an H100 training launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.schema import (  # noqa: E402
    ACTION_LABELS,
    ACTION_LABEL_SET,
    PROMPT_VERSION,
    read_jsonl,
    render_policy_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="directory containing <ablation>/*.jsonl")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ablation", default="default")
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--require-paper-ready", action="store_true")
    parser.add_argument("--require-icra-ready", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--check-model-access", action="store_true")
    parser.add_argument("--check-processor-contract", action="store_true")
    parser.add_argument("--check-all-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--min-ram-gb", type=float, default=0.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--requirements-file",
        type=Path,
        default=ROOT / "requirements-runpod.txt",
        help="pinned-package manifest to check the active interpreter against "
        "(defaults to the original Gemma4 venv's requirements-runpod.txt; pass "
        "requirements-navila.txt or similar when preflighting a different venv)",
    )
    return parser.parse_args()


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str, fatal: bool = True) -> None:
    checks.append({"name": name, "ok": bool(ok), "fatal": bool(fatal), "detail": detail})
    marker = "OK" if ok else ("FAIL" if fatal else "WARN")
    print(f"[{marker}] {name}: {detail}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pinned_requirements(path: Path) -> dict[str, str]:
    """Extract exact ``name==version`` pins from a requirements file.

    Lines without a bare ``==`` pin (range specifiers, unpinned packages,
    comments) are skipped rather than guessed at -- this only checks what
    the requirements file actually promises.
    """
    pinned: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "==" not in line or any(op in line for op in ("<", ">", ";")):
            continue
        name, _, ver = line.partition("==")
        pinned[name.strip()] = ver.strip()
    return pinned


def package_check(checks: list[dict[str, Any]], requirements_file: Path) -> None:
    # Requirements-file-driven (not hardcoded) so this works for any model's
    # venv, not just the original single-venv Gemma4 recipe -- NaVILA and any
    # future adapter each pin a different, incompatible transformers/torch
    # stack in their own requirements-*.txt (see requirements-navila.txt).
    required = parse_pinned_requirements(requirements_file)
    for package, required_version in required.items():
        try:
            installed = version(package)
            from packaging.version import Version

            ok = Version(installed).base_version == Version(required_version).base_version
            detail = f"{installed} (required == {required_version})"
        except PackageNotFoundError:
            ok, detail = False, "not installed"
        add(checks, f"package:{package}", ok, detail)
    try:
        import torch
        import torchvision

        add(
            checks,
            "torchvision_runtime",
            True,
            f"torch={torch.__version__} torchvision={torchvision.__version__} cuda_build={torch.version.cuda}",
        )
    except Exception as exc:
        add(checks, "torchvision_runtime", False, f"{type(exc).__name__}: {exc}")


def dataset_check(
    checks: list[dict[str, Any]], split_root: Path, image_root: Path, check_all_images: bool
) -> None:
    manifest_path = split_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        add(checks, "dataset_manifest", False, f"missing {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        add(checks, "dataset_manifest", False, str(exc))
        return
    add(
        checks,
        "dataset_manifest",
        True,
        f"paper_ready={manifest.get('paper_ready')} sha256={manifest.get('source_sha256', '?')}",
    )
    benchmark_ready = manifest.get("benchmark_ready", manifest.get("paper_ready", False))
    benchmark_reasons = manifest.get(
        "benchmark_ready_reasons", manifest.get("paper_ready_reasons", [])
    ) or []
    add(
        checks,
        "benchmark_readiness",
        bool(benchmark_ready),
        "benchmark-ready"
        if benchmark_ready
        else "; ".join(benchmark_reasons) or "not benchmark-ready",
        fatal=False,
    )
    icra_ready = bool(manifest.get("icra_scientific_ready", False))
    icra_reasons = manifest.get("icra_scientific_ready_reasons") or []
    add(
        checks,
        "icra_scientific_readiness",
        icra_ready,
        "ICRA-scientific-ready"
        if icra_ready
        else "; ".join(icra_reasons) or "not ICRA-scientific-ready",
        fatal=False,
    )
    recorded_noncausal = (
        manifest.get("frame_resolution", {}).get("output_noncausal_history_references")
    )
    add(
        checks,
        "manifest_causal_history",
        recorded_noncausal == 0,
        f"output_noncausal_history_references={recorded_noncausal!r}",
    )
    recorded_split_hashes = manifest.get("split_jsonl_sha256") or {}
    split_hash_errors = []
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.jsonl"
        expected = recorded_split_hashes.get(split)
        if not expected or not path.is_file():
            split_hash_errors.append(f"{split}:missing recorded hash or file")
            continue
        digest = file_sha256(path)
        if digest != expected:
            split_hash_errors.append(f"{split}:expected={expected} actual={digest}")
    add(
        checks,
        "split_jsonl_hashes",
        not split_hash_errors,
        "match dataset manifest" if not split_hash_errors else "; ".join(split_hash_errors),
    )
    repair_meta = manifest.get("repair_log") or {}
    repair_path = split_root / str(repair_meta.get("path", "repair_log.jsonl"))
    repair_errors: list[str] = []
    repair_rows = 0
    if not repair_path.is_file():
        repair_errors.append(f"missing {repair_path}")
    else:
        expected_hash = repair_meta.get("sha256")
        actual_hash = file_sha256(repair_path)
        if not expected_hash or actual_hash != expected_hash:
            repair_errors.append(
                f"sha256 expected={expected_hash!r} actual={actual_hash}"
            )
        try:
            with repair_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    repair_rows += 1
                    for key in ("missing", "resolved"):
                        value = row.get(key)
                        if not isinstance(value, str) or Path(value).is_absolute():
                            repair_errors.append(
                                f"line {line_number}: {key} must be a portable relative path"
                            )
                    if not isinstance(row.get("index_gap"), int):
                        repair_errors.append(f"line {line_number}: invalid index_gap")
                    if len(repair_errors) >= 20:
                        break
        except (OSError, json.JSONDecodeError) as exc:
            repair_errors.append(str(exc))
        expected_rows = repair_meta.get("rows")
        if expected_rows != repair_rows:
            repair_errors.append(f"rows expected={expected_rows!r} actual={repair_rows}")
    add(
        checks,
        "repair_log",
        not repair_errors,
        f"portable rows={repair_rows} sha256 verified"
        if not repair_errors
        else "; ".join(repair_errors),
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.jsonl"
        try:
            rows = read_jsonl(path)
        except (OSError, ValueError) as exc:
            add(checks, f"split:{split}", False, str(exc))
            continue
        rows_by_split[split] = rows
        add(checks, f"split:{split}", bool(rows), f"rows={len(rows):,}")

    if set(rows_by_split) != {"train", "val", "test"}:
        return
    manifest_count_errors = []
    for split, rows in rows_by_split.items():
        recorded = manifest.get("splits", {}).get(split, {}).get("rows")
        if recorded != len(rows):
            manifest_count_errors.append(f"{split}:manifest={recorded} actual={len(rows)}")
    add(
        checks,
        "manifest_split_counts",
        not manifest_count_errors,
        "match JSONL files" if not manifest_count_errors else "; ".join(manifest_count_errors),
    )
    warehouse_sets = {
        split: {str(row.get("warehouse_id")) for row in rows}
        for split, rows in rows_by_split.items()
    }
    overlaps = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = warehouse_sets[left] & warehouse_sets[right]
        if common:
            overlaps.append(f"{left}/{right}:{sorted(common)}")
    add(checks, "warehouse_leakage", not overlaps, "; ".join(overlaps) if overlaps else "none")

    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    invalid_labels: Counter[str] = Counter()
    missing_images: list[str] = []
    absolute_images: list[str] = []
    noncausal_images: list[str] = []
    invalid_camera_paths: list[str] = []
    corrupt_images: list[str] = []
    verified_paths: set[Path] = set()
    selected_image_refs: set[str] = set()
    schema_errors: list[str] = []
    try:
        from PIL import Image
    except Exception:
        Image = None
    for split, rows in rows_by_split.items():
        for row_index, row in enumerate(rows):
            raw_sample_id = row.get("sample_id")
            sample_id = str(raw_sample_id or f"{row.get('episode_id')}/{row.get('frame_id')}")
            if row.get("schema_version") != "saie_sft_v2":
                schema_errors.append(f"{split}:{row_index}:schema_version={row.get('schema_version')!r}")
            if not isinstance(raw_sample_id, str) or not raw_sample_id:
                schema_errors.append(f"{split}:{row_index}:missing sample_id")
            current_index = row.get("frame_id")
            if not isinstance(current_index, int) or isinstance(current_index, bool):
                schema_errors.append(f"{split}:{row_index}:non-numeric frame_id={current_index!r}")
            expected_group = (
                row.get("action_label", "")[1 : row.get("action_label", "").find("]")]
                if "]" in str(row.get("action_label", ""))
                else None
            )
            if row.get("action_group") != expected_group or row.get("action_category") != expected_group:
                schema_errors.append(
                    f"{split}:{row_index}:action group/category mismatch label={row.get('action_label')!r}"
                )
            if not isinstance(row.get("source_frame_id"), int) or isinstance(
                row.get("source_frame_id"), bool
            ):
                schema_errors.append(
                    f"{split}:{row_index}:non-numeric source_frame_id={row.get('source_frame_id')!r}"
                )
            for nullable_bool in (
                "hazard_present",
                "is_distractor",
                "risk_adjacent",
                "in_task_window",
            ):
                value = row.get(nullable_bool)
                if value is not None and not isinstance(value, bool):
                    schema_errors.append(
                        f"{split}:{row_index}:{nullable_bool} must be bool|null, got {value!r}"
                    )
            if sample_id in sample_ids:
                duplicate_ids.append(sample_id)
            sample_ids.add(sample_id)
            label = row.get("action_label")
            if label not in ACTION_LABEL_SET:
                invalid_labels[str(label)] += 1
            images = row.get("images") or row.get("image_history") or [row.get("image")]
            if not isinstance(images, list):
                schema_errors.append(f"{split}:{row_index}:images is not a list")
                images = []
            if images and row.get("image") != images[-1]:
                schema_errors.append(f"{split}:{row_index}:image is not the current/last image")
            for image in images:
                if not image:
                    if check_all_images or row_index < 100:
                        missing_images.append(f"{sample_id}:null")
                    continue
                relative = Path(image)
                if relative.is_absolute():
                    absolute_images.append(str(relative))
                    path = relative
                else:
                    path = image_root / relative
                    selected_image_refs.add(relative.as_posix())
                if "/spot_cam/" not in f"/{relative.as_posix()}":
                    if len(invalid_camera_paths) < 100:
                        invalid_camera_paths.append(str(relative))
                if check_all_images or row_index < 100:
                    if not path.is_file() and len(missing_images) < 100:
                        missing_images.append(str(path))
                    elif path.is_file() and path not in verified_paths and Image is not None:
                        try:
                            with Image.open(path) as source:
                                source.verify()
                            verified_paths.add(path)
                        except Exception as exc:
                            if len(corrupt_images) < 100:
                                corrupt_images.append(f"{path}: {type(exc).__name__}: {exc}")
                match = re.search(r"rgb_(\d+)\.png$", str(image))
                if match and isinstance(current_index, int) and int(match.group(1)) > current_index:
                    if len(noncausal_images) < 100:
                        noncausal_images.append(
                            f"{sample_id}: image={match.group(1)} current={current_index}"
                        )
    add(
        checks,
        "sft_schema_v2",
        not schema_errors,
        "schema/sample/frame/category/nullability contract valid"
        if not schema_errors
        else f"errors={len(schema_errors)} examples={schema_errors[:5]}",
    )
    add(
        checks,
        "sample_id_uniqueness",
        not duplicate_ids,
        "unique" if not duplicate_ids else f"duplicates={len(duplicate_ids)} examples={duplicate_ids[:5]}",
    )
    add(
        checks,
        "canonical_labels",
        not invalid_labels,
        "all 14-label contract" if not invalid_labels else str(dict(invalid_labels)),
    )
    for split in ("train", "val"):
        labels = Counter(row["action_label"] for row in rows_by_split[split])
        absent = [label for label in ACTION_LABELS if labels[label] == 0]
        add(
            checks,
            f"{split}_label_coverage",
            not absent,
            "all 14 present" if not absent else f"missing={absent}",
        )
    add(
        checks,
        "portable_image_paths",
        not absolute_images,
        "all relative" if not absolute_images else f"absolute={len(absolute_images)}",
    )
    add(
        checks,
        "spot_cam_only",
        not invalid_camera_paths,
        "all model inputs are spot_cam"
        if not invalid_camera_paths
        else f"invalid={len(invalid_camera_paths)} examples={invalid_camera_paths[:3]}",
    )
    add(
        checks,
        "image_files",
        not missing_images,
        f"checked={'all' if check_all_images else 'first 100/split'}; missing={len(missing_images)}"
        + (f" examples={missing_images[:3]}" if missing_images else ""),
    )
    add(
        checks,
        "causal_image_history",
        not noncausal_images,
        "no future frames" if not noncausal_images else f"violations={len(noncausal_images)} examples={noncausal_images[:3]}",
    )
    add(
        checks,
        "image_decode",
        not corrupt_images and Image is not None,
        f"verified_unique={len(verified_paths)}"
        if not corrupt_images and Image is not None
        else f"corrupt/examples={corrupt_images[:3]} PIL_available={Image is not None}",
    )
    if check_all_images and not missing_images and not absolute_images:
        digest = hashlib.sha256()
        for relative in sorted(selected_image_refs):
            path = image_root / relative
            digest.update(relative.encode())
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        actual_image_hash = digest.hexdigest()
        expected_image_hash = manifest.get("selected_images_sha256")
        add(
            checks,
            "selected_image_hash",
            actual_image_hash == expected_image_hash,
            f"expected={expected_image_hash} actual={actual_image_hash} files={len(selected_image_refs)}",
        )
    else:
        add(
            checks,
            "selected_image_hash",
            True,
            "skipped for smoke/partial image check",
        )


def gpu_check(checks: list[dict[str, Any]], require_h100: bool) -> None:
    try:
        import torch

        available = torch.cuda.is_available()
        add(checks, "cuda", available, f"available={available} count={torch.cuda.device_count()}")
        if not available:
            return
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        memories = [
            torch.cuda.get_device_properties(index).total_memory / (1024**3)
            for index in range(torch.cuda.device_count())
        ]
        h100_ok = all("H100" in name.upper() and memory >= 75 for name, memory in zip(names, memories))
        add(
            checks,
            "h100_80gb",
            h100_ok,
            ", ".join(f"{name} ({memory:.1f} GiB)" for name, memory in zip(names, memories)),
            fatal=require_h100,
        )
        add(checks, "bf16_gpu", torch.cuda.is_bf16_supported(), "native BF16 support")
    except Exception as exc:  # diagnostic boundary: report import/driver failures together
        add(checks, "cuda", False, f"{type(exc).__name__}: {exc}")


def processor_contract_check(
    checks: list[dict[str, Any]],
    model_id: str,
    model_revision: str | None,
    split_root: Path,
    image_root: Path,
) -> None:
    """Exercise the real remote chat template, PIL path, and answer mask once."""
    try:
        from PIL import Image
        from transformers import AutoProcessor

        from data.dataset import derive_target_mask
        from models.registry import get_adapter

        row = read_jsonl(split_root / "train.jsonl")[0]
        image_ref = row.get("image") or (row.get("images") or row.get("image_history"))[0]
        image_path = Path(image_ref)
        image_path = image_path if image_path.is_absolute() else image_root / image_path
        with Image.open(image_path) as source:
            image = source.convert("RGB").copy()
        target = row["action_label"]
        prompt = render_policy_prompt(row["instruction"], 1)
        processor = AutoProcessor.from_pretrained(
            model_id, revision=model_revision, trust_remote_code=False
        )
        adapter = get_adapter("gemma4_unified", {})
        adapter.processor = processor
        example = {
            "images": [image],
            "instruction": prompt,
            "target": target,
        }
        encoded, prefix = adapter.encode_batch([example], 2048)
        mask = derive_target_mask(
            encoded["input_ids"],
            encoded["attention_mask"],
            prefix["input_ids"],
            prefix["attention_mask"],
        )
        active_tokens = int(encoded["attention_mask"].sum())
        if active_tokens > 2048:
            raise RuntimeError(f"processor contract sample exceeds max_seq_len: {active_tokens} > 2048")
        if int(mask.sum()) > 64:
            raise RuntimeError(f"assistant suffix is unexpectedly long: {int(mask.sum())} tokens")
        tokenizer = getattr(processor, "tokenizer", processor)
        supervised = encoded["input_ids"][mask]
        decoded = tokenizer.decode(supervised, skip_special_tokens=True)
        if decoded.strip() != target:
            raise RuntimeError(
                f"target mask did not decode exactly: target={target!r} decoded={decoded!r}"
            )
        add(
            checks,
            "processor_contract",
            True,
            f"{type(processor).__name__}; prompt={PROMPT_VERSION}; "
            f"prefix_diff_supervised_tokens={int(mask.sum())}",
        )
    except Exception as exc:
        add(checks, "processor_contract", False, f"{type(exc).__name__}: {exc}")


def model_architecture_contract_check(
    checks: list[dict[str, Any]], model_id: str, model_revision: str | None
) -> None:
    """Fail if the pinned checkpoint no longer exposes both modality towers."""

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_id, revision=model_revision, trust_remote_code=False
        )
        vision = getattr(config, "vision_config", None)
        audio = getattr(config, "audio_config", None)
        if vision is None or audio is None:
            raise RuntimeError(
                "Gemma 4 config must expose non-null vision_config and audio_config"
            )
        add(
            checks,
            "model_architecture_contract",
            True,
            f"model_type={config.model_type} "
            f"vision={getattr(vision, 'model_type', type(vision).__name__)} "
            f"audio={getattr(audio, 'model_type', type(audio).__name__)}; "
            "expected runtime prefixes=model.vision_embedder/model.embed_vision/model.embed_audio "
            "(no separate audio tower; corrected 2026-08-21 against the real checkpoint)",
        )
    except Exception as exc:
        add(
            checks,
            "model_architecture_contract",
            False,
            f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []
    package_check(checks, args.requirements_file)

    data_root = args.data_root.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    split_root = data_root / args.ablation
    add(checks, "image_root", image_root.is_dir(), str(image_root))
    dataset_check(checks, split_root, image_root, args.check_all_images)

    output_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_root).free / (1024**3)
    add(
        checks,
        "output_disk",
        free_gb >= args.min_free_gb,
        f"free={free_gb:.1f} GiB required={args.min_free_gb:.1f} GiB path={output_root}",
    )
    writable_probe = output_root / ".preflight_write_probe"
    try:
        writable_probe.write_text("ok\n", encoding="utf-8")
        writable_probe.unlink()
        writable = True
        write_detail = str(output_root)
    except OSError as exc:
        writable, write_detail = False, str(exc)
    add(checks, "output_writable", writable, write_detail)
    if args.min_ram_gb > 0:
        try:
            import psutil

            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            add(
                checks,
                "host_ram",
                available_ram_gb >= args.min_ram_gb,
                f"available={available_ram_gb:.1f} GiB total={total_ram_gb:.1f} GiB "
                f"required_available={args.min_ram_gb:.1f} GiB",
            )
        except Exception as exc:
            add(checks, "host_ram", False, f"{type(exc).__name__}: {exc}")

    if args.require_gpu or args.require_h100:
        gpu_check(checks, args.require_h100)
    if args.check_model_access:
        try:
            from huggingface_hub import HfApi, get_token

            token = get_token()
            if not token:
                add(checks, "hf_token", False, "no token; run `hf auth login` or set HF_TOKEN")
            else:
                info = HfApi().model_info(
                    args.model, revision=args.model_revision, token=token
                )
                revision_ok = not args.model_revision or info.sha == args.model_revision
                add(
                    checks,
                    "model_access",
                    revision_ok,
                    f"{info.id} requested={args.model_revision or 'main'} resolved={info.sha}",
                )
        except Exception as exc:
            add(checks, "model_access", False, f"{type(exc).__name__}: {exc}")
    if args.check_model_access or args.check_processor_contract:
        model_architecture_contract_check(checks, args.model, args.model_revision)
    if args.check_processor_contract:
        processor_contract_check(
            checks, args.model, args.model_revision, split_root, image_root
        )

    if args.require_paper_ready:
        benchmark_check = next(
            (item for item in checks if item["name"] == "benchmark_readiness"), None
        )
        if benchmark_check and not benchmark_check["ok"]:
            benchmark_check["fatal"] = True
    if args.require_icra_ready:
        icra_check = next(
            (item for item in checks if item["name"] == "icra_scientific_readiness"), None
        )
        if icra_check and not icra_check["ok"]:
            icra_check["fatal"] = True

    failures = [item for item in checks if item["fatal"] and not item["ok"]]
    report = {
        "ok": not failures,
        "checks": checks,
        "environment": {
            "python": sys.version,
            "cwd": os.getcwd(),
            "data_root": str(data_root),
            "image_root": str(image_root),
            "output_root": str(output_root),
        },
    }
    if args.report:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[summary] {'PASS' if not failures else 'FAIL'} fatal_failures={len(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
