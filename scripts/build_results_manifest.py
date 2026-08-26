#!/usr/bin/env python3
"""Build a hash-linked registry for the SAIE experiment results in this workspace.

The directory names are intentionally registered here instead of inferred from
run manifests.  Several historical manifests reused tags and artifact paths
across balanced and unbalanced runs, so inference would silently conflate
scientifically distinct checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    model: str
    method: str
    dataset: str
    seed: int
    artifact_dir: str
    aggregate_relpath: str = "scores/aggregate.json"
    prediction_relpath: str = "vlm_output"
    validity: str = "valid"
    exclusion_reason: str | None = None
    allow_missing_prediction_manifest: bool = False


EXPERIMENTS = (
    Experiment(
        "gemma4__fft__original__seed0",
        "Gemma4",
        "FFT",
        "original",
        0,
        "_gemma4_fft_results",
        "judge_output/aggregate.json",
        "predictions/vlm_output",
    ),
    Experiment("gemma4__fft__balanced__seed0", "Gemma4", "FFT", "balanced", 0, "_gemma4_fft_balanced_eval"),
    Experiment("gemma4__lora__balanced__seed0", "Gemma4", "LoRA", "balanced", 0, "_gemma4_lora_eval"),
    Experiment("gemma4__lora__original__seed0", "Gemma4", "LoRA", "original", 0, "_gemma4_lora_unbalanced_eval"),
    Experiment("gemma4__lora__redesigned__seed0", "Gemma4", "LoRA", "redesigned", 0, "_gemma4_lora_redesigned_seed0_eval"),
    Experiment("gemma4__lora__redesigned__seed1", "Gemma4", "LoRA", "redesigned", 1, "_gemma4_lora_redesigned_seed1_eval"),
    Experiment("gemma4__lora__redesigned__seed2", "Gemma4", "LoRA", "redesigned", 2, "_gemma4_lora_redesigned_seed2_eval"),
    Experiment(
        "navila__fft__original__seed0",
        "NaVILA",
        "FFT",
        "original",
        0,
        "_navila_eval",
        prediction_relpath="prediction_artifacts",
        allow_missing_prediction_manifest=True,
    ),
    Experiment("navila__fft__balanced__seed0", "NaVILA", "FFT", "balanced", 0, "_navila_fft_balanced_eval"),
    Experiment("navila__lora__original__seed0", "NaVILA", "LoRA", "original", 0, "_navila_lora_eval"),
    Experiment("navila__lora__balanced__seed0", "NaVILA", "LoRA", "balanced", 0, "_navila_lora_balanced_eval"),
    Experiment("navila__lora__redesigned__seed0", "NaVILA", "LoRA", "redesigned", 0, "_navila_lora_redesigned_eval"),
    Experiment("internvla_n1__fft__original__seed0", "InternVLA-N1", "FFT", "original", 0, "_internvla_n1_fft_fixed_eval"),
    Experiment("internvla_n1__lora__balanced__seed0", "InternVLA-N1", "LoRA", "balanced", 0, "_internvla_n1_lora_fixed_eval"),
    Experiment("internvla_n1__lora__original__seed0", "InternVLA-N1", "LoRA", "original", 0, "_internvla_n1_lora_unbalanced_eval"),
    Experiment("internvla_n1__lora__redesigned__seed0", "InternVLA-N1", "LoRA", "redesigned", 0, "_internvla_n1_lora_redesigned_eval"),
    Experiment(
        "internvla_m1__fft__balanced__seed0",
        "InternVLA-M1",
        "FFT",
        "balanced",
        0,
        "_internvla_m1_fft_balanced_eval",
        prediction_relpath="prediction_artifacts",
    ),
    Experiment("internvla_m1__lora__balanced__seed0", "InternVLA-M1", "LoRA", "balanced", 0, "_internvla_m1_lora_balanced_eval"),
    Experiment("internvla_m1__lora__original__seed0", "InternVLA-M1", "LoRA", "original", 0, "_internvla_m1_lora_unbalanced_eval"),
    Experiment(
        "internvla_n1__fft__legacy_parser__seed0",
        "InternVLA-N1",
        "FFT",
        "original",
        0,
        "_internvla_n1_eval",
        validity="invalid_diagnostic",
        exclusion_reason="legacy generated-text slicing/parser bug; use the fixed eval",
    ),
    Experiment(
        "internvla_n1__lora__legacy_parser__seed0",
        "InternVLA-N1",
        "LoRA",
        "balanced",
        0,
        "_internvla_n1_lora_eval",
        validity="invalid_diagnostic",
        exclusion_reason="legacy generated-text slicing/parser bug; use the fixed eval",
    ),
)


DATASET_MANIFESTS = {
    "original": "_sft_gemma4/default/dataset_manifest.json",
    "balanced": "_sft_gemma4_balanced/default/dataset_manifest.json",
    "redesigned": "_sft_redesigned/default/dataset_manifest.json",
}


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def tree_fingerprint(root: Path) -> dict[str, Any]:
    """Hash a directory using relative names, sizes, and per-file hashes."""
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha = sha256_file(path)
        total_bytes += size
        digest.update(f"{relative}\0{size}\0{file_sha}\n".encode("utf-8"))
    return {
        "algorithm": "sha256(relative_path\\0size\\0file_sha256\\n)",
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mean_metric(aggregate: dict[str, Any], name: str) -> float | None:
    value = aggregate.get("episode_macro_distributions", {}).get(name, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def relative_or_absolute(path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return str(path)


def git_metadata(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = git("status", "--porcelain")
    return {
        "path": str(repo),
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "origin": git("remote", "get-url", "origin"),
        "dirty": None if status is None else bool(status),
    }


def build_entry(spec: Experiment, workspace: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    artifact_root = workspace / spec.artifact_dir
    aggregate_path = artifact_root / spec.aggregate_relpath
    prediction_root = artifact_root / spec.prediction_relpath
    prediction_manifest_path = prediction_root / "prediction_export_manifest.json"
    dataset_manifest_path = workspace / DATASET_MANIFESTS[spec.dataset]
    train_jsonl_path = dataset_manifest_path.parent / "train.jsonl"

    required = [artifact_root, aggregate_path, prediction_root, dataset_manifest_path, train_jsonl_path]
    if not spec.allow_missing_prediction_manifest:
        required.append(prediction_manifest_path)
    for path in required:
        if not path.exists():
            warnings.append(f"missing required artifact: {path}")

    aggregate = load_json(aggregate_path) if aggregate_path.is_file() else {}
    prediction_manifest = load_json(prediction_manifest_path) if prediction_manifest_path.is_file() else {}
    prediction_tree = tree_fingerprint(prediction_root) if prediction_root.is_dir() else None
    if spec.allow_missing_prediction_manifest and not prediction_manifest_path.is_file():
        warnings.append(
            "registered legacy provenance gap: raw prediction chunks exist but export manifest is unavailable"
        )

    diagnostic_v2 = aggregate.get("diagnostic_episode_distributions", {}).get(
        "safety_violation_rate_v2", {}
    )
    if not diagnostic_v2 or not diagnostic_v2.get("n"):
        warnings.append("safety_violation_rate_v2 unavailable; do not source paper Safety Viol. from this field")

    if spec.validity == "valid":
        expected = {"num_episodes": 210, "num_warehouses": 11, "num_frames": 12603}
        for key, value in expected.items():
            if aggregate.get(key) != value:
                warnings.append(f"judge contract mismatch: {key}={aggregate.get(key)!r}, expected {value}")

    dataset_manifest_sha = sha256_file(dataset_manifest_path) if dataset_manifest_path.is_file() else None
    aggregate_sha = sha256_file(aggregate_path) if aggregate_path.is_file() else None
    prediction_manifest_sha = (
        sha256_file(prediction_manifest_path) if prediction_manifest_path.is_file() else None
    )

    entry = {
        "experiment_id": spec.experiment_id,
        "model": spec.model,
        "method": spec.method,
        "dataset": spec.dataset,
        "seed": spec.seed,
        "validity": spec.validity,
        "exclusion_reason": spec.exclusion_reason,
        "artifact_root": relative_or_absolute(artifact_root, workspace),
        "dataset_manifest": {
            "path": relative_or_absolute(dataset_manifest_path, workspace),
            "sha256": dataset_manifest_sha,
            "train_jsonl_path": relative_or_absolute(train_jsonl_path, workspace),
            "train_jsonl_sha256": sha256_file(train_jsonl_path) if train_jsonl_path.is_file() else None,
            "train_jsonl_rows": count_lines(train_jsonl_path) if train_jsonl_path.is_file() else None,
        },
        "prediction": {
            "root": relative_or_absolute(prediction_root, workspace),
            "tree": prediction_tree,
            "export_manifest_path": relative_or_absolute(prediction_manifest_path, workspace),
            "export_manifest_sha256": prediction_manifest_sha,
            "rows": prediction_manifest.get("rows"),
            "episodes": prediction_manifest.get("episodes"),
            "normalization_rule": prediction_manifest.get("normalization_rule"),
            "wrapper_version": prediction_manifest.get("wrapper_version"),
            "source_jsonl_sha256": prediction_manifest.get("source_jsonl_sha256"),
        },
        "checkpoint": {
            "source_path": prediction_manifest.get("checkpoint"),
            "sha256": prediction_manifest.get("checkpoint_sha256"),
        },
        "judge": {
            "aggregate_path": relative_or_absolute(aggregate_path, workspace),
            "aggregate_sha256": aggregate_sha,
            "schema_version": aggregate.get("schema_version"),
            "version": aggregate.get("judge_version"),
            "num_warehouses": aggregate.get("num_warehouses"),
            "num_episodes": aggregate.get("num_episodes"),
            "num_frames": aggregate.get("num_frames"),
            "frame_weighted_source": aggregate.get("frame_weighted_metrics_source"),
        },
        "metrics": {
            "episode_macro": {
                "tool_accuracy": mean_metric(aggregate, "tool_accuracy"),
                "task_completion": mean_metric(aggregate, "task_completion"),
                "nav_deviation_m": mean_metric(aggregate, "nav_deviation_m"),
                "safety_violation_rate": mean_metric(aggregate, "safety_violation_rate"),
            },
            "frame_weighted": aggregate.get("frame_weighted_metrics", {}),
            "safety_violation_rate_v2": diagnostic_v2,
            "risk_adjacent_pooled": aggregate.get("safety_v2_risk_adjacent_pooled", {}),
        },
        "warnings": warnings,
    }
    return entry, warnings


def write_csv(path: Path, entries: list[dict[str, Any]]) -> None:
    fields = (
        "experiment_id",
        "model",
        "method",
        "dataset",
        "seed",
        "validity",
        "tool_accuracy_episode_macro",
        "task_completion_episode_macro",
        "safety_violation_rate_episode_macro",
        "num_episodes",
        "checkpoint_sha256",
        "prediction_tree_sha256",
        "aggregate_sha256",
        "artifact_root",
        "exclusion_reason",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in entries:
            macro = entry["metrics"]["episode_macro"]
            writer.writerow(
                {
                    "experiment_id": entry["experiment_id"],
                    "model": entry["model"],
                    "method": entry["method"],
                    "dataset": entry["dataset"],
                    "seed": entry["seed"],
                    "validity": entry["validity"],
                    "tool_accuracy_episode_macro": macro["tool_accuracy"],
                    "task_completion_episode_macro": macro["task_completion"],
                    "safety_violation_rate_episode_macro": macro["safety_violation_rate"],
                    "num_episodes": entry["judge"]["num_episodes"],
                    "checkpoint_sha256": entry["checkpoint"]["sha256"],
                    "prediction_tree_sha256": (entry["prediction"]["tree"] or {}).get("sha256"),
                    "aggregate_sha256": entry["judge"]["aggregate_sha256"],
                    "artifact_root": entry["artifact_root"],
                    "exclusion_reason": entry["exclusion_reason"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=Path("/data/oem4"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--strict", action="store_true", help="fail on missing or judge-contract warnings")
    args = parser.parse_args()

    workspace = args.workspace_root.resolve()
    entries: list[dict[str, Any]] = []
    hard_warnings: list[str] = []
    all_warnings: list[str] = []
    for spec in EXPERIMENTS:
        entry, warnings = build_entry(spec, workspace)
        entries.append(entry)
        for warning in warnings:
            qualified = f"{spec.experiment_id}: {warning}"
            all_warnings.append(qualified)
            if warning.startswith(("missing required artifact", "judge contract mismatch")):
                hard_warnings.append(qualified)

    document = {
        "schema_version": "saie_experiment_registry_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(workspace),
        "code": git_metadata(workspace / "runpod_trainer"),
        "identity_policy": (
            "directory + dataset manifest hash + checkpoint hash; never trust historical run tag alone"
        ),
        "summary": {
            "experiments": len(entries),
            "valid": sum(entry["validity"] == "valid" for entry in entries),
            "invalid_diagnostic": sum(entry["validity"] != "valid" for entry in entries),
            "hard_warnings": len(hard_warnings),
            "warnings": len(all_warnings),
        },
        "known_metric_provenance_risk": (
            "Current judge aggregates do not populate safety_violation_rate_v2. "
            "Verify the paper Table 2 Safety Viol. source before citation."
        ),
        "known_dataset_provenance_risk": (
            "The original and balanced dataset_manifest.json files are byte-identical despite "
            "different train JSONL contents. Identify them by canonical directory, train JSONL "
            "hash/row count, run directory/global step, and checkpoint hash, not manifest SHA alone."
        ),
        "experiments": entries,
        "warnings": all_warnings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    write_csv(csv_out, entries)
    print(
        f"wrote {args.out} and {csv_out}: "
        f"valid={document['summary']['valid']} invalid={document['summary']['invalid_diagnostic']} "
        f"hard_warnings={len(hard_warnings)}"
    )
    if args.strict and hard_warnings:
        for warning in hard_warnings:
            print(f"ERROR: {warning}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
