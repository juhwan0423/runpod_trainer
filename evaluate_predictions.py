#!/usr/bin/env python3
"""Compute label-level SAIE metrics and transparent trivial baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.schema import ACTION_LABELS, ACTION_LABEL_SET, read_jsonl  # noqa: E402


INVALID = "__invalid__"
ACTION_GROUPS = ("navigate", "gauge", "count", "state", "alert")


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def normalise_prediction(value: Any, extract: bool) -> str:
    text = str(value or "").strip()
    if text in ACTION_LABEL_SET:
        return text
    if extract:
        matches = [label for label in ACTION_LABELS if label in text]
        if len(matches) == 1:
            return matches[0]
    return INVALID


def prediction_value(row: dict[str, Any], extract: bool) -> Any:
    """Choose strict accepted output, or raw text only in diagnostic mode."""

    if extract:
        return first_value(
            row,
            (
                "native_action",
                "raw_model_output",
                "raw",
                "prediction",
                "label",
                "normalized_action_label",
                "pred",
            ),
        )
    if "pred" in row:
        # An explicit JSON null is a deliberate strict-format rejection; do
        # not fall through to the diagnostic normalized_action_label.
        return row["pred"]
    if row.get("parse_status") not in (None, "exact"):
        return None
    return first_value(row, ("prediction", "label", "normalized_action_label"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold/pred must be non-empty and aligned")
    columns = (*ACTION_LABELS, INVALID)
    matrix = {label: Counter() for label in ACTION_LABELS}
    for expected, predicted in zip(gold, pred):
        if expected not in ACTION_LABEL_SET:
            raise ValueError(f"non-canonical gold label: {expected!r}")
        matrix[expected][predicted] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    present_recalls: list[float] = []
    f1_all: list[float] = []
    f1_present: list[float] = []
    for label in ACTION_LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[other][label] for other in ACTION_LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_all.append(f1)
        if support:
            present_recalls.append(recall)
            f1_present.append(f1)

    correct = sum(expected == predicted for expected, predicted in zip(gold, pred))
    return {
        "n": len(gold),
        "accuracy": correct / len(gold),
        "balanced_accuracy_present_classes": sum(present_recalls) / len(present_recalls),
        "macro_f1_14_classes": sum(f1_all) / len(ACTION_LABELS),
        "macro_f1_present_classes": sum(f1_present) / len(f1_present),
        "invalid_prediction_rate": sum(value == INVALID for value in pred) / len(pred),
        "per_class": per_class,
        "confusion": {
            expected: {predicted: matrix[expected][predicted] for predicted in columns}
            for expected in ACTION_LABELS
        },
    }


def action_group_or_invalid(label: str) -> str:
    if label == INVALID:
        return INVALID
    if label not in ACTION_LABEL_SET:
        raise ValueError(f"cannot derive category from non-canonical label: {label!r}")
    return label[1 : label.index("]")]


def compute_category_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    """Aggregate the 14 canonical labels into the benchmark's five tools."""

    gold_groups = [action_group_or_invalid(value) for value in gold]
    pred_groups = [action_group_or_invalid(value) for value in pred]
    columns = (*ACTION_GROUPS, INVALID)
    matrix = {group: Counter() for group in ACTION_GROUPS}
    for expected, predicted in zip(gold_groups, pred_groups):
        matrix[expected][predicted] += 1

    per_group: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    f1s: list[float] = []
    for group in ACTION_GROUPS:
        tp = matrix[group][group]
        support = sum(matrix[group].values())
        predicted_count = sum(matrix[other][group] for other in ACTION_GROUPS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_group[group] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support:
            recalls.append(recall)
            f1s.append(f1)

    correct = sum(expected == predicted for expected, predicted in zip(gold_groups, pred_groups))
    return {
        "n": len(gold_groups),
        "accuracy": correct / len(gold_groups),
        "balanced_accuracy_present_groups": sum(recalls) / len(recalls),
        "macro_f1_present_groups": sum(f1s) / len(f1s),
        "macro_f1_5_groups": sum(per_group[group]["f1"] for group in ACTION_GROUPS)
        / len(ACTION_GROUPS),
        "invalid_prediction_rate": sum(value == INVALID for value in pred_groups)
        / len(pred_groups),
        "per_group": per_group,
        "confusion": {
            expected: {predicted: matrix[expected][predicted] for predicted in columns}
            for expected in ACTION_GROUPS
        },
    }


def metric_suite(gold: list[str], pred: list[str]) -> dict[str, Any]:
    label_metrics = compute_metrics(gold, pred)
    label_metrics["category_metrics"] = compute_category_metrics(gold, pred)
    return label_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        required=True,
        help="authoritative test/val JSONL; gold labels in predictions are never trusted",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="explicit diagnostic mode for predictions covering only part of the reference",
    )
    parser.add_argument("--train-jsonl", type=Path, help="enables non-leaky train-majority/prior baselines")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract-canonical", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.predictions)
    reference_rows = read_jsonl(args.reference_jsonl)
    reference_by_id: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every reference row must have a non-empty sample_id")
        if sample_id in reference_by_id:
            raise ValueError(f"duplicate sample_id in reference JSONL: {sample_id}")
        reference_by_id[sample_id] = row
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("every prediction row must have a non-empty sample_id")
        if sample_id in prediction_by_id:
            raise ValueError(f"duplicate sample_id in predictions: {sample_id}")
        if sample_id not in reference_by_id:
            raise ValueError(f"prediction sample_id is absent from reference: {sample_id}")
        prediction_by_id[sample_id] = row
    missing_ids = sorted(set(reference_by_id) - set(prediction_by_id))
    if missing_ids and not args.allow_subset:
        raise ValueError(
            f"predictions are incomplete: missing={len(missing_ids)} examples={missing_ids[:5]}"
        )
    ordered_ids = [
        row["sample_id"] for row in reference_rows if row["sample_id"] in prediction_by_id
    ]
    rows = [prediction_by_id[sample_id] for sample_id in ordered_ids]
    gold = []
    for sample_id, prediction_row in zip(ordered_ids, rows):
        reference_label = reference_by_id[sample_id].get("action_label")
        if reference_label not in ACTION_LABEL_SET:
            raise ValueError(f"reference has non-canonical label for {sample_id}: {reference_label!r}")
        claimed_gold = first_value(
            prediction_row, ("gold", "action_label", "a", "reference_action_label")
        )
        if claimed_gold is not None and claimed_gold != reference_label:
            raise ValueError(
                f"prediction gold disagrees with reference for {sample_id}: "
                f"{claimed_gold!r} != {reference_label!r}"
            )
        gold.append(reference_label)
    raw_pred = [prediction_value(row, args.extract_canonical) for row in rows]
    pred = [normalise_prediction(value, args.extract_canonical) for value in raw_pred]
    result: dict[str, Any] = {
        "schema_version": 1,
        "prediction_file": str(args.predictions.resolve()),
        "reference_file": str(args.reference_jsonl.resolve()),
        "reference_sha256": file_sha256(args.reference_jsonl),
        "reference_rows": len(reference_rows),
        "evaluated_rows": len(rows),
        "missing_reference_rows": len(missing_ids),
        "allow_subset": args.allow_subset,
        "normalization_mode": (
            "diagnostic_extract_single_canonical_substring"
            if args.extract_canonical
            else "strict_accepted_prediction_only"
        ),
        "parse_status_counts": dict(
            sorted(Counter(str(row.get("parse_status") or "unrecorded") for row in rows).items())
        ),
        "prediction_metrics": metric_suite(gold, pred),
        "trivial_baselines": {},
    }

    fixed = {
        "always_forward_75": "[navigate] move forward 75 cm",
        "always_navigate_stop": "[navigate] stop",
        "always_alert_stop": "[alert] stop",
    }
    for name, label in fixed.items():
        result["trivial_baselines"][name] = metric_suite(gold, [label] * len(gold))

    rng = random.Random(args.seed)
    result["trivial_baselines"]["uniform_random_14"] = metric_suite(
        gold, [rng.choice(ACTION_LABELS) for _ in gold]
    )
    if args.train_jsonl:
        train_rows = read_jsonl(args.train_jsonl)
        train_counts = Counter(row.get("action_label") for row in train_rows)
        invalid_train = set(train_counts) - ACTION_LABEL_SET
        if invalid_train:
            raise ValueError(f"train JSONL has invalid labels: {sorted(invalid_train)}")
        majority = max(ACTION_LABELS, key=lambda label: train_counts[label])
        result["train_label_counts"] = {label: train_counts[label] for label in ACTION_LABELS}
        result["trivial_baselines"]["train_majority"] = metric_suite(
            gold, [majority] * len(gold)
        )
        population = list(ACTION_LABELS)
        weights = [train_counts[label] for label in population]
        rng = random.Random(args.seed)
        result["trivial_baselines"]["train_prior_random"] = metric_suite(
            gold, rng.choices(population, weights=weights, k=len(gold))
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = result["prediction_metrics"]
    categories = metrics["category_metrics"]
    print(
        f"n={metrics['n']} accuracy={metrics['accuracy']:.6f} "
        f"balanced_acc={metrics['balanced_accuracy_present_classes']:.6f} "
        f"macro_f1={metrics['macro_f1_14_classes']:.6f} "
        f"category_acc={categories['accuracy']:.6f} "
        f"category_macro_f1={categories['macro_f1_5_groups']:.6f} "
        f"invalid={metrics['invalid_prediction_rate']:.6f}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
