#!/usr/bin/env python3
"""Aggregate warehouse-qualified SAIE judge scores without losing denominators."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "tool_accuracy",
    "nav_deviation_m",
    "safety_violation_rate",
    "task_completion",
)
DIAGNOSTIC_METRICS = (
    "task_completion_strict",
    "safety_violation_rate_v2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-root",
        type=Path,
        required=True,
        help="root containing <warehouse>/scores/score_*.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = [
        (float(row[key]), int(row["num_frames"]))
        for row in rows
        if row.get(key) is not None and int(row.get("num_frames") or 0) > 0
    ]
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator if denominator else None


def _frame_weighted_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, float | None], str]:
    per_frame = [row.get("per_frame") for row in rows]
    if all(
        isinstance(frames, list) and len(frames) == int(row["num_frames"])
        for row, frames in zip(rows, per_frame)
    ):
        flattened = [frame for frames in per_frame for frame in frames]
        required = ("tool_match", "nav_deviation_m", "safety_violated")
        if flattened and all(
            isinstance(frame, dict) and all(key in frame for key in required)
            for frame in flattened
        ):
            return (
                {
                    "tool_accuracy": sum(bool(frame["tool_match"]) for frame in flattened)
                    / len(flattened),
                    "nav_deviation_m": sum(float(frame["nav_deviation_m"]) for frame in flattened)
                    / len(flattened),
                    "safety_violation_rate": sum(
                        bool(frame["safety_violated"]) for frame in flattened
                    )
                    / len(flattened),
                },
                "exact_from_per_frame_records",
            )
    return (
        {
            key: _weighted_mean(rows, key)
            for key in ("tool_accuracy", "nav_deviation_m", "safety_violation_rate")
        },
        "weighted_from_rounded_episode_metrics",
    )


def aggregate(scores_root: Path) -> dict[str, Any]:
    score_paths = sorted(scores_root.glob("warehouse_s*/scores/score_*.json"))
    if not score_paths:
        raise ValueError(f"no warehouse-qualified score files under {scores_root}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    versions: set[int] = set()
    for path in score_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        warehouse_id = path.parents[1].name
        episode_id = payload.get("episode_id") or path.stem.removeprefix("score_")
        qualified_id = f"{warehouse_id}/{episode_id}"
        if qualified_id in seen:
            raise ValueError(f"duplicate warehouse-qualified score: {qualified_id}")
        seen.add(qualified_id)
        if not isinstance(payload.get("version"), int):
            raise ValueError(f"score has no integer version: {path}")
        versions.add(payload["version"])
        if not isinstance(payload.get("num_frames"), int) or payload["num_frames"] <= 0:
            raise ValueError(f"score has invalid num_frames: {path}")
        for key in METRICS:
            if not isinstance(payload.get(key), (int, float)):
                raise ValueError(f"score is missing numeric {key}: {path}")
        for key in ("tool_accuracy", "safety_violation_rate", "task_completion"):
            if not 0.0 <= float(payload[key]) <= 1.0:
                raise ValueError(f"score has out-of-range {key}: {path}")
        for key in DIAGNOSTIC_METRICS:
            if payload.get(key) is not None and not 0.0 <= float(payload[key]) <= 1.0:
                raise ValueError(f"score has out-of-range {key}: {path}")
        if float(payload["nav_deviation_m"]) < 0:
            raise ValueError(f"score has negative nav_deviation_m: {path}")
        risky_frames = int(payload.get("safety_v2_n_risky_frames") or 0)
        violating_frames = int(payload.get("safety_v2_n_violations") or 0)
        if risky_frames < 0 or violating_frames < 0 or violating_frames > risky_frames:
            raise ValueError(f"score has invalid pooled safety-v2 counts: {path}")
        rows.append(
            {
                **payload,
                "warehouse_id": warehouse_id,
                "qualified_episode_id": qualified_id,
                "score_file": str(path.resolve()),
            }
        )
    if len(versions) != 1:
        raise ValueError(f"mixed judge versions are not comparable: {sorted(versions)}")

    distributions = {
        key: _summary([float(row[key]) for row in rows]) for key in METRICS
    }
    diagnostic_distributions = {
        key: _summary([float(row[key]) for row in rows if row.get(key) is not None])
        for key in DIAGNOSTIC_METRICS
    }
    frame_weighted, frame_weighted_source = _frame_weighted_metrics(rows)
    risky = sum(int(row.get("safety_v2_n_risky_frames") or 0) for row in rows)
    violations = sum(int(row.get("safety_v2_n_violations") or 0) for row in rows)
    safety_v2 = {
        "n_risky_frames": risky,
        "n_violations": violations,
        "pooled_violation_rate": violations / risky if risky else None,
        "aggregation": "sum(n_violations) / sum(n_risky_frames)",
    }
    warehouse_counts: dict[str, int] = {}
    for row in rows:
        warehouse_counts[row["warehouse_id"]] = warehouse_counts.get(row["warehouse_id"], 0) + 1

    return {
        "schema_version": "saie_judge_aggregate_v1",
        "judge_version": versions.pop(),
        "scores_root": str(scores_root.resolve()),
        "num_warehouses": len(warehouse_counts),
        "num_episodes": len(rows),
        "num_frames": sum(int(row["num_frames"]) for row in rows),
        "warehouse_episode_counts": dict(sorted(warehouse_counts.items())),
        "episode_macro_distributions": distributions,
        "diagnostic_episode_distributions": diagnostic_distributions,
        "frame_weighted_metrics": frame_weighted,
        "frame_weighted_metrics_source": frame_weighted_source,
        "safety_v2_risk_adjacent_pooled": safety_v2,
        "episodes": [
            {
                "qualified_episode_id": row["qualified_episode_id"],
                "warehouse_id": row["warehouse_id"],
                "episode_id": row.get("episode_id"),
                "num_frames": row["num_frames"],
                **{key: row[key] for key in METRICS},
                "task_completion_strict": row.get("task_completion_strict"),
                "safety_violation_rate_v2": row.get("safety_violation_rate_v2"),
                "safety_v2_n_risky_frames": row.get("safety_v2_n_risky_frames"),
                "safety_v2_n_violations": row.get("safety_v2_n_violations"),
            }
            for row in rows
        ],
    }


def main() -> int:
    args = parse_args()
    result = aggregate(args.scores_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"aggregated warehouses={result['num_warehouses']} "
        f"episodes={result['num_episodes']} frames={result['num_frames']} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
