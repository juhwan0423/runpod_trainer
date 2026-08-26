#!/usr/bin/env python3
"""Paired episode-level comparison of two SAIE judge aggregates.

The aggregate files contain the same qualified episode IDs, so comparing
unpaired run-level means would throw away useful information.  This utility
reports paired deltas, a percentile bootstrap confidence interval, and a
Monte-Carlo sign-flip p-value without requiring SciPy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("tool_accuracy", "task_completion")


def load_episodes(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    episodes = aggregate.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"aggregate has no episodes list: {path}")
    indexed: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        key = episode.get("qualified_episode_id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"episode lacks qualified_episode_id: {path}")
        if key in indexed:
            raise ValueError(f"duplicate episode {key}: {path}")
        indexed[key] = episode
    return indexed, aggregate


def compare_metric(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
) -> dict[str, Any]:
    differences = candidate - baseline
    n = len(differences)
    if n < 2:
        raise ValueError("at least two paired episodes are required")

    bootstrap_means = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = min(5_000, bootstrap_samples)
    offset = 0
    while offset < bootstrap_samples:
        size = min(batch_size, bootstrap_samples - offset)
        indices = rng.integers(0, n, size=(size, n))
        bootstrap_means[offset : offset + size] = differences[indices].mean(axis=1)
        offset += size

    observed_abs = abs(float(differences.mean()))
    extreme = 0
    offset = 0
    while offset < permutation_samples:
        size = min(batch_size, permutation_samples - offset)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        permuted_abs = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted_abs >= observed_abs - 1e-15))
        offset += size

    return {
        "n": n,
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "paired_delta_mean": float(differences.mean()),
        "paired_delta_median": float(np.median(differences)),
        "paired_delta_sd": float(differences.std(ddof=1)),
        "bootstrap_ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "sign_flip_p_two_sided": float((extreme + 1) / (permutation_samples + 1)),
        "candidate_wins": int(np.count_nonzero(differences > 0)),
        "ties": int(np.count_nonzero(differences == 0)),
        "candidate_losses": int(np.count_nonzero(differences < 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.permutation_samples < 1:
        parser.error("sample counts must be positive")

    baseline, baseline_aggregate = load_episodes(args.baseline)
    candidate, candidate_aggregate = load_episodes(args.candidate)
    if baseline.keys() != candidate.keys():
        missing_from_candidate = sorted(baseline.keys() - candidate.keys())
        missing_from_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "episode sets differ: "
            f"missing_from_candidate={missing_from_candidate[:5]} "
            f"missing_from_baseline={missing_from_baseline[:5]}"
        )
    keys = sorted(baseline)
    rng = np.random.default_rng(args.seed)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        baseline_values = np.asarray([baseline[key][metric] for key in keys], dtype=np.float64)
        candidate_values = np.asarray([candidate[key][metric] for key in keys], dtype=np.float64)
        metrics[metric] = compare_metric(
            baseline_values,
            candidate_values,
            rng=rng,
            bootstrap_samples=args.bootstrap_samples,
            permutation_samples=args.permutation_samples,
        )

    result = {
        "schema_version": "saie_paired_episode_comparison_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "episode_set_equal": True,
        "num_episodes": len(keys),
        "judge_versions": {
            "baseline": baseline_aggregate.get("judge_version"),
            "candidate": candidate_aggregate.get("judge_version"),
        },
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for metric, values in metrics.items():
        low, high = values["bootstrap_ci95"]
        print(
            f"{metric}: delta={values['paired_delta_mean']:+.6f} "
            f"CI95=[{low:+.6f}, {high:+.6f}] "
            f"p={values['sign_flip_p_two_sided']:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
