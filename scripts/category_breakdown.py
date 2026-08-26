#!/usr/bin/env python3
"""Decompose scored predictions by task category and by individual label.

The judge's ``aggregate.json`` reports only corpus-level tool_accuracy /
task_completion / safety numbers. Those hide *where* a model actually fails:
this corpus is 83.4% ``[navigate]`` frames, so a model can post a high
tool_accuracy while getting every task-stop category (alert/gauge/count/state)
wrong -- exactly the trivial-baseline result already recorded in
``데이터셋_결함_및_개선점.md`` §2.4 (always-navigate = 83.4% tool_accuracy,
0.0% task_completion).

This reads the per-episode ``score_*.json`` files the judge writes (each has a
``per_frame`` list of ``{gt, pred, tool_match, ...}``) and reports, per result
directory:

  * per-category exact-match accuracy (pred string == gt string)
  * per-category tool_match rate (the judge's own looser tool-level check)
  * per-label exact-match accuracy, so navigate's internal spread is visible
  * the model's own prediction distribution, to expose mode collapse

Usage:
    python3 scripts/category_breakdown.py /data/oem4/_gemma4_lora_eval [...]
    python3 scripts/category_breakdown.py --all       # every _*_eval dir
    python3 scripts/category_breakdown.py --all --json out.json
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

CATEGORY_ORDER = ["navigate", "alert", "gauge", "count", "state"]


def category_of(label: str) -> str:
    """Return the bracket tag of a canonical label, e.g. '[gauge] stop' -> 'gauge'."""
    if label.startswith("[") and "]" in label:
        return label[1 : label.index("]")]
    return "<unparsed>"


def scores_root(result_dir: Path) -> Path | None:
    """Return the dir holding the per-warehouse score trees.

    Most eval dirs use ``scores/``; ``_gemma4_fft_results`` (a full training
    output dir rather than an eval-only one) uses ``judge_output/`` for the
    identical tree.
    """
    for name in ("scores", "judge_output"):
        if (result_dir / name).is_dir():
            return result_dir / name
    return None


def load_frames(result_dir: Path) -> list[dict]:
    """Collect every per-frame record under a result dir's score tree."""
    root = scores_root(result_dir)
    if root is None:
        return []
    pattern = str(root / "*" / "scores" / "*.json")
    frames: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            episode = json.load(handle)
        for frame in episode.get("per_frame", []):
            frames.append(frame)
    return frames


def analyse(result_dir: Path) -> dict | None:
    frames = load_frames(result_dir)
    if not frames:
        return None

    per_cat: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "exact": 0, "tool": 0}
    )
    per_label: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "exact": 0}
    )
    pred_counts: collections.Counter = collections.Counter()

    for frame in frames:
        gt = frame.get("gt", "")
        pred = frame.get("pred", "")
        cat = category_of(gt)
        exact = int(pred == gt)
        # tool_match is the judge's own field; absent on older score files.
        tool = int(bool(frame.get("tool_match", False)))

        per_cat[cat]["n"] += 1
        per_cat[cat]["exact"] += exact
        per_cat[cat]["tool"] += tool
        per_label[gt]["n"] += 1
        per_label[gt]["exact"] += exact
        pred_counts[pred] += 1

    total = len(frames)
    top_pred, top_pred_n = pred_counts.most_common(1)[0]

    # Task-stop = every non-navigate category pooled. This is the number the
    # trivial always-navigate baseline scores 0.0 on, so it is the honest
    # summary of whether a model learned anything beyond the majority class.
    stop_n = sum(v["n"] for k, v in per_cat.items() if k != "navigate")
    stop_exact = sum(v["exact"] for k, v in per_cat.items() if k != "navigate")

    return {
        "result_dir": str(result_dir),
        "num_frames": total,
        "num_distinct_predictions": len(pred_counts),
        "top_prediction": top_pred,
        "top_prediction_share": round(top_pred_n / total, 4),
        "overall_exact": round(
            sum(v["exact"] for v in per_cat.values()) / total, 4
        ),
        "task_stop_pooled_exact": round(stop_exact / stop_n, 4) if stop_n else None,
        "per_category": {
            cat: {
                "n": per_cat[cat]["n"],
                "exact": round(per_cat[cat]["exact"] / per_cat[cat]["n"], 4),
                "tool_match": round(per_cat[cat]["tool"] / per_cat[cat]["n"], 4),
            }
            for cat in CATEGORY_ORDER
            if cat in per_cat
        },
        "per_label": {
            label: {
                "n": stats["n"],
                "exact": round(stats["exact"] / stats["n"], 4),
            }
            for label, stats in sorted(
                per_label.items(), key=lambda kv: -kv[1]["n"]
            )
        },
    }


def print_report(report: dict) -> None:
    name = Path(report["result_dir"]).name
    print(f"\n{'=' * 78}")
    print(f"{name}   ({report['num_frames']} frames)")
    print("=" * 78)
    print(
        f"overall exact-match : {report['overall_exact']:.4f}    "
        f"task-stop pooled : "
        + (
            f"{report['task_stop_pooled_exact']:.4f}"
            if report["task_stop_pooled_exact"] is not None
            else "n/a"
        )
    )
    print(
        f"distinct preds      : {report['num_distinct_predictions']:<4d}   "
        f"most-frequent pred : {report['top_prediction_share']:.1%}  "
        f"{report['top_prediction']!r}"
    )
    print(f"\n  {'category':<10} {'n':>6}  {'exact':>7}  {'tool_match':>10}")
    print(f"  {'-' * 10} {'-' * 6}  {'-' * 7}  {'-' * 10}")
    for cat, stats in report["per_category"].items():
        print(
            f"  {cat:<10} {stats['n']:>6}  {stats['exact']:>7.4f}  "
            f"{stats['tool_match']:>10.4f}"
        )
    print(f"\n  {'label':<36} {'n':>6}  {'exact':>7}")
    print(f"  {'-' * 36} {'-' * 6}  {'-' * 7}")
    for label, stats in report["per_label"].items():
        print(f"  {label:<36} {stats['n']:>6}  {stats['exact']:>7.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="analyse every /data/oem4/_*_eval directory that has scores/",
    )
    parser.add_argument("--json", type=Path, help="also write reports to this JSON file")
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the per-label tables"
    )
    args = parser.parse_args()

    dirs = list(args.dirs)
    if args.all:
        dirs.extend(
            Path(p)
            for p in sorted(
                glob.glob("/data/oem4/_*_eval") + glob.glob("/data/oem4/_*_results")
            )
            if scores_root(Path(p)) is not None
        )
    if not dirs:
        parser.error("pass result directories, or --all")

    reports = []
    for result_dir in dirs:
        report = analyse(result_dir)
        if report is None:
            print(f"[skip] no per-frame score files under {result_dir}", file=sys.stderr)
            continue
        reports.append(report)
        if args.quiet:
            report_view = dict(report)
            report_view["per_label"] = {}
            print_report(report_view)
        else:
            print_report(report)

    if args.json:
        args.json.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {args.json} ({len(reports)} reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
