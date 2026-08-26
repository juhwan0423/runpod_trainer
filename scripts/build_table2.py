#!/usr/bin/env python3
"""Build the redesigned Table-2 comparison from judge aggregates + category breakdown.

Why this exists
---------------
The paper's Table 2 reports four columns: Category Acc., Task Comp., Nav. Dev.,
Safety Viol. Auditing this session's 15 scored SFT runs showed that three of
those four are not usable as reported for the SFT results:

* **Category Acc. (= judge tool_accuracy) does not discriminate inside
  navigate.** navigate exact-match runs 9-34% while tool_match on the *same*
  frames runs 84-97%: the metric checks the ``[navigate]`` tag, not which
  navigate action was chosen. Since the corpus is 83.4% navigate, an
  always-navigate baseline scores 83.4% (see 데이터셋_결함_및_개선점.md §2.4).
* **Nav. Dev. is ~0 for nearly every frame.** In ``_gemma4_lora_eval`` only 268
  of 10,646 mismatched predictions (2.5%) carry a nonzero deviation --
  predicting "turn right 15" when the gold is "turn left 30" scores 0.0.
* **Safety Viol. (v2) is undefined.** All 15 runs report
  ``safety_v2_risk_adjacent_pooled.n_risky_frames == 0``, so the pooled rate is
  ``None`` on every one.

So this script reports the judge's own numbers *and* the decomposed numbers
side by side, and flags the degenerate columns rather than quietly printing
them as if they were comparable to the paper's zero-shot row.

Usage:
    python3 scripts/build_table2.py                       # markdown to stdout
    python3 scripts/build_table2.py --out /data/oem4/_table2.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

# Zero-shot rows exactly as printed in the paper (main.tex Table 2).
# Kept verbatim for reference; see the caveat block in render() about why these
# are NOT directly comparable to the SFT rows below.
PAPER_ZERO_SHOT = [
    ("Gemma 4-12B-it (4-bit)", 0.357, 0.592, 0.388, 0.591),
    ("NaVILA", 0.930, 0.001, 0.373, 0.533),
    ("OmniVLA", 0.930, 0.000, 0.388, 0.450),
    ("InternVLA-M1", 0.930, 0.090, 0.300, 0.411),
]

# Human-readable (model, method, data) for each result directory. Anything not
# listed here still gets rendered, just with the raw directory name.
RUN_LABELS = {
    "_gemma4_fft_results": ("Gemma4", "FFT", "불균형"),
    "_gemma4_lora_unbalanced_eval": ("Gemma4", "LoRA", "불균형"),
    "_gemma4_lora_eval": ("Gemma4", "LoRA", "밸런스"),
    "_gemma4_fft_balanced_eval": ("Gemma4", "FFT", "밸런스"),
    "_gemma4_lora_redesigned_seed0_eval": ("Gemma4", "LoRA", "재설계 s0"),
    "_gemma4_lora_redesigned_seed1_eval": ("Gemma4", "LoRA", "재설계 s1"),
    "_gemma4_lora_redesigned_seed2_eval": ("Gemma4", "LoRA", "재설계 s2"),
    "_navila_eval": ("NaVILA", "FFT", "불균형"),
    "_navila_lora_eval": ("NaVILA", "LoRA", "불균형"),
    "_navila_lora_balanced_eval": ("NaVILA", "LoRA", "밸런스"),
    "_navila_fft_balanced_eval": ("NaVILA", "FFT", "밸런스"),
    "_internvla_n1_fft_fixed_eval": ("InternVLA-N1", "FFT", "불균형"),
    "_internvla_n1_lora_unbalanced_eval": ("InternVLA-N1", "LoRA", "불균형"),
    "_internvla_n1_lora_fixed_eval": ("InternVLA-N1", "LoRA", "밸런스"),
    "_internvla_m1_lora_unbalanced_eval": ("InternVLA-M1", "LoRA", "불균형"),
    "_internvla_m1_lora_balanced_eval": ("InternVLA-M1", "LoRA", "밸런스"),
    "_internvla_m1_fft_balanced_eval": ("InternVLA-M1", "FFT", "밸런스"),
    # Superseded by the *_fixed_* runs -- these three predate the
    # parse_native_response() fix and scored a constant "[error] stop"
    # (데이터셋_결함_및_개선점.md §9.1), so they are excluded by default.
    "_internvla_n1_eval": ("InternVLA-N1", "FFT", "불균형(버그)"),
    "_internvla_n1_lora_eval": ("InternVLA-N1", "LoRA", "불균형(버그)"),
}

SUPERSEDED = {"_internvla_n1_eval", "_internvla_n1_lora_eval"}

CATEGORIES = ["navigate", "alert", "gauge", "count", "state"]


def category_of(label: str) -> str:
    if label.startswith("[") and "]" in label:
        return label[1 : label.index("]")]
    return "<unparsed>"


def scores_root(result_dir: Path) -> Path | None:
    """Return the dir holding aggregate.json + per-warehouse score trees.

    Two layouts exist side by side in /data/oem4: most eval dirs put this under
    ``scores/``, but the earliest run (``_gemma4_fft_results``, which is a full
    training output dir rather than an eval-only one) puts the identical tree
    under ``judge_output/``. Same contents, different parent name.
    """
    for name in ("scores", "judge_output"):
        if (result_dir / name / "aggregate.json").is_file():
            return result_dir / name
    return None


def collect(result_dir: Path) -> dict | None:
    """Read one result dir's aggregate.json plus its per-frame score files."""
    root = scores_root(result_dir)
    if root is None:
        return None
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))

    per_cat: dict[str, list[int]] = {c: [0, 0, 0] for c in CATEGORIES}  # n, exact, tool
    preds: dict[str, int] = {}
    frames = 0
    for path in glob.glob(str(root / "*" / "scores" / "*.json")):
        episode = json.loads(Path(path).read_text(encoding="utf-8"))
        for frame in episode.get("per_frame", []):
            frames += 1
            gold = frame.get("gt", "")
            pred = frame.get("pred", "")
            cat = category_of(gold)
            if cat in per_cat:
                per_cat[cat][0] += 1
                per_cat[cat][1] += int(pred == gold)
                per_cat[cat][2] += int(bool(frame.get("tool_match", False)))
            preds[pred] = preds.get(pred, 0) + 1
    if not frames:
        return None

    stop_n = sum(per_cat[c][0] for c in CATEGORIES if c != "navigate")
    stop_exact = sum(per_cat[c][1] for c in CATEGORIES if c != "navigate")
    top_pred, top_n = max(preds.items(), key=lambda kv: kv[1])

    macro = aggregate["episode_macro_distributions"]
    return {
        "dir": result_dir.name,
        "frames": frames,
        "tool_accuracy": macro["tool_accuracy"]["mean"],
        "task_completion": macro["task_completion"]["mean"],
        "nav_deviation_m": macro["nav_deviation_m"]["mean"],
        "safety_v2_risky_frames": aggregate["safety_v2_risk_adjacent_pooled"][
            "n_risky_frames"
        ],
        "safety_v2": aggregate["safety_v2_risk_adjacent_pooled"]["pooled_violation_rate"],
        "exact": sum(per_cat[c][1] for c in CATEGORIES) / frames,
        "task_stop": (stop_exact / stop_n) if stop_n else None,
        "nav_exact": per_cat["navigate"][1] / per_cat["navigate"][0],
        "nav_tool": per_cat["navigate"][2] / per_cat["navigate"][0],
        "per_cat": {
            c: (per_cat[c][1] / per_cat[c][0]) if per_cat[c][0] else None
            for c in CATEGORIES
        },
        "n_pred": len(preds),
        "top_share": top_n / frames,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render(rows: list[dict]) -> str:
    out: list[str] = []
    out.append("# Table 2 (재설계) — zero-shot 원본 + 이번 세션 SFT 결과\n")
    out.append(
        "자동 생성: `scripts/build_table2.py`. 근거와 해석은 "
        "`데이터셋_결함_및_개선점.md` §9.5 / §9.6 참고.\n"
    )

    out.append("\n## A. 논문 원본 Table 2 (zero-shot, main.tex)\n")
    out.append("| Model | Category Acc. | Task Comp. | Nav. Dev. | Safety Viol. |")
    out.append("|---|---:|---:|---:|---:|")
    for name, cat, task, nav, safety in PAPER_ZERO_SHOT:
        out.append(f"| {name} | {cat:.3f} | {task:.3f} | {nav:.3f} | {safety:.3f} |")

    out.append(
        "\n> ⚠️ **아래 SFT 결과와 직접 비교 금지.** 이번 세션의 채점에서는 "
        "`Nav. Dev.`가 오답의 97.5%에서 0.0으로 나오고, `Safety Viol.`(v2)은 "
        "15개 결과셋 전부 `n_risky_frames=0`이라 값 자체가 `None`이다. 두 열은 "
        "SFT 쪽에서 재현되지 않으므로, 같은 축에 올리려면 채점 설정 차이부터 "
        "규명해야 한다(§9.6).\n"
    )

    out.append("\n## B. 이번 세션 SFT 결과 — judge 총계 지표\n")
    out.append(
        "| 모델 | 방법 | 데이터 | Category Acc.<br>(tool_acc) | Task Comp. | "
        "Nav. Dev. | Safety v2 |"
    )
    out.append("|---|---|---|---:|---:|---:|---:|")
    for row in rows:
        model, method, data = RUN_LABELS.get(row["dir"], (row["dir"], "", ""))
        safety = (
            "undef." if row["safety_v2"] is None else f"{row['safety_v2']:.3f}"
        )
        out.append(
            f"| {model} | {method} | {data} | {row['tool_accuracy']:.3f} | "
            f"{row['task_completion']:.3f} | {row['nav_deviation_m']:.5f} | {safety} |"
        )

    out.append("\n## C. 카테고리 분해 — 실제 변별력이 있는 지표\n")
    out.append(
        "`exact`는 라벨 문자열 완전일치. `task-stop`은 navigate를 제외한 "
        "4개 카테고리를 pooled한 값으로, always-navigate trivial baseline이 "
        "정확히 0.000을 받는 지표다(§2.4).\n"
    )
    out.append(
        "| 모델 | 방법 | 데이터 | exact | **task-stop** | nav exact | "
        "nav tool_match | alert | gauge | count | state | #pred | 최빈예측 |"
    )
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        model, method, data = RUN_LABELS.get(row["dir"], (row["dir"], "", ""))
        cats = " | ".join(
            fmt(row["per_cat"][c]) for c in ["alert", "gauge", "count", "state"]
        )
        out.append(
            f"| {model} | {method} | {data} | {row['exact']:.3f} | "
            f"**{fmt(row['task_stop'])}** | {row['nav_exact']:.3f} | "
            f"{row['nav_tool']:.3f} | {cats} | {row['n_pred']} | "
            f"{row['top_share']:.1%} |"
        )

    out.append(
        "\n> `nav exact` vs `nav tool_match` 열을 나란히 보면 Category Acc.의 "
        "문제가 한눈에 보인다: 같은 프레임에서 9~34% vs 84~97%. tool_match는 "
        "`[navigate]` 태그만 확인하고 어떤 navigate 행동인지는 보지 않는다.\n"
    )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/oem4"))
    parser.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    parser.add_argument(
        "--include-superseded",
        action="store_true",
        help="also include the pre-parse-fix InternVLA-N1 runs (§9.1)",
    )
    args = parser.parse_args()

    candidates = sorted(glob.glob(str(args.root / "_*_eval"))) + sorted(
        glob.glob(str(args.root / "_*_results"))
    )
    rows = []
    for path in candidates:
        name = os.path.basename(path)
        if name in SUPERSEDED and not args.include_superseded:
            continue
        row = collect(Path(path))
        if row is not None:
            rows.append(row)

    # Group by model in RUN_LABELS declaration order so the table reads as a
    # factorial block per model rather than alphabetically by directory.
    order = {name: i for i, name in enumerate(RUN_LABELS)}
    rows.sort(key=lambda r: order.get(r["dir"], len(order)))

    text = render(rows)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} runs)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
