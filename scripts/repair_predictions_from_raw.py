#!/usr/bin/env python3
"""Rebuild judge-ready vlm_output from a prediction run's stored raw text.

When ``predict_labels.py`` discards a model's answer at the parsing stage, the
run is still recoverable: ``test_predictions.jsonl`` records
``raw_model_output`` (the full decode, special tokens intact) for every frame.
This re-derives ``native_action`` -> ``canonical_from_raw`` -> ``label`` from
that stored text and rewrites the per-episode ``vlm_output`` JSON files the
judge consumes -- no GPU, no re-inference.

Used 2026-08-26 to recover both InternVLA-M1 LoRA runs. Their processor exposes
``parse_response`` but calling it raises ``AttributeError: This tokenizer does
not have a `response_schema` ...``, so all 12,523 frames were written as
``[error] stop`` while ``raw_model_output`` held clean labels throughout -- the
same artifact as 데이터셋_결함_및_개선점.md §9.1, in a different code branch.
``predict_labels.py`` is fixed for future runs; this repairs the finished ones.

Usage:
    python3 scripts/repair_predictions_from_raw.py \
        --predictions /path/to/test_predictions.jsonl \
        --vlm-output-dir /data/oem4/_internvla_m1_lora_balanced_eval/vlm_output
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import ACTION_LABELS rather than predict_labels.canonical_from_raw itself:
# importing that module pulls in torch, which this analysis box does not have.
# canonical_from_raw is a pure-text function over ACTION_LABELS, reproduced
# below verbatim so the two cannot silently drift apart in behaviour.
from data.schema import ACTION_LABELS  # noqa: E402


def canonical_from_raw(raw: str | None) -> tuple[str | None, str]:
    """Verbatim copy of predict_labels.canonical_from_raw (no torch needed)."""
    text = str(raw or "").strip()
    if text in ACTION_LABELS:
        return text, "exact"
    matches = [label for label in ACTION_LABELS if label in text]
    return (matches[0], "extracted") if len(matches) == 1 else (None, "invalid")

# Strip the chat-template end-of-turn markers the raw decode keeps. The set is
# deliberately narrow: anything else stays in place so canonical_from_raw sees
# the model's real text and can honestly report a non-exact parse.
_EOT_RE = re.compile(
    r"(<\|im_end\|>|<\|endoftext\|>|<end_of_turn>|<eos>|</s>)\s*$"
)


def native_from_raw(raw: str | None) -> str | None:
    if not raw:
        return None
    text = raw
    while True:
        stripped = _EOT_RE.sub("", text).strip()
        if stripped == text.strip():
            return stripped or None
        text = stripped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--vlm-output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-extracted-label",
        action="store_true",
        help="accept parse_status == 'extracted' as well as 'exact'",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        parser.error(f"no rows in {args.predictions}")

    episodes: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    status_counts: collections.Counter = collections.Counter()
    before_labels: collections.Counter = collections.Counter()
    after_labels: collections.Counter = collections.Counter()
    normalization_rule = rows[0].get("normalization_rule")

    for row in rows:
        native = native_from_raw(row.get("raw_model_output"))
        normalized, parse_status = canonical_from_raw(native)
        accepted = (
            normalized
            if parse_status == "exact"
            or (args.allow_extracted_label and parse_status == "extracted")
            else None
        )
        status_counts[parse_status] += 1
        before_labels[row.get("pred") or "[error] stop"] += 1
        after_labels[accepted or "[error] stop"] += 1

        source_frame_id = row.get("source_frame_id")
        if source_frame_id is None:
            source_frame_id = row.get("frame_id")
        episodes[(row["warehouse_id"], row["source_episode_id"])].append(
            {
                "frame_id": source_frame_id,
                "observation_frame_id": row.get("frame_id"),
                "t": source_frame_id / 30.0,
                "label": accepted or "[error] stop",
                "raw": row.get("raw_model_output"),
                "native_action": native,
                "parse_status": parse_status,
                "normalization_rule": normalization_rule,
                "used_ground_truth_context": False,
            }
        )

    print(f"rows: {len(rows)}   episodes: {len(episodes)}")
    print(f"parse_status: {dict(status_counts)}")
    print(f"\nbefore -- distinct labels: {len(before_labels)}")
    for label, count in before_labels.most_common(5):
        print(f"  {count:6d}  {label}")
    print(f"\nafter  -- distinct labels: {len(after_labels)}")
    for label, count in after_labels.most_common(8):
        print(f"  {count:6d}  {label}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    for (warehouse_id, episode_id), records in episodes.items():
        records.sort(key=lambda r: r["frame_id"])
        destination = args.vlm_output_dir / warehouse_id / f"{episode_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"\nwrote {len(episodes)} episode files under {args.vlm_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
