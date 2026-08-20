"""Canonical SAIE action vocabulary and JSONL helpers.

This module intentionally mirrors ``SAIE/src/export_benchmark_annotations.py``.
Training and evaluation import one immutable list from here so that a spelling
change cannot silently create a fifteenth class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


ACTION_LABELS = (
    "[navigate] move forward 25 cm",
    "[navigate] move forward 50 cm",
    "[navigate] move forward 75 cm",
    "[navigate] turn left 15 degree",
    "[navigate] turn left 30 degree",
    "[navigate] turn left 45 degree",
    "[navigate] turn right 15 degree",
    "[navigate] turn right 30 degree",
    "[navigate] turn right 45 degree",
    "[navigate] stop",
    "[gauge] stop",
    "[count] stop",
    "[state] stop",
    "[alert] stop",
)

ACTION_LABEL_SET = frozenset(ACTION_LABELS)
PROMPT_VERSION = "saie_action14_sft_v1"


def render_policy_prompt(mission: str, num_frames: int) -> str:
    """Render the fixed, model-visible policy contract around episode ``q``."""
    temporal = (
        "The images are ordered from oldest to newest; the last image is the current observation."
        if num_frames > 1
        else "The image is the robot's current front-camera observation."
    )
    labels = "\n".join(f"- {label}" for label in ACTION_LABELS)
    return (
        "You are the policy for a quadruped robot performing an industrial safety patrol.\n"
        f"{temporal}\n"
        "Follow the mission, react to visible safety hazards before resuming the patrol, and "
        "keep a safe distance from a fallen worker.\n\n"
        f"MISSION:\n{mission.strip()}\n\n"
        "Output exactly ONE canonical action label from the list below, character-for-character. "
        "Do not output reasoning, explanation, JSON, markdown, or any additional text.\n"
        f"{labels}\n\n"
        "ACTION:"
    )


def action_group(label: str) -> str:
    """Return the canonical tool/action group from a validated label."""
    if label not in ACTION_LABEL_SET:
        raise ValueError(f"unknown SAIE action label: {label!r}")
    return label[1 : label.index("]")]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
