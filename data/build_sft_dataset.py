#!/usr/bin/env python3
"""Convert the *actual* SAIE benchmark annotations into portable SFT JSONL.

Input layout (one or more warehouses below ``--saie-output``)::

    warehouse_s31500/
      annotations/bench_seed09001.json   # [{q, a, frames, ...}, ...]
      frames/bench_seed09001/spot_cam/rgb_0000.png
      master/bench_seed09001.json

An explicit warehouse-level split manifest is mandatory for paper runs.  The
``--dev-auto-split`` escape hatch exists only for local smoke testing and is
recorded as non-paper data in ``dataset_manifest.json``.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from annotation_audit import causal_history  # noqa: E402
from data.schema import ACTION_LABELS, ACTION_LABEL_SET, action_group, write_jsonl  # noqa: E402


SPLITS = ("train", "val", "test")
FRAME_RE = re.compile(r"^(?P<prefix>.*?)(?P<index>\d+)(?P<suffix>\.[^.]+)$")
VIDEO_RE = re.compile(r"_t(?P<index>\d+)$")

# Nominal (dx_m, dyaw_rad) each canonical label commands, per
# annotation_review.md's parse_action_motion() convention: positive dyaw is
# left/CCW (confirmed against that audit's own examples, e.g. a `[gauge]
# stop` row measuring dyaw_rad=+1.3481 while turning left). All task-stop
# labels command zero motion by definition -- a demonstrator that is
# genuinely reading a gauge/counting/checking state/reporting an alert
# should not be moving.
_NOMINAL_MOTION_M_RAD: dict[str, tuple[float, float]] = {
    "[navigate] move forward 25 cm": (0.25, 0.0),
    "[navigate] move forward 50 cm": (0.50, 0.0),
    "[navigate] move forward 75 cm": (0.75, 0.0),
    "[navigate] turn left 15 degree": (0.0, math.radians(15)),
    "[navigate] turn left 30 degree": (0.0, math.radians(30)),
    "[navigate] turn left 45 degree": (0.0, math.radians(45)),
    "[navigate] turn right 15 degree": (0.0, -math.radians(15)),
    "[navigate] turn right 30 degree": (0.0, -math.radians(30)),
    "[navigate] turn right 45 degree": (0.0, -math.radians(45)),
    "[navigate] stop": (0.0, 0.0),
    "[gauge] stop": (0.0, 0.0),
    "[count] stop": (0.0, 0.0),
    "[state] stop": (0.0, 0.0),
    "[alert] stop": (0.0, 0.0),
}
# SE(2) distance weight on yaw error, matching the judge's own
# NAV_SE2_K=0.5 m/rad convention (annotation_review.md section 4) so that
# "resolve-via-motion" arbitration and the physical-grounding audit use the
# same metric the paper's evaluation already relies on.
_SE2_YAW_WEIGHT_M_PER_RAD = 0.5
# Minimum absolute gap (in the same SE2-error units, ~meters) between the two
# conflicting candidates' errors before a duplicate is considered physically
# resolved rather than ambiguous. Deliberately conservative: an unresolved
# conflict falls back to exclude-conflicts (drop both) rather than guessing.
_MOTION_RESOLUTION_MARGIN_M = 0.15
_MOTION_RESOLUTION_MAX_ERROR_M = 0.15


def _error_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def _percentile(p: float) -> float:
        if n == 1:
            return ordered[0]
        rank = p * (n - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    buckets = {"<=0.05m": 0, "<=0.25m": 0, "<=0.5m": 0, "<=1.0m": 0, ">1.0m": 0}
    for value in ordered:
        if value <= 0.05:
            buckets["<=0.05m"] += 1
        elif value <= 0.25:
            buckets["<=0.25m"] += 1
        elif value <= 0.5:
            buckets["<=0.5m"] += 1
        elif value <= 1.0:
            buckets["<=1.0m"] += 1
        else:
            buckets[">1.0m"] += 1
    return {
        "n": n,
        "mean": sum(ordered) / n,
        "median": _percentile(0.5),
        "p90": _percentile(0.9),
        "max": ordered[-1],
        "histogram_m": buckets,
    }


def _se2_error(measured: tuple[float, float], nominal: tuple[float, float]) -> float:
    dx_err = measured[0] - nominal[0]
    dyaw_err = measured[1] - nominal[1]
    return math.sqrt(dx_err**2 + (_SE2_YAW_WEIGHT_M_PER_RAD * dyaw_err) ** 2)


# motion_summary events are logged on a fixed ~30-render-frame cadence
# (confirmed empirically against real sim_events.json files), while decision
# anchors (video_id / source_frame_id) land off that grid -- an exact-frame
# dict lookup therefore misses the large majority of rows. Search the
# nearest logged motion_summary within this many render frames instead
# (half the observed cadence, so every anchor has a candidate within range
# except right at an episode's very start/end).
_MOTION_LOOKUP_TOLERANCE_FRAMES = 15


class _EpisodeMotionIndex:
    """Nearest-frame lookup into an episode's motion_summary events.

    Mirrors the exporter's own `_build_motion_summary_index` / the judge's
    `_build_motion_index` pattern (both keyed by render_frame off
    `spot_events` entries with `event == "motion_summary"`), reimplemented
    here so this builder never needs to import the oem6-owned exporter or
    judge modules.
    """

    def __init__(self, warehouse_dir: Path, episode_id: str) -> None:
        path = warehouse_dir / "frames" / episode_id / "sim_events.json"
        entries: list[tuple[int, float, float]] = []
        if path.is_file():
            events = _read_json(path)
            for entry in events.get("spot_events") or []:
                if not isinstance(entry, dict) or entry.get("event") != "motion_summary":
                    continue
                render_frame = entry.get("render_frame")
                dx = entry.get("dx")
                dyaw = entry.get("dyaw")
                if isinstance(render_frame, int) and _finite_number(dx) and _finite_number(dyaw):
                    entries.append((render_frame, float(dx), float(dyaw)))
        entries.sort(key=lambda item: item[0])
        self._frames = [item[0] for item in entries]
        self._values = [(item[1], item[2]) for item in entries]

    def nearest(self, render_frame: int) -> tuple[float, float] | None:
        if not self._frames:
            return None
        idx = bisect.bisect_left(self._frames, render_frame)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(self._frames)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self._frames[i] - render_frame))
        if abs(self._frames[best] - render_frame) > _MOTION_LOOKUP_TOLERANCE_FRAMES:
            return None
        return self._values[best]


def _normalise_hazard_type(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    aliases = {
        "FALL_HUMAN": "fallen_worker",
        "fall_human": "fallen_worker",
        "human": "fallen_worker",
    }
    return aliases.get(raw, raw.lower())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _load_split_manifest(path: Path) -> dict[str, str]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("split manifest must be a JSON object")

    # Both forms are accepted:
    #   {"train": ["warehouse_s1"], "val": [...], "test": [...]}
    #   {"warehouse_s1": "train", ...}
    if any(key in raw for key in SPLITS):
        unknown = set(raw) - set(SPLITS) - {"version", "note"}
        if unknown:
            raise ValueError(f"unknown split-manifest keys: {sorted(unknown)}")
        out: dict[str, str] = {}
        for split in SPLITS:
            values = raw.get(split, [])
            if not isinstance(values, list) or not all(isinstance(x, str) for x in values):
                raise ValueError(f"split manifest field {split!r} must be a list of warehouse ids")
            for warehouse_id in values:
                if warehouse_id in out:
                    raise ValueError(f"warehouse {warehouse_id!r} appears in two splits")
                out[warehouse_id] = split
        return out

    out = {}
    for warehouse_id, split in raw.items():
        if split not in SPLITS:
            raise ValueError(f"warehouse {warehouse_id!r} has invalid split {split!r}")
        out[str(warehouse_id)] = str(split)
    return out


def _make_dev_split(warehouse_ids: list[str], seed: int) -> dict[str, str]:
    if len(warehouse_ids) < 3:
        raise ValueError("--dev-auto-split needs at least 3 warehouses (train/val/test)")
    ids = sorted(warehouse_ids)
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(len(ids) * 0.2))
    n_val = max(1, round(len(ids) * 0.1))
    if n_test + n_val >= len(ids):
        n_test, n_val = 1, 1
    out = {warehouse_id: "train" for warehouse_id in ids}
    for warehouse_id in ids[:n_test]:
        out[warehouse_id] = "test"
    for warehouse_id in ids[n_test : n_test + n_val]:
        out[warehouse_id] = "val"
    return out


def _warehouse_dirs(root: Path) -> list[Path]:
    dirs = sorted({path.parent for path in root.rglob("annotations") if path.is_dir()})
    # rglob returns the annotation directory itself; its parent is the warehouse.
    dirs = [path for path in dirs if (path / "annotations").is_dir()]
    if not dirs and (root / "annotations").is_dir():
        dirs = [root]
    return dirs


def _episode_id(annotation_path: Path, item: dict[str, Any]) -> str:
    if annotation_path.stem != "annotations":
        return annotation_path.stem
    frames = item.get("frames") or []
    if frames and isinstance(frames[0], str):
        return Path(frames[0]).parts[0]
    video_id = str(item.get("video_id", ""))
    match = re.match(r"(.+)_t\d+$", video_id)
    if match:
        return match.group(1)
    raise ValueError(f"cannot infer episode id for a row in {annotation_path}")


def _master_metadata(warehouse_dir: Path, episode_id: str) -> dict[str, Any]:
    path = warehouse_dir / "master" / f"{episode_id}.json"
    if not path.is_file():
        return {"master_path": None}
    master = _read_json(path)
    hazard = master.get("hazard") or {}
    risk_source = master.get("risk_source") or {}
    hazard_type = (
        master.get("hazard_type")
        or hazard.get("type")
        or risk_source.get("type")
        or master.get("scenario_type")
    )
    distractor = master.get("is_distractor")
    if distractor is None and isinstance(hazard, dict):
        distractor = hazard.get("is_distractor")
    return {
        "master_path": path,
        "hazard_type": _normalise_hazard_type(hazard_type),
        "hazard_present": (
            bool(master.get("hazard_present"))
            if master.get("hazard_present") is not None
            else (True if hazard_type is not None else None)
        ),
        # Preserve unknown as null.  Treating missing metadata as false would
        # silently contaminate the no-distractor ablation.
        "is_distractor": bool(distractor) if distractor is not None else None,
        "severity": master.get("severity"),
        "scenario_type": master.get("scenario_type"),
    }


def _invalid_episode(warehouse_dir: Path, episode_id: str) -> tuple[bool, str | None, Path | None]:
    path = warehouse_dir / "frames" / episode_id / "sim_events.json"
    if not path.is_file():
        return False, None, None
    events = _read_json(path)
    return bool(events.get("invalid", False)), events.get("invalid_reason"), path


class FrameResolver:
    def __init__(self, image_root: Path, max_gap: int, strict: bool):
        self.image_root = image_root.resolve()
        self.max_gap = max_gap
        self.strict = strict
        self._cache: dict[tuple[Path, str, str], tuple[list[int], dict[int, Path]]] = {}
        self.repaired = 0
        self.exact = 0
        self.gaps: Counter[int] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.repair_records: list[dict[str, Any]] = []

    def _candidates(self, missing: Path, match: re.Match[str]) -> tuple[list[int], dict[int, Path]]:
        key = (missing.parent, match.group("prefix"), match.group("suffix"))
        if key not in self._cache:
            by_index: dict[int, Path] = {}
            pattern = f"{match.group('prefix')}*{match.group('suffix')}"
            for candidate in missing.parent.glob(pattern):
                candidate_match = FRAME_RE.match(candidate.name)
                if candidate_match and candidate_match.group("prefix") == match.group("prefix"):
                    by_index[int(candidate_match.group("index"))] = candidate
            self._cache[key] = (sorted(by_index), by_index)
        return self._cache[key]

    def resolve(self, warehouse_dir: Path, frame_ref: str) -> str:
        source = Path(frame_ref)
        intended = source if source.is_absolute() else warehouse_dir / "frames" / source
        intended = intended.resolve()
        if intended.is_file():
            self.exact += 1
            return self._relative(intended)

        match = FRAME_RE.match(intended.name)
        if not match:
            raise FileNotFoundError(f"missing frame and no numeric index to repair: {intended}")
        wanted = int(match.group("index"))
        indices, by_index = self._candidates(intended, match)
        if not indices:
            raise FileNotFoundError(f"no saved frames alongside missing reference: {intended}")
        pos = bisect.bisect_left(indices, wanted)
        neighbors = indices[max(0, pos - 1) : min(len(indices), pos + 1)]
        nearest = min(neighbors, key=lambda value: (abs(value - wanted), value))
        gap = abs(nearest - wanted)
        if gap > self.max_gap:
            raise FileNotFoundError(
                f"nearest saved frame is too far from {intended}: index {nearest}, gap={gap} "
                f"> --max-frame-gap={self.max_gap}"
            )
        if self.strict:
            raise FileNotFoundError(
                f"strict frame mode forbids repair: {intended} -> {by_index[nearest]} (gap={gap})"
            )
        self.repaired += 1
        self.gaps[gap] += 1
        repair_record = {
            # Keep the repair ledger portable across the workstation and
            # /workspace on RunPod.  Both paths use the same image-root
            # relative contract as the emitted SFT rows.
            "missing": self._relative(intended),
            "resolved": self._relative(by_index[nearest].resolve()),
            "index_gap": gap,
            "wanted_frame_id": wanted,
            "selected_frame_id": nearest,
        }
        self.repair_records.append(repair_record)
        if len(self.examples) < 20:
            self.examples.append(repair_record)
        return self._relative(by_index[nearest].resolve())

    def resolve_decision(
        self, warehouse_dir: Path, episode_id: str, source_anchor: int, num_frames: int
    ) -> list[str]:
        """Snap the decision anchor, then rebuild causal history from saved PNGs."""
        anchor_ref = f"{episode_id}/spot_cam/rgb_{source_anchor:04d}.png"
        current_relative = self.resolve(warehouse_dir, anchor_ref)
        current = (self.image_root / current_relative).resolve()
        match = FRAME_RE.match(current.name)
        if match is None:
            raise ValueError(f"cannot parse selected frame index: {current}")
        selected_index = int(match.group("index"))
        indices, by_index = self._candidates(current, match)
        selected_history = causal_history(selected_index, indices, num_frames)
        if not selected_history or selected_history[-1] != selected_index:
            raise AssertionError(
                f"failed to construct causal history ending at frame {selected_index}: {selected_history}"
            )
        return [self._relative(by_index[index].resolve()) for index in selected_history]

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.image_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"frame {path} is outside portable --image-root {self.image_root}") from exc


def _hash_files(paths: list[Path], source_root: Path) -> str:
    """Hash stable source-root-relative names and bytes (never host paths)."""
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        try:
            relative = path.resolve().relative_to(source_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"source file lies outside --saie-output: {path}") from exc
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_ablation(
    rows: list[dict[str, Any]], split: str, ablation: str, held_out_hazards: set[str]
) -> list[dict[str, Any]]:
    # Validation follows the training distribution; using held-out hazards to
    # select the best checkpoint would leak the intended generalization axis.
    fit_partition = split in {"train", "val"}
    if ablation == "holdout_hazard" and fit_partition:
        return [row for row in rows if row.get("hazard_type") not in held_out_hazards]
    if ablation == "no_distractor" and fit_partition:
        return [row for row in rows if row.get("is_distractor") is False]
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["action_label"] for row in rows)
    groups = Counter(row["action_group"] for row in rows)
    hazards = Counter(str(row.get("hazard_type")) for row in rows)
    distractors = Counter(str(row.get("is_distractor")) for row in rows)
    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "warehouses": sorted({row["warehouse_id"] for row in rows}),
        "label_counts": {label: labels.get(label, 0) for label in ACTION_LABELS},
        "group_counts": dict(sorted(groups.items())),
        "hazard_counts": dict(sorted(hazards.items())),
        "distractor_metadata_counts": dict(sorted(distractors.items())),
    }


STOP_TIMING_VALUES = {
    "premature",
    "valid",
    "late",
    "false_positive",
    "not_applicable",
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_finite_number(item) for item in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _valid_world_position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) in {2, 3}
        and all(_finite_number(item) for item in value)
    )


def _valid_attention_target(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    bbox = value.get("bbox_xyxy", value.get("bbox"))
    world = value.get("world_position_m", value.get("world_position"))
    bearing = value.get("bearing_deg", value.get("target_bearing_deg"))
    return _valid_bbox(bbox) or (
        _valid_world_position(world) and _finite_number(bearing)
    )


def _candidate_safety_flags(value: Any) -> list[bool] | None:
    """Normalize a canonical-label map/list to true-means-safe flags."""

    if isinstance(value, dict):
        if set(value) != ACTION_LABEL_SET:
            return None
        candidates = [value[label] for label in ACTION_LABELS]
    elif isinstance(value, list) and len(value) == len(ACTION_LABELS):
        candidates = value
    else:
        return None
    flags: list[bool] = []
    for candidate in candidates:
        if isinstance(candidate, bool):
            flags.append(candidate)
            continue
        if not isinstance(candidate, dict):
            return None
        declared = [
            (key, candidate.get(key))
            for key in ("safe", "is_safe", "violates")
            if key in candidate
        ]
        if len(declared) != 1 or not isinstance(declared[0][1], bool):
            return None
        key, flag = declared[0]
        flags.append(not flag if key == "violates" else flag)
    return flags


def scientific_structure_errors(rows: list[dict[str, Any]]) -> Counter[str]:
    """Return row-level structural errors for the optional ICRA annotation layer."""

    errors: Counter[str] = Counter()
    for row in rows:
        for field in ("hazard_present", "is_distractor", "risk_adjacent", "in_task_window"):
            value = row.get(field)
            if value is not None and not isinstance(value, bool):
                errors[f"{field}_not_boolean"] += 1

        stop_timing = row.get("stop_timing")
        if stop_timing is not None and stop_timing not in STOP_TIMING_VALUES:
            errors["invalid_stop_timing"] += 1

        admissible = row.get("admissible_action_labels")
        if admissible is not None and (
            not isinstance(admissible, list)
            or not admissible
            or any(not isinstance(label, str) for label in admissible)
            or len(set(admissible)) != len(admissible)
            or any(label not in ACTION_LABEL_SET for label in admissible)
        ):
            errors["invalid_admissible_action_labels"] += 1

        safe_mask = row.get("safe_action_mask")
        candidate_safety = row.get("candidate_action_safety")
        valid_mask = (
            isinstance(safe_mask, list)
            and len(safe_mask) == len(ACTION_LABELS)
            and all(isinstance(value, bool) for value in safe_mask)
        )
        if safe_mask is not None:
            if not valid_mask:
                errors["invalid_safe_action_mask"] += 1
        candidate_flags = (
            _candidate_safety_flags(candidate_safety)
            if candidate_safety is not None
            else None
        )
        if candidate_safety is not None and candidate_flags is None:
            errors["invalid_candidate_action_safety"] += 1
        if valid_mask and candidate_flags is not None and list(safe_mask) != candidate_flags:
            errors["inconsistent_safe_action_supervision"] += 1

        if row.get("action_group") != "alert":
            continue
        if row.get("view_valid") is not None and not isinstance(row.get("view_valid"), bool):
            errors["view_valid_not_boolean"] += 1
        attention_target = row.get("attention_target")
        bbox = row.get("target_bbox")
        world = row.get("target_world_position")
        bearing = row.get("target_bearing_deg")
        localization_valid = False
        if attention_target is not None:
            localization_valid = _valid_attention_target(attention_target)
            if not localization_valid:
                errors["invalid_attention_target"] += 1
        if bbox is not None:
            bbox_valid = _valid_bbox(bbox)
            localization_valid = localization_valid or bbox_valid
            if not bbox_valid:
                errors["invalid_target_bbox"] += 1
        if world is not None or bearing is not None:
            world_bearing_valid = _valid_world_position(world) and _finite_number(bearing)
            localization_valid = localization_valid or world_bearing_valid
            if not world_bearing_valid:
                errors["invalid_target_world_bearing"] += 1
        if (
            any(value is not None for value in (attention_target, bbox, world, bearing))
            and not localization_valid
        ):
            errors["invalid_alert_localization"] += 1
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saie-output", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="portable root used by JSONL image paths (default: --saie-output)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--dev-auto-split",
        action="store_true",
        help="deterministic 70/10/20 warehouse split for smoke tests only",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ablation", choices=("default", "holdout_hazard", "no_distractor"), default="default"
    )
    parser.add_argument(
        "--held-out-hazards",
        default="",
        help="comma-separated exact hazard_type values; required for holdout_hazard",
    )
    parser.add_argument(
        "--max-history-frames",
        type=int,
        default=1,
        help="rebuild N causal frames from actual saved PNGs (default 1=current frame)",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=8,
        help="maximum render-frame index gap for dev repair (8 is <=0.27 s at 30 Hz)",
    )
    parser.add_argument(
        "--strict-frame-paths",
        action="store_true",
        help="fail instead of nearest-saved-frame repair (required for final paper export)",
    )
    parser.add_argument("--allow-unassigned", action="store_true")
    parser.add_argument("--allow-empty-split", action="store_true")
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "exclude-conflicts", "first", "prefer-task-stop", "resolve-via-motion"),
        default="error",
        help=(
            "paper runs must use error; repair policies are development-only and recorded. "
            "resolve-via-motion arbitrates conflicting same-video_id rows using the shared "
            "decision frame's actual sim_events.json motion_summary against each candidate "
            "label's nominal motion; falls back to exclude-conflicts when ambiguous"
        ),
    )
    parser.add_argument(
        "--balance-cap",
        type=int,
        default=None,
        help=(
            "cap each action_label's row count in the train split only (random subsample, "
            "seeded by --seed); val/test always keep the natural distribution so they stay "
            "comparable to the paper's protocol"
        ),
    )
    parser.add_argument(
        "--max-motion-error-m",
        type=float,
        default=None,
        help=(
            "exclude train-split navigate rows whose nominal-vs-measured SE(2) motion error "
            "exceeds this (meters-equivalent, same metric as resolve-via-motion). Unset by "
            "default: the full error distribution is always logged into dataset_manifest.json "
            "regardless of this flag, so the threshold can be chosen from real numbers instead "
            "of guessed up front"
        ),
    )
    parser.add_argument(
        "--require-all-train-labels", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--expected-warehouses", type=int, default=70, help="0 disables this gate")
    parser.add_argument("--expected-test-warehouses", type=int, default=11, help="0 disables this gate")
    parser.add_argument(
        "--expected-train-pool-warehouses", type=int, default=59, help="train+val; 0 disables"
    )
    parser.add_argument("--expected-episodes", type=int, default=1311, help="0 disables this gate")
    parser.add_argument("--expected-rows", type=int, default=80280, help="0 disables this gate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.saie_output.resolve()
    image_root = (args.image_root or args.saie_output).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"--saie-output does not exist: {source_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root does not exist: {image_root}")
    if bool(args.split_manifest) == bool(args.dev_auto_split):
        raise ValueError("choose exactly one of --split-manifest or --dev-auto-split")
    if args.max_history_frames < 1:
        raise ValueError("--max-history-frames must be >= 1")
    if args.max_frame_gap < 0:
        raise ValueError("--max-frame-gap must be >= 0")
    expectation_values = {
        "warehouses": args.expected_warehouses,
        "test_warehouses": args.expected_test_warehouses,
        "train_pool_warehouses": args.expected_train_pool_warehouses,
        "episodes": args.expected_episodes,
        "rows": args.expected_rows,
    }
    if any(value < 0 for value in expectation_values.values()):
        raise ValueError("expected corpus counts must be >= 0 (use 0 to disable a gate)")

    held_out_hazards = {value.strip() for value in args.held_out_hazards.split(",") if value.strip()}
    if args.ablation == "holdout_hazard" and not held_out_hazards:
        raise ValueError("holdout_hazard requires --held-out-hazards")

    warehouses = _warehouse_dirs(source_root)
    if not warehouses:
        raise FileNotFoundError(f"no warehouse */annotations directories below {source_root}")
    warehouse_ids = [path.name for path in warehouses]
    split_map = (
        _load_split_manifest(args.split_manifest.resolve())
        if args.split_manifest
        else _make_dev_split(warehouse_ids, args.seed)
    )
    missing_assignments = sorted(set(warehouse_ids) - set(split_map))
    extra_assignments = sorted(set(split_map) - set(warehouse_ids))
    if missing_assignments and not args.allow_unassigned:
        raise ValueError(f"warehouses missing from split manifest: {missing_assignments}")
    if extra_assignments:
        raise ValueError(f"split manifest names warehouses absent from input: {extra_assignments}")

    resolver = FrameResolver(image_root, args.max_frame_gap, args.strict_frame_paths)
    buckets: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_files: list[Path] = []
    seen_row_ids: dict[str, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    duplicate_rows = 0
    duplicate_examples: list[str] = []
    conflicting_duplicates = 0
    conflict_examples: list[dict[str, str]] = []
    motion_resolved_examples: list[dict[str, Any]] = []
    motion_ambiguous_examples: list[dict[str, Any]] = []
    motion_resolved_count = 0
    motion_ambiguous_count = 0
    invalid_episodes: dict[str, str | None] = {}
    valid_episodes: set[str] = set()
    missing_master_episodes: set[str] = set()
    missing_sim_events_episodes: set[str] = set()
    skipped_unassigned: list[str] = []
    source_noncausal_history_references = 0
    source_decision_order_reversals = 0
    decision_order_examples: list[dict[str, Any]] = []
    last_anchor_by_episode: dict[str, int] = {}
    source_task_groups: dict[str, set[str]] = {}
    # Physical-grounding audit: SE(2) error between every navigate row's
    # nominal primitive and its actually-measured motion at that decision's
    # render_frame -- collected for ALL navigate rows (not just conflicts),
    # extending annotation_review.md section 4's 52-row-sample audit to the
    # full corpus. Always collected regardless of --max-motion-error-m.
    navigate_motion_errors: list[float] = []

    for warehouse_dir in warehouses:
        warehouse_id = warehouse_dir.name
        split = split_map.get(warehouse_id)
        if split is None:
            skipped_unassigned.append(warehouse_id)
            continue
        for annotation_path in sorted((warehouse_dir / "annotations").glob("*.json")):
            annotations = _read_json(annotation_path)
            if not isinstance(annotations, list):
                raise ValueError(f"{annotation_path}: expected a JSON list")
            source_files.append(annotation_path)
            metadata_cache: dict[str, dict[str, Any]] = {}
            invalid_cache: dict[str, bool] = {}
            motion_index_cache: dict[str, _EpisodeMotionIndex] = {}
            for item_index, item in enumerate(annotations):
                if not isinstance(item, dict):
                    raise ValueError(f"{annotation_path}[{item_index}]: expected an object")
                label = item.get("a")
                if label not in ACTION_LABEL_SET:
                    raise ValueError(
                        f"{annotation_path}[{item_index}]: non-canonical action label {label!r}"
                    )
                instruction = item.get("q")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(f"{annotation_path}[{item_index}]: missing q instruction")
                episode_id = _episode_id(annotation_path, item)
                qualified_episode = f"{warehouse_id}/{episode_id}"
                video_id = str(item.get("video_id") or "")
                anchor_match = VIDEO_RE.search(video_id)
                if anchor_match is None:
                    raise ValueError(f"{annotation_path}[{item_index}]: invalid video_id anchor {video_id!r}")
                source_anchor = int(anchor_match.group("index"))
                previous_anchor = last_anchor_by_episode.get(qualified_episode)
                if previous_anchor is not None and source_anchor < previous_anchor:
                    source_decision_order_reversals += 1
                    if len(decision_order_examples) < 20:
                        decision_order_examples.append(
                            {
                                "episode_id": qualified_episode,
                                "previous_source_frame_id": previous_anchor,
                                "source_frame_id": source_anchor,
                                "source_annotation_index": item_index,
                            }
                        )
                last_anchor_by_episode[qualified_episode] = source_anchor
                if episode_id not in invalid_cache:
                    is_invalid, invalid_reason, events_path = _invalid_episode(warehouse_dir, episode_id)
                    invalid_cache[episode_id] = is_invalid
                    if events_path:
                        source_files.append(events_path)
                    else:
                        missing_sim_events_episodes.add(f"{warehouse_id}/{episode_id}")
                    if is_invalid:
                        invalid_episodes[f"{warehouse_id}/{episode_id}"] = invalid_reason
                if invalid_cache[episode_id]:
                    continue
                valid_episodes.add(qualified_episode)
                source_task_groups.setdefault(qualified_episode, set()).add(action_group(label))
                frame_refs = item.get("frames")
                if not isinstance(frame_refs, list) or not frame_refs or not all(
                    isinstance(value, str) for value in frame_refs
                ):
                    raise ValueError(f"{annotation_path}[{item_index}]: frames must be a non-empty string list")
                for frame_ref in frame_refs:
                    ref_match = FRAME_RE.match(Path(frame_ref).name)
                    if ref_match and int(ref_match.group("index")) > source_anchor:
                        source_noncausal_history_references += 1
                resolved_frames = resolver.resolve_decision(
                    warehouse_dir, episode_id, source_anchor, args.max_history_frames
                )

                if episode_id not in metadata_cache:
                    metadata_cache[episode_id] = _master_metadata(warehouse_dir, episode_id)
                    master_path = metadata_cache[episode_id].get("master_path")
                    if master_path:
                        source_files.append(master_path)
                    else:
                        missing_master_episodes.add(qualified_episode)
                meta = metadata_cache[episode_id]
                row_id = f"{warehouse_id}/{video_id}"
                signature = (label, resolved_frames[-1], instruction.strip())
                group = action_group(label)
                item_hazard = _normalise_hazard_type(item.get("hazard_type"))
                hazard_type = item_hazard if item_hazard is not None else meta.get("hazard_type")
                explicit_distractor = item.get("is_distractor")
                if explicit_distractor is None:
                    explicit_distractor = meta.get("is_distractor")
                hazard_present = item.get("hazard_present")
                if hazard_present is None:
                    hazard_present = meta.get("hazard_present")
                selected_match = FRAME_RE.match(Path(resolved_frames[-1]).name)
                if selected_match is None:
                    raise ValueError(f"cannot parse numeric frame id: {resolved_frames[-1]}")
                row = {
                    "schema_version": "saie_sft_v2",
                    # Native episode/video ids repeat across warehouses.  The
                    # qualified ids are the only safe keys for split audits.
                    "sample_id": row_id,
                    "frame_id": int(selected_match.group("index")),
                    "source_frame_id": source_anchor,
                    "episode_id": f"{warehouse_id}/{episode_id}",
                    "source_video_id": video_id,
                    "source_episode_id": episode_id,
                    "warehouse_id": warehouse_id,
                    "images": resolved_frames,
                    "image": resolved_frames[-1],
                    "instruction": instruction.strip(),
                    "action_label": label,
                    "action_group": group,
                    "action_category": group,
                    "hazard_type": hazard_type,
                    "hazard_present": hazard_present,
                    "is_distractor": explicit_distractor,
                    # Preserve explicit frame-aligned measurements when a
                    # future exporter provides them; otherwise retain null.
                    # None of these metadata fields enter the model prompt.
                    "dist_to_hazard_m": item.get("dist_to_hazard_m"),
                    "risk_adjacent": item.get("risk_adjacent"),
                    # A task-stop label does not prove the geometric validity
                    # window.  Only preserve an exporter-provided value.
                    "in_task_window": item.get("in_task_window"),
                    # Optional localization/gaze supervision for the extended
                    # ICRA corpus.  Keeping the raw fields makes absence
                    # auditable without inventing negative annotations.
                    "attention_target": item.get("attention_target"),
                    "target_bbox": item.get("target_bbox"),
                    "target_world_position": item.get("target_world_position"),
                    "target_bearing_deg": item.get("target_bearing_deg"),
                    "view_valid": item.get("view_valid"),
                    "stop_timing": item.get("stop_timing"),
                    "admissible_action_labels": item.get("admissible_action_labels"),
                    "safe_action_mask": item.get("safe_action_mask"),
                    "candidate_action_safety": item.get("candidate_action_safety"),
                    "motion": item.get("motion"),
                    "source_annotation": annotation_path.relative_to(source_root).as_posix(),
                    "split_tags": [f"hazard:{hazard_type}"] if hazard_type else [],
                }
                if episode_id not in motion_index_cache:
                    motion_index_cache[episode_id] = _EpisodeMotionIndex(warehouse_dir, episode_id)
                measured_motion = motion_index_cache[episode_id].nearest(source_anchor)
                motion_error_m: float | None = None
                if measured_motion is not None and label in _NOMINAL_MOTION_M_RAD:
                    motion_error_m = _se2_error(measured_motion, _NOMINAL_MOTION_M_RAD[label])
                    if group == "navigate":
                        navigate_motion_errors.append(motion_error_m)
                row["_motion_error_m"] = motion_error_m
                if row_id in seen_row_ids:
                    old_signature, old_row = seen_row_ids[row_id]
                    if args.duplicate_policy == "error":
                        kind = "exact" if old_signature == signature else "conflicting"
                        raise ValueError(f"{kind} duplicate decision-frame id: {row_id}")
                    duplicate_rows += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append(row_id)
                    if old_signature != signature:
                        conflicting_duplicates += 1
                        if len(conflict_examples) < 20:
                            conflict_examples.append(
                                {"frame_id": row_id, "kept_or_first": old_signature[0], "new": label}
                            )
                        if args.duplicate_policy == "exclude-conflicts":
                            old_row["_exclude_conflict"] = True
                        elif args.duplicate_policy == "prefer-task-stop":
                            old_group = action_group(old_signature[0])
                            old_is_task = old_group != "navigate"
                            new_is_task = group != "navigate"
                            if old_is_task == new_is_task:
                                raise ValueError(
                                    f"ambiguous conflicting duplicate (same priority): {row_id}: "
                                    f"{old_signature[0]!r} vs {label!r}"
                                )
                            if new_is_task:
                                old_row.clear()
                                old_row.update(row)
                                seen_row_ids[row_id] = (signature, old_row)
                        elif args.duplicate_policy == "resolve-via-motion":
                            old_label = old_signature[0]
                            resolvable = (
                                measured_motion is not None
                                and old_label in _NOMINAL_MOTION_M_RAD
                                and label in _NOMINAL_MOTION_M_RAD
                            )
                            old_error = new_error = None
                            if resolvable:
                                old_error = _se2_error(measured_motion, _NOMINAL_MOTION_M_RAD[old_label])
                                new_error = _se2_error(measured_motion, _NOMINAL_MOTION_M_RAD[label])
                            if (
                                resolvable
                                and min(old_error, new_error) <= _MOTION_RESOLUTION_MAX_ERROR_M
                                and abs(old_error - new_error) >= _MOTION_RESOLUTION_MARGIN_M
                            ):
                                motion_resolved_count += 1
                                winner_is_new = new_error < old_error
                                if len(motion_resolved_examples) < 20:
                                    motion_resolved_examples.append(
                                        {
                                            "frame_id": row_id,
                                            "measured_dx_m": measured_motion[0],
                                            "measured_dyaw_rad": measured_motion[1],
                                            "kept": label if winner_is_new else old_label,
                                            "kept_error_m": round(new_error if winner_is_new else old_error, 4),
                                            "dropped": old_label if winner_is_new else label,
                                            "dropped_error_m": round(old_error if winner_is_new else new_error, 4),
                                        }
                                    )
                                if winner_is_new:
                                    old_row.clear()
                                    old_row.update(row)
                                    seen_row_ids[row_id] = (signature, old_row)
                                # else: old_row already holds the winning candidate; the new
                                # `row` built above is simply discarded by the `continue` below.
                            else:
                                motion_ambiguous_count += 1
                                if len(motion_ambiguous_examples) < 20:
                                    motion_ambiguous_examples.append(
                                        {
                                            "frame_id": row_id,
                                            "reason": (
                                                "no motion_summary at shared render_frame"
                                                if not resolvable
                                                else "errors too close or both too large to trust"
                                            ),
                                            "old_label": old_label,
                                            "new_label": label,
                                            "old_error_m": round(old_error, 4) if old_error is not None else None,
                                            "new_error_m": round(new_error, 4) if new_error is not None else None,
                                        }
                                    )
                                old_row["_exclude_conflict"] = True
                    continue
                seen_row_ids[row_id] = (signature, row)
                buckets[split].append(row)

    source_valid_buckets: dict[str, list[dict[str, Any]]] = {}
    motion_error_excluded_train = 0
    for split in SPLITS:
        cleaned = []
        for row in buckets[split]:
            excluded_conflict = row.pop("_exclude_conflict", False)
            motion_error_m = row.pop("_motion_error_m", None)
            if excluded_conflict:
                continue
            # --max-motion-error-m only ever trims train: val/test must keep
            # the natural (possibly ungrounded) distribution so scores stay
            # comparable to the paper's original evaluation protocol.
            if (
                split == "train"
                and args.max_motion_error_m is not None
                and motion_error_m is not None
                and motion_error_m > args.max_motion_error_m
            ):
                motion_error_excluded_train += 1
                continue
            cleaned.append(row)
        source_valid_buckets[split] = list(cleaned)
        buckets[split] = _apply_ablation(cleaned, split, args.ablation, held_out_hazards)
        if not buckets[split] and not args.allow_empty_split:
            raise ValueError(f"{split} split is empty after applying {args.ablation!r}")

    balance_cap_before: dict[str, int] = {}
    balance_cap_after: dict[str, int] = {}
    if args.balance_cap is not None:
        rng = random.Random(args.seed)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for row in buckets["train"]:
            by_label.setdefault(row["action_label"], []).append(row)
        capped_train: list[dict[str, Any]] = []
        for label_key, rows in by_label.items():
            balance_cap_before[label_key] = len(rows)
            kept = rng.sample(rows, args.balance_cap) if len(rows) > args.balance_cap else rows
            balance_cap_after[label_key] = len(kept)
            capped_train.extend(kept)
        rng.shuffle(capped_train)
        buckets["train"] = capped_train
        if not buckets["train"] and not args.allow_empty_split:
            raise ValueError("train split is empty after applying --balance-cap")

    split_warehouses = {
        split: {row["warehouse_id"] for row in rows} for split, rows in buckets.items()
    }
    assigned_warehouse_sets = {
        split: {
            warehouse_id
            for warehouse_id in warehouse_ids
            if split_map.get(warehouse_id) == split
        }
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_warehouses[left] & split_warehouses[right]
            if overlap:
                raise AssertionError(f"warehouse leakage between {left}/{right}: {sorted(overlap)}")
    split_episodes = {split: {row["episode_id"] for row in rows} for split, rows in buckets.items()}
    if split_episodes["train"] & split_episodes["test"]:
        raise AssertionError("episode id leakage between train/test")

    missing_train_labels = sorted(ACTION_LABEL_SET - {row["action_label"] for row in buckets["train"]})
    missing_val_labels = sorted(ACTION_LABEL_SET - {row["action_label"] for row in buckets["val"]})
    if missing_train_labels and args.require_all_train_labels:
        raise ValueError(f"train split is missing canonical labels: {missing_train_labels}")

    outdir = args.out.resolve() / args.ablation
    outdir.mkdir(parents=True, exist_ok=True)
    for split, rows in buckets.items():
        write_jsonl(outdir / f"{split}.jsonl", rows)
    write_jsonl(outdir / "repair_log.jsonl", resolver.repair_records)

    benchmark_ready_reasons = [
        reason
        for reason, present in (
            ("development auto-split used", args.dev_auto_split),
            ("nearest-frame repair allowed", not args.strict_frame_paths),
            ("duplicate annotation rows repaired", duplicate_rows > 0),
            ("non-strict duplicate policy selected", args.duplicate_policy != "error"),
            ("invalid episodes excluded", bool(invalid_episodes)),
        )
        if present
    ]
    if source_noncausal_history_references:
        benchmark_ready_reasons.append(
            "source annotation contains future frame references: "
            f"{source_noncausal_history_references}"
        )
    if missing_train_labels:
        benchmark_ready_reasons.append(f"train split is missing labels: {missing_train_labels}")
    if missing_val_labels:
        benchmark_ready_reasons.append(f"validation split is missing labels: {missing_val_labels}")
    all_assigned_warehouses = set().union(*assigned_warehouse_sets.values())
    if args.expected_warehouses and len(all_assigned_warehouses) != args.expected_warehouses:
        benchmark_ready_reasons.append(
            f"expected {args.expected_warehouses} warehouses, found {len(all_assigned_warehouses)}"
        )
    if (
        args.expected_test_warehouses
        and len(assigned_warehouse_sets["test"]) != args.expected_test_warehouses
    ):
        benchmark_ready_reasons.append(
            f"expected {args.expected_test_warehouses} held-out test warehouses, "
            f"found {len(assigned_warehouse_sets['test'])}"
        )
    train_pool_warehouses = assigned_warehouse_sets["train"] | assigned_warehouse_sets["val"]
    if (
        args.expected_train_pool_warehouses
        and len(train_pool_warehouses) != args.expected_train_pool_warehouses
    ):
        benchmark_ready_reasons.append(
            f"expected {args.expected_train_pool_warehouses} train-pool warehouses "
            f"before/including validation carve-out, found {len(train_pool_warehouses)}"
        )
    all_rows = [row for rows in buckets.values() for row in rows]
    source_valid_rows = [row for rows in source_valid_buckets.values() for row in rows]
    required_task_groups = {"gauge", "count", "state", "alert"}
    missing_task_stops = {
        episode_id: sorted(required_task_groups - source_task_groups.get(episode_id, set()))
        for episode_id in sorted(valid_episodes)
        if required_task_groups - source_task_groups.get(episode_id, set())
    }
    if args.expected_episodes and len(valid_episodes) != args.expected_episodes:
        benchmark_ready_reasons.append(
            f"expected {args.expected_episodes} valid episodes, found {len(valid_episodes)}"
        )
    if args.expected_rows and len(source_valid_rows) != args.expected_rows:
        benchmark_ready_reasons.append(
            f"expected {args.expected_rows} source-valid decision rows, "
            f"found {len(source_valid_rows)}"
        )
    if source_decision_order_reversals:
        benchmark_ready_reasons.append(
            f"source episode decision order reverses {source_decision_order_reversals} times"
        )
    if missing_task_stops:
        benchmark_ready_reasons.append(
            f"valid episodes missing one or more task-stop groups: {len(missing_task_stops)}"
        )
    if missing_master_episodes:
        benchmark_ready_reasons.append(
            f"valid episodes missing master metadata: {len(missing_master_episodes)}"
        )
    if missing_sim_events_episodes:
        benchmark_ready_reasons.append(
            f"episodes missing sim_events.json: {len(missing_sim_events_episodes)}"
        )
    hazard_types = {row.get("hazard_type") for row in all_rows if row.get("hazard_type")}
    icra_scientific_ready_reasons = list(benchmark_ready_reasons)
    if len(hazard_types) < 2:
        icra_scientific_ready_reasons.append(
            f"hazard-type generalization unavailable; observed={sorted(hazard_types)}"
        )
    distractor_annotated = sum(row.get("is_distractor") is not None for row in all_rows)
    risk_annotated = sum(row.get("risk_adjacent") is not None for row in all_rows)
    hazard_present_annotated = sum(row.get("hazard_present") is not None for row in all_rows)
    task_window_annotated = sum(row.get("in_task_window") is not None for row in all_rows)
    stop_timing_annotated = sum(row.get("stop_timing") is not None for row in all_rows)
    admissible_annotated = sum(
        row.get("admissible_action_labels") is not None for row in all_rows
    )
    safe_action_annotated = sum(
        row.get("safe_action_mask") is not None
        or row.get("candidate_action_safety") is not None
        for row in all_rows
    )
    structure_errors = scientific_structure_errors(all_rows)
    if structure_errors:
        icra_scientific_ready_reasons.append(
            "scientific annotation structure is invalid: "
            + ", ".join(f"{key}={value}" for key, value in sorted(structure_errors.items()))
        )
    if distractor_annotated != len(all_rows):
        icra_scientific_ready_reasons.append(
            f"is_distractor coverage is incomplete: {distractor_annotated}/{len(all_rows)}"
        )
    if risk_annotated != len(all_rows):
        icra_scientific_ready_reasons.append(
            f"risk_adjacent coverage is incomplete: {risk_annotated}/{len(all_rows)}"
        )
    hazard_presence_values = {
        row.get("hazard_present")
        for row in all_rows
        if isinstance(row.get("hazard_present"), bool)
    }
    if hazard_present_annotated != len(all_rows) or hazard_presence_values != {False, True}:
        icra_scientific_ready_reasons.append(
            "hazard_present must cover every row and include positive/negative controls: "
            f"coverage={hazard_present_annotated}/{len(all_rows)} values={sorted(hazard_presence_values)}"
        )
    distractor_values = {
        row.get("is_distractor")
        for row in all_rows
        if isinstance(row.get("is_distractor"), bool)
    }
    if distractor_values != {False, True}:
        icra_scientific_ready_reasons.append(
            f"hard distractor controls are incomplete; is_distractor values={sorted(distractor_values)}"
        )
    risk_values = {
        row.get("risk_adjacent")
        for row in all_rows
        if isinstance(row.get("risk_adjacent"), bool)
    }
    if risk_values != {False, True}:
        icra_scientific_ready_reasons.append(
            f"risk-adjacent positive/negative coverage is incomplete; values={sorted(risk_values)}"
        )
    task_window_values = {
        row.get("in_task_window")
        for row in all_rows
        if isinstance(row.get("in_task_window"), bool)
    }
    if task_window_values != {False, True}:
        icra_scientific_ready_reasons.append(
            f"task-window positive/negative coverage is incomplete; values={sorted(task_window_values)}"
        )
    for field_name, count in (
        ("in_task_window", task_window_annotated),
        ("stop_timing", stop_timing_annotated),
        ("admissible_action_labels", admissible_annotated),
        ("safe_action_mask_or_candidate_action_safety", safe_action_annotated),
    ):
        if count != len(all_rows):
            icra_scientific_ready_reasons.append(
                f"{field_name} coverage is incomplete: {count}/{len(all_rows)}"
            )
    alert_rows = [row for row in all_rows if row.get("action_group") == "alert"]
    gaze_annotated = sum(
        row.get("view_valid") is not None
        and (
            row.get("attention_target") is not None
            or row.get("target_bbox") is not None
            or (
                row.get("target_world_position") is not None
                and row.get("target_bearing_deg") is not None
            )
        )
        for row in alert_rows
    )
    if gaze_annotated != len(alert_rows):
        icra_scientific_ready_reasons.append(
            "gaze/localization coverage for alert decisions is incomplete: "
            f"{gaze_annotated}/{len(alert_rows)}"
        )
    train_hazards = {
        row.get("hazard_type") for row in buckets["train"] + buckets["val"] if row.get("hazard_type")
    }
    test_hazards = {row.get("hazard_type") for row in buckets["test"] if row.get("hazard_type")}
    if len(train_hazards) < 2:
        icra_scientific_ready_reasons.append(
            f"at least two train-seen hazard types are required; observed={sorted(train_hazards)}"
        )
    if not held_out_hazards:
        icra_scientific_ready_reasons.append("no held-out hazard taxonomy was declared")
    else:
        leaked_hazards = held_out_hazards & train_hazards
        absent_test_hazards = held_out_hazards - test_hazards
        if leaked_hazards:
            icra_scientific_ready_reasons.append(
                f"held-out hazards leak into train/val: {sorted(leaked_hazards)}"
            )
        if absent_test_hazards:
            icra_scientific_ready_reasons.append(
                f"declared held-out hazards are absent from test: {sorted(absent_test_hazards)}"
            )
    annotation_source_hash = _hash_files(source_files, source_root)
    selected_image_paths = {
        (image_root / image).resolve() for row in all_rows for image in row["images"]
    }
    selected_images_hash = _hash_files(list(selected_image_paths), image_root)
    manifest = {
        "schema_version": "saie_sft_manifest_v2",
        # Backward compatibility: paper_ready is deliberately an alias for
        # the existing-benchmark execution gate, not the stronger scientific
        # claim gate.  New consumers should use the explicit names below.
        "paper_ready": not benchmark_ready_reasons,
        "paper_ready_reasons": benchmark_ready_reasons,
        "paper_ready_alias": "benchmark_ready",
        "benchmark_ready": not benchmark_ready_reasons,
        "benchmark_ready_reasons": benchmark_ready_reasons,
        "icra_scientific_ready": not icra_scientific_ready_reasons,
        "icra_scientific_ready_reasons": icra_scientific_ready_reasons,
        "annotation_coverage": {
            "decision_rows": len(all_rows),
            "is_distractor": distractor_annotated,
            "risk_adjacent": risk_annotated,
            "hazard_present": hazard_present_annotated,
            "in_task_window": task_window_annotated,
            "stop_timing": stop_timing_annotated,
            "admissible_action_labels": admissible_annotated,
            "safe_action_supervision": safe_action_annotated,
            "alert_decisions": len(alert_rows),
            "alert_gaze_or_localization": gaze_annotated,
            "scientific_structure_errors": dict(sorted(structure_errors.items())),
        },
        "source_root": str(source_root),
        "image_root_at_build_time": str(image_root),
        "image_paths_are_relative": True,
        "source_sha256": annotation_source_hash,
        "annotation_source_sha256": annotation_source_hash,
        "source_hash_contract": "sha256(relative_path_nul_bytes_nul)",
        "source_file_count": len(set(source_files)),
        "selected_images_sha256": selected_images_hash,
        "selected_image_count": len(selected_image_paths),
        "selected_image_hash_contract": "sha256(image_root_relative_path_nul_bytes_nul)",
        "split_jsonl_sha256": {
            split: _hash_file(outdir / f"{split}.jsonl") for split in SPLITS
        },
        "repair_log": {
            "path": "repair_log.jsonl",
            "rows": len(resolver.repair_records),
            "sha256": _hash_file(outdir / "repair_log.jsonl"),
        },
        "split_manifest_sha256": (
            hashlib.sha256(args.split_manifest.resolve().read_bytes()).hexdigest()
            if args.split_manifest
            else None
        ),
        "ablation": args.ablation,
        "expected_corpus_counts": expectation_values,
        "observed_corpus_counts": {
            "assigned_warehouses": len(all_assigned_warehouses),
            "assigned_test_warehouses": len(assigned_warehouse_sets["test"]),
            "train_pool_warehouses": len(train_pool_warehouses),
            "valid_episodes": len(valid_episodes),
            "source_valid_decision_rows": len(source_valid_rows),
            "emitted_decision_rows": len(all_rows),
            "emitted_warehouses": len(set().union(*split_warehouses.values())),
        },
        "held_out_hazards": sorted(held_out_hazards),
        "split_source": str(args.split_manifest.resolve()) if args.split_manifest else "DEV_AUTO_SPLIT",
        "split_assignment": dict(sorted(split_map.items())),
        "skipped_unassigned_warehouses": skipped_unassigned,
        "invalid_episodes_excluded": dict(sorted(invalid_episodes.items())),
        "missing_master_episodes": sorted(missing_master_episodes),
        "missing_sim_events_episodes": sorted(missing_sim_events_episodes),
        "missing_task_stops_by_valid_episode": missing_task_stops,
        "source_decision_order": {
            "reversals": source_decision_order_reversals,
            "examples": decision_order_examples,
        },
        "frame_resolution": {
            "exact_references": resolver.exact,
            "repaired_references": resolver.repaired,
            "repair_gap_histogram": {str(k): v for k, v in sorted(resolver.gaps.items())},
            "max_allowed_index_gap": args.max_frame_gap,
            "strict": args.strict_frame_paths,
            "repair_examples": resolver.examples,
            "source_noncausal_history_references_detected": source_noncausal_history_references,
            "output_noncausal_history_references": 0,
        },
        "deduplication": {
            "policy": args.duplicate_policy,
            "total_duplicate_rows_seen": duplicate_rows,
            "exact_duplicate_rows_removed": duplicate_rows - conflicting_duplicates,
            "conflicting_duplicate_rows_resolved": conflicting_duplicates,
            "examples": duplicate_examples,
            "conflict_examples": conflict_examples,
            # Only populated (non-zero) when --duplicate-policy resolve-via-motion:
            # conflicts arbitrated by comparing each candidate label's nominal
            # motion against the shared decision frame's actually-measured
            # sim_events.json motion_summary, vs. conflicts where that
            # comparison was unavailable or too close to trust (fell back to
            # exclude-conflicts either way).
            "motion_resolved_count": motion_resolved_count,
            "motion_resolved_examples": motion_resolved_examples,
            "motion_ambiguous_count": motion_ambiguous_count,
            "motion_ambiguous_examples": motion_ambiguous_examples,
        },
        # Full-corpus extension of annotation_review.md section 4's 52-row
        # sample audit: for every navigate row, the SE(2) distance between
        # its label's nominal primitive and the demonstrator's actually
        # measured motion at that decision frame (same NAV_SE2_K=0.5 metric
        # the judge itself uses). Always computed regardless of
        # --max-motion-error-m, so a filtering threshold can be chosen from
        # real numbers rather than guessed.
        "physical_grounding": {
            "metric": "sqrt(dx_err_m^2 + (0.5*dyaw_err_rad)^2)",
            "motion_lookup_tolerance_frames": _MOTION_LOOKUP_TOLERANCE_FRAMES,
            "caveat": (
                "measured motion is the NEAREST logged motion_summary within "
                f"{_MOTION_LOOKUP_TOLERANCE_FRAMES} render frames of the decision anchor, not "
                "always the exact frame (spot_events motion_summary logging runs on a fixed "
                "~30-frame cadence that decision anchors don't land on) -- part of the measured "
                "error is temporal misalignment (up to 0.5s at 30fps), not pure label/primitive "
                "mismatch, so these numbers are a plausible upper bound on the true grounding error"
            ),
            "navigate_rows_measured": _error_distribution(navigate_motion_errors),
            "max_motion_error_m_filter": args.max_motion_error_m,
            "train_rows_excluded_by_filter": motion_error_excluded_train,
        },
        "balance_policy": (
            {
                "cap": args.balance_cap,
                "seed": args.seed,
                "train_label_counts_before": balance_cap_before,
                "train_label_counts_after": balance_cap_after,
            }
            if args.balance_cap is not None
            else None
        ),
        "missing_train_labels": missing_train_labels,
        "missing_val_labels": missing_val_labels,
        "splits": {split: _summary(rows) for split, rows in buckets.items()},
    }
    (outdir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (outdir / "split_manifest.resolved.json").write_text(
        json.dumps(split_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {outdir}")
    for split in SPLITS:
        info = manifest["splits"][split]
        print(
            f"  {split:<5} rows={info['rows']:<6} episodes={info['episodes']:<4} "
            f"warehouses={len(info['warehouses'])}"
        )
    print(
        f"  frames exact={resolver.exact} repaired={resolver.repaired} "
        f"benchmark_ready={manifest['benchmark_ready']} "
        f"icra_scientific_ready={manifest['icra_scientific_ready']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
