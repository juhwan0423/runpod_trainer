#!/usr/bin/env python3
"""Audit SAIE annotations and prepare non-destructive SFT rows.

The simulator currently writes three separate sources per episode::

    warehouse_s*/master/<episode>.json
    warehouse_s*/annotations/<episode>.json
    warehouse_s*/frames/<episode>/{sim_events.json,spot_cam/rgb_*.png}

The fine-tuning draft, on the other hand, describes already-flattened decision
rows.  This tool is the explicit bridge between those formats.  It never edits
the simulator output.  When ``--prepare-jsonl`` is supplied it writes a new,
atomic JSONL file and records every derived or unavailable field in each row.

Important safety properties:

* Source image references are audited before preparation.
* A missing/non-grid action anchor is snapped to the nearest saved image only in
  the derived JSONL; the original anchor and signed delta remain in provenance.
* Multi-frame histories are rebuilt from saved frames no later than the selected
  current image.  Future-frame leakage in old annotation histories is not copied.
* Unknown action labels and episodes without any saved front-camera frame are
  skipped loudly, never coerced to a plausible label/image.
* Existing output files are not overwritten unless ``--overwrite`` is explicit.

Typical use::

    python annotation_audit.py \
      --dataset-root /path/to/SAIE/output/local_smoke \
      --report-json /tmp/saie_annotation_audit.json \
      --report-md /tmp/saie_annotation_audit.md

    python annotation_audit.py \
      --dataset-root /runpod-volume/benchmark_v1 \
      --prepare-jsonl /runpod-volume/sft/all.jsonl \
      --num-frames 1 --report-json /runpod-volume/sft/audit.json

Use ``--publication-strict`` for the paper corpus.  It intentionally fails for
smoke subsets and for the current fallen-worker-only/no-distractor corpus.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TextIO


CANONICAL_ACTIONS = (
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
CANONICAL_ACTION_SET = frozenset(CANONICAL_ACTIONS)
TASK_LABELS = (
    "[gauge] stop",
    "[count] stop",
    "[state] stop",
    "[alert] stop",
)
EXPECTED_HAZARDS = frozenset({"fallen_worker", "spill", "blocked_aisle"})
HAZARD_NAME_MAP = {
    "FALL_HUMAN": "fallen_worker",
    "FALLEN_WORKER": "fallen_worker",
    "SPILL": "spill",
    "BLOCKED_AISLE": "blocked_aisle",
}

VIDEO_SUFFIX_RE = re.compile(r"_t(?P<frame>[0-9]+)$")
IMAGE_FRAME_RE = re.compile(r"rgb_(?P<frame>[0-9]+)\.png$")
TOOL_RE = re.compile(r"^\[(?P<tool>[a-z_]+)\]")

# One canonical forward action can move at most 0.75 m.  This matches
# MAX_FORWARD_STEP_M in SAIE/src/benchmark/judge.py.
RISK_PROBE_M = 0.75
MAX_POSE_FRAME_DELTA = 30


@dataclass(frozen=True)
class EpisodePaths:
    warehouse_dir: Path
    warehouse_id: str
    master_path: Path
    annotation_path: Path
    events_path: Path
    image_dir: Path


class IssueCollector:
    """Aggregate issue counts while retaining only a few actionable examples."""

    def __init__(self, max_examples: int = 5) -> None:
        self.max_examples = max_examples
        self._items: dict[str, dict[str, Any]] = {}

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **context: Any,
    ) -> None:
        item = self._items.setdefault(
            code,
            {
                "code": code,
                "severity": severity,
                "message": message,
                "count": 0,
                "examples": [],
            },
        )
        item["count"] += 1
        if len(item["examples"]) < self.max_examples:
            item["examples"].append(context)

    def as_list(self) -> list[dict[str, Any]]:
        rank = {"error": 0, "warning": 1, "gap": 2, "info": 3}
        return sorted(
            self._items.values(),
            key=lambda x: (rank.get(x["severity"], 9), x["code"]),
        )

    def count(self, *severities: str) -> int:
        wanted = set(severities)
        return sum(
            int(item["count"])
            for item in self._items.values()
            if item["severity"] in wanted
        )

    def code_count(self, code: str) -> int:
        return int(self._items.get(code, {}).get("count", 0))


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json_dump(path: Path, value: Any, overwrite: bool) -> None:
    _check_output_path(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_text_dump(path: Path, text: str, overwrite: bool) -> None:
    _check_output_path(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _check_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {path}; pass --overwrite explicitly"
        )


def _warehouse_dirs(root: Path) -> list[Path]:
    if (root / "master").is_dir():
        return [root]
    found = {p.parent for p in root.rglob("master") if p.is_dir()}
    return sorted(found)


def _load_split_manifest(path: Path) -> dict[str, str]:
    """Load the same explicit warehouse split forms accepted by the SFT builder."""

    raw = _load_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"split manifest must be a JSON object: {path}")
    valid_splits = {"train", "val", "test"}
    if any(split in raw for split in valid_splits):
        unknown = set(raw) - valid_splits - {"version", "note"}
        if unknown:
            raise ValueError(f"unknown split-manifest keys in {path}: {sorted(unknown)}")
        assignment: dict[str, str] = {}
        for split in ("train", "val", "test"):
            warehouse_ids = raw.get(split, [])
            if not isinstance(warehouse_ids, list) or not all(
                isinstance(value, str) and value for value in warehouse_ids
            ):
                raise ValueError(
                    f"split manifest field {split!r} must be a list of warehouse ids"
                )
            for warehouse_id in warehouse_ids:
                if warehouse_id in assignment:
                    raise ValueError(
                        f"warehouse {warehouse_id!r} appears in multiple manifest splits"
                    )
                assignment[warehouse_id] = split
        return assignment

    assignment = {}
    for warehouse_id, split in raw.items():
        if split not in valid_splits:
            raise ValueError(
                f"warehouse {warehouse_id!r} has invalid manifest split {split!r}"
            )
        assignment[str(warehouse_id)] = str(split)
    return assignment


def discover_episodes(root: Path) -> tuple[list[EpisodePaths], list[Path]]:
    episodes: list[EpisodePaths] = []
    warehouse_dirs = _warehouse_dirs(root)
    for warehouse in warehouse_dirs:
        for master in sorted((warehouse / "master").glob("*.json")):
            episode = master.stem
            episodes.append(
                EpisodePaths(
                    warehouse_dir=warehouse,
                    warehouse_id=warehouse.name,
                    master_path=master,
                    annotation_path=warehouse / "annotations" / f"{episode}.json",
                    events_path=warehouse / "frames" / episode / "sim_events.json",
                    image_dir=warehouse / "frames" / episode / "spot_cam",
                )
            )
    return episodes, warehouse_dirs


def _parse_video_frame(video_id: Any) -> Optional[int]:
    if not isinstance(video_id, str):
        return None
    match = VIDEO_SUFFIX_RE.search(video_id)
    return int(match.group("frame")) if match else None


def _parse_image_frame(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    match = IMAGE_FRAME_RE.search(value)
    return int(match.group("frame")) if match else None


def _saved_images(image_dir: Path) -> tuple[list[int], dict[int, Path]]:
    by_frame: dict[int, Path] = {}
    if image_dir.is_dir():
        for path in image_dir.glob("rgb_*.png"):
            frame = _parse_image_frame(path.name)
            if frame is not None:
                by_frame[frame] = path
    return sorted(by_frame), by_frame


def nearest_saved_frame(anchor: int, saved: Sequence[int]) -> Optional[int]:
    """Return nearest saved frame; ties prefer the earlier (causal) image."""

    if not saved:
        return None
    pos = bisect.bisect_left(saved, anchor)
    if pos == 0:
        return int(saved[0])
    if pos == len(saved):
        return int(saved[-1])
    before, after = int(saved[pos - 1]), int(saved[pos])
    return before if anchor - before <= after - anchor else after


def causal_history(current: int, saved: Sequence[int], num_frames: int) -> list[int]:
    """Uniform unique history ending at ``current`` with no future frames."""

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    available = [int(frame) for frame in saved if frame <= current]
    if not available:
        return []
    if num_frames == 1:
        return [available[-1]]
    if len(available) <= num_frames:
        return available
    step = (len(available) - 1) / (num_frames - 1)
    picked = [available[round(i * step)] for i in range(num_frames)]
    picked[-1] = available[-1]
    return sorted(set(picked))


def _tool(label: str) -> Optional[str]:
    match = TOOL_RE.match(label)
    return match.group("tool") if match else None


def _hazard_type(master: Mapping[str, Any]) -> Optional[str]:
    raw = (master.get("hazard") or {}).get("type")
    if raw is None:
        return None
    raw_text = str(raw)
    return HAZARD_NAME_MAP.get(raw_text.upper(), raw_text.lower())


def _pose_index(events: Mapping[str, Any]) -> tuple[list[int], dict[int, tuple[float, float, float]]]:
    poses: dict[int, tuple[float, float, float]] = {}
    for event in events.get("spot_events", []) or []:
        if event.get("event") != "robot_pose":
            continue
        try:
            frame = int(event["render_frame"])
            poses[frame] = (
                float(event["x"]),
                float(event["y"]),
                float(event["yaw_rad"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(poses), poses


def _nearest_pose(
    frame: int,
    pose_frames: Sequence[int],
    poses: Mapping[int, tuple[float, float, float]],
) -> tuple[Optional[tuple[float, float, float]], Optional[int]]:
    nearest = nearest_saved_frame(frame, pose_frames)
    if nearest is None:
        return None, None
    delta = nearest - frame
    if abs(delta) > MAX_POSE_FRAME_DELTA:
        return None, delta
    return poses[nearest], delta


def _point_segment_distance(
    px: float,
    py: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    dx, dy = x1 - x0, y1 - y0
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
    return math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def derive_hazard_metadata(
    master: Mapping[str, Any],
    pose: Optional[tuple[float, float, float]],
) -> tuple[Optional[float], Optional[bool]]:
    """Derive human-hazard distance and risk adjacency from a sim robot pose.

    ``risk_adjacent`` here covers the hazard clearance only.  It does not claim
    to include wall/box collision adjacency, which needs floor/nav-map inputs.
    """

    hazard = master.get("hazard") or {}
    hazard_pose = hazard.get("pose_world") or []
    if pose is None or len(hazard_pose) < 2:
        return None, None
    hx, hy = float(hazard_pose[0]), float(hazard_pose[1])
    x, y, yaw = pose
    distance = math.hypot(hx - x, hy - y)
    clearance_raw = hazard.get("clearance")
    if clearance_raw is None:
        return distance, None
    clearance = float(clearance_raw)
    qx = x + math.cos(yaw) * RISK_PROBE_M
    qy = y + math.sin(yaw) * RISK_PROBE_M
    risky = _point_segment_distance(hx, hy, x, y, qx, qy) < clearance
    return distance, risky


def _finite_motion(motion: Any) -> bool:
    if not isinstance(motion, Mapping):
        return False
    required = ("vx_m_per_s", "vy_m_per_s", "omega_rad_per_s", "dx_m", "dyaw_rad", "duration_s")
    for key in required:
        try:
            value = float(motion[key])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def _relative_image_path(dataset_root: Path, image: Path) -> str:
    try:
        return image.relative_to(dataset_root).as_posix()
    except ValueError:
        return image.resolve().as_posix()


def _prepare_row(
    *,
    dataset_root: Path,
    paths: EpisodePaths,
    master: Mapping[str, Any],
    annotation: Mapping[str, Any],
    annotation_index: int,
    source_anchor: int,
    saved: Sequence[int],
    by_frame: Mapping[int, Path],
    num_frames: int,
    pose_frames: Sequence[int],
    poses: Mapping[int, tuple[float, float, float]],
) -> Optional[dict[str, Any]]:
    selected = nearest_saved_frame(source_anchor, saved)
    if selected is None:
        return None
    history_frames = causal_history(selected, saved, num_frames)
    if not history_frames:
        return None
    history_paths = [by_frame[frame] for frame in history_frames]
    pose, pose_delta = _nearest_pose(selected, pose_frames, poses)
    distance, risk_adjacent = derive_hazard_metadata(master, pose)

    source_episode = str(master.get("episode_id") or paths.master_path.stem)
    global_episode = f"{paths.warehouse_id}/{source_episode}"
    frame_id = (
        f"{paths.warehouse_id}__{source_episode}__"
        f"f{source_anchor:06d}__r{annotation_index:04d}"
    )
    label = str(annotation["a"])
    quality_flags: list[str] = []
    if selected != source_anchor:
        quality_flags.append("source_anchor_snapped_to_saved_image")
    if annotation.get("is_distractor") is None:
        quality_flags.append("is_distractor_unannotated")
    if annotation.get("in_task_window") is None:
        quality_flags.append("in_task_window_unannotated")
    if pose is None:
        quality_flags.append("sim_robot_pose_unavailable")

    explicit_distractor = annotation.get("is_distractor", master.get("is_distractor"))
    explicit_window = annotation.get("in_task_window")
    return {
        "schema_version": "saie_sft_decision_v1",
        "frame_id": frame_id,
        "frame_index": selected,
        "source_frame_index": source_anchor,
        "episode_id": global_episode,
        "source_episode_id": source_episode,
        "warehouse_id": paths.warehouse_id,
        "image": _relative_image_path(dataset_root, by_frame[selected]),
        "image_history": [
            _relative_image_path(dataset_root, path) for path in history_paths
        ],
        "instruction": annotation.get("q"),
        "action_label": label,
        # The bracketed tool is the faithful five-way group in the released
        # SAIE vocabulary: navigate/gauge/count/state/alert.
        "action_group": _tool(label),
        "motion": annotation.get("motion"),
        "hazard_type": _hazard_type(master),
        # Unknown is JSON null, never silently converted to a negative example.
        "is_distractor": explicit_distractor,
        "dist_to_hazard_m": round(distance, 4) if distance is not None else None,
        "risk_adjacent": risk_adjacent,
        "in_task_window": explicit_window,
        "split": master.get("split"),
        "split_tags": list(master.get("split_tags") or []),
        "provenance": {
            "master": _relative_image_path(dataset_root, paths.master_path),
            "annotation": _relative_image_path(dataset_root, paths.annotation_path),
            "annotation_index": annotation_index,
            "source_video_id": annotation.get("video_id"),
            "selected_frame_delta": selected - source_anchor,
            "robot_pose_frame_delta": pose_delta,
            "risk_adjacent_scope": "human_hazard_only",
        },
        "quality_flags": quality_flags,
    }


def _open_atomic_jsonl(path: Path, overwrite: bool) -> tuple[TextIO, Path]:
    _check_output_path(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    return os.fdopen(fd, "w", encoding="utf-8"), Path(tmp_name)


def audit_dataset(
    dataset_root: Path,
    *,
    prepare_jsonl: Optional[Path] = None,
    num_frames: int = 1,
    overwrite: bool = False,
    expected_warehouses: int = 70,
    expected_episodes: int = 1311,
    expected_decision_hz: float = 1.0,
    split_manifest: Optional[Path] = None,
    block_on_errors: bool = False,
    block_on_publication_gaps: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    issues = IssueCollector()
    episode_paths, warehouse_dirs = discover_episodes(dataset_root)
    split_manifest_path = split_manifest.resolve() if split_manifest is not None else None
    external_split_map = (
        _load_split_manifest(split_manifest_path) if split_manifest_path is not None else {}
    )
    stats: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    hazard_counts: Counter[str] = Counter()
    task_episode_coverage: Counter[str] = Counter()
    metadata_presence: Counter[str] = Counter()
    duration_values: list[float] = []
    decision_positive_deltas: list[int] = []
    source_episode_warehouses: defaultdict[str, set[str]] = defaultdict(set)
    split_warehouses: defaultdict[str, set[str]] = defaultdict(set)
    prepared_artifact_written = False

    if split_manifest_path is not None:
        discovered_warehouses = {path.name for path in warehouse_dirs}
        missing_assignments = sorted(discovered_warehouses - set(external_split_map))
        extra_assignments = sorted(set(external_split_map) - discovered_warehouses)
        for warehouse_id in missing_assignments:
            issues.add(
                "error",
                "split_manifest_assignment_missing",
                "A discovered warehouse is absent from the external split manifest.",
                warehouse=warehouse_id,
                split_manifest=str(split_manifest_path),
            )
        for warehouse_id in extra_assignments:
            issues.add(
                "error",
                "split_manifest_assignment_extra",
                "The external split manifest names a warehouse absent from the corpus.",
                warehouse=warehouse_id,
                split_manifest=str(split_manifest_path),
            )
        for warehouse_id, split in external_split_map.items():
            if warehouse_id in discovered_warehouses:
                split_warehouses[split].add(warehouse_id)

    writer: Optional[TextIO] = None
    tmp_jsonl: Optional[Path] = None
    if prepare_jsonl is not None:
        prepare_jsonl = prepare_jsonl.resolve()
        writer, tmp_jsonl = _open_atomic_jsonl(prepare_jsonl, overwrite)

    try:
        for paths in episode_paths:
            stats["master_files"] += 1
            try:
                master = _load_json(paths.master_path)
                if not isinstance(master, Mapping):
                    raise TypeError("master root is not an object")
            except (OSError, ValueError, TypeError) as exc:
                issues.add(
                    "error", "master_parse_error", "Master JSON cannot be parsed.",
                    path=str(paths.master_path), error=str(exc),
                )
                stats["unusable_episodes"] += 1
                continue

            source_episode = str(master.get("episode_id") or paths.master_path.stem)
            source_episode_warehouses[source_episode].add(paths.warehouse_id)
            if source_episode != paths.master_path.stem:
                issues.add(
                    "error", "master_episode_id_mismatch",
                    "Master episode_id does not match its file name.",
                    warehouse=paths.warehouse_id, file=paths.master_path.name,
                    episode_id=source_episode,
                )
            map_id = master.get("map_id")
            if map_id and str(map_id) != paths.warehouse_id:
                issues.add(
                    "error", "master_map_id_mismatch",
                    "Master map_id does not match the warehouse directory.",
                    warehouse=paths.warehouse_id, map_id=map_id, episode=source_episode,
                )

            hazard_type = _hazard_type(master)
            if hazard_type is not None:
                hazard_counts[hazard_type] += 1
            else:
                issues.add(
                    "gap", "missing_hazard_type",
                    "Episode has no explicit hazard type.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                )
            master_split = master.get("split")
            external_split = external_split_map.get(paths.warehouse_id)
            if split_manifest_path is not None:
                if master_split and external_split and str(master_split) != external_split:
                    issues.add(
                        "error",
                        "master_split_manifest_conflict",
                        "Master split disagrees with the external warehouse split manifest.",
                        warehouse=paths.warehouse_id,
                        episode=source_episode,
                        master_split=master_split,
                        manifest_split=external_split,
                    )
            elif master_split:
                split_warehouses[str(master_split)].add(paths.warehouse_id)

            if not paths.annotation_path.is_file():
                issues.add(
                    "error", "missing_annotation_file",
                    "Master has no matching per-episode annotation JSON.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                    expected=str(paths.annotation_path),
                )
                stats["unusable_episodes"] += 1
                continue
            stats["annotation_files"] += 1
            try:
                annotations = _load_json(paths.annotation_path)
                if not isinstance(annotations, list):
                    raise TypeError("annotation root is not a list")
            except (OSError, ValueError, TypeError) as exc:
                issues.add(
                    "error", "annotation_parse_error",
                    "Annotation JSON cannot be parsed as a list.",
                    path=str(paths.annotation_path), error=str(exc),
                )
                stats["unusable_episodes"] += 1
                continue

            events: Mapping[str, Any] = {}
            if paths.events_path.is_file():
                stats["sim_event_files"] += 1
                try:
                    loaded_events = _load_json(paths.events_path)
                    if isinstance(loaded_events, Mapping):
                        events = loaded_events
                    else:
                        raise TypeError("sim_events root is not an object")
                except (OSError, ValueError, TypeError) as exc:
                    issues.add(
                        "error", "sim_events_parse_error",
                        "sim_events.json cannot be parsed.",
                        path=str(paths.events_path), error=str(exc),
                    )
            else:
                issues.add(
                    "error", "missing_sim_events",
                    "Episode has no sim_events.json; event alignment cannot be verified.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                )

            invalid = bool(events.get("invalid", False))
            if invalid:
                stats["invalid_episodes"] += 1
                issues.add(
                    "error", "invalid_episode",
                    "Simulator marked the episode invalid; exclude it from training/evaluation.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                    reason=events.get("invalid_reason"),
                )

            saved, by_frame = _saved_images(paths.image_dir)
            stats["saved_front_images"] += len(saved)
            if not saved:
                issues.add(
                    "error", "missing_front_images",
                    "Episode has no saved spot_cam RGB images.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                    directory=str(paths.image_dir),
                )
            declared_spot_frames = events.get("spot_frames")
            if declared_spot_frames is not None:
                try:
                    if int(declared_spot_frames) != len(saved):
                        issues.add(
                            "warning", "spot_frame_count_mismatch",
                            "sim_events spot_frames differs from files on disk.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            declared=int(declared_spot_frames), actual=len(saved),
                        )
                except (TypeError, ValueError):
                    issues.add(
                        "warning", "invalid_spot_frame_count",
                        "sim_events spot_frames is not an integer.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        value=declared_spot_frames,
                    )

            pose_frames, poses = _pose_index(events)
            episode_labels: Counter[str] = Counter()
            seen_video_ids: set[str] = set()
            previous_anchor: Optional[int] = None
            instruction_values: set[str] = set()
            video_id_indices: defaultdict[str, list[int]] = defaultdict(list)
            video_id_labels: defaultdict[str, set[str]] = defaultdict(set)
            for raw_index, raw_annotation in enumerate(annotations):
                if not isinstance(raw_annotation, Mapping):
                    continue
                raw_video_id = raw_annotation.get("video_id")
                if isinstance(raw_video_id, str):
                    video_id_indices[raw_video_id].append(raw_index)
                    raw_label = raw_annotation.get("a")
                    video_id_labels[raw_video_id].add(
                        raw_label if isinstance(raw_label, str) else repr(raw_label)
                    )
            conflicting_video_ids = {
                video_id
                for video_id, labels in video_id_labels.items()
                if len(labels) > 1
            }
            for video_id in sorted(conflicting_video_ids):
                issues.add(
                    "error", "conflicting_duplicate_label",
                    "The same decision-frame ID has conflicting supervision labels; all copies are excluded from prepared data.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                    video_id=video_id,
                    labels=sorted(str(v) for v in video_id_labels[video_id]),
                )

            for annotation_index, annotation in enumerate(annotations):
                stats["annotation_rows"] += 1
                if not isinstance(annotation, Mapping):
                    issues.add(
                        "error", "annotation_row_not_object",
                        "Annotation row is not a JSON object.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index,
                    )
                    stats["skipped_prepare_rows"] += 1
                    continue

                missing_keys = [
                    key for key in ("video_id", "q", "a", "frames", "motion")
                    if key not in annotation
                ]
                if missing_keys:
                    issues.add(
                        "error", "annotation_required_key_missing",
                        "Annotation row is missing required source-schema keys.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index, keys=missing_keys,
                    )

                video_id = annotation.get("video_id")
                source_anchor = _parse_video_frame(video_id)
                if source_anchor is None:
                    issues.add(
                        "error", "invalid_video_id",
                        "video_id does not end in _t<render-frame>.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index, video_id=video_id,
                    )
                    stats["skipped_prepare_rows"] += 1
                    continue
                if isinstance(video_id, str):
                    if video_id in seen_video_ids:
                        issues.add(
                            "error", "duplicate_video_id",
                            "An episode contains duplicate decision-frame IDs.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            video_id=video_id,
                        )
                    seen_video_ids.add(video_id)
                    if not video_id.startswith(f"{source_episode}_"):
                        issues.add(
                            "error", "video_episode_id_mismatch",
                            "video_id prefix does not match master episode_id.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            video_id=video_id,
                        )

                if previous_anchor is not None:
                    delta = source_anchor - previous_anchor
                    if delta > 0:
                        decision_positive_deltas.append(delta)
                    elif delta < 0:
                        issues.add(
                            "warning", "non_monotonic_decision_frames",
                            "Decision frame anchors go backwards within an episode.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            previous=previous_anchor, current=source_anchor,
                            row=annotation_index,
                        )
                previous_anchor = source_anchor

                label = annotation.get("a")
                if not isinstance(label, str) or label not in CANONICAL_ACTION_SET:
                    issues.add(
                        "error", "unknown_action_label",
                        "Action is outside the released SAIE canonical 14-label vocabulary.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index, label=label,
                    )
                    stats["skipped_prepare_rows"] += 1
                    continue
                label_counts[label] += 1
                episode_labels[label] += 1
                tool = _tool(label)
                if tool:
                    tool_counts[tool] += 1

                instruction = annotation.get("q")
                if not isinstance(instruction, str) or not instruction.strip():
                    issues.add(
                        "error", "missing_instruction",
                        "Annotation row has no non-empty instruction.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index,
                    )
                    stats["skipped_prepare_rows"] += 1
                    continue
                instruction_values.add(instruction)

                motion = annotation.get("motion")
                if not _finite_motion(motion):
                    issues.add(
                        "warning", "invalid_motion_metadata",
                        "Motion auxiliary metadata is missing, non-finite, or incomplete.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index,
                    )
                else:
                    duration_values.append(float(motion["duration_s"]))
                    if label in TASK_LABELS and (
                        abs(float(motion["dx_m"])) > 1e-6
                        or abs(float(motion["dyaw_rad"])) > 1e-6
                    ):
                        issues.add(
                            "warning", "task_stop_has_nonzero_motion",
                            "Task-stop target is paired with non-zero auxiliary motion; define whether motion is preceding context or action consequence.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index, label=label, motion=motion,
                        )

                frames = annotation.get("frames")
                frame_indices: list[int] = []
                if not isinstance(frames, list) or not frames:
                    issues.add(
                        "error", "missing_frame_history",
                        "Annotation row has no non-empty frames list.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index,
                    )
                else:
                    stats["source_frame_references"] += len(frames)
                    for ref in frames:
                        ref_frame = _parse_image_frame(ref)
                        if ref_frame is None:
                            issues.add(
                                "error", "invalid_frame_reference",
                                "Frame reference is not an rgb_<frame>.png path.",
                                warehouse=paths.warehouse_id, episode=source_episode,
                                row=annotation_index, reference=ref,
                            )
                            continue
                        frame_indices.append(ref_frame)
                        if ref_frame not in by_frame:
                            stats["missing_source_frame_references"] += 1
                            issues.add(
                                "warning", "missing_source_frame_reference",
                                "Annotation references an image not present on the saved-frame grid; preparation will snap/rebuild it.",
                                warehouse=paths.warehouse_id, episode=source_episode,
                                row=annotation_index, reference=ref,
                            )
                    if len(frame_indices) != len(set(frame_indices)):
                        issues.add(
                            "warning", "duplicate_history_frames",
                            "Annotation history contains duplicate frame indices.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index,
                        )
                    if frame_indices != sorted(frame_indices):
                        issues.add(
                            "warning", "non_monotonic_history",
                            "Annotation history is not in chronological order.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index, frames=frame_indices,
                        )
                    if any(frame > source_anchor for frame in frame_indices):
                        issues.add(
                            "warning", "future_frame_leakage",
                            "Annotation history contains a frame later than its decision anchor.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index, anchor=source_anchor,
                            max_history=max(frame_indices),
                        )
                    if frame_indices and frame_indices[-1] != source_anchor:
                        issues.add(
                            "warning", "history_anchor_mismatch",
                            "Last history frame does not equal video_id decision anchor.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index, anchor=source_anchor,
                            last_history=frame_indices[-1],
                        )

                if source_anchor not in by_frame:
                    selected = nearest_saved_frame(source_anchor, saved)
                    issues.add(
                        "warning", "missing_current_image",
                        "Decision anchor has no exact PNG; derived data must snap to a saved frame.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        row=annotation_index, anchor=source_anchor,
                        nearest=selected,
                        delta=(selected - source_anchor) if selected is not None else None,
                    )

                for key in ("is_distractor", "in_task_window", "dist_to_hazard_m", "risk_adjacent"):
                    if key in annotation and annotation[key] is not None:
                        metadata_presence[key] += 1

                if writer is not None:
                    if invalid:
                        # Invalid episodes are never copied to a training artifact,
                        # even when their frames happen to remain on disk.
                        stats["excluded_invalid_prepare_rows"] += 1
                        continue
                    if video_id in conflicting_video_ids:
                        stats["excluded_conflicting_prepare_rows"] += 1
                        continue
                    if (
                        isinstance(video_id, str)
                        and len(video_id_indices[video_id]) > 1
                        and annotation_index != video_id_indices[video_id][0]
                    ):
                        # Exact duplicate labels add no signal and would silently
                        # reweight a frame.  Keep the first row only.
                        stats["excluded_duplicate_prepare_rows"] += 1
                        continue
                    row = _prepare_row(
                        dataset_root=dataset_root,
                        paths=paths,
                        master=master,
                        annotation=annotation,
                        annotation_index=annotation_index,
                        source_anchor=source_anchor,
                        saved=saved,
                        by_frame=by_frame,
                        num_frames=num_frames,
                        pose_frames=pose_frames,
                        poses=poses,
                    )
                    if row is None:
                        issues.add(
                            "error", "prepare_image_unavailable",
                            "Could not select an existing image for an otherwise valid row.",
                            warehouse=paths.warehouse_id, episode=source_episode,
                            row=annotation_index,
                        )
                        stats["skipped_prepare_rows"] += 1
                    else:
                        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stats["prepared_rows"] += 1
                        if row["frame_index"] != row["source_frame_index"]:
                            stats["prepared_rows_with_snapped_image"] += 1

            if len(instruction_values) > 1:
                issues.add(
                    "warning", "inconsistent_episode_instruction",
                    "Instruction text changes within one episode.",
                    warehouse=paths.warehouse_id, episode=source_episode,
                    variants=len(instruction_values),
                )

            for task_label in TASK_LABELS:
                if episode_labels[task_label] > 0:
                    task_episode_coverage[task_label] += 1
                else:
                    issues.add(
                        "error", "missing_task_stop_label",
                        "Episode lacks one of the four required task-stop labels.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        label=task_label,
                    )
                if episode_labels[task_label] > 1:
                    issues.add(
                        "warning", "duplicate_task_stop_label",
                        "Episode has multiple one-shot task-stop labels.",
                        warehouse=paths.warehouse_id, episode=source_episode,
                        label=task_label, count=episode_labels[task_label],
                    )

            stats["audited_episodes"] += 1

        for episode, warehouses in source_episode_warehouses.items():
            if len(warehouses) > 1:
                issues.add(
                    "gap", "non_global_source_episode_ids",
                    "Raw episode_id repeats across warehouses; use warehouse-qualified IDs for SFT/splits.",
                    episode=episode, warehouses=sorted(warehouses),
                )

        observed_hazards = set(hazard_counts)
        for missing_hazard in sorted(EXPECTED_HAZARDS - observed_hazards):
            issues.add(
                "gap", "missing_hazard_category",
                "Masterplan generalization hazard category is absent from the corpus.",
                hazard_type=missing_hazard,
            )
        if not observed_hazards:
            issues.add(
                "gap", "no_hazard_coverage", "No hazard-bearing episode was found."
            )
        if metadata_presence["is_distractor"] == 0:
            issues.add(
                "gap", "distractor_annotation_absent",
                "No row has an explicit is_distractor value; false must not be imputed from absence."
            )
        if metadata_presence["in_task_window"] == 0:
            issues.add(
                "gap", "task_window_annotation_absent",
                "No row marks valid inspection windows, so premature/false-positive stops cannot be analysed."
            )
        if metadata_presence["dist_to_hazard_m"] == 0:
            issues.add(
                "gap", "hazard_distance_annotation_absent",
                "No source row stores per-frame hazard distance; prepared rows derive it from nearest sim pose when possible."
            )
        if metadata_presence["risk_adjacent"] == 0:
            issues.add(
                "gap", "risk_adjacent_annotation_absent",
                "No source row stores risk adjacency; prepared rows derive human-hazard adjacency only when pose is available."
            )
        if not split_warehouses:
            issues.add(
                "gap", "split_manifest_absent",
                "No warehouse-level train/val/test split is encoded in masters; do not split by repeated raw episode_id."
            )
        if expected_warehouses > 0 and len(warehouse_dirs) != expected_warehouses:
            issues.add(
                "gap", "warehouse_count_below_paper_target",
                "Corpus does not match the reference paper warehouse count.",
                expected=expected_warehouses, actual=len(warehouse_dirs),
            )
        if expected_episodes > 0 and stats["audited_episodes"] != expected_episodes:
            issues.add(
                "gap", "episode_count_below_paper_target",
                "Corpus does not match the reference paper episode count.",
                expected=expected_episodes, actual=stats["audited_episodes"],
            )

        median_duration = statistics.median(duration_values) if duration_values else None
        if expected_decision_hz > 0 and median_duration is not None:
            expected_duration = 1.0 / expected_decision_hz
            if not math.isclose(median_duration, expected_duration, rel_tol=0.05, abs_tol=0.01):
                issues.add(
                    "gap", "decision_rate_contract_mismatch",
                    "Motion windows imply a different decision rate than the configured paper contract.",
                    expected_hz=expected_decision_hz,
                    expected_duration_s=expected_duration,
                    median_motion_duration_s=median_duration,
                )

        if writer is not None:
            writer.flush()
            os.fsync(writer.fileno())
            writer.close()
            writer = None
            assert tmp_jsonl is not None and prepare_jsonl is not None
            should_block = (
                block_on_publication_gaps
                and issues.count("error", "warning", "gap") > 0
            ) or (
                block_on_errors and issues.count("error") > 0
            )
            if should_block:
                tmp_jsonl.unlink()
                stats["prepared_write_blocked"] = 1
            else:
                os.replace(tmp_jsonl, prepare_jsonl)
                prepared_artifact_written = True
            tmp_jsonl = None
    except BaseException:
        if writer is not None:
            writer.close()
        if tmp_jsonl is not None:
            try:
                tmp_jsonl.unlink()
            except FileNotFoundError:
                pass
        raise

    row_count = stats["annotation_rows"]
    majority_label = label_counts.most_common(1)[0] if label_counts else (None, 0)
    majority_tool = tool_counts.most_common(1)[0] if tool_counts else (None, 0)
    summary = {
        "dataset_root": str(dataset_root),
        "contract": {
            "canonical_actions": list(CANONICAL_ACTIONS),
            "expected_hazards": sorted(EXPECTED_HAZARDS),
            "expected_warehouses": expected_warehouses,
            "expected_episodes": expected_episodes,
            "expected_decision_hz": expected_decision_hz,
            "split_manifest": str(split_manifest_path) if split_manifest_path else None,
            "invalid_episode_policy": "exclude_from_prepared_jsonl",
            "missing_boolean_policy": "preserve_as_null",
        },
        "counts": dict(sorted(stats.items())),
        "warehouse_ids": [path.name for path in warehouse_dirs],
        "split_warehouses": {
            split: sorted(warehouse_ids)
            for split, warehouse_ids in sorted(split_warehouses.items())
        },
        "hazard_episode_counts": dict(sorted(hazard_counts.items())),
        "action_label_counts": {
            label: label_counts.get(label, 0) for label in CANONICAL_ACTIONS
        },
        "tool_counts": dict(sorted(tool_counts.items())),
        "task_episode_coverage": {
            label: task_episode_coverage.get(label, 0) for label in TASK_LABELS
        },
        "source_metadata_non_null_rows": {
            key: metadata_presence.get(key, 0)
            for key in ("is_distractor", "in_task_window", "dist_to_hazard_m", "risk_adjacent")
        },
        "distribution": {
            "majority_label": majority_label[0],
            "majority_label_share": (majority_label[1] / row_count) if row_count else None,
            "majority_tool": majority_tool[0],
            "majority_tool_share": (majority_tool[1] / row_count) if row_count else None,
            "median_motion_duration_s": median_duration,
            "median_positive_decision_frame_delta": (
                statistics.median(decision_positive_deltas)
                if decision_positive_deltas else None
            ),
        },
        "requested_prepared_output": str(prepare_jsonl) if prepare_jsonl else None,
        "prepared_output": (
            str(prepare_jsonl) if prepared_artifact_written else None
        ),
        "readiness": {
            "source_mechanically_clean": issues.count("error", "warning") == 0,
            "prepared_mechanically_complete": (
                prepared_artifact_written
                and stats["skipped_prepare_rows"] == 0
                and stats["prepared_rows"]
                == stats["annotation_rows"]
                - stats["excluded_invalid_prepare_rows"]
                - stats["excluded_conflicting_prepare_rows"]
                - stats["excluded_duplicate_prepare_rows"]
            ),
            "publication_ready": issues.count("error", "warning", "gap") == 0,
            "error_count": issues.count("error"),
            "warning_count": issues.count("warning"),
            "gap_count": issues.count("gap"),
        },
        "issues": issues.as_list(),
    }
    return summary


def markdown_report(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    ready = report["readiness"]
    dist = report["distribution"]
    lines = [
        "# SAIE annotation audit",
        "",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Warehouses: {len(report['warehouse_ids'])}",
        f"- Audited episodes: {counts.get('audited_episodes', 0)}",
        f"- Invalid episodes: {counts.get('invalid_episodes', 0)}",
        f"- Decision rows: {counts.get('annotation_rows', 0)}",
        f"- Missing source image references: {counts.get('missing_source_frame_references', 0)}",
        f"- Prepared rows: {counts.get('prepared_rows', 0)}",
        f"- Excluded invalid rows: {counts.get('excluded_invalid_prepare_rows', 0)}",
        f"- Skipped unrepairable rows: {counts.get('skipped_prepare_rows', 0)}",
        f"- Majority tool/share: {dist.get('majority_tool')} / {dist.get('majority_tool_share')}",
        f"- Source mechanically clean: {ready['source_mechanically_clean']}",
        f"- Publication ready: {ready['publication_ready']}",
        "",
        "## Coverage",
        "",
        f"- Hazard episodes: `{json.dumps(report['hazard_episode_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Task-label episode coverage: `{json.dumps(report['task_episode_coverage'], ensure_ascii=False, sort_keys=True)}`",
        f"- Explicit metadata rows: `{json.dumps(report['source_metadata_non_null_rows'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Issues",
        "",
        "| Severity | Code | Count | Meaning |",
        "|---|---|---:|---|",
    ]
    for item in report["issues"]:
        message = str(item["message"]).replace("|", "\\|")
        lines.append(
            f"| {item['severity']} | `{item['code']}` | {item['count']} | {message} |"
        )
    lines.extend(["", "## Action distribution", "", "| Label | Count |", "|---|---:|"])
    for label, count in report["action_label_counts"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def _print_summary(report: Mapping[str, Any]) -> None:
    counts = report["counts"]
    ready = report["readiness"]
    print(
        "[audit] "
        f"warehouses={len(report['warehouse_ids'])} "
        f"episodes={counts.get('audited_episodes', 0)} "
        f"rows={counts.get('annotation_rows', 0)} "
        f"invalid_episodes={counts.get('invalid_episodes', 0)}"
    )
    print(
        "[audit] "
        f"missing_refs={counts.get('missing_source_frame_references', 0)} "
        f"prepared={counts.get('prepared_rows', 0)} "
        f"excluded_invalid={counts.get('excluded_invalid_prepare_rows', 0)} "
        f"skipped={counts.get('skipped_prepare_rows', 0)}"
    )
    print(
        "[audit] "
        f"errors={ready['error_count']} warnings={ready['warning_count']} "
        f"gaps={ready['gap_count']} publication_ready={ready['publication_ready']}"
    )
    for item in report["issues"]:
        print(
            f"  {item['severity'].upper():7} {item['code']:<42} "
            f"{item['count']:>7}  {item['message']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--prepare-jsonl", type=Path,
        help="Optional derived unified JSONL. Source files are never edited.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=1,
        help="Number of causal saved images in image_history (default: 1).",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-warehouses", type=int, default=70)
    parser.add_argument("--expected-episodes", type=int, default=1311)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help=(
            "Optional external warehouse-level train/val/test JSON. This is the "
            "authoritative split when masters intentionally remain immutable."
        ),
    )
    parser.add_argument(
        "--expected-decision-hz", type=float, default=1.0,
        help=(
            "Expected action-decision rate (default: 1 Hz per PDF Appendix A/Table 4). "
            "This is distinct from spot_cam image capture at 2 Hz. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Return exit 2 for any mechanical error (warnings remain non-fatal).",
    )
    parser.add_argument(
        "--publication-strict", action="store_true",
        help="Return exit 2 for errors, warnings, or scientific coverage gaps.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_dataset(
            args.dataset_root,
            prepare_jsonl=args.prepare_jsonl,
            num_frames=args.num_frames,
            overwrite=args.overwrite,
            expected_warehouses=args.expected_warehouses,
            expected_episodes=args.expected_episodes,
            expected_decision_hz=args.expected_decision_hz,
            split_manifest=args.split_manifest,
            block_on_errors=args.strict or args.publication_strict,
            block_on_publication_gaps=args.publication_strict,
        )
        if args.report_json:
            _atomic_json_dump(args.report_json.resolve(), report, args.overwrite)
        if args.report_md:
            _atomic_text_dump(
                args.report_md.resolve(), markdown_report(report), args.overwrite
            )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[audit:error] {exc}", file=sys.stderr)
        return 2

    _print_summary(report)
    readiness = report["readiness"]
    if args.publication_strict and not readiness["publication_ready"]:
        return 2
    if args.strict and readiness["error_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
