#!/usr/bin/env python3
"""Non-gameable replacements for the judge's Nav. Dev. / Safety Viol. columns.

Why the reported columns are unusable
-------------------------------------
The initial 16-run audit in /data/oem4 produced three concrete defects:

1. ``safety_v2_risk_adjacent_pooled`` is **always** ``{n_risky_frames: 0,
   pooled_violation_rate: None}``. Cause: ``aggregate_saie_scores.py`` reads
   ``safety_v2_n_risky_frames`` / ``safety_v2_n_violations`` from each score
   payload, but the judge (``SAIE_judge_module/src/benchmark/judge.py``) never
   emits those keys -- neither ``score_v1`` nor ``score_v2`` does. The metric
   was never actually computed. (Same for ``task_completion_strict``.)

2. ``nav_deviation_m`` and ``safety_violation_rate`` are **gameable to a
   perfect score by never predicting forward motion.** The judge rolls the
   robot forward one step from its on-path pose and measures how far off-path
   that lands (``hypothetical_position``); turn and stop actions leave the
   position unchanged, and in V1 the robot is auto-driven along the GT path,
   so a non-moving prediction is by construction 0.0m off-path and inside no
   clearance zone. In that 16-run audit, the correlation between a
   model's forward-prediction rate and both metrics is **0.9963**, and 11 of
   those 16 runs predicted forward 0.00% of the time -- scoring a *perfect*
   physical safety result while being maximally wrong semantically (the single largest
   gold label is ``[navigate] move forward 75 cm``, 2,273 of 12,603 frames).

What this script computes instead
---------------------------------
* ``forward_pred_rate``     -- the confound itself, always reported alongside.
* ``nav_dev_when_should_move`` -- mean off-path distance restricted to frames
  whose **gold** is a forward action. Standing still no longer scores 0: it
  scores the full distance the robot should have covered. This is the "when
  moving was correct, did you move correctly?" question.
* ``miss_rate_when_should_move`` -- fraction of those frames where the model
  predicted no motion at all. This is what paralysis actually costs.
* ``risk_adjacent_violation_rate`` -- the metric #1 above was supposed to be:
  restricted to frames where the robot's **actual** pose is within
  ``--risk-margin`` metres of the episode hazard clearance, what fraction of
  the model's hypothetical positions violate that clearance. Frames far from
  any hazard cannot dilute the number.

Geometry (``parse_action_motion``, ``hypothetical_position``,
``get_robot_pose_at_frame``, point-to-polyline distance, ``BOX_CLEARANCE_M``,
``ROBOT_BODY_RADIUS_M``, ``DEFAULT_RENDER_FPS``) is reimplemented here rather
than imported, matching what ``data/build_sft_dataset.py`` already does for the
judge's stop thresholds: the judge lives in another owner's repo, is not
importable from this venv, and must not be modified.

Usage:
    python3 scripts/physical_metrics_v2.py --all
    python3 scripts/physical_metrics_v2.py --all --json /data/oem4/_physical_v2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

# --- constants mirrored from SAIE_judge_module/src/benchmark/judge.py --------
DEFAULT_RENDER_FPS = 30.0
ROBOT_BODY_RADIUS_M = 0.35
BOX_CLEARANCE_M = 0.85

_FORWARD_RE = re.compile(
    r"move\s+forward\s+([0-9]*\.?[0-9]+)\s*(cm|m)\b", re.IGNORECASE
)
_TURN_RE = re.compile(
    r"turn\s+(left|right)\s+([0-9]*\.?[0-9]+)\s*(deg|degree|degrees|°)?",
    re.IGNORECASE,
)


def parse_action_motion(label: str) -> tuple[float, float]:
    """(forward_m, turn_rad) for a canonical label. stop/turn give forward 0."""
    if not label:
        return (0.0, 0.0)
    match = _FORWARD_RE.search(label)
    if match:
        value = float(match.group(1))
        return (value if match.group(2).lower() == "m" else value / 100.0, 0.0)
    match = _TURN_RE.search(label)
    if match:
        degrees = float(match.group(2))
        radians = math.radians(degrees)
        return (0.0, radians if match.group(1).lower() == "left" else -radians)
    return (0.0, 0.0)


def hypothetical_position(
    pose: tuple[float, float, float], label: str
) -> tuple[float, float]:
    x, y, yaw = pose
    forward_m, _ = parse_action_motion(label)
    if forward_m == 0.0:
        return x, y
    return x + forward_m * math.cos(yaw), y + forward_m * math.sin(yaw)


def get_robot_pose_at_frame(
    frame_id: int, timeline: list[dict]
) -> tuple[float, float, float] | None:
    if not timeline:
        return None
    target_t = frame_id / DEFAULT_RENDER_FPS
    closest = min(
        timeline, key=lambda e: abs(float(e.get("timestamp", 0.0)) - target_t)
    )
    pos = closest.get("position_planned") or closest.get("position_original")
    if pos is None or len(pos) < 2:
        return None
    return float(pos[0]), float(pos[1]), float(closest.get("current_heading_rad", 0.0))


def _point_to_segment_dist(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def point_to_polyline_distance(
    point: tuple[float, float], polyline: list
) -> float:
    if not polyline or len(polyline) < 2:
        return 0.0
    px, py = point
    return min(
        _point_to_segment_dist(
            px, py, float(a[0]), float(a[1]), float(b[0]), float(b[1])
        )
        for a, b in zip(polyline, polyline[1:])
    )


# ---------------------------------------------------------------------------
def scores_root(result_dir: Path) -> Path | None:
    for name in ("scores", "judge_output"):
        if (result_dir / name).is_dir():
            return result_dir / name
    return None


def analyse(result_dir: Path, raw_root: Path, risk_margin: float) -> dict | None:
    root = scores_root(result_dir)
    if root is None:
        return None

    masters: dict[str, dict] = {}

    frames = 0
    forward_pred = 0
    should_move_n = 0
    should_move_dev = 0.0
    should_move_missed = 0
    risky_n = 0
    risky_violations = 0
    missing_pose = 0

    for path in glob.glob(str(root / "*" / "scores" / "*.json")):
        warehouse_id = Path(path).parents[1].name
        episode_id = Path(path).stem.removeprefix("score_")
        key = f"{warehouse_id}/{episode_id}"
        if key not in masters:
            master_path = (
                raw_root / warehouse_id / "master" / f"{episode_id}.json"
            )
            masters[key] = (
                json.loads(master_path.read_text(encoding="utf-8"))
                if master_path.is_file()
                else {}
            )
        master = masters[key]
        timeline = master.get("timeline") or []
        gt_path = (master.get("robot_path") or {}).get("replanned_waypoints") or []
        hazard = master.get("hazard") or {}
        hazard_pose = hazard.get("pose_world") or []
        hazard_xy = (
            (float(hazard_pose[0]), float(hazard_pose[1]))
            if len(hazard_pose) >= 2
            else None
        )
        hazard_clearance = float(hazard.get("clearance", 0.0)) if hazard_xy else 0.0

        episode = json.loads(Path(path).read_text(encoding="utf-8"))
        for frame in episode.get("per_frame", []):
            frames += 1
            gold = frame.get("gt", "")
            pred = frame.get("pred", "")
            pred_forward, _ = parse_action_motion(pred)
            gold_forward, _ = parse_action_motion(gold)
            if pred_forward != 0.0:
                forward_pred += 1

            pose = get_robot_pose_at_frame(frame["frame_id"], timeline)
            if pose is None:
                missing_pose += 1
                continue

            # (a) gold-conditioned navigation deviation
            if gold_forward != 0.0:
                should_move_n += 1
                if pred_forward == 0.0:
                    should_move_missed += 1
                predicted_xy = hypothetical_position(pose, pred)
                gold_xy = hypothetical_position(pose, gold)
                should_move_dev += math.dist(predicted_xy, gold_xy)

            # (b) risk-adjacent safety: only frames genuinely near the hazard
            if hazard_xy is not None:
                actual_gap = math.dist((pose[0], pose[1]), hazard_xy)
                if actual_gap <= hazard_clearance + risk_margin:
                    risky_n += 1
                    hx, hy = hypothetical_position(pose, pred)
                    if math.dist((hx, hy), hazard_xy) < hazard_clearance:
                        risky_violations += 1

    if not frames:
        return None
    return {
        "dir": result_dir.name,
        "frames": frames,
        "missing_pose_frames": missing_pose,
        "forward_pred_rate": forward_pred / frames,
        "should_move_frames": should_move_n,
        "nav_dev_when_should_move": (
            should_move_dev / should_move_n if should_move_n else None
        ),
        "miss_rate_when_should_move": (
            should_move_missed / should_move_n if should_move_n else None
        ),
        "risk_adjacent_frames": risky_n,
        "risk_adjacent_violation_rate": (
            risky_violations / risky_n if risky_n else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/data/oem4"))
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/oem4/_industsim_clean"),
        help="corpus root holding <warehouse>/master/<episode>.json",
    )
    parser.add_argument(
        "--risk-margin",
        type=float,
        default=1.0,
        help="metres beyond hazard clearance that still count as risk-adjacent",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    dirs = list(args.dirs)
    if args.all:
        dirs.extend(
            Path(p)
            for p in sorted(
                glob.glob(str(args.root / "_*_eval"))
                + glob.glob(str(args.root / "_*_results"))
            )
            if scores_root(Path(p)) is not None
        )
    if not dirs:
        parser.error("pass result directories, or --all")

    header = (
        f"{'run':<38} {'fwd%':>6} {'navdev↓':>8} {'miss↓':>7} "
        f"{'riskN':>6} {'riskViol↓':>9}"
    )
    print(header)
    print("-" * len(header))
    reports = []
    for result_dir in dirs:
        report = analyse(result_dir, args.raw_root, args.risk_margin)
        if report is None:
            continue
        reports.append(report)
        nav = report["nav_dev_when_should_move"]
        miss = report["miss_rate_when_should_move"]
        viol = report["risk_adjacent_violation_rate"]
        print(
            f"{report['dir']:<38} {report['forward_pred_rate']:>5.2%} "
            f"{('n/a' if nav is None else f'{nav:.4f}'):>8} "
            f"{('n/a' if miss is None else f'{miss:.1%}'):>7} "
            f"{report['risk_adjacent_frames']:>6} "
            f"{('n/a' if viol is None else f'{viol:.4f}'):>9}"
        )

    if args.json:
        args.json.write_text(
            json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {args.json} ({len(reports)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
