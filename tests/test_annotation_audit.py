from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation_audit import (  # noqa: E402
    TASK_LABELS,
    audit_dataset,
    causal_history,
    derive_hazard_metadata,
    nearest_saved_frame,
)


class AnnotationAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_episode(
        self,
        annotations: list[dict],
        *,
        invalid: bool = False,
        include_images: bool = True,
    ) -> Path:
        warehouse = self.root / "warehouse_s1"
        master_dir = warehouse / "master"
        annotation_dir = warehouse / "annotations"
        frame_dir = warehouse / "frames" / "ep1"
        image_dir = frame_dir / "spot_cam"
        for directory in (master_dir, annotation_dir, frame_dir, image_dir):
            directory.mkdir(parents=True, exist_ok=True)

        master = {
            "episode_id": "ep1",
            "map_id": "warehouse_s1",
            "split": "train",
            "hazard": {
                "type": "FALL_HUMAN",
                "pose_world": [2.0, 0.0, 0.0],
                "clearance": 2.35,
            },
            "stations": [],
            "timeline": [{"timestamp": 0.0}],
        }
        (master_dir / "ep1.json").write_text(json.dumps(master), encoding="utf-8")
        (annotation_dir / "ep1.json").write_text(
            json.dumps(annotations), encoding="utf-8"
        )
        saved = [0, 15, 30, 45, 60, 75]
        if include_images:
            for frame in saved:
                (image_dir / f"rgb_{frame:04d}.png").write_bytes(b"test")
        events = {
            "invalid": invalid,
            "invalid_reason": "robot_fallen" if invalid else None,
            "spot_frames": len(saved) if include_images else 0,
            "spot_events": [
                {
                    "event": "robot_pose",
                    "render_frame": frame,
                    "x": 0.0,
                    "y": 0.0,
                    "yaw_rad": 0.0,
                }
                for frame in saved
            ],
        }
        (frame_dir / "sim_events.json").write_text(
            json.dumps(events), encoding="utf-8"
        )
        return warehouse

    @staticmethod
    def _row(frame: int, label: str, frames: list[int] | None = None) -> dict:
        refs = frames if frames is not None else [frame]
        return {
            "video_id": f"ep1_t{frame:04d}",
            "q": "Inspect safely.",
            "a": label,
            "motion": {
                "vx_m_per_s": 0.0,
                "vy_m_per_s": 0.0,
                "omega_rad_per_s": 0.0,
                "dx_m": 0.0,
                "dyaw_rad": 0.0,
                "duration_s": 1.0,
            },
            "frames": [f"ep1/spot_cam/rgb_{value:04d}.png" for value in refs],
            "is_distractor": False,
            "in_task_window": None,
        }

    def test_snap_and_causal_history_helpers(self) -> None:
        saved = [0, 15, 30, 45]
        self.assertEqual(nearest_saved_frame(7, saved), 0)
        self.assertEqual(nearest_saved_frame(8, saved), 15)
        self.assertEqual(causal_history(30, saved, 3), [0, 15, 30])
        self.assertTrue(all(frame <= 30 for frame in causal_history(30, saved, 16)))

    def test_audit_detects_bad_source_and_prepares_existing_images(self) -> None:
        rows = [self._row(14, "[navigate] move forward 25 cm", [0, 30, 14])]
        rows.extend(
            self._row(frame, label)
            for frame, label in zip((30, 45, 60, 75), TASK_LABELS)
        )
        self._write_episode(rows)
        output = self.root / "prepared" / "all.jsonl"
        report = audit_dataset(
            self.root,
            prepare_jsonl=output,
            num_frames=3,
            expected_warehouses=0,
            expected_episodes=0,
            expected_decision_hz=0,
        )

        issue_counts = {item["code"]: item["count"] for item in report["issues"]}
        self.assertGreaterEqual(issue_counts["missing_source_frame_reference"], 1)
        self.assertGreaterEqual(issue_counts["future_frame_leakage"], 1)
        self.assertEqual(report["counts"]["prepared_rows"], 5)
        prepared = [json.loads(line) for line in output.read_text().splitlines()]
        first = prepared[0]
        self.assertEqual(first["source_frame_index"], 14)
        self.assertEqual(first["frame_index"], 15)
        self.assertEqual(first["provenance"]["selected_frame_delta"], 1)
        self.assertTrue((self.root / first["image"]).is_file())
        history_indices = [
            int(Path(path).stem.split("_")[-1]) for path in first["image_history"]
        ]
        self.assertEqual(history_indices[-1], 15)
        self.assertTrue(all(frame <= 15 for frame in history_indices))
        self.assertEqual(first["episode_id"], "warehouse_s1/ep1")

    def test_conflicting_duplicate_supervision_is_excluded(self) -> None:
        rows = [
            self._row(0, "[navigate] stop"),
            self._row(0, "[gauge] stop"),
            self._row(30, "[gauge] stop"),
            self._row(45, "[count] stop"),
            self._row(60, "[state] stop"),
            self._row(75, "[alert] stop"),
        ]
        self._write_episode(rows)
        output = self.root / "all.jsonl"
        report = audit_dataset(
            self.root,
            prepare_jsonl=output,
            expected_warehouses=0,
            expected_episodes=0,
            expected_decision_hz=0,
        )
        self.assertEqual(report["counts"]["excluded_conflicting_prepare_rows"], 2)
        self.assertEqual(report["counts"]["prepared_rows"], 4)
        codes = {item["code"] for item in report["issues"]}
        self.assertIn("conflicting_duplicate_label", codes)

    def test_publication_block_keeps_existing_output_untouched(self) -> None:
        rows = [self._row(frame, label) for frame, label in zip((0, 30, 45, 60), TASK_LABELS)]
        self._write_episode(rows)
        output = self.root / "all.jsonl"
        output.write_text("sentinel\n", encoding="utf-8")
        report = audit_dataset(
            self.root,
            prepare_jsonl=output,
            overwrite=True,
            expected_warehouses=70,
            expected_episodes=1311,
            expected_decision_hz=2,
            block_on_publication_gaps=True,
        )
        self.assertEqual(output.read_text(encoding="utf-8"), "sentinel\n")
        self.assertEqual(report["counts"]["prepared_write_blocked"], 1)
        self.assertIsNone(report["prepared_output"])

    def test_hazard_distance_and_forward_risk_are_derived(self) -> None:
        master = {
            "hazard": {
                "pose_world": [2.0, 0.0, 0.0],
                "clearance": 1.5,
            }
        }
        distance, risky = derive_hazard_metadata(master, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(distance or 0.0, 2.0)
        self.assertTrue(risky)
        _, risky_away = derive_hazard_metadata(master, (0.0, 0.0, 3.1415926535))
        self.assertFalse(risky_away)

    def test_external_split_manifest_is_authoritative(self) -> None:
        rows = [self._row(frame, label) for frame, label in zip((0, 30, 45, 60), TASK_LABELS)]
        warehouse = self._write_episode(rows)
        master_path = warehouse / "master" / "ep1.json"
        master = json.loads(master_path.read_text(encoding="utf-8"))
        master.pop("split")
        master_path.write_text(json.dumps(master), encoding="utf-8")
        manifest_path = self.root / "split.json"
        manifest_path.write_text(
            json.dumps({"train": ["warehouse_s1"], "val": [], "test": []}),
            encoding="utf-8",
        )

        report = audit_dataset(
            self.root,
            split_manifest=manifest_path,
            expected_warehouses=0,
            expected_episodes=0,
            expected_decision_hz=0,
        )

        codes = {item["code"] for item in report["issues"]}
        self.assertNotIn("split_manifest_absent", codes)
        self.assertEqual(report["split_warehouses"], {"train": ["warehouse_s1"]})


if __name__ == "__main__":
    unittest.main()
