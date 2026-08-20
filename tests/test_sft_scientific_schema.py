from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_sft_dataset import scientific_structure_errors  # noqa: E402
from data.schema import ACTION_LABELS  # noqa: E402


class ScientificSchemaTests(unittest.TestCase):
    def test_valid_extended_row_has_no_structure_errors(self) -> None:
        row = {
            "action_group": "alert",
            "hazard_present": True,
            "is_distractor": False,
            "risk_adjacent": True,
            "in_task_window": False,
            "stop_timing": "not_applicable",
            "admissible_action_labels": ["[alert] stop", "[navigate] stop"],
            "safe_action_mask": [True] * len(ACTION_LABELS),
            "candidate_action_safety": None,
            "view_valid": True,
            "attention_target": {
                "bbox_xyxy": [10, 20, 100, 200],
                "world_position_m": [1.0, 2.0, 0.0],
                "bearing_deg": 12.5,
            },
            "target_bbox": None,
            "target_world_position": None,
            "target_bearing_deg": None,
        }
        self.assertFalse(scientific_structure_errors([row]))

    def test_placeholder_values_are_rejected(self) -> None:
        row = {
            "action_group": "alert",
            "hazard_present": "true",
            "is_distractor": False,
            "risk_adjacent": False,
            "in_task_window": False,
            "stop_timing": "on_time",
            "admissible_action_labels": ["not-a-label"],
            "safe_action_mask": [True, False],
            "candidate_action_safety": None,
            "view_valid": "yes",
            "attention_target": {"instance_id": "hazard_0"},
            "target_bbox": [10, 20, 5, 30],
            "target_world_position": [1.0],
            "target_bearing_deg": float("nan"),
        }
        errors = scientific_structure_errors([row])
        for key in (
            "hazard_present_not_boolean",
            "invalid_stop_timing",
            "invalid_admissible_action_labels",
            "invalid_safe_action_mask",
            "view_valid_not_boolean",
            "invalid_attention_target",
            "invalid_target_bbox",
            "invalid_target_world_bearing",
            "invalid_alert_localization",
        ):
            self.assertEqual(errors[key], 1, key)

    def test_safe_mask_and_candidate_map_must_agree(self) -> None:
        row = {
            "action_group": "navigate",
            "safe_action_mask": [True] * len(ACTION_LABELS),
            "candidate_action_safety": {
                label: {"safe": index != 0}
                for index, label in enumerate(ACTION_LABELS)
            },
        }
        errors = scientific_structure_errors([row])
        self.assertEqual(errors["inconsistent_safe_action_supervision"], 1)


if __name__ == "__main__":
    unittest.main()
