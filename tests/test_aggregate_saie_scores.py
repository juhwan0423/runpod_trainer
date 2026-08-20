from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aggregate_saie_scores import aggregate  # noqa: E402


class AggregateSaieScoresTests(unittest.TestCase):
    def test_warehouse_qualified_ids_and_pooled_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                ("warehouse_s1", 10, 0.5, 2, 1),
                ("warehouse_s2", 30, 1.0, 8, 2),
            ]
            for warehouse, frames, accuracy, risky, violations in rows:
                score_dir = root / warehouse / "scores"
                score_dir.mkdir(parents=True)
                (score_dir / "score_bench_seed1.json").write_text(
                    json.dumps(
                        {
                            "episode_id": "bench_seed1",
                            "version": 1,
                            "tool_accuracy": accuracy,
                            "nav_deviation_m": 0.25,
                            "safety_violation_rate": 0.1,
                            "safety_violation_rate_v2": violations / risky,
                            "safety_v2_n_risky_frames": risky,
                            "safety_v2_n_violations": violations,
                            "task_completion": 0.5,
                            "task_completion_strict": accuracy,
                            "num_frames": frames,
                            "per_frame": [
                                {
                                    "tool_match": index < round(frames * accuracy),
                                    "nav_deviation_m": 0.25,
                                    "safety_violated": False,
                                }
                                for index in range(frames)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            result = aggregate(root)

        self.assertEqual(result["num_warehouses"], 2)
        self.assertEqual(result["num_episodes"], 2)
        self.assertEqual(result["num_frames"], 40)
        self.assertAlmostEqual(result["frame_weighted_metrics"]["tool_accuracy"], 0.875)
        self.assertEqual(
            result["frame_weighted_metrics_source"],
            "exact_from_per_frame_records",
        )
        self.assertAlmostEqual(
            result["diagnostic_episode_distributions"]["task_completion_strict"]["mean"],
            0.75,
        )
        pooled = result["safety_v2_risk_adjacent_pooled"]
        self.assertEqual(pooled["n_risky_frames"], 10)
        self.assertEqual(pooled["n_violations"], 3)
        self.assertAlmostEqual(pooled["pooled_violation_rate"], 0.3)
        self.assertEqual(
            {row["qualified_episode_id"] for row in result["episodes"]},
            {"warehouse_s1/bench_seed1", "warehouse_s2/bench_seed1"},
        )


if __name__ == "__main__":
    unittest.main()
