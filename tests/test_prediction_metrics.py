from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.schema import ACTION_LABELS  # noqa: E402
from evaluate_predictions import (  # noqa: E402
    INVALID,
    compute_category_metrics,
    normalise_prediction,
    prediction_value,
)
from predict_labels import canonical_from_raw  # noqa: E402


class StrictPredictionTests(unittest.TestCase):
    def test_extra_text_is_diagnostic_not_strict(self) -> None:
        label = ACTION_LABELS[0]
        normalized, status = canonical_from_raw(f"I choose {label} now")
        self.assertEqual(normalized, label)
        self.assertEqual(status, "extracted")

        row = {
            "pred": None,
            "normalized_action_label": normalized,
            "native_action": f"I choose {label} now",
            "parse_status": status,
        }
        self.assertEqual(normalise_prediction(prediction_value(row, False), False), INVALID)
        self.assertEqual(normalise_prediction(prediction_value(row, True), True), label)

    def test_exact_label_is_strict(self) -> None:
        label = ACTION_LABELS[-1]
        normalized, status = canonical_from_raw(label)
        self.assertEqual((normalized, status), (label, "exact"))
        row = {"pred": normalized, "parse_status": status}
        self.assertEqual(normalise_prediction(prediction_value(row, False), False), label)

    def test_category_metrics_aggregate_navigation_labels(self) -> None:
        gold = [ACTION_LABELS[0], ACTION_LABELS[1], ACTION_LABELS[-1]]
        pred = [ACTION_LABELS[2], ACTION_LABELS[3], ACTION_LABELS[-1]]
        metrics = compute_category_metrics(gold, pred)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["per_group"]["navigate"]["support"], 2)
        self.assertEqual(metrics["per_group"]["alert"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
