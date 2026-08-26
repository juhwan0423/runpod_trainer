from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class Gemma4TrainingScopeTests(unittest.TestCase):
    def _model_config(self, filename: str) -> dict:
        return yaml.safe_load((ROOT / "configs/models" / filename).read_text(encoding="utf-8"))

    def test_text_backbone_freezes_loaded_modality_paths(self) -> None:
        config = self._model_config("gemma4.yaml")
        self.assertEqual(
            set(config["freeze_prefixes"]),
            {
                "model.embed_vision",
                "model.embed_audio",
            },
        )
        self.assertEqual(
            set(config["full_text_backbone_freeze_prefixes"]),
            set(config["freeze_prefixes"]),
        )
        counts = config["expected_parameter_counts"]
        self.assertEqual(
            counts["trainable"] + sum(counts["frozen_by_prefix"].values()),
            counts["total"],
        )

    def test_all_active_freezes_only_loaded_unused_audio_path(self) -> None:
        config = self._model_config("gemma4_all_active.yaml")
        self.assertEqual(
            set(config["freeze_prefixes"]),
            {"model.embed_audio"},
        )
        counts = config["expected_parameter_counts"]
        self.assertEqual(
            counts["trainable"] + sum(counts["frozen_by_prefix"].values()),
            counts["total"],
        )

    def test_lora_regex_cannot_match_vision_or_audio_towers(self) -> None:
        pattern = self._model_config("gemma4.yaml")["target_modules"]
        self.assertTrue(
            re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
        )
        self.assertTrue(
            re.fullmatch(pattern, "model.language_model.layers.0.mlp.down_proj")
        )
        self.assertIsNone(
            re.fullmatch(pattern, "model.vision_tower.encoder.layers.0.self_attn.q_proj")
        )
        self.assertIsNone(
            re.fullmatch(pattern, "model.audio_tower.layers.0.attention.q_proj")
        )


if __name__ == "__main__":
    unittest.main()
