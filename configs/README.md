# Fine-tuning config map

Only the nested paths below are executable. The earlier flat prototype files
are preserved verbatim under `../legacy_configs_do_not_use/original_snapshot/`
for provenance, but isolated from execution because their imports, model class,
labels, and SAIE input schema do not match the repository.

- `base.yaml`: shared data/training defaults
- `models/gemma4.yaml`: default recipe; freezes the complete vision/audio
  encoders and their input embedders/projectors
- `methods/full_ft.yaml`: BF16 `full_text_backbone` optimization; it is not
  literal all-parameter FT
- `models/gemma4_all_active.yaml` + `methods/full_all_active.yaml`: trains every
  vision+text-active parameter while freezing the complete unused audio branch
- `methods/lora_r16.yaml`: development LoRA baseline
- `deepspeed/zero3_offload.json`: conservative 1-2 H100 default
- `deepspeed/zero3.json`: opt-in >=4 H100 non-offload configuration
- `splits/local_smoke_dev.json`: development only; never a paper split
- `splits/paper_59_11.example.json`: deliberately invalid placeholder template

Run every command from any directory; `train.py` resolves relative paths against
the cloned repository root.

The launcher defaults to `gemma4.yaml` + `full_ft.yaml`.  To opt into the wider
scope after a smoke test, export:

```bash
export SAIE_MODEL_CONFIG=configs/models/gemma4_all_active.yaml
export SAIE_METHOD_CONFIG=configs/methods/full_all_active.yaml
```
