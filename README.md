# SAIE RunPod Trainer

A fine-tuning and evaluation pipeline for the IndustSim-Quad (SAIE) industrial patrol benchmark. It fine-tunes four VLM/VLA models on the benchmark's 14-label action task, generates predictions and scores them.

| Model | Full fine-tuning | LoRA | Setup |
|---|:---:|:---:|---|
| Gemma 4 12B | ✓ | ✓ | DeepSpeed ZeRO-3, BF16 |
| NaVILA (LLaMA3 8B) | ✓ | ✓ | DeepSpeed ZeRO-2, 4× H100 |
| InternVLA-N1 | ✓ | ✓ | Built on InternNav |
| InternVLA-M1 | ✓ | ✓ | |

## Purpose

Existing VLAs scored close to zero on IndustSim-Quad without training. This study asks whether the cause lies in the data or in the models.

- **Does the benchmark data carry a learning signal?** All four models get the same standard recipe (full fine-tuning and LoRA), with no per-model tuning.
- **Does the kind of pretraining change the result?** The models are a legged-navigation VLA (NaVILA), a navigation VLA (InternVLA-N1), a general VLA (InternVLA-M1) and a VLM with no robot pretraining (Gemma 4).
- **Do the metrics measure the ability they claim to?** Predictions of the trained models are broken down by category and compared with the existing metrics.

## Results

19 runs are compared on the same 11 test warehouses and 210 evaluation episodes. `task-stop` is the accuracy on the four inspection tasks (alert, gauge, count, state); a model that always answers "navigate" scores 0.

| Model | Method | Training data | Task completion | task-stop |
|---|---|---|---:|---:|
| InternVLA-M1 | Full FT | balanced | **0.748** | **0.634** |
| InternVLA-N1 | LoRA | balanced | 0.733 | 0.597 |
| Gemma 4 | LoRA | redesigned (seed 0) | 0.687 | 0.515 |
| NaVILA | Full FT | balanced | 0.647 | 0.488 |

- Full fine-tuning collapsed to near-constant outputs on Gemma 4 and InternVLA-N1 and stayed stable on NaVILA and InternVLA-M1. Which of LoRA and full fine-tuning is better depends on the model.
- On navigation frames, a metric that matches only the `[navigate]` tag gives 0.84–0.97, while exact-match accuracy on the same frames is 0.09–0.34.
- `scripts/build_table2.py` and `scripts/category_breakdown.py` rebuild the full table of 19 runs and the per-category breakdown.

## Repository

```text
.
├── train.py                       # multimodal supervised fine-tuning
├── predict_labels.py              # 14-label generation
├── predict_labels_navila.py       # NaVILA prediction path
├── evaluate_predictions.py        # classification and balance diagnostics
├── aggregate_saie_scores.py       # per-warehouse judge aggregation
├── annotation_audit.py            # raw annotation audit
├── data/                          # labels, prompt, SFT dataset builder, loader
├── models/registry.py             # per-model adapters, freeze and LoRA rules
├── configs/                       # models, methods, DeepSpeed, splits
├── scripts/                       # RunPod setup and launch, sharded evaluation, result registry, analysis
└── tests/
```

## Requirements

- Linux, Python 3.12, CUDA GPU (full training: H100 80 GB or larger)
- A Hugging Face token with access to the gated models
- The raw SAIE warehouse corpus

Pinned dependencies are in `requirements-*.txt`. The single-H100 recipe offloads the optimizer to CPU, so it needs ample host RAM and disk.

## Data layout

```text
<SAIE_RAW_ROOT>/
└── warehouse_s<seed>/
    ├── annotations/<episode>.json
    ├── master/<episode>.json
    └── frames/<episode>/
        ├── sim_events.json
        └── spot_cam/rgb_XXXX.png
```

Paper runs use a warehouse-level split manifest (59 training and 11 test warehouses). `configs/splits/paper_59_11.example.json` shows the format.

## 1. Set up RunPod

```bash
git clone https://github.com/juhwan0423/runpod_trainer.git
cd runpod_trainer
bash scripts/setup_runpod.sh
source /workspace/venvs/saie-ft/bin/activate
export HF_TOKEN='...'
```

## 2. Audit the annotations

```bash
export SAIE_RAW_ROOT=/workspace/raw

python annotation_audit.py \
  --dataset-root "$SAIE_RAW_ROOT" \
  --split-manifest /workspace/manifests/paper_59_11.json \
  --report-json /workspace/data/audit.json \
  --report-md /workspace/data/audit.md \
  --publication-strict
```

## 3. Build the SFT dataset

```bash
export SAIE_SFT_ROOT=/workspace/data/sft

python data/build_sft_dataset.py \
  --saie-output "$SAIE_RAW_ROOT" \
  --image-root "$SAIE_RAW_ROOT" \
  --out "$SAIE_SFT_ROOT" \
  --split-manifest /workspace/manifests/paper_59_11.json \
  --max-history-frames 1 \
  --strict-frame-paths \
  --duplicate-policy error
```

This writes `<out>/<ablation>/{train,val,test}.jsonl` and a dataset manifest. Each example is the current `spot_cam` image with the mission text, and the target is one canonical label.

## 4. Preflight

```bash
export SAIE_DATA_ROOT="$SAIE_SFT_ROOT"
export SAIE_IMAGE_ROOT="$SAIE_RAW_ROOT"
export SAIE_OUTPUT_ROOT=/workspace/saie-runs

python scripts/preflight.py \
  --data-root "$SAIE_DATA_ROOT" \
  --image-root "$SAIE_IMAGE_ROOT" \
  --output-root "$SAIE_OUTPUT_ROOT" \
  --ablation default \
  --require-gpu \
  --require-h100 \
  --check-model-access \
  --check-processor-contract
```

## 5. Train

```bash
bash scripts/launch_runpod.sh smoke 0
bash scripts/launch_runpod.sh smoke-resume 0
bash scripts/launch_runpod.sh full 0
```

The default is Gemma 4 with `configs/methods/full_ft.yaml`, which trains the whole text backbone and keeps the vision tower frozen. Select another model or method with environment variables:

```bash
export SAIE_MODEL_CONFIG=configs/models/gemma4_all_active.yaml
export SAIE_METHOD_CONFIG=configs/methods/full_all_active.yaml   # vision + text
export SAIE_METHOD_CONFIG=configs/methods/lora_r16.yaml          # LoRA
```

NaVILA has its own launcher, `scripts/launch_navila_runpod.sh`, and the `scripts/setup_*_venv.sh` scripts build the environments for NaVILA and InternVLA-N1.

## 6. Predict and evaluate

```bash
python predict_labels.py \
  --config configs/models/gemma4.yaml \
  --checkpoint /workspace/saie-runs/<run>/final \
  --jsonl "$SAIE_SFT_ROOT/default/test.jsonl" \
  --image-root "$SAIE_IMAGE_ROOT" \
  --out /workspace/saie-runs/predictions/test.jsonl \
  --vlm-output-dir /workspace/saie-runs/vlm_output

python evaluate_predictions.py \
  --predictions /workspace/saie-runs/predictions/test.jsonl \
  --reference-jsonl "$SAIE_SFT_ROOT/default/test.jsonl" \
  --train-jsonl "$SAIE_SFT_ROOT/default/train.jsonl" \
  --out /workspace/saie-runs/predictions/metrics.json
```

`scripts/evaluate_saie_judge.sh` runs the benchmark's four-metric judge per warehouse and aggregates the scores.

## 7. Local checks

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

## Documents

- [`ANNOTATION_AUDIT.md`](./ANNOTATION_AUDIT.md): what the annotation audit checks
- [`configs/README.md`](./configs/README.md): map of the run configurations
- [`configs/data_format_decision.md`](./configs/data_format_decision.md): data, prompt and split decisions
- [`가이드.md`](./가이드.md): full operations guide (Korean)
