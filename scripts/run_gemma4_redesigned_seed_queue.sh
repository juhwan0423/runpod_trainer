#!/usr/bin/env bash
set -euo pipefail

# Queue seed-1 prediction, seed-2 training, and seed-2 prediction after an
# already-running seed-1 torchrun launcher. Authentication is inherited from
# the caller; never place a token in this file.
SEED1_LAUNCHER_PID=${1:?usage: $0 SEED1_LAUNCHER_PID}

ROOT=${SAIE_CODE_ROOT:-/workspace/code/runpod_trainer}
VENV=${SAIE_VENV_DIR:-/workspace/venvs/saie-ft}
DATA_ROOT=${SAIE_DATA_ROOT:-/workspace/saie-data/sft}
IMAGE_ROOT=${SAIE_IMAGE_ROOT:-/workspace/saie-data/raw/benchmark_v1_3}
RUN_ROOT=${SAIE_RUN_ROOT:-/workspace/saie-runs}

cd "$ROOT"
export PATH="$VENV/bin:$PATH"

wait_for_external_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
}

require_final() {
  local final_dir=$1
  [[ -d "$final_dir" ]] || {
    echo "FATAL: expected completed checkpoint missing: $final_dir" >&2
    exit 1
  }
}

predict_seed() {
  local seed=$1
  local checkpoint="$RUN_ROOT/gemma4-12b-it__lora__default__seed${seed}/final"
  local prediction_root="$RUN_ROOT/predictions_gemma4_lora_redesigned_seed${seed}"
  require_final "$checkpoint"
  python predict_labels.py \
    --config configs/models/gemma4.yaml \
    --checkpoint "$checkpoint" \
    --jsonl "$DATA_ROOT/default/test.jsonl" \
    --image-root "$IMAGE_ROOT" \
    --out "$prediction_root/test_predictions.jsonl" \
    --vlm-output-dir "$prediction_root/vlm_output" \
    --num-frames 1 --max-new-tokens 32
}

wait_for_external_pid "$SEED1_LAUNCHER_PID"
predict_seed 1 > /workspace/predict_gemma4_lora_redesigned_seed1.log 2>&1

python -m torch.distributed.run --standalone --nproc_per_node 4 \
  train.py --config configs/models/gemma4.yaml \
  --method configs/methods/lora_r16.yaml \
  --seed 2 --ablation default \
  --data_root "$DATA_ROOT" --image_root "$IMAGE_ROOT" \
  --output_root "$RUN_ROOT" --deepspeed none \
  --report_to none --resume none \
  > /workspace/gemma4_lora_redesigned_seed2_train.log 2>&1

predict_seed 2 > /workspace/predict_gemma4_lora_redesigned_seed2.log 2>&1
echo "seed queue complete"
