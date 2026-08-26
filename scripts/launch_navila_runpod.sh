#!/usr/bin/env bash
# Usage: scripts/launch_navila_runpod.sh smoke|full [seed]
#
# Dedicated NaVILA launch script -- deliberately NOT a merge into
# launch_runpod.sh. That script is actively used by the in-flight Gemma4
# full-FT run; a shared/generalized launcher is real future work (see
# adapter_implement_plan.md section 5) but is out of scope while that run is
# live, so this duplicates its smoke/full structure against NaVILA's own
# venv (scripts/setup_navila_venv.sh) and model config instead of editing
# the Gemma4 path.
set -euo pipefail

MODE=${1:?usage: launch_navila_runpod.sh smoke|full [seed]}
SEED=${2:-0}
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "mode must be smoke or full" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${NAVILA_VENV_DIR:-${PERSIST_ROOT}/venvs/navila-ft}
PYTHON=${NAVILA_PYTHON:-${VENV_DIR}/bin/python}
DATA_ROOT=${SAIE_DATA_ROOT:?export SAIE_DATA_ROOT=/persistent/path/to/data/sft}
IMAGE_ROOT=${SAIE_IMAGE_ROOT:?export SAIE_IMAGE_ROOT=/persistent/path/to/portable/image/root}
OUTPUT_ROOT=${SAIE_OUTPUT_ROOT:-${PERSIST_ROOT}/saie-runs}
ABLATION=${SAIE_ABLATION:-default}
MODEL_CONFIG=${NAVILA_MODEL_CONFIG:-configs/models/navila.yaml}
METHOD_CONFIG=${NAVILA_METHOD_CONFIG:-configs/methods/full_ft.yaml}
REPORT_TO=${SAIE_REPORT_TO:-none}
REQUIREMENTS_FILE=${NAVILA_REQUIREMENTS_FILE:-$ROOT/requirements-navila.txt}
HF_HOME=${HF_HOME:-${PERSIST_ROOT}/cache/huggingface}
export HF_HOME
# See setup_navila_venv.sh -- llava is pip-installed editable inside this
# venv, no separate repo path needs to be on PYTHONPATH.
export PATH="${VENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

[[ -x "$PYTHON" ]] || {
  echo "python not found: $PYTHON; run scripts/setup_navila_venv.sh first" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT/logs" "$HF_HOME"

if [[ "$MODEL_CONFIG" = /* ]]; then
  MODEL_CONFIG_PATH=$MODEL_CONFIG
else
  MODEL_CONFIG_PATH=$ROOT/$MODEL_CONFIG
fi
mapfile -t MODEL_META < <(
  "$PYTHON" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(c["hf_id"]); print(c.get("revision") or "")' \
    "$MODEL_CONFIG_PATH"
)
MODEL_ID=${MODEL_META[0]:?model config has no hf_id}
MODEL_REVISION=${MODEL_META[1]:?model config must pin revision}

AVAILABLE_GPUS=$($PYTHON -c 'import torch; print(torch.cuda.device_count())')
NUM_GPUS=${SAIE_NUM_GPUS:-$AVAILABLE_GPUS}
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || (( NUM_GPUS > AVAILABLE_GPUS )); then
  echo "invalid SAIE_NUM_GPUS=$NUM_GPUS (available=$AVAILABLE_GPUS)" >&2
  exit 2
fi

# ZeRO-3, not ZeRO-2, is what Gemma4 uses -- deliberately NOT the default
# here. Confirmed empirically 2026-08-22: llava's load_pretrained_model
# hardcodes low_cpu_mem_usage=True (llava/model/builder.py, not something
# this adapter can turn off), and transformers refuses that combination
# under ZeRO-3's zero.Init() with a hard error ("DeepSpeed Zero-3 is not
# compatible with low_cpu_mem_usage=True or with passing a device_map").
# ZeRO-2 doesn't use zero.Init (it doesn't partition parameters, only
# gradients/optimizer state), so it doesn't hit this at all.
if [[ -n "${SAIE_DEEPSPEED_CONFIG:-}" ]]; then
  DS_CONFIG=$SAIE_DEEPSPEED_CONFIG
elif (( NUM_GPUS >= 4 )) && [[ "${SAIE_ALLOW_NO_OFFLOAD:-0}" == "1" ]]; then
  DS_CONFIG=configs/deepspeed/zero2.json
else
  DS_CONFIG=configs/deepspeed/zero2_offload.json
fi

# NaVILA's checkpoint access doesn't go through --check-model-access /
# --check-processor-contract (those assume an AutoProcessor-based adapter);
# navila_unified.load() does its own revision-pinned snapshot_download and
# tokenizer sanity checks instead (see scripts/verify_navila_exact_prefix.py
# for the standalone pre-flight version of that check).
PREFLIGHT=(
  "$PYTHON" "$ROOT/scripts/preflight.py"
  --data-root "$DATA_ROOT"
  --image-root "$IMAGE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --ablation "$ABLATION"
  --require-gpu
  --requirements-file "$REQUIREMENTS_FILE"
  --report "$OUTPUT_ROOT/preflight_navila_${MODE}.json"
)

TRAIN_BASE=(
  "$ROOT/train.py"
  --config "$MODEL_CONFIG"
  --method "$METHOD_CONFIG"
  --seed "$SEED"
  --ablation "$ABLATION"
  --data_root "$DATA_ROOT"
  --image_root "$IMAGE_ROOT"
  --output_root "$OUTPUT_ROOT"
  --deepspeed "$DS_CONFIG"
  --report_to "$REPORT_TO"
)

if [[ "$MODE" == "smoke" ]]; then
  PREFLIGHT+=(--min-free-gb "${SAIE_MIN_FREE_GB:-100}" --min-ram-gb "${SAIE_MIN_RAM_GB:-64}" --no-check-all-images)
else
  PREFLIGHT+=(--min-free-gb "${SAIE_MIN_FREE_GB:-200}" --min-ram-gb "${SAIE_MIN_RAM_GB:-64}")
fi

"${PREFLIGHT[@]}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTPUT_ROOT/logs/${MODE}_navila_seed${SEED}_${STAMP}.log"

dry_run() {
  "$PYTHON" "$@" --dry-run
}

run_train() {
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_GPUS" "$@" 2>&1 | tee -a "$LOG"
}

if [[ "$MODE" == "full" ]]; then
  TRAIN=("${TRAIN_BASE[@]}" --resume "${SAIE_RESUME:-auto}")
  dry_run "${TRAIN[@]}"
  echo "mode=$MODE model=$MODEL_ID@$MODEL_REVISION gpus=$NUM_GPUS deepspeed=$DS_CONFIG log=$LOG"
  run_train "${TRAIN[@]}"
else
  STEPS=${SAIE_SMOKE_STEPS:-2}
  if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAIE_SMOKE_STEPS must be a positive integer" >&2
    exit 2
  fi
  SMOKE_DIR=${SAIE_SMOKE_OUTPUT_DIR:-$OUTPUT_ROOT/smoke_navila_seed${SEED}_${STAMP}}
  TRAIN=(
    "${TRAIN_BASE[@]}"
    --max_steps "$STEPS"
    --save_steps "$STEPS"
    --eval_steps "$STEPS"
    --save_total_limit 1
    --max_eval_samples "${SAIE_SMOKE_EVAL_SAMPLES:-4}"
    --resume none
    --output_dir "$SMOKE_DIR"
  )
  dry_run "${TRAIN[@]}"
  echo "mode=$MODE steps=$STEPS model=$MODEL_ID@$MODEL_REVISION gpus=$NUM_GPUS deepspeed=$DS_CONFIG output=$SMOKE_DIR log=$LOG"
  run_train "${TRAIN[@]}"
fi
