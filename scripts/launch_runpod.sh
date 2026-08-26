#!/usr/bin/env bash
# Usage: scripts/launch_runpod.sh smoke|smoke-resume|full [seed]
set -euo pipefail

MODE=${1:?usage: launch_runpod.sh smoke|smoke-resume|full [seed]}
SEED=${2:-0}
if [[ "$MODE" != "smoke" && "$MODE" != "smoke-resume" && "$MODE" != "full" ]]; then
  echo "mode must be smoke, smoke-resume, or full" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${SAIE_VENV_DIR:-${PERSIST_ROOT}/venvs/saie-ft}
PYTHON=${SAIE_PYTHON:-${VENV_DIR}/bin/python}
DATA_ROOT=${SAIE_DATA_ROOT:?export SAIE_DATA_ROOT=/persistent/path/to/data/sft}
IMAGE_ROOT=${SAIE_IMAGE_ROOT:?export SAIE_IMAGE_ROOT=/persistent/path/to/portable/image/root}
OUTPUT_ROOT=${SAIE_OUTPUT_ROOT:-${PERSIST_ROOT}/saie-runs}
ABLATION=${SAIE_ABLATION:-default}
MODEL_CONFIG=${SAIE_MODEL_CONFIG:-configs/models/gemma4.yaml}
METHOD_CONFIG=${SAIE_METHOD_CONFIG:-configs/methods/full_ft.yaml}
# The default pair is honestly tagged full_text_backbone because Gemma's
# complete vision/audio input stacks are frozen. Set both variables to the
# *_all_active recipes to optimize the vision encoder/projector too; the unused
# audio path remains frozen for DDP.
REPORT_TO=${SAIE_REPORT_TO:-none}
HF_HOME=${HF_HOME:-${PERSIST_ROOT}/cache/huggingface}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${PERSIST_ROOT}/cache/torch_extensions}
export HF_HOME TORCH_EXTENSIONS_DIR
# $PYTHON is invoked by full path below, never via `source .../activate`, so
# PATH would otherwise lack $VENV_DIR/bin. DeepSpeed's CPUAdam JIT build
# shells out to `ninja` by name (not by full path) inside every torchrun
# worker process, so without this the offload optimizer fails with "Unable
# to JIT load the cpu_adam op due to ninja not being installed" even though
# pip installed ninja straight into this venv.
export PATH="${VENV_DIR}/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

[[ -x "$PYTHON" ]] || {
  echo "python not found: $PYTHON; run scripts/setup_runpod.sh" >&2
  exit 2
}
mkdir -p "$OUTPUT_ROOT/logs" "$HF_HOME" "$TORCH_EXTENSIONS_DIR"

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

# One or two 80 GB GPUs remain optimizer-offload jobs. Non-offload ZeRO-3 is
# selected automatically only when >=4 GPUs *and* the user explicitly opts in.
if [[ -n "${SAIE_DEEPSPEED_CONFIG:-}" ]]; then
  DS_CONFIG=$SAIE_DEEPSPEED_CONFIG
elif (( NUM_GPUS >= 4 )) && [[ "${SAIE_ALLOW_NO_OFFLOAD:-0}" == "1" ]]; then
  DS_CONFIG=configs/deepspeed/zero3.json
else
  DS_CONFIG=configs/deepspeed/zero3_offload.json
fi

PREFLIGHT=(
  "$PYTHON" "$ROOT/scripts/preflight.py"
  --data-root "$DATA_ROOT"
  --image-root "$IMAGE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --ablation "$ABLATION"
  --require-gpu
  --require-h100
  --check-model-access
  --check-processor-contract
  --model "$MODEL_ID"
  --model-revision "$MODEL_REVISION"
  --report "$OUTPUT_ROOT/preflight_${MODE}.json"
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

if [[ "$MODE" == "full" ]]; then
  PREFLIGHT+=(
    --require-paper-ready
    --min-free-gb "${SAIE_MIN_FREE_GB:-700}"
    --min-ram-gb "${SAIE_MIN_RAM_GB:-200}"
  )
else
  # Even a two-step full-FT smoke test materializes CPU Adam state and one
  # resumable checkpoint. These defaults assume a >=500 GB pod volume and
  # >=200 GiB host RAM; 1 TB / 256 GiB remains recommended.
  PREFLIGHT+=(
    --min-free-gb "${SAIE_MIN_FREE_GB:-350}"
    --min-ram-gb "${SAIE_MIN_RAM_GB:-200}"
    --no-check-all-images
  )
fi
if [[ "$MODE" == "full" && "${SAIE_REQUIRE_ICRA_READY:-0}" == "1" ]]; then
  PREFLIGHT+=(--require-icra-ready)
fi

"${PREFLIGHT[@]}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$OUTPUT_ROOT/logs/${MODE}_gemma4_seed${SEED}_${STAMP}.log"

dry_run() {
  "$PYTHON" "$@" --dry-run
}

run_train() {
  # torchrun is used even for one GPU so ZeRO-3 sees an initialized distributed
  # environment and never falls back to MPI/mpi4py discovery.
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_GPUS" "$@" 2>&1 | tee -a "$LOG"
}

if [[ "$MODE" == "full" ]]; then
  TRAIN=("${TRAIN_BASE[@]}" --require-paper-ready --resume "${SAIE_RESUME:-auto}")
  if [[ "${SAIE_REQUIRE_ICRA_READY:-0}" == "1" ]]; then
    TRAIN+=(--require-icra-ready)
  fi
  dry_run "${TRAIN[@]}"
  echo "mode=$MODE gpus=$NUM_GPUS deepspeed=$DS_CONFIG log=$LOG"
  run_train "${TRAIN[@]}"
elif [[ "$MODE" == "smoke" ]]; then
  STEPS=${SAIE_SMOKE_STEPS:-2}
  if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAIE_SMOKE_STEPS must be a positive integer" >&2
    exit 2
  fi
  SMOKE_DIR=${SAIE_SMOKE_OUTPUT_DIR:-$OUTPUT_ROOT/smoke_gemma4_seed${SEED}_${STAMP}}
  TRAIN=(
    "${TRAIN_BASE[@]}"
    --max_steps "$STEPS"
    --save_steps "$STEPS"
    --eval_steps "$STEPS"
    --save_total_limit 1
    --max_eval_samples "${SAIE_SMOKE_EVAL_SAMPLES:-4}"
    --resume "${SAIE_RESUME:-none}"
    --output_dir "$SMOKE_DIR"
  )
  dry_run "${TRAIN[@]}"
  echo "mode=$MODE steps=$STEPS gpus=$NUM_GPUS deepspeed=$DS_CONFIG output=$SMOKE_DIR log=$LOG"
  run_train "${TRAIN[@]}"
else
  FIRST_STEPS=${SAIE_SMOKE_FIRST_STEPS:-1}
  RESUME_STEPS=${SAIE_SMOKE_RESUME_STEPS:-2}
  if ! [[ "$FIRST_STEPS" =~ ^[1-9][0-9]*$ && "$RESUME_STEPS" =~ ^[1-9][0-9]*$ ]] \
    || (( RESUME_STEPS <= FIRST_STEPS )); then
    echo "resume smoke requires positive SAIE_SMOKE_FIRST_STEPS < SAIE_SMOKE_RESUME_STEPS" >&2
    exit 2
  fi
  SMOKE_DIR=${SAIE_SMOKE_OUTPUT_DIR:-$OUTPUT_ROOT/smoke_resume_gemma4_seed${SEED}_${STAMP}}
  FIRST=(
    "${TRAIN_BASE[@]}"
    --max_steps "$FIRST_STEPS"
    --save_steps "$FIRST_STEPS"
    --eval_steps "$FIRST_STEPS"
    --save_total_limit 1
    --max_eval_samples "${SAIE_SMOKE_EVAL_SAMPLES:-4}"
    --resume none
    --output_dir "$SMOKE_DIR"
  )
  SECOND=(
    "${TRAIN_BASE[@]}"
    --max_steps "$RESUME_STEPS"
    --save_steps "$RESUME_STEPS"
    --eval_steps "$RESUME_STEPS"
    --save_total_limit 1
    --max_eval_samples "${SAIE_SMOKE_EVAL_SAMPLES:-4}"
    --resume auto
    --output_dir "$SMOKE_DIR"
  )
  dry_run "${FIRST[@]}"
  dry_run "${SECOND[@]}"
  echo "mode=$MODE steps=$FIRST_STEPS->$RESUME_STEPS gpus=$NUM_GPUS deepspeed=$DS_CONFIG output=$SMOKE_DIR log=$LOG"
  run_train "${FIRST[@]}"
  [[ -d "$SMOKE_DIR/checkpoint-$FIRST_STEPS" ]] || {
    echo "first smoke phase did not create checkpoint-$FIRST_STEPS" >&2
    exit 2
  }
  run_train "${SECOND[@]}"
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="complete", p' \
    "$SMOKE_DIR/run_manifest.json"
  echo "resume smoke PASS: $SMOKE_DIR"
fi
