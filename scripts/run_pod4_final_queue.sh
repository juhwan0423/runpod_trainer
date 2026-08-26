#!/usr/bin/env bash
# Finish the Pod 4 B200 comparison without allowing one failed stage to
# contaminate the next one.
#
# Usage:
#   scripts/run_pod4_final_queue.sh [already-running-navila-launcher-pid]
#
# The optional PID is only monitored; an unrelated/reused PID is rejected.
# Since a shell cannot recover the exit status of a process it did not start,
# the NaVILA manifest and final adapter are the authoritative completion gate.
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${SAIE_DATA_ROOT:-/workspace/saie/saie-data/sft}
IMAGE_ROOT=${SAIE_IMAGE_ROOT:-/workspace/saie/saie-data/raw/benchmark_v1_3}
RUN_ROOT=${SAIE_OUTPUT_ROOT:-/workspace/saie/saie-runs}
HF_CACHE=${HF_HOME:-/workspace/saie/cache/huggingface}
NAVILA_PY=${NAVILA_PYTHON:-/workspace/saie/venvs/navila-ft/bin/python}
INTERN_PY=${INTERNVLA_PYTHON:-/workspace/saie/venvs/saie-internvla-n1/bin/python}
CURRENT_NAVILA_PID=${1:-}

TEST_JSONL=$DATA_ROOT/default/test.jsonl
EXPECTED_TEST_SHA=92f5b9589efbf4c4c020a22eb3fd5041088c4ea623f1ef97456ceee962b533cc
EXPECTED_TEST_ROWS=12544
EXPECTED_EPISODES=210
EXPECTED_WAREHOUSES=11
NUM_GPUS=4

NAVILA_RUN=$RUN_ROOT/navila-llama3-8b-8f__lora__default__seed0
INTERN_RUN=$RUN_ROOT/internvla-n1-dualvln__lora__default__seed0
NAVILA_FINAL=$NAVILA_RUN/final
INTERN_FINAL=$INTERN_RUN/final
NAVILA_EVAL=$NAVILA_RUN/eval_test_4gpu
INTERN_EVAL=$INTERN_RUN/eval_test_4gpu

NAVILA_PREDICT_SHA=e041537dada0ca4373d5526f40fde708c5a703daeece1cc23c61917a99ddcd0c
INTERN_PREDICT_SHA=b9cd4acc51f20b404eb389cfe36b579a935d0542d86f4009f9ebb94b908c5bd4

mkdir -p "$RUN_ROOT/logs"
QUEUE_LOG=${POD4_QUEUE_LOG:-$RUN_ROOT/logs/pod4_final_queue.log}
exec > >(tee -a "$QUEUE_LOG") 2>&1

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

on_error() {
  local rc=$?
  log "QUEUE FAILED rc=$rc line=${BASH_LINENO[0]:-unknown}"
  exit "$rc"
}
trap on_error ERR

command -v flock >/dev/null || fail "flock is required"
exec 9>"$RUN_ROOT/.pod4_final_queue.lock"
flock -n 9 || fail "another Pod 4 final queue already holds the lock"

[[ -x "$NAVILA_PY" ]] || fail "NaVILA Python is missing: $NAVILA_PY"
[[ -x "$INTERN_PY" ]] || fail "InternVLA Python is missing: $INTERN_PY"
[[ -f "$TEST_JSONL" ]] || fail "test JSONL is missing: $TEST_JSONL"
[[ -f "$ROOT/scripts/episode_sharded_eval.py" ]] || fail "episode shard helper is missing"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | xargs)
[[ "$GPU_COUNT" == "$NUM_GPUS" ]] || fail "expected exactly $NUM_GPUS GPUs, found $GPU_COUNT"

require_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || fail "SHA256 mismatch for $path: expected=$expected actual=$actual"
}

verify_complete_lora_run() {
  local run_dir=$1
  local expected_tag=$2
  "$NAVILA_PY" - "$run_dir" "$expected_tag" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
expected_tag = sys.argv[2]
manifest_path = run_dir / "run_manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing run manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
checks = {
    "status": manifest.get("status") == "complete",
    "tag": manifest.get("tag") == expected_tag,
    "world_size": int(manifest.get("world_size", -1)) == 4,
    "training_scope": manifest.get("training_scope") == "lora",
}
bad = [name for name, passed in checks.items() if not passed]
if bad:
    raise SystemExit(f"invalid completed-run contract ({', '.join(bad)}): {manifest}")
final_dir = run_dir / "final"
required = [final_dir / "adapter_config.json", final_dir / "adapter_model.safetensors"]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"missing/empty final LoRA artifacts: {missing}")
metrics = manifest.get("train_metrics") or {}
for name, value in metrics.items():
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise SystemExit(f"non-finite train metric {name}={value}")
print(f"verified complete LoRA run: {run_dir}")
PY
}

wait_for_external_navila() {
  local pid=$1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || fail "invalid external PID: $pid"
  if [[ ! -r "/proc/$pid/stat" ]]; then
    log "NaVILA launcher PID $pid has already exited; validating artifacts"
    return
  fi

  local command_line start_time current_start state ticks=0
  if ! command_line=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null); then
    log "NaVILA launcher PID $pid exited during initial inspection"
    return
  fi
  [[ "$command_line" == *"launch_navila_runpod.sh full 0"* ]] || \
    fail "PID $pid is not the expected NaVILA full launcher: $command_line"
  if ! start_time=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null); then
    log "NaVILA launcher PID $pid exited during start-time inspection"
    return
  fi
  log "monitoring existing NaVILA launcher pid=$pid start_ticks=$start_time"

  while [[ -r "/proc/$pid/stat" ]]; do
    if ! current_start=$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null); then
      break
    fi
    [[ "$current_start" == "$start_time" ]] || fail "PID $pid was reused while monitoring"
    if ! state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null); then
      break
    fi
    [[ "$state" != "Z" ]] || break
    if (( ticks % 10 == 0 )); then
      log "NaVILA launcher still active pid=$pid elapsed=$(ps -o etime= -p "$pid" | xargs)"
    fi
    sleep 30
    ((ticks += 1))
  done
  log "NaVILA launcher pid=$pid ended; validating manifest and final adapter"
}

isolated_python() {
  local python=$1
  local gpu=$2
  local env_name
  env_name=$(basename "$(dirname "$(dirname "$python")")")
  shift 2
  env -u PYTHONPATH -u VIRTUAL_ENV \
    PATH="$(dirname "$python"):/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    HF_HOME="$HF_CACHE" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_NVLS_ENABLE=0 \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TORCH_EXTENSIONS_DIR="$RUN_ROOT/.torch_extensions_${env_name}_gpu${gpu}" \
    "$python" "$@"
}

verify_prediction_smoke() {
  local prediction=$1
  local model_name=$2
  "$NAVILA_PY" - "$prediction" "$model_name" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
model_name = sys.argv[2]
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"{model_name} smoke expected 1 row, got {len(rows)}")
row = rows[0]
if not str(row.get("native_action") or "").strip():
    raise SystemExit(f"{model_name} smoke produced an empty native_action: {row}")
if row.get("parse_error") is not None:
    raise SystemExit(f"{model_name} smoke parse_error: {row['parse_error']}")
if row.get("used_ground_truth_context") is not False:
    raise SystemExit(f"{model_name} smoke violated no-ground-truth-context contract")
print(f"verified {model_name} one-row inference smoke: parse_status={row.get('parse_status')}")
PY
}

run_episode_sharded_prediction() {
  local model_name=$1
  local python=$2
  local predictor=$3
  local config=$4
  local checkpoint=$5
  local eval_root=$6
  local work_root="${eval_root}.work"

  [[ ! -e "$eval_root" ]] || fail "refusing to overwrite existing eval root: $eval_root"
  [[ ! -e "$work_root" ]] || fail "refusing to overwrite existing eval work root: $work_root"
  "$NAVILA_PY" "$ROOT/scripts/episode_sharded_eval.py" split \
    --source-jsonl "$TEST_JSONL" \
    --output-dir "$work_root" \
    --expected-sha256 "$EXPECTED_TEST_SHA" \
    --expected-rows "$EXPECTED_TEST_ROWS" \
    --expected-episodes "$EXPECTED_EPISODES" \
    --expected-warehouses "$EXPECTED_WAREHOUSES"

  mkdir -p "$work_root/smoke"
  log "$model_name: loading final adapter for a one-row inference smoke"
  isolated_python "$python" 0 "$predictor" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --jsonl "$TEST_JSONL" \
    --image-root "$IMAGE_ROOT" \
    --out "$work_root/smoke/test_prediction.jsonl" \
    --vlm-output-dir "$work_root/smoke/vlm_output" \
    --num-frames 1 \
    --max-new-tokens 32 \
    --limit 1 >"$work_root/smoke/predict.log" 2>&1
  verify_prediction_smoke "$work_root/smoke/test_prediction.jsonl" "$model_name"

  log "$model_name: starting four episode-disjoint prediction workers"
  local pids=() gpu shard failed=0 rc
  for gpu in 0 1 2 3; do
    shard="$work_root/shards/shard${gpu}"
    isolated_python "$python" "$gpu" "$predictor" \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --jsonl "$shard/input.jsonl" \
      --image-root "$IMAGE_ROOT" \
      --out "$shard/test_predictions.jsonl" \
      --vlm-output-dir "$shard/vlm_output" \
      --num-frames 1 \
      --max-new-tokens 32 >"$shard/predict.log" 2>&1 &
    pids[$gpu]=$!
    log "$model_name: gpu=$gpu prediction pid=${pids[$gpu]}"
  done
  for gpu in 0 1 2 3; do
    if wait "${pids[$gpu]}"; then
      log "$model_name: gpu=$gpu prediction complete"
    else
      rc=$?
      failed=1
      log "$model_name: gpu=$gpu prediction failed rc=$rc"
    fi
  done
  (( failed == 0 )) || fail "$model_name: at least one prediction worker failed"

  "$NAVILA_PY" "$ROOT/scripts/episode_sharded_eval.py" merge \
    --source-jsonl "$TEST_JSONL" \
    --shards-root "$work_root" \
    --output-dir "$eval_root" \
    --expected-sha256 "$EXPECTED_TEST_SHA" \
    --expected-rows "$EXPECTED_TEST_ROWS" \
    --expected-episodes "$EXPECTED_EPISODES" \
    --expected-warehouses "$EXPECTED_WAREHOUSES"
  log "$model_name: merged and validated $eval_root/test_predictions.jsonl"
}

run_intern_full() {
  local args=(
    "$ROOT/train.py"
    --config "$ROOT/configs/models/internvla_n1.yaml"
    --method "$ROOT/configs/methods/lora_r16.yaml"
    --seed 0
    --ablation default
    --data_root "$DATA_ROOT"
    --image_root "$IMAGE_ROOT"
    --output_root "$RUN_ROOT"
    --deepspeed none
    --report_to none
    --resume none
  )

  [[ ! -e "$INTERN_RUN" ]] || fail "refusing to overwrite existing InternVLA run: $INTERN_RUN"
  log "InternVLA-N1: validating full-run configuration"
  isolated_python "$INTERN_PY" 0 "${args[@]}" --dry-run
  log "InternVLA-N1: starting 4-GPU LoRA full run (DeepSpeed disabled on B200)"
  env -u PYTHONPATH -u VIRTUAL_ENV \
    PATH="$(dirname "$INTERN_PY"):/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    HF_HOME="$HF_CACHE" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NCCL_NVLS_ENABLE=0 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TORCH_EXTENSIONS_DIR="$RUN_ROOT/.torch_extensions_intern_train" \
    "$INTERN_PY" -m torch.distributed.run \
      --standalone --nproc_per_node "$NUM_GPUS" "${args[@]}" \
      2>&1 | tee -a "$RUN_ROOT/logs/full_internvla_n1_seed0.log"
  verify_complete_lora_run "$INTERN_RUN" "internvla-n1-dualvln__lora__default__seed0"
}

log "Pod 4 final queue starting"
require_sha256 "$ROOT/predict_labels_navila.py" "$NAVILA_PREDICT_SHA"
require_sha256 "$ROOT/predict_labels.py" "$INTERN_PREDICT_SHA"

if [[ -n "$CURRENT_NAVILA_PID" ]]; then
  wait_for_external_navila "$CURRENT_NAVILA_PID"
fi
verify_complete_lora_run "$NAVILA_RUN" "navila-llama3-8b-8f__lora__default__seed0"
require_sha256 "$ROOT/predict_labels_navila.py" "$NAVILA_PREDICT_SHA"

run_episode_sharded_prediction \
  "NaVILA" \
  "$NAVILA_PY" \
  "$ROOT/predict_labels_navila.py" \
  "$ROOT/configs/models/navila.yaml" \
  "$NAVILA_FINAL" \
  "$NAVILA_EVAL"

run_intern_full

# Re-check the exact predictor after the long training stage. This catches a
# stale code sync before it can silently turn every InternVLA label into the
# judge's error fallback.
require_sha256 "$ROOT/predict_labels.py" "$INTERN_PREDICT_SHA"
run_episode_sharded_prediction \
  "InternVLA-N1" \
  "$INTERN_PY" \
  "$ROOT/predict_labels.py" \
  "$ROOT/configs/models/internvla_n1.yaml" \
  "$INTERN_FINAL" \
  "$INTERN_EVAL"

log "QUEUE COMPLETE: NaVILA and InternVLA-N1 full runs and merged test predictions are validated"
