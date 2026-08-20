#!/usr/bin/env bash
# Evaluate pre-generated <warehouse>/<episode>.json outputs with the SAIE V1 judge.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SAIE_EVAL_CODE_ROOT=${1:?usage: evaluate_saie_judge.sh SAIE_CODE_ROOT RAW_ROOT VLM_ROOT OUT_ROOT}
RAW_ROOT=${2:?usage: evaluate_saie_judge.sh SAIE_CODE_ROOT RAW_ROOT VLM_ROOT OUT_ROOT}
VLM_ROOT=${3:?usage: evaluate_saie_judge.sh SAIE_CODE_ROOT RAW_ROOT VLM_ROOT OUT_ROOT}
OUT_ROOT=${4:?usage: evaluate_saie_judge.sh SAIE_CODE_ROOT RAW_ROOT VLM_ROOT OUT_ROOT}
PYTHON=${SAIE_PYTHON:-python}
INCLUDE_PER_FRAME=${SAIE_JUDGE_INCLUDE_PER_FRAME:-1}

[[ -f "$SAIE_EVAL_CODE_ROOT/src/benchmark/judge.py" ]] || {
  echo "SAIE judge not found: $SAIE_EVAL_CODE_ROOT/src/benchmark/judge.py" >&2
  exit 2
}
[[ -d "$RAW_ROOT" ]] || { echo "raw root not found: $RAW_ROOT" >&2; exit 2; }
[[ -d "$VLM_ROOT" ]] || { echo "VLM output root not found: $VLM_ROOT" >&2; exit 2; }
[[ "$INCLUDE_PER_FRAME" == "0" || "$INCLUDE_PER_FRAME" == "1" ]] || {
  echo "SAIE_JUDGE_INCLUDE_PER_FRAME must be 0 or 1" >&2
  exit 2
}
mkdir -p "$OUT_ROOT"

shopt -s nullglob
PREDICTIONS=("$VLM_ROOT"/warehouse_s*/*.json)
if (( ${#PREDICTIONS[@]} == 0 )); then
  echo "no predictions matching $VLM_ROOT/warehouse_s*/*.json" >&2
  exit 2
fi

SCORED=0
SKIPPED=0
for PREDICTION in "${PREDICTIONS[@]}"; do
  WAREHOUSE_ID=$(basename "$(dirname "$PREDICTION")")
  EPISODE_ID=$(basename "$PREDICTION" .json)
  WAREHOUSE_ROOT="$RAW_ROOT/$WAREHOUSE_ID"
  MASTER="$WAREHOUSE_ROOT/master/$EPISODE_ID.json"
  ANNOTATION="$WAREHOUSE_ROOT/annotations/$EPISODE_ID.json"
  SIM_EVENTS="$WAREHOUSE_ROOT/frames/$EPISODE_ID/sim_events.json"
  FLOOR_META="$WAREHOUSE_ROOT/floor_metadata.json"
  SCORE_DIR="$OUT_ROOT/$WAREHOUSE_ID/scores"
  SCORE="$SCORE_DIR/score_$EPISODE_ID.json"

  for REQUIRED in "$MASTER" "$ANNOTATION" "$SIM_EVENTS"; do
    [[ -f "$REQUIRED" ]] || {
      echo "required judge input missing: $REQUIRED" >&2
      exit 2
    }
  done
  if [[ -f "$SCORE" && "${SAIE_JUDGE_OVERWRITE:-0}" != "1" ]]; then
    echo "[skip existing] $WAREHOUSE_ID/$EPISODE_ID"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  mkdir -p "$SCORE_DIR"
  JUDGE_ARGS=(
    --master "$MASTER"
    --annotation "$ANNOTATION"
    --vlm-output "$PREDICTION"
    --sim-events "$SIM_EVENTS"
    --version 1
    --output "$SCORE"
  )
  [[ -f "$FLOOR_META" ]] && JUDGE_ARGS+=(--floor-meta "$FLOOR_META")
  [[ "$INCLUDE_PER_FRAME" == "1" ]] && JUDGE_ARGS+=(--include-per-frame)
  (
    cd "$SAIE_EVAL_CODE_ROOT"
    "$PYTHON" -m src.benchmark.judge "${JUDGE_ARGS[@]}"
  )
  SCORED=$((SCORED + 1))
  echo "[scored] $WAREHOUSE_ID/$EPISODE_ID"
done

"$PYTHON" "$ROOT/aggregate_saie_scores.py" \
  --scores-root "$OUT_ROOT" \
  --out "$OUT_ROOT/aggregate.json"

echo "SAIE judge complete: new=$SCORED existing=$SKIPPED aggregate=$OUT_ROOT/aggregate.json"
