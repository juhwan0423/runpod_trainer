#!/usr/bin/env bash
# Build a code-only upload archive.  Dataset/images stay separate because a
# paper corpus is hundreds of GB and should be staged on the persistent volume.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${1:-$ROOT/dist}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE="$DEST/saie-finetuning-code_${STAMP}.tar.gz"
mkdir -p "$DEST"

# Keep this an explicit public-code allowlist. The working directory may also
# contain paper drafts, raw annotations, private screenshots, or a Git checkout;
# none of those belong in a RunPod code archive.
INCLUDE=(
  README.md
  .gitignore
  requirements-runpod.txt
  ANNOTATION_AUDIT.md
  가이드.md
  aggregate_saie_scores.py
  annotation_audit.py
  evaluate_predictions.py
  predict_labels.py
  train.py
  configs
  data
  models
  scripts
  tests
)
for relative in "${INCLUDE[@]}"; do
  [[ -e "$ROOT/$relative" ]] || {
    echo "required bundle member is missing: $ROOT/$relative" >&2
    exit 2
  }
done

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  -C "$ROOT" \
  -czf "$ARCHIVE" \
  "${INCLUDE[@]}"

(
  cd "$DEST"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)
echo "$ARCHIVE"
echo "$ARCHIVE.sha256"
