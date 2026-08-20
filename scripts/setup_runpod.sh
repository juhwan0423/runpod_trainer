#!/usr/bin/env bash
# Idempotent setup on a RunPod persistent volume.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${SAIE_VENV_DIR:-${PERSIST_ROOT}/venvs/saie-ft}
HF_CACHE=${HF_HOME:-${PERSIST_ROOT}/cache/huggingface}
PIP_CACHE=${PIP_CACHE_DIR:-${PERSIST_ROOT}/cache/pip}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${PERSIST_ROOT}/cache/torch_extensions}
TORCH_INDEX_URL=${SAIE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
export TORCH_EXTENSIONS_DIR

mkdir -p "$PERSIST_ROOT/venvs" "$HF_CACHE" "$PIP_CACHE" "$TORCH_EXTENSIONS_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  torch==2.12.1 torchvision==0.27.1
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements-runpod.txt"

"$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]={prop.name} vram={prop.total_memory / 1024**3:.1f} GiB")
PY

# The single-H100 recipe needs DeepSpeedCPUAdam.  Build it now so paid training
# time does not begin with a surprise compiler failure. Only users who have
# explicitly selected and smoke-tested the non-offload zero3.json recipe should
# set SAIE_SKIP_CPU_ADAM_BUILD=1.
if [[ "${SAIE_SKIP_CPU_ADAM_BUILD:-0}" != "1" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
from deepspeed.ops.op_builder import CPUAdamBuilder
print("building/loading DeepSpeedCPUAdam...")
CPUAdamBuilder().load(verbose=True)
print("DeepSpeedCPUAdam ready")
PY
fi

printf '%s\n' \
  "setup complete" \
  "activate: source $VENV_DIR/bin/activate" \
  "HF cache: $HF_CACHE" \
  "Torch extensions: $TORCH_EXTENSIONS_DIR" \
  "Torch wheel index: $TORCH_INDEX_URL" \
  "next: export HF_TOKEN without writing it into this repository"
