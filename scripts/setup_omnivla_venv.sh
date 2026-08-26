#!/usr/bin/env bash
# Build a dedicated venv for omnivla_unified. Follows /home/oem6/OmniVLA/SETUP.md's
# own tested pins (torch==2.2.0, python3.10, flash-attn==2.5.5) as closely as
# possible rather than guessing newer versions -- this repo pulls a forked
# transformers (moojink/transformers-openvla-oft, bidirectional-attention
# patched) via `pip install -e .`, so drifting versions risks incompatibility
# with that fork in ways this project has no way to pre-verify.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${SAIE_OMNIVLA_VENV_DIR:-${PERSIST_ROOT}/venvs/saie-omnivla}
OMNIVLA_SRC=${OMNIVLA_SRC:-/home/oem6/OmniVLA}
TORCH_INDEX_URL=${SAIE_OMNIVLA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}

[[ -d "$OMNIVLA_SRC" ]] || {
  echo "OmniVLA source checkout not found at $OMNIVLA_SRC -- set OMNIVLA_SRC" >&2
  exit 1
}

PYTHON_BIN=${SAIE_PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
      if [[ "$version" == "3.10" ]]; then
        PYTHON_BIN=$(command -v "$candidate")
        break
      fi
    fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || { echo "no python3.10 found -- OmniVLA's own SETUP.md pins python3.10" >&2; exit 1; }
echo "using python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

mkdir -p "$PERSIST_ROOT/venvs"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
export PATH="${VENV_DIR}/bin:${PATH}"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Pin exactly what SETUP.md pins, from the official index + PyPI fallback
# for pure-Python deps that index alone can't resolve (same pattern this
# project already hit and fixed for B200's torch install).
"$VENV_DIR/bin/python" -m pip install \
  --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple \
  numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Installs the pyproject.toml deps, including the forked transformers.
"$VENV_DIR/bin/python" -m pip install -e "$OMNIVLA_SRC"

"$VENV_DIR/bin/python" -m pip install packaging ninja
"$VENV_DIR/bin/python" -m pip install "flash-attn==2.5.5" --no-build-isolation

"$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} ({transformers.__file__})")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]={prop.name} vram={prop.total_memory / 1024**3:.1f} GiB")
PY

printf '%s\n' \
  "setup complete" \
  "activate: source $VENV_DIR/bin/activate" \
  "OmniVLA source: $OMNIVLA_SRC"
