#!/usr/bin/env bash
# Build a dedicated venv for internvla_n1_unified. Cannot share a venv with
# gemma4_unified (transformers 5.12.1) or navila_unified (transformers
# 4.37.2, patched) -- this model's own repo pins transformers==4.51.0.
#
# Reuses the InternNav source checkout at /home/oem6/InternNav (read-only,
# `pip install -e` only creates a .pth link, never writes into that repo) --
# this is the exact checkout whose zero-shot integration
# (examples/wrappers/internvla_offline.py) the team already validated, so we
# deliberately don't re-clone a separate copy.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${SAIE_INTERNVLA_N1_VENV_DIR:-${PERSIST_ROOT}/venvs/saie-internvla-n1}
INTERNNAV_SRC=${INTERNNAV_SRC:-/home/oem6/InternNav}
TORCH_INDEX_URL=${SAIE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}
TORCH_VERSION=${SAIE_TORCH_VERSION:-2.6.0}
TORCHVISION_VERSION=${SAIE_TORCHVISION_VERSION:-0.21.0}

# Lesson from NaVILA (NaVILA_implementation.md §14): a bare `python3` can
# silently resolve to a version the pinned wheel set doesn't support. Find
# python3.10 explicitly (matches the known-working reference venv at
# /data/oem6/SAIE_runtime_envs/internnav-venv) rather than trusting whatever
# `python3` happens to mean on this machine.
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
if [[ -z "$PYTHON_BIN" ]]; then
  echo "no python3.10 found on PATH -- InternVLA-N1's pinned wheel set targets cp310;" >&2
  echo "install python3.10 (e.g. apt-get install python3.10-venv) and re-run" >&2
  exit 1
fi
echo "using python: $PYTHON_BIN ($("$PYTHON_BIN" --version))"

[[ -d "$INTERNNAV_SRC" ]] || {
  echo "InternNav source checkout not found at $INTERNNAV_SRC -- set INTERNNAV_SRC" >&2
  exit 1
}
DEPTH_CKPT="$INTERNNAV_SRC/checkpoints/depth_anything_v2_metric_hypersim_vits.pth"
[[ -f "$DEPTH_CKPT" ]] || {
  echo "missing $DEPTH_CKPT -- InternVLAN1Model.__init__ hard-requires this separate" >&2
  echo "DepthAnythingV2 weight (not part of the HF checkpoint download); download it from" >&2
  echo "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small" >&2
  exit 1
}

mkdir -p "$PERSIST_ROOT/venvs"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # Prefer stdlib venv, but fall back to the `virtualenv` package: some
  # hosts (confirmed 2026-08-22 on this workstation) lack the
  # python3.10-venv apt package (ensurepip missing) and there's no sudo to
  # install it. virtualenv bootstraps its own pip without relying on
  # ensurepip, so it works unmodified on those hosts.
  if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/tmp/venv_stderr_$$; then
    :
  elif grep -q "ensurepip is not available" /tmp/venv_stderr_$$; then
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
  else
    cat /tmp/venv_stderr_$$ >&2
    rm -f /tmp/venv_stderr_$$
    exit 1
  fi
  rm -f /tmp/venv_stderr_$$
fi
# See launch_runpod.sh/setup_runpod.sh's identical fix for gemma4_unified:
# $VENV_DIR/bin isn't on PATH just because we invoke $VENV_DIR/bin/python by
# full path, and DeepSpeed's CPUAdam JIT builder shells out to `ninja` by
# bare name.
export PATH="${VENV_DIR}/bin:${PATH}"

# setuptools<81 is intentional, not a stale pin: setuptools>=81 removed
# pkg_resources entirely, and InternNav's own LongCLIP submodule
# (internnav/model/basemodel/LongCLIP/model/longclip.py) does
# `from pkg_resources import packaging` at import time -- confirmed
# 2026-08-22, a bare `pip install --upgrade setuptools` here produced
# setuptools==84.0.0 and a real `ModuleNotFoundError: No module named
# 'pkg_resources'` crash on every rank during the pod smoke test.
"$VENV_DIR/bin/python" -m pip install --upgrade pip "setuptools<81" wheel
"$VENV_DIR/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
"$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements-internvla-n1.txt"
# Transitive metadata from diffusion_policy/depth-camera-filtering may pull
# InternNav's historical torch==2.6 build back in. Reassert the explicitly
# selected target pair after dependency resolution so B200 retains sm_100.
"$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall --no-deps \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
"$VENV_DIR/bin/python" -m pip install -e "$INTERNNAV_SRC" --no-deps

"$VENV_DIR/bin/python" - <<'PY'
import torch, transformers
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
import internnav
print(f"internnav={internnav.__file__}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]={prop.name} vram={prop.total_memory / 1024**3:.1f} GiB")
PY

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
  "next: cd $INTERNNAV_SRC before running any script that touches InternVLAN1Model" \
  "  (build_depthanythingv2's relative-path constant is monkeypatched by" \
  "  models/registry.py's InternVLAN1UnifiedAdapter, so train.py itself does" \
  "  NOT need to run from this cwd -- only ad-hoc verification scripts do" \
  "  unless they also import+patch internvla_n1_arch.MODEL_PATH_TO first)"
