#!/usr/bin/env bash
# Idempotent NaVILA-only venv setup on a RunPod persistent volume.
#
# This is a SEPARATE venv from scripts/setup_runpod.sh's saie-ft venv. NaVILA
# needs transformers==4.37.2 with its site-packages files monkey-patched by
# the upstream VILA repo (llava/train/transformers_replace + deepspeed_replace)
# -- incompatible with the transformers==5.12.1 pin used for Gemma4. Do not
# point SAIE_VENV_DIR at the same directory as the Gemma4 setup.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PERSIST_ROOT=${SAIE_PERSIST_ROOT:-/workspace}
VENV_DIR=${NAVILA_VENV_DIR:-${PERSIST_ROOT}/venvs/navila-ft}
VILA_REPO_DIR=${NAVILA_REPO_DIR:-${PERSIST_ROOT}/vila-repo}
# Pinned 2026-08-21 to the commit actually checked out in the local reference
# clone at /home/oem6/NaVILA (github.com/AnjieCheng/NaVILA, branch main).
VILA_REPO_URL=${NAVILA_REPO_URL:-https://github.com/AnjieCheng/NaVILA.git}
VILA_REPO_COMMIT=${NAVILA_REPO_COMMIT:-76b98f233dd0fff05dfcd69435eec6740febff9d}
HF_CACHE=${HF_HOME:-${PERSIST_ROOT}/cache/huggingface}
PIP_CACHE=${PIP_CACHE_DIR:-${PERSIST_ROOT}/cache/pip}
TORCH_INDEX_URL=${NAVILA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
TORCH_VERSION=${NAVILA_TORCH_VERSION:-2.3.0}
TORCHVISION_VERSION=${NAVILA_TORCHVISION_VERSION:-0.18.0}
REQUIREMENTS_FILE=${NAVILA_REQUIREMENTS_FILE:-$ROOT/requirements-navila.txt}
# flash-attn is a HARD requirement, not an optional accelerator -- confirmed
# empirically 2026-08-22 (see NaVILA_implementation.md) that
# transformers_replace's patched modeling_llama.py calls
# self._flash_attention_forward() unconditionally from LlamaAttention.forward,
# not gated on attn_implementation at all, so training crashes/errors
# regardless of "eager"/"sdpa" without a real working flash-attn install.
#
# The exact prebuilt wheel from upstream environment_setup.sh (torch 2.3,
# cu122, cp310, cxx11abiFALSE) is the default target. On the local A6000 dev
# box this wheel installs cleanly but SIGSEGVs on the first real
# forward+backward -- root cause not fully isolated (that box's system nvcc
# is 11.5, too old to build flash-attn from source at all as a way to
# cross-check: FlashAttention requires nvcc >= 11.6), so it's unresolved
# whether the wheel itself is bad or something else about that box is. DO
# NOT ship configs/models/navila.yaml as smoke-tested until this passes a
# real forward+backward on the actual target pod -- verify with
# NAVILA_SKIP_FLASH_ATTN=0 (default) and a real smoke run, not the import
# stub below, before trusting it.
FLASH_ATTN_WHEEL=${NAVILA_FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}
# flash-attn 2.8.x otherwise compiles every supported CUDA architecture.  On
# a homogeneous Blackwell pod that needlessly multiplies source-build time;
# set NAVILA_FLASH_ATTN_CUDA_ARCHS=100 to build only the B200 kernels.  Leave
# this unset for the historical prebuilt-wheel path above.
FLASH_ATTN_CACHE_ARGS=()
if [[ -n "${NAVILA_FLASH_ATTN_CUDA_ARCHS:-}" ]]; then
  export FLASH_ATTN_CUDA_ARCHS="$NAVILA_FLASH_ATTN_CUDA_ARCHS"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE
  # pip's locally-built wheel cache key does not encode the Torch ABI.  A
  # wheel previously built against torch 2.3 must never be reused after a
  # Blackwell torch upgrade.
  FLASH_ATTN_CACHE_ARGS+=(--no-cache-dir)
fi

mkdir -p "$PERSIST_ROOT/venvs" "$HF_CACHE" "$PIP_CACHE"
[[ -f "$REQUIREMENTS_FILE" ]] || {
  echo "NaVILA requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
}

if [[ ! -d "$VILA_REPO_DIR/.git" ]]; then
  git clone "$VILA_REPO_URL" "$VILA_REPO_DIR"
fi
git -C "$VILA_REPO_DIR" fetch origin
git -C "$VILA_REPO_DIR" checkout "$VILA_REPO_COMMIT"

# LlavaMetaModel.save_pretrained()'s own signature is
# (self, output_dir, state_dict=None) -- confirmed 2026-08-21 on the H100
# pod via a real TypeError("...got an unexpected keyword argument
# 'safe_serialization'") the first time Trainer._save() actually ran (this
# venv's transformers always passes that kwarg). The method's own body never
# needs it (each submodule's own .save_pretrained() call picks its own
# serialization format), so accepting-and-ignoring extra kwargs is
# sufficient -- not a vendored-repo bug we're routing around, just an older
# method signature meeting a newer Trainer. Idempotent: only patches if the
# unpatched signature is still present (a fresh git checkout every run).
LLAVA_ARCH_PY="$VILA_REPO_DIR/llava/model/llava_arch.py"
if grep -q 'def save_pretrained(self, output_dir, state_dict=None):' "$LLAVA_ARCH_PY"; then
  sed -i 's/def save_pretrained(self, output_dir, state_dict=None):/def save_pretrained(self, output_dir, state_dict=None, **kwargs):/' "$LLAVA_ARCH_PY"
fi

# repack_multimodal_data()'s non-sequence-parallel branch (the one this
# project actually uses -- we never set up SP) receives attention_mask as
# int64, not bool, then uses it directly as a boolean mask:
# labels[sorted_idx[i]][attention_mask[sorted_idx[i]]]. With an int64 index
# tensor, PyTorch silently does advanced/fancy INTEGER indexing instead of
# boolean masking -- for an unpadded single-sequence batch (attention_mask
# all 1s) this selects element [1] of the sequence (deep in the ignored
# prompt prefix, so almost always -100) repeated for the full length,
# discarding every real label. Confirmed empirically 2026-08-22 by
# instrumenting a real training step: outputs.loss was NaN on every step
# (silently logged as 0.0 by HF Trainer's logging_nan_inf_filter -- not
# "no learning happening", literally NaN gradients on every backward()).
# VILA's own sequence-parallel branch a few hundred lines above already does
# `attention_mask = attention_mask.bool()` -- this is a missing cast in the
# non-SP branch specifically, not something intentional. Idempotent: only
# patches if the unpatched line is still present.
if grep -q '        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)' "$LLAVA_ARCH_PY"; then
  python3 - "$LLAVA_ARCH_PY" <<'PY'
import sys

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()
target = "        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)\n"
idx = [i for i, l in enumerate(lines) if l == target]
assert len(idx) == 1, idx
i = idx[0]
lines[i:i] = ["        attention_mask = attention_mask.bool()\n"]
with open(path, "w") as f:
    f.writelines(lines)
PY
fi

# Must be 3.10 specifically, not whatever `python3` happens to resolve to --
# confirmed 2026-08-22 on a fresh RunPod H100 image where system `python3` is
# 3.12: a venv built from bare `python3 -m venv` silently accepted that,
# then the flash-attn wheel (cp310-tagged, matching the VILA repo's own
# tested version) failed with "not a supported wheel on this platform" only
# at the flash-attn install step, well after everything else had appeared to
# work. Pin the interpreter explicitly so this fails at venv-creation time
# instead, if 3.10 isn't present at all.
PYTHON_BIN=${NAVILA_PYTHON_BIN:-python3.10}
command -v "$PYTHON_BIN" >/dev/null || {
  echo "FATAL: $PYTHON_BIN not found -- NaVILA's transformers_replace patches and the" >&2
  echo "  flash-attn cp310 wheel both target Python 3.10 specifically. Install it" >&2
  echo "  (e.g. apt-get install -y python3.10 python3.10-venv) or set NAVILA_PYTHON_BIN" >&2
  echo "  to a 3.10 interpreter." >&2
  exit 1
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  # python3 -m venv silently produces a python with no pip if the
  # python3-venv OS package is missing (seen on the local A6000 dev box,
  # which has no sudo access to install it) -- fall back to the pip-only
  # `virtualenv` package, which needs no OS package at all.
  "$PYTHON_BIN" -m venv "$VENV_DIR" || true
  if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m pip install --user -q virtualenv
    "$PYTHON_BIN" -m virtualenv "$VENV_DIR"
  fi
fi
export PATH="${VENV_DIR}/bin:${PATH}"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

# Install VILA itself (editable) before pinning transformers -- pyproject.toml
# pulls in a newer transformers as a transitive dep otherwise, which the next
# step then downgrades and patches.
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install --no-deps -e "$VILA_REPO_DIR"
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install --no-deps -e "${VILA_REPO_DIR}[train]"

# The selected requirements manifest may document a Torch pair, but the
# explicitly selected target above remains authoritative (notably cu128 on
# Blackwell).
FILTERED_REQUIREMENTS=$(mktemp)
trap 'rm -f "$FILTERED_REQUIREMENTS"' EXIT
grep -vE '^(torch|torchvision)==' "$REQUIREMENTS_FILE" > "$FILTERED_REQUIREMENTS"
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install \
  -r "$FILTERED_REQUIREMENTS"

# VILA's own editable metadata pins its historical torch==2.3 pair and may
# silently downgrade an explicitly selected Blackwell-compatible build.
# Reassert the target pair after all upstream dependency resolution and
# before compiling flash-attn, which is ABI-coupled to Torch.
PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall --no-deps \
  --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"

# The transformers/deepspeed monkey-patch: overwrites specific files inside
# this venv's own site-packages, not the system install. Must run after both
# packages are installed and must be re-run any time this venv's transformers
# or deepspeed version changes.
SITE_PKG=$("$VENV_DIR/bin/python" -c 'import site; print(site.getsitepackages()[0])')
cp -rv "$VILA_REPO_DIR/llava/train/transformers_replace/." "$SITE_PKG/transformers/"
cp -rv "$VILA_REPO_DIR/llava/train/deepspeed_replace/." "$SITE_PKG/deepspeed/"

if [[ "${NAVILA_SKIP_FLASH_ATTN:-0}" == "1" ]]; then
  echo "NAVILA_SKIP_FLASH_ATTN=1: installing the import-only stub instead of real flash-attn." >&2
  echo "  Training WILL fail (or worse, silently misbehave) with only the stub installed --" >&2
  echo "  this mode exists for import/tokenizer/module-name checks only." >&2
  PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install -e "$ROOT/scripts/navila_flash_attn_stub"
elif ! PIP_CACHE_DIR="$PIP_CACHE" "$VENV_DIR/bin/python" -m pip install \
  --no-build-isolation --force-reinstall --no-deps "${FLASH_ATTN_CACHE_ARGS[@]}" \
  "$FLASH_ATTN_WHEEL"; then
  echo "FATAL: flash-attn wheel install failed. This is a hard requirement for training" >&2
  echo "  (transformers_replace's patched modeling_llama.py calls flash attention" >&2
  echo "  unconditionally, not just when requested) -- set NAVILA_FLASH_ATTN_WHEEL to a" >&2
  echo "  wheel matching this venv's exact torch+CUDA build, or build from source with" >&2
  echo "  nvcc >= 11.6 (pip install --no-build-isolation flash-attn==2.5.8)." >&2
  exit 1
fi

"$VENV_DIR/bin/python" - <<'PY'
import torch
import transformers
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
import llava  # noqa: F401
print("llava package importable: OK")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        print(f"gpu[{index}]={prop.name} vram={prop.total_memory / 1024**3:.1f} GiB")
PY

printf '%s\n' \
  "setup complete" \
  "activate: source $VENV_DIR/bin/activate" \
  "vila repo: $VILA_REPO_DIR @ $VILA_REPO_COMMIT" \
  "HF cache: $HF_CACHE" \
  "next: export HF_TOKEN without writing it into this repository, then run" \
  "  scripts/verify_navila_exact_prefix.py"
