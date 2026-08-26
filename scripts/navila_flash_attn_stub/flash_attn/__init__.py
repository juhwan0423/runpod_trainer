"""Import-satisfying stub, not a real flash-attn implementation.

llava/model/multimodal_encoder/intern/flash_attention.py (and a handful of
other files under llava/train/, used only for MPT/Mixtral/sequence-parallel
training paths this project never exercises) import from `flash_attn`
unconditionally at module load time, even though the actual vision tower we
load (SigLIP, per configs/models/navila.yaml) never reaches that code. A
prebuilt flash-attn wheel matching this torch/CUDA build reproducibly
SIGSEGVed on load here (2026-08-22, see NaVILA_implementation.md) and is not
needed for anything this adapter actually calls -- every symbol below exists
only so those `import` statements succeed; calling one is a bug (a code path
this project doesn't use got reached) and raises loudly instead of running
mismatched native code.
"""

__version__ = "0.0.0-import-stub"


def _unimplemented(name):
    def _fn(*args, **kwargs):
        raise RuntimeError(
            f"flash_attn.{name} was actually called, but this venv has only the "
            "import-satisfying stub installed (scripts/navila_flash_attn_stub) -- "
            "install a real flash-attn build matching this torch/CUDA if a code "
            "path now genuinely needs it"
        )

    _fn.__name__ = name
    return _fn


flash_attn_func = _unimplemented("flash_attn_func")
flash_attn_varlen_func = _unimplemented("flash_attn_varlen_func")
