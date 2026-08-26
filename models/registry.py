"""Native model adapters for the SAIE training harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_REGISTRY: dict[str, type["BaseAdapter"]] = {}


def parameter_numel(parameter) -> int:
    """Return logical size for both ordinary and ZeRO-3 partitioned params."""

    deepspeed_numel = getattr(parameter, "ds_numel", None)
    return int(deepspeed_numel) if deepspeed_numel else int(parameter.numel())


def register(name: str):
    def decorator(cls: type["BaseAdapter"]):
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str, cfg: dict[str, Any]) -> "BaseAdapter":
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; available={sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)


class BaseAdapter:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        raise NotImplementedError

    def freeze_input_modules(self, model) -> dict[str, int]:
        """Freeze configured modality modules, failing on stale parameter prefixes."""
        counts: dict[str, int] = {}
        for prefix in self.cfg.get("freeze_prefixes", []):
            matched = 0
            matched_parameters = 0
            for name, parameter in model.named_parameters():
                if name == prefix or name.startswith(prefix + "."):
                    parameter.requires_grad_(False)
                    matched += parameter_numel(parameter)
                    matched_parameters += 1
            if matched_parameters == 0:
                raise ValueError(f"freeze prefix {prefix!r} matched zero model parameters")
            counts[prefix] = matched
        return counts


@register("gemma4_unified")
class Gemma4UnifiedAdapter(BaseAdapter):
    """Official Transformers path for ``google/gemma-4-12B-it``.

    The official 12B Unified config contains ``model.vision_tower`` and
    ``model.audio_tower`` modality encoders plus ``embed_vision`` and
    ``embed_audio`` soft-token embedders/projectors. Training recipes must name
    all four explicitly so a stale or incomplete freeze policy fails closed.
    """

    def load(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        hf_id = self.cfg["hf_id"]
        revision = self.cfg.get("revision")
        hub_kwargs: dict[str, Any] = {}
        if revision and not Path(str(hf_id)).expanduser().is_dir():
            hub_kwargs["revision"] = revision
        dtype_name = self.cfg.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")
        load_kwargs: dict[str, Any] = {
            "dtype": getattr(torch, dtype_name),
            "attn_implementation": self.cfg.get("attn_implementation", "sdpa"),
            "trust_remote_code": self.cfg.get("trust_remote_code", False),
            "low_cpu_mem_usage": True,
            **hub_kwargs,
        }
        quantization = self.cfg.get("quantization") or {}
        if quantization.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        self.processor = AutoProcessor.from_pretrained(
            hf_id,
            trust_remote_code=self.cfg.get("trust_remote_code", False),
            **hub_kwargs,
        )
        self.model = AutoModelForMultimodalLM.from_pretrained(hf_id, **load_kwargs)
        return self.model, self.processor

    @staticmethod
    def _user_conversation(example: dict[str, Any]) -> list[dict[str, Any]]:
        # PIL objects avoid URL/path reinterpretation differences across
        # processor versions and guarantee images were opened locally first.
        content = [{"type": "image", "image": image} for image in example["images"]]
        content.append({"type": "text", "text": example["instruction"]})
        return [{"role": "user", "content": content}]

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        conversations = [self._user_conversation(example) for example in examples]
        # Gemma 4's completed-assistant template is not token-prefix-equivalent
        # to its `enable_thinking=False` generation prompt. Render the exact
        # inference prefix first, then append the supervised action and EOT.
        prefix_texts = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(prefix_texts, str):
            prefix_texts = [prefix_texts]
        tokenizer = getattr(self.processor, "tokenizer", None)
        eot_token = getattr(tokenizer, "eot_token", None)
        if tokenizer is None or not isinstance(eot_token, str) or not eot_token:
            raise RuntimeError("Gemma processor tokenizer exposes no eot_token")
        if len(prefix_texts) != len(examples):
            raise RuntimeError("chat template returned the wrong number of generation prefixes")
        if tokenizer.bos_token and not all(text.startswith(tokenizer.bos_token) for text in prefix_texts):
            raise RuntimeError("Gemma generation prefix does not begin with tokenizer.bos_token")
        training_texts = [
            prefix + example["target"] + eot_token + "\n"
            for prefix, example in zip(prefix_texts, examples)
        ]
        image_batches = [example["images"] for example in examples]
        common = {
            "return_tensors": "pt",
            "padding": True,
            # Never silently truncate the answer at the end. The collator
            # validates active sequence lengths against max_seq_len.
            "truncation": False,
            "add_special_tokens": False,
        }
        encoded = self.processor(
            text=training_texts,
            images=image_batches,
            **common,
        )
        prefix = self.processor(
            text=prefix_texts,
            images=image_batches,
            **common,
        )
        for key in ("pixel_values", "pixel_values_videos"):
            if key in encoded and key in prefix and encoded[key].shape != prefix[key].shape:
                raise RuntimeError(
                    f"full/prefix multimodal tensor shape differs for {key}: "
                    f"{tuple(encoded[key].shape)} vs {tuple(prefix[key].shape)}"
                )
        for key in ("token_type_ids", "mm_token_type_ids"):
            if key not in encoded or key not in prefix:
                continue
            full_feature = encoded[key]
            prefix_feature = prefix[key]
            if full_feature.ndim != 2 or prefix_feature.ndim != 2:
                raise RuntimeError(f"unexpected {key} rank in processor output")
            for row_index in range(full_feature.shape[0]):
                full_active = full_feature[row_index][encoded["attention_mask"][row_index].bool()]
                prefix_active = prefix_feature[row_index][prefix["attention_mask"][row_index].bool()]
                if not torch.equal(full_active[: len(prefix_active)], prefix_active):
                    raise RuntimeError(f"full/prefix {key} mismatch for batch row {row_index}")
        return encoded, prefix


@register("navila_unified")
class NaVILAUnifiedAdapter(BaseAdapter):
    """VILA/LLaVA loader for NaVILA (``a8cheng/navila-llama3-8b-8f``: Llama-3-8B
    + SigLIP-SO400M, ``mlp_downsample`` projector).

    Requires the ``llava`` package (the VILA training repo pinned in
    ``requirements-navila.txt``) importable, and that package's
    ``environment_setup.sh`` monkey-patch applied (it overwrites several
    ``transformers==4.37.2`` site-packages files -- see
    ``scripts/setup_navila_venv.sh``). This adapter cannot share a venv with
    ``gemma4_unified`` (transformers 5.12.1) or any other model in this
    registry; ``train.py``/``launch_navila_runpod.sh`` must invoke it with
    ``configs/models/navila.yaml``'s own interpreter.

    Module names below were read directly from ``LlavaMetaModel.__init__``
    in the local checkout at ``/home/oem6/NaVILA/llava/model/llava_arch.py``
    on 2026-08-21 (``self.llm``, ``self.vision_tower``, ``self.mm_projector``
    set as direct top-level submodules) and cross-checked against the actual
    HF Hub file listing for the pinned revision (``llm/``, ``vision_tower/``,
    ``mm_projector/`` subfolders, each with their own ``config.json``). This
    is a stronger source than the plan doc alone, but per the project's own
    lesson (config/doc claims about module names have been wrong 4/4 times
    for Gemma4) it must still be re-confirmed with a live
    ``named_parameters()`` dump the first time this adapter actually loads a
    checkpoint (ideally under the real DeepSpeed ZeRO-3 path, not a bare
    single-process load -- Gemma4's embed_vision parameter count differed
    between the two).

    Key architectural difference from ``gemma4_unified`` that the shared
    ``MultimodalSFTCollator``/``derive_target_mask`` machinery is NOT aware
    of: NaVILA does not expand ``<image>`` into per-patch placeholder tokens
    inside ``input_ids``. Each ``<image>`` tag becomes exactly one sentinel
    id (``IMAGE_TOKEN_INDEX = -200``) in the token sequence; the real vision
    patch embeddings (~196 tokens/frame after the ``mlp_downsample`` 2x2
    pool of SigLIP's 27x27 patch grid, per the pinned checkpoint's
    ``vision_tower/config.json``: image_size=384, patch_size=14) are spliced
    in as *embeddings* inside the model's own ``forward()``, after
    tokenization. This means ``data.max_seq_len`` / the collator's
    ``active_lengths`` check operate on a token count that undercounts the
    model's true effective sequence length by roughly
    ``195 * num_frames`` tokens. Not a memory risk at the 8-frame scale used
    here (~1.6-1.8k effective tokens), but do not raise ``num_frames`` far
    without accounting for this gap.
    """

    _CONV_SYSTEM_PROMPT = (
        "You are a helpful language and vision assistant. You are able to understand the visual "
        "content that the user provides, and assist the user with a variety of tasks using "
        "natural language."
    )

    def _resolve_local_snapshot(self, hf_id: str) -> str:
        if Path(hf_id).expanduser().is_dir():
            return str(Path(hf_id).expanduser())
        revision = self.cfg.get("revision")
        if not revision:
            raise ValueError(
                "configs/models/navila.yaml must pin model.revision to a specific commit hash "
                "(load_pretrained_model has no revision kwarg of its own -- this adapter "
                "downloads the exact pinned snapshot itself)"
            )
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=hf_id, revision=revision)

    def load(self):
        try:
            from llava.constants import DEFAULT_IMAGE_TOKEN  # noqa: F401
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "the `llava` package is not importable in this interpreter -- navila_unified "
                "needs its own venv built by scripts/setup_navila_venv.sh, it cannot share the "
                "gemma4 saie-ft venv (incompatible transformers pin, 4.37.2 patched vs 5.12.1)"
            ) from exc

        local_dir = self._resolve_local_snapshot(self.cfg["hf_id"])
        model_name = get_model_name_from_path(local_dir)
        dtype_name = self.cfg.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=local_dir,
            model_name=model_name,
            device_map=None,
            device="cuda",
            torch_dtype=getattr(torch, dtype_name),
            # IMPORTANT (confirmed empirically 2026-08-22, see
            # NaVILA_implementation.md): this kwarg does NOT make flash-attn
            # optional for this adapter, despite what an earlier version of
            # this comment claimed. The transformers_replace-patched
            # modeling_llama.py this venv installs (llava/train/
            # transformers_replace/models/llama/modeling_llama.py) has its
            # self-attention forward() call self._flash_attention_forward()
            # unconditionally -- not gated on config._attn_implementation at
            # all -- so neither "eager" nor "sdpa" avoids requiring a real,
            # working flash-attn install ("sdpa" additionally fails outright
            # at load time: this patched LlamaForCausalLM raises ValueError
            # from _check_and_enable_sdpa, it doesn't support that path at
            # all). "eager" is kept as the least-wrong default for the small
            # part of attn_implementation resolution that does still run
            # (avoids the auto-upgrade-to-flash_attention_2 log/config
            # rewrite), but a real flash-attn build matching this venv's
            # exact torch+CUDA is a hard requirement for training, not an
            # optional accelerator. scripts/navila_flash_attn_stub exists
            # only to make *import-only* testing (module names, tokenizer,
            # a no_grad() forward) possible without flash-attn installed --
            # it deliberately raises if anything actually calls into it,
            # which is exactly what a real forward+backward does.
            attn_implementation=self.cfg.get("attn_implementation", "eager"),
        )
        # Confirmed empirically (2026-08-22, single-GPU local load) that the
        # torch_dtype kwarg above is silently ignored: llava's
        # load_pretrained_model unconditionally does
        # `kwargs["torch_dtype"] = torch.float16` for the non-quantized path
        # (llava/model/builder.py) regardless of what the caller passed, and
        # that string is what every submodule (llm/vision_tower/mm_projector)
        # actually loads with via `eval(config.model_dtype)`. Left alone this
        # produces a real forward-pass crash: encode_batch below casts the
        # image tensor to cfg["dtype"] (bfloat16 by default), which then
        # collides with a float16 mm_projector LayerNorm
        # ("expected scalar type BFloat16 but found Half"). Force the whole
        # model to the configured dtype uniformly after load instead of
        # trusting the loader's own dtype handling.
        model = model.to(getattr(torch, dtype_name))
        if tokenizer.pad_token_id is None:
            raise RuntimeError(
                "NaVILA tokenizer exposes no pad_token_id; the pinned checkpoint's tokenizer "
                "is expected to already define one (verified for revision "
                f"{self.cfg.get('revision')!r}: pad_token='[PAD]') -- re-run the exact-prefix "
                "verification script if this fires on a different revision"
            )
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is None or eot_id == tokenizer.unk_token_id:
            raise RuntimeError("NaVILA tokenizer does not recognize the <|eot_id|> stop token")
        # Required manual init step, not something from_pretrained sets on its
        # own -- confirmed 2026-08-21 on the H100 pod via a real
        # AttributeError('LlamaForCausalLM' object has no attribute
        # 'pad_token_id') inside llava_arch.py's repack_multimodal_data
        # (training-mode sequence-packing path, entered unconditionally
        # whenever self.llm.forward accepts seqlens_in_batch, which the
        # transformers_replace-patched LlamaForCausalLM always does -- so
        # this isn't optional/skippable). VILA's own training entrypoint
        # sets this the same way right after construction
        # (llava/train/train.py:649: model.llm.pad_token_id =
        # tokenizer.pad_token_id) -- we don't go through that entrypoint, so
        # replicate the assignment here instead of trusting from_pretrained
        # to have set it.
        model.llm.pad_token_id = tokenizer.pad_token_id
        # Same category of missing manual init as pad_token_id above,
        # confirmed 2026-08-21 on the H100 pod via a real AttributeError
        # ('LlavaLlamaConfig' object has no attribute 'time_token_ids')
        # inside LlavaLlamaModel.forward's training-mode branch. VILA's own
        # training entrypoint always sets this -- to real token ids only
        # when config.num_time_tokens > 0 (a video-temporal-reasoning
        # feature this project's 14-label SAIE task doesn't use), otherwise
        # to [] (llava/train/train.py:677-687). Our pinned checkpoint's
        # persisted config has neither field at all, so replicate the
        # num_time_tokens=0 fallback explicitly rather than leave it unset.
        model.config.num_time_tokens = 0
        model.config.time_token_ids = []

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
        self.eot_id = eot_id
        self.conv_template = conv_templates["llama_3"]
        # train.py sets adapter.model = None right after load() returns, so
        # DataLoader workers never pickle the full model (see the Gemma4
        # adapter's identical comment) -- but that means encode_batch (which
        # runs later, per-batch, in the collator) cannot read self.model.config
        # the way process_images() expects (confirmed by a real
        # AttributeError: 'NoneType' object has no attribute 'config' on the
        # H100 pod, 2026-08-22). process_images only ever reads
        # image_aspect_ratio off that object and assigns its own
        # `.image_processor` onto it, so a small owned namespace with just
        # that one field is sufficient -- no need to keep the real model
        # object alive.
        from types import SimpleNamespace

        self.image_processor_cfg = SimpleNamespace(image_aspect_ratio=model.config.image_aspect_ratio)
        # BaseAdapter/MultimodalSFTCollator only ever read `.processor` and,
        # via `getattr(processor, "tokenizer", processor)`, fall back to
        # treating it as the tokenizer directly when it has no `.tokenizer`
        # attribute of its own -- so the bare HF tokenizer satisfies the
        # adapter contract without a wrapper class.
        self.processor = tokenizer
        return self.model, self.processor

    def _build_prefix_text(self, instruction: str, num_frames: int) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN

        conv = self.conv_template.copy()
        image_tags = (DEFAULT_IMAGE_TOKEN + "\n") * num_frames
        conv.append_message(conv.roles[0], image_tags + instruction)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        prefix_id_lists: list[list[int]] = []
        full_id_lists: list[list[int]] = []
        image_batches = []
        for example in examples:
            num_frames = len(example["images"])
            prefix_text = self._build_prefix_text(example["instruction"], num_frames)
            full_text = prefix_text + example["target"] + "<|eot_id|>"

            prefix_ids = tokenizer_image_token(
                prefix_text, self.tokenizer, image_token_index=IMAGE_TOKEN_INDEX
            )
            full_ids = tokenizer_image_token(
                full_text, self.tokenizer, image_token_index=IMAGE_TOKEN_INDEX
            )
            # tokenizer_image_token tokenizes the text on either side of each
            # `<image>` split independently, so this equality is expected to
            # hold structurally (verified empirically against the pinned
            # tokenizer for all 14 canonical labels -- see
            # scripts/verify_navila_exact_prefix.py); assert rather than
            # trust so a future tokenizer/template change fails closed here
            # instead of silently mis-training, matching derive_target_mask's
            # fail-closed contract downstream.
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise RuntimeError(
                    "NaVILA full-conversation token ids do not begin with the tokenized "
                    "generation prefix; naive string-concat is no longer prefix-exact for this "
                    "tokenizer/template -- switch this adapter to ID-level concatenation "
                    "(tokenize prefix and answer separately, then splice the id lists) instead "
                    "of tokenizing prefix_text and full_text as two independent strings"
                )

            prefix_id_lists.append(prefix_ids)
            full_id_lists.append(full_ids)

            processed = process_images(example["images"], self.image_processor, self.image_processor_cfg)
            image_batches.append(processed)

        def _pad_batch(id_lists: list[list[int]]) -> dict[str, torch.Tensor]:
            max_len = max(len(ids) for ids in id_lists)
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
            for row, ids in enumerate(id_lists):
                # IMAGE_TOKEN_INDEX (-200) values pass through untouched; the
                # model's forward() replaces them with spliced vision
                # embeddings before the (real) embedding lookup ever sees
                # them, so a plain torch.long tensor is safe here even though
                # -200 is not a valid vocab id.
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        encoded = _pad_batch(full_id_lists)
        prefix = _pad_batch(prefix_id_lists)
        encoded["images"] = torch.stack(image_batches, dim=0).to(getattr(torch, self.cfg.get("dtype", "bfloat16")))
        return encoded, prefix


class InternVLAN1TextWrapper(torch.nn.Module):
    """Bypasses ``InternVLAN1ForCausalLM.forward()``'s labels-triggered
    diffusion (System-1) loss branch.

    Confirmed empirically 2026-08-22 against the real pinned checkpoint
    (``scripts/verify_internvla_n1_module_names.py``,
    ``scripts/verify_internvla_n1_multimodal_forward.py``, both real-image
    and dummy-input variants): ``InternVLAN1ForCausalLM.forward()`` only
    computes a loss when ``labels is not None``, and that branch
    unconditionally indexes ``t_s_pos``/``traj_poses``/``traj_images`` (the
    trajectory-diffusion System-1 head's inputs) -- passing plain SFT text
    labels the way a stock causal-LM ``Trainer`` does crashes immediately.
    ``logits`` (``self.lm_head(hidden_states)``) is computed unconditionally
    and is unaffected by the labels branch -- a live ``forward(labels=None)``
    call, including with a real image and real ``pixel_values``/
    ``image_grid_thw``, returns correct-shaped logits with ``loss=None`` and
    no crash. This wrapper always calls the inner model with ``labels=None``,
    reads back ``.logits``, and computes an ordinary shifted cross-entropy
    itself, so ``train.py``'s generic HF ``Trainer`` (which just calls
    ``model(**batch)`` and reads ``.loss``, see the `Trainer(` construction
    site) needs no changes for this model -- see
    ``adapter_implement_plan.md`` §3.3.

    Parameter-naming consequence for ``freeze_prefixes``/``target_modules``:
    wrapping adds this class's own attribute name as an extra
    ``named_parameters()`` prefix. The inner model is stored as
    ``self.backbone``, so real InternVLA-N1 prefixes (``visual.*``,
    ``model.layers.*``, ``lm_head.*`` -- confirmed via a live dump, NOT
    ``model.visual``/``model.model.layers``/``model.lm_head`` as an earlier,
    unverified version of the plan doc guessed) become
    ``backbone.visual.*``, ``backbone.model.layers.*``, ``backbone.lm_head.*``
    on this wrapper. ``configs/models/internvla_n1.yaml`` accounts for this.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return self.backbone.device

    @property
    def generation_config(self):
        return self.backbone.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.backbone.generation_config = value

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def generate(self, *args, **kwargs):
        # This wrapper only exists to intercept forward()'s labels-triggered
        # System-1 loss branch (see class docstring) -- training never calls
        # .generate(), but predict_labels.py does, and a bare
        # torch.nn.Module (this class's base, not PreTrainedModel) has no
        # such method by default. Delegate straight to the backbone, which
        # is the real InternVLAN1ForCausalLM / GenerationMixin.
        return self.backbone.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Same category as generate() above -- confirmed needed 2026-08-24:
        # peft's get_peft_model()/PeftModelForCausalLM wrapping (LoRA method,
        # unlike full_text_backbone FFT which never touches this) probes for
        # this GenerationMixin method on the wrapped model during setup, not
        # only at actual generation time. A bare nn.Module doesn't have it.
        return self.backbone.prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        labels: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs,
    ):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        # HF Trainer (confirmed 2026-08-22 with this venv's newer Trainer)
        # passes num_items_in_batch into model.forward() for cross-batch loss
        # averaging under gradient accumulation -- InternVLAN1ForCausalLM's
        # own forward() signature predates that convention and doesn't
        # accept it (TypeError otherwise). This wrapper always computes its
        # own loss (see below) rather than delegating averaging to the
        # backbone, so silently absorbing the kwarg here is correct, not a
        # workaround that loses information: Trainer's own reduction (mean
        # over `num_items_in_batch` when set, else per-batch mean) still
        # runs on the `.loss` we return either way.
        outputs = self.backbone(labels=None, **kwargs)
        loss = None
        if labels is not None:
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=outputs.logits)


@register("internvla_n1_unified")
class InternVLAN1UnifiedAdapter(BaseAdapter):
    """InternVLA-N1 (``InternRobotics/InternVLA-N1-DualVLN``): Qwen2.5-VL-7B
    backbone plus a dual-system (System-1 diffusion trajectory head) design.

    Requires the ``internnav`` package (InternNav repo, pinned in
    ``requirements-internvla-n1.txt``) importable -- there is no ``auto_map``
    in the checkpoint's ``config.json`` and no ``AutoConfig``/``AutoModel``
    registration anywhere in that repo, so ``trust_remote_code=True`` does
    NOT work; the model class must be imported directly. Needs its own venv,
    separate from ``gemma4_unified`` (transformers 5.12.1) and
    ``navila_unified`` (transformers==4.37.2, patched) -- this model's repo
    pins ``transformers==4.51.0``.

    All architectural claims below were re-verified live against the pinned
    checkpoint on 2026-08-22 (see ``scripts/verify_internvla_n1_*.py`` --
    keep re-running these after any checkpoint/library upgrade, per this
    project's established lesson that config/plan-doc claims about module
    names have been wrong before, including an earlier draft of this same
    plan doc for this exact model):

    - Real top-level prefixes on the raw ``InternVLAN1ForCausalLM`` object:
      ``visual.*`` (ViT, NOT ``model.visual``), ``model.layers.*`` /
      ``model.embed_tokens`` / ``model.norm`` (NOT ``model.model.layers``),
      ``lm_head.*`` (NOT ``model.lm_head``, sibling of ``.model`` not nested
      inside it).
    - Dual-system (System-1) modules, all present for this checkpoint
      (``config.system1 == "nextdit_async"``): ``model.latent_queries``,
      ``model.traj_dit``, ``model.action_encoder``, ``model.action_decoder``,
      ``model.cond_projector``, ``model.memory_encoder``,
      ``model.rgb_resampler``, ``model.rgb_model`` (DepthAnythingV2).
    - Total 8,383,538,891 params; frozen (visual + all System-1 modules)
      767,922,379; trainable (full language stack: layers+norm+embed+lm_head)
      7,615,616,512 -- these are the real ``expected_parameter_counts`` for
      ``full_text_backbone``, measured directly, not estimated.
    - ``build_depthanythingv2()`` (``internvla_n1_arch.py``) loads a *separate*
      pretrained weight file via a hardcoded **relative** path
      (``MODEL_PATH_TO = "checkpoints"``, i.e. ``checkpoints/depth_anything_v2_metric_hypersim_vits.pth``
      under whatever the process's cwd happens to be) -- NOT part of the HF
      checkpoint we download. Rather than depend on process cwd (fragile for
      a long-running training job that may chdir elsewhere) or a
      hardcoded absolute path (broke on first pod deployment 2026-08-22 --
      the workstation's checkout lives at ``/home/oem6/InternNav`` but the
      pod's lives at ``/workspace/InternNav``), this adapter derives the
      directory from wherever the ``internnav`` package actually is on disk
      (``Path(internnav.__file__).parent.parent / "checkpoints"``) and
      monkeypatches the ``MODEL_PATH_TO`` module constant to that absolute
      path before constructing the model. If this asset is missing/moved,
      loading fails with a plain ``FileNotFoundError`` pointing at
      the relative path -- that error is not a bug in this adapter, go find
      (or re-download, see the source comment's HF Hub URL in
      ``internvla_n1_arch.py``) the weight file.
    - ``forward(labels=...)`` cannot be used directly -- see
      ``InternVLAN1TextWrapper`` above. ``load()`` returns that wrapper, not
      the raw ``InternVLAN1ForCausalLM``, so downstream ``freeze_prefixes``/
      ``target_modules`` must be written against ``backbone.<real name>``.
    - Naive prefix/full **string** tokenize-then-compare (the same simple
      pattern ``gemma4_unified`` uses) is exact-prefix-safe for this
      tokenizer -- verified 28/28 canonical-label x instruction combinations
      text-only, and separately confirmed with a real image's expanded
      ``<|image_pad|>`` tokens in the prefix
      (``scripts/verify_internvla_n1_exact_prefix.py``,
      ``scripts/verify_internvla_n1_multimodal_forward.py``). No ID-level
      splicing needed, unlike what an earlier, unverified draft of the plan
      doc assumed was necessary.
    """

    def load(self):
        try:
            import internnav
            import internnav.model.basemodel.internvla_n1.internvla_n1_arch as _arch
            from internnav.model.basemodel.internvla_n1.internvla_n1 import (
                InternVLAN1ForCausalLM,
            )
        except ImportError as exc:
            raise RuntimeError(
                "the `internnav` package is not importable in this interpreter -- "
                "internvla_n1_unified needs its own venv (see "
                "requirements-internvla-n1.txt), it cannot share the gemma4/navila "
                "venvs (incompatible transformers pins)"
            ) from exc

        # See the class docstring: build_depthanythingv2() resolves this
        # constant against the process cwd otherwise, which is fragile. This
        # must NOT be a hardcoded absolute path (an earlier version of this
        # adapter hardcoded the workstation's /home/oem6/InternNav path,
        # which broke the first pod deployment 2026-08-22 --
        # FileNotFoundError since the pod's checkout lives at
        # /workspace/InternNav instead). Deriving it from internnav's own
        # __file__ instead makes this correct in every environment that has
        # `internnav` installed editable from a `checkpoints/`-containing
        # checkout, matching how scripts/setup_internvla_n1_venv.sh always
        # installs it (`pip install -e "$INTERNNAV_SRC"`).
        internnav_root = Path(internnav.__file__).resolve().parent.parent
        model_path_to = self.cfg.get("internnav_checkpoints_dir") or str(internnav_root / "checkpoints")
        depth_checkpoint = Path(model_path_to) / "depth_anything_v2_metric_hypersim_vits.pth"
        if not depth_checkpoint.is_file():
            raise FileNotFoundError(
                f"InternVLA-N1's DepthAnythingV2 weight is missing: {depth_checkpoint} "
                "(this is a separate asset, not part of the HF checkpoint download -- "
                "see internvla_n1_arch.py's build_depthanythingv2 for the source URL, or "
                "set model.internnav_checkpoints_dir in configs/models/internvla_n1.yaml "
                "if internnav is installed from a checkout that keeps checkpoints/ "
                "somewhere other than its own repo root)"
            )
        _arch.MODEL_PATH_TO = model_path_to

        hf_id = self.cfg["hf_id"]
        revision = self.cfg.get("revision")
        hub_kwargs: dict[str, Any] = {}
        if revision and not Path(str(hf_id)).expanduser().is_dir():
            hub_kwargs["revision"] = revision
        dtype_name = self.cfg.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")

        device_map = self.cfg.get("device_map")
        from_pretrained_kwargs: dict[str, Any] = {}
        if device_map:
            # Inference-only escape hatch: a single-GPU load of this
            # checkpoint was observed to consume ~79GiB on one H100 (vs. the
            # ~17GiB the bf16 state dict alone accounts for at 8.38B params
            # -- root cause not chased down, plausibly ignore_mismatched_sizes
            # below forcing a non-memory-efficient load path). Never pass
            # this during training: DeepSpeed already shards/places the
            # model, and accelerate's device_map dispatch is incompatible
            # with that.
            from_pretrained_kwargs["device_map"] = device_map

        backbone = InternVLAN1ForCausalLM.from_pretrained(
            hf_id,
            # NOTE: torch_dtype, not dtype -- transformers==4.51.0 (this
            # model's pin) predates the dtype= rename that gemma4_unified's
            # transformers==5.12.1 uses; torch_dtype is what
            # scripts/verify_internvla_n1_module_names.py confirmed working
            # against this exact checkpoint/library pairing 2026-08-22.
            torch_dtype=getattr(torch, dtype_name),
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
            **from_pretrained_kwargs,
            # traj_dit (System-1)'s LuminaFeedForward inner_dim, computed by
            # internvla_n1/nextdit_traj.py, does not match this exact
            # checkpoint's saved shapes for at least one block (confirmed
            # 2026-08-22 on the pod: real from_pretrained() error, "checkpoint
            # 1024" vs "current model 768" for one feed_forward.*.weight).
            # Root cause not chased further because it does not matter: this
            # adapter freezes the entire System-1 stack (freeze_prefixes
            # above) and InternVLAN1TextWrapper never calls into it --
            # ignore_mismatched_sizes lets from_pretrained keep that
            # submodule's own (randomly-initialized-then-partially-loaded)
            # weights instead of hard-failing the whole load over params we
            # never read.
            ignore_mismatched_sizes=True,
            **hub_kwargs,
        )
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(hf_id, **hub_kwargs)
        self.model = InternVLAN1TextWrapper(backbone)
        return self.model, self.processor

    @staticmethod
    def _user_conversation(example: dict[str, Any]) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in example["images"]]
        content.append({"type": "text", "text": example["instruction"]})
        return [{"role": "user", "content": content}]

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        # Mirrors gemma4_unified's string-based dual-render pattern -- see
        # the class docstring for why this is exact-prefix-safe here (28/28
        # verified, including with real expanded image tokens), unlike the
        # more conservative ID-level-splice approach other adapters in this
        # registry needed.
        conversations = [self._user_conversation(example) for example in examples]
        prefix_texts = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        if isinstance(prefix_texts, str):
            prefix_texts = [prefix_texts]
        tokenizer = self.processor.tokenizer
        eot_token = "<|im_end|>\n"
        training_texts = [
            prefix + example["target"] + eot_token
            for prefix, example in zip(prefix_texts, examples)
        ]
        image_batches = [example["images"] for example in examples]
        common = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
            "add_special_tokens": False,
        }
        encoded = self.processor(text=training_texts, images=image_batches, **common)
        prefix = self.processor(text=prefix_texts, images=image_batches, **common)
        for key in ("pixel_values", "image_grid_thw"):
            if key in encoded and key in prefix and not torch.equal(encoded[key], prefix[key]):
                raise RuntimeError(
                    f"full/prefix multimodal tensor differs for {key} -- images must be "
                    "identical between the prefix-only and prefix+answer renders"
                )
        return encoded, prefix


class OmniVLATextWrapper(torch.nn.Module):
    """Adds a working ``.generate()`` on top of a bare ``OpenVLAForActionPrediction``.

    Confirmed 2026-08-25 by reading ``prismatic/extern/hf/modeling_prismatic.py``
    directly (not guessed): the loaded backbone has NO ``generate()``
    override of its own (grep found none), and the repo's own inference
    script never calls ``.generate()`` either -- ``predict_action()`` always
    hand-loops. Its own ``forward()`` (the plain, non-MMN
    ``OpenVLAForActionPrediction`` base class this adapter deliberately
    loads instead of the checkpoint's registered
    ``OpenVLAForActionPrediction_MMNv1`` -- see the adapter class docstring)
    already handles both a "multimodal prefill" branch (real
    `input_ids`+`pixel_values`, computes the vision splice) and a "cached
    decode" branch (`input_ids.shape[1] == 1` + `past_key_values`, skips the
    splice) -- so a correct incremental generate loop is just calling
    ``self.backbone(...)`` twice with those two branches, never touching
    ``GenerationMixin``'s default `prepare_inputs_for_generation` (unverified
    for this model's multimodal kwargs, and the repo's own avoidance of
    `.generate()` is itself a signal it may not "just work").

    Unlike ``InternVLAN1TextWrapper``, this wrapper does NOT need to
    override ``forward()`` for training -- confirmed by reading the base
    class's multimodal branch fully: it already calls
    ``self._build_multimodal_labels(labels, projected_patch_embeddings)``
    (auto-pads labels for the inserted image-patch positions) and dispatches
    to a plain ``LlamaForCausalLM`` with ``labels=``, which computes loss
    natively. ``forward()`` here is pure passthrough.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return self.backbone.device

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        return self.backbone(**kwargs)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        max_new_tokens: int = 32,
        do_sample: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if do_sample:
            raise NotImplementedError(
                "OmniVLATextWrapper.generate() only implements greedy decoding "
                "(this project's whole eval protocol uses do_sample=False everywhere)"
            )
        if input_ids.shape[0] != 1:
            raise NotImplementedError(
                "batch size 1 only -- matches predict_labels.py's own per-example loop, "
                "and the backbone's cached-decode branch itself asserts batch size 1"
            )
        eos_token_id = getattr(self.backbone.config, "eos_token_id", None)
        tokens = input_ids
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=None,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break
            outputs = self.backbone(
                input_ids=next_token,
                attention_mask=None,
                pixel_values=None,
                labels=None,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            past_key_values = outputs.past_key_values
        return tokens


@register("omnivla_unified")
class OmniVLAUnifiedAdapter(BaseAdapter):
    """OmniVLA (local repo ``/home/oem6/OmniVLA``, HF checkpoints under
    ``NHirose/omnivla-*``): a Prismatic-VLM-family OpenVLA fork (SigLIP+DINOv2
    dual vision backbone, Vicuna-v1.5-7B/Llama backbone, ``openvla`` model
    type) co-trained for navigation with a discrete action-bin output head,
    NOT a text-generation model out of the box.

    Confirmed 2026-08-25 by reading the GitHub repo's code directly (not the
    plan doc's earlier summary, which turned out to be more pessimistic than
    the actual code -- see the 2026-08-25 research note this class was
    written from for the full investigation):

    - The underlying LLM (``self.language_model``, set in
      ``modeling_prismatic.py``'s ``PrismaticForConditionalGeneration.__init__``
      via ``AutoModelForCausalLM.from_config(config.text_config, ...)`` with
      ``llm_backbone_id="vicuna-v15-7b"``) resolves to a completely plain,
      unmodified ``transformers.LlamaForCausalLM``.
    - ``ACTION_TOKEN_BEGIN_IDX=31743``'s embedding-zeroing logic
      (``_process_action_masks``) keys off ``labels``, not ``input_ids`` --
      it only ever fires because OmniVLA's own training data sets
      ``labels`` to the top-256 action-bin token ids by construction. Our
      SFT targets are plain English label strings (ordinary vocab ids, all
      well below 31743), so the mask is `False` everywhere and this logic
      is a structural no-op for us -- not something that needs bypassing.
    - HOWEVER, the ``vla-scripts/train_omnivla.py`` TRAINING script (in the
      GitHub repo) registers ``OpenVLAForActionPrediction_MMNv1`` for its
      own navigation-conditioned training, a subclass whose ``forward()``
      hard-requires a ``modality_id`` (0-8) and a fixed 1025-slot multi-image
      layout (nav-goal image + satellite image + pose) this project's
      single-current-frame SFT task has no equivalent inputs for. **This
      turned out not to matter**: confirmed 2026-08-25 by fetching the
      actual published checkpoint's own ``config.json`` from the HF Hub API
      (``NHirose/omnivla-original``) -- its own ``auto_map`` already points
      at the PLAIN base class:
      ``{"AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction"}``,
      not the MMNv1 training-script convention. So a completely standard
      ``trust_remote_code=True`` load (the checkpoint repo bundles its own
      ``modeling_prismatic.py``/``configuration_prismatic.py``/
      ``processing_prismatic.py``, dynamically imported by transformers)
      already gets the simple, single-image-compatible ``forward()``
      described above with zero extra registration code -- no need to
      import the local ``/home/oem6/OmniVLA`` checkout's ``prismatic``
      package or its forked-transformers/torch==2.2.0 pin at all, and no
      need for a dedicated venv: this can share ``gemma4_unified``'s.
    - Prompt format is OpenVLA's own convention (confirmed in
      ``prismatic/models/backbones/llm/prompting/base_prompter.py`` in the
      local repo checkout, which the checkpoint's bundled code matches):
      ``f"In: {instruction}\\nOut: "``, not a chat template -- this is not a
      chat-tuned model.
    - This specific checkpoint's backbone is Llama-2-7B (config.json:
      ``hf_llm_id: meta-llama/Llama-2-7b-hf``, ``llm_backbone_id:
      llama2-7b-pure``), not Vicuna-v1.5 -- the generic code's *default*
      when no checkpoint config is loaded, not what this checkpoint uses.
    - No native ``.generate()`` -- see ``OmniVLATextWrapper`` above.

    Loading note (confirmed 2026-08-25, reverted from an earlier, simpler
    design): the checkpoint's ``config.json`` ``auto_map`` correctly points
    at its own bundled ``modeling_prismatic.OpenVLAForActionPrediction``,
    but a plain ``trust_remote_code=True`` load still fails --
    transformers' dynamic-module loader separately checks (via
    ``importlib.metadata``, not just whether ``import prismatic`` succeeds)
    that a pip-installed distribution literally named ``prismatic`` exists,
    because the bundled file does ``from prismatic.xxx import yyy``. This
    repo's own ``pyproject.toml`` declares ``name = "OmniVLA"`` (the
    importable top-level package is coincidentally also called
    ``prismatic``, but the pip *distribution* name differs), so that check
    fails even in a venv where ``pip install -e /path/to/OmniVLA`` already
    succeeded and ``import prismatic`` works fine standalone. Importing the
    local package explicitly (below) and registering it with Auto* classes
    ourselves sidesteps that check entirely -- this needs the OmniVLA
    checkout's own venv (``scripts/setup_omnivla_venv.sh``), it cannot share
    ``gemma4_unified``'s.

    KNOWN UNRESOLVED BLOCKER (2026-08-25, do not re-attempt without reading
    this first): even with the above fixed, loading still fails with a
    genuinely irreconcilable transitive dependency conflict, not something
    `pip install`-ing a different version can solve. Root cause chain:
    ``prismatic/__init__.py`` (and, it turns out, something reachable from
    the target ``prismatic.extern.hf.*`` submodules too -- the
    ``_stub_prismatic_package_init`` trick above only patches the *first*
    layer) eventually imports ``tensorflow_datasets`` -> ``tensorflow_metadata``,
    whose generated protobuf file (``anomalies_pb2.py`` etc.) was compiled
    with protoc gencode 6.31.1 (requires protobuf runtime >=6.31.1 per
    protobuf's own cross-version guarantee), while
    ``tensorflow==2.15.0`` (also a transitive dependency of this same repo)
    hard-pins ``protobuf<5.0``. **No protobuf version satisfies both
    constraints simultaneously** -- confirmed by directly attempting
    protobuf 4.25, 5.29, and 6.33 in turn, each failing a different real
    constraint. This is a genuine defect in the upstream OmniVLA/prismatic
    repo's own pinned dependency set (its RLDS/TFDS training-data stack and
    its own bundled HF-format inference code were evidently never tested
    together against the same environment), not an integration mistake on
    this project's side. Real ways forward, none attempted: (a) vendor/copy
    just the 3 needed ``extern/hf/*.py`` files out of the package entirely
    so nothing ever triggers ``prismatic/__init__.py`` or its submodule
    tree, (b) patch the local OmniVLA checkout to lazy-import
    ``tensorflow_datasets`` only inside the functions that actually need it,
    or (c) find/build a newer tensorflow build compatible with protobuf>=6.
    Per this project's explicit direction (2026-08-25), this blocker itself
    -- not a working OmniVLA baseline -- is being recorded as a legitimate
    negative result: OmniVLA's own release is not cleanly integrable as a
    text-generation SFT baseline without upstream fixes this project does
    not own.
    """

    @staticmethod
    def _stub_prismatic_package_init():
        """Skip ``prismatic/__init__.py``'s own import chain.

        Confirmed 2026-08-25: that file does ``from .models import ...``,
        which transitively imports the RLDS/``tensorflow_datasets`` training
        data stack this project never uses (we only need
        ``prismatic.extern.hf.*``, the plain HF-format model/processor
        classes). That chain pulls in a ``tensorflow_metadata``-generated
        protobuf file requiring ``google.protobuf.runtime_version``
        (protobuf>=5.26), while ``tensorflow==2.15.0`` (also pulled in
        transitively) itself pins ``protobuf<5.0`` -- a genuine, irreconcilable
        version conflict in this package's own transitive dependency tree,
        not something this project can `pip install` its way out of.
        Pre-registering an empty stub module for the top-level ``prismatic``
        name (with a real ``__path__`` so Python's finder can still locate
        real submodules on disk) makes ``import prismatic.extern.hf.X`` skip
        running the real, heavy ``__init__.py`` while still correctly
        resolving ``extern``/``extern.hf`` (both genuinely empty
        ``__init__.py`` files) and the actual target submodule.
        """
        import sys
        import types

        if "prismatic" in sys.modules:
            return
        import importlib.util

        spec = importlib.util.find_spec("prismatic")
        if spec is None or spec.submodule_search_locations is None:
            return  # not installed at all; let the real import below fail normally
        stub = types.ModuleType("prismatic")
        stub.__path__ = list(spec.submodule_search_locations)
        sys.modules["prismatic"] = stub

    def load(self):
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

            self._stub_prismatic_package_init()
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from prismatic.extern.hf.processing_prismatic import (
                PrismaticImageProcessor,
                PrismaticProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "the `prismatic` package is not importable in this interpreter -- omnivla_unified "
                "needs its own venv (scripts/setup_omnivla_venv.sh), it cannot share gemma4_unified's "
                "or any other model's venv in this registry"
            ) from exc

        try:
            # Renamed in newer transformers; this adapter's own dedicated
            # venv (setup_omnivla_venv.sh, torch==2.2.0 + the repo's forked
            # transformers) still has the older name.
            from transformers import AutoModelForImageTextToText as AutoModelForVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelForVLM

        try:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVLM.register(OpenVLAConfig, OpenVLAForActionPrediction)
        except ValueError:
            pass  # already registered by an earlier adapter.load() call this process

        hf_id = self.cfg["hf_id"]
        revision = self.cfg.get("revision")
        hub_kwargs: dict[str, Any] = {}
        if revision and not Path(str(hf_id)).expanduser().is_dir():
            hub_kwargs["revision"] = revision
        dtype_name = self.cfg.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")
        self.processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True, **hub_kwargs)
        backbone = AutoModelForVLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype_name),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **hub_kwargs,
        )
        self.model = OmniVLATextWrapper(backbone)
        return self.model, self.processor

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        eos_token = tokenizer.eos_token or "</s>"
        prefix_texts = [f"In: {example['instruction']}\nOut: " for example in examples]
        training_texts = [
            prefix + example["target"] + eos_token for prefix, example in zip(prefix_texts, examples)
        ]
        # single current-frame image only -- the processor's own __call__
        # requires images.shape[0] == text batch size (no per-example image
        # lists), matching this project's num_frames=1 default.
        images = [example["images"][-1] for example in examples]
        common = {"return_tensors": "pt", "padding": True, "truncation": False}
        encoded = self.processor(text=training_texts, images=images, **common)
        prefix = self.processor(text=prefix_texts, images=images, **common)
        if not torch.equal(encoded["pixel_values"], prefix["pixel_values"]):
            raise RuntimeError(
                "full/prefix pixel_values differ -- images must be identical between the "
                "prefix-only and prefix+answer renders"
            )
        return encoded, prefix


@register("internvla_m1_unified")
class InternVLAM1UnifiedAdapter(BaseAdapter):
    """InternVLA-M1 (``InternRobotics/InternVLA-M1``): a Qwen2.5-VL-3B
    backbone co-trained with a separate DINO/QFormer/DiT action-diffusion
    head for manipulation, per the released paper/GitHub repo
    (``InternRobotics/InternVLA-M1`` on GitHub, arXiv 2510.13778).

    Confirmed 2026-08-25 by reading the GitHub repo directly (not guessed):
    ``InternVLA/model/framework/M1.py``'s ``InternVLA_M1.forward()`` only
    ever computes the action-diffusion loss (needs a DINO encoder + a
    layer-wise QFormer + a DiT head, none of which exist in this HF release
    -- see below); the text-capable half lives entirely in
    ``self.qwen_vl_interface``, a thin wrapper (``InternVLA/model/modules/
    vlm/QWen2_5.py``) around a completely standard
    ``Qwen2_5_VLForConditionalGeneration`` + ``AutoProcessor`` pair loaded
    via plain ``from_pretrained(model_id)`` (default
    ``Qwen/Qwen2.5-VL-3B-Instruct``), whose own ``forward()`` just forwards
    ``labels=`` straight into the Qwen backbone -- i.e. standard HF
    cross-entropy training, no custom loss-bypass wrapper needed (unlike
    ``InternVLAN1TextWrapper`` above, which exists precisely because
    InternVLA-N1's raw ``forward()`` does NOT handle plain ``labels=``
    cleanly). ``chat_with_M1()`` in that same file independently confirms
    the inference path is exactly this repo's own
    ``processor.apply_chat_template(...) -> model.generate(**inputs)``
    pattern -- the same shape ``gemma4_unified``/``internvla_n1_unified``
    already use here.

    Critically, the checkpoint published at ``InternRobotics/InternVLA-M1``
    on the HF Hub (confirmed via the HF Hub API's model-info JSON, not
    guessed) is packaged as a **plain, standard `transformers` release**:
    ``architectures: ["Qwen2_5_VLForConditionalGeneration"]``,
    ``library_name: transformers``, ``auto_model:
    AutoModelForMultimodalLM`` -- i.e. it is JUST the (M1-pretraining
    fine-tuned) Qwen2.5-VL weights in safetensors form, with no DINO/
    QFormer/action-head state at all. That means this adapter can load it
    exactly like ``gemma4_unified`` does (plain ``AutoModelForMultimodalLM``
    + ``AutoProcessor``, no need to construct the full GitHub repo's
    ``InternVLA_M1`` class, its DINO encoder, or download the separate
    ``.pt`` checkpoint the GitHub README's demo script loads) -- there is
    nothing to freeze/skip beyond the vision tower, unlike InternVLA-N1's
    frozen System-1 diffusion stack.

    License: ``cc-by-nc-sa-4.0`` (non-commercial) -- fine for research use,
    flag this explicitly if this baseline's numbers go in the paper.

    Module names below (``model.visual``, ``model.language_model.layers.N``)
    are read from the real ``Qwen2_5_VLForConditionalGeneration``/
    ``Qwen2_5_VLModel`` source in an already-installed transformers copy
    (``transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py``:
    ``Qwen2_5_VLModel.__init__`` sets ``self.visual`` and
    ``self.language_model``; ``Qwen2_5_VLForConditionalGeneration.__init__``
    sets ``self.model = Qwen2_5_VLModel(...)`` and ``self.lm_head``) -- NOT
    guessed from InternVLA-N1's differently-nested naming (that model is a
    custom fork that does NOT nest visual/language_model under `.model`,
    this one does). Still needs a live ``named_parameters()`` dump before
    trusting ``expected_parameter_counts`` for a real training run, per this
    project's established lesson that module-name claims from docs/source
    reading alone have been wrong before (see
    ``scripts/verify_internvla_n1_module_names.py`` for the pattern to
    replicate here).
    """

    def load(self):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        hf_id = self.cfg["hf_id"]
        revision = self.cfg.get("revision")
        hub_kwargs: dict[str, Any] = {}
        if revision and not Path(str(hf_id)).expanduser().is_dir():
            hub_kwargs["revision"] = revision
        dtype_name = self.cfg.get("dtype", "bfloat16")
        if not hasattr(torch, dtype_name):
            raise ValueError(f"unsupported torch dtype {dtype_name!r}")
        self.processor = AutoProcessor.from_pretrained(hf_id, **hub_kwargs)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            hf_id,
            dtype=getattr(torch, dtype_name),
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            **hub_kwargs,
        )
        return self.model, self.processor

    @staticmethod
    def _user_conversation(example: dict[str, Any]) -> list[dict[str, Any]]:
        content = [{"type": "image", "image": image} for image in example["images"]]
        content.append({"type": "text", "text": example["instruction"]})
        return [{"role": "user", "content": content}]

    def encode_batch(self, examples: list[dict[str, Any]], max_seq_len: int):
        # Same Qwen-family string-render pattern as internvla_n1_unified
        # above (both are Qwen2.5-VL-tokenizer-family models): `<|im_end|>`
        # is this tokenizer's real eos/turn-end token (confirmed via the HF
        # Hub API's tokenizer_config.json: eos_token="<|im_end|>", unlike
        # Gemma4's differently-named `.eot_token` attribute).
        conversations = [self._user_conversation(example) for example in examples]
        prefix_texts = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        if isinstance(prefix_texts, str):
            prefix_texts = [prefix_texts]
        eot_token = "<|im_end|>\n"
        training_texts = [
            prefix + example["target"] + eot_token
            for prefix, example in zip(prefix_texts, examples)
        ]
        image_batches = [example["images"] for example in examples]
        common = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
            "add_special_tokens": False,
        }
        encoded = self.processor(text=training_texts, images=image_batches, **common)
        prefix = self.processor(text=prefix_texts, images=image_batches, **common)
        for key in ("pixel_values", "image_grid_thw"):
            if key in encoded and key in prefix and not torch.equal(encoded[key], prefix[key]):
                raise RuntimeError(
                    f"full/prefix multimodal tensor differs for {key} -- images must be "
                    "identical between the prefix-only and prefix+answer renders"
                )
        return encoded, prefix
