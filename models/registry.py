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
