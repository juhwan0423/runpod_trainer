"""Lazy JSONL dataset and multimodal SFT collator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.schema import ACTION_LABEL_SET, PROMPT_VERSION, read_jsonl, render_policy_prompt


def derive_target_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_input_ids: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Locate assistant targets by exact prompt-prefix token comparison.

    Native assistant character-offset masks have been unreliable for expanded
    multimodal placeholders in some Transformers releases.  Tokenizing the
    identical image+user conversation once with a generation prompt and once
    with the gold assistant answer avoids that offset path.  Any template,
    truncation, or padding mismatch fails closed.
    """

    if input_ids.ndim != 2 or prefix_input_ids.ndim != 2:
        raise RuntimeError("chat template input_ids must be rank-2 batched tensors")
    if input_ids.shape != attention_mask.shape:
        raise RuntimeError("full input_ids/attention_mask shapes differ")
    if prefix_input_ids.shape != prefix_attention_mask.shape:
        raise RuntimeError("prefix input_ids/attention_mask shapes differ")
    if input_ids.shape[0] != prefix_input_ids.shape[0]:
        raise RuntimeError("full and prefix chat batches have different sizes")

    target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row_index in range(input_ids.shape[0]):
        full_positions = torch.nonzero(attention_mask[row_index].bool()).flatten()
        prefix_positions = torch.nonzero(prefix_attention_mask[row_index].bool()).flatten()
        full_tokens = input_ids[row_index, full_positions]
        prefix_tokens = prefix_input_ids[row_index, prefix_positions]
        if len(full_tokens) <= len(prefix_tokens):
            raise RuntimeError(
                f"assistant target is empty or truncated for batch row {row_index}: "
                f"full={len(full_tokens)} prefix={len(prefix_tokens)}"
            )
        if not torch.equal(full_tokens[: len(prefix_tokens)], prefix_tokens):
            mismatch = torch.nonzero(full_tokens[: len(prefix_tokens)] != prefix_tokens).flatten()
            first = int(mismatch[0]) if len(mismatch) else -1
            raise RuntimeError(
                "full conversation does not begin with the tokenized generation prefix "
                f"for batch row {row_index}; first_mismatch={first}"
            )
        target_mask[row_index, full_positions[len(prefix_tokens) :]] = True
    return target_mask


def select_history(images: list[str], num_frames: int) -> list[str]:
    """Select ordered frames including the current (last) observation."""
    if not images:
        raise ValueError("SFT row has no images")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames == 1:
        return [images[-1]]
    if len(images) <= num_frames:
        return images
    indices = [round(i * (len(images) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [images[index] for index in indices]


class InspectionSFTDataset(Dataset):
    def __init__(self, jsonl: str | Path):
        self.path = Path(jsonl)
        if not self.path.is_file():
            raise FileNotFoundError(f"SFT split does not exist: {self.path}")
        self.rows = read_jsonl(self.path)
        if not self.rows:
            raise ValueError(f"SFT split is empty: {self.path}")
        for index, row in enumerate(self.rows):
            if row.get("action_label") not in ACTION_LABEL_SET:
                raise ValueError(f"{self.path}: row {index} has invalid action_label")
            if not row.get("instruction"):
                raise ValueError(f"{self.path}: row {index} has no instruction")
            if not (row.get("images") or row.get("image")):
                raise ValueError(f"{self.path}: row {index} has no image reference")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class MultimodalSFTCollator:
    """Render native chat prompts in a worker and mask every non-answer token."""

    def __init__(self, adapter, image_root: str | Path, max_seq_len: int, num_frames: int):
        self.adapter = adapter
        self.image_root = Path(image_root)
        self.max_seq_len = max_seq_len
        self.num_frames = num_frames
        self._mask_contract_validated = False

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        examples = []
        for row in rows:
            images = row.get("images") or [row["image"]]
            selected = select_history(list(images), self.num_frames)
            loaded_images = []
            for relative in selected:
                path = Path(relative)
                path = path if path.is_absolute() else self.image_root / path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"missing training image {path} (frame_id={row.get('frame_id')})"
                    )
                with Image.open(path) as image:
                    loaded_images.append(image.convert("RGB").copy())
            examples.append(
                {
                    "images": loaded_images,
                    "instruction": render_policy_prompt(row["instruction"], len(loaded_images)),
                    "target": row["action_label"],
                    "prompt_version": PROMPT_VERSION,
                }
            )
        encoded, prefix = self.adapter.encode_batch(examples, self.max_seq_len)
        if "attention_mask" not in encoded or "attention_mask" not in prefix:
            raise RuntimeError("processor must return attention_mask for target-only SFT")
        assistant_mask = derive_target_mask(
            encoded["input_ids"],
            encoded["attention_mask"],
            prefix["input_ids"],
            prefix["attention_mask"],
        )
        active_lengths = encoded["attention_mask"].sum(dim=1)
        if torch.any(active_lengths > self.max_seq_len):
            bad = torch.nonzero(active_lengths > self.max_seq_len).flatten().tolist()
            lengths = [int(active_lengths[index]) for index in bad]
            raise ValueError(
                f"tokenized samples exceed data.max_seq_len={self.max_seq_len}: "
                f"batch_rows={bad} active_lengths={lengths}"
            )
        labels = encoded["input_ids"].clone()
        labels[~assistant_mask] = -100
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = -100
        target_counts = (labels != -100).sum(dim=1)
        if torch.any(target_counts == 0):
            bad = torch.nonzero(target_counts == 0).flatten().tolist()
            raise ValueError(
                f"assistant target was truncated or unmarked for batch rows {bad}; "
                "increase data.max_seq_len or inspect the model chat template"
            )
        if torch.any(target_counts > 64):
            bad = torch.nonzero(target_counts > 64).flatten().tolist()
            raise RuntimeError(
                f"assistant suffix unexpectedly exceeds 64 tokens for batch rows {bad}; "
                "refusing to supervise possible prompt/template content"
            )
        if not self._mask_contract_validated:
            tokenizer = getattr(self.adapter.processor, "tokenizer", self.adapter.processor)
            for index, example in enumerate(examples):
                supervised = labels[index][labels[index] != -100]
                decoded = tokenizer.decode(supervised, skip_special_tokens=True)
                if decoded.strip() != example["target"]:
                    raise RuntimeError(
                        "prefix-derived assistant mask does not decode to exactly the target label: "
                        f"target={example['target']!r} decoded_supervision={decoded!r}"
                    )
            self._mask_contract_validated = True
        encoded["labels"] = labels
        return encoded
