"""Collators for VLM and Diffusion training batches.

These take raw dataset dicts and produce model-ready tensor batches.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


def _pil_to_vae_tensor(image: Any, *, image_size: int) -> Tensor:
    """Convert PIL Image to VAE input tensor: [3, H, W] in [-1, 1]."""
    from PIL import Image, ImageOps

    image = ImageOps.fit(
        image.convert("RGB"),
        (image_size, image_size),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    raw = raw.view(image_size, image_size, 3).permute(2, 0, 1).float()
    return raw.div(127.5).sub(1.0)


@dataclass(frozen=True)
class VLMCollator:
    """Collate examples for VLM (autoregressive captioning) training.

    Uses Qwen3-VL processor to tokenize image+text into model inputs.
    Produces:
        input_ids, attention_mask, position_ids, pixel_values,
        image_grid_thw, labels
    """

    processor: Any
    prompt: str = "Describe this image in detail."
    label_ignore_index: int = -100
    max_seq_len: int | None = None

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("VLM collator received empty batch")

        messages_batch = []
        images = []
        for ex in examples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["caption"]}],
                },
            ]
            messages_batch.append(messages)
            images.append(ex["image"])

        # Apply chat template for full text (with answer)
        full_texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_batch
        ]
        # Apply template for prompt only (to compute label mask)
        prompt_messages = [
            [msgs[0]] for msgs in messages_batch  # user turn only
        ]
        prompt_texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in prompt_messages
        ]

        # Tokenize
        kwargs: dict[str, Any] = {
            "text": full_texts,
            "images": images,
            "padding": "max_length" if self.max_seq_len else True,
            "return_tensors": "pt",
        }
        if self.max_seq_len:
            kwargs["max_length"] = self.max_seq_len
            kwargs["truncation"] = True
        batch = dict(self.processor(**kwargs))

        # Build labels: mask prompt tokens
        labels = batch["input_ids"].clone()
        for row, (pt, img) in enumerate(zip(prompt_texts, images)):
            prompt_enc = self.processor(
                text=[pt], images=[img], padding=False, return_tensors="pt"
            )
            prompt_len = min(int(prompt_enc["input_ids"].shape[-1]), labels.shape[1])
            labels[row, :prompt_len] = self.label_ignore_index

        # Mask padding
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(~attention_mask.bool(), self.label_ignore_index)
        batch["labels"] = labels
        return batch


@dataclass(frozen=True)
class DiffusionCollator:
    """Collate examples for diffusion training.

    Produces:
        condition: dict with input_ids, attention_mask (text condition)
        vae_pixel_values: [B, 3, H, W] in [-1, 1] for VAE encoding
    """

    processor: Any
    image_size: int = 256
    condition_prefix: str = "Caption: "
    condition_suffix: str = " <|wm_predict_image|>"
    text_condition_dropout_prob: float = 0.0
    condition_max_seq_len: int | None = None

    def _conditional_text(self, caption: str) -> str:
        return self.condition_prefix + caption + self.condition_suffix

    def _unconditional_text(self) -> str:
        return self.condition_suffix.strip()

    def __call__(self, examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not examples:
            raise ValueError("Diffusion collator received empty batch")

        # Build condition texts (with possible CFG dropout)
        texts = []
        drop_mask = []
        for ex in examples:
            should_drop = random.random() < self.text_condition_dropout_prob
            drop_mask.append(should_drop)
            if should_drop:
                texts.append(self._unconditional_text())
            else:
                texts.append(self._conditional_text(ex["caption"]))

        # Tokenize condition
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tok_kwargs: dict[str, Any] = {
            "padding": "max_length" if self.condition_max_seq_len else True,
            "return_tensors": "pt",
        }
        if self.condition_max_seq_len:
            tok_kwargs["max_length"] = self.condition_max_seq_len
            tok_kwargs["truncation"] = True
        condition = dict(tokenizer(texts, **tok_kwargs))

        # Prepare VAE pixel values
        vae_pixel_values = torch.stack([
            _pil_to_vae_tensor(ex["image"], image_size=self.image_size)
            for ex in examples
        ])

        return {
            "condition": condition,
            "vae_pixel_values": vae_pixel_values,
            "condition_dropped_mask": torch.tensor(drop_mask, dtype=torch.bool),
        }
