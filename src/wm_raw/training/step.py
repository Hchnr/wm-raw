"""Single training step logic."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from ..models import WorldModel
from .config import TrainingConfig

logger = logging.getLogger(__name__)


class FrozenVAECodec:
    """Frozen VAE encoder for converting pixels → latent tokens."""

    def __init__(self, ae: nn.Module, device: torch.device, dtype: torch.dtype):
        self.ae = ae
        self.device = device
        self.dtype = dtype
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, pixel_values: Tensor) -> Tensor:
        """Encode [B, 3, H, W] → [B, H*W, C] flat latent tokens."""
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        # Standard VAE encode: returns [B, C, h, w]
        latent = self.ae.encode(pixel_values)
        if hasattr(latent, "latent_dist"):
            latent = latent.latent_dist.sample()
        elif hasattr(latent, "sample"):
            latent = latent.sample()
        # Reshape [B, C, h, w] → [B, h*w, C]
        b, c, h, w = latent.shape
        return latent.permute(0, 2, 3, 1).reshape(b, h * w, c)


def train_step(
    model: WorldModel,
    batch: dict[str, Any],
    *,
    codec: FrozenVAECodec,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, float]:
    """One training step: forward + backward + optimizer step.

    Returns dict of scalar metrics.
    """
    optimizer.zero_grad(set_to_none=True)

    # Encode images to latents
    vae_pixels = batch["vae_pixel_values"].to(device=device)
    state_target = codec.encode(vae_pixels)  # [B, H*W, C]

    # Derive latent spatial dimensions from VAE output
    # VAE downsamples by 8x: 512px → 64 latent, 448px → 56, etc.
    # state_target shape is [B, H*W, C] — we need H and W separately
    # For fixed image_size: H = W = image_size / 8
    # For variable (buckets): get from batch metadata
    if "image_height" in batch and "image_width" in batch:
        latent_h = batch["image_height"] // 8
        latent_w = batch["image_width"] // 8
    else:
        # Infer from state_target: assume square if not specified
        num_latent_pixels = state_target.shape[1]
        latent_side = int(num_latent_pixels ** 0.5)
        assert latent_side * latent_side == num_latent_pixels, (
            f"Cannot infer latent dimensions from {num_latent_pixels} pixels. "
            f"Pass image_height/image_width in batch."
        )
        latent_h = latent_w = latent_side

    # Condition tokens
    condition = {k: v.to(device=device) for k, v in batch["condition"].items() if torch.is_tensor(v)}

    # Build 3D position IDs for MRoPE
    seq_len = condition["input_ids"].shape[1]
    batch_size = condition["input_ids"].shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]

    # Build causal attention mask [B, 1, S, S]
    # Start from padding mask [B, S] (1 = attend, 0 = ignore)
    padding_mask = condition.get("attention_mask")
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=compute_dtype),
        diagonal=1,
    )  # [S, S] upper-triangular = -inf
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)  # [B, 1, S, S]
    if padding_mask is not None and not padding_mask.all():
        # Mask out padded positions in the key dimension
        # Use where() to avoid -inf * 0 = nan
        pad_expanded = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        causal_mask = causal_mask.clone()
        causal_mask.masked_fill_(~pad_expanded, float("-inf"))

    # Forward with autocast
    with torch.amp.autocast("cuda", dtype=compute_dtype):
        output = model({
            "task_type": "diffusion",
            "condition": {
                "input_ids": condition["input_ids"],
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            },
            "state_target": state_target.to(dtype=compute_dtype),
            "latent_h": latent_h,
            "latent_w": latent_w,
        })

    loss = output.diffusion_loss * config.diffusion_loss_weight
    loss.backward()

    # Gradient clipping
    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )

    optimizer.step()
    scheduler.step()

    return {
        "loss": loss.item(),
        "diffusion_loss": output.diffusion_loss.item(),
        "lr": scheduler.get_last_lr()[0],
    }
