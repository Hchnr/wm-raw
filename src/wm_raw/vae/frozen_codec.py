"""Frozen VAE codec for latent encoding/decoding.

Wraps the BAGEL autoencoder (or compatible) as a non-trainable encode-only
module for diffusion training. All parameters are frozen (requires_grad=False).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from ..models.embeddings import patchify_latent, unpatchify_latent

logger = logging.getLogger(__name__)


class VAEProtocol(Protocol):
    """Interface that any loaded VAE must satisfy."""

    def encode(self, x: Tensor) -> Tensor: ...


class FrozenCodec(nn.Module):
    """Frozen VAE codec that encodes images to patchified latent tokens.

    Wraps an external VAE model and provides a clean interface for the
    diffusion training loop:
        pixel_images → encode → patchify → [B, num_patches, patch_dim]

    All VAE parameters are frozen; this module has no trainable params.

    Attributes:
        latent_channels: number of channels in VAE latent (e.g., 16)
        latent_h: spatial height of latent grid
        latent_w: spatial width of latent grid
        patch_size: spatial patch size for latent tokenization
        num_latent_tokens: (latent_h / patch_size) * (latent_w / patch_size)
        latent_token_dim: patch_size * patch_size * latent_channels
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        image_size: int = 256,
        latent_channels: int = 16,
        downsample_factor: int = 8,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.image_size = image_size
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.patch_size = patch_size

        # Derived constants
        self.latent_h = image_size // downsample_factor
        self.latent_w = image_size // downsample_factor
        self.num_latent_tokens = (self.latent_h // patch_size) * (self.latent_w // patch_size)
        self.latent_token_dim = patch_size * patch_size * latent_channels

        # Freeze VAE
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """Encode images to patchified latent tokens.

        Args:
            images: [B, 3, H, W] pixel images (normalized to model's expected range)

        Returns:
            tokens: [B, num_latent_tokens, latent_token_dim]
                where num_latent_tokens = (H/downsample/patch)^2
                and latent_token_dim = patch_size^2 * latent_channels
        """
        # VAE encode: [B, 3, H, W] → [B, C, H_lat, W_lat]
        latent = self.vae.encode(images)

        # Handle different VAE output formats
        if hasattr(latent, "latent_dist"):
            latent = latent.latent_dist.sample()
        elif hasattr(latent, "sample"):
            latent = latent.sample()
        elif isinstance(latent, tuple):
            latent = latent[0]

        assert latent.ndim == 4, f"Expected 4D latent, got shape {latent.shape}"
        batch, channels, h, w = latent.shape

        # Flatten spatial: [B, C, H, W] → [B, H*W, C]
        tokens_flat = latent.permute(0, 2, 3, 1).reshape(batch, h * w, channels)

        # Patchify: [B, H*W, C] → [B, num_patches, patch_dim]
        tokens = patchify_latent(
            tokens_flat,
            height=h,
            width=w,
            patch_size=self.patch_size,
        )
        return tokens

    @torch.no_grad()
    def decode(self, tokens: Tensor) -> Tensor:
        """Decode patchified latent tokens back to pixel images.

        Args:
            tokens: [B, num_latent_tokens, latent_token_dim]

        Returns:
            images: [B, 3, H, W] reconstructed images
        """
        # Unpatchify: [B, num_patches, patch_dim] → [B, H*W, C]
        tokens_flat = unpatchify_latent(
            tokens,
            height=self.latent_h,
            width=self.latent_w,
            channels=self.latent_channels,
            patch_size=self.patch_size,
        )

        # Reshape to spatial: [B, H*W, C] → [B, C, H, W]
        batch = tokens_flat.shape[0]
        latent = (
            tokens_flat.reshape(batch, self.latent_h, self.latent_w, self.latent_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # VAE decode
        if hasattr(self.vae, "decode"):
            decoded = self.vae.decode(latent)
            if hasattr(decoded, "sample"):
                return decoded.sample
            return decoded
        raise NotImplementedError("VAE does not have a decode method")

    def forward(self, images: Tensor) -> Tensor:
        """Alias for encode (default forward is encode for training)."""
        return self.encode(images)

    def train(self, mode: bool = True) -> "FrozenCodec":
        """Override to keep VAE always in eval mode."""
        super().train(mode)
        self.vae.eval()
        return self


def load_frozen_codec(
    vae_path: str | Path,
    *,
    image_size: int = 256,
    latent_channels: int = 16,
    downsample_factor: int = 8,
    patch_size: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> FrozenCodec:
    """Load a frozen VAE codec from checkpoint path.

    Tries loading in order:
    1. BAGEL autoencoder (wm_training.models.bagel.autoencoder)
    2. diffusers AutoencoderKL

    Args:
        vae_path: path to .safetensors file or HF model dir
        image_size: input image spatial size
        latent_channels: VAE latent channels
        downsample_factor: spatial downsample ratio (pixel → latent)
        patch_size: latent patch size for tokenization
        device: target device
        dtype: target dtype
    """
    vae_path = Path(vae_path)

    # Try BAGEL autoencoder first
    try:
        from wm_training.models.bagel.autoencoder import load_ae

        vae = load_ae(str(vae_path))
        vae = vae.to(device=device, dtype=dtype)
        logger.info("Loaded BAGEL autoencoder from %s", vae_path)
    except ImportError:
        # Fallback: diffusers AutoencoderKL
        try:
            from diffusers import AutoencoderKL

            model_path = str(vae_path.parent) if vae_path.is_file() else str(vae_path)
            vae = AutoencoderKL.from_pretrained(model_path, torch_dtype=dtype)
            vae = vae.to(device=device)
            logger.info("Loaded diffusers AutoencoderKL from %s", model_path)
        except (ImportError, Exception) as e:
            raise RuntimeError(
                f"Failed to load VAE from {vae_path}. "
                "Install wm_training or diffusers to load the autoencoder."
            ) from e

    return FrozenCodec(
        vae,
        image_size=image_size,
        latent_channels=latent_channels,
        downsample_factor=downsample_factor,
        patch_size=patch_size,
    )
