"""Latent tokenization and position embedding utilities.

Handles:
- Patchification of VAE latent grids into token sequences
- 2D sinusoidal position embeddings for latent tokens
- Continuous token projection (LayerNorm + Linear)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def patchify_latent(
    tokens: Tensor,  # [B, H*W, C]
    *,
    height: int,
    width: int,
    patch_size: int,
) -> Tensor:
    """Patchify flat VAE latent grid into larger tokens.

    Reshapes [B, H*W, C] → [B, (H/P)*(W/P), P*P*C] where P = patch_size.
    Patch ordering follows BAGEL convention: row-major patches, within each
    patch the order is [row, col, channel].

    Args:
        tokens: [B, H*W, C] — flat row-major VAE latents
        height: spatial height of latent grid
        width: spatial width of latent grid
        patch_size: spatial patch size (P)

    Returns:
        patchified: [B, num_patches, patch_dim] where
            num_patches = (H/P) * (W/P)
            patch_dim = P * P * C
    """
    if patch_size == 1:
        return tokens

    batch, seq_len, channels = tokens.shape
    assert seq_len == height * width, f"tokens length {seq_len} != {height}*{width}"
    assert height % patch_size == 0 and width % patch_size == 0

    patch_h = height // patch_size
    patch_w = width // patch_size

    # [B, H*W, C] → [B, H, W, C]
    grid = tokens.reshape(batch, height, width, channels)
    # [B, H, W, C] → [B, pH, P, pW, P, C]
    grid = grid.reshape(batch, patch_h, patch_size, patch_w, patch_size, channels)
    # [B, pH, pW, P, P, C] → [B, pH*pW, P*P*C]
    patchified = (
        grid.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, patch_h * patch_w, patch_size * patch_size * channels)
    )
    return patchified


def unpatchify_latent(
    tokens: Tensor,  # [B, num_patches, patch_dim]
    *,
    height: int,
    width: int,
    channels: int,
    patch_size: int,
) -> Tensor:
    """Inverse of patchify_latent: [B, pH*pW, P*P*C] → [B, H*W, C]."""
    if patch_size == 1:
        return tokens

    batch = tokens.shape[0]
    patch_h = height // patch_size
    patch_w = width // patch_size

    # [B, pH*pW, P*P*C] → [B, pH, pW, P, P, C]
    grid = tokens.reshape(batch, patch_h, patch_w, patch_size, patch_size, channels)
    # [B, pH, pW, P, P, C] → [B, pH, P, pW, P, C] → [B, H, W, C] → [B, H*W, C]
    result = (
        grid.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, height * width, channels)
    )
    return result


class Sincos2DPositionEmbedding(nn.Module):
    """Fixed 2D sinusoidal position embedding for patchified latent grids.

    Given a grid of (grid_h, grid_w) patches, produces a fixed embedding
    table of shape [grid_h * grid_w, embed_dim]. The embedding is computed
    once at init and stored as a buffer (no learnable parameters).
    """

    def __init__(self, grid_h: int, grid_w: int, embed_dim: int) -> None:
        super().__init__()
        pe = self._build_2d_sincos(grid_h, grid_w, embed_dim)
        self.register_buffer("pe", pe, persistent=False)  # [grid_h*grid_w, embed_dim]

    def forward(self, batch_size: int) -> Tensor:
        """Return position embeddings expanded for batch.

        Returns:
            pos_embed: [B, num_positions, embed_dim]
        """
        return self.pe.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _build_2d_sincos(grid_h: int, grid_w: int, embed_dim: int) -> Tensor:
        """Build 2D sinusoidal embeddings.

        Split embed_dim evenly between height and width. Each half uses
        sinusoidal encoding along its respective axis.
        """
        assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos"
        half_dim = embed_dim // 2

        # Height positions: [grid_h, 1] × freq [1, half_dim//2]
        pos_h = torch.arange(grid_h, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(grid_w, dtype=torch.float32).unsqueeze(1)

        dim_h = half_dim // 2
        dim_w = half_dim // 2
        omega_h = 1.0 / (10000.0 ** (torch.arange(dim_h, dtype=torch.float32) / dim_h))
        omega_w = 1.0 / (10000.0 ** (torch.arange(dim_w, dtype=torch.float32) / dim_w))

        # [grid_h, dim_h] → sin/cos → [grid_h, half_dim]
        pe_h = torch.cat([
            (pos_h * omega_h).sin(),
            (pos_h * omega_h).cos(),
        ], dim=-1)  # [grid_h, half_dim]

        pe_w = torch.cat([
            (pos_w * omega_w).sin(),
            (pos_w * omega_w).cos(),
        ], dim=-1)  # [grid_w, half_dim]

        # Outer product: [grid_h, grid_w, embed_dim]
        # Each position (i, j) gets concat of pe_h[i] and pe_w[j]
        pe_h_expanded = pe_h.unsqueeze(1).expand(-1, grid_w, -1)  # [H, W, half]
        pe_w_expanded = pe_w.unsqueeze(0).expand(grid_h, -1, -1)  # [H, W, half]
        pe = torch.cat([pe_h_expanded, pe_w_expanded], dim=-1)  # [H, W, embed_dim]

        return pe.reshape(grid_h * grid_w, embed_dim)


class ContinuousTokenProjector(nn.Module):
    """Projects continuous-valued tokens into model hidden dimension.

    When normalize_input=True: LayerNorm → Linear.
    When normalize_input=False: Linear only (+ input_projection_version buffer).
    """

    def __init__(
        self, input_dim: int, hidden_size: int, normalize_input: bool = True
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim) if normalize_input else None
        self.proj = nn.Linear(input_dim, hidden_size)
        if not normalize_input:
            self.register_buffer(
                "input_projection_version",
                torch.tensor(2, dtype=torch.int64),
                persistent=True,
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """Project tokens: [B, S, input_dim] → [B, S, hidden_size]."""
        if self.norm is not None:
            tokens = self.norm(tokens)
        return self.proj(tokens)
