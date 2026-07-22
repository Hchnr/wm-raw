"""AdaLN-Zero timestep conditioning for diffusion decoder layers.

Components:
- SinusoidalTimestepEmbedding: maps scalar timestep → hidden vector
- AdaLNZero: produces per-layer modulation params (shift, scale, gate) × 2
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP → hidden vector.

    Maps scalar timesteps to dense vectors using sinusoidal frequencies
    followed by a 2-layer MLP with SiLU activation.
    """

    def __init__(self, frequency_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        """Embed timesteps into hidden dimension.

        Args:
            timesteps: [B] scalar timestep values

        Returns:
            embedding: [B, hidden_size]
        """
        timesteps = timesteps.float().reshape(-1)  # [B]
        half_dim = self.frequency_dim // 2

        # Exponential frequency spacing: exp(-log(10000) * i / (half-1))
        exponent = -math.log(10000.0) * torch.arange(
            max(half_dim, 1), device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        frequencies = timesteps[:, None] * exponent.exp()[None]  # [B, half_dim]

        # Concat sin and cos, truncate to frequency_dim
        embedding = torch.cat([frequencies.cos(), frequencies.sin()], dim=-1)
        embedding = embedding[:, : self.frequency_dim]  # [B, frequency_dim]

        # MLP projection
        embedding = embedding.to(
            device=self.mlp[0].weight.device,
            dtype=self.mlp[0].weight.dtype,
        )
        return self.mlp(embedding)  # [B, hidden_size]


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero modulation for diffusion decoder layers.

    From the timestep embedding, produces 6 modulation parameters per layer:
    (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)

    Initialized to zero so that at the start of training, the diffusion
    layers behave as identity (no modulation).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init the linear layer for identity-like behavior at init
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, time_hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Produce AdaLN modulation parameters from timestep embedding.

        Args:
            time_hidden: [B, D] timestep embedding

        Returns:
            Tuple of 6 tensors, each [B, D]:
            (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        """
        params = self.modulation(time_hidden.to(self.modulation[-1].weight.dtype))  # [B, 6*D]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)
        return shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
