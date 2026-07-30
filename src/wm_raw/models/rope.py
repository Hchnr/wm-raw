"""Rotary Position Embedding implementations for Qwen3-VL.

Two variants:
- VisionRotaryEmbedding: 2D rotary for vision encoder (simple, theta=10000)
- TextMRoPE: 3D multi-resolution rotary for text decoder (temporal, height, width)

All implementations are fully static — no .item(), .tolist(), or dynamic shapes.
Position IDs are passed as pre-computed tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs: [-x2, x1] from [x1, x2]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,  # [B, H, S, D]
    k: Tensor,  # [B, H, S, D]
    cos: Tensor,  # [B, S, D] or [B, 1, S, D]
    sin: Tensor,  # [B, S, D] or [B, 1, S, D]
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # Broadcast cos/sin to [B, 1, S, D] for head dimension
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)  # [B, 1, S, D]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_query_only(
    q: Tensor,  # [B, H, S, D]
    cos: Tensor,  # [B, S, D] or [B, 1, S, D]
    sin: Tensor,  # [B, S, D] or [B, 1, S, D]
) -> Tensor:
    """Apply rotary position embedding to query only (for cross-attention)."""
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


class VisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding for vision encoder.

    Position IDs are flattened spatial positions. The embedding is a simple
    sinusoidal rotation with no multi-resolution structure.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor) -> Tensor:
        """Compute rotary frequencies for vision positions.

        Args:
            position_ids: [N] integer position indices

        Returns:
            freqs: [N, dim] — to be used as (cos(freqs), sin(freqs))
        """
        # position_ids: [N] → [N, 1]
        # inv_freq: [dim//2] → [1, dim//2]
        # result: [N, dim//2]
        return position_ids.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)


class TextMRoPE(nn.Module):
    """Multi-resolution Rotary Position Embedding for Qwen3-VL text decoder.

    Produces interleaved 3D rope: temporal, height, width dimensions are
    assigned to different slices of the head dimension via mrope_section.

    Unlike the HF implementation, this never calls .item() or .tolist().
    Position IDs must be pre-computed and passed in as [3, B, S] tensors.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 500000.0,
        mrope_section: tuple[int, ...] = (24, 20, 20),
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        # inv_freq: [head_dim // 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: Tensor,  # [3, B, S] — temporal, height, width
    ) -> tuple[Tensor, Tensor]:
        """Compute cos/sin embeddings with interleaved MRoPE.

        Args:
            position_ids: [3, B, S] integer positions for 3 axes

        Returns:
            cos: [B, S, head_dim]
            sin: [B, S, head_dim]
        """
        # inv_freq: [D/2], expand to [3, B, D/2, 1] for matmul with positions
        # position_ids: [3, B, S] → [3, B, 1, S] for matmul
        inv_freq = self.inv_freq.float()  # [D/2]
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )  # [3, B, D/2, 1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, S]

        # freqs: [3, B, S, D/2]
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        # Apply interleaved MRoPE: assign each axis to its section of head_dim
        freqs = self._apply_interleaved_mrope(freqs)  # [B, S, D/2]

        # Double for cos/sin pair
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

    def _apply_interleaved_mrope(self, freqs: Tensor) -> Tensor:
        """Reorganize from [3, B, S, D/2] to interleaved [B, S, D/2].

        HF Qwen3-VL interleave: assigns axes to frequency indices as:
          index 0,3,6,...  → temporal (axis 0)
          index 1,4,7,...  → height  (axis 1)
          index 2,5,8,...  → width   (axis 2)

        Implementation: start with temporal freqs for ALL dims, then overwrite
        height/width at their interleaved positions.
        """
        # Start with temporal as base (covers all dims including overflow)
        result = freqs[0].clone()  # [B, S, D/2]

        # Overwrite height dims: indices 1, 4, 7, ... up to mrope_section[1]*3
        h_length = self.mrope_section[1] * 3
        h_indices = list(range(1, h_length, 3))
        for idx in h_indices:
            if idx < result.shape[-1]:
                result[..., idx] = freqs[1, ..., idx]

        # Overwrite width dims: indices 2, 5, 8, ... up to mrope_section[2]*3
        w_length = self.mrope_section[2] * 3
        w_indices = list(range(2, w_length, 3))
        for idx in w_indices:
            if idx < result.shape[-1]:
                result[..., idx] = freqs[2, ..., idx]

        return result
