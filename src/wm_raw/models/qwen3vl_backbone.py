"""Qwen3-VL text decoder building blocks.

Self-contained implementation of:
- RMSNorm
- TextAttention (GQA + QK-norm + MRoPE)
- TextMLP (SwiGLU)
- DecoderLayer (pre-norm transformer block)

All shapes are explicitly annotated. No dynamic dispatch or HF model inheritance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x: [*, D]
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class TextAttention(nn.Module):
    """Multi-head attention with GQA and QK-norm for Qwen3-VL text decoder.

    Supports optional external K/V concatenation for cross_kv_down
    communication policy — external keys and values are prepended to the
    self-attention K/V before the attention computation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # QK-norm (per-head RMSNorm on head_dim dimension)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,  # [B, S, D]
        position_embeddings: tuple[Tensor, Tensor],  # (cos, sin): each [B, S, head_dim]
        attention_mask: Optional[Tensor] = None,  # [B, 1, S, S] or [B, 1, S, S+K]
        external_kv: Optional[tuple[Tensor, Tensor]] = None,  # ([B, H_kv, K, D], [B, H_kv, K, D])
    ) -> Tensor:
        """
        Args:
            hidden_states: input features [B, S, D]
            position_embeddings: (cos, sin) from TextMRoPE, each [B, S, head_dim]
            attention_mask: causal mask [B, 1, S, total_kv_len]
            external_kv: optional pre-projected K/V to prepend (for cross_kv_down)

        Returns:
            output: [B, S, D]
        """
        batch, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)  # [B, S, H*D]
        k = self.k_proj(hidden_states)  # [B, S, Hkv*D]
        v = self.v_proj(hidden_states)  # [B, S, Hkv*D]

        # Reshape to [B, H, S, D]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-norm (on head_dim, i.e. last dimension)
        q = self.q_norm(q)  # [B, H, S, D]
        k = self.k_norm(k)  # [B, Hkv, S, D]

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepend external K/V for cross_kv_concat
        if external_kv is not None:
            ext_k, ext_v = external_kv  # [B, Hkv, K, D]
            # Apply k_norm to external K (matches online: k_norm normalizes both
            # native and external keys before they share one softmax)
            ext_k = self.k_norm(ext_k)
            k = torch.cat([ext_k, k], dim=2)  # [B, Hkv, K+S, D]
            v = torch.cat([ext_v, v], dim=2)  # [B, Hkv, K+S, D]

        # Expand KV for GQA: [B, Hkv, L, D] → [B, H, L, D]
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        # Use is_causal=True when no explicit mask and no external KV (VLM causal path)
        use_causal = attention_mask is None and external_kv is None
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=use_causal,
            scale=self.scaling,
        )  # [B, H, S, D]

        # Merge heads: [B, H, S, D] → [B, S, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class TextMLP(nn.Module):
    """SwiGLU MLP for Qwen3-VL text decoder."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Single Qwen3-VL text decoder layer (pre-norm transformer block).

    Supports optional external K/V injection for cross-attention via the
    cross_kv_down policy (concatenated before self-attention keys/values).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = TextAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = TextMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,  # [B, S, D]
        position_embeddings: tuple[Tensor, Tensor],  # (cos, sin)
        attention_mask: Optional[Tensor] = None,  # [B, 1, S, S+K]
        external_kv: Optional[tuple[Tensor, Tensor]] = None,  # cross-attn K/V
    ) -> Tensor:
        """Forward pass. Returns [B, S, D]."""
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            external_kv=external_kv,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
