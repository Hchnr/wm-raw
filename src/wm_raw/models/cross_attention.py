"""Cross-Attention from VLM hidden states to diffusion branch.

Implements the `cross_kv_down` communication policy:
- Per diffusion layer: project VLM context → K, V (smaller dimension)
- Diffusion hidden → Q (in diffusion's own dimension)
- Standard cross-attention + gated residual

Layer mapping determines which VLM layer's hidden state feeds each
diffusion layer's cross-attention.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .qwen3vl_backbone import RMSNorm


class CrossAttentionAdapter(nn.Module):
    """Single cross-attention layer: diffusion queries attend to VLM context.

    Architecture (cross_kv_down):
        Q = q_proj(norm_q(diffusion_hidden))       → [B, H_q, S_diff, D]
        K = k_proj(norm_ctx(vlm_context))          → [B, H_kv, S_vlm, D]
        V = v_proj(norm_ctx(vlm_context))          → [B, H_kv, S_vlm, D]
        out = gate * o_proj(attention(Q, K, V))
    """

    def __init__(
        self,
        diffusion_hidden_size: int,
        vlm_hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        gate_init: float = 0.01,
        zero_init_output: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_query_heads // num_kv_heads

        query_width = num_query_heads * head_dim
        kv_width = num_kv_heads * head_dim

        # Norms
        self.query_norm = RMSNorm(diffusion_hidden_size, eps=norm_eps)
        self.context_norm = RMSNorm(vlm_hidden_size, eps=norm_eps)

        # Projections
        self.q_proj = nn.Linear(diffusion_hidden_size, query_width, bias=False)
        self.k_proj = nn.Linear(vlm_hidden_size, kv_width, bias=False)
        self.v_proj = nn.Linear(vlm_hidden_size, kv_width, bias=False)
        self.o_proj = nn.Linear(query_width, diffusion_hidden_size, bias=False)

        # Gating (tanh-gated residual)
        self.gate = nn.Parameter(torch.tensor([float(gate_init)], dtype=torch.float32))

        if zero_init_output:
            nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        diffusion_hidden: Tensor | None = None,  # [B, S_diff, D_diff]
        context_hidden: Tensor | None = None,  # [B, S_vlm, D_vlm]
        attention_mask: Tensor | None = None,  # [B, 1, S_diff, S_vlm]
        *,
        mode: str = "full",  # "full" | "kv_only"
        target_device: torch.device | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Cross-attend from diffusion to VLM context.

        Modes:
            "full": full cross-attention (requires diffusion_hidden + context_hidden).
                Returns: [B, S_diff, D_diff] — gated residual added to input.
            "kv_only": project context into K/V only (requires context_hidden).
                Returns: (k, v) with shape [B, H_kv, S_vlm, D].
        """
        if mode == "kv_only":
            assert context_hidden is not None
            if target_device is not None or target_dtype is not None:
                context_hidden = context_hidden.to(
                    device=target_device or context_hidden.device,
                    dtype=target_dtype or context_hidden.dtype,
                )
            kv_input = self.context_norm(context_hidden)
            batch, seq_vlm, _ = kv_input.shape

            k = self.k_proj(kv_input.to(self.k_proj.weight.dtype))
            v = self.v_proj(kv_input.to(self.v_proj.weight.dtype))
            k = k.view(batch, seq_vlm, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_vlm, self.num_kv_heads, self.head_dim).transpose(1, 2)
            return k, v

        # mode == "full"
        assert diffusion_hidden is not None and context_hidden is not None
        residual = diffusion_hidden
        batch, seq_diff, _ = diffusion_hidden.shape
        _, seq_vlm, _ = context_hidden.shape

        # Normalize
        q_input = self.query_norm(diffusion_hidden)
        kv_input = self.context_norm(context_hidden)

        # Project
        q = self.q_proj(q_input)  # [B, S_diff, H_q * D]
        k = self.k_proj(kv_input)  # [B, S_vlm, H_kv * D]
        v = self.v_proj(kv_input)  # [B, S_vlm, H_kv * D]

        # Reshape to [B, H, S, D]
        q = q.view(batch, seq_diff, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_vlm, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_vlm, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )  # [B, H_q, S_diff, D]

        # Merge heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch, seq_diff, -1)
        )  # [B, S_diff, H_q*D]

        # Output projection + gated residual
        output = self.o_proj(attn_output)
        gate = torch.tanh(self.gate).to(output.dtype)
        return residual + gate * output

    def project_context_kv(
        self,
        context_hidden: Tensor,  # [B, S_vlm, D_vlm]
        *,
        target_device: torch.device | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Pre-project context K/V for cross_kv_concat mode.

        Delegates to forward(mode="kv_only") so that FSDP2 hooks fire correctly.
        """
        return self(
            context_hidden=context_hidden,
            mode="kv_only",
            target_device=target_device,
            target_dtype=target_dtype,
        )


def build_layer_map(
    num_diffusion_layers: int,
    num_vlm_layers: int,
    policy: str = "middle_n",
) -> list[int]:
    """Map each diffusion layer to a VLM layer index.

    Policies:
      - middle_n: center the diffusion range within the VLM depth
      - first_n: use the first N vlm layers
      - last_n: use the last N vlm layers
    """
    if num_diffusion_layers > num_vlm_layers:
        raise ValueError(
            f"Cannot map {num_diffusion_layers} diffusion layers to "
            f"{num_vlm_layers} VLM layers with {policy}"
        )

    if policy == "first_n":
        return list(range(num_diffusion_layers))
    elif policy == "last_n":
        start = num_vlm_layers - num_diffusion_layers
        return list(range(start, num_vlm_layers))
    elif policy == "middle_n":
        start = (num_vlm_layers - num_diffusion_layers) // 2
        return list(range(start, start + num_diffusion_layers))
    else:
        raise ValueError(f"Unknown layer mapping policy: {policy}")


class CrossAttentionStack(nn.Module):
    """Full cross-attention stack: one adapter per diffusion layer.

    Manages the layer mapping from diffusion layers → VLM layers, and
    provides the `condition_layer` interface used during diffusion forward.
    """

    def __init__(
        self,
        num_diffusion_layers: int,
        num_vlm_layers: int,
        diffusion_hidden_size: int,
        vlm_hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_mapping_policy: str = "middle_n",
        hidden_state_layer_offset: int = 1,
        gate_init: float = 0.01,
        zero_init_output: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.layer_map = build_layer_map(
            num_diffusion_layers, num_vlm_layers, layer_mapping_policy
        )
        self.hidden_state_layer_offset = hidden_state_layer_offset

        self.adapters = nn.ModuleList([
            CrossAttentionAdapter(
                diffusion_hidden_size=diffusion_hidden_size,
                vlm_hidden_size=vlm_hidden_size,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                gate_init=gate_init,
                zero_init_output=zero_init_output,
                norm_eps=norm_eps,
            )
            for _ in range(num_diffusion_layers)
        ])

    def project_all_context_kv(
        self,
        vlm_hidden_states: Sequence[Tensor],
        *,
        target_device: torch.device | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Project VLM context into K/V for all diffusion layers at once.

        Avoids per-layer dynamo recompilation by unrolling all adapters in a
        single traced function (layer count is a compile-time constant).

        Returns:
            List of (k, v) tuples, one per diffusion layer.
            k/v shape: [B, H_kv, S_vlm, D]
        """
        results: list[tuple[Tensor, Tensor]] = []
        for layer_idx in range(len(self.adapters)):
            state_index = self.layer_map[layer_idx] + self.hidden_state_layer_offset
            context = vlm_hidden_states[state_index]
            adapter = self.adapters[layer_idx]
            results.append(
                adapter(context_hidden=context, mode="kv_only",
                        target_device=target_device, target_dtype=target_dtype)
            )
        return results

    def project_context_kv(
        self,
        diffusion_layer_idx: int,
        vlm_hidden_states: Sequence[Tensor],
        *,
        target_device: torch.device | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Project VLM context into K/V for cross_kv_concat at a given layer.

        Selects the VLM hidden state via layer_map + offset, then projects
        through the adapter's context_norm → k_proj / v_proj.

        Args:
            diffusion_layer_idx: index of the current diffusion layer
            vlm_hidden_states: all VLM hidden states (length = num_vlm_layers + 1)
            target_device: optional device to move context to before projection
            target_dtype: optional dtype to cast context to before projection

        Returns:
            k: [B, H_kv, S_vlm, D]
            v: [B, H_kv, S_vlm, D]
        """
        vlm_layer_idx = self.layer_map[diffusion_layer_idx]
        state_index = vlm_layer_idx + self.hidden_state_layer_offset
        context = vlm_hidden_states[state_index]

        adapter = self.adapters[diffusion_layer_idx]
        return adapter.project_context_kv(
            context, target_device=target_device, target_dtype=target_dtype
        )

    def condition_layer(
        self,
        diffusion_layer_idx: int,
        diffusion_hidden: Tensor,  # [B, S_diff, D_diff]
        vlm_hidden_states: Sequence[Tensor],  # list of [B, S_vlm, D_vlm]
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-attention for one diffusion layer (legacy separate cross-attn).

        NOTE: This is NOT the cross_kv_concat path. Use project_context_kv instead
        for the correct behavior matching the online model.

        Selects the appropriate VLM hidden state based on layer_map + offset,
        then runs the cross-attention adapter.

        Args:
            diffusion_layer_idx: index of the current diffusion layer
            diffusion_hidden: [B, S_diff, D_diff]
            vlm_hidden_states: all VLM hidden states (length = num_vlm_layers + 1)
            attention_mask: optional cross-attention mask

        Returns:
            conditioned_hidden: [B, S_diff, D_diff]
        """
        vlm_layer_idx = self.layer_map[diffusion_layer_idx]
        # Apply offset: layer i's output is at index i+1 in hidden_states list
        state_index = vlm_layer_idx + self.hidden_state_layer_offset
        context = vlm_hidden_states[state_index]

        adapter = self.adapters[diffusion_layer_idx]
        return adapter(diffusion_hidden, context, attention_mask=attention_mask)
