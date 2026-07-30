"""Qwen3-VL Vision Encoder — static shape implementation.

Components:
- PatchEmbed (Conv3D)
- VisionAttention (with 2D RoPE)
- VisionBlock (pre-norm)
- PatchMerger (spatial 2x2 → linear)
- VisionEncoder (full stack)

All shapes are deterministic when image size is fixed (256x256 in our case).
No .item(), .tolist(), or dynamic indexing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import VisionRotaryEmbedding


class PatchEmbed(nn.Module):
    """3D patch embedding via Conv3D.

    Input: [B*T*H*W/(patch**2 * temporal_patch), C, temporal_patch, patch, patch]
    Actually we reshape externally and feed flattened patches.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 1152,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Embed pixel patches into hidden dimension.

        Args:
            pixel_values: [N, C, temporal_patch, patch, patch]
                N = total number of patches across batch

        Returns:
            hidden: [N, D] — one vector per patch
        """
        # pixel_values: [N, C, T, P, P]
        hidden = self.proj(pixel_values.to(self.proj.weight.dtype))  # [N, D, 1, 1, 1]
        return hidden.view(hidden.shape[0], -1)  # [N, D]


class VisionMLP(nn.Module):
    """Vision encoder MLP (GELU activation, with bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder with 2D RoPE.

    Uses flash-attention-friendly approach: packs all image tokens in a
    single sequence, with cu_seqlens marking image boundaries for
    variable-length attention (or fixed-length with padding when static).
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        hidden_states: Tensor,  # [total_tokens, D]
        position_embeddings: tuple[Tensor, Tensor],  # (cos, sin): [total_tokens, D]
        cu_seqlens: Tensor,  # [num_images + 1] cumulative lengths
    ) -> Tensor:
        """
        For static shapes (single fixed-size image per sample), this reduces to
        standard batched attention. We keep cu_seqlens for generality.

        Args:
            hidden_states: [N, D] packed tokens for all images
            position_embeddings: (cos, sin) each [N, D]
            cu_seqlens: [num_images + 1] — boundaries

        Returns:
            output: [N, D]
        """
        total_tokens = hidden_states.shape[0]
        head_dim = self.head_dim

        q = self.q_proj(hidden_states).view(total_tokens, self.num_heads, head_dim)
        k = self.k_proj(hidden_states).view(total_tokens, self.num_heads, head_dim)
        v = self.v_proj(hidden_states).view(total_tokens, self.num_heads, head_dim)

        # Apply 2D rotary embeddings
        cos, sin = position_embeddings  # [N, D]
        cos = cos.unsqueeze(1)  # [N, 1, D]
        sin = sin.unsqueeze(1)  # [N, 1, D]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin

        # For fixed image sizes, we can reshape to batched attention
        # num_images = cu_seqlens.shape[0] - 1
        # tokens_per_image = fixed (e.g., 324 for 256x256 with patch=16, merge=2)
        # Reshape: [N, H, D] → [num_images, tokens_per_image, H, D] → [num_images, H, T, D]
        num_images = cu_seqlens.shape[0] - 1
        tokens_per_image = total_tokens // num_images

        q = q.view(num_images, tokens_per_image, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(num_images, tokens_per_image, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(num_images, tokens_per_image, self.num_heads, head_dim).transpose(1, 2)

        # [num_images, H, T, D]
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # [num_images, H, T, D] → [N, D]
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(total_tokens, self.hidden_size)
        )
        return self.o_proj(attn_output)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class VisionBlock(nn.Module):
    """Single vision transformer block (pre-norm)."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,  # [N, D]
        position_embeddings: tuple[Tensor, Tensor],
        cu_seqlens: Tensor,
    ) -> Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), position_embeddings, cu_seqlens
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """Spatial 2x2 patch merger: reduces spatial resolution by 2x.

    Takes groups of spatial_merge_size^2 adjacent tokens and projects
    their concatenation to the text hidden dimension.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
    ) -> None:
        super().__init__()
        self.merge_size = spatial_merge_size
        merged_dim = vision_hidden_size * (spatial_merge_size**2)
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(merged_dim, merged_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(merged_dim, out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Merge spatial patches.

        Args:
            x: [N, D_vision] — N must be divisible by merge_size^2

        Returns:
            merged: [N // merge_size^2, D_text]
        """
        x = self.norm(x)
        # Reshape to group merge_size^2 tokens
        x = x.view(-1, self.merge_size**2 * x.shape[-1])
        x = self.fc2(self.act(self.fc1(x)))
        return x


class DeepstackMerger(nn.Module):
    """Deepstack patch merger (post-shuffle norm variant).

    Same spatial merge as PatchMerger but normalizes AFTER the spatial
    reshape (post-shuffle). Uses HF-compatible parameter naming
    (linear_fc1/linear_fc2) so checkpoint keys map directly.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
    ) -> None:
        super().__init__()
        self.merge_size = spatial_merge_size
        merged_dim = vision_hidden_size * (spatial_merge_size**2)
        # Post-shuffle norm: normalizes the concatenated vector
        self.norm = nn.LayerNorm(merged_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged_dim, merged_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_dim, out_hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Merge spatial patches with post-shuffle normalization.

        Args:
            x: [N, D_vision] — N must be divisible by merge_size^2

        Returns:
            merged: [N // merge_size^2, D_text]
        """
        # Reshape first (shuffle), then normalize
        x = x.view(-1, self.merge_size**2 * x.shape[-1])
        x = self.norm(x)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionEncoderOutput:
    """Output from VisionEncoder forward pass."""

    __slots__ = ("visual_tokens", "deepstack_features")

    def __init__(
        self,
        visual_tokens: Tensor,
        deepstack_features: list[Tensor],
    ) -> None:
        self.visual_tokens = visual_tokens
        self.deepstack_features = deepstack_features


class VisionEncoder(nn.Module):
    """Complete Qwen3-VL vision encoder.

    For a fixed 256x256 image with patch_size=16, temporal_patch=2:
    - Raw patches: (256/16)^2 = 256 spatial patches (single frame → padded to 2)
    - After merger (2x2): 256/4 = 64 visual tokens
    - Output dimension: out_hidden_size (= text hidden_size)
    """

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        rope_theta: float = 10000.0,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.deepstack_visual_indexes = deepstack_visual_indexes
        head_dim = hidden_size // num_heads

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )
        # Learnable 2D position embeddings (bilinear interpolation grid)
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim, theta=rope_theta)

        self.blocks = nn.ModuleList([
            VisionBlock(hidden_size, intermediate_size, num_heads) for _ in range(depth)
        ])
        self.merger = PatchMerger(hidden_size, spatial_merge_size, out_hidden_size)

        # DeepStack mergers: one per deepstack_visual_indexes layer
        self.deepstack_merger_list = nn.ModuleList([
            DeepstackMerger(hidden_size, spatial_merge_size, out_hidden_size)
            for _ in range(len(deepstack_visual_indexes))
        ])

    def forward(
        self,
        pixel_values: Tensor,  # [N, C, T, P, P]
        position_ids: Tensor,  # [N] spatial position indices for RoPE
        cu_seqlens: Tensor,  # [num_images + 1]
        bilinear_indices: Tensor,  # [N, 4] for position embedding interpolation
        bilinear_weights: Tensor,  # [N, 4]
    ) -> VisionEncoderOutput:
        """Run vision encoder.

        Returns:
            VisionEncoderOutput with:
                visual_tokens: [total_merged_tokens, D_text]
                deepstack_features: list of [total_merged_tokens, D_text], one per deepstack layer
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)  # [N, D]

        # Position embedding via bilinear interpolation
        pos_embeds = self.pos_embed(bilinear_indices)  # [N, 4, D]
        pos_embeds = (pos_embeds * bilinear_weights[:, :, None]).sum(dim=1)  # [N, D]
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        # Rotary embeddings for all tokens
        rotary_freqs = self.rotary_pos_emb(position_ids)  # [N, head_dim/2]
        emb = torch.cat((rotary_freqs, rotary_freqs), dim=-1)  # [N, head_dim]
        position_embeddings = (emb.cos(), emb.sin())

        # Transformer blocks with deepstack feature extraction
        deepstack_features: list[Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings, cu_seqlens)
            if layer_idx in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_idx)
                deepstack_features.append(
                    self.deepstack_merger_list[merger_idx](hidden_states)
                )

        # Patch merger: reduce spatial resolution
        merged = self.merger(hidden_states)  # [N/merge^2, D_text]
        return VisionEncoderOutput(
            visual_tokens=merged,
            deepstack_features=deepstack_features,
        )
