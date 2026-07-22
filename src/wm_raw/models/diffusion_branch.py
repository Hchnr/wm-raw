"""State Diffusion Branch — denoising backbone for image latent generation.

Assembles:
- ContinuousTokenProjector (latent → hidden)
- SinusoidalTimestepEmbedding + timestep conditioner
- Decoder layers (shared architecture with VLM but separate weights)
- Output head (hidden → latent prediction)

Uses flow matching: predicts velocity v = noise - clean.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..config import DiffusionConfig, LatentConfig
from .adaln import AdaLNZero, SinusoidalTimestepEmbedding
from .embeddings import ContinuousTokenProjector, Sincos2DPositionEmbedding, patchify_latent
from .qwen3vl_backbone import DecoderLayer, RMSNorm
from .rope import TextMRoPE


class DiffusionDecoderLayer(nn.Module):
    """Decoder layer with AdaLN-Zero timestep modulation.

    Wraps a standard DecoderLayer and applies adaptive modulation:
        x_attn = gate_attn * attn(shift_attn + scale_attn * norm(x))
        x_mlp  = gate_mlp  * mlp(shift_mlp + scale_mlp * norm(x))

    When AdaLN params are all zero (init), the layer degenerates to standard
    pre-norm transformer.
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
        self.hidden_size = hidden_size
        self.layer = DecoderLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,  # [B, S, D]
        attention_mask: Optional[Tensor],  # [B, 1, S, S+K]
        position_embeddings: tuple[Tensor, Tensor],  # (cos, sin)
        external_kv: Optional[tuple[Tensor, Tensor]] = None,
        adaln_params: Optional[tuple[Tensor, ...]] = None,  # 6 tensors each [B, D]
    ) -> Tensor:
        """Forward with optional AdaLN modulation and external KV.

        When adaln_params is None, behaves as standard decoder layer.
        When provided, applies shift/scale/gate modulation.
        """
        if adaln_params is None:
            # Standard forward (no modulation)
            return self.layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                external_kv=external_kv,
            )

        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaln_params

        # --- Attention sub-layer with modulation ---
        residual = hidden_states
        # Modulated pre-norm: (1 + scale) * norm(x) + shift
        normed = self.layer.input_layernorm(hidden_states)  # [B, S, D]
        normed = normed * (1.0 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)

        # Self-attention (with optional external KV)
        attn_out = self.layer.self_attn(
            normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            external_kv=external_kv,
        )
        # Gated residual
        hidden_states = residual + gate_attn.unsqueeze(1) * attn_out

        # --- MLP sub-layer with modulation ---
        residual = hidden_states
        normed = self.layer.post_attention_layernorm(hidden_states)
        normed = normed * (1.0 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        mlp_out = self.layer.mlp(normed)
        hidden_states = residual + gate_mlp.unsqueeze(1) * mlp_out

        return hidden_states


class StateDiffusionBranch(nn.Module):
    """Diffusion branch for image latent generation via flow matching.

    Architecture:
        1. Patchify VAE latents → token sequence
        2. Project tokens into hidden dimension
        3. Add timestep conditioning (input_add or adaln)
        4. Run through decoder layers with cross-attention from VLM
        5. Project back to latent dimension (velocity prediction)
    """

    def __init__(
        self,
        config: DiffusionConfig,
        latent_config: LatentConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.latent_config = latent_config

        # Patchified token dimensions
        patch_h = latent_config.latent_height // latent_config.patch_size
        patch_w = latent_config.latent_width // latent_config.patch_size
        token_dim = (
            latent_config.patch_size ** 2 * latent_config.latent_channels
        )
        self.num_latent_tokens = patch_h * patch_w
        self.token_dim = token_dim

        # Input projection: latent token → hidden
        self.input_proj = ContinuousTokenProjector(token_dim, config.hidden_size)

        # Timestep embedding
        self.time_embedder = SinusoidalTimestepEmbedding(
            frequency_dim=config.hidden_size,
            hidden_size=config.hidden_size,
        )
        # Timestep → hidden additive conditioning
        self.time_conditioner = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Position embedding for latent grid
        self.position_embedding = Sincos2DPositionEmbedding(
            grid_h=patch_h,
            grid_w=patch_w,
            embed_dim=config.hidden_size,
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            DiffusionDecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        # AdaLN-Zero per layer (optional, for adaln_zero timestep conditioning)
        self.adaln_layers = nn.ModuleList([
            AdaLNZero(config.hidden_size) for _ in range(config.num_hidden_layers)
        ])

        # Final norm + output projection
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_size, token_dim, bias=False)

        # Rotary embedding for diffusion sequence
        self.rotary_emb = TextMRoPE(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )

    def prepare_inputs(
        self,
        noisy_latent: Tensor,  # [B, H*W, C] — noisy VAE latents (flat grid)
        timesteps: Tensor,  # [B]
    ) -> tuple[Tensor, Tensor]:
        """Tokenize noisy latents and apply timestep conditioning.

        Returns:
            hidden: [B, num_tokens, D] — ready for decoder layers
            time_hidden: [B, D] — timestep embedding for AdaLN
        """
        batch = noisy_latent.shape[0]

        # 1. Patchify: [B, H*W, C] → [B, num_tokens, token_dim]
        tokens = patchify_latent(
            noisy_latent,
            height=self.latent_config.latent_height,
            width=self.latent_config.latent_width,
            patch_size=self.latent_config.patch_size,
        )

        # 2. Project to hidden dimension: [B, num_tokens, D]
        hidden = self.input_proj(tokens)

        # 3. Add position embedding
        pos_embed = self.position_embedding(batch)  # [B, num_tokens, D]
        hidden = hidden + pos_embed

        # 4. Timestep conditioning
        time_hidden = self.time_embedder(timesteps)  # [B, D]
        # Add timestep to all token positions
        time_cond = self.time_conditioner(time_hidden)  # [B, D]
        hidden = hidden + time_cond.unsqueeze(1)  # [B, num_tokens, D]

        return hidden, time_hidden

    def forward(
        self,
        noisy_latent: Tensor,  # [B, H*W, C]
        timesteps: Tensor,  # [B]
        cross_attention_contexts: list[Tensor] | None = None,  # per-layer [B, S_vlm, D_vlm]
        cross_attention_fn=None,  # CrossAttentionStack.condition_layer
        attention_mask: Optional[Tensor] = None,  # [B, 1, S, S]
        cross_attention_mask: Optional[Tensor] = None,  # [B, 1, S_diff, S_vlm]
    ) -> Tensor:
        """Full diffusion forward: noisy latent → velocity prediction.

        Args:
            noisy_latent: [B, H*W, C] noisy (flow-interpolated) latent
            timesteps: [B] timestep values in [0, 1]
            cross_attention_contexts: VLM hidden states per diffusion layer
            cross_attention_fn: callable to apply cross-attention
            attention_mask: self-attention mask for diffusion tokens
            cross_attention_mask: mask for cross-attention to VLM

        Returns:
            prediction: [B, num_tokens, token_dim] — velocity prediction
        """
        # Prepare inputs
        hidden, time_hidden = self.prepare_inputs(noisy_latent, timesteps)
        batch, num_tokens, _ = hidden.shape

        # Build position IDs for MRoPE (simple 1D positions for latent tokens)
        # Use same position ID for all 3 axes (no temporal/spatial distinction in latent)
        pos_ids = torch.arange(
            num_tokens, device=hidden.device, dtype=torch.long
        ).unsqueeze(0).expand(batch, -1)  # [B, S]
        # Expand to [3, B, S] for MRoPE
        position_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]
        cos, sin = self.rotary_emb(position_ids)

        # Run decoder layers
        for layer_idx, (layer, adaln) in enumerate(zip(self.layers, self.adaln_layers)):
            # Compute AdaLN params from timestep
            adaln_params = adaln(time_hidden)

            # Apply cross-attention if available
            if cross_attention_fn is not None and cross_attention_contexts is not None:
                hidden = cross_attention_fn(
                    layer_idx, hidden, cross_attention_contexts, cross_attention_mask
                )

            # Decoder layer with AdaLN modulation
            hidden = layer(
                hidden,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                adaln_params=adaln_params,
            )

        # Final norm + output projection
        hidden = self.final_norm(hidden)  # [B, num_tokens, D]
        prediction = self.output_head(hidden)  # [B, num_tokens, token_dim]

        return prediction
