"""State Diffusion Branch — denoising backbone for image latent generation.

Assembles:
- ContinuousTokenProjector (latent → hidden)
- SinusoidalTimestepEmbedding + timestep conditioner (input_add)
- AdaLN-Zero per-layer modulation
- Decoder layers with external KV (cross_kv_concat)
- Output head (hidden → latent prediction)

Uses flow matching: predicts velocity v = noise - clean.

Timestep conditioning is dual:
  1. input_add: time_conditioner(time_hidden) added to all tokens before layers
  2. adaln_zero: per-layer adaptive modulation (shift/scale/gate)
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn

from ..config import DiffusionConfig, LatentConfig
from .adaln import AdaLNZero, SinusoidalTimestepEmbedding
from .embeddings import BagelGridPositionEmbedding, ContinuousTokenProjector
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
        1. Project patchified latent tokens into hidden dimension
        2. Add 2D sincos position embedding (based on patch grid)
        3. Add timestep conditioning (input_add: broadcast to all tokens)
        4. Run through decoder layers with AdaLN-Zero + external KV
        5. Final norm → output head (velocity prediction)

    Supports variable latent sizes (resolution buckets): the patch grid
    dimensions (patch_h, patch_w) are passed at forward time, not fixed.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        latent_config: LatentConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.latent_config = latent_config

        # Token dimension: P*P*C (e.g. 2*2*16 = 64)
        token_dim = latent_config.token_dim

        # Input projection: latent token → hidden (no normalization, matches online)
        self.input_proj = ContinuousTokenProjector(
            token_dim, config.hidden_size, normalize_input=False
        )

        # Timestep embedding
        self.time_embedder = SinusoidalTimestepEmbedding(
            frequency_dim=config.timestep_frequency_dim,
            hidden_size=config.hidden_size,
        )
        # Timestep → hidden additive conditioning (input_add)
        self.time_conditioner = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Position embedding for latent grid (BAGEL-style frozen 2D sincos table)
        self.latent_position_embedding = BagelGridPositionEmbedding(
            max_num_patch_per_side=latent_config.max_position_size,
            hidden_size=config.hidden_size,
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

        # AdaLN-Zero per layer
        self.adaln_layers = nn.ModuleList([
            AdaLNZero(config.hidden_size) for _ in range(config.num_hidden_layers)
        ])

        # Final norm + output projection
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_head = nn.Linear(config.hidden_size, token_dim, bias=True)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        # Rotary embedding for diffusion sequence
        self.rotary_emb = TextMRoPE(
            head_dim=config.head_dim,
            theta=config.rope_theta,
            mrope_section=config.mrope_section,
        )

    def prepare_inputs(
        self,
        noisy_tokens: Tensor,  # [B, num_tokens, token_dim]
        timesteps: Tensor,  # [B]
        patch_h: int,  # patch grid height (latent_h / patch_size)
        patch_w: int,  # patch grid width (latent_w / patch_size)
    ) -> tuple[Tensor, Tensor]:
        """Project noisy tokens and apply position + timestep conditioning.

        Args:
            noisy_tokens: [B, num_tokens, token_dim] already patchified
            timesteps: [B] flow matching timesteps
            patch_h: height of patch grid
            patch_w: width of patch grid

        Returns:
            hidden: [B, num_tokens, D] — ready for decoder layers
            time_hidden: [B, D] — timestep embedding for AdaLN
        """
        batch = noisy_tokens.shape[0]

        # 1. Project to hidden dimension: [B, num_tokens, D]
        hidden = self.input_proj(noisy_tokens)

        # 2. Add position embedding (BAGEL-style: row * max_pos + col indexing)
        pos_ids = self.latent_position_embedding.make_position_ids(
            patch_h, patch_w, batch
        ).to(hidden.device)
        pos_embed = self.latent_position_embedding(pos_ids)  # [B, num_tokens, D]
        hidden = hidden + pos_embed.to(hidden.dtype)

        # 3. Timestep conditioning — input_add (only if not using adaln_zero)
        time_hidden = self.time_embedder(timesteps)  # [B, D]
        if self.latent_config.timestep_conditioning == "input_add":
            time_cond = self.time_conditioner(time_hidden)  # [B, D]
            hidden = hidden + time_cond.unsqueeze(1)  # [B, num_tokens, D]

        return hidden, time_hidden

    def forward(
        self,
        noisy_latent: Tensor,  # [B, num_tokens, token_dim]
        timesteps: Tensor,  # [B]
        patch_h: int,  # patch grid height
        patch_w: int,  # patch grid width
        cross_attention_stack=None,  # CrossAttentionStack instance
        vlm_hidden_states: Sequence[Tensor] | None = None,
        attention_mask: Optional[Tensor] = None,  # [B, 1, S, S+K] combined mask
        cross_attention_mask: Optional[Tensor] = None,  # unused for now
    ) -> Tensor:
        """Full diffusion forward: noisy tokens → velocity prediction.

        Uses cross_kv_concat: VLM context K/V are projected and concatenated
        to self-attention K/V inside each decoder layer.

        Args:
            noisy_latent: [B, num_tokens, token_dim] patchified noisy tokens
            timesteps: [B] timestep values in [0, 1]
            patch_h: height of patch grid (varies per resolution bucket)
            patch_w: width of patch grid (varies per resolution bucket)
            cross_attention_stack: CrossAttentionStack for projecting VLM context
            vlm_hidden_states: VLM hidden states (list, indexed by layer map)
            attention_mask: combined self+cross attention mask [B, 1, S, K+S]
            cross_attention_mask: unused placeholder

        Returns:
            prediction: [B, num_tokens, token_dim] — velocity prediction
        """
        # Prepare inputs (project + pos embed + timestep cond)
        hidden, time_hidden = self.prepare_inputs(noisy_latent, timesteps, patch_h, patch_w)
        batch, num_tokens, _ = hidden.shape

        # Build MRoPE position IDs for latent tokens: [3, B, num_tokens]
        # Online model uses simple sequential positions (arange) for ALL 3 MRoPE axes.
        # The 2D spatial information is already encoded via BagelGridPositionEmbedding
        # (additive pos embed in prepare_inputs), so RoPE just provides sequential ordering.
        device = hidden.device
        pos_ids = torch.arange(
            num_tokens, device=device, dtype=torch.long
        ).unsqueeze(0).expand(batch, -1)  # [B, S]
        position_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]
        cos, sin = self.rotary_emb(position_ids)

        # Build combined attention mask for cross_kv_concat.
        # Online model builds an explicit all-zeros mask [B, 1, S_q, S_ext+S_self]
        # when external_kv is present. We must do the same to match SDPA kernel behavior.
        combined_mask = attention_mask
        if combined_mask is None and cross_attention_stack is not None and vlm_hidden_states is not None:
            # Determine external KV length from the first VLM hidden state
            # that will be used (layer_map[0] + offset)
            vlm_seq_len = vlm_hidden_states[0].shape[1]
            combined_mask = torch.zeros(
                batch, 1, num_tokens, vlm_seq_len + num_tokens,
                device=hidden.device, dtype=hidden.dtype,
            )

        # Pre-project all VLM context K/V outside the layer loop.
        # This avoids dynamo recompilation per diffusion_layer_idx.
        all_external_kv: list[tuple[Tensor, Tensor]] | None = None
        if cross_attention_stack is not None and vlm_hidden_states is not None:
            all_external_kv = cross_attention_stack.project_all_context_kv(
                vlm_hidden_states,
                target_device=hidden.device,
                target_dtype=hidden.dtype,
            )

        # Run decoder layers
        for layer_idx, (layer, adaln) in enumerate(zip(self.layers, self.adaln_layers)):
            # Compute AdaLN params from timestep
            adaln_params = adaln(time_hidden)

            # Retrieve pre-projected K/V for this layer
            external_kv = all_external_kv[layer_idx] if all_external_kv is not None else None

            # Decoder layer with AdaLN modulation + external KV
            hidden = layer(
                hidden,
                attention_mask=combined_mask,
                position_embeddings=(cos, sin),
                external_kv=external_kv,
                adaln_params=adaln_params,
            )

        # Final norm + output projection
        hidden = self.final_norm(hidden)  # [B, num_tokens, D]
        prediction = self.output_head(hidden)  # [B, num_tokens, token_dim]

        return prediction
