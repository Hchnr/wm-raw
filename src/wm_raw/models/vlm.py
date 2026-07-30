"""VLM Branch — Qwen3-VL backbone for text/image understanding.

Assembles:
- Token embedding
- Vision encoder (processes image patches → visual tokens)
- Text decoder stack (with MRoPE)
- LM head (for AR loss)

Produces hidden states at selected layers for cross-attention to diffusion.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..config import TextModelConfig, VisionModelConfig
from .qwen3vl_backbone import DecoderLayer, RMSNorm
from .rope import TextMRoPE
from .vision_encoder import VisionEncoder, VisionEncoderOutput


class VLMBranch(nn.Module):
    """Qwen3-VL VLM backbone for conditioning the diffusion branch.

    Forward produces:
    - ar_loss: autoregressive cross-entropy on text tokens (if labels given)
    - hidden_states: list of intermediate hidden states for cross-attention

    The list of hidden states is indexed by diffusion layer via layer_mapping.
    """

    def __init__(
        self,
        text_config: TextModelConfig,
        vision_config: VisionModelConfig,
    ) -> None:
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config

        # Token embedding
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            depth=vision_config.depth,
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            num_heads=vision_config.num_heads,
            in_channels=vision_config.in_channels,
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            spatial_merge_size=vision_config.spatial_merge_size,
            out_hidden_size=vision_config.out_hidden_size,
            rope_theta=vision_config.rope_theta,
            deepstack_visual_indexes=vision_config.deepstack_visual_indexes,
        )

        # Text decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=text_config.hidden_size,
                intermediate_size=text_config.intermediate_size,
                num_attention_heads=text_config.num_attention_heads,
                num_key_value_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                rms_norm_eps=text_config.rms_norm_eps,
            )
            for _ in range(text_config.num_hidden_layers)
        ])

        # Final norm + LM head
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        # Rotary embedding
        self.rotary_emb = TextMRoPE(
            head_dim=text_config.head_dim,
            theta=text_config.rope_theta,
            mrope_section=text_config.mrope_section,
        )

    def forward(
        self,
        input_ids: Tensor,  # [B, S]
        attention_mask: Tensor,  # [B, 1, S, S] causal mask
        position_ids: Tensor,  # [3, B, S] MRoPE positions
        pixel_values: Optional[Tensor] = None,  # [N_patches, C, T, P, P]
        image_grid_thw: Optional[Tensor] = None,  # [num_images, 3]
        image_token_mask: Optional[Tensor] = None,  # [B, S] bool: where to insert visual tokens
        labels: Optional[Tensor] = None,  # [B, S] for AR loss (-100 = ignore)
    ) -> VLMOutput:
        """Forward pass through VLM branch.

        Returns hidden states from all layers (for cross-attention)
        and optionally the AR loss.
        """
        # 1. Token embeddings
        hidden_states = self.embed_tokens(input_ids)  # [B, S, D]

        # 2. Process and merge visual tokens (if images present)
        deepstack_features: list[Tensor] = []
        if pixel_values is not None and image_grid_thw is not None:
            vision_output: VisionEncoderOutput = self.vision_encoder(
                pixel_values, image_grid_thw
            )
            visual_tokens = vision_output.visual_tokens  # [total_visual_tokens, D]
            deepstack_features = vision_output.deepstack_features

            # Scatter visual tokens into the sequence at image_token positions
            if image_token_mask is not None:
                hidden_states = self._merge_visual_tokens(
                    hidden_states, visual_tokens, image_token_mask
                )

        # 3. Compute position embeddings (MRoPE)
        cos, sin = self.rotary_emb(position_ids)  # [B, S, head_dim]

        # 4. Run through decoder layers, collecting hidden states
        all_hidden_states: list[Tensor] = [hidden_states]
        num_deepstack = len(deepstack_features)

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )
            # DeepStack: inject visual features into early decoder layers
            if num_deepstack > 0 and layer_idx < num_deepstack and image_token_mask is not None:
                hidden_states = self._deepstack_process(
                    hidden_states, image_token_mask, deepstack_features[layer_idx]
                )
            all_hidden_states.append(hidden_states)

        # 5. Final norm
        hidden_states = self.norm(hidden_states)  # [B, S, D]

        # 6. Compute AR loss if labels provided
        ar_loss = None
        if labels is not None:
            logits = self.lm_head(hidden_states)  # [B, S, V]
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, V]
            shift_labels = labels[:, 1:].contiguous()  # [B, S-1]
            ar_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return VLMOutput(
            hidden_states=all_hidden_states,  # list of [B, S, D], length = num_layers + 1
            ar_loss=ar_loss,
            last_hidden_state=hidden_states,
        )

    def _merge_visual_tokens(
        self,
        text_embeddings: Tensor,  # [B, S, D]
        visual_tokens: Tensor,  # [total_visual_tokens, D]
        mask: Tensor,  # [B, S] bool mask indicating visual token positions
    ) -> Tensor:
        """Replace masked positions in text_embeddings with visual tokens.

        Uses scatter with cumulative indexing (no boolean indexing for static shapes).
        The mask has exactly total_visual_tokens True values across the batch.
        """
        batch, seq_len, dim = text_embeddings.shape

        # Compute flat indices for visual positions via cumsum
        flat_mask = mask.reshape(-1)  # [B*S]
        # Cumsum gives 1-based index for each True position
        cum_indices = flat_mask.to(torch.int64).cumsum(0) - 1  # [B*S]
        # Create output by clone and scatter
        output = text_embeddings.clone()
        flat_output = output.reshape(-1, dim)  # [B*S, D]

        # Gather visual tokens in order and place them
        # flat_mask positions get visual_tokens[cum_indices[i]]
        visual_positions = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_vis]
        flat_output[visual_positions] = visual_tokens.to(flat_output.dtype)

        return flat_output.reshape(batch, seq_len, dim)

    def _deepstack_process(
        self,
        hidden_states: Tensor,  # [B, S, D]
        visual_pos_mask: Tensor,  # [B, S] bool mask for visual token positions
        visual_embeds: Tensor,  # [total_visual_tokens, D] deepstack feature
    ) -> Tensor:
        """Add deepstack visual features to hidden states at visual positions.

        This injects multi-scale vision features from intermediate vision encoder
        layers into the early language model decoder layers, following the DeepStack
        paper (https://arxiv.org/abs/2406.04334).
        """
        hidden_states = hidden_states.clone()
        visual_embeds = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states[visual_pos_mask] = hidden_states[visual_pos_mask] + visual_embeds
        return hidden_states


class VLMOutput:
    """Output from VLM branch forward pass."""

    __slots__ = ("hidden_states", "ar_loss", "last_hidden_state")

    def __init__(
        self,
        hidden_states: list[Tensor],
        ar_loss: Optional[Tensor],
        last_hidden_state: Tensor,
    ) -> None:
        self.hidden_states = hidden_states
        self.ar_loss = ar_loss
        self.last_hidden_state = last_hidden_state
