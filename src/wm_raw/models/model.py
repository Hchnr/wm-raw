"""WorldModel — top-level module assembling VLM + Diffusion + CrossAttention.

This is the single nn.Module that gets wrapped in FSDP/DDP and called
during training. It handles:
- VLM forward (conditioning)
- Diffusion forward (latent denoising)
- Cross-attention routing
- Loss computation and aggregation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..config import WorldModelConfig
from ..diffusion import add_flow_noise, flow_matching_loss, flow_matching_target, sample_timesteps
from .cross_attention import CrossAttentionStack
from .embeddings import patchify_latent
from .vlm import VLMBranch, VLMOutput


@dataclass
class DiffusionOutput:
    """Output from diffusion branch forward pass."""

    loss: Optional[Tensor]  # scalar MSE loss
    prediction: Tensor  # [B, S, token_dim] velocity prediction


@dataclass
class WorldModelOutput:
    """Combined output from the full model."""

    loss: Optional[Tensor]  # total weighted loss
    ar_loss: Optional[Tensor]  # VLM autoregressive loss
    diffusion_loss: Optional[Tensor]  # state diffusion loss
    metadata: dict  # task info


class WorldModel(nn.Module):
    """World Model: VLM backbone + state diffusion branch + cross-attention.

    Training workflow (diffusion-only mode, matching online config):
        1. VLM branch processes condition text → hidden states (no AR loss)
        2. Diffusion branch denoises latent tokens conditioned on VLM hidden
        3. Total loss = state_diffusion_loss_weight * diff_loss
    """

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # VLM branch (Qwen3-VL 4B)
        self.vlm = VLMBranch(
            text_config=config.text,
            vision_config=config.vision,
        )

        # State diffusion branch (Qwen3-VL 2B)
        from .diffusion_branch import StateDiffusionBranch

        self.state_diffusion = StateDiffusionBranch(
            config=config.diffusion,
            latent_config=config.latent,
        )

        # Cross-attention stack
        self.cross_attention = CrossAttentionStack(
            num_diffusion_layers=config.diffusion.num_hidden_layers,
            num_vlm_layers=config.text.num_hidden_layers,
            diffusion_hidden_size=config.diffusion.hidden_size,
            vlm_hidden_size=config.text.hidden_size,
            num_query_heads=config.diffusion.num_attention_heads,
            num_kv_heads=config.diffusion.num_key_value_heads,
            head_dim=config.diffusion.head_dim,
            layer_mapping_policy=config.layer_mapping.policy,
            gate_init=config.cross_attention.gate_init,
            zero_init_output=config.cross_attention.zero_init_output,
            norm_eps=config.cross_attention.norm_eps,
            hidden_state_layer_offset=config.cross_attention.hidden_state_layer_offset,
        )

    def forward_vlm(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        pixel_values: Optional[Tensor] = None,
        image_grid_thw: Optional[Tensor] = None,
        image_token_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> VLMOutput:
        """Run VLM forward pass."""
        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_token_mask=image_token_mask,
            labels=labels,
        )

    def forward_diffusion(
        self,
        state_target: Tensor,  # [B, C, H_lat, W_lat] or [B, H*W, C] raw VAE latents
        vlm_hidden_states: list[Tensor],  # from VLM forward
        latent_h: int,  # latent height (before patchify)
        latent_w: int,  # latent width (before patchify)
        timesteps: Optional[Tensor] = None,  # [B] optional pre-sampled
        noise: Optional[Tensor] = None,  # [B, num_tokens, token_dim] optional
        loss_mask: Optional[Tensor] = None,  # [B, num_tokens]
        cross_attention_mask: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """Run diffusion forward pass with flow matching.

        Steps:
            1. Patchify latent target
            2. Sample timesteps (logit-normal) and noise
            3. Create noisy input via linear interpolation
            4. Run diffusion backbone with cross-attention
            5. Compute flow matching loss
        """
        batch = state_target.shape[0]
        device = state_target.device
        latent_cfg = self.config.latent
        patch_size = latent_cfg.patch_size

        # Patch grid dimensions
        patch_h = latent_h // patch_size
        patch_w = latent_w // patch_size

        # 1. Patchify: [B, H*W, C] → [B, num_patches, patch_dim]
        clean_tokens = patchify_latent(
            state_target,
            height=latent_h,
            width=latent_w,
            patch_size=patch_size,
        )  # [B, patch_h*patch_w, token_dim]

        num_tokens = clean_tokens.shape[1]
        token_dim = clean_tokens.shape[2]

        # 2. Sample timesteps and noise
        if timesteps is None:
            ts_cfg = latent_cfg.timestep_sampling
            timesteps = sample_timesteps(
                batch,
                device=device,
                sampling_type=ts_cfg.type,
                shift=latent_cfg.timestep_shift,
                mean=ts_cfg.mean,
                std=ts_cfg.std,
            )
        if noise is None:
            noise = torch.randn_like(clean_tokens)

        # 3. Create noisy sample and velocity target
        noisy_tokens = add_flow_noise(clean_tokens, noise, timesteps)
        velocity_target = flow_matching_target(clean_tokens, noise)

        # 4. Run diffusion backbone (cross_kv_concat: VLM K/V concat to self-attn)
        prediction = self.state_diffusion(
            noisy_latent=noisy_tokens,
            timesteps=timesteps,
            patch_h=patch_h,
            patch_w=patch_w,
            cross_attention_stack=self.cross_attention,
            vlm_hidden_states=vlm_hidden_states,
            cross_attention_mask=cross_attention_mask,
        )  # [B, num_tokens, token_dim]

        # 5. Compute loss
        loss = flow_matching_loss(prediction, velocity_target, mask=loss_mask)

        return DiffusionOutput(loss=loss, prediction=prediction)

    def forward(
        self,
        batch: dict,
    ) -> WorldModelOutput:
        """Dispatch training batch.

        Expected batch keys:
            task_type: "vlm" | "diffusion" | "joint"

            For diffusion (primary mode):
                condition.{input_ids, attention_mask, position_ids, ...}
                state_target: [B, H*W, C] VAE encoded latents
                latent_h, latent_w: latent spatial dimensions
                [timesteps, noise, loss_mask] — optional, will be sampled if absent
        """
        task_type = batch.get("task_type", "diffusion")

        if task_type == "vlm":
            vlm_out = self.forward_vlm(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                image_token_mask=batch.get("image_token_mask"),
                labels=batch.get("labels"),
            )
            return WorldModelOutput(
                loss=vlm_out.ar_loss,
                ar_loss=vlm_out.ar_loss,
                diffusion_loss=None,
                metadata={"task_type": "vlm"},
            )

        elif task_type == "diffusion":
            # First run VLM on condition
            cond = batch["condition"]
            vlm_out = self.forward_vlm(
                input_ids=cond["input_ids"],
                attention_mask=cond["attention_mask"],
                position_ids=cond["position_ids"],
                pixel_values=cond.get("pixel_values"),
                image_grid_thw=cond.get("image_grid_thw"),
                image_token_mask=cond.get("image_token_mask"),
                labels=None,  # no AR loss for conditioning pass
            )

            diff_out = self.forward_diffusion(
                state_target=batch["state_target"],
                vlm_hidden_states=vlm_out.hidden_states,
                latent_h=batch["latent_h"],
                latent_w=batch["latent_w"],
                timesteps=batch.get("timesteps"),
                noise=batch.get("noise"),
                loss_mask=batch.get("loss_mask"),
                cross_attention_mask=batch.get("cross_attention_mask"),
            )
            return WorldModelOutput(
                loss=diff_out.loss,
                ar_loss=None,
                diffusion_loss=diff_out.loss,
                metadata={"task_type": "diffusion"},
            )

        else:  # joint
            cond = batch.get("vlm", batch.get("condition", {}))
            vlm_out = self.forward_vlm(
                input_ids=cond["input_ids"],
                attention_mask=cond["attention_mask"],
                position_ids=cond["position_ids"],
                pixel_values=cond.get("pixel_values"),
                image_grid_thw=cond.get("image_grid_thw"),
                image_token_mask=cond.get("image_token_mask"),
                labels=cond.get("labels"),
            )

            diff_out = self.forward_diffusion(
                state_target=batch["state_target"],
                vlm_hidden_states=vlm_out.hidden_states,
                latent_h=batch["latent_h"],
                latent_w=batch["latent_w"],
                timesteps=batch.get("timesteps"),
                noise=batch.get("noise"),
                loss_mask=batch.get("loss_mask"),
                cross_attention_mask=batch.get("cross_attention_mask"),
            )

            ar_loss = vlm_out.ar_loss
            diff_loss = diff_out.loss
            total_loss = torch.tensor(0.0, device=diff_loss.device)
            if ar_loss is not None:
                total_loss = total_loss + self.config.ar_loss_weight * ar_loss
            total_loss = total_loss + self.config.state_diffusion_loss_weight * diff_loss

            return WorldModelOutput(
                loss=total_loss,
                ar_loss=ar_loss,
                diffusion_loss=diff_loss,
                metadata={"task_type": "joint"},
            )
