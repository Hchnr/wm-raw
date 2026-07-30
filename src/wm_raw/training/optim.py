"""Optimizer, scheduler, and trainable parameter configuration."""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

from ..models import WorldModel
from .config import TrainingConfig

logger = logging.getLogger(__name__)


def build_optimizer(
    model: WorldModel,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with per-component learning rates."""
    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, "module") else model

    seen: set[int] = set()
    adapter_params: list[Tensor] = []
    diffusion_params: list[Tensor] = []
    vlm_params: list[Tensor] = []

    # Adapter params: cross-attention + input_proj + time embedder + output_head + adaln
    adapter_modules = [
        raw_model.cross_attention,
        raw_model.state_diffusion.input_proj,
        raw_model.state_diffusion.time_embedder,
        raw_model.state_diffusion.time_conditioner,
        raw_model.state_diffusion.adaln_layers,
        raw_model.state_diffusion.output_head,
        raw_model.state_diffusion.latent_position_embedding,
    ]
    for module in adapter_modules:
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                adapter_params.append(p)
                seen.add(id(p))

    # Diffusion backbone params: decoder layers + final norm + rotary
    diffusion_modules = [
        raw_model.state_diffusion.layers,
        raw_model.state_diffusion.final_norm,
        raw_model.state_diffusion.rotary_emb,
    ]
    for module in diffusion_modules:
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad and id(p) not in seen:
                diffusion_params.append(p)
                seen.add(id(p))

    # VLM params: everything else
    for p in raw_model.vlm.parameters():
        if p.requires_grad and id(p) not in seen:
            vlm_params.append(p)
            seen.add(id(p))

    groups: list[dict[str, Any]] = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": config.adapter_lr, "name": "adapters"})
    if diffusion_params:
        groups.append({"params": diffusion_params, "lr": config.diffusion_lr, "name": "diffusion"})
    if vlm_params:
        groups.append({"params": vlm_params, "lr": config.vlm_lr, "name": "vlm"})

    if not groups:
        raise RuntimeError("No trainable parameters for optimizer")

    logger.info(
        "Optimizer groups: %s",
        {g["name"]: len(g["params"]) for g in groups},
    )
    return torch.optim.AdamW(
        groups,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
        fused=config.adam_fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler: constant or cosine, both with linear warmup."""

    if config.scheduler_type == "constant":
        # Constant LR after warmup
        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / max(config.warmup_steps, 1)
            return 1.0
    else:
        # Cosine decay after warmup
        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / max(config.warmup_steps, 1)
            progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_trainable(model: WorldModel, config: TrainingConfig) -> None:
    """Freeze/unfreeze parameters based on trainable_mode."""
    # Start frozen
    for p in model.parameters():
        p.requires_grad = False

    mode = config.trainable_mode

    if mode in ("diffusion", "all"):
        # Always train adapters (cross-attn, input/output proj, time, adaln)
        for module in [
            model.cross_attention,
            model.state_diffusion.input_proj,
            model.state_diffusion.time_embedder,
            model.state_diffusion.time_conditioner,
            model.state_diffusion.adaln_layers,
            model.state_diffusion.output_head,
            model.state_diffusion.latent_position_embedding,
        ]:
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

        # Optionally train diffusion backbone
        if config.train_diffusion_backbone:
            for p in model.state_diffusion.layers.parameters():
                p.requires_grad = True
            for p in model.state_diffusion.final_norm.parameters():
                p.requires_grad = True

    if mode in ("vlm", "all"):
        for p in model.vlm.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100 * trainable / total:.1f}%)")
