"""FSDP2 training loop for wm-raw WorldModel.

Minimal, production-ready training loop supporting:
- FSDP2 data-parallel sharding
- Mixed precision (bf16)
- Per-group learning rates (adapter / diffusion backbone / VLM)
- Gradient clipping
- Cosine LR schedule with warmup
- Checkpointing (DCP or manual)
- Logging (JSONL + optional wandb)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler

from .config import WorldModelConfig
from .models import WorldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All training hyperparameters."""

    # Paths
    output_dir: str = "outputs/wm-raw"
    vlm_path: str = ""
    diffusion_path: str = ""
    vae_path: str = ""

    # Data
    train_manifest: str = ""
    image_root: str = ""
    image_size: int = 256
    max_samples: int | None = None
    batch_size: int = 1
    num_workers: int = 4
    seed: int = 42

    # Optimizer
    adapter_lr: float = 1e-4
    diffusion_lr: float = 1e-5
    vlm_lr: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 10000
    warmup_steps: int = 200
    scheduler_type: str = "cosine"

    # Loss weights
    ar_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    vlm_microbatches: int = 0
    diffusion_microbatches: int = 4

    # Mixed precision
    compute_dtype: str = "bfloat16"

    # FSDP2
    fsdp2_enabled: bool = True

    # Checkpoint
    save_every_steps: int = 1000
    resume_from: str | None = None

    # Logging
    log_every: int = 10
    wandb_enabled: bool = False
    wandb_project: str = "wm-raw"
    wandb_name: str = ""

    # Condition text
    condition_prefix: str = "Caption: "
    condition_suffix: str = " <|wm_predict_image|>"
    text_condition_dropout_prob: float = 0.0

    # Trainable mode
    train_diffusion_backbone: bool = True
    trainable_mode: str = "diffusion"  # "diffusion" | "all" | "vlm"

    # EMA
    ema_enabled: bool = False
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 0


# ---------------------------------------------------------------------------
# Distributed utilities
# ---------------------------------------------------------------------------


@dataclass
class DistContext:
    """Distributed context wrapper."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistContext:
    """Initialize distributed process group and return context."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return DistContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# FSDP2 wrapping
# ---------------------------------------------------------------------------


def apply_fsdp2(
    model: WorldModel,
    *,
    ctx: DistContext,
    mp_policy: Any | None = None,
) -> WorldModel:
    """Apply FSDP2 per-layer wrapping to WorldModel.

    Wrapping strategy (matches training_code pattern):
    - Each VLM decoder layer is a shard unit
    - Each diffusion decoder layer is a shard unit
    - Cross-attention adapters are shard units
    - Root model gets final fully_shard call
    """
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    # Shard VLM decoder layers
    for layer in model.vlm.layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard VLM vision encoder blocks
    if hasattr(model.vlm, "vision_encoder") and model.vlm.vision_encoder is not None:
        for block in model.vlm.vision_encoder.blocks:
            fully_shard(block, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard VLM branch as a unit (captures embed_tokens, norm, lm_head)
    fully_shard(model.vlm, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard diffusion decoder layers
    for layer in model.state_diffusion.layers:
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard diffusion branch as a unit (captures input_proj, time_embedder, output_head, etc.)
    fully_shard(model.state_diffusion, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard cross-attention adapters
    for adapter in model.cross_attention.adapters:
        fully_shard(adapter, mp_policy=mp_policy, reshard_after_forward=True)

    # Shard cross-attention as a unit
    fully_shard(model.cross_attention, mp_policy=mp_policy, reshard_after_forward=True)

    # Root model
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=False)
    return model


# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------


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
        raw_model.state_diffusion.position_embedding,
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


# ---------------------------------------------------------------------------
# Trainable parameter configuration
# ---------------------------------------------------------------------------


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
            model.state_diffusion.position_embedding,
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


# ---------------------------------------------------------------------------
# VAE codec
# ---------------------------------------------------------------------------


class FrozenVAECodec:
    """Frozen VAE encoder for converting pixels → latent tokens."""

    def __init__(self, ae: nn.Module, device: torch.device, dtype: torch.dtype):
        self.ae = ae
        self.device = device
        self.dtype = dtype
        self.ae.eval()
        for p in self.ae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, pixel_values: Tensor) -> Tensor:
        """Encode [B, 3, H, W] → [B, H*W, C] flat latent tokens."""
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        # Standard VAE encode: returns [B, C, h, w]
        latent = self.ae.encode(pixel_values)
        if hasattr(latent, "latent_dist"):
            latent = latent.latent_dist.sample()
        elif hasattr(latent, "sample"):
            latent = latent.sample()
        # Reshape [B, C, h, w] → [B, h*w, C]
        b, c, h, w = latent.shape
        return latent.permute(0, 2, 3, 1).reshape(b, h * w, c)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: WorldModel,
    batch: dict[str, Any],
    *,
    codec: FrozenVAECodec,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, float]:
    """One training step: forward + backward + optimizer step.

    Returns dict of scalar metrics.
    """
    optimizer.zero_grad(set_to_none=True)

    # Encode images to latents
    vae_pixels = batch["vae_pixel_values"].to(device=device)
    state_target = codec.encode(vae_pixels)  # [B, H*W, C]

    # Condition tokens
    condition = {k: v.to(device=device) for k, v in batch["condition"].items() if torch.is_tensor(v)}

    # Build 3D position IDs for MRoPE
    seq_len = condition["input_ids"].shape[1]
    batch_size = condition["input_ids"].shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]

    # Forward with autocast
    with torch.amp.autocast("cuda", dtype=compute_dtype):
        # VLM forward (condition pass, no AR loss for diffusion-only)
        vlm_output = model.forward_vlm(
            input_ids=condition["input_ids"],
            attention_mask=condition.get("attention_mask"),
            position_ids=position_ids,
        )

        # Diffusion forward
        diff_output = model.forward_diffusion(
            state_target=state_target.to(dtype=compute_dtype),
            vlm_hidden_states=vlm_output.hidden_states,
        )

    loss = diff_output.loss * config.diffusion_loss_weight
    loss.backward()

    # Gradient clipping
    if config.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )

    optimizer.step()
    scheduler.step()

    return {
        "loss": loss.item(),
        "diffusion_loss": diff_output.loss.item(),
        "lr": scheduler.get_last_lr()[0],
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: WorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    output_dir: Path,
    *,
    ctx: DistContext,
) -> None:
    """Save a training checkpoint (rank 0 only for non-FSDP, DCP for FSDP)."""
    checkpoint_dir = output_dir / "checkpoints" / f"step-{step:06d}"
    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # For FSDP2, use distributed checkpoint
    try:
        from torch.distributed.checkpoint import save
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            StateDictOptions,
        )

        model_sd = get_model_state_dict(model)
        optim_sd = get_optimizer_state_dict(model, optimizer)
        state = {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": scheduler.state_dict(),
            "step": step,
        }
        save(state, checkpoint_dir=str(checkpoint_dir))
    except (ImportError, RuntimeError):
        # Fallback: single-file checkpoint (rank 0)
        if ctx.is_main:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            }
            torch.save(state, checkpoint_dir / "checkpoint.pt")

    if ctx.is_main:
        logger.info(f"Saved checkpoint at step {step}")


def load_checkpoint(
    path: str | Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> int:
    """Load a training checkpoint. Returns the global step."""
    path = Path(path)

    # Try DCP first
    if (path / "metadata.pt").exists() or (path / ".metadata").exists():
        from torch.distributed.checkpoint import load
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            set_optimizer_state_dict,
        )

        state = {"model": {}, "optimizer": {}, "scheduler": {}, "step": 0}
        load(state, checkpoint_dir=str(path))
        set_model_state_dict(model, state["model"])
        if optimizer is not None and state.get("optimizer"):
            set_optimizer_state_dict(model, optimizer, state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        return int(state.get("step", 0))

    # Fallback: single file
    ckpt_file = path / "checkpoint.pt" if path.is_dir() else path
    state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(config: TrainingConfig) -> None:
    """Full training entry point.

    Usage:
        torchrun --nproc_per_node=8 -m wm_raw.training --config config.yaml
    """
    # Setup
    ctx = setup_distributed()
    if ctx.is_main:
        logger.info(f"Training on {ctx.world_size} GPUs")

    output_dir = Path(config.output_dir)
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = getattr(torch, config.compute_dtype, torch.bfloat16)

    # Build model
    model_config = WorldModelConfig()  # Use defaults (4B VLM + 2B diffusion)
    model = WorldModel(model_config)

    # Load pretrained weights
    if config.vlm_path:
        from .checkpoint import load_vlm_weights
        report = load_vlm_weights(model, config.vlm_path, dtype=compute_dtype)
        if ctx.is_main:
            logger.info(report.format())

    if config.diffusion_path:
        from .checkpoint import load_diffusion_weights
        report = load_diffusion_weights(model, config.diffusion_path, dtype=compute_dtype)
        if ctx.is_main:
            logger.info(report.format())

    model = model.to(device=ctx.device, dtype=compute_dtype)

    # Configure trainable parameters
    configure_trainable(model, config)

    # FSDP2
    if config.fsdp2_enabled and ctx.world_size > 1:
        model = apply_fsdp2(model, ctx=ctx)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Resume
    global_step = 0
    if config.resume_from:
        global_step = load_checkpoint(config.resume_from, model, optimizer, scheduler)
        if ctx.is_main:
            logger.info(f"Resumed from step {global_step}")

    # Data
    from .data import DiffusionCollator, ImageCaptionDataset, load_manifest
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.vlm_path, trust_remote_code=True)
    records = load_manifest(
        config.train_manifest,
        image_root=config.image_root or None,
        max_samples=config.max_samples,
        seed=config.seed,
        shuffle=True,
    )
    dataset = ImageCaptionDataset(records)
    sampler = DistributedSampler(
        dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=config.seed
    )
    collator = DiffusionCollator(
        processor=processor,
        image_size=config.image_size,
        condition_prefix=config.condition_prefix,
        condition_suffix=config.condition_suffix,
        text_condition_dropout_prob=config.text_condition_dropout_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # VAE
    codec: FrozenVAECodec | None = None
    if config.vae_path:
        from .checkpoint import load_vae
        ae = load_vae(config.vae_path, device=ctx.device, dtype=compute_dtype)
        codec = FrozenVAECodec(ae, device=ctx.device, dtype=compute_dtype)

    if codec is None:
        raise RuntimeError("VAE path required for diffusion training")

    # Metrics file
    metrics_path = output_dir / "train_metrics.jsonl" if ctx.is_main else None

    # WandB
    if ctx.is_main and config.wandb_enabled:
        try:
            import wandb

            wandb.init(
                project=config.wandb_project,
                name=config.wandb_name or None,
                config={
                    "max_steps": config.max_steps,
                    "batch_size": config.batch_size,
                    "adapter_lr": config.adapter_lr,
                    "diffusion_lr": config.diffusion_lr,
                    "vlm_lr": config.vlm_lr,
                    "trainable_mode": config.trainable_mode,
                    "compute_dtype": config.compute_dtype,
                },
            )
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            config.wandb_enabled = False

    # EMA
    ema = None
    if config.ema_enabled:
        from .utils.ema import EMAManager

        ema = EMAManager(model, decay=config.ema_decay, warmup_steps=config.ema_warmup_steps)

    # Training loop
    model.train()
    data_iter: Iterator = iter([])
    epoch = 0

    if ctx.is_main:
        logger.info(f"Starting training: max_steps={config.max_steps}, batch_size={config.batch_size}")

    while global_step < config.max_steps:
        # Reload data iterator on exhaustion
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        step_start = time.perf_counter()
        metrics = train_step(
            model,
            batch,
            codec=codec,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=ctx.device,
            compute_dtype=compute_dtype,
        )
        global_step += 1
        step_time = time.perf_counter() - step_start

        # EMA update
        if ema is not None:
            ema.update()

        # Logging
        if ctx.is_main and global_step % config.log_every == 0:
            metrics["step"] = global_step
            metrics["epoch"] = epoch
            metrics["step_time_s"] = step_time
            logger.info(
                f"step={global_step} loss={metrics['loss']:.4f} "
                f"diff_loss={metrics['diffusion_loss']:.4f} "
                f"lr={metrics['lr']:.2e} time={step_time:.2f}s"
            )
            if metrics_path:
                with metrics_path.open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
            if config.wandb_enabled:
                import wandb

                wandb.log(metrics, step=global_step)

        # Checkpointing
        if global_step % config.save_every_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, output_dir, ctx=ctx)

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, output_dir, ctx=ctx)
    cleanup_distributed()
    if ctx.is_main:
        if config.wandb_enabled:
            import wandb

            wandb.finish()
        logger.info(f"Training complete. Final step: {global_step}")
