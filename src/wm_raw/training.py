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
    dataset_type: str = "manifest"  # "manifest" | "wm_sequence_prepared"
    prepared_root: str = ""

    # Optimizer
    adapter_lr: float = 1e-4
    diffusion_lr: float = 1e-5
    vlm_lr: float = 1e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_fused: bool = True
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

    # torch.compile
    compile_enabled: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"

    # Checkpoint
    save_every_steps: int = 1000
    keep_last_n: int | None = None  # Keep only last N checkpoints (None = keep all)
    save_final: bool = True  # Save checkpoint at end of training
    resume_from: str | None = None

    # Logging
    log_every: int = 10
    log_dir: str | None = None  # Separate log directory (default: <output_dir>/logs)
    wandb_enabled: bool = False
    wandb_project: str = "wm-raw"
    wandb_entity: str = ""
    wandb_name: str = ""

    # Condition text
    condition_prefix: str = "Caption: "
    condition_suffix: str = " <|wm_predict_image|>"
    text_condition_dropout_prob: float = 0.0
    condition_max_seq_len: int | None = None  # Fixed length for CUDA Graph compatibility

    # Trainable mode
    train_diffusion_backbone: bool = True
    trainable_mode: str = "diffusion"  # "diffusion" | "all" | "vlm"

    # EMA
    ema_enabled: bool = False
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 0

    # Dry run (skip pretrained weight loading)
    skip_pretrained: bool = False


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
    """Initialize distributed process group and return context.

    If torchrun env vars aren't set, falls back to single-GPU mode.
    """
    if "RANK" not in os.environ:
        # Single-GPU fallback (no torchrun)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return DistContext(rank=0, world_size=1, local_rank=0, device=device)

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

    # Derive latent spatial dimensions from VAE output
    # VAE downsamples by 8x: 512px → 64 latent, 448px → 56, etc.
    # state_target shape is [B, H*W, C] — we need H and W separately
    # For fixed image_size: H = W = image_size / 8
    # For variable (buckets): get from batch metadata
    if "image_height" in batch and "image_width" in batch:
        latent_h = batch["image_height"] // 8
        latent_w = batch["image_width"] // 8
    else:
        # Infer from state_target: assume square if not specified
        num_latent_pixels = state_target.shape[1]
        latent_side = int(num_latent_pixels ** 0.5)
        assert latent_side * latent_side == num_latent_pixels, (
            f"Cannot infer latent dimensions from {num_latent_pixels} pixels. "
            f"Pass image_height/image_width in batch."
        )
        latent_h = latent_w = latent_side

    # Condition tokens
    condition = {k: v.to(device=device) for k, v in batch["condition"].items() if torch.is_tensor(v)}

    # Build 3D position IDs for MRoPE
    seq_len = condition["input_ids"].shape[1]
    batch_size = condition["input_ids"].shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]

    # Build causal attention mask [B, 1, S, S]
    # Start from padding mask [B, S] (1 = attend, 0 = ignore)
    padding_mask = condition.get("attention_mask")
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=compute_dtype),
        diagonal=1,
    )  # [S, S] upper-triangular = -inf
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)  # [B, 1, S, S]
    if padding_mask is not None and not padding_mask.all():
        # Mask out padded positions in the key dimension
        # Use where() to avoid -inf * 0 = nan
        pad_expanded = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        causal_mask = causal_mask.clone()
        causal_mask.masked_fill_(~pad_expanded, float("-inf"))

    # Forward with autocast
    with torch.amp.autocast("cuda", dtype=compute_dtype):
        output = model({
            "task_type": "diffusion",
            "condition": {
                "input_ids": condition["input_ids"],
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            },
            "state_target": state_target.to(dtype=compute_dtype),
            "latent_h": latent_h,
            "latent_w": latent_w,
        })

    loss = output.diffusion_loss * config.diffusion_loss_weight
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
        "diffusion_loss": output.diffusion_loss.item(),
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
    keep_last_n: int | None = None,
) -> None:
    """Save a training checkpoint (rank 0 only for non-FSDP, DCP for FSDP)."""
    checkpoint_dir = output_dir / "checkpoints" / f"step-{step:06d}"
    if ctx.is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # For FSDP2, use distributed checkpoint
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            StateDictOptions,
        )

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=options)
        optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
        state = {
            "model": model_sd,
            "optimizer": optim_sd,
            "scheduler": scheduler.state_dict(),
            "step": step,
        }
        dcp.save(state, checkpoint_id=str(checkpoint_dir))
    except (ImportError, RuntimeError, TypeError):
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

    # Cleanup old checkpoints (rank 0 only)
    if keep_last_n is not None and ctx.is_main:
        ckpt_root = output_dir / "checkpoints"
        existing = sorted(ckpt_root.iterdir()) if ckpt_root.exists() else []
        existing = [d for d in existing if d.is_dir() and d.name.startswith("step-")]
        if len(existing) > keep_last_n:
            import shutil

            for old_dir in existing[: len(existing) - keep_last_n]:
                shutil.rmtree(old_dir)
                logger.info(f"Removed old checkpoint: {old_dir.name}")


def load_checkpoint(
    path: str | Path,
    model: WorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> int:
    """Load a training checkpoint. Returns the global step.

    Supports:
    - DCP (distributed checkpoint) with automatic resharding across different
      world sizes (e.g. saved on 8 GPUs, resumed on 2 GPUs).
    - Single-file checkpoint fallback for non-FSDP runs.

    In both paths, validates structure alignment between the checkpoint and the
    current model. Logs ERROR for any missing, unexpected, or shape-mismatched
    keys before proceeding with the loadable subset.
    """
    from .checkpoint import CheckpointReport

    path = Path(path)

    # Try DCP first (FSDP2 distributed checkpoint)
    if (path / ".metadata").exists():
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            get_optimizer_state_dict,
            set_model_state_dict,
            set_optimizer_state_dict,
            StateDictOptions,
        )

        # --- Structure validation: compare DCP metadata keys vs model keys ---
        reader = FileSystemReader(str(path))
        metadata = reader.read_metadata()
        ckpt_model_keys = set()
        ckpt_model_shapes: dict[str, tuple] = {}
        for k, md in metadata.state_dict_metadata.items():
            if k.startswith("model."):
                stripped = k[len("model."):]
                ckpt_model_keys.add(stripped)
                if isinstance(md, TensorStorageMetadata):
                    ckpt_model_shapes[stripped] = tuple(md.size)

        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())

        missing_keys = sorted(model_keys - ckpt_model_keys)
        unexpected_keys = sorted(ckpt_model_keys - model_keys)
        shape_mismatch: list[str] = []
        matched_count = 0

        for key in sorted(model_keys & ckpt_model_keys):
            if key in ckpt_model_shapes:
                model_shape = tuple(model_sd[key].shape)
                if model_shape != ckpt_model_shapes[key]:
                    shape_mismatch.append(
                        f"{key}: model={list(model_shape)} vs ckpt={list(ckpt_model_shapes[key])}"
                    )
                else:
                    matched_count += 1
            else:
                matched_count += 1  # non-tensor metadata, assume ok

        report = CheckpointReport(
            matched=matched_count,
            missing=tuple(missing_keys),
            unexpected=tuple(unexpected_keys),
            shape_mismatch=tuple(shape_mismatch),
        )
        logger.info(f"DCP checkpoint structure check: {report.format()}")
        report.log_errors(context="load_checkpoint (DCP)")

        # --- Proceed with DCP loading ---
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)

        # Build target state dicts that DCP can reshard into.
        # These carry the current FSDP sharding info so DCP knows how to
        # redistribute tensors from a different world size.
        model_state = get_model_state_dict(model, options=options)
        payload: dict[str, Any] = {"model": model_state}

        if optimizer is not None:
            optim_state = get_optimizer_state_dict(model, optimizer, options=options)
            payload["optimizer"] = optim_state

        # Scheduler and step are non-sharded scalars stored in the same DCP
        payload["scheduler"] = {}
        payload["step"] = 0

        dcp.load(payload, checkpoint_id=str(path))

        # Apply loaded state back to model/optimizer
        set_model_state_dict(model, payload["model"], options=options)
        if optimizer is not None and payload.get("optimizer"):
            set_optimizer_state_dict(
                model, optimizer, payload["optimizer"], options=options
            )
        if scheduler is not None and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])

        step = int(payload.get("step", 0))
        logger.info(f"Resumed from DCP checkpoint: {path} at step {step}")
        return step

    # Fallback: single file (non-FSDP)
    ckpt_file = path / "checkpoint.pt" if path.is_dir() else path
    state = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)

    # --- Structure validation for single-file checkpoint ---
    ckpt_sd = state.get("model", {})
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(ckpt_sd.keys())

    missing_keys_sf = sorted(model_keys - ckpt_keys)
    unexpected_keys_sf = sorted(ckpt_keys - model_keys)
    shape_mismatch_sf: list[str] = []
    matched_keys_sf: list[str] = []

    for key in sorted(model_keys & ckpt_keys):
        if model_sd[key].shape != ckpt_sd[key].shape:
            shape_mismatch_sf.append(
                f"{key}: model={list(model_sd[key].shape)} vs ckpt={list(ckpt_sd[key].shape)}"
            )
        else:
            matched_keys_sf.append(key)

    report = CheckpointReport(
        matched=len(matched_keys_sf),
        missing=tuple(missing_keys_sf),
        unexpected=tuple(unexpected_keys_sf),
        shape_mismatch=tuple(shape_mismatch_sf),
    )
    logger.info(f"Single-file checkpoint structure check: {report.format()}")
    report.log_errors(context="load_checkpoint (single-file)")

    # Load only matched keys
    load_dict = {k: ckpt_sd[k] for k in matched_keys_sf}
    model.load_state_dict(load_dict, strict=False)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    step = int(state.get("step", 0))
    logger.info(f"Resumed from single-file checkpoint: {ckpt_file} at step {step}")
    return step


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
    if ctx.is_main:
        logger.info("Building model...")
    model_config = WorldModelConfig()  # Use defaults (4B VLM + 2B diffusion)
    model = WorldModel(model_config)
    if ctx.is_main:
        logger.info("Model built")

    # Detect if resume_from is an online (wm-training) DCP checkpoint
    _is_online_resume = False
    if config.resume_from:
        _resume_path = Path(config.resume_from)
        if (_resume_path / ".metadata").exists():
            from torch.distributed.checkpoint import FileSystemReader
            _reader = FileSystemReader(str(_resume_path))
            _meta = _reader.read_metadata()
            _sample_keys = list(_meta.state_dict_metadata.keys())[:5]
            _is_online_resume = any("vlm_branch" in k for k in _sample_keys)

    # Load pretrained weights (skip if resuming from online DCP — it already has them)
    if not _is_online_resume:
        if config.vlm_path and not config.skip_pretrained:
            if ctx.is_main:
                logger.info("Loading VLM weights from %s ...", config.vlm_path)
            from .checkpoint import load_vlm_weights
            report = load_vlm_weights(model, config.vlm_path, dtype=compute_dtype)
            if ctx.is_main:
                logger.info(report.format())

        if config.diffusion_path and not config.skip_pretrained:
            if ctx.is_main:
                logger.info("Loading diffusion weights from %s ...", config.diffusion_path)
            from .checkpoint import load_diffusion_weights
            report = load_diffusion_weights(model, config.diffusion_path, dtype=compute_dtype)
            if ctx.is_main:
                logger.info(report.format())
    else:
        # Load from online DCP (before FSDP wrapping, model weights only)
        if ctx.is_main:
            logger.info("Loading online DCP checkpoint from %s ...", config.resume_from)
        from .checkpoint import load_online_dcp_weights
        report = load_online_dcp_weights(model, config.resume_from, dtype=compute_dtype)
        if ctx.is_main:
            logger.info(report.format())

    if ctx.is_main:
        logger.info("Moving model to device %s ...", ctx.device)
    model = model.to(device=ctx.device, dtype=compute_dtype)

    # Configure trainable parameters
    configure_trainable(model, config)

    # FSDP2
    if config.fsdp2_enabled and ctx.world_size > 1:
        if ctx.is_main:
            logger.info("Applying FSDP2...")
        model = apply_fsdp2(model, ctx=ctx)

    # torch.compile
    if config.compile_enabled:
        # Compile the root model. train_step calls model(batch) which enters
        # model.forward — dynamo traces the full VLM + cross_attention + diffusion
        # graph in one shot, ensuring FSDP2 DTensor dispatch is consistent.
        model = torch.compile(model, mode=config.compile_mode, dynamic=False)
        if ctx.is_main:
            logger.info("torch.compile enabled (mode=%s, dynamic=False)", config.compile_mode)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Resume (only for our own DCP checkpoints — online DCP was loaded above)
    global_step = 0
    if config.resume_from and not _is_online_resume:
        global_step = load_checkpoint(config.resume_from, model, optimizer, scheduler)
        if ctx.is_main:
            logger.info(f"Resumed from step {global_step}")

    # Data
    from .data import DiffusionCollator, ImageCaptionDataset, PreparedImageCaptionDataset, load_manifest
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.vlm_path, trust_remote_code=True)

    if config.dataset_type == "wm_sequence_prepared":
        if not config.prepared_root:
            raise ValueError("data.prepared_root is required for dataset_type=wm_sequence_prepared")
        dataset = PreparedImageCaptionDataset(
            config.prepared_root,
            image_size=config.image_size,
            center_crop=True,
            max_samples=config.max_samples,
        )
    else:
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
        condition_max_seq_len=config.condition_max_seq_len,
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
    log_dir = Path(config.log_dir) if config.log_dir else output_dir
    if ctx.is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "train_metrics.jsonl" if ctx.is_main else None

    # WandB
    if ctx.is_main and config.wandb_enabled:
        try:
            import wandb

            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity or None,
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
                try:
                    import wandb

                    wandb.log(metrics, step=global_step)
                except Exception:
                    pass  # Don't let wandb crash training

        # Checkpointing
        if global_step % config.save_every_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, output_dir, ctx=ctx, keep_last_n=config.keep_last_n)

    # Final save
    if config.save_final:
        save_checkpoint(model, optimizer, scheduler, global_step, output_dir, ctx=ctx, keep_last_n=config.keep_last_n)
    cleanup_distributed()
    if ctx.is_main:
        if config.wandb_enabled:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass
        logger.info(f"Training complete. Final step: {global_step}")
