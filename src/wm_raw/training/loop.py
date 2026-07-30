"""Main training loop."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..config import WorldModelConfig
from ..models import WorldModel
from .checkpointing import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .distributed import DistContext, apply_fsdp2, cleanup_distributed, setup_distributed
from .optim import build_optimizer, build_scheduler, configure_trainable
from .step import FrozenVAECodec, train_step

logger = logging.getLogger(__name__)


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
            from ..checkpoint import load_vlm_weights
            report = load_vlm_weights(model, config.vlm_path, dtype=compute_dtype)
            if ctx.is_main:
                logger.info(report.format())

        if config.diffusion_path and not config.skip_pretrained:
            if ctx.is_main:
                logger.info("Loading diffusion weights from %s ...", config.diffusion_path)
            from ..checkpoint import load_diffusion_weights
            report = load_diffusion_weights(model, config.diffusion_path, dtype=compute_dtype)
            if ctx.is_main:
                logger.info(report.format())
    else:
        # Load from online DCP (before FSDP wrapping, model weights only)
        if ctx.is_main:
            logger.info("Loading online DCP checkpoint from %s ...", config.resume_from)
        from ..checkpoint import load_online_dcp_weights
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
        model = apply_fsdp2(model, ctx=ctx, hsdp=config.hsdp_enabled)

    # torch.compile
    if config.compile_enabled:
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
    from ..data import DiffusionCollator, ImageCaptionDataset, PreparedImageCaptionDataset, load_manifest
    from ..data.resolution_buckets import ResolutionBucketBatchSampler, find_bucket_assignment_path
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.vlm_path, trust_remote_code=True)

    # Parse resolution bucket config (if present in YAML)
    bucket_sizes: list[tuple[int, int]] | None = None
    assignment_path = None

    if config.dataset_type == "wm_sequence_prepared":
        if not config.prepared_root:
            raise ValueError("data.prepared_root is required for dataset_type=wm_sequence_prepared")

        # Try to find bucket assignment cache for 512px buckets
        bucket_sizes_512 = [
            (512, 512), (448, 608), (608, 448),
            (416, 640), (640, 416), (384, 704), (704, 384),
        ]
        if config.image_size >= 512:
            assignment_path = find_bucket_assignment_path(config.prepared_root, bucket_sizes_512)
            if assignment_path is not None:
                bucket_sizes = bucket_sizes_512
                if ctx.is_main:
                    logger.info(f"Using resolution buckets: {bucket_sizes}")
                    logger.info(f"  assignment cache: {assignment_path}")

        dataset = PreparedImageCaptionDataset(
            config.prepared_root,
            image_size=config.image_size,
            center_crop=False,  # online uses center_crop=false with buckets
            max_samples=config.max_samples,
            bucket_sizes=bucket_sizes,
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

    # Build sampler: bucket sampler if available, otherwise standard distributed
    sampler: Any = None  # epoch-aware sampler (DistributedSampler or BucketBatchSampler)
    if bucket_sizes is not None and assignment_path is not None:
        batch_sampler = ResolutionBucketBatchSampler(
            assignment_path=assignment_path,
            dataset_size=len(dataset),
            bucket_sizes=bucket_sizes,
            batch_size=config.batch_size,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            seed=config.seed,
            shuffle=True,
        )
        sampler = batch_sampler
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
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
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
        from ..checkpoint import load_vae
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
        from ..utils.ema import EMAManager

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
