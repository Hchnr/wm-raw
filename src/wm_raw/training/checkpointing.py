"""Training checkpoint save/load utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ..models import WorldModel
from .distributed import DistContext

logger = logging.getLogger(__name__)


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
    from ..checkpoint import CheckpointReport

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
