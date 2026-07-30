"""Distributed training utilities: context, process group, FSDP2 wrapping."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from ..models import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class DistContext:
    """Distributed context wrapper."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    local_world_size: int = 1  # GPUs per node

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.local_world_size


def setup_distributed() -> DistContext:
    """Initialize distributed process group and return context.

    If torchrun env vars aren't set, falls back to single-GPU mode.
    """
    if "RANK" not in os.environ:
        # Single-GPU fallback (no torchrun)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return DistContext(rank=0, world_size=1, local_rank=0, device=device, local_world_size=1)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return DistContext(
        rank=rank, world_size=world_size, local_rank=local_rank,
        device=device, local_world_size=local_world_size,
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def apply_fsdp2(
    model: WorldModel,
    *,
    ctx: DistContext,
    mp_policy: Any | None = None,
    hsdp: bool = False,
) -> WorldModel:
    """Apply FSDP2 per-layer wrapping to WorldModel.

    Wrapping strategy (matches training_code pattern):
    - Each VLM decoder layer is a shard unit
    - Each diffusion decoder layer is a shard unit
    - Cross-attention adapters are shard units
    - Root model gets final fully_shard call

    When hsdp=True, uses Hybrid Sharded Data Parallel:
    - Intra-node: FSDP (shard parameters across GPUs within a node)
    - Inter-node: replicate (allreduce gradients across nodes)
    This reduces cross-node communication volume compared to pure FSDP.
    """
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    # Build device mesh for HSDP if requested
    mesh = None
    if hsdp and ctx.num_nodes > 1:
        from torch.distributed.device_mesh import init_device_mesh
        # 2-D mesh: (replicate across nodes, shard within node)
        mesh = init_device_mesh(
            "cuda",
            (ctx.num_nodes, ctx.local_world_size),
            mesh_dim_names=("replicate", "shard"),
        )
        if ctx.is_main:
            logger.info(
                f"HSDP enabled: {ctx.num_nodes} nodes × {ctx.local_world_size} GPUs/node, "
                f"shard within node, replicate across nodes"
            )

    # Common kwargs for fully_shard
    def _shard(module: Any, reshard: bool = True) -> None:
        kwargs: dict[str, Any] = {
            "mp_policy": mp_policy,
            "reshard_after_forward": reshard,
        }
        if mesh is not None:
            kwargs["mesh"] = mesh["shard"]
        fully_shard(module, **kwargs)

    # Shard VLM decoder layers
    for layer in model.vlm.layers:
        _shard(layer)

    # Shard VLM vision encoder blocks
    if hasattr(model.vlm, "vision_encoder") and model.vlm.vision_encoder is not None:
        for block in model.vlm.vision_encoder.blocks:
            _shard(block)

    # Shard VLM branch as a unit (captures embed_tokens, norm, lm_head)
    _shard(model.vlm)

    # Shard diffusion decoder layers
    for layer in model.state_diffusion.layers:
        _shard(layer)

    # Shard diffusion branch as a unit
    _shard(model.state_diffusion)

    # Shard cross-attention adapters
    for adapter in model.cross_attention.adapters:
        _shard(adapter)

    # Shard cross-attention as a unit
    _shard(model.cross_attention)

    # Root model (reshard_after_forward=False for root)
    _shard(model, reshard=False)

    return model
