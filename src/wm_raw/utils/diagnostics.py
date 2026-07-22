"""Tensor statistics and logging utilities for debugging.

Provides lightweight diagnostic helpers for inspecting tensor values,
gradient norms, and parameter counts during training.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def log_tensor_stats(
    name: str,
    tensor: Tensor,
    *,
    level: int = logging.DEBUG,
) -> dict[str, float]:
    """Log basic statistics of a tensor and return them as a dict.

    Args:
        name: identifier for the tensor (e.g., "vlm_hidden.layer_12")
        tensor: the tensor to inspect
        level: logging level

    Returns:
        Dict with keys: mean, std, min, max, norm, shape
    """
    with torch.no_grad():
        t = tensor.detach().float()
        stats = {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "min": t.min().item(),
            "max": t.max().item(),
            "norm": t.norm().item(),
        }

    shape_str = "x".join(str(s) for s in tensor.shape)
    logger.log(
        level,
        "%s [%s %s]: mean=%.4e std=%.4e min=%.4e max=%.4e norm=%.4e",
        name,
        shape_str,
        tensor.dtype,
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["max"],
        stats["norm"],
    )
    return stats


def param_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Count total parameters in a module.

    Args:
        module: PyTorch module
        trainable_only: if True, only count parameters with requires_grad=True

    Returns:
        Total number of parameters (elements)
    """
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def summarize_gradients(
    model: nn.Module,
    *,
    prefix_groups: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-group gradient norm statistics.

    Args:
        model: model after backward pass
        prefix_groups: mapping from group_name → param name prefix.
            If None, uses default groups: vlm, state_diffusion, cross_attention.

    Returns:
        Dict of group_name → {grad_norm, param_norm, ratio, num_params}
    """
    if prefix_groups is None:
        prefix_groups = {
            "vlm": "vlm.",
            "diffusion": "state_diffusion.",
            "cross_attn": "cross_attention.",
        }

    results: dict[str, dict[str, float]] = {}

    for group_name, prefix in prefix_groups.items():
        grad_sq_sum = 0.0
        param_sq_sum = 0.0
        count = 0

        for name, param in model.named_parameters():
            if not name.startswith(prefix):
                continue
            if param.grad is not None:
                grad_sq_sum += param.grad.detach().float().norm().item() ** 2
                param_sq_sum += param.detach().float().norm().item() ** 2
                count += 1

        if count > 0:
            grad_norm = grad_sq_sum**0.5
            param_norm = param_sq_sum**0.5
            results[group_name] = {
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "ratio": grad_norm / max(param_norm, 1e-8),
                "num_params": count,
            }
        else:
            results[group_name] = {
                "grad_norm": 0.0,
                "param_norm": 0.0,
                "ratio": 0.0,
                "num_params": 0,
            }

    return results


def format_param_summary(model: nn.Module) -> str:
    """Format a human-readable parameter summary by top-level submodule.

    Returns a multi-line string showing total/trainable param counts per submodule.
    """
    lines = []
    total = 0
    total_trainable = 0

    for name, child in model.named_children():
        n_all = param_count(child, trainable_only=False)
        n_train = param_count(child, trainable_only=True)
        total += n_all
        total_trainable += n_train
        lines.append(f"  {name}: {n_all / 1e6:.1f}M total, {n_train / 1e6:.1f}M trainable")

    header = f"Model: {total / 1e6:.1f}M total, {total_trainable / 1e6:.1f}M trainable"
    return "\n".join([header] + lines)
