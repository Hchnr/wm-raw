"""Flow matching diffusion utilities.

Implements:
- Timestep sampling (uniform [0, 1] with optional shift)
- Noise scheduling (linear interpolation for flow matching)
- Flow matching loss (MSE on velocity prediction)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    shift: float = 1.0,
) -> Tensor:
    """Sample timesteps from uniform distribution with optional logit-normal shift.

    For shift=1.0 (default), this is simply U[0,1].
    For shift>1.0, applies rectified flow timestep shifting:
        t_shifted = shift * t / (1 + (shift - 1) * t)

    Args:
        batch_size: number of timesteps to sample
        device: target device
        shift: timestep shift factor (1.0 = no shift)

    Returns:
        timesteps: [B] values in (0, 1)
    """
    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    # Clamp away from exact 0 and 1 for numerical stability
    t = t.clamp(1e-5, 1.0 - 1e-5)

    if shift != 1.0:
        # Rectified flow shift: t → shift*t / (1 + (shift-1)*t)
        t = shift * t / (1.0 + (shift - 1.0) * t)

    return t


def add_flow_noise(
    clean: Tensor,  # [B, S, D]
    noise: Tensor,  # [B, S, D]
    timesteps: Tensor,  # [B]
) -> Tensor:
    """Create noisy sample via linear interpolation (flow matching).

    noisy = (1 - t) * clean + t * noise

    Args:
        clean: clean target tokens [B, S, D]
        noise: Gaussian noise [B, S, D]
        timesteps: [B] values in [0, 1]

    Returns:
        noisy_sample: [B, S, D]
    """
    # Reshape t for broadcasting: [B] → [B, 1, 1]
    t = timesteps[:, None, None]
    return (1.0 - t) * clean + t * noise


def flow_matching_target(
    clean: Tensor,  # [B, S, D]
    noise: Tensor,  # [B, S, D]
) -> Tensor:
    """Compute flow matching velocity target.

    v_target = noise - clean  (derivative of linear interpolation)

    Args:
        clean: clean target tokens [B, S, D]
        noise: Gaussian noise [B, S, D]

    Returns:
        velocity_target: [B, S, D]
    """
    return noise - clean


def flow_matching_loss(
    prediction: Tensor,  # [B, S, D]
    target: Tensor,  # [B, S, D]
    mask: Optional[Tensor] = None,  # [B, S]
) -> Tensor:
    """Compute MSE loss for flow matching velocity prediction.

    Args:
        prediction: predicted velocity [B, S, D]
        target: ground truth velocity [B, S, D]
        mask: optional token-level mask [B, S] (1 = keep, 0 = ignore)

    Returns:
        loss: scalar MSE loss
    """
    error = (prediction.float() - target.float()).pow(2)

    if mask is None:
        return error.mean()

    # Expand mask: [B, S] → [B, S, 1] → broadcast with [B, S, D]
    weight = mask.float().unsqueeze(-1)  # [B, S, 1]
    return (error * weight).sum() / weight.sum().clamp_min(1.0)
