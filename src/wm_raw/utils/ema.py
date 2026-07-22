"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of model parameters with exponential decay.
Used for evaluation/inference without affecting training dynamics.
"""

from __future__ import annotations

import copy
import logging
from typing import Iterable

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class EMAManager:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMAManager(model, decay=0.9999)
        # After each optimizer step:
        ema.update()
        # For evaluation:
        with ema.average_parameters():
            evaluate(model)
        # For checkpointing:
        ema.state_dict() / ema.load_state_dict(...)

    Attributes:
        decay: EMA decay coefficient (higher = slower update)
        num_updates: number of update() calls made
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        trainable_only: bool = True,
    ) -> None:
        """Initialize EMA with a copy of model parameters.

        Args:
            model: the model whose parameters to track
            decay: EMA decay coefficient
            warmup_steps: linearly ramp decay from 0 to target over this many steps
            trainable_only: only track parameters with requires_grad=True
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self._model = model
        self._trainable_only = trainable_only

        # Store shadow parameters
        self._shadow: list[Tensor] = []
        self._params: list[nn.Parameter] = []

        for param in model.parameters():
            if trainable_only and not param.requires_grad:
                continue
            self._params.append(param)
            self._shadow.append(param.data.clone().detach())

        logger.info(
            "EMA initialized: tracking %d params, decay=%.6f, warmup=%d",
            len(self._shadow),
            decay,
            warmup_steps,
        )

    def _get_decay(self) -> float:
        """Get current decay, accounting for warmup."""
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            # Linear warmup from 0 to target decay
            return self.decay * (self.num_updates / self.warmup_steps)
        return self.decay

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        decay = self._get_decay()
        self.num_updates += 1

        for shadow, param in zip(self._shadow, self._params):
            # shadow = decay * shadow + (1 - decay) * param
            shadow.lerp_(param.data, 1.0 - decay)

    def apply_shadow(self) -> list[Tensor]:
        """Swap model parameters with EMA shadow. Returns original params for restore."""
        originals = []
        for shadow, param in zip(self._shadow, self._params):
            originals.append(param.data.clone())
            param.data.copy_(shadow)
        return originals

    def restore(self, originals: list[Tensor]) -> None:
        """Restore original model parameters after apply_shadow."""
        for original, param in zip(originals, self._params):
            param.data.copy_(original)

    class _AverageContext:
        """Context manager for temporarily applying EMA parameters."""

        def __init__(self, ema: "EMAManager") -> None:
            self._ema = ema
            self._originals: list[Tensor] = []

        def __enter__(self) -> None:
            self._originals = self._ema.apply_shadow()

        def __exit__(self, *args) -> None:
            self._ema.restore(self._originals)

    def average_parameters(self) -> _AverageContext:
        """Context manager to temporarily use EMA parameters for evaluation.

        Example:
            with ema.average_parameters():
                val_loss = evaluate(model)
        """
        return self._AverageContext(self)

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "shadow": [s.cpu() for s in self._shadow],
            "decay": self.decay,
            "num_updates": self.num_updates,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state from checkpoint."""
        shadows = state["shadow"]
        if len(shadows) != len(self._shadow):
            raise ValueError(
                f"EMA state_dict has {len(shadows)} params, "
                f"but model has {len(self._shadow)} tracked params"
            )
        for i, s in enumerate(shadows):
            self._shadow[i].copy_(s.to(self._shadow[i].device))
        self.decay = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        logger.info("EMA state restored: num_updates=%d", self.num_updates)
