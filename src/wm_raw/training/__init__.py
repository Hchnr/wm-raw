"""Training package for wm-raw WorldModel.

Supports FSDP2, mixed precision, per-group learning rates, checkpointing,
and resolution-bucket data loading.
"""

from .checkpointing import load_checkpoint, save_checkpoint
from .config import TrainingConfig
from .distributed import DistContext, apply_fsdp2, cleanup_distributed, setup_distributed
from .loop import run_training
from .optim import build_optimizer, build_scheduler, configure_trainable
from .step import FrozenVAECodec, train_step

__all__ = [
    "TrainingConfig",
    "run_training",
    "DistContext",
    "setup_distributed",
    "cleanup_distributed",
    "apply_fsdp2",
    "build_optimizer",
    "build_scheduler",
    "configure_trainable",
    "FrozenVAECodec",
    "train_step",
    "save_checkpoint",
    "load_checkpoint",
]
