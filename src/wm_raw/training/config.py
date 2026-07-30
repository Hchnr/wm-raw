"""Training configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


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
    hsdp_enabled: bool = False  # Hybrid: shard within node, replicate across nodes

    # torch.compile
    compile_enabled: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"

    # Checkpoint
    save_every_steps: int = 1000
    keep_last_n: int | None = None  # Keep only last N checkpoints (None = keep all)
    save_final: bool = True  # Save checkpoint at end of training
    resume_from: str | None = None
    # Resume mode: "auto" (detect), "wm_raw" (own DCP, restore optimizer+scheduler),
    # "wm_training" (online DCP, model weights only)
    resume_mode: str = "auto"

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
    ema_update_every: int = 1

    # Dry run (skip pretrained weight loading)
    skip_pretrained: bool = False
