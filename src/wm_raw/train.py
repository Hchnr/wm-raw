"""Training entry point for wm-raw.

Usage:
    torchrun --nproc_per_node=8 -m wm_raw.train --config configs/gpic_image_diffusion.yaml

Or with a single GPU:
    python -m wm_raw.train --config configs/gpic_image_diffusion.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="wm-raw training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    raw_config = load_config(args.config)

    from wm_raw.training import TrainingConfig, run_training

    # Flatten nested YAML sections into TrainingConfig fields
    tc = TrainingConfig()

    # Model paths
    model_cfg = raw_config.get("model", {})
    tc.vlm_path = model_cfg.get("vlm_path", "")
    tc.diffusion_path = model_cfg.get("diffusion_path", "")
    tc.vae_path = raw_config.get("vae", {}).get("model_path", "")

    # Data
    data_cfg = raw_config.get("data", {})
    tc.dataset_type = data_cfg.get("dataset_type", "manifest")
    tc.prepared_root = data_cfg.get("prepared_root", "")
    tc.train_manifest = data_cfg.get("train_manifest", "")
    tc.image_root = data_cfg.get("image_root", "")
    tc.image_size = int(data_cfg.get("image_size", 256))
    tc.max_samples = data_cfg.get("max_samples")
    tc.batch_size = int(data_cfg.get("batch_size", 1))
    tc.num_workers = int(data_cfg.get("num_workers", 4))
    tc.seed = int(data_cfg.get("seed", 42))
    tc.condition_prefix = data_cfg.get("condition_prefix", "Caption: ")
    tc.condition_suffix = data_cfg.get("condition_suffix", " <|wm_predict_image|>")
    tc.text_condition_dropout_prob = float(data_cfg.get("text_condition_dropout_prob", 0.0))

    # Optimizer
    opt_cfg = raw_config.get("optimizer", {})
    tc.adapter_lr = float(opt_cfg.get("adapter_learning_rate", 1e-4))
    tc.diffusion_lr = float(opt_cfg.get("diffusion_learning_rate", 1e-5))
    tc.vlm_lr = float(opt_cfg.get("vlm_learning_rate", 1e-6))
    tc.weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    tc.adam_beta1 = float(opt_cfg.get("adam_beta1", 0.9))
    tc.adam_beta2 = float(opt_cfg.get("adam_beta2", 0.95))
    tc.max_grad_norm = float(opt_cfg.get("max_grad_norm", 1.0))

    # Scheduler
    sched_cfg = raw_config.get("scheduler", {})
    tc.warmup_steps = int(sched_cfg.get("warmup_steps", 200))
    tc.scheduler_type = sched_cfg.get("type", "cosine")

    # Training
    train_cfg = raw_config.get("training", {})
    tc.max_steps = int(train_cfg.get("max_steps", 10000))
    tc.log_every = int(train_cfg.get("log_every", 10))
    tc.output_dir = train_cfg.get("output_dir", "outputs/wm-raw")

    # Multitask
    mt_cfg = raw_config.get("multitask", {})
    tc.vlm_microbatches = int(mt_cfg.get("vlm_microbatches_per_step", 0))
    tc.diffusion_microbatches = int(mt_cfg.get("diffusion_microbatches_per_step", 4))
    tc.ar_loss_weight = float(mt_cfg.get("vlm_loss_weight", 0.0))
    tc.diffusion_loss_weight = float(mt_cfg.get("state_loss_weight", 1.0))
    tc.train_diffusion_backbone = bool(mt_cfg.get("train_diffusion_backbone", True))
    tc.trainable_mode = mt_cfg.get("trainable_mode", "diffusion")

    # Checkpoint
    ckpt_cfg = raw_config.get("checkpoint", {})
    tc.save_every_steps = int(ckpt_cfg.get("save_every_steps", 1000))
    tc.resume_from = args.resume or ckpt_cfg.get("resume_from")

    # Distributed
    dist_cfg = raw_config.get("distributed", {})
    tc.fsdp2_enabled = dist_cfg.get("strategy", "fsdp2") == "fsdp2"

    # Logging
    log_cfg = raw_config.get("logging", {})
    tc.wandb_enabled = bool(log_cfg.get("wandb_enabled", False))
    tc.wandb_project = log_cfg.get("wandb_project", "wm-raw")
    tc.wandb_name = log_cfg.get("wandb_name", "")

    # EMA
    ema_cfg = raw_config.get("ema", {})
    tc.ema_enabled = bool(ema_cfg.get("enabled", False))
    tc.ema_decay = float(ema_cfg.get("decay", 0.9999))
    tc.ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))

    # Compute dtype
    tc.compute_dtype = model_cfg.get("torch_dtype", "bfloat16")

    run_training(tc)


if __name__ == "__main__":
    main()
