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
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip pretrained weight loading (test pipeline with random weights)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--adam-fused", type=str, default=None, choices=["true", "false"],
                        help="Enable/disable fused AdamW (overrides config)")
    # Output / logging overrides
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override training output directory")
    parser.add_argument("--compile", type=str, default=None, choices=["true", "false"],
                        help="Enable/disable torch.compile (overrides config)")
    parser.add_argument("--compile-mode", type=str, default=None,
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override data.batch_size from config")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override data.num_workers from config")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Override wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Override wandb entity (team/user)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Override wandb run name")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Override log file directory (default: <output_dir>/logs)")
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
    tc.diffusion_lr = float(opt_cfg.get("diffusion_learning_rate", 1e-4))
    tc.vlm_lr = float(opt_cfg.get("vlm_learning_rate", 0.0))
    tc.weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    tc.adam_beta1 = float(opt_cfg.get("adam_beta1", 0.9))
    tc.adam_beta2 = float(opt_cfg.get("adam_beta2", 0.95))
    tc.adam_fused = bool(opt_cfg.get("fused", True))
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

    # Fixed sequence length (for CUDA Graph / static compilation)
    fixed_seqlen_cfg = train_cfg.get("fixed_seqlen", {})
    if fixed_seqlen_cfg.get("enabled"):
        tc.condition_max_seq_len = int(fixed_seqlen_cfg.get("condition_max_seq_len", 256))

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
    tc.keep_last_n = ckpt_cfg.get("keep_last_n", None)
    if tc.keep_last_n is not None:
        tc.keep_last_n = int(tc.keep_last_n)
    tc.save_final = bool(ckpt_cfg.get("save_final", True))
    tc.resume_from = args.resume or ckpt_cfg.get("resume_from")
    tc.resume_mode = ckpt_cfg.get("resume_mode", "auto")

    # Distributed
    dist_cfg = raw_config.get("distributed", {})
    tc.fsdp2_enabled = dist_cfg.get("strategy", "fsdp2") == "fsdp2"
    tc.hsdp_enabled = bool(dist_cfg.get("hsdp", False))

    # torch.compile — check both locations (new: training.torch_compile, old: distributed.compile)
    compile_cfg = train_cfg.get("torch_compile", {})
    if compile_cfg:
        tc.compile_enabled = bool(compile_cfg.get("enabled", False))
        tc.compile_mode = compile_cfg.get("mode", "default")
    else:
        tc.compile_enabled = bool(dist_cfg.get("compile", False))
        tc.compile_mode = dist_cfg.get("compile_mode", "default")

    # Logging
    log_cfg = raw_config.get("logging", {})
    tc.wandb_enabled = bool(log_cfg.get("wandb_enabled", False))
    tc.wandb_project = log_cfg.get("wandb_project", "wm-raw")
    tc.wandb_entity = log_cfg.get("wandb_entity", "")
    tc.wandb_name = log_cfg.get("wandb_name", "")

    # EMA
    ema_cfg = raw_config.get("ema", {})
    tc.ema_enabled = bool(ema_cfg.get("enabled", False))
    tc.ema_decay = float(ema_cfg.get("decay", 0.9999))
    tc.ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))
    tc.ema_update_every = int(ema_cfg.get("update_every", 1))

    # Compute dtype
    tc.compute_dtype = model_cfg.get("torch_dtype", "bfloat16")

    # CLI overrides
    if args.dry_run:
        tc.skip_pretrained = True
    if args.max_steps is not None:
        tc.max_steps = args.max_steps
    if args.adam_fused is not None:
        tc.adam_fused = args.adam_fused.lower() == "true"
    if args.output_dir is not None:
        tc.output_dir = args.output_dir
    if args.compile is not None:
        tc.compile_enabled = args.compile.lower() == "true"
    if args.compile_mode is not None:
        tc.compile_mode = args.compile_mode
    if args.wandb_project is not None:
        tc.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        tc.wandb_entity = args.wandb_entity
    if args.wandb_name is not None:
        tc.wandb_name = args.wandb_name
    if args.log_dir is not None:
        tc.log_dir = args.log_dir
    if args.batch_size is not None:
        tc.batch_size = args.batch_size
    if args.num_workers is not None:
        tc.num_workers = args.num_workers

    run_training(tc)


if __name__ == "__main__":
    main()
