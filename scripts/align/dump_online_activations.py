"""Dump intermediate activations from wm-training (online model) for alignment.

Usage:
    # Set up path
    export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH

    # Dump activations from a real GPIC sample
    python scripts/dump_online_activations.py \
        --checkpoint /share/project/eai_pwm/repos/wm-training/outputs/.../checkpoints/step_275000.dcp \
        --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
        --diffusion-path /share/project/eai_pwm/models/Qwen3-VL-2B-Instruct \
        --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
        --prepared-root /share/project/eai_pwm/prepared_datasets/gpic_train_v1_94_incremental \
        --output alignment_fixture.pt

    # Then compare in wm-raw:
    python scripts/align_check.py compare --reference alignment_fixture.pt

This script:
1. Loads a real GPIC sample from the prepared dataset
2. Collates it using the exact same collator as training
3. Encodes with VAE to get state_target
4. Constructs the online model (Qwen3VLBagelModel)
5. Loads the DCP checkpoint
6. Registers forward hooks to capture per-layer activations
7. Runs one forward step (no backward)
8. Saves: fixture batch tensors + all intermediate activations

Prerequisites:
    - wm-training/src in PYTHONPATH
    - Single GPU with enough memory for full model (~24GB bf16)
    - Access to model weights and prepared dataset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def set_deterministic(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_online_model(
    vlm_path: str,
    diffusion_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Build Qwen3VLBagelModel matching the online training config."""
    from transformers import AutoConfig

    from wm_training.models.qwen3vl_bagel.configuration import Qwen3VLBagelConfig
    from wm_training.models.qwen3vl_bagel.modeling import Qwen3VLBagelModel

    print(f"Building model...")
    print(f"  VLM: {vlm_path}")
    print(f"  Diffusion: {diffusion_path}")

    # Load HF configs for layout info
    vlm_hf_config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
    diffusion_hf_config = AutoConfig.from_pretrained(diffusion_path, trust_remote_code=True)

    # Build BAGEL config matching production config
    bagel_config = Qwen3VLBagelConfig.from_mapping({
        "architecture": "qwen3vl_bagel",
        "vlm_backbone_path": vlm_path,
        "diffusion_backbone_path": diffusion_path,
        "torch_dtype": "bfloat16",
        "state_target_dim": 64,  # patchified: 2*2*16 = 64
        "action_target_dim": 1,
        "state_target_format": "image_dit",
        "action_target_format": "continuous_tensor",
        "ar_loss_weight": 0.0,
        "state_diffusion_loss_weight": 1.0,
        "action_diffusion_loss_weight": 0.0,
        "latent": {
            "objective": "flow_matching",
            "prediction_type": "flow",
            "timestep_shift": 1.0,
            "timestep_sampling": {
                "type": "logit_normal",
                "mean": 0.0,
                "std": 1.0,
            },
            "timestep_frequency_dim": 256,
            "timestep_conditioning": "adaln_zero",
            "tokenization": "patchified",
            "patch_size": 2,
            "position_embedding": "bagel_2d_sincos",
            "max_position_size": 64,
            "edm": {},
        },
        "trajectory_context": {
            "enabled": False,
            "state_dim": 0,
            "tactile_force_dim": 0,
            "action_dim": 0,
        },
        "cross_attention": {
            "enabled": True,
            "communication_policy": "cross_kv_concat",
            "layer_mapping_policy": "middle_n",
            "hidden_state_layer_offset": 1,
            "gate_init": 0.01,
            "zero_init_output": False,
        },
    }).with_hf_layouts(
        vlm_config=vlm_hf_config,
        diffusion_config=diffusion_hf_config,
    )
    bagel_config.validate()

    # Construct model + load pretrained backbones
    model = Qwen3VLBagelModel(bagel_config)
    print("  Loading VLM backbone...")
    model.vlm_branch.load_backbone()
    print("  Loading Diffusion backbone...")
    model.state_diffusion_branch.load_backbone()

    model = model.to(device=device, dtype=dtype)
    print(f"  Model on {device}, dtype={dtype}")
    return model


def load_dcp_checkpoint(model: Any, checkpoint_path: str) -> None:
    """Load a DCP checkpoint into the model (single GPU, no FSDP)."""
    from wm_training.inference.checkpoint_loader import load_model_weights

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"Loading checkpoint: {path}")
    report = load_model_weights(
        model,
        str(path),
        strict=False,
        map_location="cpu",
        allow_partial_dcp_load=False,
        use_ema=False,
    )
    n_loaded = len(report.loaded_keys) if hasattr(report, "loaded_keys") else "unknown"
    n_missing = len(report.missing_keys) if hasattr(report, "missing_keys") else "unknown"
    print(f"  Loaded: {n_loaded} keys, Missing: {n_missing} keys")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gpic_sample(
    prepared_root: str,
    vlm_path: str,
    target_height: int = 512,
    target_width: int = 512,
) -> dict[str, Any]:
    """Load one real GPIC sample and collate it for diffusion training.

    Returns a dict with:
        condition: {input_ids: [1, L], attention_mask: [1, L]}
        vae_pixel_values: [1, 3, H, W] float32 in [-1, 1]
        caption: str
    """
    from transformers import AutoProcessor

    from wm_training.data.prepared_views import PreparedImageCaptionDataset
    from wm_training.data.image_caption_collator import Qwen3VLImageCaptionDiffusionCollator

    print(f"Loading GPIC sample from: {prepared_root}")

    # Load dataset (skip shard validation — hashing 55k shards is very slow)
    dataset = PreparedImageCaptionDataset(
        prepared_root=prepared_root,
        image_size=target_height,
        center_crop=False,
        max_read_retries=3,
        validate=False,
        max_samples=1,
    )
    print(f"  Dataset size: {len(dataset)}")

    # Load a sample (index 0 for reproducibility)
    # With bucket metadata: pass (index, bucket_id) tuple
    # bucket_id=0 corresponds to the first bucket size (512x512)
    example = dataset[0]
    print(f"  Caption: {example.caption[:80]}...")
    print(f"  Image size: {example.image.size}")

    # Set up collator
    processor = AutoProcessor.from_pretrained(
        vlm_path, trust_remote_code=True,
        min_pixels=1024, max_pixels=1048576,
    )

    collator = Qwen3VLImageCaptionDiffusionCollator(
        processor=processor,
        image_size=target_height,
        condition_prefix="Caption: ",
        condition_suffix=" <|wm_predict_image|>",
        text_condition_dropout_prob=0.0,  # No dropout for alignment test
        cfg_dropout_mode="sentinel_only",
        unconditional_condition_text=None,
        condition_max_seq_len=None,
    )

    # Override example metadata with target resolution
    from dataclasses import replace as _dc_replace

    meta = dict(example.metadata) if example.metadata else {}
    meta["target_height"] = target_height
    meta["target_width"] = target_width
    example = _dc_replace(example, metadata=meta)

    # Collate single sample
    batch = collator([example])
    print(f"  Condition input_ids shape: {batch['condition']['input_ids'].shape}")
    print(f"  VAE pixel_values shape: {batch['vae_pixel_values'].shape}")

    return batch


def encode_with_vae(
    vae_path: str,
    vae_pixel_values: Tensor,
    device: torch.device,
    patch_size: int = 2,
) -> tuple[Tensor, int, int]:
    """Encode pixels with BAGEL VAE and patchify.

    Returns:
        state_target: [B, num_patches, patch_dim] — patchified latents
        latent_h: int
        latent_w: int
    """
    from wm_training.models.qwen3vl_bagel.latent_codec import FrozenImageLatentCodec

    print(f"  Loading VAE: {vae_path}")
    codec = FrozenImageLatentCodec.from_pretrained(
        vae_path,
        device=device,
        dtype=torch.bfloat16,
    )

    print(f"  Encoding image ({vae_pixel_values.shape})...")
    with torch.no_grad():
        latent_batch = codec.encode(vae_pixel_values.to(device))

    # latent_batch.tokens: [B, H*W, C] e.g. [1, 4096, 16] for 512x512
    tokens = latent_batch.tokens
    latent_shape = latent_batch.latent_shape  # (C, H, W) e.g. (16, 64, 64)
    latent_h = latent_shape[1]
    latent_w = latent_shape[2]

    print(f"  Latent shape: {latent_shape} → tokens {tokens.shape}")

    # Patchify: [B, H*W, C] → [B, (H/P)*(W/P), P*P*C]
    from wm_training.models.qwen3vl_bagel.image_latent import patchify_image_latent_tokens

    state_target, patched_shape = patchify_image_latent_tokens(
        torch, tokens, latent_shape=latent_shape, patch_size=patch_size,
    )
    print(f"  Patchified state_target: {state_target.shape}")
    return state_target, latent_h, latent_w


# ---------------------------------------------------------------------------
# Activation hooking
# ---------------------------------------------------------------------------


class OnlineActivationCapture:
    """Register hooks on wm-training model to capture intermediate activations.

    Captures:
        vlm.layer.{i}           — VLM decoder layer outputs [B, S, D_vlm]
        cross_attn.kv.{i}       — Cross-attention projected K,V [B, H_kv, S_vlm, D]
        diffusion.input_proj    — After input projection [B, S_diff, D_diff]
        diffusion.adaln.{i}     — AdaLN params (6 tensors) per layer
        diffusion.layer.{i}     — Diffusion decoder layer outputs [B, S_diff, D_diff]
        diffusion.final_norm    — After final RMSNorm
        diffusion.prediction    — Output head result [B, S_diff, token_dim]
    """

    def __init__(self) -> None:
        self.activations: dict[str, Tensor] = {}
        self._hooks: list[Any] = []

    def register(self, model) -> None:
        """Register hooks on wm-training model."""
        # --- VLM decoder layers ---
        try:
            vlm_layers = model.vlm_branch.backbone.model.language_model.model.layers
        except AttributeError:
            try:
                # Try other paths the model uses internally
                backbone = model.vlm_branch.backbone
                for path_fn in [
                    lambda b: b.model.language_model.model.layers,
                    lambda b: b.model.language_model.layers,
                    lambda b: b.language_model.model.layers,
                    lambda b: b.language_model.layers,
                    lambda b: b.model.model.layers,
                    lambda b: b.model.layers,
                    lambda b: b.layers,
                ]:
                    try:
                        vlm_layers = path_fn(backbone)
                        break
                    except AttributeError:
                        continue
                else:
                    vlm_layers = []
                    print("  WARNING: Could not find VLM layers for hooking")
            except Exception:
                vlm_layers = []
                print("  WARNING: Could not find VLM layers for hooking")

        for i, layer in enumerate(vlm_layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_output_hook(f"vlm.layer.{i}"))
            )

        # --- Diffusion branch ---
        branch = model.state_diffusion_branch

        # Input projector
        self._hooks.append(
            branch.input_projector.register_forward_hook(
                self._make_output_hook("diffusion.input_proj")
            )
        )

        # Time embedder
        self._hooks.append(
            branch.time_embedder.register_forward_hook(
                self._make_output_hook("diffusion.time_embedder")
            )
        )

        # Time conditioner
        self._hooks.append(
            branch.time_conditioner.register_forward_hook(
                self._make_output_hook("diffusion.time_conditioner")
            )
        )

        # AdaLN modulations per layer
        if branch.adaln_modulations is not None:
            for i, adaln in enumerate(branch.adaln_modulations):
                self._hooks.append(
                    adaln.register_forward_hook(
                        self._make_adaln_hook(f"diffusion.adaln.{i}")
                    )
                )

        # Diffusion decoder layers
        diff_layers = branch.adapter.layers
        for i, layer in enumerate(diff_layers):
            self._hooks.append(
                layer.register_forward_hook(
                    self._make_output_hook(f"diffusion.layer.{i}")
                )
            )

        # Final norm
        self._hooks.append(
            branch.adapter.final_norm.register_forward_hook(
                self._make_output_hook("diffusion.final_norm")
            )
        )

        # Output head
        self._hooks.append(
            branch.output_head.register_forward_hook(
                self._make_output_hook("diffusion.output_head")
            )
        )

        # --- Cross-attention adapters ---
        xattn_adapters = model.cross_attention.adapters.get("state", None)
        if xattn_adapters is not None:
            for i, adapter in enumerate(xattn_adapters):
                self._hooks.append(
                    adapter.register_forward_hook(
                        self._make_xattn_hook(f"cross_attn.adapter.{i}")
                    )
                )

        print(f"  Registered {len(self._hooks)} hooks")

    def _make_output_hook(self, name: str):
        """Capture the first tensor output."""
        def hook(module, input, output):
            if isinstance(output, Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                for item in output:
                    if isinstance(item, Tensor):
                        self.activations[name] = item.detach().cpu()
                        break
        return hook

    def _make_adaln_hook(self, name: str):
        """Capture AdaLN output (tuple of 6 tensors: shift/scale/gate × attn/mlp)."""
        def hook(module, input, output):
            if isinstance(output, tuple) and all(isinstance(t, Tensor) for t in output):
                # Stack 6 modulation vectors: [6, B, D]
                self.activations[name] = torch.stack(
                    [t.detach().cpu() for t in output]
                )
            elif isinstance(output, Tensor):
                # Some implementations return a single [B, 6*D] tensor
                self.activations[name] = output.detach().cpu()
        return hook

    def _make_xattn_hook(self, name: str):
        """Capture cross-attention adapter forward output (K, V projection)."""
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                k, v = output
                if isinstance(k, Tensor) and isinstance(v, Tensor):
                    self.activations[f"{name}.k"] = k.detach().cpu()
                    self.activations[f"{name}.v"] = v.detach().cpu()
            elif isinstance(output, Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump wm-training activations for alignment testing"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="DCP checkpoint directory (e.g. step_275000.dcp)")
    parser.add_argument("--vlm-path", type=str,
                        default="/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--diffusion-path", type=str,
                        default="/share/project/eai_pwm/models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vae-path", type=str,
                        default="/share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--prepared-root", type=str,
                        default="/share/project/eai_pwm/prepared_datasets/gpic_train_v1_94_incremental")
    parser.add_argument("--output", type=str, default="alignment_fixture.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--sample-index", type=int, default=0,
                        help="Index in prepared dataset to use as fixture")

    args = parser.parse_args()

    print("=" * 70)
    print(" wm-training Activation Dumper for Alignment")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    set_deterministic(args.seed)

    # 1. Load a real GPIC sample and collate
    print("[Step 1] Loading GPIC data...")
    batch = load_gpic_sample(
        prepared_root=args.prepared_root,
        vlm_path=args.vlm_path,
        target_height=args.image_height,
        target_width=args.image_width,
    )
    print()

    # 2. VAE encode
    print("[Step 2] VAE encoding...")
    state_target, latent_h, latent_w = encode_with_vae(
        vae_path=args.vae_path,
        vae_pixel_values=batch["vae_pixel_values"],
        device=device,
        patch_size=2,
    )
    print()

    # 3. Build model
    print("[Step 3] Building model...")
    model = build_online_model(
        vlm_path=args.vlm_path,
        diffusion_path=args.diffusion_path,
        device=device,
        dtype=dtype,
    )
    print()

    # 4. Load checkpoint
    print("[Step 4] Loading checkpoint...")
    load_dcp_checkpoint(model, args.checkpoint)
    model.eval()
    print()

    # 5. Register hooks
    print("[Step 5] Registering activation hooks...")
    capture = OnlineActivationCapture()
    capture.register(model)
    print()

    # 6. Prepare forward inputs
    print("[Step 6] Running forward pass...")

    # Fix timestep and noise for reproducibility
    set_deterministic(args.seed)
    patch_h = latent_h // 2
    patch_w = latent_w // 2
    num_tokens = patch_h * patch_w
    token_dim = 64  # patch_size^2 * latent_channels = 2*2*16

    timesteps = torch.tensor([0.5], device=device, dtype=torch.float32)
    noise = torch.randn(1, num_tokens, token_dim, device=device, dtype=dtype)

    # Build latent_position_ids for bagel_2d_sincos
    # pos_id = row * max_position_size + col
    max_pos = 64
    pos_ids_list = []
    for r in range(patch_h):
        for c in range(patch_w):
            pos_ids_list.append(r * max_pos + c)
    latent_position_ids = torch.tensor(
        pos_ids_list, device=device, dtype=torch.long
    ).unsqueeze(0)  # [1, num_tokens]

    # Build the diffusion batch using model's high-level API
    # This exactly mirrors the training loop in qwen3vl_image_joint.py
    condition = {k: v.to(device) for k, v in batch["condition"].items()}
    # Ensure output_hidden_states for cross-attention
    condition["output_hidden_states"] = True

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # Use the model's own forward_diffusion_training for exact parity
        output = model.forward_diffusion_training({
            "condition": condition,
            "state_target": state_target,
            "state_timesteps": timesteps,
            "state_noise": noise,
            "state_latent_position_ids": latent_position_ids,
        })

    loss = output.loss
    capture.activations["_loss"] = loss.detach().cpu()

    # Also save prediction from output_head hook (already captured)
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Captured {len(capture.activations)} activation tensors")
    print()

    # Reconstruct noisy_tokens and velocity_target for fixture
    # (needed so wm-raw can replay the exact same computation)
    clean_tokens = state_target  # already patchified: [B, num_tokens, token_dim]
    t_expand = timesteps[:, None, None]
    noisy_tokens = (1.0 - t_expand) * clean_tokens + t_expand * noise
    velocity_target = noise - clean_tokens

    # 7. Save everything
    print("[Step 7] Saving fixture...")
    output_data = {
        # Activations (all hooks captured during forward)
        **{k: v for k, v in capture.activations.items()},

        # Fixture batch (for replay in wm-raw)
        "_fixture": {
            "condition_input_ids": condition["input_ids"].cpu(),
            "condition_attention_mask": condition["attention_mask"].cpu(),
            "state_target": state_target.cpu(),  # patchified [B, num_tokens, token_dim]
            "latent_h": latent_h,
            "latent_w": latent_w,
            "timesteps": timesteps.cpu(),
            "noise": noise.cpu(),
            "noisy_tokens": noisy_tokens.detach().cpu(),
            "velocity_target": velocity_target.detach().cpu(),
            "clean_tokens": clean_tokens.cpu(),
            "latent_position_ids": latent_position_ids.cpu(),
        },

        # Metadata
        "_meta": {
            "seed": args.seed,
            "image_height": args.image_height,
            "image_width": args.image_width,
            "sample_index": args.sample_index,
            "checkpoint": args.checkpoint,
            "patch_h": patch_h,
            "patch_w": patch_w,
            "num_tokens": num_tokens,
            "token_dim": token_dim,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, output_path)
    print(f"  Saved to: {output_path}")
    print(f"  Keys: {sorted(k for k in output_data if not k.startswith('_'))}")
    print()
    print("Done! Next step: python scripts/align_check.py compare --reference", args.output)

    capture.remove()


if __name__ == "__main__":
    main()
