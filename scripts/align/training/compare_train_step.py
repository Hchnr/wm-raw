#!/usr/bin/env python3
"""Single-step training comparison: wm-raw vs wm-training.

Runs ONE forward+backward step with identical inputs on both models,
comparing loss, prediction, and gradients to verify training alignment.

Usage:
    export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH

    CUDA_VISIBLE_DEVICES=0 python scripts/align/training/compare_train_step.py \
        --checkpoint /path/to/step_295000_training.pt \
        --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
        --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def compare(name: str, a: Tensor, b: Tensor, atol: float = 1e-3) -> bool:
    a_f, b_f = a.float().cpu(), b.float().cpu()
    if a_f.shape != b_f.shape:
        print(f"  ✗ {name}: SHAPE MISMATCH {list(a_f.shape)} vs {list(b_f.shape)}")
        return False
    diff = (a_f - b_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < atol
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: max={max_err:.2e}, mean={mean_err:.2e}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print("=" * 70)
    print("Training Step Comparison: wm-raw vs wm-training")
    print("=" * 70)

    # =========================================================================
    # Create shared fixture: condition tokens + patchified state_target + noise
    # =========================================================================
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Fixed condition text
    condition_text = "Caption: A peaceful lake surrounded by green mountains. <|wm_predict_image|>"
    encoded = tokenizer([condition_text], padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask_1d = encoded["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    # Fixed state_target: simulate a 512x512 image → VAE latent [1, 4096, 16] → patchify [1, 1024, 64]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    latent_h, latent_w, latent_c = 64, 64, 16
    patch_size = 2
    # Raw latent (before patchify)
    state_target_raw = torch.randn(1, latent_h * latent_w, latent_c, device=device, dtype=dtype)

    # Patchify (for wm-training which expects patchified input)
    from wm_raw.models.embeddings import patchify_latent
    state_target_patched = patchify_latent(
        state_target_raw, height=latent_h, width=latent_w, patch_size=patch_size
    )  # [1, 1024, 64]

    # Fixed noise and timesteps (so both models see identical randomness)
    torch.manual_seed(args.seed + 1000)
    noise = torch.randn_like(state_target_patched)
    from wm_raw.diffusion import sample_timesteps
    timesteps = sample_timesteps(
        1, device=device, sampling_type="logit_normal", mean=0.0, std=1.0, shift=1.0
    )

    print(f"Fixture: seq_len={seq_len}, state_target_raw={state_target_raw.shape}, "
          f"state_target_patched={state_target_patched.shape}")
    print(f"  timesteps={timesteps.item():.6f}, noise_std={noise.std():.4f}")
    print()

    # =========================================================================
    # Part 1: wm-training forward
    # =========================================================================
    print("=" * 70)
    print("[wm-training] Forward...")
    print("=" * 70)

    from wm_training.models.qwen3vl_bagel.modeling import Qwen3VLBagelModel, Qwen3VLBagelConfig
    from wm_training.inference.checkpoint_loader import load_model_weights
    from wm_training.models.qwen3vl_bagel.image_latent import (
        make_image_grid_position_ids_from_latent_shape,
        make_bagel_latent_position_ids_from_latent_shape,
    )
    from transformers import AutoConfig
    import yaml

    config_path = "/share/project/eai_pwm/home/hcr/repos/wm-training/configs_cuihu/qwen3vl_gpic_patchlatent_2dpos_adaln_fm_logitnormal_buckets_512_stage2_from_step145000_fsdp.yaml"
    with open(config_path) as f:
        train_config = yaml.safe_load(f)
    mc = train_config["model"]
    vlm_hf = AutoConfig.from_pretrained(args.vlm_path, trust_remote_code=True)
    diff_hf = AutoConfig.from_pretrained(mc["diffusion_path"], trust_remote_code=True)

    bc = Qwen3VLBagelConfig.from_mapping({
        "architecture": "qwen3vl_bagel",
        "vlm_backbone_path": args.vlm_path,
        "diffusion_backbone_path": mc["diffusion_path"],
        "torch_dtype": "bfloat16",
        "state_target_dim": 64,
        "action_target_dim": 1,
        "state_target_format": "image_dit",
        "action_target_format": "continuous_tensor",
        "ar_loss_weight": 0.0,
        "state_diffusion_loss_weight": 1.0,
        "action_diffusion_loss_weight": 0.0,
        "latent": mc["latent"],
        "trajectory_context": {"enabled": False, "state_dim": 0, "tactile_force_dim": 0, "action_dim": 0},
        "cross_attention": {
            "enabled": True,
            "communication_policy": mc.get("communication_policy", "cross_kv_concat"),
            "layer_mapping_policy": mc.get("layer_mapping_policy", "middle_n"),
            "hidden_state_layer_offset": int(mc.get("hidden_state_layer_offset", 1)),
            "gate_init": float(mc.get("cross_attention_gate_init", 0.01)),
            "zero_init_output": False,
        },
    }).with_hf_layouts(vlm_config=vlm_hf, diffusion_config=diff_hf)
    bc.validate()

    om = Qwen3VLBagelModel(bc)
    om.vlm_branch.load_backbone()
    om.state_diffusion_branch.load_backbone()
    load_model_weights(om, args.checkpoint, strict=False)
    om = om.to(device=device, dtype=dtype)
    om.train()

    # Build position IDs (same as online train step)
    patch_h = latent_h // patch_size
    patch_w = latent_w // patch_size
    state_latent_shape = (latent_c * patch_size * patch_size, patch_h, patch_w)  # (64, 32, 32)

    state_position_ids = make_image_grid_position_ids_from_latent_shape(
        torch, latent_shape=state_latent_shape, batch_size=1, device=device
    )
    state_latent_position_ids = make_bagel_latent_position_ids_from_latent_shape(
        torch, latent_shape=state_latent_shape, batch_size=1,
        max_position_size=64, device=device
    )

    # Online forward
    online_batch = {
        "task_type": "diffusion",
        "condition": {"input_ids": input_ids, "attention_mask": attention_mask_1d},
        "state_target": state_target_patched,
        "state_loss_mask": None,
        "state_position_ids": state_position_ids,
        "state_latent_position_ids": state_latent_position_ids,
        "state_loss_weight": 1.0,
        "action_loss_weight": 0.0,
    }

    with torch.amp.autocast("cuda", dtype=dtype):
        online_output = om(online_batch)

    online_loss = online_output.loss
    print(f"  Online loss: {online_loss.item():.6f}")

    # Backward
    online_loss.backward()

    # Collect gradients
    online_grads = {}
    for name, p in om.named_parameters():
        if p.grad is not None and "state_diffusion_branch" in name:
            online_grads[name] = p.grad.detach().clone()
    print(f"  Online grads collected: {len(online_grads)} params")

    # =========================================================================
    # Part 2: wm-raw forward
    # =========================================================================
    print()
    print("=" * 70)
    print("[wm-raw] Forward...")
    print("=" * 70)

    from wm_raw.config import WorldModelConfig
    from wm_raw.models.model import WorldModel
    from wm_raw.checkpoint import _map_online_key, load_vlm_weights

    cfg = WorldModelConfig()
    rm = WorldModel(cfg)
    cd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_vlm_weights(rm, args.vlm_path)
    sd = rm.state_dict()
    remap = {}
    for k, v in cd["model"].items():
        m = _map_online_key(f"model.{k}")
        if m is None:
            continue
        s = m[len("model."):] if m.startswith("model.") else m
        if s in sd and v.shape == sd[s].shape:
            remap[s] = v
    rm.load_state_dict(remap, strict=False)
    rm = rm.to(device=device, dtype=dtype)
    rm.train()

    # wm-raw forward: state_target is UN-patchified [B, H*W, C]
    # model.forward_diffusion patchifies internally
    # Also needs latent_h, latent_w
    # For VLM: pass 1D attention_mask (wm-raw VLM handles it internally... or not?)
    # Actually wm-raw VLM expects 4D mask. Let's use the HF VLM hidden states instead.

    # Use HF VLM for condition (same approach as generate_image.py for exact alignment)
    from transformers import AutoModelForImageTextToText
    hf_vlm = AutoModelForImageTextToText.from_pretrained(
        args.vlm_path, trust_remote_code=True, dtype=dtype
    ).to(device).eval()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        hf_out = hf_vlm(
            input_ids=input_ids, attention_mask=attention_mask_1d,
            output_hidden_states=True, return_dict=True,
        )
        vlm_hidden_states = list(hf_out.hidden_states)

    # wm-raw forward_diffusion directly (bypass VLM)
    with torch.amp.autocast("cuda", dtype=dtype):
        raw_diff_out = rm.forward_diffusion(
            state_target=state_target_raw,
            vlm_hidden_states=vlm_hidden_states,
            latent_h=latent_h,
            latent_w=latent_w,
            timesteps=timesteps,
            noise=noise,
        )

    raw_loss = raw_diff_out.loss
    print(f"  wm-raw loss: {raw_loss.item():.6f}")

    # Backward
    raw_loss.backward()

    # Collect gradients
    raw_grads = {}
    for name, p in rm.named_parameters():
        if p.grad is not None and "state_diffusion" in name:
            raw_grads[name] = p.grad.detach().clone()
    print(f"  wm-raw grads collected: {len(raw_grads)} params")

    # =========================================================================
    # Part 3: Compare
    # =========================================================================
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n[Loss]")
    loss_diff = abs(online_loss.item() - raw_loss.item())
    status = "✓" if loss_diff < 1e-4 else "✗"
    print(f"  {status} online={online_loss.item():.6f}, raw={raw_loss.item():.6f}, diff={loss_diff:.2e}")

    print(f"\n[Gradients] (diffusion branch)")
    # Map online grad keys to raw keys for comparison
    from wm_raw.checkpoint import _map_online_key
    grad_compared = 0
    grad_passed = 0
    grad_failed_examples = []

    for online_name, online_grad in sorted(online_grads.items())[:50]:
        mapped = _map_online_key(f"model.{online_name}")
        if mapped is None:
            continue
        raw_name = mapped[len("model."):] if mapped.startswith("model.") else mapped
        if raw_name not in raw_grads:
            continue

        raw_grad = raw_grads[raw_name]
        diff = (online_grad.float() - raw_grad.float()).abs()
        max_d = diff.max().item()
        grad_compared += 1
        if max_d < 0.01:
            grad_passed += 1
        else:
            grad_failed_examples.append((online_name, max_d))

    print(f"  Compared: {grad_compared}, Passed (<0.01): {grad_passed}, Failed: {grad_compared - grad_passed}")
    if grad_failed_examples:
        print(f"  Worst failures:")
        for name, d in sorted(grad_failed_examples, key=lambda x: -x[1])[:5]:
            print(f"    {name}: max_diff={d:.4f}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
