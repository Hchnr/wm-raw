#!/usr/bin/env python3
"""Side-by-side numerical comparison of wm-raw vs wm-training inference.

Runs the SAME prompt with SAME seed through both models and compares
intermediate hidden states at each stage to find where they diverge.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_inference.py \
        --checkpoint /share/project/eai_pwm/repos/wm-training/outputs/.../step_295000.pt \
        --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
        --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
        --prompt "A peaceful lake surrounded by green mountains" \
        --seed 1234

Requires: wm-training/src in PYTHONPATH
    export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def compare(name: str, a: Tensor, b: Tensor, atol: float = 1e-3) -> bool:
    """Compare two tensors, print stats, return True if close."""
    a_f = a.float().cpu()
    b_f = b.float().cpu()
    if a_f.shape != b_f.shape:
        print(f"  ✗ {name}: SHAPE MISMATCH {list(a_f.shape)} vs {list(b_f.shape)}")
        return False
    diff = (a_f - b_f).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    ok = max_err < atol
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: max={max_err:.2e}, mean={mean_err:.2e}, shape={list(a_f.shape)}")
    if not ok:
        # Show where the max error is
        flat_idx = diff.argmax().item()
        print(f"      a_val={a_f.flatten()[flat_idx]:.6f}, b_val={b_f.flatten()[flat_idx]:.6f}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare wm-raw vs wm-training inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--prompt", default="A peaceful lake surrounded by green mountains, clear blue sky, soft sunlight, realistic photography style.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition-prefix", default="Caption: ")
    parser.add_argument("--condition-suffix", default=" <|wm_predict_image|>")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    condition_text = f"{args.condition_prefix}{args.prompt}{args.condition_suffix}"
    print("=" * 70)
    print("Inference Comparison: wm-raw vs wm-training")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {condition_text!r}")
    print(f"Seed: {args.seed}")
    print()

    # =========================================================================
    # Part 1: Load tokenizer (shared)
    # =========================================================================
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Tokenize condition
    encoded = tokenizer([condition_text], padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask_1d = encoded["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    print(f"Tokenized: seq_len={seq_len}, input_ids[:10]={input_ids[0, :10].tolist()}")
    print()

    # =========================================================================
    # Part 2: wm-training forward
    # =========================================================================
    print("=" * 70)
    print("[wm-training] Building model and running forward...")
    print("=" * 70)

    from wm_raw.utils.timing import Timer

    from wm_training.models.qwen3vl_bagel.modeling import Qwen3VLBagelModel
    from wm_training.models.qwen3vl_bagel.configuration import Qwen3VLBagelConfig
    from wm_training.inference.checkpoint_loader import load_model_weights
    from transformers import AutoConfig
    import yaml

    timer_online = Timer()

    config_path = "/share/project/eai_pwm/home/hcr/repos/wm-training/configs_cuihu/qwen3vl_gpic_patchlatent_2dpos_adaln_fm_logitnormal_buckets_512_stage2_from_step145000_fsdp.yaml"
    with open(config_path) as f:
        train_config = yaml.safe_load(f)

    model_cfg = train_config["model"]

    with timer_online.stage("load HF configs"):
        vlm_hf_config = AutoConfig.from_pretrained(args.vlm_path, trust_remote_code=True)
        diffusion_hf_config = AutoConfig.from_pretrained(
            model_cfg["diffusion_path"], trust_remote_code=True
        )

    bagel_config = Qwen3VLBagelConfig.from_mapping({
        "architecture": "qwen3vl_bagel",
        "vlm_backbone_path": args.vlm_path,
        "diffusion_backbone_path": model_cfg["diffusion_path"],
        "torch_dtype": "bfloat16",
        "state_target_dim": int(model_cfg.get("state_target_dim", 64)),
        "action_target_dim": 1,
        "state_target_format": "image_dit",
        "action_target_format": "continuous_tensor",
        "ar_loss_weight": 0.0,
        "state_diffusion_loss_weight": 1.0,
        "action_diffusion_loss_weight": 0.0,
        "latent": model_cfg["latent"],
        "trajectory_context": {"enabled": False, "state_dim": 0, "tactile_force_dim": 0, "action_dim": 0},
        "cross_attention": {
            "enabled": True,
            "communication_policy": model_cfg.get("communication_policy", "cross_kv_concat"),
            "layer_mapping_policy": model_cfg.get("layer_mapping_policy", "middle_n"),
            "hidden_state_layer_offset": int(model_cfg.get("hidden_state_layer_offset", 1)),
            "gate_init": float(model_cfg.get("cross_attention_gate_init", 0.01)),
            "zero_init_output": False,
        },
    }).with_hf_layouts(vlm_config=vlm_hf_config, diffusion_config=diffusion_hf_config)
    bagel_config.validate()

    with timer_online.stage("construct Qwen3VLBagelModel"):
        online_model = Qwen3VLBagelModel(bagel_config)

    with timer_online.stage("load VLM backbone"):
        online_model.vlm_branch.load_backbone()

    with timer_online.stage("load diffusion backbone"):
        online_model.state_diffusion_branch.load_backbone()

    with timer_online.stage("load .pt checkpoint"):
        load_model_weights(online_model, args.checkpoint, strict=False)

    with timer_online.stage("model to device"):
        online_model = online_model.to(device=device, dtype=dtype)
        online_model.eval()

    timer_online.summary()

    # VLM forward
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        vlm_out_online = online_model.forward_vlm({
            "input_ids": input_ids,
            "attention_mask": attention_mask_1d,
            "output_hidden_states": True,
        })
    online_hidden_states = vlm_out_online.hidden_states
    print(f"  VLM hidden_states: {len(online_hidden_states)} layers, shape={online_hidden_states[0].shape}")

    # Diffusion forward (one step at t=1.0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Generate noise matching 512x512 image
    patch_h, patch_w = 32, 32  # 512/8/2 = 32
    num_tokens = patch_h * patch_w
    token_dim = 64
    noise = torch.randn(1, num_tokens, token_dim, device=device, dtype=dtype)
    timesteps = torch.tensor([1.0], device=device, dtype=torch.float32)

    # Build latent position ids
    from wm_training.models.qwen3vl_bagel.image_latent import make_bagel_latent_position_ids
    latent_pos_ids = make_bagel_latent_position_ids(
        torch, batch_size=1, height=patch_h, width=patch_w, max_position_size=64, device=device
    )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        prepared = online_model.state_diffusion_branch.prepare_inputs(
            noisy_sample=noise,
            timesteps=timesteps,
            latent_position_ids=latent_pos_ids,
        )

    online_prepared_hidden = prepared.hidden.detach().clone()
    online_time_hidden = prepared.time_hidden.detach().clone()
    print(f"  Prepared hidden: {online_prepared_hidden.shape}, time_hidden: {online_time_hidden.shape}")

    # Run diffusion layers
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        adapter = online_model.state_diffusion_branch.require_adapter()
        hidden = prepared.hidden
        pos_ids, pos_emb = adapter.make_position_embeddings(hidden, position_ids=None)

        online_layer_outputs = []
        for layer_idx in range(adapter.num_layers):
            external_kv = online_model.cross_attention.project_context_key_value(
                "state", layer_idx, hidden, vlm_hidden_states=online_hidden_states,
            )
            adaln = online_model.state_diffusion_branch.adaln_params(layer_idx, prepared.time_hidden)
            hidden = adapter.forward_layer(
                layer_idx, hidden, attention_mask=None,
                position_ids=pos_ids, position_embeddings=pos_emb,
                external_key_value=external_kv, external_attention_mask=None,
                adaln=adaln,
            )
            if layer_idx < 3 or layer_idx == adapter.num_layers - 1:
                online_layer_outputs.append((layer_idx, hidden.detach().clone()))

        online_final = adapter.finalize(hidden).detach().clone()
        online_prediction = online_model.state_diffusion_branch.output_head(
            online_final.to(online_model.state_diffusion_branch.output_head.weight.dtype)
        ).detach().clone()

    print(f"  Final hidden: {online_final.shape}, prediction: {online_prediction.shape}")
    print()

    # =========================================================================
    # Part 3: wm-raw forward
    # =========================================================================
    print("=" * 70)
    print("[wm-raw] Building model and running forward...")
    print("=" * 70)

    from wm_raw.config import WorldModelConfig
    from wm_raw.models.model import WorldModel

    timer_raw = Timer()

    with timer_raw.stage("construct WorldModel"):
        config = WorldModelConfig()
        raw_model = WorldModel(config)

    # Load .pt checkpoint
    from wm_raw.checkpoint import _map_online_key

    with timer_raw.stage("torch.load .pt file"):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]

    # Load VLM from pretrained
    from wm_raw.checkpoint import load_vlm_weights
    with timer_raw.stage("load VLM weights"):
        load_vlm_weights(raw_model, args.vlm_path)

    # Load diffusion+CA from .pt
    with timer_raw.stage("remap + load diffusion/CA weights"):
        model_sd = raw_model.state_dict()
        remapped = {}
        for key, tensor in state_dict.items():
            mapped = _map_online_key(f"model.{key}")
            if mapped is None:
                continue
            stripped = mapped[len("model."):] if mapped.startswith("model.") else mapped
            if stripped in model_sd and tensor.shape == model_sd[stripped].shape:
                remapped[stripped] = tensor
        raw_model.load_state_dict(remapped, strict=False)
        print(f"  Loaded {len(remapped)} params")

    with timer_raw.stage("model to device"):
        raw_model = raw_model.to(device=device, dtype=dtype)
        raw_model.eval()

    timer_raw.summary()

    # VLM forward — build same mask as generate_image.py
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    attn_mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        vlm_out_raw = raw_model.forward_vlm(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
    raw_hidden_states = vlm_out_raw.hidden_states
    print(f"  VLM hidden_states: {len(raw_hidden_states)} layers, shape={raw_hidden_states[0].shape}")

    # Diffusion forward (same noise, same timestep)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        raw_prepared_hidden, raw_time_hidden = raw_model.state_diffusion.prepare_inputs(
            noise, timesteps, patch_h=patch_h, patch_w=patch_w
        )

    print(f"  Prepared hidden: {raw_prepared_hidden.shape}, time_hidden: {raw_time_hidden.shape}")

    # Run diffusion layers
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        hidden_raw = raw_prepared_hidden
        batch = hidden_raw.shape[0]
        num_tok = hidden_raw.shape[1]

        # Build MRoPE position ids (2D spatial)
        row_ids = torch.arange(patch_h, device=device, dtype=torch.long).repeat_interleave(patch_w)
        col_ids = torch.arange(patch_w, device=device, dtype=torch.long).repeat(patch_h)
        temporal_ids = torch.zeros(num_tok, device=device, dtype=torch.long)
        mrope_pos_ids = torch.stack([temporal_ids, row_ids, col_ids], dim=0).unsqueeze(1).expand(-1, batch, -1)
        cos, sin = raw_model.state_diffusion.rotary_emb(mrope_pos_ids)

        all_ext_kv = raw_model.cross_attention.project_all_context_kv(
            raw_hidden_states, target_device=device, target_dtype=dtype
        )

        raw_layer_outputs = []
        for layer_idx, (layer, adaln) in enumerate(
            zip(raw_model.state_diffusion.layers, raw_model.state_diffusion.adaln_layers)
        ):
            adaln_params = adaln(raw_time_hidden)
            ext_kv = all_ext_kv[layer_idx]
            hidden_raw = layer(
                hidden_raw, attention_mask=None,
                position_embeddings=(cos, sin),
                external_kv=ext_kv, adaln_params=adaln_params,
            )
            if layer_idx < 3 or layer_idx == 27:
                raw_layer_outputs.append((layer_idx, hidden_raw.detach().clone()))

        raw_final = raw_model.state_diffusion.final_norm(hidden_raw).detach().clone()
        raw_prediction = raw_model.state_diffusion.output_head(raw_final).detach().clone()

    print(f"  Final hidden: {raw_final.shape}, prediction: {raw_prediction.shape}")
    print()

    # =========================================================================
    # Part 4: Compare
    # =========================================================================
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\n[VLM Hidden States]")
    # Compare a few VLM layers
    for i in [0, 1, 18, 35, 36]:
        if i < len(online_hidden_states) and i < len(raw_hidden_states):
            compare(f"vlm.hidden_state[{i}]", raw_hidden_states[i], online_hidden_states[i])

    print("\n[Diffusion Prepared Hidden]")
    compare("prepared_hidden", raw_prepared_hidden, online_prepared_hidden)
    compare("time_hidden", raw_time_hidden, online_time_hidden)

    print("\n[Diffusion Layer Outputs]")
    for (idx_o, out_o), (idx_r, out_r) in zip(online_layer_outputs, raw_layer_outputs):
        assert idx_o == idx_r
        compare(f"layer[{idx_o}]", out_r, out_o)

    print("\n[Final Output]")
    compare("final_norm", raw_final, online_final)
    compare("prediction (velocity)", raw_prediction, online_prediction)

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
