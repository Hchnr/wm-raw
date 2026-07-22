"""Full-model weight loading test against real Qwen3-VL-4B + 2B checkpoints.

Run on GPU machine with:
    python tests/test_full_model_load.py
"""

import sys
import time

sys.path.insert(0, "src")

import torch
from wm_raw.config import (
    CrossAttentionConfig,
    DiffusionConfig,
    LatentConfig,
    LayerMappingConfig,
    TextModelConfig,
    VisionModelConfig,
    WorldModelConfig,
)
from wm_raw.models import WorldModel
from wm_raw.checkpoint import (
    load_vlm_weights,
    load_diffusion_weights,
    load_hf_safetensors,
    _map_vlm_key,
    _map_diffusion_key,
)

# Real model paths
VLM_PATH = "/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct"
DIFFUSION_PATH = "/share/project/eai_pwm/models/Qwen3-VL-2B-Instruct"


def make_full_config() -> WorldModelConfig:
    """Config matching real Qwen3-VL-4B (VLM) + Qwen3-VL-2B (diffusion)."""
    return WorldModelConfig(
        text=TextModelConfig(
            vocab_size=151936,
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            mrope_section=(24, 20, 20),
        ),
        vision=VisionModelConfig(
            depth=24,
            hidden_size=1024,
            intermediate_size=4096,
            num_heads=16,
            head_dim=64,  # 1024 / 16 = 64
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=2560,
            rope_theta=10000.0,
        ),
        diffusion=DiffusionConfig(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            mrope_section=(24, 20, 20),
            target_dim=16,
        ),
        cross_attention=CrossAttentionConfig(gate_init=0.01),
        latent=LatentConfig(
            latent_channels=64,
            latent_height=16,
            latent_width=16,
            patch_size=2,
        ),
        layer_mapping=LayerMappingConfig(policy="middle_n"),
    )


def test_key_mapping_coverage():
    """Test that HF checkpoint keys map correctly to model keys."""
    print("=== test_key_mapping_coverage ===")

    # Load HF state dict keys (don't load tensors to GPU)
    vlm_sd = load_hf_safetensors(VLM_PATH)
    diff_sd = load_hf_safetensors(DIFFUSION_PATH)

    # Check VLM key mapping
    vlm_mapped = 0
    vlm_unmapped = []
    for key in vlm_sd:
        raw_key = _map_vlm_key(key)
        if raw_key is not None:
            vlm_mapped += 1
        else:
            vlm_unmapped.append(key)

    print(f"  VLM: {vlm_mapped} mapped, {len(vlm_unmapped)} unmapped")
    if vlm_unmapped:
        print(f"  First 10 unmapped: {vlm_unmapped[:10]}")

    # Check diffusion key mapping
    diff_mapped = 0
    diff_unmapped = []
    for key in diff_sd:
        raw_key = _map_diffusion_key(key)
        if raw_key is not None:
            diff_mapped += 1
        else:
            diff_unmapped.append(key)

    print(f"  Diffusion: {diff_mapped} mapped, {len(diff_unmapped)} unmapped (expected: embed/lm_head/vision)")
    # Unmapped should only be embed_tokens, lm_head, visual.*
    unexpected_unmapped = [k for k in diff_unmapped if not any(
        k.startswith(p) for p in ("model.embed_tokens", "lm_head", "visual.", "model.visual.")
    )]
    if unexpected_unmapped:
        print(f"  WARNING: unexpected unmapped keys: {unexpected_unmapped[:10]}")

    print("  PASSED")
    del vlm_sd, diff_sd


def test_full_model_load():
    """Build full-size model and load real HF weights."""
    print("\n=== test_full_model_load ===")
    config = make_full_config()

    t0 = time.time()
    print("  Building model...")
    model = WorldModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built: {total_params / 1e9:.2f}B params ({time.time() - t0:.1f}s)")

    # Load VLM weights
    t0 = time.time()
    print("  Loading VLM weights...")
    vlm_report = load_vlm_weights(model, VLM_PATH, dtype=torch.bfloat16)
    print(f"  VLM: matched={vlm_report.matched}, missing={len(vlm_report.missing)}, "
          f"unexpected={len(vlm_report.unexpected)} ({time.time() - t0:.1f}s)")
    if vlm_report.missing:
        print(f"    Missing (first 20): {list(vlm_report.missing[:20])}")
    if vlm_report.shape_mismatch:
        print(f"    Shape mismatch: {list(vlm_report.shape_mismatch[:10])}")

    # Load diffusion weights
    t0 = time.time()
    print("  Loading diffusion weights...")
    diff_report = load_diffusion_weights(model, DIFFUSION_PATH, dtype=torch.bfloat16)
    print(f"  Diffusion: matched={diff_report.matched}, missing={len(diff_report.missing)}, "
          f"unexpected={len(diff_report.unexpected)} ({time.time() - t0:.1f}s)")
    if diff_report.missing:
        print(f"    Missing (first 20): {list(diff_report.missing[:20])}")
    if diff_report.shape_mismatch:
        print(f"    Shape mismatch: {list(diff_report.shape_mismatch[:10])}")

    # Validation
    assert vlm_report.matched > 0, "No VLM weights loaded!"
    assert diff_report.matched > 0, "No diffusion weights loaded!"
    assert not vlm_report.shape_mismatch, f"VLM shape mismatches: {vlm_report.shape_mismatch}"
    assert not diff_report.shape_mismatch, f"Diff shape mismatches: {diff_report.shape_mismatch}"

    # Report coverage
    vlm_model_keys = [k for k in model.state_dict() if k.startswith("vlm.")]
    diff_model_keys = [k for k in model.state_dict()
                       if k.startswith("state_diffusion.layers.") or k.startswith("state_diffusion.final_norm.")]
    print(f"  VLM coverage: {vlm_report.matched}/{len(vlm_model_keys)} "
          f"({100*vlm_report.matched/max(len(vlm_model_keys),1):.1f}%)")
    print(f"  Diffusion coverage: {diff_report.matched}/{len(diff_model_keys)} "
          f"({100*diff_report.matched/max(len(diff_model_keys),1):.1f}%)")

    print("  PASSED")
    return model


def test_forward_with_loaded_weights(model):
    """Run a forward pass on GPU with loaded weights."""
    print("\n=== test_forward_with_loaded_weights ===")
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    device = torch.device("cuda:0")
    model = model.to(device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()

    batch_size, seq_len = 1, 32
    config = model.config

    input_ids = torch.randint(0, config.text.vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    # latent_height=16, latent_width=16, patch_size=2 → 64 tokens, dim=256
    latent_h = config.latent.latent_height // config.latent.patch_size  # 8
    latent_w = config.latent.latent_width // config.latent.patch_size  # 8
    n_tokens = latent_h * latent_w  # 64
    target_dim = config.latent.latent_channels * config.latent.patch_size ** 2  # 64 * 4 = 256
    state_target = torch.randn(batch_size, n_tokens, target_dim, device=device, dtype=torch.bfloat16)

    print(f"  Forward pass: batch={batch_size}, seq_len={seq_len}, latent_tokens={n_tokens}")

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            output = model.forward_joint(
                input_ids=input_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                labels=input_ids,
                state_target=state_target,
            )

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total loss: {output.loss.item():.4f}")
    print(f"  AR loss: {output.ar_loss.item():.4f}")
    print(f"  Diffusion loss: {output.diffusion_loss.item():.4f}")
    print(f"  Peak GPU memory: {peak_gb:.2f} GB")
    print("  PASSED")


if __name__ == "__main__":
    print("Running full-model loading tests...\n")
    test_key_mapping_coverage()
    model = test_full_model_load()
    test_forward_with_loaded_weights(model)
    print("\n=== ALL FULL-MODEL TESTS DONE ===")
