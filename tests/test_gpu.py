"""GPU + torch.compile smoke test for wm-raw.

Tests:
1. Model forward on CUDA with bf16
2. torch.compile (fullgraph) compatibility
3. Memory usage sanity check

Run with: python tests/test_gpu.py
"""

import sys
sys.path.insert(0, "src")

import time
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


def make_small_config() -> WorldModelConfig:
    """Slightly larger config for GPU testing (still not full model)."""
    return WorldModelConfig(
        text=TextModelConfig(
            vocab_size=4096,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            mrope_section=(12, 10, 10),
        ),
        vision=VisionModelConfig(
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            head_dim=32,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=256,
            rope_theta=10000.0,
        ),
        diffusion=DiffusionConfig(
            hidden_size=192,
            intermediate_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            num_key_value_heads=2,
            head_dim=32,
            mrope_section=(12, 10, 10),
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


def make_inputs(config: WorldModelConfig, batch: int, seq_len: int, device: torch.device):
    """Create dummy inputs on device."""
    input_ids = torch.randint(0, config.text.vocab_size, (batch, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
    labels = torch.randint(0, config.text.vocab_size, (batch, seq_len), device=device)
    state_target = torch.randn(
        batch,
        config.latent.latent_height * config.latent.latent_width,
        config.latent.latent_channels,
        device=device,
    )
    return input_ids, causal_mask, position_ids, labels, state_target


def test_cuda_bf16():
    """Test forward + backward on CUDA with bf16 autocast."""
    print("=== test_cuda_bf16 ===")
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    device = torch.device("cuda:0")
    config = make_small_config()
    model = WorldModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {params / 1e6:.2f}M")

    batch, seq_len = 4, 64
    input_ids, causal_mask, position_ids, labels, state_target = make_inputs(
        config, batch, seq_len, device
    )

    torch.cuda.reset_peak_memory_stats()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_joint(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            labels=labels,
            state_target=state_target,
        )

    output.loss.backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"  Total loss: {output.loss.item():.4f}")
    print(f"  AR loss: {output.ar_loss.item():.4f}")
    print(f"  Diffusion loss: {output.diffusion_loss.item():.4f}")
    print(f"  Peak GPU memory: {peak_mb:.1f} MB")
    print("  PASSED")


def test_torch_compile():
    """Test torch.compile compatibility (fullgraph mode)."""
    print("\n=== test_torch_compile ===")
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    device = torch.device("cuda:0")
    config = make_small_config()
    model = WorldModel(config).to(device)

    batch, seq_len = 2, 32
    input_ids, causal_mask, position_ids, labels, state_target = make_inputs(
        config, batch, seq_len, device
    )

    # Try compiling just the VLM branch
    try:
        compiled_vlm = torch.compile(model.vlm, fullgraph=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                vlm_out = compiled_vlm(
                    input_ids=input_ids,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    labels=labels,
                )
        print(f"  VLM compile: OK (loss={vlm_out.ar_loss.item():.4f})")
    except Exception as e:
        print(f"  VLM compile FAILED: {e}")

    # Try compiling diffusion branch
    try:
        compiled_diff = torch.compile(model.state_diffusion, fullgraph=True)
        vlm_hidden_states = [
            torch.randn(batch, seq_len, config.text.hidden_size, device=device)
            for _ in range(config.text.num_hidden_layers + 1)
        ]
        from wm_raw.diffusion import sample_timesteps
        from wm_raw.models.embeddings import patchify_latent

        clean_tokens = patchify_latent(
            state_target,
            height=config.latent.latent_height,
            width=config.latent.latent_width,
            patch_size=config.latent.patch_size,
        )
        timesteps = sample_timesteps(batch, device=device)
        noisy = clean_tokens + 0.1 * torch.randn_like(clean_tokens)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                pred = compiled_diff(
                    noisy_latent=noisy,
                    timesteps=timesteps,
                    cross_attention_contexts=vlm_hidden_states,
                    cross_attention_fn=model.cross_attention.condition_layer,
                )
        print(f"  Diffusion compile: OK (pred shape={pred.shape})")
    except Exception as e:
        print(f"  Diffusion compile FAILED: {e}")

    print("  DONE")


def test_throughput():
    """Measure training throughput (steps/sec)."""
    print("\n=== test_throughput ===")
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)")
        return

    device = torch.device("cuda:0")
    config = make_small_config()
    model = WorldModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch, seq_len = 4, 64
    input_ids, causal_mask, position_ids, labels, state_target = make_inputs(
        config, batch, seq_len, device
    )

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_joint(
                input_ids=input_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                labels=labels,
                state_target=state_target,
            )
        output.loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    num_steps = 10
    start = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_joint(
                input_ids=input_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                labels=labels,
                state_target=state_target,
            )
        output.loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"  {num_steps} steps in {elapsed:.2f}s = {num_steps/elapsed:.1f} steps/sec")
    print(f"  Final loss: {output.loss.item():.4f}")
    print("  PASSED")


if __name__ == "__main__":
    print("Running wm-raw GPU tests...\n")
    test_cuda_bf16()
    test_torch_compile()
    test_throughput()
    print("\n=== ALL GPU TESTS DONE ===")
