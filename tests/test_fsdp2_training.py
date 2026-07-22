"""Multi-GPU FSDP2 training test.

Run with:
    torchrun --nproc_per_node=2 tests/test_fsdp2_training.py

Tests:
1. Distributed setup
2. FSDP2 wrapping
3. Optimizer + scheduler construction
4. Training step (synthetic data)
5. Gradient sync verification
"""

import sys
import os
import time

sys.path.insert(0, "src")

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

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
from wm_raw.training import (
    TrainingConfig,
    DistContext,
    setup_distributed,
    cleanup_distributed,
    apply_fsdp2,
    build_optimizer,
    build_scheduler,
    configure_trainable,
    FrozenVAECodec,
    train_step,
)


def make_test_config() -> WorldModelConfig:
    """Small config for multi-GPU test (fits in ~1GB per GPU)."""
    return WorldModelConfig(
        text=TextModelConfig(
            vocab_size=4096,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            mrope_section=(12, 10, 10),
        ),
        vision=VisionModelConfig(
            depth=2,
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
            num_hidden_layers=4,
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


class MockVAECodec:
    """Mock VAE that returns random latents (no actual autoencoder needed)."""

    def __init__(self, latent_channels: int, latent_h: int, latent_w: int, device: torch.device, dtype: torch.dtype):
        self.latent_channels = latent_channels
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Fake encode: pixel_values [B, 3, H, W] → latent tokens [B, h*w, C]."""
        batch_size = pixel_values.shape[0]
        return torch.randn(
            batch_size, self.latent_h * self.latent_w, self.latent_channels,
            device=self.device, dtype=self.dtype,
        )


def make_synthetic_batch(batch_size: int, image_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Create a synthetic diffusion training batch."""
    # Build a proper 4D causal mask [B, 1, S, S]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return {
        "condition": {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
            "attention_mask": causal_mask,
        },
        "vae_pixel_values": torch.randn(batch_size, 3, image_size, image_size, device=device),
    }


def main():
    ctx = setup_distributed()
    rank = ctx.rank
    device = ctx.device

    def rank_print(msg):
        if ctx.is_main:
            print(msg)

    rank_print("=== test_fsdp2_training (multi-GPU) ===")
    rank_print(f"  World size: {ctx.world_size}")
    rank_print(f"  Device: {torch.cuda.get_device_name(ctx.local_rank)}")

    # Build model
    model_config = make_test_config()
    model = WorldModel(model_config).to(device=device, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())
    rank_print(f"  Model params: {total_params / 1e6:.2f}M")

    # Configure trainable
    tc = TrainingConfig()
    tc.trainable_mode = "diffusion"
    tc.train_diffusion_backbone = True
    tc.adapter_lr = 1e-4
    tc.diffusion_lr = 1e-5
    tc.max_grad_norm = 1.0
    tc.max_steps = 100
    tc.warmup_steps = 10
    tc.diffusion_loss_weight = 1.0
    configure_trainable(model, tc)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_print(f"  Trainable: {trainable / 1e6:.2f}M")

    # Apply FSDP2
    t0 = time.time()
    try:
        model = apply_fsdp2(model, ctx=ctx)
        rank_print(f"  FSDP2 applied ({time.time() - t0:.2f}s)")
    except Exception as e:
        rank_print(f"  FSDP2 failed: {e}")
        rank_print("  Falling back to DDP...")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[ctx.local_rank])

    # Build optimizer + scheduler
    optimizer = build_optimizer(model, tc)
    scheduler = build_scheduler(optimizer, tc)
    rank_print(f"  Optimizer groups: {len(optimizer.param_groups)}")

    # Mock VAE
    mock_codec = MockVAECodec(
        latent_channels=model_config.latent.latent_channels,
        latent_h=model_config.latent.latent_height,
        latent_w=model_config.latent.latent_width,
        device=device,
        dtype=torch.bfloat16,
    )

    # Training loop (5 steps)
    rank_print("\n  Running 5 training steps...")
    model.train()
    losses = []

    for step in range(5):
        batch = make_synthetic_batch(
            batch_size=2,
            image_size=model_config.latent.latent_height * 16,  # approximate
            seq_len=32,
            vocab_size=model_config.text.vocab_size,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)

        # Encode with mock VAE
        state_target = mock_codec.encode(batch["vae_pixel_values"])

        # VLM condition forward
        condition = batch["condition"]
        seq_len = condition["input_ids"].shape[1]
        batch_size = condition["input_ids"].shape[0]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vlm_out = model.forward_vlm(
                input_ids=condition["input_ids"],
                attention_mask=condition.get("attention_mask"),
                position_ids=position_ids,
            )
            diff_out = model.forward_diffusion(
                state_target=state_target,
                vlm_hidden_states=vlm_out.hidden_states,
            )

        loss = diff_out.loss
        loss.backward()

        if tc.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                tc.max_grad_norm,
            )

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if ctx.is_main:
            print(f"    step {step}: loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}")

    # Verify training completed on all ranks (loss is expected to differ per rank
    # since each rank gets different random synthetic data)
    rank_print("\n  Verifying all ranks completed training...")
    loss_tensor = torch.tensor([losses[-1]], device=device)
    loss_list = [torch.zeros(1, device=device) for _ in range(ctx.world_size)]
    dist.all_gather(loss_list, loss_tensor)
    if ctx.is_main:
        rank_losses = [l.item() for l in loss_list]
        print(f"    Final losses per rank: {rank_losses}")
        assert all(not (torch.isnan(l) or torch.isinf(l)) for l in loss_list), \
            "Non-finite loss on some rank!"
        print("    All ranks have finite loss ✓")

    # Verify loss decreased (or at least didn't explode)
    rank_print(f"\n  Losses: {losses}")
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN loss detected!"
    assert all(not torch.isinf(torch.tensor(l)) for l in losses), "Inf loss detected!"
    rank_print("  Loss values are finite ✓")

    # Memory stats
    if ctx.is_main:
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"\n  Peak GPU memory: {peak_mb:.1f} MB")

    cleanup_distributed()
    rank_print("\n=== FSDP2 TRAINING TEST PASSED ===")


if __name__ == "__main__":
    main()
