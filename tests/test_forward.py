"""End-to-end smoke test for WorldModel forward pass.

Instantiates a tiny config, runs VLM + Diffusion forward, verifies shapes and loss.
Run with: python tests/test_forward.py
"""

import sys
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


def make_tiny_config() -> WorldModelConfig:
    return WorldModelConfig(
        text=TextModelConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mrope_section=(6, 5, 5),
        ),
        vision=VisionModelConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            head_dim=8,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            rope_theta=10000.0,
        ),
        diffusion=DiffusionConfig(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=3,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=16,
            mrope_section=(6, 5, 5),
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


def test_vlm_forward():
    """Test VLM branch forward produces hidden states and AR loss."""
    print("=== test_vlm_forward ===")
    config = make_tiny_config()
    model = WorldModel(config)
    model.eval()

    batch, seq_len = 2, 32
    input_ids = torch.randint(0, 1000, (batch, seq_len))
    # 3D MRoPE positions: [3, B, S]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]
    # Causal mask: [B, 1, S, S]
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
    labels = torch.randint(0, 1000, (batch, seq_len))

    with torch.no_grad():
        vlm_output = model.forward_vlm(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            labels=labels,
        )

    assert vlm_output.ar_loss is not None, "AR loss should be computed"
    assert vlm_output.ar_loss.ndim == 0, "AR loss should be scalar"
    # hidden_states: list of num_layers + 1 tensors
    assert len(vlm_output.hidden_states) == config.text.num_hidden_layers + 1
    for hs in vlm_output.hidden_states:
        assert hs.shape == (batch, seq_len, config.text.hidden_size)

    print(f"  AR loss: {vlm_output.ar_loss.item():.4f}")
    print(f"  Hidden states: {len(vlm_output.hidden_states)} layers")
    print(f"  Shape: {vlm_output.hidden_states[0].shape}")
    print("  PASSED")


def test_diffusion_forward():
    """Test diffusion branch forward produces loss."""
    print("\n=== test_diffusion_forward ===")
    config = make_tiny_config()
    model = WorldModel(config)
    model.eval()

    batch = 2
    # Simulate VLM hidden states (normally from VLM forward)
    num_vlm_states = config.text.num_hidden_layers + 1
    vlm_seq_len = 32
    vlm_hidden_states = [
        torch.randn(batch, vlm_seq_len, config.text.hidden_size)
        for _ in range(num_vlm_states)
    ]

    # State target: raw VAE latents [B, H*W, C]
    state_target = torch.randn(
        batch,
        config.latent.latent_height * config.latent.latent_width,
        config.latent.latent_channels,
    )

    with torch.no_grad():
        diff_output = model.forward_diffusion(
            state_target=state_target,
            vlm_hidden_states=vlm_hidden_states,
        )

    assert diff_output.loss is not None, "Diffusion loss should be computed"
    assert diff_output.loss.ndim == 0, "Loss should be scalar"
    # Prediction shape: [B, num_patches, patch_dim]
    expected_tokens = (config.latent.latent_height // config.latent.patch_size) * (
        config.latent.latent_width // config.latent.patch_size
    )
    expected_dim = config.latent.patch_size**2 * config.latent.latent_channels
    assert diff_output.prediction.shape == (batch, expected_tokens, expected_dim)

    print(f"  Diffusion loss: {diff_output.loss.item():.4f}")
    print(f"  Prediction shape: {diff_output.prediction.shape}")
    print("  PASSED")


def test_full_forward():
    """Test combined VLM + Diffusion forward (joint training step)."""
    print("\n=== test_full_forward ===")
    config = make_tiny_config()
    model = WorldModel(config)
    model.eval()

    batch, seq_len = 2, 32

    # VLM inputs
    input_ids = torch.randint(0, 1000, (batch, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
    labels = torch.randint(0, 1000, (batch, seq_len))

    # Diffusion inputs
    state_target = torch.randn(
        batch,
        config.latent.latent_height * config.latent.latent_width,
        config.latent.latent_channels,
    )

    with torch.no_grad():
        output = model.forward_joint(
            input_ids=input_ids,
            attention_mask=causal_mask,
            position_ids=position_ids,
            labels=labels,
            state_target=state_target,
        )

    assert output.loss is not None
    assert output.ar_loss is not None
    assert output.diffusion_loss is not None
    print(f"  Total loss: {output.loss.item():.4f}")
    print(f"  AR loss: {output.ar_loss.item():.4f}")
    print(f"  Diffusion loss: {output.diffusion_loss.item():.4f}")
    print("  PASSED")


def test_backward():
    """Test that gradients flow through the full model."""
    print("\n=== test_backward ===")
    config = make_tiny_config()
    model = WorldModel(config)
    model.train()

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, 1000, (batch, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    ).unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
    labels = torch.randint(0, 1000, (batch, seq_len))
    state_target = torch.randn(
        batch,
        config.latent.latent_height * config.latent.latent_width,
        config.latent.latent_channels,
    )

    output = model.forward_joint(
        input_ids=input_ids,
        attention_mask=causal_mask,
        position_ids=position_ids,
        labels=labels,
        state_target=state_target,
    )
    output.loss.backward()

    # Check some gradients are non-zero
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    non_zero = sum(1 for v in grad_norms.values() if v > 0)
    total = len(grad_norms)
    print(f"  Parameters with grad: {total}")
    print(f"  Non-zero grads: {non_zero}/{total}")
    assert non_zero > 0, "At least some gradients should be non-zero"
    print("  PASSED")


if __name__ == "__main__":
    print("Running wm-raw forward tests...\n")
    test_vlm_forward()
    test_diffusion_forward()
    test_full_forward()
    test_backward()
    print("\n=== ALL TESTS PASSED ===")
