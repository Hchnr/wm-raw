"""Model configuration dataclasses for wm-raw.

All configs are plain dataclasses with explicit types — no dynamic dispatch,
no Mapping[str, Any] bags. Validation happens at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextModelConfig:
    """Qwen3-VL text decoder configuration (matches HF Qwen3-VL-4B-Instruct)."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    max_position_embeddings: int = 262144


@dataclass(frozen=True)
class VisionModelConfig:
    """Qwen3-VL vision encoder configuration (matches Qwen3-VL-4B-Instruct)."""

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    head_dim: int = 64  # hidden_size // num_heads
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2560  # projects to text hidden_size via merger
    rope_theta: float = 10000.0
    num_position_embeddings: int = 2304


@dataclass(frozen=True)
class CrossAttentionConfig:
    """Cross-attention from VLM hidden states to diffusion branch."""

    enabled: bool = True
    # cross_kv_concat: concat VLM KV to diffusion self-attention KV
    communication_policy: str = "cross_kv_concat"
    gate_init: float = 0.01
    zero_init_output: bool = False
    norm_eps: float = 1e-6
    hidden_state_layer_offset: int = 1


@dataclass(frozen=True)
class LatentConfig:
    """Image latent tokenization settings."""

    # VAE latent shape: [C, H, W] for 256x256 images with BAGEL AE (downsample 8x)
    latent_channels: int = 16
    latent_height: int = 32
    latent_width: int = 32
    # Patchify: 2x2 patches → 256 tokens of dim 64
    # num_tokens = (H/P)*(W/P) = 256, token_dim = P*P*C = 64
    patch_size: int = 2
    position_embedding: str = "bagel_2d_sincos"
    max_position_size: int = 64


@dataclass(frozen=True)
class DiffusionConfig:
    """State diffusion branch configuration (matches Qwen3-VL-2B-Instruct)."""

    # Diffusion backbone — Qwen3-VL-2B
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)

    # Diffusion objective
    prediction_type: str = "flow"  # flow matching velocity prediction
    timestep_shift: float = 2.0
    timestep_frequency_dim: int = 256

    # Target (state_target_dim: patchified latent token dim = P*P*C = 2*2*16 = 64)
    target_dim: int = 64


@dataclass(frozen=True)
class LayerMappingConfig:
    """How diffusion layers map to VLM layers for cross-attention."""

    policy: str = "middle_n"


@dataclass(frozen=True)
class WorldModelConfig:
    """Top-level model configuration assembling all components."""

    text: TextModelConfig = field(default_factory=TextModelConfig)
    vision: VisionModelConfig = field(default_factory=VisionModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    layer_mapping: LayerMappingConfig = field(default_factory=LayerMappingConfig)

    # Loss weights
    ar_loss_weight: float = 1.0
    state_diffusion_loss_weight: float = 1.0

    # Precision
    torch_dtype: str = "bfloat16"

    def validate(self) -> None:
        """Check config consistency."""
        supported_policies = ("cross_kv_down", "cross_kv_concat")
        if self.cross_attention.communication_policy not in supported_policies:
            raise ValueError(
                f"wm-raw only supports {supported_policies}, "
                f"got {self.cross_attention.communication_policy!r}"
            )
        if self.diffusion.prediction_type != "flow":
            raise ValueError("wm-raw only supports flow matching prediction type")
        if self.latent.patch_size < 1:
            raise ValueError("latent patch_size must be >= 1")
