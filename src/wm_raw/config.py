"""Model configuration dataclasses for wm-raw.

All configs are plain dataclasses with explicit types — no dynamic dispatch,
no Mapping[str, Any] bags. Validation happens at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextModelConfig:
    """Qwen3-VL text decoder configuration (matches HF Qwen3VLTextConfig)."""

    vocab_size: int = 151936
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 36
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    max_position_embeddings: int = 128000


@dataclass(frozen=True)
class VisionModelConfig:
    """Qwen3-VL vision encoder configuration."""

    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    head_dim: int = 72  # hidden_size // num_heads
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584  # projects to text hidden_size
    rope_theta: float = 10000.0
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)


@dataclass(frozen=True)
class CrossAttentionConfig:
    """Cross-attention from VLM hidden states to diffusion branch."""

    enabled: bool = True
    # Only cross_kv_down is implemented in wm-raw
    communication_policy: str = "cross_kv_down"
    gate_init: float = 0.01
    zero_init_output: bool = False
    norm_eps: float = 1e-6
    hidden_state_layer_offset: int = 1


@dataclass(frozen=True)
class LatentConfig:
    """Image latent tokenization settings."""

    # VAE latent shape: [C, H, W] for 256x256 images with BAGEL AE
    latent_channels: int = 64
    latent_height: int = 16
    latent_width: int = 16
    # Patchify: 2x2 patches → 256 tokens of dim 256
    patch_size: int = 2
    # After patchify: num_tokens = (H/P)*(W/P) = 64, token_dim = P*P*C = 256
    position_embedding: str = "bagel_2d_sincos"


@dataclass(frozen=True)
class DiffusionConfig:
    """State diffusion branch configuration."""

    # Diffusion backbone — typically Qwen3-VL-2B
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)

    # Diffusion objective
    prediction_type: str = "flow"  # flow matching velocity prediction
    timestep_shift: float = 1.0

    # Target
    target_dim: int = 16  # state_target_dim from config


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
        if self.cross_attention.communication_policy != "cross_kv_down":
            raise ValueError("wm-raw only supports cross_kv_down communication policy")
        if self.diffusion.prediction_type != "flow":
            raise ValueError("wm-raw only supports flow matching prediction type")
        if self.latent.patch_size < 1:
            raise ValueError("latent patch_size must be >= 1")
