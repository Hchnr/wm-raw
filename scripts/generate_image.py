#!/usr/bin/env python3
"""Text-to-image generation using a wm-raw checkpoint.

Loads a trained wm-raw checkpoint (.pt or FSDP DCP directory) and generates
images via Euler sampling of the flow-matching diffusion model with optional
classifier-free guidance.

Condition format matches training: "Caption: {prompt} <|wm_predict_image|>"
CFG unconditional uses sentinel-only: "<|wm_predict_image|>"

Usage (our own DCP checkpoint):
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_image.py \
        --checkpoint outputs/gpic_image_diffusion/checkpoints/step-000010 \
        --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
        --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
        --prompt "a photo of a cat sitting on a windowsill" \
        --num-steps 50 --timestep-shift 1.0 --cfg-scale 5.0 --seed 42 \
        --image-height 512 --image-width 512 \
        --output generated.png --device cuda

Usage (online DCP checkpoint, comparable to interactive_generate_image.py):
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_image.py \
        --checkpoint /share/project/eai_pwm/repos/wm-training/outputs/qwen3vl_gpic_patchlatent_2dpos_adaln_fm_logitnormal_buckets_512_stage2_from_step145000_fsdp/checkpoints/step_285000.dcp \
        --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
        --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
        --prompt "a photo of a cat sitting on a windowsill" \
        --num-steps 50 --timestep-shift 1.0 --cfg-scale 5.0 --seed 42 \
        --image-height 512 --image-width 512 \
        --output generated.png --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wm_raw.checkpoint import load_checkpoint, load_vae
from wm_raw.config import WorldModelConfig
from wm_raw.models.embeddings import unpatchify_latent
from wm_raw.models.model import WorldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DCP loading utility for our own checkpoints
# ---------------------------------------------------------------------------


def load_own_dcp_checkpoint(model: WorldModel, dcp_path: Path) -> None:
    """Load a wm-raw FSDP DCP checkpoint (keys prefixed with 'model.').

    Uses DCP no_dist mode for single-process inference loading.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # For single-process loading: provide a plain state dict matching the DCP key structure
    state_dict: dict[str, object] = {"model": model.state_dict()}
    dcp.load(
        state_dict,
        checkpoint_id=str(dcp_path),
        no_dist=True,
    )
    # The DCP stores 'model.*' which maps directly to our model state_dict
    model.load_state_dict(state_dict["model"], strict=True)
    logger.info("Loaded own DCP checkpoint: %s (keys=%d)", dcp_path, len(state_dict["model"]))


def load_online_dcp_for_inference(
    model: WorldModel,
    dcp_path: Path,
    vlm_path: str,
    *,
    dtype: torch.dtype | None = None,
) -> None:
    """Load an online (wm-training) DCP checkpoint for inference.

    Strategy:
    - VLM branch: loaded from pretrained Qwen3-VL weights (fast, ~8B from safetensors)
    - Diffusion + cross-attention: loaded from EMA state in the DCP (5 GB, much faster
      than loading the full 22 GB model state + 74 GB total checkpoint)

    This avoids the slow full-DCP single-process load.
    """
    import functools
    import operator

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    # 1. Load VLM from pretrained weights (fast path)
    logger.info("Loading VLM weights from %s ...", vlm_path)
    from wm_raw.checkpoint import load_vlm_weights
    vlm_report = load_vlm_weights(model, vlm_path)
    logger.info("VLM: %s", vlm_report.format())

    # 2. Load EMA (diffusion + cross_attention) from DCP
    logger.info("Loading EMA weights from DCP (diffusion + cross_attention)...")
    reader = FileSystemReader(str(dcp_path))
    metadata = reader.read_metadata()
    all_keys = list(metadata.state_dict_metadata.keys())

    # Select EMA keys for state_diffusion_branch and cross_attention only
    ema_keys = [
        k for k in all_keys
        if k.startswith("ema.") and "action_diffusion_branch" not in k
    ]
    logger.info("  EMA keys to load: %d", len(ema_keys))

    # Build placeholder state dict
    state_dict: dict[str, Tensor] = {}
    for key in ema_keys:
        md = metadata.state_dict_metadata[key]
        if isinstance(md, TensorStorageMetadata):
            state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype)

    dcp.load(state_dict, checkpoint_id=str(dcp_path))
    logger.info("  DCP load complete, remapping keys...")

    # 3. Remap EMA keys to model keys
    # EMA key format: ema.{cross_attention|state_diffusion_branch}.adapters.state.X.Y
    # We need to map to: {cross_attention|state_diffusion_branch}.X.Y
    # But first, the online format uses different naming than our model.
    # Let's use the existing _map_online_key with "model." prefix substitution.
    from wm_raw.checkpoint import _map_online_key

    remapped: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        # Convert ema.X... to model.X... for the key mapper
        # EMA stores: ema.cross_attention.adapters.state.0.context_norm.weight
        #             ema.state_diffusion_branch.adaln_modulations.state.0.modulation.1.weight
        #             ema.state_diffusion_branch.backbone.layers.state.0.self_attn.q_proj.weight
        model_key = key.replace("ema.", "model.", 1)

        # The _map_online_key prefix map already handles cross_attention ".state." removal.
        # For diffusion branch, the online model doesn't have ".state." in its model keys,
        # but EMA tracking inserts it for ModuleList. Remove it for diffusion sub-modules.
        if "state_diffusion_branch" in model_key:
            model_key = model_key.replace(".adaln_modulations.state.", ".adaln_modulations.", 1)
            model_key = model_key.replace(".layers.state.", ".layers.", 1)

        mapped = _map_online_key(model_key)
        if mapped is None:
            continue
        # Strip 'model.' prefix
        if mapped.startswith("model."):
            mapped = mapped[len("model."):]
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        remapped[mapped] = tensor

    # Load into model (non-strict, VLM already loaded)
    model_sd = model.state_dict()
    to_load: dict[str, Tensor] = {}
    skipped = []
    for key, tensor in remapped.items():
        if key not in model_sd:
            skipped.append(key)
            continue
        if tensor.shape != model_sd[key].shape:
            skipped.append(f"{key} (shape mismatch)")
            continue
        to_load[key] = tensor

    model.load_state_dict(to_load, strict=False)
    logger.info("  Loaded %d EMA params into model (skipped %d)", len(to_load), len(skipped))
    if skipped[:5]:
        for s in skipped[:5]:
            logger.debug("    skipped: %s", s)


def load_online_pt_checkpoint(
    model: WorldModel,
    pt_path: Path,
    vlm_path: str,
    *,
    dtype: torch.dtype | None = None,
) -> None:
    """Load a monolithic .pt checkpoint exported from wm-training.

    Format: {'model': state_dict, 'global_step': int, ...}
    Keys are like 'state_diffusion_branch.X' (no 'model.' prefix).

    Strategy: load VLM from pretrained, load diffusion+CA from .pt state_dict.
    """
    from wm_raw.checkpoint import _map_online_key

    # 1. Load VLM from pretrained
    logger.info("Loading VLM weights from %s ...", vlm_path)
    from wm_raw.checkpoint import load_vlm_weights
    vlm_report = load_vlm_weights(model, vlm_path)
    logger.info("VLM: %s", vlm_report.format())

    # 2. Load .pt state dict
    logger.info("Loading .pt checkpoint: %s", pt_path)
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        step = ckpt.get("global_step", "?")
        logger.info("  global_step=%s, keys=%d", step, len(state_dict))
    else:
        state_dict = ckpt
        logger.info("  Raw state_dict, keys=%d", len(state_dict))

    # 3. Remap keys (add 'model.' prefix for _map_online_key compatibility)
    model_sd = model.state_dict()
    remapped: dict[str, Tensor] = {}
    skipped: list[str] = []

    for key, tensor in state_dict.items():
        mapped = _map_online_key(f"model.{key}")
        if mapped is None:
            continue
        stripped = mapped[len("model."):] if mapped.startswith("model.") else mapped
        if stripped not in model_sd:
            skipped.append(stripped)
            continue
        if tensor.shape != model_sd[stripped].shape:
            skipped.append(f"{stripped} (shape mismatch)")
            continue
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        remapped[stripped] = tensor

    model.load_state_dict(remapped, strict=False)
    logger.info("  Loaded %d params (skipped %d)", len(remapped), len(skipped))
    if skipped[:5]:
        for s in skipped[:5]:
            logger.debug("    skipped: %s", s)


# ---------------------------------------------------------------------------
# Flow-matching sampling utilities
# ---------------------------------------------------------------------------


def apply_timestep_shift(t: Tensor, shift: float) -> Tensor:
    """Cosmos-style rectified-flow timestep shift: s*t / (1 + (s-1)*t)."""
    if shift == 1.0:
        return t
    s = t.new_tensor(shift)
    return s * t / (1.0 + (s - 1.0) * t)


def build_schedule(num_steps: int, shift: float, device: torch.device) -> tuple[Tensor, Tensor]:
    """Build raw and shifted timestep schedules for Euler sampling.

    Returns:
        raw_schedule: [num_steps+1] linear from 1→0
        shifted_schedule: [num_steps+1] with shift applied
    """
    raw = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    shifted = apply_timestep_shift(raw, shift)
    return raw, shifted


# ---------------------------------------------------------------------------
# Condition encoding
# ---------------------------------------------------------------------------


def encode_text_condition(
    model: WorldModel,
    processor,
    prompt: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Tensor]:
    """Tokenize a text prompt and run through VLM to get hidden states.

    Returns list of hidden states from all VLM layers (for cross-attention).
    """
    # Tokenize using the processor's tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    encoded = tokenizer(
        [prompt],
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask_1d = encoded["attention_mask"].to(device)  # [B, S]

    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    # Build 3D MRoPE position_ids [3, B, S] — all axes use sequential positions
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)  # [3, B, S]

    # Build causal attention mask [B, 1, S, S]
    # SDPA interprets float mask as additive: 0 = attend, -inf = mask out
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )  # upper triangle = -inf, diagonal and below = 0
    attn_mask = causal.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # [B, 1, S, S]

    # Apply padding mask: set columns of padding tokens to -inf
    # Use where() to avoid 0 * -inf = NaN
    pad_positions = (attention_mask_1d == 0)  # [B, S] — True for padding
    if pad_positions.any():
        # Expand to [B, 1, 1, S] and fill -inf where padding
        pad_mask = pad_positions.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        attn_mask = attn_mask.masked_fill(pad_mask, float("-inf"))

    # Run VLM forward (no images, text-only)
    with torch.inference_mode(), torch.amp.autocast(device.type, dtype=dtype):
        vlm_output = model.forward_vlm(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )

    return vlm_output.hidden_states


# ---------------------------------------------------------------------------
# Velocity prediction
# ---------------------------------------------------------------------------


def predict_velocity(
    model: WorldModel,
    noisy_tokens: Tensor,  # [B, num_tokens, token_dim]
    timesteps: Tensor,  # [B]
    vlm_hidden_states: list[Tensor],
    *,
    patch_h: int,
    patch_w: int,
    cross_attention_mask: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_type: str = "cuda",
) -> Tensor:
    """Single diffusion velocity prediction."""
    with torch.amp.autocast(device_type, dtype=dtype):
        return model.state_diffusion(
            noisy_latent=noisy_tokens,
            timesteps=timesteps,
            patch_h=patch_h,
            patch_w=patch_w,
            cross_attention_stack=model.cross_attention,
            vlm_hidden_states=vlm_hidden_states,
            cross_attention_mask=cross_attention_mask,
        )


# ---------------------------------------------------------------------------
# CFG-guided velocity
# ---------------------------------------------------------------------------


def predict_guided_velocity(
    model: WorldModel,
    noisy_tokens: Tensor,
    timesteps: Tensor,
    cond_hidden_states: list[Tensor],
    uncond_hidden_states: list[Tensor] | None,
    *,
    patch_h: int,
    patch_w: int,
    cfg_scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device_type: str = "cuda",
) -> Tensor:
    """Velocity prediction with optional classifier-free guidance."""
    cond_vel = predict_velocity(
        model, noisy_tokens, timesteps, cond_hidden_states,
        patch_h=patch_h, patch_w=patch_w,
        dtype=dtype, device_type=device_type,
    )

    if uncond_hidden_states is None or cfg_scale == 1.0:
        return cond_vel

    uncond_vel = predict_velocity(
        model, noisy_tokens, timesteps, uncond_hidden_states,
        patch_h=patch_h, patch_w=patch_w,
        dtype=dtype, device_type=device_type,
    )
    return uncond_vel + cfg_scale * (cond_vel - uncond_vel)


# ---------------------------------------------------------------------------
# Euler sampling
# ---------------------------------------------------------------------------


@torch.inference_mode()
def sample_euler(
    model: WorldModel,
    cond_hidden_states: list[Tensor],
    uncond_hidden_states: list[Tensor] | None,
    *,
    num_steps: int = 50,
    timestep_shift: float = 1.0,
    cfg_scale: float = 5.0,
    seed: int = 42,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    config: WorldModelConfig,
    image_height: int = 512,
    image_width: int = 512,
) -> Tensor:
    """Generate latent tokens via Euler sampling of the flow model.

    Args:
        image_height/width: target image resolution (must match training buckets).
            Default 512x512 matches the primary training bucket.

    Returns:
        clean_tokens: [1, num_tokens, token_dim] denoised latent tokens
    """
    latent_cfg = config.latent
    diff_cfg = config.diffusion

    # Compute latent and patch dimensions from target image size
    patch_h, patch_w = latent_cfg.patch_shape_for_image(image_height, image_width)
    num_tokens = patch_h * patch_w
    token_dim = latent_cfg.token_dim

    logger.info("  Image %dx%d → latent %dx%d → patch %dx%d → %d tokens (dim=%d)",
                image_height, image_width,
                image_height // latent_cfg.vae_downsample_factor,
                image_width // latent_cfg.vae_downsample_factor,
                patch_h, patch_w, num_tokens, token_dim)

    # Generate noise in float32 for numerical stability
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, num_tokens, token_dim,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )

    raw_schedule, shifted_schedule = build_schedule(num_steps, timestep_shift, device)

    device_type = device.type
    sample = noise
    for i, (raw_t, shifted_t, next_shifted_t) in enumerate(
        zip(raw_schedule[:-1], shifted_schedule[:-1], shifted_schedule[1:])
    ):
        timestep = torch.full((1,), float(raw_t), device=device, dtype=torch.float32)

        velocity = predict_guided_velocity(
            model,
            sample,
            timestep,
            cond_hidden_states,
            uncond_hidden_states,
            patch_h=patch_h,
            patch_w=patch_w,
            cfg_scale=cfg_scale,
            dtype=dtype,
            device_type=device_type,
        )

        # Euler step
        dt = (shifted_t - next_shifted_t).to(sample.dtype)
        sample = sample - velocity.to(sample.dtype) * dt

        if not torch.isfinite(sample).all():
            raise RuntimeError(f"NaN/Inf detected at step {i}")

    return sample


# ---------------------------------------------------------------------------
# VAE decoding
# ---------------------------------------------------------------------------


def decode_latent_to_image(
    latent_tokens: Tensor,  # [1, num_tokens, token_dim]
    vae,
    config: WorldModelConfig,
    image_height: int = 512,
    image_width: int = 512,
) -> Image.Image:
    """Unpatchify latent tokens and decode via VAE to PIL image."""
    latent_cfg = config.latent

    latent_h = image_height // latent_cfg.vae_downsample_factor
    latent_w = image_width // latent_cfg.vae_downsample_factor

    # Unpatchify: [1, num_patches, patch_dim] → [1, H*W, C]
    flat_latent = unpatchify_latent(
        latent_tokens,
        height=latent_h,
        width=latent_w,
        channels=latent_cfg.latent_channels,
        patch_size=latent_cfg.patch_size,
    )  # [1, H*W, C]

    # Reshape to spatial: [1, C, H, W]
    b = flat_latent.shape[0]
    spatial = flat_latent.reshape(
        b,
        latent_h,
        latent_w,
        latent_cfg.latent_channels,
    ).permute(0, 3, 1, 2)  # [1, C, latent_h, latent_w]

    # VAE decode
    spatial = spatial.to(dtype=next(vae.parameters()).dtype)
    with torch.inference_mode():
        if hasattr(vae, "decode"):
            decoded = vae.decode(spatial)
            if hasattr(decoded, "sample"):
                decoded = decoded.sample
        else:
            decoded = vae(spatial)

    # Convert to PIL: [-1, 1] → [0, 255]
    decoded = decoded.clamp(-1, 1).float()
    decoded = ((decoded + 1) * 127.5).round().to(torch.uint8)
    decoded = decoded[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    return Image.fromarray(decoded)


# ---------------------------------------------------------------------------
# Evaluation prompts (same as wm-training batch_generate_image.py)
# ---------------------------------------------------------------------------

EVAL_PROMPTS: list[dict[str, str]] = [
    {"level": "1", "category": "landscape", "name": "simple_natural_landscape",
     "prompt": "A peaceful lake surrounded by green mountains, clear blue sky, soft sunlight, realistic photography style."},
    {"level": "2", "category": "landscape", "name": "complex_natural_environment",
     "prompt": "A vast snowy mountain valley at sunrise, a crystal clear river flowing through the valley, pine forests covering the slopes, golden sunlight reflecting on the snow, ultra realistic landscape photography."},
    {"level": "3", "category": "landscape", "name": "cinematic_landscape",
     "prompt": "A cinematic view of an ancient castle on a cliff above the ocean, surrounded by mist and dark clouds, dramatic lighting, waves crashing against the rocks, fantasy movie scene, highly detailed."},
    {"level": "4", "category": "landscape", "name": "city_architecture_crowd",
     "prompt": "A futuristic Tokyo street at night, neon signs glowing everywhere, rain-soaked streets reflecting colorful lights, pedestrians walking with umbrellas, cyberpunk atmosphere, cinematic photography, ultra detailed."},
    {"level": "5", "category": "portrait", "name": "basic_portrait",
     "prompt": "Portrait of a young woman with long brown hair, wearing a white dress, natural expression, soft studio lighting, realistic photography, high detail."},
    {"level": "6", "category": "portrait", "name": "portrait_environment_clothing",
     "prompt": "A detailed portrait of a young Asian woman wearing a traditional red silk dress, standing in an ancient Chinese garden, delicate embroidery patterns on the fabric, natural makeup, soft afternoon sunlight, shallow depth of field, professional photography."},
    {"level": "7", "category": "portrait", "name": "complex_character_scene",
     "prompt": "A cinematic portrait of an elderly Japanese craftsman working in a traditional wooden workshop, wearing a worn blue apron, detailed wrinkles on his face, wooden tools and handmade objects around him, warm sunlight coming through the window, emotional storytelling photography, ultra realistic."},
    {"level": "8", "category": "portrait", "name": "multi_person_group",
     "prompt": "A group portrait of five people from different generations standing together in a cozy living room, each person has unique facial features and clothing styles, grandmother, parents and children smiling naturally, warm indoor lighting, realistic photography, highly detailed faces, accurate human anatomy."},
    {"level": "9", "category": "portrait", "name": "extreme_composite_test",
     "prompt": "A cinematic photo of a young female astronaut exploring an alien planet, wearing a detailed futuristic spacesuit with realistic fabric textures, holding a holographic device, a massive alien city in the background, strange plants and creatures around her, dramatic sunset lighting, reflections on the helmet glass, Hollywood sci-fi movie style, ultra realistic, 8K detail."},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate images from wm-raw checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt or .dcp dir)")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to VAE safetensors")
    parser.add_argument("--vlm-path", type=str, required=True, help="Path to Qwen3-VL processor")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt (single-image mode)")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--condition-prefix", type=str, default="Caption: ",
                        help="Prefix prepended to prompt (must match training format)")
    parser.add_argument("--condition-suffix", type=str, default=" <|wm_predict_image|>",
                        help="Suffix appended to prompt (must match training format)")
    parser.add_argument("--num-steps", type=int, default=50, help="Euler sampling steps")
    parser.add_argument("--timestep-shift", type=float, default=1.0,
                        help="Flow timestep shift (must match training: 1.0)")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale (1.0 = no CFG)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="generated.png",
                        help="Output path (single) or output directory (batch)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images per prompt")
    parser.add_argument("--image-height", type=int, default=512,
                        help="Target image height (must match training bucket)")
    parser.add_argument("--image-width", type=int, default=512,
                        help="Target image width (must match training bucket)")
    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Run batch eval with EVAL_PROMPTS (ignores --prompt)")
    parser.add_argument("--levels", type=str, default="",
                        help="Comma-separated levels to run in batch mode (e.g. '1,3,9')")
    parser.add_argument("--seed-stride", type=int, default=1,
                        help="Seed increment between prompts in batch mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16

    t_start = time.time()

    # 1. Load processor/tokenizer
    logger.info("Loading processor from %s", args.vlm_path)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    logger.info("  Processor loaded in %.1fs", time.time() - t_start)

    # 2. Build model
    t_step = time.time()
    logger.info("Building model...")
    config = WorldModelConfig()
    model = WorldModel(config)
    logger.info("  Model built in %.1fs", time.time() - t_step)

    # 3. Load checkpoint — auto-detect format from path
    t_step = time.time()
    ckpt_path = Path(args.checkpoint)
    logger.info("Loading checkpoint: %s", ckpt_path)
    if ckpt_path.is_dir():
        # DCP directory — check if it's our own format or online format.
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        reader = FileSystemReader(str(ckpt_path))
        metadata = reader.read_metadata()
        ckpt_keys = set(metadata.state_dict_metadata.keys())
        is_online = any(k.startswith("model.vlm_branch.") for k in ckpt_keys)
        if not is_online:
            load_own_dcp_checkpoint(model, ckpt_path)
        else:
            load_online_dcp_for_inference(model, ckpt_path, args.vlm_path, dtype=dtype)
    elif ckpt_path.suffix == ".pt":
        # Monolithic .pt checkpoint from wm-training
        load_online_pt_checkpoint(model, ckpt_path, args.vlm_path, dtype=dtype)
    else:
        # Legacy: direct load_checkpoint (safetensors or raw state_dict)
        load_checkpoint(ckpt_path, model)
    logger.info("  Checkpoint loaded in %.1fs", time.time() - t_step)

    t_step = time.time()
    model = model.to(device=device, dtype=dtype)
    model.eval()
    logger.info("  Model to device in %.1fs", time.time() - t_step)

    # 4. Load VAE
    t_step = time.time()
    logger.info("Loading VAE: %s", args.vae_path)
    vae = load_vae(args.vae_path, device=device, dtype=dtype)
    logger.info("  VAE loaded in %.1fs", time.time() - t_step)

    # 5. Build prompt list (single or batch mode)
    if args.batch:
        prompts_to_run = EVAL_PROMPTS
        if args.levels:
            selected = {l.strip() for l in args.levels.split(",")}
            prompts_to_run = [p for p in prompts_to_run if p["level"] in selected]
        if not prompts_to_run:
            raise SystemExit(f"No prompts matched --levels={args.levels}")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Batch mode: %d prompts, output_dir=%s", len(prompts_to_run), output_dir)
    else:
        if not args.prompt:
            raise SystemExit("--prompt is required (or use --batch for eval prompts)")
        prompts_to_run = [{"level": "0", "category": "custom", "name": "custom", "prompt": args.prompt}]

    # 6. Encode negative condition (shared across all prompts for CFG)
    uncond_hidden = None
    if args.cfg_scale != 1.0:
        t_step = time.time()
        if args.negative_prompt:
            neg_text = f"{args.condition_prefix}{args.negative_prompt}{args.condition_suffix}"
        else:
            neg_text = args.condition_suffix.strip()
        logger.info("Encoding negative condition: %r", neg_text)
        uncond_hidden = encode_text_condition(model, processor, neg_text, device=device, dtype=dtype)
        logger.info("  Negative condition encoded in %.1fs", time.time() - t_step)

    # 7. Generate
    logger.info("Total setup time: %.1fs", time.time() - t_start)
    generated = 0
    failed = 0

    for idx, entry in enumerate(prompts_to_run):
        prompt = entry["prompt"]
        level = entry["level"]
        name = entry["name"]
        seed = args.seed + idx * args.seed_stride

        # Encode condition
        t_step = time.time()
        condition_text = f"{args.condition_prefix}{prompt}{args.condition_suffix}"
        logger.info("[%d/%d] L%s %s (seed=%d)", idx + 1, len(prompts_to_run), level, name, seed)
        cond_hidden = encode_text_condition(model, processor, condition_text, device=device, dtype=dtype)

        for img_idx in range(args.num_images):
            cur_seed = seed + img_idx
            try:
                t0 = time.time()
                latent_tokens = sample_euler(
                    model,
                    cond_hidden,
                    uncond_hidden,
                    num_steps=args.num_steps,
                    timestep_shift=args.timestep_shift,
                    cfg_scale=args.cfg_scale,
                    seed=cur_seed,
                    device=device,
                    dtype=dtype,
                    config=config,
                    image_height=args.image_height,
                    image_width=args.image_width,
                )
                t_sample = time.time() - t0

                image = decode_latent_to_image(
                    latent_tokens, vae, config,
                    image_height=args.image_height,
                    image_width=args.image_width,
                )

                # Determine save path
                if args.batch:
                    save_path = output_dir / f"L{level}_{name}_seed{cur_seed}.png"
                elif args.num_images > 1:
                    save_path = Path(args.output).with_stem(f"{Path(args.output).stem}_{img_idx:03d}")
                else:
                    save_path = Path(args.output)

                save_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(save_path)
                logger.info("  Saved: %s (%.1fs)", save_path, t_sample)
                generated += 1

            except RuntimeError as e:
                logger.error("  FAILED: %s", e)
                failed += 1

    logger.info("Done. Generated %d images, %d failed.", generated, failed)


if __name__ == "__main__":
    main()
