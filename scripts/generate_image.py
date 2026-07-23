#!/usr/bin/env python3
"""Text-to-image generation using a wm-raw checkpoint.

Loads a trained wm-raw .pt checkpoint and generates images via Euler sampling
of the flow-matching diffusion model with optional classifier-free guidance.

Usage:
    python scripts/generate_image.py \
        --checkpoint outputs/gpic_image_diffusion/checkpoints/checkpoint_step_10000.pt \
        --vae-path /path/to/ae.safetensors \
        --vlm-path /path/to/Qwen3-VL-4B-Instruct \
        --prompt "a photo of a cat sitting on a windowsill" \
        --output generated.png
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
    # Causal: lower-triangular, masked where padding
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=dtype))
    # Apply padding mask: zero out columns for padding tokens
    pad_mask = attention_mask_1d.unsqueeze(1).unsqueeze(2).to(dtype)  # [B, 1, 1, S]
    attn_mask = causal.unsqueeze(0) * pad_mask  # [B, 1, S, S]

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
    cross_attention_mask: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_type: str = "cuda",
) -> Tensor:
    """Single diffusion velocity prediction."""
    with torch.amp.autocast(device_type, dtype=dtype):
        return model.state_diffusion(
            noisy_latent=noisy_tokens,
            timesteps=timesteps,
            cross_attention_contexts=vlm_hidden_states,
            cross_attention_fn=model.cross_attention.condition_layer,
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
    cfg_scale: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device_type: str = "cuda",
) -> Tensor:
    """Velocity prediction with optional classifier-free guidance."""
    cond_vel = predict_velocity(
        model, noisy_tokens, timesteps, cond_hidden_states,
        dtype=dtype, device_type=device_type,
    )

    if uncond_hidden_states is None or cfg_scale == 1.0:
        return cond_vel

    uncond_vel = predict_velocity(
        model, noisy_tokens, timesteps, uncond_hidden_states,
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
    timestep_shift: float = 2.0,
    cfg_scale: float = 5.0,
    seed: int = 42,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    config: WorldModelConfig,
) -> Tensor:
    """Generate latent tokens via Euler sampling of the flow model.

    Returns:
        clean_tokens: [1, num_tokens, token_dim] denoised latent tokens
    """
    latent_cfg = config.latent
    diff_cfg = config.diffusion

    num_tokens = (latent_cfg.latent_height // latent_cfg.patch_size) * (
        latent_cfg.latent_width // latent_cfg.patch_size
    )
    token_dim = diff_cfg.target_dim

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
) -> Image.Image:
    """Unpatchify latent tokens and decode via VAE to PIL image."""
    latent_cfg = config.latent

    # Unpatchify: [1, 256, 64] → [1, 1024, 16]
    flat_latent = unpatchify_latent(
        latent_tokens,
        height=latent_cfg.latent_height,
        width=latent_cfg.latent_width,
        channels=latent_cfg.latent_channels,
        patch_size=latent_cfg.patch_size,
    )  # [1, H*W, C] = [1, 1024, 16]

    # Reshape to spatial: [1, C, H, W]
    b = flat_latent.shape[0]
    spatial = flat_latent.reshape(
        b,
        latent_cfg.latent_height,
        latent_cfg.latent_width,
        latent_cfg.latent_channels,
    ).permute(0, 3, 1, 2)  # [1, 16, 32, 32]

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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate images from wm-raw checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to VAE safetensors")
    parser.add_argument("--vlm-path", type=str, required=True, help="Path to Qwen3-VL processor")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt for CFG")
    parser.add_argument("--num-steps", type=int, default=50, help="Euler sampling steps")
    parser.add_argument("--timestep-shift", type=float, default=2.0, help="Flow timestep shift")
    parser.add_argument("--cfg-scale", type=float, default=5.0, help="CFG scale (1.0 = no CFG)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # 1. Load processor/tokenizer
    logger.info("Loading processor from %s", args.vlm_path)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)

    # 2. Build model
    logger.info("Building model...")
    config = WorldModelConfig()
    model = WorldModel(config)

    # 3. Load checkpoint (.pt or .dcp directory)
    ckpt_path = Path(args.checkpoint)
    logger.info("Loading checkpoint: %s", ckpt_path)
    if ckpt_path.is_dir():
        # DCP directory — check if it's our own format (has 'model.' prefix)
        # or online format (has different key mapping)
        from torch.distributed.checkpoint.filesystem import FileSystemReader
        reader = FileSystemReader(str(ckpt_path))
        metadata = reader.read_metadata()
        first_key = next(iter(metadata.state_dict_metadata.keys()), "")
        if first_key.startswith("model."):
            # Our own FSDP DCP checkpoint
            load_own_dcp_checkpoint(model, ckpt_path)
        else:
            # Online format DCP — use key remapping loader
            from wm_raw.checkpoint import load_online_dcp_weights
            report = load_online_dcp_weights(model, ckpt_path)
            logger.info("Online DCP checkpoint: %s", report.format())
    else:
        load_checkpoint(ckpt_path, model)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # 4. Load VAE
    logger.info("Loading VAE: %s", args.vae_path)
    vae = load_vae(args.vae_path, device=device, dtype=dtype)

    # 5. Encode condition
    logger.info("Encoding prompt: %r", args.prompt)
    cond_hidden = encode_text_condition(model, processor, args.prompt, device=device, dtype=dtype)

    # 6. Encode negative condition (for CFG)
    uncond_hidden = None
    if args.cfg_scale != 1.0:
        # Use a single pad/space token as unconditional embedding if prompt is empty
        neg_prompt = args.negative_prompt if args.negative_prompt else " "
        logger.info("Encoding negative prompt: %r", neg_prompt)
        uncond_hidden = encode_text_condition(
            model, processor, neg_prompt, device=device, dtype=dtype
        )

    # 7. Generate
    output_path = Path(args.output)
    for i in range(args.num_images):
        seed = args.seed + i
        logger.info("Generating image %d/%d (seed=%d, steps=%d, shift=%.1f, cfg=%.1f)",
                    i + 1, args.num_images, seed, args.num_steps,
                    args.timestep_shift, args.cfg_scale)

        t0 = time.time()
        latent_tokens = sample_euler(
            model,
            cond_hidden,
            uncond_hidden,
            num_steps=args.num_steps,
            timestep_shift=args.timestep_shift,
            cfg_scale=args.cfg_scale,
            seed=seed,
            device=device,
            dtype=dtype,
            config=config,
        )
        t_sample = time.time() - t0
        logger.info("Sampling took %.1fs", t_sample)

        # 8. Decode to image
        image = decode_latent_to_image(latent_tokens, vae, config)

        # Save
        if args.num_images > 1:
            save_path = output_path.with_stem(f"{output_path.stem}_{i:03d}")
        else:
            save_path = output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(save_path)
        logger.info("Saved: %s", save_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
