"""Checkpoint utilities: load HF pretrained weights into wm-raw WorldModel.

Supports loading from:
- Qwen3-VL-4B-Instruct (for VLM branch)
- Qwen3-VL-2B-Instruct (for diffusion branch backbone)

Key mapping strategy:
- VLM branch weights map directly from HF Qwen3-VL (model.layers → vlm.decoder.layers)
- Diffusion branch backbone maps from a separate Qwen3-VL checkpoint
- Cross-attention and adapter weights are initialized fresh (no pretrained source)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointReport:
    """Summary of a weight loading operation."""

    matched: int = 0
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    shape_mismatch: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and not self.shape_mismatch

    def format(self, max_items: int = 30) -> str:
        lines = [
            f"Checkpoint report: matched={self.matched}, "
            f"missing={len(self.missing)}, unexpected={len(self.unexpected)}, "
            f"shape_mismatch={len(self.shape_mismatch)}",
        ]
        for label, keys in [
            ("missing", self.missing),
            ("unexpected", self.unexpected),
            ("shape_mismatch", self.shape_mismatch),
        ]:
            if keys:
                lines.append(f"  {label}:")
                for k in keys[:max_items]:
                    lines.append(f"    - {k}")
                if len(keys) > max_items:
                    lines.append(f"    ... and {len(keys) - max_items} more")
        return "\n".join(lines)


def _strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Remove a prefix from all keys in a state dict."""
    result = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            result[key[len(prefix):]] = value
        else:
            result[key] = value
    return result


def _find_hf_text_model_prefix(state_dict: dict[str, Tensor]) -> str:
    """Auto-detect the prefix path to the text model in an HF checkpoint.

    Common patterns:
        model.layers.0.self_attn... (Qwen3-VL ImageTextToText)
        language_model.model.layers.0... (some wrappers)
    """
    for key in state_dict:
        if "model.layers.0.self_attn.q_proj.weight" in key:
            return key.split("model.layers.0.self_attn.q_proj.weight")[0]
        if "layers.0.self_attn.q_proj.weight" in key:
            return key.split("layers.0.self_attn.q_proj.weight")[0]
    return ""


# ---------------------------------------------------------------------------
# VLM branch key mapping: HF Qwen3-VL → wm-raw VLMBranch
# ---------------------------------------------------------------------------

_VLM_TEXT_PREFIX_MAP = {
    # HF key prefix → wm-raw key prefix
    "model.embed_tokens.": "vlm.embed_tokens.",
    "model.layers.": "vlm.decoder.layers.",
    "model.norm.": "vlm.norm.",
    "lm_head.": "vlm.lm_head.",
}

_VLM_VISION_PREFIX_MAP = {
    "visual.": "vlm.vision_encoder.",
}


def _map_vlm_key(hf_key: str) -> str | None:
    """Map a single HF Qwen3-VL key to the wm-raw VLM branch key."""
    for hf_prefix, raw_prefix in _VLM_TEXT_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            return raw_prefix + hf_key[len(hf_prefix):]
    for hf_prefix, raw_prefix in _VLM_VISION_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            return raw_prefix + hf_key[len(hf_prefix):]
    return None


# ---------------------------------------------------------------------------
# Diffusion branch key mapping: HF Qwen3-VL text backbone → StateDiffusionBranch
# ---------------------------------------------------------------------------

_DIFFUSION_TEXT_PREFIX_MAP = {
    "model.layers.": "state_diffusion.layers.",
    "model.norm.": "state_diffusion.final_norm.",
}


def _map_diffusion_key(hf_key: str) -> str | None:
    """Map an HF Qwen3-VL text key to diffusion branch decoder layer key.

    Only maps decoder layers and final norm. Embedding/lm_head are skipped
    since diffusion uses its own input_proj/output_head.
    """
    for hf_prefix, raw_prefix in _DIFFUSION_TEXT_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            return raw_prefix + hf_key[len(hf_prefix):]
    return None


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------


def load_hf_safetensors(model_path: str | Path) -> dict[str, Tensor]:
    """Load all safetensors shards from an HF model directory."""
    from safetensors.torch import load_file

    model_path = Path(model_path)
    if model_path.is_file():
        return load_file(str(model_path))

    # Load all .safetensors files in the directory
    shard_files = sorted(model_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_path}")

    merged: dict[str, Tensor] = {}
    for shard in shard_files:
        merged.update(load_file(str(shard)))
    return merged


def load_vlm_weights(
    model: Any,
    vlm_path: str | Path,
    *,
    strict: bool = False,
    dtype: torch.dtype | None = None,
) -> CheckpointReport:
    """Load VLM branch weights from an HF Qwen3-VL checkpoint.

    Args:
        model: WorldModel instance
        vlm_path: path to HF Qwen3-VL-4B-Instruct directory
        strict: if True, raise on missing/unexpected keys
        dtype: cast loaded tensors to this dtype

    Returns:
        CheckpointReport with loading stats
    """
    hf_sd = load_hf_safetensors(vlm_path)
    prefix = _find_hf_text_model_prefix(hf_sd)
    if prefix:
        hf_sd = _strip_prefix(hf_sd, prefix)

    # Map HF keys → wm-raw keys
    mapped: dict[str, Tensor] = {}
    unmapped_keys: list[str] = []
    for hf_key, tensor in hf_sd.items():
        raw_key = _map_vlm_key(hf_key)
        if raw_key is not None:
            mapped[raw_key] = tensor if dtype is None else tensor.to(dtype)
        else:
            unmapped_keys.append(hf_key)

    # Load into model
    model_sd = model.state_dict()
    matched_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatch: list[str] = []

    for key in model_sd:
        if not key.startswith("vlm."):
            continue
        if key in mapped:
            if model_sd[key].shape == mapped[key].shape:
                matched_keys.append(key)
            else:
                shape_mismatch.append(
                    f"{key}: model={tuple(model_sd[key].shape)} vs ckpt={tuple(mapped[key].shape)}"
                )
        else:
            missing_keys.append(key)

    # Actually load the matched weights
    load_dict = {k: mapped[k] for k in matched_keys}
    model.load_state_dict(load_dict, strict=False)

    report = CheckpointReport(
        matched=len(matched_keys),
        missing=tuple(missing_keys),
        unexpected=tuple(unmapped_keys[:100]),
        shape_mismatch=tuple(shape_mismatch),
    )

    if strict and not report.ok:
        raise RuntimeError(f"Strict checkpoint loading failed:\n{report.format()}")

    logger.info("VLM weights loaded: %s", report.format())
    return report


def load_diffusion_weights(
    model: Any,
    diffusion_path: str | Path,
    *,
    strict: bool = False,
    dtype: torch.dtype | None = None,
) -> CheckpointReport:
    """Load diffusion branch backbone from an HF Qwen3-VL checkpoint.

    Only loads the decoder layer weights and final norm. The input_proj,
    output_head, timestep embedder, AdaLN, and position embeddings are
    initialized fresh (not present in the HF checkpoint).

    Args:
        model: WorldModel instance
        diffusion_path: path to HF Qwen3-VL-2B-Instruct directory
        strict: if True, raise on missing keys in the backbone
        dtype: cast loaded tensors to this dtype
    """
    hf_sd = load_hf_safetensors(diffusion_path)
    prefix = _find_hf_text_model_prefix(hf_sd)
    if prefix:
        hf_sd = _strip_prefix(hf_sd, prefix)

    # Map HF keys → wm-raw diffusion keys
    mapped: dict[str, Tensor] = {}
    unmapped_keys: list[str] = []
    for hf_key, tensor in hf_sd.items():
        raw_key = _map_diffusion_key(hf_key)
        if raw_key is not None:
            mapped[raw_key] = tensor if dtype is None else tensor.to(dtype)
        else:
            unmapped_keys.append(hf_key)

    # Load into model
    model_sd = model.state_dict()
    matched_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatch: list[str] = []

    # Only check state_diffusion.layers.* and state_diffusion.final_norm.*
    backbone_prefixes = ("state_diffusion.layers.", "state_diffusion.final_norm.")
    for key in model_sd:
        if not any(key.startswith(p) for p in backbone_prefixes):
            continue
        if key in mapped:
            if model_sd[key].shape == mapped[key].shape:
                matched_keys.append(key)
            else:
                shape_mismatch.append(
                    f"{key}: model={tuple(model_sd[key].shape)} vs ckpt={tuple(mapped[key].shape)}"
                )
        else:
            missing_keys.append(key)

    load_dict = {k: mapped[k] for k in matched_keys}
    model.load_state_dict(load_dict, strict=False)

    report = CheckpointReport(
        matched=len(matched_keys),
        missing=tuple(missing_keys),
        unexpected=tuple(unmapped_keys[:50]),
        shape_mismatch=tuple(shape_mismatch),
    )

    if strict and not report.ok:
        raise RuntimeError(f"Strict diffusion loading failed:\n{report.format()}")

    logger.info("Diffusion weights loaded: %s", report.format())
    return report


def load_pretrained_weights(
    model: Any,
    vlm_path: str | Path,
    diffusion_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    strict: bool = False,
) -> tuple[CheckpointReport, CheckpointReport]:
    """Load both VLM and diffusion branch weights from HF checkpoints.

    Args:
        model: WorldModel instance
        vlm_path: path to HF Qwen3-VL-4B-Instruct
        diffusion_path: path to HF Qwen3-VL-2B-Instruct
        dtype: target dtype for loaded weights
        strict: raise on any loading issues

    Returns:
        (vlm_report, diffusion_report)
    """
    vlm_report = load_vlm_weights(model, vlm_path, strict=strict, dtype=dtype)
    diff_report = load_diffusion_weights(model, diffusion_path, strict=strict, dtype=dtype)
    return vlm_report, diff_report


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    output_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save a training checkpoint.

    Saves model state_dict, optimizer, scheduler, and step counter.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        checkpoint["config"] = config

    ckpt_file = output_path / f"checkpoint_step_{global_step}.pt"
    torch.save(checkpoint, ckpt_file)
    logger.info("Saved checkpoint: %s", ckpt_file)
    return ckpt_file


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    *,
    map_location: str = "cpu",
) -> int:
    """Load a training checkpoint. Returns global_step."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    global_step = checkpoint.get("global_step", 0)
    logger.info("Loaded checkpoint from step %d: %s", global_step, path)
    return global_step


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------


def load_vae(
    vae_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Any:
    """Load a VAE autoencoder model from a safetensors checkpoint.

    Supports the BAGEL ae.safetensors format. Returns an nn.Module with
    an encode() method.

    For now, delegates to the actual VAE class if available in the environment.
    Falls back to loading raw state_dict if the model class isn't importable.
    """
    from safetensors.torch import load_file

    vae_path = Path(vae_path)
    state_dict = load_file(str(vae_path)) if vae_path.is_file() else load_hf_safetensors(vae_path)

    # Try importing the BAGEL autoencoder
    try:
        from wm_training.models.bagel.autoencoder import load_ae

        ae = load_ae(str(vae_path))
        ae = ae.to(device=device, dtype=dtype)
        ae.eval()
        return ae
    except ImportError:
        pass

    # Fallback: try diffusers AutoencoderKL
    try:
        from diffusers import AutoencoderKL

        ae = AutoencoderKL.from_pretrained(
            str(vae_path.parent) if vae_path.is_file() else str(vae_path),
            torch_dtype=dtype,
        )
        ae = ae.to(device=device)
        ae.eval()
        return ae
    except (ImportError, Exception) as e:
        logger.warning("Could not load VAE with diffusers: %s", e)

    raise RuntimeError(
        f"Failed to load VAE from {vae_path}. "
        "Install wm_training or diffusers to load the autoencoder."
    )

