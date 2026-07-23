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
        model.layers.0.self_attn... (Qwen3-VL standalone)
        model.language_model.layers.0... (Qwen3-VL ConditionalGeneration)
        language_model.model.layers.0... (some wrappers)
    """
    for key in state_dict:
        # Qwen3-VL ConditionalGeneration: model.language_model.layers.0.self_attn.q_proj.weight
        if "language_model.layers.0.self_attn.q_proj.weight" in key:
            return key.split("language_model.layers.0.self_attn.q_proj.weight")[0] + "language_model."
        if "model.layers.0.self_attn.q_proj.weight" in key:
            return key.split("model.layers.0.self_attn.q_proj.weight")[0]
        if "layers.0.self_attn.q_proj.weight" in key:
            return key.split("layers.0.self_attn.q_proj.weight")[0]
    return ""


def _find_hf_vision_prefix(state_dict: dict[str, Tensor]) -> str:
    """Auto-detect the prefix for vision encoder keys.

    Common patterns:
        visual.blocks.0... (standalone)
        model.visual.blocks.0... (ConditionalGeneration wrapper)
    """
    for key in state_dict:
        if ".visual.blocks.0." in key or key.startswith("visual.blocks.0."):
            idx = key.find("visual.")
            return key[:idx]
    return ""


# ---------------------------------------------------------------------------
# VLM branch key mapping: HF Qwen3-VL → wm-raw VLMBranch
# ---------------------------------------------------------------------------

_VLM_TEXT_PREFIX_MAP = {
    # HF key prefix → wm-raw key prefix
    # After stripping prefix, keys look like: layers.0.self_attn.q_proj.weight
    # or: embed_tokens.weight, norm.weight
    "embed_tokens.": "vlm.embed_tokens.",
    "layers.": "vlm.layers.",
    "norm.": "vlm.norm.",
    "lm_head.": "vlm.lm_head.",
    # Legacy: some checkpoints don't get prefix stripped
    "model.embed_tokens.": "vlm.embed_tokens.",
    "model.layers.": "vlm.layers.",
    "model.norm.": "vlm.norm.",
}

_VLM_VISION_PREFIX_MAP = {
    "visual.": "vlm.vision_encoder.",
}

# Vision encoder key renames (within a block)
# HF → wm-raw (after prefix mapping to vlm.vision_encoder.blocks.N.)
_VISION_KEY_RENAMES = {
    "attn.proj.": "attn.o_proj.",
    "mlp.linear_fc1.": "mlp.fc1.",
    "mlp.linear_fc2.": "mlp.fc2.",
}

# Merger key renames
_VISION_MERGER_RENAMES = {
    "merger.linear_fc1.": "merger.fc1.",
    "merger.linear_fc2.": "merger.fc2.",
}


def _map_vlm_key(hf_key: str) -> str | None:
    """Map a single HF Qwen3-VL key to the wm-raw VLM branch key.

    Returns None if the key should not be loaded (e.g., fused QKV that
    needs special handling via _split_vlm_fused_qkv).
    """
    # Text keys
    for hf_prefix, raw_prefix in _VLM_TEXT_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            return raw_prefix + hf_key[len(hf_prefix):]

    # Vision keys
    for hf_prefix, raw_prefix in _VLM_VISION_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            suffix = hf_key[len(hf_prefix):]
            # Skip fused QKV — handled separately
            if "attn.qkv." in suffix:
                return None
            # Apply renames
            for old, new in _VISION_KEY_RENAMES.items():
                if old in suffix:
                    suffix = suffix.replace(old, new)
                    break
            # Merger renames
            for old, new in _VISION_MERGER_RENAMES.items():
                if old in suffix:
                    suffix = suffix.replace(old, new)
                    break
            return raw_prefix + suffix
    return None


def _split_vision_fused_qkv(
    hf_sd: dict[str, Tensor],
    vision_prefix: str = "visual.",
) -> dict[str, Tensor]:
    """Split fused QKV weights from HF vision encoder into separate Q, K, V.

    HF key: visual.blocks.N.attn.qkv.{weight,bias} → shape [3*D, D] or [3*D]
    wm-raw: vlm.vision_encoder.blocks.N.attn.{q,k,v}_proj.{weight,bias}
    """
    result: dict[str, Tensor] = {}
    target_prefix = "vlm.vision_encoder."

    for key, tensor in hf_sd.items():
        if not key.startswith(vision_prefix):
            continue
        suffix = key[len(vision_prefix):]
        if "attn.qkv." not in suffix:
            continue

        # Determine if weight or bias
        param_type = "weight" if suffix.endswith(".weight") else "bias"
        # Extract block path: blocks.N.attn.qkv.weight
        block_path = suffix.replace(f"attn.qkv.{param_type}", "")

        # Split into 3 equal chunks
        chunks = tensor.chunk(3, dim=0)
        result[f"{target_prefix}{block_path}attn.q_proj.{param_type}"] = chunks[0]
        result[f"{target_prefix}{block_path}attn.k_proj.{param_type}"] = chunks[1]
        result[f"{target_prefix}{block_path}attn.v_proj.{param_type}"] = chunks[2]

    return result


# ---------------------------------------------------------------------------
# Diffusion branch key mapping: HF Qwen3-VL text backbone → StateDiffusionBranch
# ---------------------------------------------------------------------------

_DIFFUSION_TEXT_PREFIX_MAP = {
    # After stripping prefix, keys look like: layers.0.self_attn.q_proj.weight
    "layers.": "state_diffusion.layers.",
    "norm.": "state_diffusion.final_norm.",
    # Legacy: some checkpoints don't get prefix stripped
    "model.layers.": "state_diffusion.layers.",
    "model.norm.": "state_diffusion.final_norm.",
}


def _map_diffusion_key(hf_key: str) -> str | None:
    """Map an HF Qwen3-VL text key to diffusion branch decoder layer key.

    Only maps decoder layers and final norm. Embedding/lm_head are skipped
    since diffusion uses its own input_proj/output_head.

    The wm-raw DiffusionDecoderLayer wraps a DecoderLayer as self.layer, so:
        HF: layers.0.self_attn.q_proj.weight
        → wm-raw: state_diffusion.layers.0.layer.self_attn.q_proj.weight
    """
    import re

    for hf_prefix, raw_prefix in _DIFFUSION_TEXT_PREFIX_MAP.items():
        if hf_key.startswith(hf_prefix):
            suffix = hf_key[len(hf_prefix):]
            if hf_prefix.endswith("layers."):
                # Insert .layer. after the layer index
                # suffix is like "0.self_attn.q_proj.weight"
                match = re.match(r"(\d+)\.(.*)", suffix)
                if match:
                    layer_idx = match.group(1)
                    rest = match.group(2)
                    return f"{raw_prefix}{layer_idx}.layer.{rest}"
            return raw_prefix + suffix
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

    # Separate text and vision keys, strip their respective prefixes
    text_prefix = _find_hf_text_model_prefix(hf_sd)
    vision_prefix = _find_hf_vision_prefix(hf_sd)

    stripped: dict[str, Tensor] = {}
    for key, tensor in hf_sd.items():
        if text_prefix and key.startswith(text_prefix):
            stripped[key[len(text_prefix):]] = tensor
        elif vision_prefix and key.startswith(vision_prefix + "visual."):
            # Strip to "visual.blocks.0..." format
            stripped[key[len(vision_prefix):]] = tensor
        elif key.startswith("visual."):
            stripped[key] = tensor
        elif key.startswith("lm_head."):
            stripped[key] = tensor
        else:
            # Keys that don't match either prefix — keep original
            stripped[key] = tensor
    hf_sd = stripped

    # Map HF keys → wm-raw keys
    mapped: dict[str, Tensor] = {}
    unmapped_keys: list[str] = []
    for hf_key, tensor in hf_sd.items():
        raw_key = _map_vlm_key(hf_key)
        if raw_key is not None:
            mapped[raw_key] = tensor if dtype is None else tensor.to(dtype)
        else:
            unmapped_keys.append(hf_key)

    # Split fused vision QKV into separate Q, K, V
    qkv_splits = _split_vision_fused_qkv(hf_sd, vision_prefix="visual.")
    for key, tensor in qkv_splits.items():
        mapped[key] = tensor if dtype is None else tensor.to(dtype)
    # Remove qkv keys from unmapped since we handled them
    unmapped_keys = [
        k for k in unmapped_keys if "attn.qkv." not in k
    ]

    # Handle tied word embeddings: copy embed_tokens → lm_head if missing
    if "vlm.lm_head.weight" not in mapped and "vlm.embed_tokens.weight" in mapped:
        mapped["vlm.lm_head.weight"] = mapped["vlm.embed_tokens.weight"]

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


# ---------------------------------------------------------------------------
# Online DCP checkpoint loading (wm-training → wm-raw key remapping)
# ---------------------------------------------------------------------------

# Key mapping: online (wm-training) prefix → wm-raw prefix
_ONLINE_PREFIX_MAP = [
    # VLM text (language model)
    ("model.vlm_branch.backbone.model.language_model.", "model.vlm."),
    # VLM lm_head
    ("model.vlm_branch.backbone.lm_head.", "model.vlm.lm_head."),
    # VLM vision encoder
    ("model.vlm_branch.backbone.model.visual.", "model.vlm.vision_encoder."),
    # Cross attention: drop "state." sub-key (action adapters are ignored)
    ("model.cross_attention.adapters.state.", "model.cross_attention.adapters."),
    # Diffusion adaln
    ("model.state_diffusion_branch.adaln_modulations.", "model.state_diffusion.adaln_layers."),
    # Diffusion backbone layers
    ("model.state_diffusion_branch.backbone.layers.", "model.state_diffusion.layers."),
    ("model.state_diffusion_branch.backbone.embed_tokens.", None),  # skip
    # Diffusion input projector
    ("model.state_diffusion_branch.input_projector.", "model.state_diffusion.input_proj."),
    # Diffusion time conditioner/embedder
    ("model.state_diffusion_branch.time_conditioner.", "model.state_diffusion.time_conditioner."),
    ("model.state_diffusion_branch.time_embedder.", "model.state_diffusion.time_embedder."),
    # Diffusion output head
    ("model.state_diffusion_branch.output_head.", "model.state_diffusion.output_head."),
    # Skip online-only components
    ("model.state_diffusion_branch.peer_conditioner.", None),
    ("model.state_diffusion_branch.latent_position_embedding.", None),
    ("model.state_diffusion_branch.adapter.", None),
    ("model.action_diffusion_branch.", None),
    ("model.cross_attention.adapters.action.", None),
]


def _map_online_key(key: str) -> str | None:
    """Map an online (wm-training) DCP key to wm-raw key.

    Returns None for keys that should be skipped (no equivalent in wm-raw).
    Returns the key itself for fused QKV that needs special handling.
    """
    import re

    for online_prefix, raw_prefix in _ONLINE_PREFIX_MAP:
        if not key.startswith(online_prefix):
            continue
        if raw_prefix is None:
            return None  # skip this key

        suffix = key[len(online_prefix):]

        # Diffusion backbone layers need .layer. insertion
        if online_prefix == "model.state_diffusion_branch.backbone.layers.":
            match = re.match(r"(\d+)\.(.*)", suffix)
            if match:
                layer_idx, rest = match.group(1), match.group(2)
                return f"{raw_prefix}{layer_idx}.layer.{rest}"

        # Vision encoder: rename keys and skip fused QKV
        if online_prefix == "model.vlm_branch.backbone.model.visual.":
            if "attn.qkv." in suffix:
                return None  # handled by _split_online_vision_qkv
            for old, new in _VISION_KEY_RENAMES.items():
                if old in suffix:
                    suffix = suffix.replace(old, new)
                    break
            for old, new in _VISION_MERGER_RENAMES.items():
                if old in suffix:
                    suffix = suffix.replace(old, new)
                    break

        return raw_prefix + suffix

    return None  # not matched


def _split_online_vision_qkv(
    state_dict: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Split fused QKV from online vision encoder into separate Q, K, V.

    Online key: model.vlm_branch.backbone.model.visual.blocks.N.attn.qkv.{weight,bias}
    wm-raw:     model.vlm.vision_encoder.blocks.N.attn.{q,k,v}_proj.{weight,bias}
    """
    import re

    result: dict[str, Tensor] = {}
    prefix = "model.vlm_branch.backbone.model.visual."

    for key, tensor in state_dict.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if "attn.qkv." not in suffix:
            continue

        # suffix like: blocks.0.attn.qkv.weight
        match = re.match(r"blocks\.(\d+)\.attn\.qkv\.(weight|bias)", suffix)
        if not match:
            continue

        block_idx = match.group(1)
        param_type = match.group(2)
        # Split [3*D, ...] → 3 × [D, ...]
        chunks = tensor.chunk(3, dim=0)
        for proj_name, chunk in zip(("q_proj", "k_proj", "v_proj"), chunks):
            out_key = f"model.vlm.vision_encoder.blocks.{block_idx}.attn.{proj_name}.{param_type}"
            result[out_key] = chunk

    return result


def load_online_dcp_weights(
    model: Any,
    dcp_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
) -> CheckpointReport:
    """Load model weights from an online (wm-training) DCP checkpoint.

    This handles the key remapping between wm-training's model structure and
    wm-raw's model structure. Only model weights are loaded (no optimizer/scheduler).
    Supports resharding across different world sizes (e.g. 8 GPU → 2 GPU).

    Args:
        model: WorldModel instance (before FSDP wrapping)
        dcp_path: path to the .dcp directory
        dtype: cast loaded tensors to this dtype

    Returns:
        CheckpointReport summarizing the loading
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader

    dcp_path = Path(dcp_path)
    if not (dcp_path / ".metadata").exists():
        raise FileNotFoundError(f"Not a DCP checkpoint: {dcp_path}")

    # Read metadata to get all model keys
    reader = FileSystemReader(str(dcp_path))
    metadata = reader.read_metadata()
    all_keys = list(metadata.state_dict_metadata.keys())
    model_keys = [k for k in all_keys if k.startswith("model.")]

    # Load only model keys as full state dict using DCP
    # Build a placeholder state dict to load into
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # Use dcp.load with the model keys we want
    # We load into a flat dict - need to create empty tensors as targets
    state_dict: dict[str, Tensor] = {}
    for key in model_keys:
        md = metadata.state_dict_metadata[key]
        # Only load TensorStorageMetadata (skip BytesStorageMetadata)
        from torch.distributed.checkpoint.metadata import TensorStorageMetadata
        if isinstance(md, TensorStorageMetadata):
            state_dict[key] = torch.empty(md.size, dtype=md.properties.dtype)

    dcp.load(state_dict, checkpoint_id=str(dcp_path))

    # Remap keys
    remapped: dict[str, Tensor] = {}
    unmapped: list[str] = []

    # First handle fused QKV split
    qkv_split = _split_online_vision_qkv(state_dict)
    remapped.update(qkv_split)

    # Map remaining keys
    for key, tensor in state_dict.items():
        mapped = _map_online_key(key)
        if mapped is None:
            continue  # skip
        if mapped in remapped:
            continue  # already handled by QKV split
        remapped[mapped] = tensor

    # Strip "model." prefix since model.load_state_dict expects unprefixed keys
    final_sd: dict[str, Tensor] = {}
    for key, tensor in remapped.items():
        if key.startswith("model."):
            key = key[len("model."):]
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        final_sd[key] = tensor

    # Load into model
    model_sd = model.state_dict()
    model_keys_expected = set(model_sd.keys())
    shape_mismatch: list[str] = []
    to_load: dict[str, Tensor] = {}

    for key, tensor in final_sd.items():
        if key not in model_keys_expected:
            unmapped.append(key)
            continue
        # Check shape
        if tensor.shape != model_sd[key].shape:
            shape_mismatch.append(f"{key}: online={list(tensor.shape)} vs ours={list(model_sd[key].shape)}")
            continue
        to_load[key] = tensor

    missing_keys = model_keys_expected - set(to_load.keys())
    model.load_state_dict(to_load, strict=False)

    report = CheckpointReport(
        matched=len(to_load),
        missing=tuple(sorted(missing_keys)),
        unexpected=tuple(unmapped[:50]),
        shape_mismatch=tuple(shape_mismatch),
    )
    logger.info("Online DCP weights loaded: %s", report.format())
    return report


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
        from wm_training.models.bagel.autoencoder import load_bagel_autoencoder

        ae, _params = load_bagel_autoencoder(str(vae_path))
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

