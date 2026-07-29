"""Layer-by-layer numerical alignment between wm-raw and wm-training.

Workflow:
    # 1. Dump online model activations (uses wm-training code)
    python scripts/dump_online_activations.py \
        --checkpoint /path/to/step_275000.dcp \
        --output alignment_fixture.pt

    # 2. Replay the same inputs in wm-raw + compare layer-by-layer
    python scripts/align_check.py replay \
        --checkpoint /path/to/step_275000.dcp \
        --reference alignment_fixture.pt \
        --atol 1e-3

    # 3. Standalone smoke test (synthetic data)
    python scripts/align_check.py smoke --checkpoint /path/to/ckpt

Environment:
    - Single GPU, bf16, fixed seed, no FSDP
    - Both scripts must use identical checkpoint + inputs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_deterministic(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TensorComparison:
    """Result of comparing two tensors."""

    name: str
    shape: tuple[int, ...]
    max_abs_err: float
    mean_abs_err: float
    max_rel_err: float
    passed: bool

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return (
            f"  {status} {self.name}: "
            f"shape={list(self.shape)}, "
            f"max_abs={self.max_abs_err:.2e}, "
            f"mean_abs={self.mean_abs_err:.2e}, "
            f"max_rel={self.max_rel_err:.2e}"
        )


def compare_tensors(
    name: str,
    actual: Tensor,
    expected: Tensor,
    atol: float = 1e-3,
) -> TensorComparison:
    """Compare two tensors and return a comparison result."""
    a = actual.float().cpu()
    e = expected.float().cpu()

    if a.shape != e.shape:
        return TensorComparison(
            name=f"{name} [SHAPE MISMATCH: {list(a.shape)} vs {list(e.shape)}]",
            shape=tuple(a.shape),
            max_abs_err=float("inf"),
            mean_abs_err=float("inf"),
            max_rel_err=float("inf"),
            passed=False,
        )

    abs_err = (a - e).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()

    denom = e.abs().clamp_min(1e-8)
    rel_err = (abs_err / denom).max().item()

    return TensorComparison(
        name=name,
        shape=tuple(a.shape),
        max_abs_err=max_abs,
        mean_abs_err=mean_abs,
        max_rel_err=rel_err,
        passed=max_abs < atol,
    )


# ---------------------------------------------------------------------------
# Activation capture for wm-raw
# ---------------------------------------------------------------------------


class WMRawActivationCapture:
    """Register hooks on wm-raw model to capture intermediate activations.

    Captures the same keys as dump_online_activations.py to enable comparison.
    """

    def __init__(self) -> None:
        self.activations: dict[str, Tensor] = {}
        self._hooks: list[Any] = []

    def register(self, model) -> None:
        """Register forward hooks on key wm-raw modules."""
        # --- VLM decoder layers ---
        for i, layer in enumerate(model.vlm.layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_hook(f"vlm.layer.{i}"))
            )

        # --- Diffusion branch ---
        diff = model.state_diffusion

        # Input projection
        self._hooks.append(
            diff.input_proj.register_forward_hook(
                self._make_hook("diffusion.input_proj")
            )
        )

        # Time embedder
        self._hooks.append(
            diff.time_embedder.register_forward_hook(
                self._make_hook("diffusion.time_embedder")
            )
        )

        # Time conditioner
        self._hooks.append(
            diff.time_conditioner.register_forward_hook(
                self._make_hook("diffusion.time_conditioner")
            )
        )

        # AdaLN per-layer
        for i, adaln in enumerate(diff.adaln_layers):
            self._hooks.append(
                adaln.register_forward_hook(self._make_adaln_hook(f"diffusion.adaln.{i}"))
            )

        # Diffusion decoder layers
        for i, layer in enumerate(diff.layers):
            self._hooks.append(
                layer.register_forward_hook(self._make_hook(f"diffusion.layer.{i}"))
            )

        # Final norm
        self._hooks.append(
            diff.final_norm.register_forward_hook(
                self._make_hook("diffusion.final_norm")
            )
        )

        # Output head
        self._hooks.append(
            diff.output_head.register_forward_hook(
                self._make_hook("diffusion.output_head")
            )
        )

        # --- Cross-attention adapters ---
        for i, adapter in enumerate(model.cross_attention.adapters):
            self._hooks.append(
                adapter.register_forward_hook(
                    self._make_xattn_hook(f"cross_attn.adapter.{i}")
                )
            )

        print(f"  Registered {len(self._hooks)} hooks on wm-raw model")

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                for item in output:
                    if isinstance(item, Tensor):
                        self.activations[name] = item.detach().cpu()
                        break
        return hook

    def _make_adaln_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple) and all(isinstance(t, Tensor) for t in output):
                self.activations[name] = torch.stack(
                    [t.detach().cpu() for t in output]
                )
            elif isinstance(output, Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def _make_xattn_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                k, v = output
                if isinstance(k, Tensor) and isinstance(v, Tensor):
                    self.activations[f"{name}.k"] = k.detach().cpu()
                    self.activations[f"{name}.v"] = v.detach().cpu()
            elif isinstance(output, Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> None:
    """Quick sanity check: run wm-raw forward with synthetic data."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from wm_raw.checkpoint import load_checkpoint
    from wm_raw.config import WorldModelConfig
    from wm_raw.models import WorldModel

    print("=== Smoke Test: wm-raw forward ===")
    set_deterministic(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    config = WorldModelConfig()
    model = WorldModel(config)
    model = model.to(device=device, dtype=dtype)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)

    model.eval()

    # Synthetic batch
    S = 64
    lat_h, lat_w = 64, 64  # 512x512 / 8
    patch_h, patch_w = lat_h // 2, lat_w // 2
    num_tokens = patch_h * patch_w
    token_dim = 64

    causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=device), diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    batch = {
        "task_type": "diffusion",
        "condition": {
            "input_ids": torch.randint(0, 151936, (1, S), device=device),
            "attention_mask": causal_mask,
            "position_ids": torch.arange(S, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1),
        },
        "state_target": torch.randn(1, lat_h * lat_w, 16, device=device, dtype=dtype),
        "latent_h": lat_h,
        "latent_w": lat_w,
        "timesteps": torch.tensor([0.5], device=device),
        "noise": torch.randn(1, num_tokens, token_dim, device=device, dtype=dtype),
    }

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        output = model(batch)

    print(f"Loss: {output.loss.item():.6f}")
    print("Smoke test passed ✓")


def cmd_replay(args: argparse.Namespace) -> None:
    """Load reference fixture, replay in wm-raw, compare layer-by-layer."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from wm_raw.checkpoint import load_online_dcp_weights
    from wm_raw.config import WorldModelConfig
    from wm_raw.models import WorldModel

    print("=== Replay & Compare ===")

    # Load reference
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"ERROR: reference file not found: {ref_path}")
        sys.exit(1)

    print(f"Loading reference: {ref_path}")
    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)

    fixture = ref_data["_fixture"]
    meta = ref_data.get("_meta", {})
    print(f"  Checkpoint: {meta.get('checkpoint', 'unknown')}")
    print(f"  Image: {meta.get('image_height', '?')}x{meta.get('image_width', '?')}")
    print(f"  Latent: {fixture['latent_h']}x{fixture['latent_w']}")
    print(f"  Tokens: {fixture['noisy_tokens'].shape}")
    print()

    # Build wm-raw model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    set_deterministic(meta.get("seed", 42))

    print("Building wm-raw model...")
    config = WorldModelConfig()
    model = WorldModel(config)
    model = model.to(device=device, dtype=dtype)

    # Load checkpoint
    checkpoint_path = args.checkpoint or meta.get("checkpoint", "")
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        load_online_dcp_weights(model, checkpoint_path)
    else:
        print("WARNING: No checkpoint specified, using random weights")

    model.eval()

    # Register hooks
    capture = WMRawActivationCapture()
    capture.register(model)

    # Replay the exact same forward pass
    print("\nRunning wm-raw forward...")
    lat_h = fixture["latent_h"]
    lat_w = fixture["latent_w"]
    patch_h = lat_h // 2
    patch_w = lat_w // 2

    condition_input_ids = fixture["condition_input_ids"].to(device)
    condition_attention_mask_raw = fixture["condition_attention_mask"].to(device)

    # Build proper 4D attention mask [B, 1, S, S] for VLM
    # Conditioning uses bidirectional (non-causal) attention.
    # If all tokens are valid (no padding), pass None for full attention.
    if condition_attention_mask_raw.ndim == 2:
        B, S = condition_attention_mask_raw.shape
        if condition_attention_mask_raw.all():
            # No padding — full bidirectional attention
            condition_attention_mask = None
        else:
            # Mask out padding key positions: [B, 1, 1, S] → broadcast to [B, 1, S, S]
            mask_4d = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
            pad_cols = (1 - condition_attention_mask_raw.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(dtype).min
            condition_attention_mask = mask_4d + pad_cols
    else:
        condition_attention_mask = condition_attention_mask_raw

    # Position IDs [3, B, S] — simple sequential
    B, S = condition_input_ids.shape
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    noisy_tokens = fixture["noisy_tokens"].to(device=device, dtype=dtype)
    timesteps = fixture["timesteps"].to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # VLM forward (condition)
        vlm_out = model.forward_vlm(
            input_ids=condition_input_ids,
            attention_mask=condition_attention_mask,
            position_ids=position_ids,
        )

        # Diffusion forward (using pre-computed noisy_tokens — no resampling)
        prediction = model.state_diffusion(
            noisy_latent=noisy_tokens,
            timesteps=timesteps,
            patch_h=patch_h,
            patch_w=patch_w,
            cross_attention_stack=model.cross_attention,
            vlm_hidden_states=vlm_out.hidden_states,
        )

    capture.activations["diffusion.prediction"] = prediction.detach().cpu()

    # Compute loss
    velocity_target = fixture["velocity_target"].to(device=device, dtype=dtype)
    loss = (prediction.float() - velocity_target.float()).pow(2).mean()
    capture.activations["_loss"] = loss.cpu()
    print(f"  wm-raw loss: {loss.item():.6f}")

    ref_loss = ref_data.get("_loss")
    if ref_loss is not None:
        print(f"  online loss: {ref_loss.item():.6f}")
        print(f"  loss diff:   {abs(loss.item() - ref_loss.item()):.2e}")
    print()

    # Compare activations
    print(f"Comparing activations (atol={args.atol:.1e})...")
    print("=" * 70)

    # Find common activation keys (exclude metadata keys starting with _)
    ref_keys = {k for k in ref_data if not k.startswith("_") and isinstance(ref_data[k], Tensor)}
    raw_keys = {k for k in capture.activations if isinstance(capture.activations[k], Tensor)}
    common = sorted(ref_keys & raw_keys)
    only_ref = sorted(ref_keys - raw_keys)
    only_raw = sorted(raw_keys - ref_keys)

    if only_ref:
        print(f"\n  Keys only in reference ({len(only_ref)}): {only_ref[:5]}...")
    if only_raw:
        print(f"\n  Keys only in wm-raw ({len(only_raw)}): {only_raw[:5]}...")
    print()

    results: list[TensorComparison] = []
    for key in common:
        result = compare_tensors(key, capture.activations[key], ref_data[key], atol=args.atol)
        results.append(result)

    # Also compare loss
    if ref_loss is not None:
        results.append(compare_tensors("_loss", loss.cpu(), ref_loss, atol=args.atol))

    # Print results grouped by component
    for prefix in ["vlm.layer", "cross_attn", "diffusion.input", "diffusion.time",
                   "diffusion.adaln", "diffusion.layer", "diffusion.final",
                   "diffusion.output", "diffusion.prediction", "_loss"]:
        group = [r for r in results if r.name.startswith(prefix)]
        if group:
            print(f"\n  [{prefix}*]")
            for r in group:
                print(str(r))

    # Summary
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("\n" + "=" * 70)
    print(f"Total: {len(results)} comparisons, {passed} passed, {failed} failed")

    if failed > 0:
        print("\n⚠ ALIGNMENT FAILED")
        # Show worst offenders
        worst = sorted(results, key=lambda r: r.max_abs_err, reverse=True)[:5]
        print("\nTop 5 worst:")
        for r in worst:
            print(f"    {r.name}: max_abs={r.max_abs_err:.2e}")
        sys.exit(1)
    else:
        print("\n✓ ALL ALIGNED")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two pre-captured activation dumps (offline)."""
    print("=== Offline Comparison ===")

    ref_path = Path(args.reference)
    act_path = Path(args.activations)

    for p, label in [(ref_path, "reference"), (act_path, "activations")]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    act = torch.load(act_path, map_location="cpu", weights_only=False)

    ref_keys = {k for k in ref if not k.startswith("_") and isinstance(ref[k], Tensor)}
    act_keys = {k for k in act if not k.startswith("_") and isinstance(act[k], Tensor)}
    common = sorted(ref_keys & act_keys)

    print(f"Reference keys: {len(ref_keys)}")
    print(f"Activation keys: {len(act_keys)}")
    print(f"Common keys: {len(common)}")
    print()

    results: list[TensorComparison] = []
    for key in common:
        result = compare_tensors(key, act[key], ref[key], atol=args.atol)
        results.append(result)
        print(str(result))

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="wm-raw numerical alignment tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # smoke
    p_smoke = sub.add_parser("smoke", help="Quick forward pass sanity check")
    p_smoke.add_argument("--checkpoint", type=str, default="")

    # replay
    p_replay = sub.add_parser("replay",
        help="Replay reference fixture in wm-raw and compare layer-by-layer")
    p_replay.add_argument("--reference", type=str, required=True,
                          help="Output from dump_online_activations.py")
    p_replay.add_argument("--checkpoint", type=str, default="",
                          help="Checkpoint path (auto-detected from reference if absent)")
    p_replay.add_argument("--atol", type=float, default=1e-3,
                          help="Absolute tolerance for bf16 comparison")

    # compare (offline, two dumps)
    p_cmp = sub.add_parser("compare", help="Compare two activation dump files")
    p_cmp.add_argument("--reference", type=str, required=True)
    p_cmp.add_argument("--activations", type=str, required=True)
    p_cmp.add_argument("--atol", type=float, default=1e-3)

    args = parser.parse_args()

    if args.command == "smoke":
        cmd_smoke(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
