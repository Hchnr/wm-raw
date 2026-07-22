"""Frozen VAE codec for wm-raw diffusion training."""

from .frozen_codec import FrozenCodec, load_frozen_codec

__all__ = ["FrozenCodec", "load_frozen_codec"]
