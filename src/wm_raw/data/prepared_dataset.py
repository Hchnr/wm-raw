"""Prepared image-caption dataset for wm_sequence_prepared format.

Reads GPIC prepared shards and exposes the same interface as ImageCaptionDataset,
so DiffusionCollator works without changes.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .prepared_manifest import ShardRecord, load_shard_records, validate_prepared_snapshot

logger = logging.getLogger(__name__)


class PreparedImageCaptionDataset(Dataset):
    """Map-style dataset over prepared wm_sequence_prepared shards.

    Each __getitem__ returns:
        {"image": PIL.Image (RGB), "caption": str, "image_path": str}

    This matches the interface of ImageCaptionDataset so DiffusionCollator
    works unchanged.
    """

    def __init__(
        self,
        prepared_root: str | Path,
        *,
        image_size: int = 256,
        center_crop: bool = True,
        max_read_retries: int = 3,
        max_samples: int | None = None,
        validate: bool = False,
    ) -> None:
        self.prepared_root = Path(prepared_root).expanduser()
        if not self.prepared_root.is_dir():
            raise FileNotFoundError(f"prepared_root does not exist: {self.prepared_root}")

        if validate:
            _, shards = validate_prepared_snapshot(self.prepared_root)
        else:
            shards = load_shard_records(self.prepared_root)

        if not shards:
            raise ValueError(f"No shards found in {self.prepared_root}")

        self.shards = shards
        self.image_size = image_size
        self.center_crop = center_crop
        self.max_read_retries = max_read_retries
        self.max_samples = max_samples

        # Build prefix-sum array for global→(shard, local) mapping
        self._prefix_counts = _build_prefix_counts(self.shards)
        self._total = self._prefix_counts[-1]
        if max_samples is not None:
            self._total = min(self._total, max_samples)

        logger.info(
            f"PreparedImageCaptionDataset: {len(self.shards)} shards, "
            f"{self._total} examples (prepared_root={self.prepared_root})"
        )

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        last_error: Exception | None = None
        for attempt in range(self.max_read_retries + 1):
            candidate = (index + attempt) % self._prefix_counts[-1]
            try:
                return self._load_example(candidate)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 0:
                    logger.debug(f"Retry {attempt+1} for index {index}: {exc}")

        raise RuntimeError(
            f"Failed to load example at index {index} after "
            f"{self.max_read_retries + 1} attempts"
        ) from last_error

    def _locate(self, index: int) -> tuple[ShardRecord, int]:
        """Map global index to (shard, local_index)."""
        shard_idx = bisect_right(self._prefix_counts, index) - 1
        local_index = index - self._prefix_counts[shard_idx]
        return self.shards[shard_idx], local_index

    def _load_example(self, index: int) -> dict[str, Any]:
        """Load a single example by global index."""
        shard, local_index = self._locate(index)
        shard_dir = self.prepared_root / shard.relative_path

        # Read the target line from samples.jsonl
        raw_line = _read_jsonl_line(shard_dir / "samples.jsonl", local_index)
        example = json.loads(raw_line)

        # Extract caption and image path from the sequence example
        caption, image_uri = _extract_image_caption(example)
        image_path = shard_dir / image_uri

        # Load and resize PIL image
        image = _load_pil_image(image_path, image_size=self.image_size, center_crop=self.center_crop)

        return {
            "image": image,
            "caption": caption,
            "image_path": str(image_path),
        }


def _build_prefix_counts(shards: tuple[ShardRecord, ...]) -> tuple[int, ...]:
    """Build cumulative sum: prefix_counts[i] = sum of examples in shards[:i]."""
    counts = [0]
    running = 0
    for shard in shards:
        running += shard.num_examples
        counts.append(running)
    return tuple(counts)


def _read_jsonl_line(path: Path, target_index: int) -> str:
    """Read a specific line (by index, skipping blank lines) from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        current = 0
        for line in f:
            if not line.strip():
                continue
            if current == target_index:
                return line
            current += 1
    raise IndexError(f"Line index {target_index} out of range in {path}")


def _extract_image_caption(example: dict[str, Any]) -> tuple[str, str]:
    """Extract (caption_text, image_uri) from a WMSequenceExample dict.

    Looks for `image_diffusion` supervision to identify the condition (text)
    and target (image) segments.
    """
    segments_by_id = {seg["segment_id"]: seg for seg in example["segments"]}

    # Find image_diffusion supervision
    for supervision in example.get("supervisions", []):
        if supervision["kind"] == "image_diffusion":
            # Condition segments contain the text caption
            for seg_id in supervision.get("condition_segment_ids", []):
                seg = segments_by_id.get(seg_id)
                if seg and seg["modality"] == "text":
                    caption = seg["payload"]["text"]
                    break
            else:
                caption = ""

            # Target segments contain the image
            for seg_id in supervision.get("target_segment_ids", []):
                seg = segments_by_id.get(seg_id)
                if seg and seg["modality"] == "image":
                    image_uri = seg["payload"]["uri"]
                    return caption.strip(), image_uri

    # Fallback: find first text + first image segment
    caption = ""
    image_uri = ""
    for seg in example["segments"]:
        if seg["modality"] == "text" and not caption:
            caption = seg["payload"]["text"]
        elif seg["modality"] == "image" and not image_uri:
            image_uri = seg["payload"]["uri"]

    if not image_uri:
        raise ValueError(f"No image segment found in example {example.get('example_id', '?')}")
    if not caption:
        raise ValueError(f"No caption found in example {example.get('example_id', '?')}")

    return caption.strip(), image_uri


def _load_pil_image(
    path: Path,
    *,
    image_size: int,
    center_crop: bool,
    preserve_original_size: bool = False,
) -> Any:
    """Load and optionally resize a PIL image.

    When preserve_original_size=True (resolution bucket mode), the image is
    returned at full resolution — resizing happens later in the collator
    based on bucket assignment.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = img.convert("RGB")
        if preserve_original_size:
            return img.copy()
        if center_crop:
            img = ImageOps.fit(
                img,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
        else:
            img = img.resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        return img.copy()
