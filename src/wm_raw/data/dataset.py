"""Image-caption dataset for text-to-image diffusion training.

Manifest format (JSONL):
    {"image_path": "path/to/img.png", "caption": "A cat sitting on a mat"}
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImageCaptionRecord:
    """One (image_path, caption) entry from the manifest."""

    image_path: Path
    caption: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_manifest(
    manifest_path: str | Path,
    *,
    image_root: str | Path | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    shuffle: bool = False,
) -> list[ImageCaptionRecord]:
    """Load a JSONL manifest into a list of records.

    Args:
        manifest_path: path to .jsonl file
        image_root: base dir for relative image paths (defaults to manifest dir)
        max_samples: cap on number of samples
        seed: random seed for shuffling
        shuffle: whether to shuffle before capping
    """
    manifest_path = Path(manifest_path).expanduser()
    manifest_dir = manifest_path.parent
    root = Path(image_root).expanduser() if image_root else manifest_dir

    records: list[ImageCaptionRecord] = []
    with manifest_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            raw_path = payload.get("image_path") or payload.get("image") or payload.get("file_name")
            if not raw_path:
                raise ValueError(f"Line {line_num}: missing image_path")
            caption = payload.get("caption") or payload.get("text") or ""
            if not caption.strip():
                raise ValueError(f"Line {line_num}: empty caption")

            img_path = Path(raw_path)
            if not img_path.is_absolute():
                img_path = root / img_path

            meta = {k: v for k, v in payload.items() if k not in {"image_path", "image", "file_name", "caption", "text"}}
            records.append(ImageCaptionRecord(image_path=img_path, caption=caption.strip(), metadata=meta))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
    if max_samples is not None:
        records = records[:max_samples]
    return records


class ImageCaptionDataset(Dataset):
    """PyTorch Dataset over (image, caption) pairs.

    Loads images lazily. Returns dicts with:
        - image: PIL.Image.Image (RGB)
        - caption: str
        - image_path: str
    """

    def __init__(
        self,
        records: Sequence[ImageCaptionRecord],
    ) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return {
            "image": image,
            "caption": record.caption,
            "image_path": str(record.image_path),
        }
