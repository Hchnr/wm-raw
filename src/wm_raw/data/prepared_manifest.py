"""Prepared dataset manifest reading for wm_sequence_prepared format.

Ported from wm_training.data.prepared_manifest — minimal subset needed
for reading prepared GPIC shards during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FORMAT_NAME = "wm_sequence_prepared"
FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata from dataset_info.json."""

    dataset_name: str
    split: str
    fingerprint: str
    num_shards: int
    num_examples: int
    format: str = FORMAT_NAME
    format_version: str = FORMAT_VERSION
    samples_per_shard: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetInfo:
        return cls(
            dataset_name=str(payload["dataset_name"]),
            split=str(payload["split"]),
            fingerprint=str(payload["fingerprint"]),
            num_shards=int(payload["num_shards"]),
            num_examples=int(payload["num_examples"]),
            format=str(payload.get("format", FORMAT_NAME)),
            format_version=str(payload.get("format_version", FORMAT_VERSION)),
            samples_per_shard=(
                None if payload.get("samples_per_shard") is None
                else int(payload["samples_per_shard"])
            ),
        )


@dataclass(frozen=True)
class ShardRecord:
    """One shard entry from shards.jsonl."""

    shard_id: str
    relative_path: str
    num_examples: int
    sha256: str | None = None
    status: str = "complete"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShardRecord:
        return cls(
            shard_id=str(payload["shard_id"]),
            relative_path=str(payload["relative_path"]),
            num_examples=int(payload["num_examples"]),
            sha256=payload.get("sha256"),
            status=str(payload.get("status", "complete")),
        )


def load_dataset_info(prepared_root: Path) -> DatasetInfo:
    """Load dataset_info.json from the prepared root directory."""
    info_path = Path(prepared_root) / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return DatasetInfo.from_dict(json.load(f))


def load_shard_records(prepared_root: Path) -> tuple[ShardRecord, ...]:
    """Load shard records from shards.jsonl."""
    shards_path = Path(prepared_root) / "shards.jsonl"
    if not shards_path.is_file():
        raise FileNotFoundError(
            f"shards.jsonl not found at: {shards_path}\n"
            f"Ensure prepared_root points to a valid wm_sequence_prepared directory."
        )
    records: list[ShardRecord] = []
    with shards_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ShardRecord.from_dict(json.loads(line)))
    return tuple(records)


def validate_prepared_snapshot(prepared_root: Path) -> tuple[DatasetInfo, tuple[ShardRecord, ...]]:
    """Basic structural validation — checks counts match, shard dirs exist."""
    root = Path(prepared_root)
    info = load_dataset_info(root)
    shards = load_shard_records(root)

    if info.num_shards != len(shards):
        raise ValueError(
            f"dataset_info num_shards={info.num_shards} but shards.jsonl has "
            f"{len(shards)} records"
        )

    total_examples = sum(r.num_examples for r in shards)
    if info.num_examples != total_examples:
        raise ValueError(
            f"dataset_info num_examples={info.num_examples} but shards sum to "
            f"{total_examples}"
        )

    # Spot-check that shard directories exist (don't hash-verify for speed)
    for record in shards:
        shard_dir = root / record.relative_path
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"shard directory missing: {shard_dir}")
        samples_path = shard_dir / "samples.jsonl"
        if not samples_path.is_file():
            raise FileNotFoundError(f"samples.jsonl missing: {samples_path}")

    return info, shards
