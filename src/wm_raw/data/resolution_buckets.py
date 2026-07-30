"""Resolution bucket batch sampler for variable-size image training.

Each batch contains samples from the same resolution bucket, ensuring uniform
tensor shapes within a batch (required for efficient GPU computation and
torch.compile static shapes).

The bucket assignment is pre-computed and stored as a binary file where each
byte is the bucket_id for the corresponding dataset index.
"""

from __future__ import annotations

import logging
import random
from array import array
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

REJECTED_BUCKET_ID = 255  # Samples that don't fit any bucket


class ResolutionBucketBatchSampler:
    """Batch sampler that groups samples by resolution bucket.

    Each yielded batch is a list of (sample_index, bucket_id) tuples,
    all from the same bucket. This ensures uniform image dimensions within
    a batch for efficient batched VAE encoding and diffusion training.

    Compatible with DistributedDataParallel: set num_replicas and rank
    to partition samples across GPUs.
    """

    def __init__(
        self,
        *,
        assignment_path: str | Path,
        dataset_size: int,
        bucket_sizes: list[tuple[int, int]],
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.assignment_path = Path(assignment_path)
        self.dataset_size = int(dataset_size)
        self.bucket_count = len(bucket_sizes)
        self.bucket_sizes = bucket_sizes  # [(H, W), ...]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

        if not self.assignment_path.is_file():
            raise FileNotFoundError(
                f"Bucket assignment file not found: {self.assignment_path}"
            )

        # Partition samples into per-rank bucket lists
        self._local_bucket_indices, self._global_bucket_counts = self._partition()

        # How many full batches each bucket can produce (globally)
        self._batches_per_bucket = tuple(
            count // (self.batch_size * self.num_replicas)
            for count in self._global_bucket_counts
        )

        total_batches = sum(self._batches_per_bucket)
        if total_batches <= 0:
            raise ValueError(
                f"No complete batches possible with bucket_count={self.bucket_count}, "
                f"batch_size={self.batch_size}, num_replicas={self.num_replicas}"
            )

        logger.info(
            f"ResolutionBucketBatchSampler: {total_batches} batches/epoch, "
            f"bucket_counts={self._global_bucket_counts}, rank={self.rank}/{self.num_replicas}"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(self._batches_per_bucket)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        """Yield batches of (sample_index, bucket_id) tuples."""
        # Copy local indices for shuffling
        local_indices = [array("Q", indices) for indices in self._local_bucket_indices]

        if self.shuffle:
            for bucket_id, indices in enumerate(local_indices):
                random.Random(
                    self.seed + self.epoch * 1_000_003 + self.rank * 10_007 + bucket_id
                ).shuffle(indices)

        # Build batch schedule: which bucket to sample from next
        schedule = [
            bucket_id
            for bucket_id, batch_count in enumerate(self._batches_per_bucket)
            for _ in range(batch_count)
        ]
        if self.shuffle:
            random.Random(self.seed + self.epoch * 1_000_003).shuffle(schedule)

        # Yield batches
        offsets = [0] * self.bucket_count
        for bucket_id in schedule:
            start = offsets[bucket_id]
            stop = start + self.batch_size
            indices = local_indices[bucket_id][start:stop]
            offsets[bucket_id] = stop
            yield [(int(idx), bucket_id) for idx in indices]

    def _partition(self) -> tuple[tuple[array, ...], tuple[int, ...]]:
        """Read assignment file and partition samples by bucket and rank."""
        local = [array("Q") for _ in range(self.bucket_count)]
        global_counts = [0] * self.bucket_count

        with self.assignment_path.open("rb") as f:
            data = f.read(self.dataset_size)

        for global_index, bucket_id in enumerate(data):
            if bucket_id >= self.bucket_count:
                continue  # rejected or invalid
            occurrence = global_counts[bucket_id]
            if occurrence % self.num_replicas == self.rank:
                local[bucket_id].append(global_index)
            global_counts[bucket_id] += 1

        return tuple(local), tuple(global_counts)


def find_bucket_assignment_path(
    prepared_root: str | Path,
    bucket_sizes: list[tuple[int, int]],
) -> Path | None:
    """Find an existing bucket assignment cache matching the given bucket sizes.

    Searches the .wm_training_cache directory for a matching assignment file.
    Returns None if no cache exists (needs to be built by wm-training first).
    """
    import json

    cache_base = Path(prepared_root) / ".wm_training_cache" / "resolution_buckets"
    if not cache_base.is_dir():
        return None

    target_sizes = [[h, w] for h, w in bucket_sizes]
    for cache_dir in cache_base.iterdir():
        if not cache_dir.is_dir():
            continue
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            meta = json.loads(metadata_path.read_text())
            cached_sizes = meta.get("bucket_config", {}).get("sizes", [])
            if cached_sizes == target_sizes:
                assignment_path = cache_dir / "assignments.u8"
                if assignment_path.is_file():
                    return assignment_path
        except (json.JSONDecodeError, OSError):
            continue

    return None
