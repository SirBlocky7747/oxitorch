"""torch.utils.data-style pipeline (plan.md Phase 5).

Two execution paths behind one `DataLoader` API:

- **Native fast path** — datasets exposing a `tensors` attribute (tuple of
  column tensors with a shared dim-0 row count, e.g. `TensorDataset`, and
  `MNIST`/`CIFAR10` without transforms) delegate to
  `_native.DataLoader`: batches are gathered by Rust worker threads with
  no GIL and no per-item Python. This is the plan's "Rust-side threads
  instead of multiprocessing" selling point.

- **Generic path** — any `Dataset`/`IterableDataset` with per-item
  transforms and `default_collate`. With `num_workers > 0` items are
  fetched through a thread pool (transforms do numpy work that releases
  the GIL, so this parallelizes real augmentation pipelines).

Deviations from torch, by design: labels are f32 tensors (f32-only
compute) and `cross_entropy` accepts them directly; the native path's
shuffle uses the engine's seeded MT19937 while the generic path uses
NumPy — both reproducible per seed, but not the same stream.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import _native

__all__ = [
    "DataLoader",
    "Dataset",
    "IterableDataset",
    "RandomSampler",
    "SequentialSampler",
    "Subset",
    "TensorDataset",
    "default_collate",
]


class Dataset:
    """Base class: map-style datasets implement `__getitem__` + `__len__`."""

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class IterableDataset(Dataset):
    """Base class: stream-style datasets implement `__iter__`."""

    def __iter__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Zips tensors along dim 0: `TensorDataset(features, labels)`."""

    def __init__(self, *tensors):
        n = tensors[0].shape[0]
        for i, t in enumerate(tensors):
            if t.shape[0] != n:
                raise ValueError(f"tensor {i} has {t.shape[0]} rows, expected {n}")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


class Subset(Dataset):
    """A fixed-index view of another dataset (no native fast path)."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class Sampler:
    """Base class for index streams."""

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    """Yields `0 .. len(data_source)` in order."""

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    """Permuted index stream; `seed` gives reproducible epoch sequences."""

    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        if self.seed is None:
            order = np.random.permutation(n)
        else:
            order = np.random.RandomState(self.seed + self._epoch).permutation(n)
            self._epoch += 1
        return iter(int(i) for i in order)

    def __len__(self):
        return len(self.data_source)


def default_collate(items):
    """Collates fetched items into batch structures.

    - tuples/lists -> per-field collation (the `(x, y)` dataset case)
    - tensors      -> stacked batch tensor (leaf; keeps `requires_grad`)
    - ints/floats  -> f32 label tensor (torch uses integer dtypes; oxitorch
      is f32-compute, and `F.cross_entropy` accepts the result directly)
    - numpy arrays -> stacked f32 tensor
    """
    if not items:
        raise ValueError("cannot collate an empty item list")
    first = items[0]
    if isinstance(first, (tuple, list)):
        return tuple(default_collate([item[i] for item in items]) for i in range(len(first)))
    if isinstance(first, np.ndarray):
        # NB: per-item ascontiguousarray would promote 0-d arrays to (1,)
        # (numpy guarantees ndim >= 1); stack first, then make contiguous.
        stacked = np.stack([np.asarray(x, dtype=np.float32) for x in items])
        return _native.from_numpy(np.ascontiguousarray(stacked))
    if hasattr(first, "shape") and hasattr(first, "numpy"):  # oxitorch Tensor
        stacked = np.stack(
            [np.asarray(x.detach().numpy(), dtype=np.float32) for x in items]
        )
        requires_grad = all(bool(getattr(x, "requires_grad", False)) for x in items)
        return _native.from_numpy(np.ascontiguousarray(stacked), requires_grad)
    if isinstance(first, (int, float)):
        return _native.from_numpy(
            np.ascontiguousarray(np.asarray(items, dtype=np.float32))
        )
    raise TypeError(f"default_collate cannot handle items of type {type(first)!r}")


class DataLoader:
    """Batches a dataset; mirrors torch's constructor surface.

    `num_workers` is the *Rust* thread count on the native fast path (no
    GIL) and a thread-pool size on the generic path (numpy transforms
    release the GIL). `seed` makes shuffled epoch orders reproducible.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=default_collate,
        drop_last=False,
        seed=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.seed = seed
        # Unseeded loaders still get per-run randomness (shuffle order and
        # native transform draws): a random base picked once, so epochs vary
        # while a single run stays reproducible. Explicit `seed` overrides.
        self._auto_seed = int.from_bytes(os.urandom(4), "little")
        self._epoch = 0

    def __len__(self):
        n = self._length_source()
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _length_source(self):
        if isinstance(self.dataset, IterableDataset):
            raise TypeError("IterableDataset has no length; len(DataLoader) is undefined")
        return len(self.dataset)

    def _native_loader_for(self, epoch):
        """The `_native.DataLoader` for tensor-backed datasets, or None."""
        if self.sampler is not None or self.batch_sampler is not None:
            return None
        if self.collate_fn is not default_collate:
            return None  # custom collation needs per-item Python
        # Each pass gets its own epoch-derived seed so shuffled epochs differ
        # while staying reproducible (the native loader's internal counter
        # would otherwise restart from 0 on every construction).
        base = self.seed if self.seed is not None else self._auto_seed
        epoch_seed = base + epoch
        # Augmented fast path: datasets exposing `native_parts` hand over the
        # raw image columns plus a transform repr the engine runs *inside the
        # workers*; the loader seed governs crop/flip randomness.
        parts = getattr(self.dataset, "native_parts", None)
        if parts is not None:
            raw, native_repr = parts
            columns = [
                t if isinstance(t, _native.Tensor) else _native.from_numpy(np.ascontiguousarray(t))
                for t in raw
            ]
            return _native.DataLoader(
                columns,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                seed=epoch_seed,
                num_workers=self.num_workers,
                transforms=_native.NativeTransform(native_repr),
                transform_seed=epoch_seed,
            )
        tensors = getattr(self.dataset, "tensors", None)
        if tensors is None:
            return None
        columns = [
            t if isinstance(t, _native.Tensor) else _native.from_numpy(np.ascontiguousarray(t))
            for t in tensors
        ]
        return _native.DataLoader(
            columns,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            seed=epoch_seed,
            num_workers=self.num_workers,
            transforms=None,
            transform_seed=epoch_seed,
        )

    def __iter__(self):
        # The epoch counter advances on every `__iter__` (torch semantics):
        # each pass — full or partially consumed — draws a fresh slice of the
        # seed stream, so a seeded run is reproducible epoch-for-epoch while
        # unseeded runs vary.
        epoch = self._epoch
        self._epoch += 1
        native = self._native_loader_for(epoch)
        if native is not None:
            # Fast path: batches built by Rust threads (gather + native
            # transforms), seeded from this pass's epoch.
            for batch in native.__iter__():
                yield batch
            return

        if self.batch_sampler is not None:
            for idx in self.batch_sampler:
                yield self._collate_index_batch(list(idx))
            return

        if isinstance(self.dataset, IterableDataset):
            yield from self._iter_stream()
            return

        if self.sampler is not None:
            indices = iter(self.sampler)
        elif self.shuffle:
            indices = iter(RandomSampler(self.dataset, seed=self._seed_for_epoch(epoch)))
        else:
            indices = iter(SequentialSampler(self.dataset))
        yield from self._iter_index_batches(indices)

    def _seed_for_epoch(self, epoch):
        if self.seed is None:
            return None
        return self.seed + epoch

    def _iter_index_batches(self, indices):
        """Chunks an index stream into batches and collates them."""
        batch = []
        for i in indices:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield self._collate_index_batch(batch)
                batch = []
        if batch and (not self.drop_last or len(batch) == self.batch_size):
            yield self._collate_index_batch(batch)

    def _iter_stream(self):
        """Collates a streamed item sequence into fixed-size batches."""
        items = []
        for item in self.dataset:
            items.append(item)
            if len(items) == self.batch_size:
                yield self.collate_fn(items)
                items = []
        if items and not self.drop_last:
            yield self.collate_fn(items)

    def _collate_index_batch(self, batch):
        """Fetches (and optionally parallelizes) then collates one batch."""
        if self.num_workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                items = list(pool.map(self.dataset.__getitem__, batch))
        else:
            items = [self.dataset[i] for i in batch]
        return self.collate_fn(items)
