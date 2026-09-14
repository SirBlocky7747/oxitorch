"""Tests for the native transform fast path: augmentation runs as engine
kernels *inside* the loader workers (no GIL, no per-item Python), with
loader-governed seeding."""

import struct

import numpy as np
import pytest

import oxitorch as ox
from oxitorch.utils.data import DataLoader, TensorDataset
from oxitorch.datasets import Compose, Normalize, RandomCrop, RandomHorizontalFlip


def _write_mnist(root, n=8, size=6):
    """Synthesizes an IDX3/IDX1 pair; returns the dataset."""
    raw = struct.pack(">IIII", 0x803, n, size, size) + bytes(
        i % 256 for i in range(n * size * size)
    )
    labels = struct.pack(">II", 0x801, n) + bytes(i % 8 for i in range(n))
    root.mkdir(parents=True, exist_ok=True)
    (root / "train-images-idx3-ubyte").write_bytes(raw)
    (root / "train-labels-idx1-ubyte").write_bytes(labels)
    return ox.datasets.MNIST(root, train=True)


@pytest.fixture()
def mnist_ds(tmp_path):
    return _write_mnist(tmp_path / "mnist")


# ---- path selection ------------------------------------------------------------


def test_native_chain_takes_fast_path(mnist_ds):
    chain = Compose([RandomCrop(4), Normalize(0.5, 0.25)])
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=chain)
    assert ds.native_parts is not None
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    batch = next(iter(loader))
    assert tuple(batch[0].shape) == (4, 1, 4, 4)
    assert batch[1].numpy().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_non_native_member_falls_back_to_per_item(mnist_ds):
    class Exotic:  # no native_repr: not runnable in the workers
        def __call__(self, img):
            return img

    ds = ox.datasets.MNIST(
        mnist_ds.root, train=True, transform=Compose([RandomCrop(4), Exotic()])
    )
    assert ds.native_parts is None
    loader = DataLoader(ds, batch_size=8)
    batch = next(iter(loader))
    assert tuple(batch[0].shape) == (8, 1, 4, 4)


def test_bare_native_transforms_work_without_compose(mnist_ds):
    for t in (RandomCrop(4), RandomHorizontalFlip(), Normalize(0.5, 0.25)):
        ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=t)
        assert ds.native_parts is not None


# ---- native batch semantics ------------------------------------------------------


def _batches(ds, batch_size, seed, workers, epochs=1):
    loader = DataLoader(ds, batch_size=batch_size, seed=seed, num_workers=workers)
    out = []
    for _ in range(epochs):
        for xb, _yb in loader:
            out.append(xb.numpy())
    return out


def test_native_batch_matches_direct_per_image_semantics(mnist_ds):
    """The native kernel and the per-image numpy path apply the same
    deterministic Normalize; batch values are bit-identical."""
    norm = Normalize(0.5, 0.25)
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=norm)
    (xb, _) = next(iter(DataLoader(ds, batch_size=8)))
    raw = ds.images.numpy()  # (n, 1, 6, 6) in [0, 1]
    want = (raw - 0.5) / 0.25
    assert np.array_equal(xb.numpy(), want)


def test_native_crop_positions_are_per_image(mnist_ds):
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=RandomCrop(4))
    loader = DataLoader(ds, batch_size=8, seed=1)
    xb = next(iter(loader))[0].numpy()
    raw = ds.images.numpy()  # (n, 1, 6, 6)
    # Each cropped 4x4 patch must appear somewhere in its source image.
    for i in range(8):
        patch = xb[i, 0]
        src = raw[i, 0]
        found = any(
            np.array_equal(patch, src[t : t + 4, l : l + 4])
            for t in range(3)
            for l in range(3)
        )
        assert found, f"image {i}: crop matches no window"


def test_seed_makes_epochs_reproducible_and_distinct(mnist_ds):
    ds = ox.datasets.MNIST(
        mnist_ds.root,
        train=True,
        transform=Compose([RandomCrop(4), RandomHorizontalFlip()]),
    )
    a = _batches(ds, 4, seed=9, workers=2, epochs=2)
    b = _batches(ds, 4, seed=9, workers=2, epochs=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], a[1]), "epoch 2 must draw fresh crops/flips"


def test_inline_and_worker_paths_are_bit_identical(mnist_ds):
    """Same seeds, same kernels: num_workers is invisible to the output."""
    ds = ox.datasets.MNIST(
        mnist_ds.root,
        train=True,
        transform=Compose([RandomCrop(4), Normalize(0.5, 0.25)]),
    )
    inline = _batches(ds, 3, seed=4, workers=0)
    for workers in (1, 3):
        par = _batches(ds, 3, seed=4, workers=workers)
        for x_inline, x_par in zip(inline, par):
            assert np.array_equal(x_inline, x_par)


def test_unseeded_loader_varies_without_explicit_seed(mnist_ds):
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=RandomCrop(4))
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    b1 = next(iter(loader))[0].numpy()
    b2 = next(iter(loader))[0].numpy()
    assert not np.array_equal(b1, b2)


def test_partial_epoch_advances_the_seed_stream(mnist_ds):
    """torch semantics: every __iter__ consumes the next slice of the seed
    stream, even if the pass is abandoned early."""
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=RandomCrop(4))
    loader = DataLoader(ds, batch_size=4, seed=3, num_workers=2)
    next(iter(loader))  # abandoned after one batch
    second_pass_first = next(iter(loader))[0].numpy()
    fresh = next(iter(DataLoader(ds, batch_size=4, seed=3, num_workers=2)))[0].numpy()
    assert not np.array_equal(second_pass_first, fresh)


def test_native_path_respects_shuffle_and_drop_last(mnist_ds):
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=Normalize(0.0, 1.0))
    loader = DataLoader(
        ds, batch_size=3, shuffle=True, drop_last=True, seed=2, num_workers=2
    )
    labels = [b[1].numpy().tolist() for b in loader]
    assert len(labels) == 2  # 8 rows // 3, drop_last
    flat = sorted(x for b in labels for x in b)
    assert len(flat) == 6 and len(set(flat)) == 6


# ---- training through the augmented loader ----------------------------------------


def test_training_with_native_augmentation_converges(mnist_ds):
    """A small CNN-family model trains through the augmented native loader —
    the full Phase-4 stack driven by the Phase-5 pipeline."""
    from oxitorch import nn
    from oxitorch.optim import AdamW

    chain = Compose([RandomCrop(4), Normalize(0.5, 0.25)])
    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=chain)
    loader = DataLoader(ds, batch_size=4, shuffle=True, seed=0, num_workers=2)

    model = nn.Sequential(nn.Linear(16, 8))
    opt = AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    first = last = None
    for _ in range(15):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb.reshape([4, 16]))
            target = yb.numpy().astype(int).tolist()  # cross_entropy's label form
            loss = nn.functional.cross_entropy(logits, target)
            loss.backward()
            opt.step()
            first = float(loss.numpy()) if first is None else first
            last = float(loss.numpy())
    assert np.isfinite(last) and last < first


def test_cross_entropy_through_native_loader(mnist_ds):
    """End-to-end: native batches feed nn functional loss directly."""
    from oxitorch import nn

    ds = ox.datasets.MNIST(mnist_ds.root, train=True, transform=Normalize(0.5, 0.5))
    loader = DataLoader(ds, batch_size=8, num_workers=2)
    model = nn.Sequential(nn.Linear(36, 10))
    for xb, _yb in loader:
        logits = model(xb.reshape([8, 36]))
        assert tuple(logits.shape) == (8, 10)
        # Labels arrive as an f32 batch tensor (native path); convert to the
        # integer-list form cross_entropy accepts (same as train_mnist.py).
        loss = nn.functional.cross_entropy(logits, _yb.numpy().astype(int).tolist())
        assert np.isfinite(float(loss.numpy()))
        break
