"""Phase 5 data-pipeline tests: batching correctness, worker parity, RNG
NumPy-compatibility, and training end-to-end through the loader."""

import numpy as np
import pytest

import oxitorch as ox
from oxitorch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
    TensorDataset,
    default_collate,
)


@pytest.fixture()
def tiny():
    x = ox.from_numpy(np.arange(24, dtype=np.float32).reshape(12, 2))
    y = ox.from_numpy(np.arange(12, dtype=np.float32))
    return TensorDataset(x, y)


def _labels(loader):
    return [b[1].numpy().tolist() for b in loader]


# ---- batching structure ------------------------------------------------------


def test_sequential_batches_with_remainder(tiny):
    loader = DataLoader(tiny, batch_size=5)
    assert _labels(loader) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]]


def test_len_matches_batch_math(tiny):
    assert len(DataLoader(tiny, batch_size=5)) == 3
    assert len(DataLoader(tiny, batch_size=5, drop_last=True)) == 2
    assert len(DataLoader(tiny, batch_size=12)) == 1


def test_drop_last_discards_only_the_tail(tiny):
    loader = DataLoader(tiny, batch_size=5, drop_last=True)
    assert _labels(loader) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_batch_size_one_and_full(tiny):
    assert len(_labels(DataLoader(tiny, batch_size=1))) == 12
    assert _labels(DataLoader(tiny, batch_size=12)) == [list(range(12))]


def test_columns_keep_trailing_dims():
    imgs = ox.from_numpy(np.arange(24, dtype=np.float32).reshape(6, 1, 2, 2))
    ds = TensorDataset(imgs)
    batch = next(iter(DataLoader(ds, batch_size=6)))
    assert tuple(batch[0].shape) == (6, 1, 2, 2)


# ---- native worker path (Rust threads) ---------------------------------------


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_worker_parity_with_inline_path(tiny, workers):
    seq = _labels(DataLoader(tiny, batch_size=3))
    got = _labels(DataLoader(tiny, batch_size=3, num_workers=workers))
    assert got == seq


def test_worker_shuffle_differs_from_sequential(tiny):
    shuffled = _labels(DataLoader(tiny, batch_size=3, shuffle=True, seed=1, num_workers=2))
    assert shuffled != _labels(DataLoader(tiny, batch_size=3))
    assert sorted(x for b in shuffled for x in b) == list(range(12))


def test_worker_epochs_differ_but_are_reproducible(tiny):
    loader = DataLoader(tiny, batch_size=4, shuffle=True, seed=42, num_workers=2)
    e0 = _labels(loader)
    e1 = _labels(loader)
    assert e0 != e1
    fresh = DataLoader(tiny, batch_size=4, shuffle=True, seed=42, num_workers=2)
    assert _labels(fresh) == e0


def test_worker_drop_last_and_last_batch_shapes(tiny):
    loader = DataLoader(tiny, batch_size=5, drop_last=True, num_workers=3)
    assert _labels(loader) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_single_epoch_covers_every_row_once(tiny):
    for workers in (2, 3):
        loader = DataLoader(tiny, batch_size=4, shuffle=True, seed=7, num_workers=workers)
        seen = [x for b in loader for x in b[1].numpy().tolist()]
        assert sorted(seen) == list(range(12))


def test_batch_tensors_are_leaf_constants(tiny):
    batch = next(iter(DataLoader(tiny, batch_size=4, num_workers=2)))
    assert batch[0].requires_grad is False
    assert batch[1].requires_grad is False


def test_empty_loader_rejected():
    x = ox.from_numpy(np.zeros((0, 2), dtype=np.float32))
    with pytest.raises(Exception):
        ox._native.DataLoader([x])


# ---- generic per-item path ----------------------------------------------------


def test_numpy_dataset_with_collate():
    class NumpyDS:
        def __len__(self):
            return 7

        def __getitem__(self, i):
            return np.full((3,), float(i), dtype=np.float32), float(i)

    loader = DataLoader(NumpyDS(), batch_size=4)
    batches = list(loader)
    assert len(batches) == 2
    x1, y1 = batches[0]
    assert tuple(x1.shape) == (4, 3)
    assert y1.numpy().tolist() == [0, 1, 2, 3]
    x2, y2 = batches[1]
    assert tuple(x2.shape) == (3, 3)
    assert y2.numpy().tolist() == [4, 5, 6]


def test_generic_path_with_workers_and_shuffle():
    class NumpyDS:
        def __len__(self):
            return 20

        def __getitem__(self, i):
            return float(i)

    loader = DataLoader(NumpyDS(), batch_size=6, shuffle=True, seed=3, num_workers=3)
    seen = [v for b in loader for v in b.numpy().tolist()]
    assert sorted(seen) == list(range(20))


def test_samplers_sequential_and_random(tiny):
    seq = DataLoader(tiny, batch_size=6, sampler=SequentialSampler(tiny))
    assert _labels(seq) == [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    rnd = DataLoader(tiny, batch_size=12, sampler=RandomSampler(tiny, seed=5))
    (only,) = _labels(rnd)
    assert sorted(only) == list(range(12)) and only != list(range(12))


def test_subset_restricts_the_view(tiny):
    sub = Subset(tiny, [3, 1, 9])
    assert len(sub) == 3
    # Subset items are (x, y) tuples fetched per item; collated labels are
    # a batch tensor via default_collate's tuple path.
    loader = DataLoader(sub, batch_size=3)
    xs, ys = next(iter(loader))
    assert tuple(xs.shape) == (3, 2)
    assert ys.numpy().tolist() == [3.0, 1.0, 9.0]


def test_default_collate_scalar_and_tensor_mix():
    items = [(np.ones(2, dtype=np.float32), 5), (np.ones(2, dtype=np.float32) * 2, 7)]
    xs, ys = default_collate(items)
    assert tuple(xs.shape) == (2, 2)
    assert ys.numpy().tolist() == [5.0, 7.0]


# ---- transforms + dataset plumbing -------------------------------------------


def test_mnist_idx_parsing(tmp_path):
    # Hand-build a tiny IDX3/IDX1 pair and read it through the Rust parser.
    import struct

    images = struct.pack(">IIII", 0x803, 3, 2, 2) + bytes(
        [0, 255, 128, 64, 10, 20, 30, 40, 50, 60, 70, 80]
    )
    labels = struct.pack(">II", 0x801, 3) + bytes([9, 4, 1])
    root = tmp_path / "mnist"
    root.mkdir()
    (root / "train-images-idx3-ubyte").write_bytes(images)
    (root / "train-labels-idx1-ubyte").write_bytes(labels)

    import gzip

    (root / "t10k-images-idx3-ubyte.gz").write_bytes(
        gzip.compress(struct.pack(">IIII", 0x803, 2, 2, 2) + bytes(range(16)))
    )
    (root / "t10k-labels-idx1-ubyte.gz").write_bytes(
        gzip.compress(struct.pack(">II", 0x801, 2) + bytes([7, 2]))
    )

    ds = ox.datasets.MNIST(root, train=True)
    assert len(ds) == 3
    imgs, labels = ds.tensors
    assert tuple(imgs.shape) == (3, 1, 2, 2)
    assert labels.numpy().tolist() == [9.0, 4.0, 1.0]
    px = imgs.numpy()
    assert float(px[0, 0, 0, 1]) == pytest.approx(1.0)
    assert float(px[0, 0, 1, 0]) == pytest.approx(128 / 255, abs=1e-6)

    tds = ox.datasets.MNIST(root, train=False)
    imgs, labels = tds.tensors
    assert tuple(imgs.shape) == (2, 1, 2, 2)
    assert labels.numpy().tolist() == [7.0, 2.0]


def test_transforms_numpy_domain():
    from oxitorch.datasets import Compose, Normalize, RandomCrop

    img = np.arange(48, dtype=np.float32).reshape(3, 4, 4)

    crop = RandomCrop(2, seed=0)
    out = crop(img)
    assert tuple(out.shape) == (3, 2, 2)
    # An explicitly-seeded transform recomputes its RNG per call, so the
    # same image always yields the same crop.
    crop_fresh = RandomCrop(2, seed=0)
    out2 = crop_fresh(img)
    assert np.allclose(out.numpy(), out2.numpy())

    norm = Normalize([0.5, 0.4, 0.3], [2.0, 2.0, 2.0])
    out = norm(img)
    assert float(out.numpy()[0, 0, 0]) == pytest.approx((0.0 - 0.5) / 2.0)

    comp = Compose([RandomCrop(2, seed=1), Normalize(0.0, 1.0)])
    out = comp(img)
    assert tuple(out.shape) == (3, 2, 2)


def test_transformed_dataset_uses_per_item_path(tmp_path):
    from oxitorch.datasets import Normalize

    images = np.zeros((5, 1, 4, 4), dtype=np.float32)
    root = tmp_path / "mnist"
    root.mkdir()
    # Load via the Rust parser from a synthesized IDX file.
    import struct

    raw = struct.pack(">IIII", 0x803, 5, 4, 4) + b"\x00" * 80
    labels = struct.pack(">II", 0x801, 5) + bytes([1, 2, 3, 4, 5])
    (root / "train-images-idx3-ubyte").write_bytes(raw)
    (root / "train-labels-idx1-ubyte").write_bytes(labels)

    ds = ox.datasets.MNIST(root, train=True, transform=Normalize(0.0, 1.0))
    assert ds.tensors is None  # fast path disabled with transforms
    loader = DataLoader(ds, batch_size=5)
    (x, y) = next(iter(loader))
    assert tuple(x.shape) == (5, 1, 4, 4)
    assert y.numpy().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_cifar10_from_synthetic_batches(tmp_path):
    # CIFAR-10's real distribution is 163 MB; build the same pickle format.
    import pickle

    base = tmp_path / "cifar-10-batches-py"
    base.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 6):
        data = rng.integers(0, 256, size=(10, 3072), dtype=np.uint8)
        entry = {"data": data, "labels": list(range(10))}
        with open(base / f"data_batch_{i}", "wb") as f:
            pickle.dump(entry, f)
    with open(base / "test_batch", "wb") as f:
        pickle.dump(
            {"data": rng.integers(0, 256, size=(4, 3072), dtype=np.uint8), "labels": [9, 8, 7, 6]},
            f,
        )

    ds = ox.datasets.CIFAR10(tmp_path, train=True)
    assert len(ds) == 50
    imgs, labels = ds.tensors
    assert tuple(imgs.shape) == (50, 3, 32, 32)
    assert float(imgs.numpy().max()) <= 1.0
    assert labels.numpy().tolist() == [float(i % 10) for i in range(50)]

    tds = ox.datasets.CIFAR10(tmp_path, train=False)
    imgs, labels = tds.tensors
    assert tuple(imgs.shape) == (4, 3, 32, 32)
    assert labels.numpy().tolist() == [9.0, 8.0, 7.0, 6.0]

    # And it flows through the native loader fast path.
    loader = DataLoader(ds, batch_size=2, num_workers=2)
    batch = next(iter(loader))
    assert tuple(batch[0].shape) == (2, 3, 32, 32)


# ---- training through the loader (end-to-end integration) --------------------


def test_train_mlp_through_loader_workers(tiny):
    from oxitorch import nn
    from oxitorch.optim import AdamW

    model = nn.Sequential(nn.Linear(2, 1))
    rng = np.random.default_rng(0)
    model.layers[0].weight.tensor = ox.from_numpy(
        (rng.standard_normal((1, 2)) * 0.5).astype(np.float32), requires_grad=True
    )
    model.layers[0].bias.tensor = ox.from_numpy(
        np.zeros(1, dtype=np.float32), requires_grad=True
    )

    # Linear data: y = 3*x0 - 2*x1 + 1 over x in [-1, 1].
    feats = rng.uniform(-1, 1, size=(64, 2)).astype(np.float32)
    targets_np = (3 * feats[:, 0] - 2 * feats[:, 1] + 1).astype(np.float32)
    ds = TensorDataset(
        ox.from_numpy(feats), ox.from_numpy(targets_np)
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True, seed=0, num_workers=2)

    opt = AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    first_loss = None
    last_loss = None
    for _epoch in range(30):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).reshape([-1])  # match the (b,) label layout
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            opt.step()
            if first_loss is None:
                first_loss = float(loss.numpy())
            last_loss = float(loss.numpy())
    assert np.isfinite(last_loss)
    assert last_loss < first_loss * 0.5

    with ox.no_grad():
        x_test = ox.from_numpy(np.array([[1.0, -1.0]], dtype=np.float32))
        pred = model(x_test)
    assert float(pred.numpy()[0, 0]) == pytest.approx(3 + 2 + 1, abs=0.35)


def test_training_results_identical_with_and_without_workers(tiny):
    """Same seed, same batches, same trained weights — worker count is
    invisible to the math (batch order is deterministic per seed)."""
    from oxitorch import nn
    from oxitorch.optim import Sgd

    def run(workers):
        rng = np.random.default_rng(7)
        feats = rng.uniform(-1, 1, size=(32, 2)).astype(np.float32)
        targets = (feats[:, 0] + feats[:, 1]).astype(np.float32)
        ds = TensorDataset(ox.from_numpy(feats), ox.from_numpy(targets))
        loader = DataLoader(ds, batch_size=8, shuffle=True, seed=11, num_workers=workers)
        model = nn.Sequential(nn.Linear(2, 1))
        init_w = ox.from_numpy(np.array([[0.4, 0.4]], dtype=np.float32), requires_grad=True)
        init_b = ox.from_numpy(np.array([0.1], dtype=np.float32), requires_grad=True)
        model.layers[0].weight.tensor = init_w
        model.layers[0].bias.tensor = init_b
        opt = Sgd(model.parameters(), lr=0.1)
        for _ in range(8):
            for xb, yb in loader:
                opt.zero_grad()
                pred = model(xb).reshape([-1])  # match the (b,) label layout
                loss = ((pred - yb) ** 2).mean()
                loss.backward()
                opt.step()
        return model.state_dict()

    a = run(0)
    b = run(3)
    for key in a:
        assert np.allclose(a[key], b[key], atol=1e-6)
