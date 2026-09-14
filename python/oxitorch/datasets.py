"""torchvision-lite: MNIST / CIFAR10 datasets and image transforms.

MNIST downloads the official IDX files (gzip) from the same mirrors torch
uses, parses them in pure Rust (`_native.load_mnist_dir`), and returns
`(n, 1, 28, 28)` images in `[0, 1]` with `(n,)` f32 labels. CIFAR10 parses
the standard python-pickle distribution with numpy and normalizes to
`[0, 1]`. Transforms compose over numpy arrays (`PIL` is deliberately not
a dependency).
"""

import os
import pickle
import tarfile
import urllib.request

import numpy as np

from oxitorch import _native
from oxitorch.utils.data import Dataset

__all__ = ["CIFAR10", "MNIST", "Compose", "Normalize", "RandomCrop", "RandomHorizontalFlip"]

_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

_MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download(url, dest):
    """Downloads `url` to `dest` (atomic-ish via a .part file)."""
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


class MNIST(Dataset):
    """The MNIST handwritten-digits dataset, parsed in pure Rust.

    With `download=True` the IDX files are fetched from the official
    mirrors on first use. Images stay `(n, 1, 28, 28)` f32 in `[0, 1]`
    (torchvision's `ToTensor` semantics); labels are `(n,)` f32 class
    indices. With no transforms the dataset carries a `tensors` attribute,
    so `DataLoader` runs it on the Rust worker fast path.
    """

    def __init__(self, root, train=True, download=False, transform=None):
        self.root = str(root)
        self.train = train
        self.transform = transform
        os.makedirs(self.root, exist_ok=True)
        self._load(download)

    def _load(self, download):
        try:
            images, labels = _native.load_mnist_dir(self.root, self.train)
        except RuntimeError as e:
            if not download or "not found" not in str(e):
                raise
            self._download_files()
            images, labels = _native.load_mnist_dir(self.root, self.train)
        self.images = images
        self.labels = labels
        self._tensors = (images, labels)

    def _download_files(self):
        prefix = "train" if self.train else "t10k"
        names = [n for n in _MNIST_FILES if n.startswith(prefix)]
        for name in names:
            dest = os.path.join(self.root, name)
            if os.path.exists(dest):
                continue
            last_err = None
            for mirror in _MNIST_MIRRORS:
                try:
                    _download(mirror + name, dest)
                    last_err = None
                    break
                except OSError as e:  # noqa: PERF203 - retry loop by design
                    last_err = e
            if last_err is not None:
                raise RuntimeError(f"could not download {name}: {last_err}")

    @property
    def tensors(self):
        """Column tensors enabling the native worker fast path."""
        if self.transform is not None:
            return None  # per-item path applies transforms
        return self._tensors

    @property
    def native_parts(self):
        """`(raw_column_tensors, native_transform_repr)` for the augmented
        native fast path: batches are gathered and transformed entirely in
        Rust workers. `None` when the chain can't run natively (per-item
        path); raw (unnormalized) image columns always.
        """
        if self.transform is None:
            return None  # plain fast path via `tensors` covers this
        repr_ = None
        fn = getattr(self.transform, "native_repr", None)
        repr_ = fn() if fn is not None else None
        if repr_ is None:
            return None
        return self._tensors, repr_

    def __getitem__(self, index):
        img = self.images[index]  # (1, 28, 28) oxitorch tensor
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return int(self.labels.shape[0])


class CIFAR10(Dataset):
    """The CIFAR-10 dataset (python-pickle distribution).

    Images are `(n, 3, 32, 32)` f32 in `[0, 1]`; labels `(n,)` f32 class
    indices (planes=0 ... truck=9). With no transforms, `DataLoader` uses
    the Rust worker fast path.
    """

    _BATCH_FILES = [f"data_batch_{i}" for i in range(1, 6)]
    _TEST_FILE = "test_batch"

    def __init__(self, root, train=True, download=False, transform=None):
        self.root = str(root)
        self.train = train
        self.transform = transform
        base = os.path.join(self.root, "cifar-10-batches-py")
        if download and not os.path.isdir(base):
            self._download()
        if not os.path.isdir(base):
            raise FileNotFoundError(
                f"CIFAR10 not found under {self.root}; pass download=True to fetch it"
            )
        files = self._BATCH_FILES if train else [self._TEST_FILE]
        images, labels = [], []
        for name in files:
            with open(os.path.join(base, name), "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data = np.asarray(entry["data"], dtype=np.float32) / 255.0
            images.append(data.reshape(-1, 3, 32, 32))
            labels.append(np.asarray(entry["labels"], dtype=np.float32))
        self.images_np = np.concatenate(images)  # (n, 3, 32, 32)
        self.labels_np = np.concatenate(labels)  # (n,)
        self.images = _native.from_numpy(np.ascontiguousarray(self.images_np))
        self.labels = _native.from_numpy(np.ascontiguousarray(self.labels_np))
        self._tensors = (self.images, self.labels)

    def _download(self):
        dest = os.path.join(self.root, "cifar-10-python.tar.gz")
        if not os.path.exists(dest):
            _download(_CIFAR_URL, dest)
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(self.root)

    @property
    def tensors(self):
        if self.transform is not None:
            return None
        return self._tensors

    @property
    def native_parts(self):
        """`(raw_column_tensors, native_transform_repr)` for the augmented
        native fast path; `None` when the chain can't run natively."""
        if self.transform is None:
            return None
        fn = getattr(self.transform, "native_repr", None)
        repr_ = fn() if fn is not None else None
        if repr_ is None:
            return None
        return self._tensors, repr_

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return int(self.labels.shape[0])


# ---- transforms (numpy domain; native fast path inside loader workers) ----
#
# Transforms work in two domains:
#
# - **Directly** (per image, numpy or tensor in -> tensor out) — unchanged
#   semantics, honouring an explicit `seed`.
# - **Batched natively** — classes exposing `native_repr()` describe their
#   effect as engine kernels the DataLoader applies to whole gathered
#   batches *inside the Rust workers* (no GIL, no per-item Python). On that
#   path the loader's `seed` governs the randomness; a transform's own
#   `seed` applies only to direct calls.


class Compose:
    """Chains transforms: `Compose([RandomCrop(28), Normalize(...)])`.

    Native fast path: a composition whose members all expose `native_repr()`
    runs as engine kernels inside the loader workers; any non-native member
    falls back to the per-item numpy path.
    """

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def native_repr(self):
        """Wire format for the native batch path, or None if any member is
        non-native."""
        steps = []
        for t in self.transforms:
            fn = getattr(t, "native_repr", None)
            part = fn() if fn is not None else None
            if part is None:
                return None
            steps.extend(part)
        return steps or None

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomCrop:
    """Randomly crops an `(c, h, w)` numpy/tensor image via numpy views.

    Native fast path: `("crop", [size])` — per-image random positions drawn
    in the worker from the loader's seeded stream.
    """

    def native_repr(self):
        return [("crop", [float(self.size)])]

    def __init__(self, size, seed=None):
        self.size = size
        self.seed = seed
        self._epoch = 0

    def __call__(self, img):
        rng = np.random if self.seed is None else np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        c, h, w = img.shape
        if self.size > h or self.size > w:
            raise ValueError(f"crop {self.size} exceeds image {h}x{w}")
        top = int(rng.randint(0, h - self.size + 1))
        left = int(rng.randint(0, w - self.size + 1))
        arr = img if isinstance(img, np.ndarray) else img.numpy()
        crop = arr[:, top : top + self.size, left : left + self.size]
        return _native.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))


class RandomHorizontalFlip:
    """Flips `(c, h, w)` images horizontally with probability 0.5.

    Native fast path: `("flip", [])` — per-image coin flips drawn in the
    worker from the loader's seeded stream.
    """

    def native_repr(self):
        return [("flip", [])]

    def __init__(self, seed=None):
        self.seed = seed
        self._epoch = 0

    def __call__(self, img):
        rng = np.random if self.seed is None else np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        arr = img if isinstance(img, np.ndarray) else img.numpy()
        if (rng.random() < 0.5) if self.seed is None else (rng.rand() < 0.5):
            arr = arr[:, :, ::-1]
        return _native.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class Normalize:
    """Standardizes with per-channel mean/std over an `(c, h, w)` image.

    Native fast path: `("normalize", [means..., stds...])` — deterministic,
    applied batch-wide inside the workers.
    """

    def native_repr(self):
        return [("normalize", self.mean.ravel().tolist() + self.std.ravel().tolist())]

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img):
        arr = img if isinstance(img, np.ndarray) else img.numpy()
        return _native.from_numpy(
            np.ascontiguousarray((arr - self.mean) / self.std, dtype=np.float32)
        )
