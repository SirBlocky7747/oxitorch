"""oxitorch: a PyTorch-equivalent deep-learning library with a Rust core.

Phase 3 surface: strided, broadcast-capable ``Tensor`` with numpy interop,
an autograd engine (``backward``/``grad``/``no_grad``/``enable_grad``), the
``nn`` module system, and optimizers.
"""

from contextlib import contextmanager

from oxitorch._native import (
    Tensor,
    __version__,
    device_count,
    eye,
    full,
    from_numpy,
    is_grad_enabled,
    set_grad_enabled,
    version,
    zeros,
)
from oxitorch import nn, optim
from oxitorch import utils
from oxitorch import datasets

__all__ = [
    "Tensor",
    "__version__",
    "device_count",
    "datasets",
    "enable_grad",
    "eye",
    "full",
    "from_numpy",
    "is_grad_enabled",
    "no_grad",
    "set_grad_enabled",
    "utils",
    "version",
    "zeros",
    "nn",
    "optim",
]


@contextmanager
def _grad_mode(enabled):
    """Shared implementation of `no_grad` / `enable_grad` (save/restore)."""
    prev = set_grad_enabled(enabled)
    try:
        yield
    finally:
        set_grad_enabled(prev)


def no_grad():
    """Context manager disabling graph recording (torch parity).

    Inside the block, ops return detached tensors and ``backward()``
    cannot run through them — use for inference and computing metrics.
    Nesting composes: the previous mode is restored on exit.

    >>> with ox.no_grad():
    ...     out = model(x)
    """
    return _grad_mode(False)


def enable_grad():
    """Context manager force-enabling graph recording (torch parity).

    Useful to train through a helper that runs under `no_grad`. The
    previous mode is restored on exit, however deep the nesting.
    """
    return _grad_mode(True)
