"""Layer classes (Phase 3 subset + Phase 4 conv layers): Linear, norms,
Embedding, Dropout, activations, Conv2d, MaxPool2d.

Each layer owns its `Parameter`s with torch-compatible init (kaiming-uniform
for Linear, fan-in uniform for Conv2d, ones/zeros for norms) and composes
the functional API.
"""

import numpy as np

from . import functional as F
from .module import Module, Parameter


def _kaiming_uniform_bound(fan_in):
    """torch.nn.Linear default bound: 1/sqrt(fan_in) (kaiming-uniform, a=sqrt(5))."""
    return 1.0 / np.sqrt(fan_in)


class Linear(Module):
    """Affine layer `y = x @ W.T + b` with torch's kaiming-uniform init."""

    def __init__(self, in_features, out_features, bias=True, seed=None):
        rng = np.random.default_rng(seed)
        # torch.nn.Linear default: kaiming-uniform with a=sqrt(5) gives
        # U(-1/sqrt(fan_in), 1/sqrt(fan_in)); bias likewise.
        bound = _kaiming_uniform_bound(in_features)
        w = rng.uniform(-bound, bound, (out_features, in_features))
        self.weight = Parameter(F._native_from_numpy(w.astype(np.float32), True))
        self.bias = (
            Parameter(
                F._native_from_numpy(
                    rng.uniform(-bound, bound, out_features).astype(np.float32), True
                )
            )
            if bias
            else None
        )
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return F.linear(x, self.weight.tensor, self.bias.tensor if self.bias else None)


class Conv2d(Module):
    """2-D convolution over NCHW input, torch layout and semantics.

    `weight` is `(out_channels, in_channels, kh, kw)` initialized like
    `torch.nn.Conv2d` (kaiming-uniform, a=sqrt(5) over the receptive-field
    fan-in); optional `bias` starts at zero. Forward is `F.conv2d` —
    im2col + bmm — so every gradient path is already graph-connected.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw  # torch's Conv2d fan_in (receptive field)
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(fan_in)
        w = rng.uniform(-bound, bound, (out_channels, in_channels, kh, kw))
        self.weight = Parameter(F._native_from_numpy(w.astype(np.float32), True))
        self.bias = (
            Parameter(F._native_from_numpy(np.zeros(out_channels, dtype=np.float32), True))
            if bias
            else None
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight.tensor,
            self.bias.tensor if self.bias else None,
            stride=self.stride,
            padding=self.padding,
        )


class MaxPool2d(Module):
    """Max pooling over NCHW input; stride defaults to the kernel size."""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride)


class LayerNorm(Module):
    """LayerNorm over the last dim with learned affine."""

    def __init__(self, normalized_shape, eps=1e-5, seed=None):
        self.eps = eps
        shape = np.ones(normalized_shape, dtype=np.float32)
        self.weight = Parameter(F._native_from_numpy(shape, True))
        self.bias = Parameter(F._native_from_numpy(np.zeros(normalized_shape, dtype=np.float32), True))

    def forward(self, x):
        return F.layer_norm(x, self.weight.tensor, self.bias.tensor, self.eps)


class RMSNorm(Module):
    """RMSNorm over the last dim (scale-only, modern LLM style)."""

    def __init__(self, normalized_shape, eps=1e-6, seed=None):
        self.eps = eps
        self.weight = Parameter(
            F._native_from_numpy(np.ones(normalized_shape, dtype=np.float32), True)
        )

    def forward(self, x):
        return F.rms_norm(x, self.weight.tensor, self.eps)


class Embedding(Module):
    """Lookup table: rows of `weight` selected by integer indices."""

    def __init__(self, num_embeddings, embedding_dim, seed=None):
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((num_embeddings, embedding_dim)) * 0.05
        self.weight = Parameter(F._native_from_numpy(w.astype(np.float32), True))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices):
        return F.embedding(indices, self.weight.tensor)


class Dropout(Module):
    """Inverted dropout; `self.training` toggles (train()/eval())."""

    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.training = True
        self.seed = seed

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        if self.seed is not None:
            np.random.seed(self.seed)
        return F.dropout(x, self.p, self.training)


class _Activation(Module):
    def __init__(self, fn):
        self._fn = fn

    def forward(self, x):
        return self._fn(x)


def ReLU():
    return _Activation(F.relu)


def GELU():
    return _Activation(F.gelu)


def SiLU():
    return _Activation(F.silu)


def Tanh():
    return _Activation(F.tanh)


def Sigmoid():
    return _Activation(F.sigmoid)


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class Softmax(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, self.dim)
