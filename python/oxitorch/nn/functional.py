"""Functional API mirroring `torch.nn.functional` (Phase 3 subset).

Every function here is autograd-connected: it composes `_native` ops that
record into the graph, so `backward()` works through any composition.
"""

import numpy as np

from .. import _native


def _native_from_numpy(arr, requires_grad=False):
    """Build a graph-connected tensor from a numpy array."""
    return _native.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), requires_grad)


def _param(arr):
    return _native_from_numpy(arr, requires_grad=True)


# ---- linear & matmul --------------------------------------------------------

def linear(x, weight, bias=None):
    """`x @ weight.T + bias` with `weight` of shape `(out, in)` (torch layout).

    `x` may be `(in,)`, `(batch, in)`, or any higher-rank `(..., in)` tensor
    (minibatches of vectors or of feature maps) — matching `torch.nn.Linear`.
    Higher-rank inputs are folded to 2-D for the GEMM and unfolded after,
    so every rank is one fused kernel (and one backward) deep.
    """
    out_features = weight.shape[0]
    if x.ndim == 1:
        out = (x.reshape([1, -1]) @ weight.transpose()).reshape([out_features])
    elif x.ndim == 2:
        out = x @ weight.transpose()
    else:
        # Fold leading dims: (..., in) -> (N, in) @ (out, in).T -> (N, out).
        lead = x.shape[:-1]
        n = 1
        for d in lead:
            n *= d
        out = x.reshape([n, x.shape[-1]]) @ weight.transpose()
        out = out.reshape([*lead, out_features])
    if bias is not None:
        out = out + bias  # broadcasts across the leading dims
    return out


def bmm(a, b):
    """Batched matrix multiply `(b, m, k) @ (b, k, n) -> (b, m, n)`.

    Autograd-connected: gradients flow to both operands, summed over the
    batch exactly like torch's `torch.bmm`.
    """
    return a.bmm(b)


# ---- conv & pooling (Phase 4) -------------------------------------------------

def _pad2d(x, ph, pw):
    """Zero-pad the last two dims of `(...)` NCHW input, keeping the graph.

    Built with `scatter` into a zero canvas: the scatter VJP routes gradient
    exactly to the written (center) region, so padding is transparent to
    autograd. Pure views cannot express padding (narrow+add reallocates).
    """
    if ph == 0 and pw == 0:
        return x
    n, c, h, w = x.shape
    out_h, out_w = h + 2 * ph, w + 2 * pw
    canvas = _native.zeros([n * c * out_h * out_w])
    offs = [
        ni * c * out_h * out_w + ci * out_h * out_w + (i + ph) * out_w + (j + pw)
        for ni in range(n)
        for ci in range(c)
        for i in range(h)
        for j in range(w)
    ]
    return canvas.scatter(offs, x.reshape([n * c * h * w])).reshape([n, c, out_h, out_w])


def _window_offsets(shape4, kh, kw, oh, ow, sh, sw):
    """Flat storage offsets of every kernel window element (im2col order).

    For each `(n, c)` plane, walks output positions `(i, j)` and kernel
    positions `(ki, kj)` — so the gathered vector per plane is
    `[oh*ow*kh*kw]` in `[i, j, ki, kj]` order, the layout im2col needs.
    """
    n, c, h, w = shape4
    offs = []
    for ni in range(n):
        for ci in range(c):
            base = (ni * c + ci) * h * w
            for i in range(oh):
                for j in range(ow):
                    top = base + (i * sh) * w + (j * sw)
                    for ki in range(kh):
                        row = top + ki * w
                        for kj in range(kw):
                            offs.append(row + kj)
    return offs


def conv2d(x, weight, bias=None, stride=1, padding=0):
    """2-D convolution (cross-correlation, torch semantics) via im2col + bmm.

    `x` is `(N, C, H, W)`; `weight` is `(out, C, kh, kw)`; optional `bias`
    is `(out,)`. `stride`/`padding` are ints applied to both spatial dims.

    Composition, all autograd-connected: zero-pad via scatter, gather every
    kernel window with one `index_select`, reshape to
    `(N, oh*ow, C*kh*kw)`, and multiply per-image by the unfolded weights —
    `(N, C*kh*kw, out)` bmm — so gradients reach `x`, `weight`, and `bias`
    through the standard index_select/reshape/bmm VJPs (accumulating where
    windows overlap, which is the correct convolution backward).
    """
    if x.ndim != 4:
        raise ValueError(f"conv2d expects a 4-D NCHW input, got ndim {x.ndim}")
    n, c, h, w = x.shape
    out_ch, in_ch, kh, kw = weight.shape
    if in_ch != c:
        raise ValueError(
            f"conv2d weight expects {c} input channels, got {in_ch}"
        )
    sh = sw = int(stride)
    ph = pw = int(padding)
    xp = _pad2d(x, ph, pw)
    hp, wp = h + 2 * ph, w + 2 * pw
    oh = (hp - kh) // sh + 1
    ow = (wp - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f"kernel {(kh, kw)} with stride {stride} does not fit padded "
            f"input ({hp}, {wp})"
        )

    offs = _window_offsets(xp.shape, kh, kw, oh, ow, sh, sw)
    windows = xp.reshape([n * c * hp * wp]).index_select(0, offs)
    windows = windows.reshape([n, c, oh, ow, kh * kw])
    # Reorder to channel-major kernel layout: [i, j, c, ki*kj] per image.
    cols = windows.permute([0, 2, 3, 1, 4]).reshape([n, oh * ow, c * kh * kw])
    wmat = weight.reshape([out_ch, c * kh * kw])
    out = cols.bmm(wmat.transpose().broadcast_to([n, c * kh * kw, out_ch]))
    out = out.permute([0, 2, 1]).reshape([n, out_ch, oh, ow])
    if bias is not None:
        out = out + bias.reshape([1, out_ch, 1, 1])
    return out


def max_pool2d(x, kernel_size, stride=None):
    """Max pooling over the last two dims, via window gather + Max reduce.

    `x` is `(N, C, H, W)`; `kernel_size`/`stride` are ints or `(kh, kw)`
    tuples (stride defaults to the kernel, torch style). Windows are
    gathered with one `index_select` into `(N, C, oh, ow, kh*kw)` and the
    max runs over the kernel axis — gradients route to each window's argmax
    through the reduce VJP (duplicates split, summing to 1).
    """
    if x.ndim != 4:
        raise ValueError(f"max_pool2d expects a 4-D NCHW input, got ndim {x.ndim}")
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    if stride is None:
        sh, sw = kh, kw
    else:
        sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    n, c, h, w = x.shape
    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError(
            f"pool kernel {(kh, kw)} with stride {(sh, sw)} does not fit "
            f"input ({h}, {w})"
        )
    offs = _window_offsets(x.shape, kh, kw, oh, ow, sh, sw)
    windows = x.reshape([n * c * h * w]).index_select(0, offs)
    windows = windows.reshape([n, c, oh, ow, kh * kw])
    return windows.max(-1, False)


# ---- activations ------------------------------------------------------------

def relu(x):
    return x.apply("relu")


def sigmoid(x):
    return x.apply("sigmoid")


def tanh(x):
    return x.apply("tanh")


def gelu(x):
    """0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) — tanh approximation."""
    c = np.sqrt(2.0 / np.pi)
    x32 = x * x
    inner = (x + x * x32 * 0.044715) * float(c)
    return 0.5 * x * (1.0 + inner.apply("tanh"))


def silu(x):
    """SiLU / swish: `x * sigmoid(x)`."""
    return x * x.apply("sigmoid")


def leaky_relu(x, negative_slope=0.01):
    return x.apply("relu") + (x * float(negative_slope)).apply("neg").apply("relu").apply("neg")


def softmax(x, dim=-1):
    """Numerically stable softmax along `dim`."""
    m = x.max(dim, True)
    e = (x - m).apply("exp")
    return e / e.sum(dim, True)


def log_softmax(x, dim=-1):
    """`log_softmax = x - max - log(sum(exp(x - max)))`, stable."""
    m = x.max(dim, True)
    z = x - m
    return z - z.apply("exp").sum(dim, True).apply("log")


# ---- normalization ----------------------------------------------------------

def layer_norm(x, weight, bias, eps=1e-5):
    """LayerNorm over the last dim with affine `weight`/`bias`."""
    mu = x.mean(-1, True)
    centered = x - mu
    var = (centered * centered).mean(-1, True)
    inv = (var + float(eps)).apply("rsqrt")
    return centered * inv * weight + bias


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm over the last dim: `x / rms(x) * weight`."""
    ms = (x * x).mean(-1, True)
    inv = (ms + float(eps)).apply("rsqrt")
    return x * inv * weight


# ---- embedding & dropout -----------------------------------------------------

def embedding(indices, weight):
    """`weight` is `(num_embeddings, dim)`; `indices` is a flat list of i64."""
    return weight.index_select(0, indices)


class DropoutMode:
    """Holds the current global dropout state (module-level in torch)."""
    enabled = True


def dropout(x, p=0.5, training=True):
    """Inverted dropout: zero with prob p, scale survivors by 1/(1-p).

    Uses a seeded RNG call per invocation (numpy), applied through
    differentiable mul/mask ops so gradients flow. Eval mode is identity.
    """
    if not training or p == 0.0:
        return x
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout probability {p} out of range [0, 1)")
    mask = (np.random.random_sample(x.shape) >= p).astype(np.float32)
    mask /= 1.0 - p
    return x * _native_from_numpy(mask)


# ---- losses -------------------------------------------------------------------

def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits, targets):
    """`targets` is a flat list of int class indices; mean over batch.

    `-log_softmax(logits)[i, target[i]]` averaged. Built from differentiable
    pieces: pick target columns via a one-hot matmul (`onehot @ logits`
    gathers one entry per row and keeps autograd intact).
    """
    ls = log_softmax(logits, -1)
    n, classes = logits.shape
    if len(targets) != n:
        raise ValueError(f"{len(targets)} targets for batch of {n}")
    onehot = np.zeros((n, classes), dtype=np.float32)
    onehot[np.arange(n), targets] = 1.0
    # picked[i] = sum_c onehot[i,c] * ls[i,c] = ls[i, target[i]] — elementwise
    # mul + sum over the graph, so gradients flow to `ls` (onehot is constant).
    picked = (_native_from_numpy(onehot) * ls).sum(-1)
    return picked.mean().apply("neg")
