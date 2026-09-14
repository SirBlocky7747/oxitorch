"""Phase 4 conv/pooling tests: `conv2d` and `max_pool2d` via im2col built
from graph-connected primitives (index_select + reshape/permute + bmm/max).

Values are checked against a direct NumPy loop implementation across
stride/padding/kernel configs; gradients are verified by central finite
differences on x/weight/bias; and a small conv→pool→linear net trains,
matching the same training run reimplemented in NumPy."""

import numpy as np
import pytest

import oxitorch as ox
import oxitorch.nn as nn
from oxitorch.nn import functional as F

RTOL = 1e-4
ATOL = 1e-4


def _t(arr, requires_grad=False):
    return ox.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), requires_grad=requires_grad
    )


def _conv_np(x, w, b=None, stride=1, padding=0):
    """Direct NumPy reference (cross-correlation, torch semantics)."""
    n, c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    oh = (xp.shape[2] - kh) // stride + 1
    ow = (xp.shape[3] - kw) // stride + 1
    out = np.zeros((n, o, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            patch = xp[:, :, i * stride : i * stride + kh, j * stride : j * stride + kw]
            # (n, c, kh, kw) x (o, c, kh, kw) -> (n, o) via einsum
            out[:, :, i, j] = np.einsum("nckl,ockl->no", patch, w)
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def _pool_np(x, k, s):
    n, c, h, w = x.shape
    oh = (h - k) // s + 1
    ow = (w - k) // s + 1
    out = np.zeros((n, c, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * s : i * s + k, j * s : j * s + k].max(axis=(2, 3))
    return out


@pytest.mark.parametrize(
    "stride,padding",
    [(1, 0), (1, 1), (2, 0), (2, 1), (3, 1), (1, 2)],
)
def test_conv2d_values_match_numpy(stride, padding):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    got = F.conv2d(_t(x), _t(w), _t(b), stride=stride, padding=padding)
    want = _conv_np(x, w, b, stride=stride, padding=padding)
    np.testing.assert_allclose(got.numpy(), want, rtol=RTOL, atol=ATOL)


def test_conv2d_no_bias_matches():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 5, 5)).astype(np.float32)
    w = rng.standard_normal((3, 2, 3, 3)).astype(np.float32)
    got = F.conv2d(_t(x), _t(w), stride=2)
    want = _conv_np(x, w, stride=2)
    np.testing.assert_allclose(got.numpy(), want, rtol=RTOL, atol=ATOL)


def test_conv2d_1x1_kernel_is_per_pixel_matmul():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 5, 4, 4)).astype(np.float32)
    w = rng.standard_normal((6, 5, 1, 1)).astype(np.float32)
    got = F.conv2d(_t(x), _t(w))
    want = np.einsum("nchw,oc->nohw", x, w.reshape(6, 5))
    np.testing.assert_allclose(got.numpy(), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kernel,stride", [(2, 2), (3, 1), (3, 2), (2, 1)])
def test_max_pool2d_values_match_numpy(kernel, stride):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    got = F.max_pool2d(_t(x), kernel, stride)
    want = _pool_np(x, kernel, stride)
    np.testing.assert_array_equal(got.numpy(), want)


def test_max_pool2d_stride_defaults_to_kernel():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 2, 6, 6)).astype(np.float32)
    got = F.max_pool2d(_t(x), 2)
    np.testing.assert_array_equal(got.numpy(), _pool_np(x, 2, 2))
    assert got.numpy().shape == (1, 2, 3, 3)


def test_conv2d_grads_match_finite_differences():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    stride, padding = 2, 1
    seed = rng.standard_normal((2, 4, 4, 4)).astype(np.float32)

    xt, wt, bt = _t(x, True), _t(w, True), _t(b, True)
    loss = (F.conv2d(xt, wt, bt, stride=stride, padding=padding) * _t(seed)).sum()
    loss.backward()

    h = 1e-2

    def loss_of(xa=None, wa=None, ba=None):
        xa = x if xa is None else xa
        wa = w if wa is None else wa
        ba = b if ba is None else ba
        return float((_conv_np(xa, wa, ba, stride, padding) * seed).sum())

    for ni, ci, i, j in [(0, 0, 0, 0), (1, 2, 5, 7), (0, 1, 3, 4)]:
        xp, xm = x.copy(), x.copy()
        xp[ni, ci, i, j] += h
        xm[ni, ci, i, j] -= h
        num = (loss_of(xa=xp) - loss_of(xa=xm)) / (2 * h)
        assert xt.grad.numpy()[ni, ci, i, j] == pytest.approx(num, rel=1e-2, abs=1e-3)

    for oi, ci, ki, kj in [(0, 0, 0, 0), (3, 2, 1, 2)]:
        wp, wm = w.copy(), w.copy()
        wp[oi, ci, ki, kj] += h
        wm[oi, ci, ki, kj] -= h
        num = (loss_of(wa=wp) - loss_of(wa=wm)) / (2 * h)
        assert wt.grad.numpy()[oi, ci, ki, kj] == pytest.approx(num, rel=1e-2, abs=1e-3)

    bp, bm = b.copy(), b.copy()
    bp[2] += h
    bm[2] -= h
    num = (loss_of(ba=bp) - loss_of(ba=bm)) / (2 * h)
    assert bt.grad.numpy()[2] == pytest.approx(num, rel=1e-2, abs=1e-3)


def test_max_pool2d_grad_routes_to_argmax():
    # One decisive window: x = [[1, 2], [3, 4]] -> pool picks 4; grad lands
    # only on the 4.
    x = _t([[[
        [1.0, 2.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]]], requires_grad=True)
    out = F.max_pool2d(x, 2)
    out.backward(_t([[[[1.0, 1.0], [1.0, 1.0]]]]))
    got = x.grad.numpy()[0, 0]
    # Window (0,0) max is 4 at (1,1); all-zero windows route to their first
    # element (argmax picks first index on ties): (0,1)->(0,2), (1,0)->(2,0),
    # (1,1)->(2,2).
    assert got[1, 1] == pytest.approx(1.0)
    assert got[0, 2] == pytest.approx(1.0)
    assert got[2, 0] == pytest.approx(1.0)
    assert got[2, 2] == pytest.approx(1.0)
    assert got[0, 0] + got[0, 1] + got[1, 0] + got[2, 1] + got[1, 2] == pytest.approx(0.0)


def test_max_pool2d_grad_ties_go_to_first_index():
    # A window whose maximum appears twice routes the entire gradient to the
    # FIRST tied index (matches our Max reduce's argmax convention, and the
    # common "first argmax wins" backward variant).
    x = _t([[[
        [5.0, 5.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]]], requires_grad=True)
    out = F.max_pool2d(x, 2)  # windows (0,0), (0,1), (1,0), (1,1)
    out.backward(_t([[[[2.0, 1.0], [1.0, 1.0]]]]))
    got = x.grad.numpy()[0, 0]
    # Window (0,0) has max 5 at both (0,0) and (0,1): all 2.0 to (0,0).
    assert got[0, 0] == pytest.approx(2.0)
    assert got[0, 1] == pytest.approx(0.0)


def test_pad2d_grad_flows_to_center_only():
    x = _t(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), requires_grad=True)
    padded = F._pad2d(x, 1, 1)
    assert padded.numpy().shape == (1, 1, 6, 6)
    np.testing.assert_array_equal(padded.numpy()[0, 0, 1:5, 1:5], np.arange(16, dtype=np.float32).reshape(4, 4))
    padded.sum().backward()
    grad = x.grad.numpy()[0, 0]
    np.testing.assert_array_equal(grad, np.ones((4, 4), dtype=np.float32))
    # Padding ring itself is zero.
    assert padded.numpy()[0, 0, 0, 0] == 0.0


def test_conv_layer_init_and_state_dict():
    conv = nn.Conv2d(3, 8, 3, stride=2, padding=1, seed=1)
    assert conv.weight.tensor.shape == [8, 3, 3, 3]
    assert conv.bias.tensor.shape == [8]
    # torch Conv2d init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)), fan_in = c*kh*kw.
    bound = 1.0 / np.sqrt(3 * 3 * 3)
    w = conv.weight.tensor.numpy()
    assert w.min() >= -bound and w.max() <= bound
    assert np.abs(conv.bias.tensor.numpy()).max() == 0.0

    sd = conv.state_dict()
    assert set(sd) == {"weight", "bias"}
    conv2 = nn.Conv2d(3, 8, 3, stride=2, padding=1, seed=99)
    conv2.load_state_dict(sd)
    np.testing.assert_array_equal(conv2.weight.tensor.numpy(), sd["weight"])
    np.testing.assert_array_equal(conv2.bias.tensor.numpy(), sd["bias"])


def test_conv_layer_forward_matches_functional():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 3, 8, 8)).astype(np.float32)
    conv = nn.Conv2d(3, 4, 3, seed=7)
    out = conv(_t(x))
    want = _conv_np(x, conv.weight.tensor.numpy(), conv.bias.tensor.numpy())
    np.testing.assert_allclose(out.numpy(), want, rtol=RTOL, atol=ATOL)


def test_cnn_training_loop_matches_numpy_reference():
    """End-to-end: conv → relu → pool → flatten → linear, full-batch SGD.

    The same network and data are reimplemented in NumPy and stepped with
    identical SGD; after a few steps the losses must agree — proving the
    graph's forward AND backward agree with the reference throughout.
    """
    rng = np.random.default_rng(8)
    x_np = rng.standard_normal((4, 2, 8, 8)).astype(np.float32)
    y_np = rng.standard_normal((4, 3)).astype(np.float32)

    # Shared init values (kaiming bound known: fan_in 2*3*3 = 18).
    bound = 1.0 / np.sqrt(18)
    w1 = rng.uniform(-bound, bound, (2, 2, 3, 3)).astype(np.float32)
    b1 = np.zeros(2, dtype=np.float32)
    fb = 1.0 / np.sqrt(32)  # linear fan_in: 2 channels * 4x4 pooled
    w2 = rng.uniform(-fb, fb, (3, 32)).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    lr, steps = 0.05, 12

    # --- oxitorch network ---
    conv = nn.Conv2d(2, 2, 3, padding=1, seed=0)
    conv.weight.tensor = _t(w1, True)
    conv.bias.tensor = _t(b1, True)
    fc = nn.Linear(32, 3, seed=0)
    fc.weight.tensor = _t(w2, True)
    fc.bias.tensor = _t(b2, True)

    def ox_step(xa, ya):
        xt = _t(xa, True)
        h = F.max_pool2d(conv(xt).apply("relu"), 2)          # (4,2,4,4)
        flat = h.reshape([4, 32])
        pred = F.linear(flat, fc.weight.tensor, fc.bias.tensor)
        loss = F.mse_loss(pred, _t(ya))
        loss.backward()
        for p in (conv.weight, conv.bias, fc.weight, fc.bias):
            p.tensor = _t(p.tensor.numpy() - lr * p.grad.numpy(), True)
        conv.zero_grad()
        fc.zero_grad()
        return float(loss.numpy())

    ox_losses = [ox_step(x_np, y_np) for _ in range(steps)]

    # --- NumPy reference network ---
    w1r, b1r, w2r, b2r = w1.copy(), b1.copy(), w2.copy(), b2.copy()

    def np_forward(xa, wa, ba, wb, bb):
        hp = np.pad(xa, ((0, 0), (0, 0), (1, 1), (1, 1)))
        conv_out = _conv_np(hp, wa, ba)  # reuse direct conv (padding already applied)
        act = np.maximum(conv_out, 0.0)
        pooled = _pool_np(act, 2, 2)
        flat = pooled.reshape(4, 32)
        pred = flat @ wb.T + bb
        return pred, flat

    def np_loss_grad(xa, ya, wa, ba, wb, bb):
        pred, flat = np_forward(xa, wa, ba, wb, bb)
        diff = pred - ya
        loss = float((diff * diff).mean())
        dpred = 2.0 * diff / diff.size
        gwb = dpred.T @ flat
        gbb = dpred.sum(0)
        gflat = dpred @ wb
        gpooled = gflat.reshape(4, 2, 4, 4)
        act = np.maximum(_conv_np(np.pad(xa, ((0, 0), (0, 0), (1, 1), (1, 1))), wa, ba), 0.0)
        # pool backward: route each pooled grad to its window argmax (ties
        # to the first index, matching the engine's Max reduce).
        gact = np.zeros_like(act)
        for i in range(4):
            for j in range(4):
                window = act[:, :, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2]
                mx = window.max(axis=(2, 3), keepdims=True)
                mask = (window == mx).astype(np.float32)
                mask /= mask.sum(axis=(2, 3), keepdims=True)
                gact[:, :, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2] = gpooled[:, :, i, j, None, None] * mask
        gpre = gact * (act > 0)
        xp = np.pad(xa, ((0, 0), (0, 0), (1, 1), (1, 1)))
        gwa = np.zeros_like(wa)
        gba = np.zeros_like(ba)
        for i in range(8):
            for j in range(8):
                patch = xp[:, :, i : i + 3, j : j + 3]
                gwa += np.einsum("nckl,no->ockl", patch, gpre[:, :, i, j])
                gba += gpre[:, :, i, j].sum(0)
        return loss, gwa, gba, gwb, gbb

    np_losses = []
    for _ in range(steps):
        loss, gwa, gba, gwb, gbb = np_loss_grad(x_np, y_np, w1r, b1r, w2r, b2r)
        np_losses.append(loss)
        w1r -= lr * gwa
        b1r -= lr * gba
        w2r -= lr * gwb
        b2r -= lr * gbb

    assert len(ox_losses) == steps
    for i, (a, b_) in enumerate(zip(ox_losses, np_losses)):
        assert a == pytest.approx(b_, rel=1e-3, abs=1e-4), f"step {i}: {a} vs {b_}"
    # And training actually reduces loss.
    assert ox_losses[-1] < ox_losses[0]


def test_conv_weight_grad_accumulates_overlapping_windows():
    # A 1x1-kernel conv on a constant input: every output element shares the
    # same weight, so its grad is the sum of all upstream grads.
    x = _t(np.full((1, 1, 2, 2), 3.0, dtype=np.float32), requires_grad=True)
    w = _t(np.ones((1, 1, 1, 1), dtype=np.float32), requires_grad=True)
    out = F.conv2d(x, w)
    out.backward(_t(np.full((1, 1, 2, 2), 2.0, dtype=np.float32)))
    # grad_w = sum over the 4 windows of (upstream 2.0 * x 3.0) = 24.
    assert w.grad.numpy()[0, 0, 0, 0] == pytest.approx(24.0)
    # 1x1 stride-1 windows are disjoint: each x element feeds exactly one
    # output, so grad_x[i,j] = up[i,j] * w = 2.
    assert x.grad.numpy()[0, 0, 0, 0] == pytest.approx(2.0)
