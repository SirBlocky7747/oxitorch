"""Rust-backed optimizer tests: the update math still matches the closed
forms (Phase 4 contract), plus the new guarantees the Rust step brings —
object identity preserved across steps, no numpy in the loop, lazy per-
parameter state, scheduler lr mutation propagating into the engine, and
state continuity across externally rebound parameter tensors.
"""

import numpy as np

import oxitorch as ox
import oxitorch.nn as nn
import oxitorch.optim as op
from oxitorch.nn import functional as F


def _t(arr, requires_grad=False):
    return ox.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), requires_grad=requires_grad
    )


def _regression_model(w0, seed=0):
    layer = nn.Linear(3, 1, bias=False, seed=seed)
    layer.weight.tensor = _t(w0, True)
    return layer


def _regression_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    y = rng.standard_normal((4, 1)).astype(np.float32)
    return x, y


def test_adamw_matches_numpy_closed_form():
    """The Rust update reproduces the AdamW closed form over real grads."""
    x, y = _regression_data()
    rng = np.random.default_rng(1)
    w0 = (rng.standard_normal((1, 3)) * 0.5).astype(np.float32)
    lr, b1, b2, eps, wd = 0.05, 0.9, 0.999, 1e-8, 0.1
    layer = _regression_model(w0)
    opt = op.AdamW(layer.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    w = w0.copy()
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    for step in range(1, 4):
        pred = w @ x.T
        g = ((2 * (pred - y.T)) @ x / x.shape[0]).astype(np.float32)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**step)
        vh = v / (1 - b2**step)
        w = w - lr * mh / (np.sqrt(vh) + eps) - lr * wd * w

        layer.zero_grad()
        loss = F.mse_loss(layer(_t(x)), _t(y))
        loss.backward()
        opt.step()
        np.testing.assert_allclose(layer.weight.tensor.numpy(), w, rtol=1e-4, atol=1e-6)


def test_sgd_momentum_matches_numpy_closed_form():
    x, y = _regression_data()
    rng = np.random.default_rng(2)
    w0 = (rng.standard_normal((1, 3)) * 0.5).astype(np.float32)
    lr, mom = 0.1, 0.9
    layer = _regression_model(w0)
    opt = op.Sgd(layer.parameters(), lr=lr, momentum=mom)

    w = w0.copy()
    buf = np.zeros_like(w)
    for _ in range(3):
        pred = w @ x.T
        g = ((2 * (pred - y.T)) @ x / x.shape[0]).astype(np.float32)
        buf = mom * buf + g
        w = w - lr * buf

        layer.zero_grad()
        loss = F.mse_loss(layer(_t(x)), _t(y))
        loss.backward()
        opt.step()
        np.testing.assert_allclose(layer.weight.tensor.numpy(), w, rtol=1e-4, atol=1e-6)


def test_step_preserves_tensor_identity():
    """The engine rewrites values in place: the same Python Tensor objects
    (and therefore the Parameter wrappers) stay bound across steps."""
    x, y = _regression_data()
    layer = nn.Linear(3, 1, seed=0)
    weight_obj = layer.weight.tensor
    bias_obj = layer.bias.tensor
    weight_before = weight_obj.numpy().copy()
    opt = op.AdamW(layer.parameters(), lr=0.05)

    for _ in range(3):
        layer.zero_grad()
        F.mse_loss(layer(_t(x)), _t(y)).backward()
        opt.step()
        assert layer.weight.tensor is weight_obj
        assert layer.bias.tensor is bias_obj
    # And the values did move:
    assert not np.allclose(weight_before, weight_obj.numpy())


def test_lazy_state_param_without_grad_gets_fresh_buffers():
    """A parameter first producing a grad on step 2 gets fresh state (and a
    step counter of t=1), exactly as if the optimizer had just been built —
    torch's per-parameter state semantics."""
    g = np.array([0.5, -0.5], dtype=np.float32)  # signs +1, -1

    # A fresh AdamW's first step moves each element by exactly lr*sign(g):
    # bias-corrected first moment is g, second moment g², update lr*g/(|g|+eps).
    w_fresh = _t([1.0, 2.0], True)
    fresh = op.AdamW([w_fresh], lr=0.1, weight_decay=0.0)
    w_fresh.grad = _t(g)
    fresh.step()
    one_step = w_fresh.numpy().copy()

    # Now the lazy scenario: the param has no grad on step 1, gets one on 2.
    w_lazy = _t([1.0, 2.0], True)
    w_other = _t([1.0, 2.0], True)
    lazy = op.AdamW([w_other, w_lazy], lr=0.1, weight_decay=0.0)
    w_other.grad = _t(g)
    lazy.step()  # w_lazy skipped — no state, no counter advance
    w_other.grad = _t(g)
    w_lazy.grad = _t(g)
    lazy.step()  # w_lazy's first real step

    np.testing.assert_allclose(
        w_lazy.numpy(), one_step, rtol=1e-5, atol=1e-7
    ), "lazy param's first real step must equal a fresh optimizer's step 1"


def test_scheduler_lr_mutation_reaches_engine():
    """`optimizer.lr = x` (what schedulers do) must affect the next Rust step."""
    w = _t([1.0], True)
    opt = op.Sgd([w], lr=0.1)
    opt.lr = 0.5  # scheduler-style mutation
    assert opt.lr == 0.5

    w.grad = _t([1.0])
    opt.step()
    np.testing.assert_allclose(w.numpy(), [0.5], rtol=1e-6)


def test_state_survives_parameter_rebinding():
    """The engine keys state by position, and each step reads the *current*
    tensors — so a user rebinding `param.tensor` keeps momentum continuity."""
    g = np.array([1.0], dtype=np.float32)
    w = _t([1.0], True)
    opt = op.Sgd([w], lr=0.1, momentum=0.9)
    w.grad = _t(g)
    opt.step()  # buf = 1.0, w = 0.9

    # External rebinding (e.g. load_state_dict-style): new Tensor object.
    w2 = _t([0.9], True)
    opt.params = [w2]  # raw tensor, new object — same list position
    w2.grad = _t(g)
    opt.step()  # buf = 0.9*1.0 + 1.0 = 1.9, w = 0.9 - 0.19 = 0.71
    np.testing.assert_allclose(w2.numpy(), [0.71], rtol=1e-5, atol=1e-7)


def test_training_loop_reaches_solution():
    """End-to-end: AdamW trains a small MLP to fit a regression target."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((32, 4)).astype(np.float32)
    target_w = rng.standard_normal((4, 2)).astype(np.float32)
    y = x @ target_w + 0.1 * rng.standard_normal((32, 2)).astype(np.float32)

    model = nn.Sequential(nn.Linear(4, 16, seed=0), nn.ReLU(), nn.Linear(16, 2, seed=1))
    opt = op.AdamW(model.parameters(), lr=0.01)

    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = F.mse_loss(model(_t(x)), _t(y))
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.numpy())
    final = float(loss.numpy())
    assert np.isfinite(final)
    assert final < first * 0.1, f"loss {first} -> {final}"
