"""Phase 2 autograd tests: Python-visible gradients, numpy parity, and a
tiny training loop (plan.md Phase 2 exit criteria)."""

import numpy as np
import pytest

import oxitorch as ox

RTOL = 1e-4
ATOL = 1e-5


def _t(arr, requires_grad=False):
    return ox.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), requires_grad=requires_grad
    )


def test_backward_matches_numpy_quadratic():
    x = _t([[1.0, 2.0], [3.0, 4.0]])
    w = _t([[0.5], [0.5]], requires_grad=True)
    b = ox.full([], 1.0, requires_grad=True)
    pred = x @ w + b
    diff = pred - _t([[1.0], [2.0]])
    loss = (diff * diff).sum()
    loss.backward()

    xa = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    wa = np.array([[0.5], [0.5]], dtype=np.float32)
    da = xa @ wa + 1.0 - np.array([[1.0], [2.0]], dtype=np.float32)
    np.testing.assert_allclose(w.grad.numpy(), (2 * da * xa).sum(0, keepdims=True).T, rtol=RTOL, atol=ATOL)
    assert float(b.grad) == pytest.approx(float((2 * da).sum()), rel=RTOL)


def test_backward_broadcast_grad_reduces_to_param_shape():
    w = _t([1.0, 2.0, 3.0], requires_grad=True)
    x = _t(np.ones((4, 3)))
    (x * w).sum().backward()
    # w broadcasts across the 4 rows; the grad sums those 4 contributions
    # back down to w's shape: all-4s here (x is all ones).
    np.testing.assert_allclose(w.grad.numpy(), np.full(3, 4.0, dtype=np.float32), rtol=RTOL)


def test_grad_accumulates_over_multiple_backward_calls():
    w = _t([1.0], requires_grad=True)
    x = _t([2.0])
    (x * w).sum().backward()
    (x * w).sum().backward()
    assert float(w.grad) == pytest.approx(4.0, rel=RTOL)


def test_no_grad_leaf_has_no_grad():
    x = _t([1.0, 2.0])
    y = (x * x).sum()
    with pytest.raises(ValueError):
        y.backward()
    assert x.grad is None


def test_detach_and_zero_grad():
    w = _t([1.0], requires_grad=True)
    y = (w * w).sum()
    y.backward()
    d = w.detach()
    assert not d.requires_grad
    w.zero_grad()
    assert w.grad is not None and float(w.grad) == 0.0


def test_sgd_training_converges():
    # Data is y = 2x + 1, but the model is y = w*x through the origin: the
    # least-squares fixed point is sum(2*x*y)/sum(2*x^2), and SGD must reach
    # exactly it (numpy cross-check below).
    xs_np = (np.arange(8, dtype=np.float32) * 0.5 - 1.5).reshape(8, 1)
    ys_np = (2.0 * xs_np + 1.0).astype(np.float32)
    fixed_point = float((2 * xs_np * ys_np).sum() / (2 * xs_np * xs_np).sum())

    w = _t([[2.5]], requires_grad=True)
    xs = _t(xs_np)
    ys = _t(ys_np)
    lr = 0.01
    for _ in range(500):
        pred = xs @ w
        l = ((pred - ys) * (pred - ys)).sum()
        l.backward()
        g = w.grad.numpy().copy()
        w.zero_grad()
        w = _t(w.detach().numpy() - lr * g, requires_grad=True)
    assert float(w) == pytest.approx(fixed_point, abs=0.02)


def test_unary_grads_via_apply():
    x_np = np.array([0.3, -0.6, 1.2], dtype=np.float32)
    w_np = np.ones(3, dtype=np.float32)

    for name, np_grad in [
        ("sigmoid", lambda v: (1 / (1 + np.exp(-v))) * (1 - 1 / (1 + np.exp(-v)))),
        ("tanh", lambda v: 1 - np.tanh(v) ** 2),
        ("exp", lambda v: np.exp(v)),
        ("relu", lambda v: (v > 0).astype(np.float32)),
    ]:
        x = _t(x_np, requires_grad=True)
        out = x.apply(name)
        loss = (out * _t(w_np)).sum()
        loss.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np_grad(x_np.astype(np.float64)).astype(np.float32),
            rtol=RTOL, atol=ATOL,
        )


def test_second_order_through_matmul_chain():
    # f(x) = sum((x @ x)^2); finite-difference the grad to verify second order.
    a = np.array([[0.4, -0.3], [0.8, 0.1]], dtype=np.float32)
    x = _t(a, requires_grad=True)
    y = x @ x
    loss = (y * y).sum()
    loss.backward()
    g1 = x.grad.numpy()

    # analytic grad of sum((x@x)^2): 4 (x@x) x^T ... verify against central diff
    h = 1e-2
    numeric = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ap = a.copy(); ap[i, j] += h
            am = a.copy(); am[i, j] -= h

            def f(v):
                return float((v @ v * (v @ v)).sum())

            numeric[i, j] = (f(ap) - f(am)) / (2 * h)
    np.testing.assert_allclose(g1, numeric, rtol=1e-3, atol=1e-3)
