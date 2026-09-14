"""Batched matmul (`bmm`) and batched `Linear` tests (Phase 3/4 seam).

Values are checked against `numpy.matmul`; gradients against the closed
forms `gA = g @ B^T`, `gB = A^T @ g` per batch, plus gradcheck-style
finite differences through the fold/unfold `Linear` path.
"""

import numpy as np
import pytest

import oxitorch as ox
import oxitorch.nn as nn
import oxitorch.nn.functional as F


def _t(arr, requires_grad=False):
    return ox.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), requires_grad)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---- bmm values & gradients -----------------------------------------------------

def test_bmm_value_matches_numpy():
    rng = _rng(1)
    a = _t(rng.standard_normal((3, 4, 5)))
    b = _t(rng.standard_normal((3, 5, 2)))
    out = a.bmm(b)
    assert out.shape == [3, 4, 2]
    assert np.allclose(out.numpy(), np.matmul(a.numpy(), b.numpy()), atol=1e-5)


def test_bmm_grads_match_closed_form():
    rng = _rng(2)
    A = rng.standard_normal((3, 4, 5)).astype(np.float32)
    B = rng.standard_normal((3, 5, 2)).astype(np.float32)
    a, b = _t(A, True), _t(B, True)
    out = a.bmm(b)
    g = rng.standard_normal((3, 4, 2)).astype(np.float32)
    (out * _t(g)).sum().backward()
    gA = np.matmul(g, np.transpose(B, (0, 2, 1)))
    gB = np.matmul(np.transpose(A, (0, 2, 1)), g)
    assert np.allclose(a.grad.numpy(), gA, atol=1e-4)
    assert np.allclose(b.grad.numpy(), gB, atol=1e-4)


def test_bmm_accepts_non_contiguous_operands():
    rng = _rng(3)
    A = rng.standard_normal((2, 3, 4)).astype(np.float32)
    B = rng.standard_normal((2, 4, 5)).astype(np.float32)
    a = _t(A)
    bt = _t(np.ascontiguousarray(np.transpose(B, (0, 2, 1))))
    out = a.bmm(bt.permute([0, 2, 1]))  # transposed views on both sides
    assert np.allclose(out.numpy(), np.matmul(A, B), atol=1e-5)


def test_bmm_rejects_bad_shapes():
    a = _t(np.zeros((2, 3, 4), dtype=np.float32))
    b = _t(np.zeros((3, 4, 2), dtype=np.float32))  # batch mismatch
    with pytest.raises(ValueError):
        a.bmm(b)
    c = _t(np.zeros((2, 5, 2), dtype=np.float32))  # inner-dim mismatch
    with pytest.raises(ValueError):
        a.bmm(c)
    m = _t(np.zeros((3, 4), dtype=np.float32))  # rank guard
    with pytest.raises(ValueError):
        a.bmm(m)


def test_bmm_respects_grad_mode():
    a = _t(np.ones((2, 2, 2), dtype=np.float32), requires_grad=True)
    b = _t(np.stack([np.eye(2), np.eye(2)]).astype(np.float32), requires_grad=True)
    with ox.no_grad():
        detached = a.bmm(b)
    assert detached.requires_grad is False
    recorded = a.bmm(b)
    assert recorded.requires_grad is True
    recorded.sum().backward()
    assert a.grad is not None


def test_bmm_finite_difference_gradcheck():
    # Central differences on sum(g * bmm(a, b)) w.r.t. one element of a.
    rng = _rng(4)
    A = rng.standard_normal((2, 3, 3)).astype(np.float32)
    B = rng.standard_normal((2, 3, 2)).astype(np.float32)
    g = rng.standard_normal((2, 3, 2)).astype(np.float32)
    a = _t(A, True)
    out = a.bmm(_t(B))
    (out * _t(g)).sum().backward()
    analytic = a.grad.numpy()
    h = 1e-2
    for idx in [(0, 0, 0), (1, 2, 1), (0, 1, 2)]:
        i, j, k = idx
        Ap = A.copy(); Ap[i, j, k] += h
        Am = A.copy(); Am[i, j, k] -= h
        num = (g * np.matmul(Ap, B)).sum() - (g * np.matmul(Am, B)).sum()
        num /= 2 * h
        assert abs(analytic[i, j, k] - num) < 1e-3 * max(1.0, abs(num))


# ---- batched Linear ---------------------------------------------------------------

def test_linear_batched_values_and_grads():
    lin = nn.Linear(4, 3, seed=1)
    x = _t(_rng(5).standard_normal((7, 4)), requires_grad=True)
    out = lin(x)
    W = lin.weight.tensor.numpy()
    bias = lin.bias.tensor.numpy()
    assert out.shape == [7, 3]
    assert np.allclose(out.numpy(), x.numpy() @ W.T + bias, atol=1e-5)
    out.sum().backward()
    assert x.grad is not None and lin.weight.grad is not None


def test_linear_rank3_fold_path():
    lin = nn.Linear(4, 3, seed=1)
    x = _t(_rng(6).standard_normal((2, 5, 4)), requires_grad=True)
    out = lin(x)
    W = lin.weight.tensor.numpy()
    bias = lin.bias.tensor.numpy()
    assert out.shape == [2, 5, 3]
    assert np.allclose(out.numpy(), x.numpy() @ W.T + bias, atol=1e-5)

    # Gradients equal the fold math: dL/dW = g_folded^T @ x_folded.
    g = _rng(7).standard_normal((2, 5, 3)).astype(np.float32)
    fresh = nn.Linear(4, 3, seed=1)
    xf = _t(_rng(6).standard_normal((2, 5, 4)), requires_grad=True)
    (fresh(xf) * _t(g)).sum().backward()
    gf = g.reshape(10, 3)
    expected_w = gf.T @ xf.numpy().reshape(10, 4)
    assert np.allclose(fresh.weight.grad.numpy(), expected_w, atol=1e-4)
    assert np.allclose(fresh.bias.grad.numpy(), gf.sum(0), atol=1e-5)


def test_linear_1d_input():
    lin = nn.Linear(4, 3, seed=1)
    x = _t(_rng(8).standard_normal(4), requires_grad=True)
    out = lin(x)
    assert out.shape == [3]
    W = lin.weight.tensor.numpy()
    bias = lin.bias.tensor.numpy()
    assert np.allclose(out.numpy(), x.numpy() @ W.T + bias, atol=1e-5)
    out.sum().backward()
    assert x.grad is not None


# ---- end-to-end batched training ----------------------------------------------------

def test_batched_training_converges():
    rng = _rng(9)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    Y = X @ rng.standard_normal((4, 2)).astype(np.float32)
    model = nn.Linear(4, 2, seed=9)
    opt = ox.optim.Sgd(model.parameters(), lr=0.1)
    Xt, Yv = _t(X), _t(Y)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = F.mse_loss(model(Xt), Yv)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    final = F.mse_loss(model(Xt), Yv).item()
    assert final < first * 0.05, (first, final)


def test_batched_two_layer_mlp_trains():
    rng = _rng(10)
    X = rng.standard_normal((64, 2)).astype(np.float32)
    # y = sin-ish nonlinear target
    Y = np.stack([np.sin(X[:, 0]), X[:, 1] ** 2], axis=1).astype(np.float32)
    model = nn.Sequential(nn.Linear(2, 16, seed=11), nn.Tanh(), nn.Linear(16, 2, seed=12))
    opt = ox.optim.Sgd(model.parameters(), lr=0.05, momentum=0.9)
    Xt, Yv = _t(X), _t(Y)
    first = None
    for _ in range(400):
        opt.zero_grad()
        loss = F.mse_loss(model(Xt), Yv)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    final = F.mse_loss(model(Xt), Yv).item()
    assert final < first * 0.2, (first, final)
