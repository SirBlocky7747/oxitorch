"""Grad-mode (`no_grad`/`enable_grad`) and `retain_graph` tests.

These cover the plan.md Phase 2 carry-over items, at the Python boundary:
mode save/restore, nesting, detachment semantics, graph retention with
grad accumulation, and the freed-graph error on double backward.
"""

import numpy as np
import pytest

import oxitorch as ox
import oxitorch.nn as nn


def _t(arr, requires_grad=False):
    return ox.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), requires_grad)


# ---- grad mode -----------------------------------------------------------------

def test_is_grad_enabled_default():
    assert ox.is_grad_enabled() is True


def test_no_grad_detaches_op_results():
    x = _t([1.0, 2.0], requires_grad=True)
    with ox.no_grad():
        assert ox.is_grad_enabled() is False
        y = x * x
        assert y.requires_grad is False
        assert np.allclose(y.numpy(), [1.0, 4.0])  # values still compute
    z = x * x
    assert z.requires_grad is True


def test_no_grad_restores_mode_on_exception():
    x = _t([1.0], requires_grad=True)
    with pytest.raises(RuntimeError, match="boom"):
        with ox.no_grad():
            raise RuntimeError("boom")
    assert ox.is_grad_enabled() is True
    assert (x * x).requires_grad is True  # sanity: usable after the block


def test_nested_enable_grad_inside_no_grad():
    x = _t([3.0, -1.0], requires_grad=True)
    with ox.no_grad():
        with ox.enable_grad():
            assert ox.is_grad_enabled() is True
            z = x * x
            assert z.requires_grad is True
        assert ox.is_grad_enabled() is False  # inner scope closed
        w = x * x
        assert w.requires_grad is False
    assert ox.is_grad_enabled() is True


def test_enable_grad_inside_no_grad_backward_flows():
    # Torch's "reparameterization under no_grad" idiom.
    x = _t([2.0], requires_grad=True)
    with ox.no_grad():
        with ox.enable_grad():
            y = x * 3.0
    loss = (y * y).sum()
    loss.backward()
    # d/dx (3x)^2 = 18x
    assert np.allclose(x.grad.numpy(), [36.0], atol=1e-6)


def test_backward_through_no_grad_output_rejected():
    w = nn.Linear(2, 2, seed=1)
    x = _t([[1.0, 2.0]])
    with ox.no_grad():
        out = w(x)
    with pytest.raises(ValueError, match="does not require grad"):
        out.sum().backward()
    assert w.weight.grad is None and w.bias.grad is None


def test_no_grad_inference_leaves_params_untouched():
    mlp = nn.Sequential(nn.Linear(2, 4, seed=3), nn.ReLU(), nn.Linear(4, 1, seed=4))
    mlp.eval()
    x = _t([[0.5, -1.5]])
    with ox.no_grad():
        out = mlp(x)
    assert out.requires_grad is False
    assert all(p.grad is None for p in mlp.parameters())


def test_set_grad_enabled_manual_save_restore():
    prev = ox.set_grad_enabled(False)
    try:
        assert ox.is_grad_enabled() is False
    finally:
        ox.set_grad_enabled(prev)
    assert ox.is_grad_enabled() is True


# ---- retain_graph ----------------------------------------------------------------

def test_default_backward_consumes_graph():
    w = _t([2.0], requires_grad=True)
    loss = (w * w).sum()
    loss.backward()
    assert np.allclose(w.grad.numpy(), [4.0])
    with pytest.raises(ValueError, match="retain_graph"):
        loss.backward()


def test_retain_graph_accumulates_leaf_grads():
    w = _t([2.0], requires_grad=True)
    loss = (w * w).sum()
    loss.backward(retain_graph=True)
    assert np.allclose(w.grad.numpy(), [4.0])
    loss.backward(retain_graph=True)
    assert np.allclose(w.grad.numpy(), [8.0])  # accumulated, not replaced
    loss.backward()  # final pass frees the graph
    assert np.allclose(w.grad.numpy(), [12.0])


def test_retain_passes_do_not_compound_intermediates():
    # Leaf grads accumulate across passes, but each pass recomputes its own
    # intermediate grads — a second pass must yield the same leaf grad again.
    a = _t([3.0], requires_grad=True)
    b = (a * a).sum()  # intermediate
    c = (b * 2.0).sum()
    c.backward(retain_graph=True)
    first = a.grad.item()
    c.backward(retain_graph=True)
    second = a.grad.item()
    assert abs(first - 12.0) < 1e-5 and abs(second - 24.0) < 1e-5


def test_two_losses_into_one_leaf_accumulate():
    w = _t([1.0, 2.0], requires_grad=True)
    l1 = (w * w).sum()
    l2 = (w * 3.0).sum()
    l1.backward(retain_graph=True)
    l2.backward()
    assert np.allclose(w.grad.numpy(), [2.0 * 1.0 + 3.0, 2.0 * 2.0 + 3.0])


# ---- training-loop integration ----------------------------------------------------

def test_train_step_with_no_grad_metrics():
    model = nn.Linear(2, 2, seed=7)
    opt = ox.optim.Sgd(model.parameters(), lr=0.1)
    x = _t([[0.5, -1.5]])
    target = _t([[1.0, 0.0]])
    for _ in range(3):
        opt.zero_grad()
        loss = ((model(x) - target) ** 2).sum()
        loss.backward()
        opt.step()
        with ox.no_grad():
            metric = ((model(x) - target) ** 2).sum().item()
        assert np.isfinite(metric)
