"""Phase 3 tests: nn.Module system, functional API, optimizers (plan.md).

Every value/gradient is checked against a NumPy reference; the training
test exercises the full stack (Module -> functional -> autograd -> SGD).
"""

import numpy as np
import pytest

import oxitorch as ox
import oxitorch.nn as nn
import oxitorch.nn.functional as F


def _t(arr, requires_grad=False):
    return ox.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), requires_grad)


# ---- functional values vs numpy ---------------------------------------------

def test_linear_value():
    x = _t([[1.0, 2.0]])
    w = _t([[0.5, -1.0], [2.0, 0.0]])
    b = _t([1.0, -1.0])
    out = F.linear(x, w, b)
    ref = np.array([[1.0, 2.0]]) @ np.array([[0.5, -1.0], [2.0, 0.0]]).T + np.array([1.0, -1.0])
    assert np.allclose(out.numpy(), ref, atol=1e-6)


def test_softmax_and_log_softmax_parity():
    x = _t([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    xr = x.detach().numpy()
    sm = F.softmax(x, -1).numpy()
    ref = np.exp(xr) / np.exp(xr).sum(-1, keepdims=True)
    assert np.allclose(sm, ref, atol=1e-6)
    ls = F.log_softmax(x, -1).numpy()
    assert np.allclose(np.exp(ls).sum(-1), 1.0, atol=1e-6)


def test_log_softmax_stability_large_inputs():
    big = _t([[1000.0, 1001.0, 1002.0]])
    ls = F.log_softmax(big, -1).numpy()
    assert np.all(np.isfinite(ls))
    sm = np.exp(ls)
    sm = sm / sm.sum()
    want = np.exp(np.array([1000.0, 1001.0, 1002.0]) - 1002.0)
    want = want / want.sum()
    assert np.allclose(sm, want, atol=1e-6)


def test_activations_values():
    x = _t([-1.0, 0.0, 1.0, 2.0])
    r = x.detach().numpy()
    assert np.allclose(F.relu(x).numpy(), np.maximum(r, 0), atol=1e-7)
    assert np.allclose(F.sigmoid(x).numpy(), 1 / (1 + np.exp(-r)), atol=1e-6)
    assert np.allclose(F.tanh(x).numpy(), np.tanh(r), atol=1e-6)
    assert np.allclose(F.silu(x).numpy(), r / (1 + np.exp(-r)), atol=1e-6)
    c = np.sqrt(2 / np.pi)
    gelu_ref = 0.5 * r * (1 + np.tanh(c * (r + 0.044715 * r**3)))
    assert np.allclose(F.gelu(x).numpy(), gelu_ref, atol=1e-5)


def test_layer_norm_value_and_grads():
    x = _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    w = _t([1.0, 1.0, 1.0], requires_grad=True)
    b = _t([0.0, 0.0, 0.0], requires_grad=True)
    out = F.layer_norm(x, w, b)
    xr = x.detach().numpy()
    mu = xr.mean(-1, keepdims=True)
    v = ((xr - mu) ** 2).mean(-1, keepdims=True)
    ref = (xr - mu) / np.sqrt(v + 1e-5)
    assert np.allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)
    # Weighted sum: a plain sum would have a mathematically-zero x grad
    # (normalized rows sum to zero), so weight the output to probe the path.
    coef = np.array([[1.0, -2.0, 0.5], [0.25, 1.0, -1.0]], dtype=np.float32)
    (out * _t(coef)).sum().backward()
    assert x.grad is not None and w.grad is not None and b.grad is not None
    assert np.abs(x.grad.numpy()).sum() > 1e-6


def test_rms_norm_value():
    x = _t([[1.0, -2.0, 3.0]])
    w = _t([1.0, 2.0, 0.5])
    out = F.rms_norm(x, w)
    xr = x.detach().numpy()
    ref = xr / np.sqrt((xr**2).mean(-1, keepdims=True) + 1e-6) * np.array([[1.0, 2.0, 0.5]])
    assert np.allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)


def test_embedding_lookup_and_grad():
    w = _t([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    out = F.embedding([2, 0, 2], w)
    assert np.allclose(out.numpy(), [[3, 3], [1, 1], [3, 3]], atol=1e-7)
    out.sum().backward()
    # Index 0 picked once, index 2 picked twice.
    assert np.allclose(w.grad.numpy(), [[1, 1], [0, 0], [2, 2]], atol=1e-6)


def test_mse_loss_value_and_grad():
    pred = _t([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = np.array([[0.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    loss = F.mse_loss(pred, _t(target))
    d = pred.detach().numpy() - target
    assert abs(loss.item() - (d * d).mean()) < 1e-6
    loss.backward()
    assert np.allclose(pred.grad.numpy(), 2 * d / d.size, atol=1e-6)


# ---- cross_entropy ------------------------------------------------------------

def test_cross_entropy_value_and_grad():
    logits = _t([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], requires_grad=True)
    loss = F.cross_entropy(logits, [2, 0])
    lg = logits.detach().numpy()
    m = lg.max(-1, keepdims=True)
    ls = lg - m - np.log(np.exp(lg - m).sum(-1, keepdims=True))
    ref = -np.mean([ls[0, 2], ls[1, 0]])
    assert abs(loss.item() - ref) < 1e-5
    loss.backward()
    sm = np.exp(ls)
    g = (sm - np.eye(3, dtype=np.float32)[[2, 0]]) / 2
    assert np.allclose(logits.grad.numpy(), g, rtol=1e-4, atol=1e-6)


def test_cross_entropy_rejects_wrong_batch():
    logits = _t(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="targets"):
        F.cross_entropy(logits, [0, 1, 2])


# ---- scalar autograd paths ----------------------------------------------------

def test_scalar_arithmetic_autograd():
    t = _t([1.0, 2.0], requires_grad=True)
    (t * 2.0 + 1.0).sum().backward()
    assert np.allclose(t.grad.numpy(), [2.0, 2.0], atol=1e-6)

    t.zero_grad()
    (2.0 - t).sum().backward()
    assert np.allclose(t.grad.numpy(), [-1.0, -1.0], atol=1e-6)

    t.zero_grad()
    (2.0 / t).sum().backward()
    # d/dt (2/t) = -2/t^2
    assert np.allclose(t.grad.numpy(), -2.0 / np.array([1.0, 4.0]), atol=1e-6)

    t.zero_grad()
    (t**3).sum().backward()
    assert np.allclose(t.grad.numpy(), [3.0, 12.0], atol=1e-5)


def test_apply_neg_abs_are_differentiable():
    t = _t([1.0, -2.0], requires_grad=True)
    t.mean().apply("neg").backward()
    assert np.allclose(t.grad.numpy(), [-0.5, -0.5], atol=1e-6)

    t.zero_grad()
    y = t.apply("abs")
    y.sum().backward()
    assert np.allclose(t.grad.numpy(), [1.0, -1.0], atol=1e-6)


# ---- Module mechanics ----------------------------------------------------------

def test_named_parameters_paths():
    model = nn.Sequential(nn.Linear(2, 8, seed=1), nn.ReLU(), nn.Linear(8, 2, seed=2))
    names = [n for n, _ in model.named_parameters()]
    assert names == ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias"]
    assert len(model.parameters()) == 4


def test_state_dict_roundtrip():
    model = nn.Sequential(nn.Linear(3, 4, seed=5), nn.Tanh(), nn.Linear(4, 1, seed=6))
    x = _t(np.random.default_rng(0).standard_normal((2, 3)).astype(np.float32))
    sd = model.state_dict()
    clone = nn.Sequential(nn.Linear(3, 4, seed=7), nn.Tanh(), nn.Linear(4, 1, seed=8))
    clone.load_state_dict(sd)
    assert np.allclose(model(x).numpy(), clone(x).numpy(), atol=1e-6)


def test_state_dict_shape_mismatch_raises():
    model = nn.Linear(3, 4, seed=1)
    bad = {"weight": np.zeros((2, 2), dtype=np.float32), "bias": np.zeros(4, dtype=np.float32)}
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_state_dict(bad)


def test_train_eval_recursion():
    model = nn.Sequential(nn.Linear(2, 2, seed=1), nn.Dropout(0.5))
    model.eval()
    assert model[1].training is False
    model.train()
    assert model[1].training is True


def test_dropout_train_eval():
    x = _t(np.ones((1000,), dtype=np.float32), requires_grad=True)
    layer = nn.Dropout(p=0.5, seed=123)
    out = layer(x)
    kept = out.numpy() != 0
    assert 0.3 < kept.mean() < 0.7  # roughly half survive
    assert np.allclose(out.numpy()[kept], 2.0, atol=1e-6)  # inverted scaling
    layer.eval()
    assert np.allclose(layer(x).numpy(), 1.0, atol=1e-7)  # identity in eval


# ---- optimizers -----------------------------------------------------------------

def test_sgd_matches_numpy_reference():
    w = nn.Parameter(_t([[1.0, 2.0]], requires_grad=True))
    x = _t([[3.0]])
    target = _t([[1.0]])
    opt = ox.optim.Sgd([w], lr=0.1)
    opt.zero_grad()
    loss = F.mse_loss(w.tensor * x, target)
    loss.backward()
    opt.step()
    # dL/dw = 2*(w*x - t)*x / numel = 2*[[2,5]]*3/2 = [[6, 15]]
    assert np.allclose(w.tensor.numpy(), [[1.0, 2.0] - 0.1 * np.array([6.0, 15.0])], atol=1e-5)


def test_adam_reduces_loss():
    w = nn.Parameter(_t([[2.0]], requires_grad=True))
    opt = ox.optim.Adam([w], lr=0.1)
    first = None
    for _ in range(20):
        opt.zero_grad()
        loss = (w.tensor * w.tensor).sum()  # w^2
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first * 0.5


# ---- end-to-end training ---------------------------------------------------------

@pytest.fixture(scope="module")
def xor_model():
    np.random.seed(0)
    model = nn.Sequential(nn.Linear(2, 8, seed=42), nn.ReLU(), nn.Linear(8, 2, seed=43))
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    Y = [0, 1, 1, 0]
    Xt = _t(X)
    opt = ox.optim.Sgd(model.parameters(), lr=0.5, momentum=0.9)
    for _ in range(500):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(Xt), Y)
        loss.backward()
        opt.step()
    return model, Xt, Y, loss.item()


def test_xor_training_converges(xor_model):
    model, Xt, Y, final_loss = xor_model
    assert final_loss < 0.01
    pred = model(Xt).numpy().argmax(1)
    assert (pred == np.array(Y)).all()


def test_xor_model_reproducible_from_state_dict(xor_model):
    model, Xt, _, _ = xor_model
    sd = model.state_dict()
    clone = nn.Sequential(nn.Linear(2, 8, seed=99), nn.ReLU(), nn.Linear(8, 2, seed=98))
    clone.load_state_dict(sd)
    assert np.allclose(model(Xt).numpy(), clone(Xt).numpy(), atol=1e-6)
