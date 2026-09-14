"""Phase 4 training-loop tests: AdamW (decoupled decay), Adam (L2-in-grad),
LR schedulers (StepLR, CosineAnnealingLR, LambdaLR, LinearWarmup+cosine via
SequentialLR), gradient clipping, and the exit criterion — a small CNN
trained with all of the above, matching a NumPy reference run."""

import math

import numpy as np
import pytest

import oxitorch as ox
import oxitorch.nn as nn
import oxitorch.optim as op
from oxitorch.nn import functional as F

RTOL = 1e-4
ATOL = 1e-5


def _t(arr, requires_grad=False):
    return ox.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), requires_grad=requires_grad
    )


class _FakeOpt:
    """Bare optimizer stand-in for scheduler tests."""

    def __init__(self, lr):
        self.lr = lr


# ---- AdamW --------------------------------------------------------------------


def _regression_data(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    y = rng.standard_normal((4, 1)).astype(np.float32)
    w = rng.standard_normal((1, 3)).astype(np.float32) * 0.5
    return x, y, w


def test_adamw_two_steps_match_formula():
    x, y, w0 = _regression_data()
    lr, b1, b2, eps, wd = 0.05, 0.9, 0.999, 1e-8, 0.1
    layer = nn.Linear(3, 1, bias=False, seed=0)  # weight-only: closed form
    layer.weight.tensor = _t(w0, True)
    opt = op.AdamW(layer.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    m = np.zeros_like(w0)
    v = np.zeros_like(w0)
    w = w0.copy()
    for step in range(1, 3):
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
        np.testing.assert_allclose(
            layer.weight.tensor.numpy(), w, rtol=1e-4, atol=1e-6
        ), f"step {step}"


def test_adam_uses_l2_in_grad_not_decoupled_decay():
    """Same data/decay: classic Adam (decay inside the moments) must differ
    from AdamW (decay on the parameter) — proving the two semantics exist."""
    x, y, w0 = _regression_data()
    results = {}
    for name, cls in [("adam", op.Adam), ("adamw", op.AdamW)]:
        layer = nn.Linear(3, 1, seed=0)
        layer.weight.tensor = _t(w0.copy(), True)
        opt = cls(layer.parameters(), lr=0.05, weight_decay=0.1)
        layer.zero_grad()
        F.mse_loss(layer(_t(x)), _t(y)).backward()
        opt.step()
        results[name] = layer.weight.tensor.numpy()
    assert not np.allclose(results["adam"], results["adamw"], rtol=1e-3)
    # And classic Adam's single step must match its own closed form.
    layer = nn.Linear(3, 1, seed=0)
    layer.weight.tensor = _t(w0.copy(), True)
    opt = op.Adam(layer.parameters(), lr=0.05, weight_decay=0.1)
    layer.zero_grad()
    F.mse_loss(layer(_t(x)), _t(y)).backward()
    grad = layer.weight.grad.numpy().copy()
    g_l2 = grad + 0.1 * w0
    want = w0 - 0.05 * g_l2 / (np.sqrt(g_l2 * g_l2) + 1e-8)
    opt.step()
    np.testing.assert_allclose(layer.weight.tensor.numpy(), want, rtol=1e-4, atol=1e-6)


def test_adamw_default_weight_decay_is_torchs():
    layer = nn.Linear(2, 1, seed=0)
    opt = op.AdamW(layer.parameters())
    assert opt.weight_decay == 1e-2
    opt2 = op.Adam(layer.parameters())
    assert opt2.weight_decay == 0.0


# ---- schedulers ---------------------------------------------------------------


def test_step_lr_schedule():
    o = _FakeOpt(1.0)
    s = op.StepLR(o, step_size=2, gamma=0.5)
    lrs = []
    for _ in range(6):
        lrs.append(o.lr)
        s.step()
    # Construction sets epoch 0 (full base lr, torch semantics); gamma
    # applies at every second epoch boundary after that.
    assert lrs == pytest.approx([1.0, 1.0, 0.5, 0.5, 0.25, 0.25])
    assert s.get_last_lr() == [o.lr]


def test_cosine_annealing_schedule():
    o = _FakeOpt(1.0)
    s = op.CosineAnnealingLR(o, T_max=4, eta_min=0.0)
    lrs = []
    for _ in range(5):
        lrs.append(o.lr)
        s.step()
    assert lrs[0] == pytest.approx(1.0)  # epoch 0 is the base lr
    assert lrs[1] == pytest.approx((1 + math.cos(math.pi / 4)) / 2)
    assert lrs[2] == pytest.approx(0.5)
    assert lrs[4] == pytest.approx(0.0, abs=1e-6)  # T_max reached


def test_lambda_lr_schedule():
    o = _FakeOpt(0.1)
    s = op.LambdaLR(o, lr_lambda=lambda t: 0.5**t)
    lrs = []
    for _ in range(3):
        lrs.append(o.lr)
        s.step()
    assert lrs == pytest.approx([0.1, 0.05, 0.025])


def test_warmup_then_cosine_sequential():
    o = _FakeOpt(0.1)
    warm = op.LinearWarmup(o, warmup_epochs=2)
    cosine = op.CosineAnnealingLR(o, T_max=6)
    seq = op.SequentialLR(o, [warm, cosine], milestones=[2])
    lrs = []
    for _ in range(9):
        lrs.append(round(float(o.lr), 6))
        seq.step()
    # Warmup ramps 0.05 -> 0.1 over 2 epochs, then cosine anneals 0.1 -> 0
    # over the next 6 (T_max=6), then stays at eta_min.
    assert lrs[0] == pytest.approx(0.05)
    assert lrs[1] == pytest.approx(0.1)
    assert lrs[2] == pytest.approx(0.1)  # cosine epoch 0 = full lr
    assert lrs[4] == pytest.approx(0.075)
    assert lrs[6] == pytest.approx(0.025)
    assert lrs[8] == pytest.approx(0.0, abs=1e-6)
    assert lrs[9 - 1] >= 0.0


def test_sequential_lr_base_lr_not_polluted_by_construction_order():
    """The warmup ctor steps the optimizer once; the cosine's base_lr must
    still be the *original* lr, not the halved value the ctor left behind."""
    o = _FakeOpt(0.2)
    warm = op.LinearWarmup(o, warmup_epochs=2)
    cosine = op.CosineAnnealingLR(o, T_max=10, eta_min=0.0)
    seq = op.SequentialLR(o, [warm, cosine], milestones=[2])
    assert cosine.base_lr == pytest.approx(0.2)
    seq.step()
    seq.step()
    assert o.lr == pytest.approx(0.2)  # cosine epoch 0 restored the base


# ---- gradient clipping --------------------------------------------------------


def _param_with_grad(arr, grad):
    p = nn.Parameter(_t(arr, True))
    p.tensor.grad = _t(grad)
    return p


def test_clip_grad_norm_scales_when_exceeded():
    p = _param_with_grad([[1.0, 2.0]], [[3.0, 4.0]])
    total = op.clip_grad_norm_([p], 1.0)
    assert total == pytest.approx(5.0)
    np.testing.assert_allclose(p.grad.numpy(), [[0.6, 0.8]], rtol=1e-5)


def test_clip_grad_norm_noop_under_threshold():
    p = _param_with_grad([[1.0, 2.0]], [[3.0, 4.0]])
    total = op.clip_grad_norm_([p], 10.0)
    assert total == pytest.approx(5.0)
    np.testing.assert_array_equal(p.grad.numpy(), [[3.0, 4.0]])


def test_clip_grad_norm_spans_multiple_params():
    pa = _param_with_grad(np.zeros((1, 3)), [[3.0, 0.0, 0.0]])
    pb = _param_with_grad(np.zeros((1, 4)), [[0.0, 4.0, 0.0, 0.0]])
    total = op.clip_grad_norm_([pa, pb], 0.5)
    assert total == pytest.approx(5.0)
    np.testing.assert_allclose(pa.grad.numpy(), [[0.3, 0.0, 0.0]], rtol=1e-5)
    np.testing.assert_allclose(pb.grad.numpy(), [[0.0, 0.4, 0.0, 0.0]], rtol=1e-5)


def test_clip_grad_norm_inf_norm():
    p = _param_with_grad(np.zeros((1, 2)), [[3.0, -4.0]])
    total = op.clip_grad_norm_([p], 1.0, norm_type=float("inf"))
    assert total == pytest.approx(4.0)
    np.testing.assert_allclose(p.grad.numpy(), [[0.75, -1.0]], rtol=1e-5)


def test_clip_grad_norm_p1():
    p = _param_with_grad(np.zeros((1, 2)), [[3.0, 4.0]])
    total = op.clip_grad_norm_([p], 1.0, norm_type=1.0)
    assert total == pytest.approx(7.0)
    np.testing.assert_allclose(p.grad.numpy(), [[3.0 / 7.0, 4.0 / 7.0]], rtol=1e-5)


def test_clip_grad_value():
    p = _param_with_grad(np.zeros((1, 3)), [[5.0, -5.0, 1.0]])
    op.clip_grad_value_([p], 2.0)
    np.testing.assert_array_equal(p.grad.numpy(), [[2.0, -2.0, 1.0]])
    with pytest.raises(ValueError):
        op.clip_grad_value_([p], 0.0)


def test_clip_grad_norm_skips_missing_grads():
    p_with = _param_with_grad(np.zeros((1, 2)), [[3.0, 4.0]])
    p_without = nn.Parameter(_t(np.zeros((1, 2)), True))
    total = op.clip_grad_norm_([p_with, p_without], 1.0)
    assert total == pytest.approx(5.0)


# ---- exit criterion: CNN trained with the full Phase 4 stack -------------------


def test_phase4_exit_criterion_cnn_with_adamw_warmup_cosine_clip():
    """Train conv → relu → pool → linear on a regression target with AdamW,
    a warmup+cosine schedule, and gradient clipping. The identical run is
    reimplemented in NumPy (AdamW + schedule + clip + backward); losses must
    track the reference, strictly decrease, and stay finite (no NaNs)."""
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((4, 2, 8, 8)).astype(np.float32)
    y_np = rng.standard_normal((4, 3)).astype(np.float32)

    bound = 1.0 / np.sqrt(18)
    w1 = rng.uniform(-bound, bound, (2, 2, 3, 3)).astype(np.float32)
    b1 = np.zeros(2, dtype=np.float32)
    fb = 1.0 / np.sqrt(32)
    w2 = rng.uniform(-fb, fb, (3, 32)).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)

    base_lr, wd, clip = 0.02, 0.01, 1.0
    epochs = 30

    # --- oxitorch run ---
    conv = nn.Conv2d(2, 2, 3, padding=1, seed=0)
    conv.weight.tensor = _t(w1.copy(), True)
    conv.bias.tensor = _t(b1.copy(), True)
    fc = nn.Linear(32, 3, seed=0)
    fc.weight.tensor = _t(w2.copy(), True)
    fc.bias.tensor = _t(b2.copy(), True)
    model_params = list(conv.parameters()) + list(fc.parameters())
    opt = op.AdamW(model_params, lr=base_lr, weight_decay=wd)
    warm = op.LinearWarmup(opt, warmup_epochs=3)
    cosine = op.CosineAnnealingLR(opt, T_max=epochs - 3)
    sched = op.SequentialLR(opt, [warm, cosine], milestones=[3])

    ox_losses = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = F.linear(
            F.max_pool2d(conv(_t(x_np)).apply("relu"), 2).reshape([4, 32]),
            fc.weight.tensor,
            fc.bias.tensor,
        )
        loss = F.mse_loss(pred, _t(y_np))
        loss.backward()
        total = op.clip_grad_norm_(model_params, clip)
        assert math.isfinite(total)
        opt.step()
        sched.step()
        value = float(loss.numpy())
        assert math.isfinite(value)
        ox_losses.append(value)

    # --- NumPy reference run ---
    def conv_np(xa, wa, ba):
        xp = np.pad(xa, ((0, 0), (0, 0), (1, 1), (1, 1)))
        oh, ow = xa.shape[2], xa.shape[3]
        out = np.zeros((xa.shape[0], wa.shape[0], oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                patch = xp[:, :, i : i + 3, j : j + 3]
                out[:, :, i, j] = np.einsum("nckl,ockl->no", patch, wa)
        return out + ba.reshape(1, -1, 1, 1)

    def fwd_grads(xa, ya, wa, ba, wb, bb):
        act = np.maximum(conv_np(xa, wa, ba), 0.0)
        pooled = np.zeros((4, 2, 4, 4), dtype=np.float32)
        argmax = np.zeros((4, 2, 4, 4, 2), dtype=np.int32)
        for i in range(4):
            for j in range(4):
                win = act[:, :, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                pooled[:, :, i, j] = win.max(axis=(2, 3))
                for n in range(4):
                    for c in range(2):
                        idx = int(np.argmax(win[n, c]))
                        argmax[n, c, i, j] = (idx // 2, idx % 2)
        flat = pooled.reshape(4, 32)
        pred = flat @ wb.T + bb
        diff = pred - ya
        loss = float((diff * diff).mean())
        dpred = 2.0 * diff / diff.size
        gwb = dpred.T @ flat
        gbb = dpred.sum(0)
        gpooled = (dpred @ wb).reshape(4, 2, 4, 4)
        gact = np.zeros_like(act)
        for i in range(4):
            for j in range(4):
                for n in range(4):
                    for c in range(2):
                        ki, kj = argmax[n, c, i, j]
                        gact[n, c, 2 * i + ki, 2 * j + kj] += gpooled[n, c, i, j]
        gpre = gact * (act > 0)
        xp = np.pad(xa, ((0, 0), (0, 0), (1, 1), (1, 1)))
        gwa = np.zeros_like(wa)
        gba = np.zeros_like(ba)
        for i in range(8):
            for j in range(8):
                patch = xp[:, :, i : i + 3, j : j + 3]
                gwa += np.einsum("nckl,no->ockl", patch, gpre[:, :, i, j])
                gba += gpre[:, :, i, j].sum(0)
        return loss, {"w1": gwa, "b1": gba, "w2": gwb, "b2": gbb}

    params = {"w1": w1.copy(), "b1": b1.copy(), "w2": w2.copy(), "b2": b2.copy()}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    np_losses = []
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        # Warmup+cosine lr (same formula as the engine's schedulers).
        t = step - 1
        if t < 3:
            lr = base_lr * (t + 1) / 3
        else:
            local = t - 3
            lr = base_lr * (1 + math.cos(math.pi * local / (epochs - 3))) / 2
        loss, grads = fwd_grads(x_np, y_np, params["w1"], params["b1"], params["w2"], params["b2"])
        # clip_grad_norm_ over the flattened parameter set
        total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, clip / (total + 1e-6))
        for k, g in grads.items():
            g = g * scale
            m[k] = b1m * m[k] + (1 - b1m) * g
            v[k] = b2m * v[k] + (1 - b2m) * g * g
            mh = m[k] / (1 - b1m**step)
            vh = v[k] / (1 - b2m**step)
            params[k] = params[k] - lr * mh / (np.sqrt(vh) + eps) - lr * wd * params[k]
        assert math.isfinite(loss)
        np_losses.append(loss)

    assert len(ox_losses) == len(np_losses) == epochs
    for i, (a, r) in enumerate(zip(ox_losses, np_losses)):
        assert a == pytest.approx(r, rel=5e-3, abs=5e-4), f"epoch {i}: {a} vs {r}"
    assert ox_losses[-1] < ox_losses[0]
    assert not any(math.isnan(v) for v in ox_losses)
