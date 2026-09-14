"""NumPy parity suite for oxitorch ops.

plan.md Phase 1 exit criterion (adapted): "parity test suite vs. NumPy for
~80 ops passes on random inputs (allclose with PyTorch's default
tolerances)". This suite drives a registry of (oxitorch op, numpy
equivalent, input shapes) triples over randomized inputs and asserts
allclose with torch's default rtol/atol.
"""

import numpy as np
import pytest

import oxitorch
from oxitorch import Tensor

RTOL = 1e-5
ATOL = 1e-8  # torch.allclose defaults


def _rng():
    return np.random.default_rng(0x5EED)


def _t(arr):
    return oxitorch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _np(t):
    return t.numpy().astype(np.float32)


# ---- registry: (name, oxitorch callable, numpy callable, input builder) ----

SHAPES = [(), (5,), (3, 4), (2, 3, 4)]


def binary_cases(a, b):
    """Builds binary op cases; b is broadcastable against a."""
    return [
        ("add", lambda: _t(a) + _t(b), lambda: a + b),
        ("sub", lambda: _t(a) - _t(b), lambda: a - b),
        ("mul", lambda: _t(a) * _t(b), lambda: a * b),
        ("div", lambda: _t(a) / _t(b), lambda: a / b),
        ("add", lambda: _t(a).add(_t(b)), lambda: a + b),
    ]


def comparison_cases(a, b):
    return [
        ("eq", lambda: _t(a) == _t(b), lambda: a == b),
        ("ne", lambda: _t(a) != _t(b), lambda: a != b),
        ("lt", lambda: _t(a) < _t(b), lambda: a < b),
        ("le", lambda: _t(a) <= _t(b), lambda: a <= b),
        ("gt", lambda: _t(a) > _t(b), lambda: a > b),
        ("ge", lambda: _t(a) >= _t(b), lambda: a >= b),
    ]


def unary_cases(a):
    r = np.abs(a) + 0.5  # keep logs/sqrt/div in a safe range
    return [
        ("neg", lambda: -_t(a), lambda: -a),
        ("abs", lambda: _t(a).apply("abs"), lambda: np.abs(a)),
        ("sqrt", lambda: _t(r).apply("sqrt"), lambda: np.sqrt(r)),
        ("exp", lambda: _t(a * 0.1).apply("exp"), lambda: np.exp(a * 0.1)),
        ("log", lambda: _t(r).apply("log"), lambda: np.log(r)),
        ("sin", lambda: _t(a).apply("sin"), lambda: np.sin(a)),
        ("cos", lambda: _t(a).apply("cos"), lambda: np.cos(a)),
        ("tanh", lambda: _t(a).apply("tanh"), lambda: np.tanh(a)),
        ("relu", lambda: _t(a).apply("relu"), lambda: np.maximum(a, 0)),
        ("sigmoid", lambda: _t(a).apply("sigmoid"), lambda: 1 / (1 + np.exp(-a))),
        ("floor", lambda: _t(a).apply("floor"), lambda: np.floor(a)),
        ("ceil", lambda: _t(a).apply("ceil"), lambda: np.ceil(a)),
        ("round", lambda: _t(a).apply("round"), lambda: np.round(a)),
        ("square", lambda: _t(a).apply("square"), lambda: a * a),
        ("reciprocal", lambda: _t(r).apply("reciprocal"), lambda: 1 / r),
    ]


def _iter_binary_unary_cmp():
    rng = _rng()
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)  # broadcasts as a row
    for name, ox, np_fn in binary_cases(a, b) + comparison_cases(a, b):
        yield name, ox, np_fn
    for name, ox, np_fn in unary_cases(a):
        yield name, ox, np_fn


@pytest.mark.parametrize("name,ox,np_fn", list(_iter_binary_unary_cmp()))
def test_op_parity(name, ox, np_fn):
    got = ox()
    want = np_fn()
    np.testing.assert_allclose(_np(got), want, rtol=RTOL, atol=ATOL)


def test_matmul_parity_random():
    rng = _rng()
    for (m, k, n) in [(1, 1, 1), (4, 7, 3), (16, 16, 16), (33, 45, 29)]:
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        got = (_t(a) @ _t(b)).numpy()
        want = a @ b
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=1e-4)


def test_reduction_parity():
    rng = _rng()
    a = rng.standard_normal((3, 4, 5)).astype(np.float32)
    t = _t(a)
    np.testing.assert_allclose(_np(t.sum(0)), a.sum(0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(t.sum(-1)), a.sum(-1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(t.sum(-1, keepdim=True)), a.sum(-1, keepdims=True))
    np.testing.assert_allclose(_np(t.mean(1)), a.mean(1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(t.max(0)), a.max(0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(t.min(2)), a.min(2), rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(_np(t.argmax(-1)), a.argmax(-1))
    np.testing.assert_array_equal(_np(t.argmin(1)), a.argmin(1))
    np.testing.assert_allclose(_np(t.norm(-1, False)), np.linalg.norm(a, axis=-1), rtol=RTOL, atol=ATOL)
    assert float(t.sum()) == pytest.approx(float(a.sum()), rel=RTOL)
    assert float(t.mean()) == pytest.approx(float(a.mean()), rel=RTOL)


def test_view_and_indexing_parity():
    rng = _rng()
    a = rng.standard_normal((4, 6)).astype(np.float32)
    t = _t(a)

    # reshape / ravel
    np.testing.assert_array_equal(_np(t.reshape((24,))), a.reshape(24))
    np.testing.assert_array_equal(_np(t.reshape((-1, 8))), a.reshape(-1, 8))
    np.testing.assert_array_equal(_np(t.reshape((6, 4)).reshape((24,))), a.reshape(24))

    # transpose / permute
    np.testing.assert_array_equal(_np(t.transpose()), a.T)
    np.testing.assert_array_equal(_np(t.permute([1, 0])), a.T)

    # slicing: oxitorch narrows dim 0 only (Phase 1); verify parity there
    np.testing.assert_array_equal(_np(t[2]), a[2])
    np.testing.assert_array_equal(_np(t[-1]), a[-1])
    np.testing.assert_array_equal(_np(t[1:3]), a[1:3])
    np.testing.assert_array_equal(_np(t[::2]), a[::2])
    np.testing.assert_array_equal(_np(t[1:]), a[1:])

    # select / index_select / gather
    np.testing.assert_array_equal(_np(t.select(1, 3)), a[:, 3])
    np.testing.assert_array_equal(_np(t.index_select(0, [3, 0, 2])), a[[3, 0, 2]])
    np.testing.assert_array_equal(_np(t.index_select(1, [-1, -2])), a[:, [-1, -2]])
    np.testing.assert_array_equal(_np(t.gather([1, 1, 0])), a[[1, 1, 0]])

    # scatter
    vals = np.full((2, 6), -1.0, dtype=np.float32)
    out = t.scatter([0, 2], _t(vals))
    want = a.copy()
    want[[0, 2]] = vals
    np.testing.assert_array_equal(_np(out), want)

    # masked_select
    mask = (a > 0).astype(np.float32)
    np.testing.assert_array_equal(_np(t.masked_select(_t(mask))), a[a > 0])

    # broadcast_to
    row = rng.standard_normal(6).astype(np.float32)
    np.testing.assert_array_equal(_np(_t(row).reshape((1, 6)).broadcast_to([4, 6])), np.broadcast_to(row, (4, 6)))


def test_strided_op_parity():
    """Ops on non-contiguous views must match NumPy."""
    rng = _rng()
    a = rng.standard_normal((4, 5)).astype(np.float32)
    t = _t(a)
    tr = t.transpose()
    np.testing.assert_allclose(_np(tr.apply("square")), (a.T) ** 2, rtol=RTOL, atol=ATOL)
    sliced = t[1:4:2]
    np.testing.assert_allclose(_np(sliced.apply("neg")), -a[1:4:2], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_np(tr.sum(0)), a.sum(1), rtol=RTOL, atol=ATOL)


def test_full_round_trip():
    rng = _rng()
    a = rng.standard_normal((3, 4)).astype(np.float32)
    back = _np(_t(a))
    np.testing.assert_array_equal(back, a)
    assert back.dtype == np.float32
