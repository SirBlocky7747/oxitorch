"""Indexing ops in the autograd graph: gather/index_select/scatter/
masked_select/fancy-index all keep gradients flowing, with duplicate-index
accumulation as the mathematically correct VJP (plan.md Phase 3 carry-over)."""

import numpy as np

import oxitorch as ox
from oxitorch.nn import functional as F

RTOL = 1e-4
ATOL = 1e-5


def _t(arr, requires_grad=False):
    return ox.from_numpy(
        np.ascontiguousarray(arr, dtype=np.float32), requires_grad=requires_grad
    )


def test_scatter_grads_last_write_wins():
    base = _t([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], requires_grad=True)
    vals = _t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    out = base.scatter([2, 0, 2], vals)
    out.sum().backward()

    np.testing.assert_array_equal(
        _np := out.numpy(), [[3.0, 4.0], [20.0, 21.0], [5.0, 6.0]]
    )
    # Written rows of `base` are overwritten -> zero grad; row 1 survives.
    np.testing.assert_array_equal(base.grad.numpy(), [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    # Slot 0 wrote row 2 but lost to slot 2 -> zero; winners get their row grad.
    np.testing.assert_array_equal(vals.grad.numpy(), [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])


def test_scatter_forward_matches_numpy():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 3)).astype(np.float32)
    vals = rng.standard_normal((3, 3)).astype(np.float32)
    out = _t(a, requires_grad=True).scatter([4, 1, 4], _t(vals))
    want = a.copy()
    want[[4, 1, 4]] = vals  # last write wins, same as NumPy assignment
    np.testing.assert_array_equal(out.numpy(), want)


def test_masked_select_broadcast_accumulation():
    # Column vector broadcast against a (2,3) mask: element (0,0) is picked
    # twice -> grads 1+2; element (1,0) picked twice -> grads 3+4.
    col = _t([[7.0], [9.0]], requires_grad=True)
    mask = _t([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    picked = col.masked_select(mask)
    assert picked.numpy().shape == (4,)
    picked.backward(_t([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(picked.numpy(), [7.0, 7.0, 9.0, 9.0])
    np.testing.assert_array_equal(col.grad.numpy(), [[3.0], [7.0]])


def test_masked_select_grad_matches_dense_multiply():
    # masked_select(x, m) grad == (m * upstream) summed over the broadcast —
    # verified against the equivalent dense computation in NumPy.
    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    up = rng.standard_normal((2, 3)).astype(np.float32)
    mask = (rng.standard_normal((2, 3)) > 0).astype(np.float32)

    x = _t(a, requires_grad=True)
    picked = x.masked_select(_t(mask))
    picked.backward(_t(up[mask > 0]))
    want = mask * up
    np.testing.assert_allclose(x.grad.numpy(), want, rtol=RTOL, atol=ATOL)


def test_index_select_duplicate_accumulation():
    w = _t(np.zeros((3, 2), dtype=np.float32), requires_grad=True)
    emb = w.index_select(0, [1, 1, 2])
    emb.backward(_t([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    np.testing.assert_array_equal(w.grad.numpy(), [[0.0, 0.0], [3.0, 30.0], [3.0, 30.0]])


def test_embedding_duplicate_index_accumulation():
    # The headline case: `F.embedding` gradients accumulate over duplicate
    # token ids, matching the torch reference exactly.
    weight = _t(np.zeros((4, 3), dtype=np.float32), requires_grad=True)
    out = F.embedding([2, 2, 0, 2], weight)
    out.backward(_t(np.arange(1, 13, dtype=np.float32).reshape(4, 3)))
    want = np.zeros((4, 3), dtype=np.float32)
    # Row 2 is picked by slots 0, 1, 3 -> [1,2,3] + [4,5,6] + [10,11,12].
    want[2] = np.array([1, 2, 3]) + np.array([4, 5, 6]) + np.array([10, 11, 12])
    want[0] = np.array([7, 8, 9])
    np.testing.assert_array_equal(weight.grad.numpy(), want)


def test_fancy_index_getitem_backward():
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    t = _t(a, requires_grad=True)
    picked = t[[2, 1, 1]]
    picked.backward(_t(np.ones((3, 3), dtype=np.float32)))
    np.testing.assert_array_equal(picked.numpy(), a[[2, 1, 1]])
    np.testing.assert_array_equal(
        t.grad.numpy(), [[0, 0, 0], [2, 2, 2], [1, 1, 1]]
    )


def test_gather_backward_accumulates_duplicates():
    t = _t([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = t.gather([1, 1])
    out.backward(_t([[5.0, 5.0], [6.0, 6.0]]))
    np.testing.assert_array_equal(out.numpy(), [[3.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_equal(t.grad.numpy(), [[0.0, 0.0], [11.0, 11.0]])


def test_detached_scatter_still_works():
    # No-grad path: forward values only, no graph.
    a = np.ones((2, 2), dtype=np.float32)
    vals = np.full((1, 2), -1.0, dtype=np.float32)
    out = _t(a).scatter([0], _t(vals))
    np.testing.assert_array_equal(out.numpy(), [[-1.0, -1.0], [1.0, 1.0]])
    assert out.grad is None
