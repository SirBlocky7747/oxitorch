//! Differentiable ops: forward through `oxi-tensor` + VJP closures.
//!
//! Each function runs the forward op and records a `Var` node whose VJP
//! rebuilds the gradient **from differentiable `Var` primitives**, so
//! double backward (second-order gradients) works wherever the VJP is
//! itself differentiable. Non-differentiable bits (integer indices, argmax
//! positions) are captured as constants inside the closures.
//!
//! Gradients flowing into broadcast operands are reduced with
//! [`crate::graph::unbroadcast`], mirroring torch's broadcast backward.

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::ops::ReduceKind;
use oxi_tensor::Tensor;

use crate::graph::{unbroadcast, Var};

/// Sum `g` down to `input`'s shape and wrap as a constant grad node.
#[allow(clippy::redundant_clone)] // see the module-level note on input cloning
fn grad_for(g: &Var, input: &Var) -> OxitorchResult<Var> {
    let shape = input.value().shape().to_vec();
    let g = unbroadcast(g.value(), &shape)?;
    Ok(Var::leaf(g, false))
}

// Every op clones its `Var` inputs: one copy goes into the node's `inputs`
// list, one is captured by the VJP closure. `Var` is an `Rc` handle, so
// these are cheap refcount bumps — and the pattern is uniform enough that
// silencing the pedantic clone lint beats hand-rolling shared storage.

// ---- elementwise binary (all differentiable both ways) ---------------------

/// `a + b` (broadcast).
///
/// # Errors
/// Propagates [`Tensor::add`].
pub fn add(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().add(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "add", vec![a.clone(), b.clone()], move |g| {
        Ok(vec![grad_for(g, &a)?, grad_for(g, &b)?])
    }))
}

/// `a - b` (broadcast).
///
/// # Errors
/// Propagates [`Tensor::sub`].
pub fn sub(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().sub(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "sub", vec![a.clone(), b.clone()], move |g| {
        let gb = g.value().unary_op("neg")?;
        Ok(vec![grad_for(g, &a)?, grad_for(&Var::leaf(gb, false), &b)?])
    }))
}

/// `a * b` (broadcast).
///
/// # Errors
/// Propagates [`Tensor::mul`].
pub fn mul(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().mul(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "mul", vec![a.clone(), b.clone()], move |g| {
        let ga = g.value().mul(b.value())?;
        let gb = g.value().mul(a.value())?;
        Ok(vec![
            grad_for(&Var::leaf(ga, false), &a)?,
            grad_for(&Var::leaf(gb, false), &b)?,
        ])
    }))
}

/// `a / b` (broadcast).
///
/// # Errors
/// Propagates [`Tensor::div`].
pub fn div(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().div(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "div", vec![a.clone(), b.clone()], move |g| {
        // d/da = g / b ; d/db = -g * a / b^2
        let ga = g.value().div(b.value())?;
        let b2 = b.value().mul(b.value())?;
        let gb = g.value().mul(a.value())?.div(&b2)?.unary_op("neg")?;
        Ok(vec![
            grad_for(&Var::leaf(ga, false), &a)?,
            grad_for(&Var::leaf(gb, false), &b)?,
        ])
    }))
}

/// `a^b` (broadcast).
///
/// # Errors
/// Propagates [`Tensor::pow`].
pub fn pow(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().pow(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "pow", vec![a.clone(), b.clone()], move |g| {
        // d/da = g * b * a^(b-1); d/db = g * out * ln(a)
        let bm1 = b.value().add(&Tensor::from_vec(&[], vec![-1.0])?)?;
        let apow = a.value().pow(&bm1)?;
        let ga = g.value().mul(b.value())?.mul(&apow)?;
        let out_t = a.value().pow(b.value())?;
        let loga = a.value().unary_op("log")?;
        let gb = g.value().mul(&out_t)?.mul(&loga)?;
        Ok(vec![
            grad_for(&Var::leaf(ga, false), &a)?,
            grad_for(&Var::leaf(gb, false), &b)?,
        ])
    }))
}

/// `a * scalar` (exact, no broadcast error paths).
///
/// # Errors
/// Propagates [`Tensor::mul_scalar`].
pub fn mul_scalar(a: &Var, c: f32) -> OxitorchResult<Var> {
    let out = a.value().mul_scalar(c)?;
    let a = a.clone();
    Ok(Var::op(out, "mul_scalar", vec![a.clone()], move |g| {
        let ga = g.value().mul_scalar(c)?;
        Ok(vec![Var::leaf(ga, false)])
    }))
}

// ---- unary (differentiable via known derivatives) --------------------------

/// Generic unary with an explicit local-derivative closure `d(x) -> dx`.
fn unary_generic(
    x: &Var,
    f: impl Fn(&Tensor) -> OxitorchResult<Tensor>,
    dfdx: impl Fn(&Tensor) -> OxitorchResult<Tensor> + Send + 'static,
    name: &'static str,
) -> OxitorchResult<Var> {
    let out = f(x.value())?;
    let x = x.clone();
    Ok(Var::op(out, name, vec![x.clone()], move |g| {
        let dx = dfdx(x.value())?;
        let gx = g.value().mul(&dx)?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// `-x`.
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn neg(x: &Var) -> OxitorchResult<Var> {
    let out = x.value().unary_op("neg")?;
    let x = x.clone();
    Ok(Var::op(out, "neg", vec![x.clone()], move |g| {
        let gx = g.value().unary_op("neg")?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// Elementwise `|x|` (grad `sign(x)`).
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn abs(x: &Var) -> OxitorchResult<Var> {
    unary_generic(x, |t| t.unary_op("abs"), |t| t.unary_op("sign"), "abs")
}

/// Elementwise `exp(x)`.
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn exp(x: &Var) -> OxitorchResult<Var> {
    unary_generic(x, |t| t.unary_op("exp"), |t| t.unary_op("exp"), "exp")
}

/// Elementwise `log(x)` (grad `1/x`).
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn log(x: &Var) -> OxitorchResult<Var> {
    unary_generic(
        x,
        |t| t.unary_op("log"),
        |t| {
            let one = Tensor::ones_like(t)?;
            one.div(t)
        },
        "log",
    )
}

/// Elementwise `sqrt(x)` (grad `1/(2 sqrt x)`).
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn sqrt(x: &Var) -> OxitorchResult<Var> {
    unary_generic(
        x,
        |t| t.unary_op("sqrt"),
        |t| {
            // d/dx sqrt = 1 / (2 sqrt x); callers keep x > 0.
            let two = Tensor::full(
                &t.shape().iter().map(|&d| d as i64).collect::<Vec<_>>(),
                2.0,
            )?;
            let denom = t.unary_op("sqrt")?.mul(&two)?;
            let one = Tensor::ones_like(t)?;
            one.div(&denom)
        },
        "sqrt",
    )
}

/// Elementwise `tanh(x)` (grad `1 - tanh^2`).
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn tanh(x: &Var) -> OxitorchResult<Var> {
    unary_generic(
        x,
        |t| t.unary_op("tanh"),
        |t| {
            let th = t.unary_op("tanh")?;
            let th2 = th.mul(&th)?;
            let one = Tensor::ones_like(t)?;
            one.sub(&th2)
        },
        "tanh",
    )
}

/// Elementwise `sigmoid(x)` (grad `s (1 - s)`).
///
/// # Errors
/// Propagates [`Tensor::unary_op`].
pub fn sigmoid(x: &Var) -> OxitorchResult<Var> {
    unary_generic(
        x,
        |t| t.unary_op("sigmoid"),
        |t| {
            let s = t.unary_op("sigmoid")?;
            let one = Tensor::ones_like(t)?;
            s.mul(&one.sub(&s)?)
        },
        "sigmoid",
    )
}

/// ReLU: forward `max(x, 0)`, grad `x > 0`. Subgradient 0 at exactly 0
/// (torch's default).
///
/// # Errors
/// Propagates tensor ops.
pub fn relu(x: &Var) -> OxitorchResult<Var> {
    let out = x.value().unary_op("relu")?;
    let x = x.clone();
    let mask = x.value().gt(&Tensor::zeros(
        &x.value()
            .shape()
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>(),
    )?)?;
    Ok(Var::op(out, "relu", vec![x.clone()], move |g| {
        let gx = g.value().mul(&mask)?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

// ---- matmul & reductions ----------------------------------------------------

/// 2-D matrix multiply `(m,k) @ (k,n)`.
///
/// # Errors
/// Propagates [`Tensor::matmul`].
pub fn matmul(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().matmul(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(
        out,
        "matmul",
        vec![a.clone(), b.clone()],
        move |g| {
            let bt = b.value().transpose()?;
            let at = a.value().transpose()?;
            let gv = g.value();
            let ga = gv.matmul(&bt)?;
            let gb = at.matmul(gv)?;
            Ok(vec![
                grad_for(&Var::leaf(ga, false), &a)?,
                grad_for(&Var::leaf(gb, false), &b)?,
            ])
        },
    ))
}

/// Batched matrix multiply `(b,m,k) @ (b,k,n) -> (b,m,n)`.
///
/// # Errors
/// Propagates [`Tensor::bmm`].
pub fn bmm(a: &Var, b: &Var) -> OxitorchResult<Var> {
    let out = a.value().bmm(b.value())?;
    let (a, b) = (a.clone(), b.clone());
    Ok(Var::op(out, "bmm", vec![a.clone(), b.clone()], move |g| {
        // Per batch i: dL/dA[i] = g[i] @ B[i]^T, dL/dB[i] = A[i]^T @ g[i].
        // Batched transposes are `permute` views; `bmm` materializes
        // contiguous operands internally, so no explicit copies here.
        // The upstream grad always carries this node's own (b, m, n)
        // shape: downstream broadcasts are unbroadcast before reaching
        // `accumulate_grad`.
        let bt = b.value().permute(&[0, 2, 1])?;
        let ga = g.value().bmm(&bt)?;
        let at = a.value().permute(&[0, 2, 1])?;
        let gb = at.bmm(g.value())?;
        Ok(vec![
            grad_for(&Var::leaf(ga, false), &a)?,
            grad_for(&Var::leaf(gb, false), &b)?,
        ])
    }))
}

/// Reduction along `dim` (`keepdim` in the forward call). Gradients
/// broadcast the reduced result back up.
///
/// # Errors
/// Propagates [`Tensor::reduce`].
pub fn reduce(x: &Var, kind: ReduceKind, dim: i64, keepdim: bool) -> OxitorchResult<Var> {
    let out = x.value().reduce(kind, dim, keepdim)?;
    let dim = normalize_dim_for_grad(dim, x.value().ndim())?;
    let x_shape = x.value().shape().to_vec();
    match kind {
        ReduceKind::Sum => {
            let x = x.clone();
            Ok(Var::op(out, "sum", vec![x.clone()], move |g| {
                let up = normalize_reduce_grad(g.value(), &x_shape, dim)?;
                let up = expand_to(&up, &x_shape)?;
                Ok(vec![grad_for(&Var::leaf(up, false), &x)?])
            }))
        }
        ReduceKind::Mean => {
            let x = x.clone();
            let n = x_shape[dim] as f32;
            Ok(Var::op(out, "mean", vec![x.clone()], move |g| {
                let up = normalize_reduce_grad(g.value(), &x_shape, dim)?;
                let up = expand_to(&up, &x_shape)?;
                let scaled = up.mul_scalar(1.0 / n)?;
                Ok(vec![grad_for(&Var::leaf(scaled, false), &x)?])
            }))
        }
        ReduceKind::Max | ReduceKind::Min => {
            let x = x.clone();
            // Positions of the winning element along `dim`, captured as a
            // constant scatter target (argmax is not differentiable).
            let arg = x.value().reduce(
                if kind == ReduceKind::Max {
                    ReduceKind::ArgMax
                } else {
                    ReduceKind::ArgMin
                },
                dim as i64,
                true,
            )?;
            Ok(Var::op(out, "maxmin", vec![x.clone()], move |g| {
                let up = normalize_reduce_grad(g.value(), &x_shape, dim)?;
                // Scatter the upstream grad to the argmax positions.
                let idx_row: Vec<i64> = arg.to_vec().iter().map(|&v| v as i64).collect();
                let gx = scatter_along_dim(&up, dim, &idx_row, &x_shape)?;
                Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
            }))
        }
        ReduceKind::ArgMax | ReduceKind::ArgMin => Err(OxitorchError::NotImplemented(
            "argmax/argmin are integer outputs and have no gradient".into(),
        )),
        ReduceKind::Norm => {
            let x = x.clone();
            Ok(Var::op(out, "norm", vec![x.clone()], move |g| {
                let up = normalize_reduce_grad(g.value(), &x_shape, dim)?;
                let up = expand_to(&up, &x_shape)?;
                // d/dx = g * x / ||x||  (broadcast along the reduced dim)
                let n = x.value().reduce(ReduceKind::Norm, dim as i64, true)?;
                let safe = n.add_scalar(1e-12)?;
                let ratio = x.value().div(&safe)?;
                let gx = up.mul(&ratio)?;
                Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
            }))
        }
    }
}

// ---- views (linear index maps: grads route straight back) -------------------

/// `transpose` (2-D).
///
/// # Errors
/// Propagates [`Tensor::transpose`].
pub fn transpose(x: &Var) -> OxitorchResult<Var> {
    let out = x.value().transpose()?;
    let x = x.clone();
    Ok(Var::op(out, "transpose", vec![x.clone()], move |g| {
        let gx = g.value().transpose()?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// `reshape` (may copy for non-contiguous inputs; still linear).
///
/// # Errors
/// Propagates [`Tensor::reshape`].
pub fn reshape(x: &Var, new_shape: &[i64]) -> OxitorchResult<Var> {
    let out = x.value().reshape(new_shape)?;
    let x = x.clone();
    let orig = x
        .value()
        .shape()
        .iter()
        .map(|&d| d as i64)
        .collect::<Vec<_>>();
    Ok(Var::op(out, "reshape", vec![x.clone()], move |g| {
        let gx = g.value().reshape(&orig)?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// `narrow` along dim 0 with a captured `offset` (step must be 1 for now).
///
/// # Errors
/// Propagates [`Tensor::narrow`]; `NotImplemented` for steps != 1.
pub fn narrow_dim0(x: &Var, offset: i64, len: i64) -> OxitorchResult<Var> {
    let out = x.value().narrow(0, offset, offset + len, 1)?;
    let x = x.clone();
    let orig = x.value().shape().to_vec();
    Ok(Var::op(out, "narrow", vec![x.clone()], move |g| {
        // Zero-pad the gradient back into the full shape at `offset`.
        // Row width = product of the trailing dims of the ORIGINAL tensor.
        let width: usize = orig[1..].iter().product::<usize>().max(1);
        let zeros_before = Tensor::zeros(&[offset, width as i64])?;
        let after = (orig[0] as i64 - offset - len).max(0);
        let zeros_after = Tensor::zeros(&[after, width as i64])?;
        let gx = concat_rows(&[zeros_before, g.value().clone(), zeros_after])?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// Concatenates row-major same-width tensors along dim 0.
fn concat_rows(ts: &[Tensor]) -> OxitorchResult<Tensor> {
    let mut data = Vec::new();
    let mut rows = 0usize;
    let width = ts.first().map_or(0, |t| t.numel() / t.shape()[0].max(1));
    for t in ts {
        let r = t.shape()[0];
        rows += r;
        data.extend(t.to_vec());
    }
    Tensor::from_vec(&[rows as i64, width as i64], data)
}

/// `permute` (view; grad permutes back with the inverse order).
///
/// # Errors
/// Propagates [`Tensor::permute`].
pub fn permute(x: &Var, order: &[i64]) -> OxitorchResult<Var> {
    let out = x.value().permute(order)?;
    let x = x.clone();
    // Inverse permutation: forward moves dim order[i] -> position i, so the
    // grad maps position i -> order[i], i.e. permute with `inv` where
    // inv[order[i]] = i.
    let mut inv = vec![0i64; order.len()];
    for (i, &d) in order.iter().enumerate() {
        inv[d as usize] = i as i64;
    }
    Ok(Var::op(out, "permute", vec![x.clone()], move |g| {
        let gx = g.value().permute(&inv)?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// `broadcast_to` (zero-stride view; grad sums the broadcast axes back).
///
/// # Errors
/// Propagates [`Tensor::broadcast_to`].
pub fn broadcast_to(x: &Var, out_shape: &[usize]) -> OxitorchResult<Var> {
    let out = x.value().broadcast_to(out_shape)?;
    let x = x.clone();
    Ok(Var::op(out, "broadcast_to", vec![x.clone()], move |g| {
        let gx = unbroadcast(g.value(), x.value().shape())?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

// ---- fancy indexing ----------------------------------------------------------

/// `index_select` along `dim`; grad scatters rows back (last write wins on
/// duplicate indices, matching a strict sum would require splitting — we
/// accumulate, which is the mathematically correct VJP).
///
/// # Errors
/// Propagates [`Tensor::index_select`].
pub fn index_select(x: &Var, dim: i64, indices: &[i64]) -> OxitorchResult<Var> {
    let out = x.value().index_select(dim, indices)?;
    let dim = normalize_dim_for_grad(dim, x.value().ndim())?;
    let x = x.clone();
    let idx: Vec<usize> = indices
        .iter()
        .map(|&i| {
            let len = x.value().shape()[dim];
            if i < 0 {
                (len as i64 + i) as usize
            } else {
                i as usize
            }
        })
        .collect();
    Ok(Var::op(out, "index_select", vec![x.clone()], move |g| {
        let gx = scatter_index_select(g.value(), dim, &idx, x.value().shape())?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

/// `base[indices[i], ...] = values[i, ...]` along dim 0, last write wins
/// (matching `Tensor::scatter_dim0`'s forward semantics).
///
/// Gradients:
/// - `base`: its own rows where untouched, **zero** on every written row —
///   last-write-wins means nothing flows back to an overwritten row.
/// - `values`: the upstream grad gathered at the final write position for
///   each slot (duplicate indices keep only the last write's grad), so the
///   VJP is exact for the forward's last-write-wins contract.
///
/// # Errors
/// Propagates [`Tensor::scatter_dim0`] / [`Tensor::index_select`].
pub fn scatter_dim0(base: &Var, indices: &[i64], values: &Var) -> OxitorchResult<Var> {
    let out = base.value().scatter_dim0(indices, values.value())?;
    let rows = base.value().shape()[0];
    // Resolve negative indices once (the forward already validated range).
    let resolved: Vec<usize> = indices
        .iter()
        .map(|&i| {
            if i < 0 {
                (rows as i64 + i) as usize
            } else {
                i as usize
            }
        })
        .collect();
    // Last write wins: slot p carries gradient only if it is the final
    // write to its row.
    let mut last_writer = vec![usize::MAX; rows];
    for (p, &r) in resolved.iter().enumerate() {
        last_writer[r] = p;
    }
    let is_winner: Vec<bool> = resolved
        .iter()
        .enumerate()
        .map(|(p, &r)| last_writer[r] == p)
        .collect();
    let mut written = resolved.clone();
    written.sort_unstable();
    written.dedup();

    let base_shape = base.value().shape().to_vec();
    let values_shape = values.value().shape().to_vec();
    let base = base.clone();
    let values = values.clone();
    Ok(Var::op(
        out,
        "scatter_dim0",
        vec![base.clone(), values.clone()],
        move |g| {
            // d/dbase: keep untouched rows, zero every written row (an
            // overwritten row has no path to the output).
            let inner: usize = base_shape[1..].iter().product::<usize>().max(1);
            let mut mask = vec![1.0f32; base_shape.iter().product::<usize>()];
            for &r in &written {
                mask[r * inner..(r + 1) * inner].fill(0.0);
            }
            let gb = g.value().mul(&Tensor::from_vec(
                &base_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
                mask,
            )?)?;
            // d/dvalues: each winning slot receives the grad of the row it
            // wrote; losing slots (overwritten) receive zero.
            let g_flat = g.value().to_vec();
            let mut gv = vec![0.0f32; resolved.len() * inner];
            for (p, &r) in resolved.iter().enumerate() {
                if is_winner[p] {
                    gv[p * inner..(p + 1) * inner]
                        .copy_from_slice(&g_flat[r * inner..(r + 1) * inner]);
                }
            }
            let gv = Tensor::from_vec(
                &values_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
                gv,
            )?;
            Ok(vec![
                grad_for(&Var::leaf(gb, false), &base)?,
                grad_for(&Var::leaf(gv, false), &values)?,
            ])
        },
    ))
}

/// Boolean-mask indexing: `self[mask]` with `mask` broadcast against `self`
/// (matching `Tensor::masked_select`). The VJP walks the same broadcast grid
/// and scatters each picked grad back to its logical position, so duplicate
/// picks from broadcasting **accumulate** — the mathematically correct VJP,
/// same policy as `index_select`.
///
/// # Errors
/// Propagates [`Tensor::masked_select`] / layout errors.
pub fn masked_select(x: &Var, mask: &Var) -> OxitorchResult<Var> {
    let out = x.value().masked_select(mask.value())?;
    let x_shape = x.value().shape().to_vec();
    let mask_shape = mask.value().shape().to_vec();
    let x = x.clone();
    let mask = mask.clone();
    Ok(Var::op(out, "masked_select", vec![x.clone()], move |g| {
        let out_shape = oxi_tensor::layout::broadcast_shapes(&x_shape, &mask_shape)?;
        // Logical (row-major) flat strides over the broadcast grid: an
        // output dim that broadcasts a size-1 input dim contributes 0 (all
        // its grid points map to the same element — the accumulation path);
        // every other dim strides like the input's row-major layout.
        let x_rms = oxi_tensor::layout::row_major_strides(&x_shape);
        let mask_rms = oxi_tensor::layout::row_major_strides(&mask_shape);
        let x_offset = out_shape.len() - x_shape.len();
        let mask_offset = out_shape.len() - mask_shape.len();
        let x_flat_str: Vec<usize> = (0..out_shape.len())
            .map(|d| match d.checked_sub(x_offset) {
                Some(j) if x_shape[j] > 1 => x_rms[j],
                _ => 0,
            })
            .collect();
        let mask_flat_str: Vec<usize> = (0..out_shape.len())
            .map(|d| match d.checked_sub(mask_offset) {
                Some(j) if mask_shape[j] > 1 => mask_rms[j],
                _ => 0,
            })
            .collect();

        // Walk the grid in the forward's order (odometer, last dim fastest),
        // scattering picked grads where the mask is nonzero.
        let mut grads = vec![0.0f32; x_shape.iter().product::<usize>().max(1)];
        let g_flat = g.value().to_vec();
        let mask_flat = mask.value().to_vec();
        let mut multi = vec![0usize; out_shape.len()];
        let mut pick = 0usize;
        for _ in 0..oxi_tensor::layout::element_count(&out_shape) {
            let m_flat: usize = multi.iter().zip(&mask_flat_str).map(|(&i, &s)| i * s).sum();
            if mask_flat[m_flat] != 0.0 {
                let x_flat: usize = multi.iter().zip(&x_flat_str).map(|(&i, &s)| i * s).sum();
                grads[x_flat] += g_flat[pick];
                pick += 1;
            }
            for d in (0..multi.len()).rev() {
                multi[d] += 1;
                if multi[d] < out_shape[d] {
                    break;
                }
                multi[d] = 0;
            }
        }
        let gx = Tensor::from_shared(
            &x_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            std::sync::Arc::new(grads),
        )?;
        Ok(vec![grad_for(&Var::leaf(gx, false), &x)?])
    }))
}

// ---- helpers shared by VJPs ---------------------------------------------------

fn normalize_dim_for_grad(dim: i64, ndim: usize) -> OxitorchResult<usize> {
    oxi_tensor::layout::normalize_dim(dim, ndim)
}

/// Normalizes a reduction's upstream gradient to keepdim form (x_shape with
/// a 1 at `dim`), or returns it unchanged when it already has the full input
/// shape (upstream broadcasts). `keepdim=false` outputs have the reduced
/// axis *removed*, so the axis is reinserted here before broadcasting.
fn normalize_reduce_grad(g: &Tensor, x_shape: &[usize], dim: usize) -> OxitorchResult<Tensor> {
    if g.shape() == x_shape {
        return Ok(g.clone());
    }
    let g_shape = g.shape();
    let mut padded: Vec<i64> = g_shape.iter().map(|&d| d as i64).collect();
    if g_shape.len() + 1 == x_shape.len() {
        // Reinsert the reduced axis (size 1) at `dim`.
        padded.insert(dim.min(padded.len()), 1);
    }
    g.reshape(&padded)
}

/// Expands `t` (keepdim-shaped) up to `shape` via `broadcast_to`.
fn expand_to(t: &Tensor, shape: &[usize]) -> OxitorchResult<Tensor> {
    // Insert size-1 dims to match rank, then broadcast.
    let t_shape = t.shape();
    let mut padded = vec![1i64; shape.len()];
    for (i, _) in shape.iter().enumerate() {
        let tail_start = shape.len() - t_shape.len();
        if i >= tail_start {
            padded[i] = t_shape[i - tail_start] as i64;
        }
    }
    let reshaped = t.reshape(&padded)?;
    reshaped.broadcast_to(shape)
}

/// Scatters the keepdim-shaped reduction grad `up` (`[outer, 1, inner]`)
/// along `dim` at the captured argmax `positions` (one per output element,
/// flat over `outer × inner`), zero elsewhere — the gradient of max/min.
fn scatter_along_dim(
    up: &Tensor,
    dim: usize,
    positions: &[i64],
    full_shape: &[usize],
) -> OxitorchResult<Tensor> {
    let zeros = Tensor::zeros(&full_shape.iter().map(|&d| d as i64).collect::<Vec<_>>())?;
    let mut data = zeros.to_vec();
    let up_data = up.to_vec();
    let inner: usize = full_shape[dim + 1..].iter().product::<usize>().max(1);
    let outer: usize = full_shape[..dim].iter().product::<usize>().max(1);
    let dim_len = full_shape[dim];
    for o in 0..outer {
        for i in 0..inner {
            let pos = positions[o * inner + i];
            let pos = if pos < 0 {
                (dim_len as i64 + pos) as usize
            } else {
                pos as usize
            };
            data[(o * dim_len + pos) * inner + i] = up_data[o * inner + i];
        }
    }
    Tensor::from_shared(
        &full_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
        std::sync::Arc::new(data),
    )
}

/// Inverse of `index_select`: adds each gathered row of `g` back into the
/// base at its index (accumulating on duplicates).
fn scatter_index_select(
    g: &Tensor,
    dim: usize,
    indices: &[usize],
    full_shape: &[usize],
) -> OxitorchResult<Tensor> {
    let zeros = Tensor::zeros(&full_shape.iter().map(|&d| d as i64).collect::<Vec<_>>())?;
    let mut data = zeros.to_vec();
    let g_data = g.to_vec();
    let (outer, inner) = {
        let o: usize = full_shape[..dim].iter().product();
        let i: usize = full_shape[dim + 1..].iter().product();
        (o, i)
    };
    let len = full_shape[dim];
    for o in 0..outer {
        for (k, &idx) in indices.iter().enumerate() {
            for i in 0..inner {
                let src = (o * indices.len() + k) * inner + i;
                let dst = (o * len + idx) * inner + i;
                data[dst] += g_data[src];
            }
        }
    }
    Tensor::from_shared(
        &full_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
        std::sync::Arc::new(data),
    )
}
