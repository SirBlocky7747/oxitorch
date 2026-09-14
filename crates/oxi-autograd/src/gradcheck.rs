//! Finite-difference gradcheck, in the spirit of
//! `torch.autograd.gradcheck`.
//!
//! Central-difference verification: for scalar outputs
//! `(f(x+h) - f(x-h)) / 2h` must match the analytic gradient within
//! tolerance. Nonsmooth ops (relu at 0) are checked with randomized
//! offsets in the tests.

use oxi_core::OxitorchResult;
use oxi_tensor::Tensor;

use crate::graph::{backward, Var};
use crate::ops;

/// Reduces any tensor to a true 0-d scalar via repeated sum reductions.
fn sum_to_scalar(v: &Var) -> OxitorchResult<Var> {
    let mut t = v.clone();
    while t.value().ndim() > 0 {
        t = ops::reduce(&t, oxi_tensor::ops::ReduceKind::Sum, 0, false)?;
    }
    Ok(t)
}

/// Checks `f`'s gradient at `input` by central differences on a scalar
/// output, returning `Ok(true)` when every element matches within
/// `tol * max(1, |analytic|)`.
///
/// # Errors
/// Propagates any op error from the forward/backward runs.
pub fn check_scalar(
    input: &Tensor,
    f: impl Fn(&Var) -> OxitorchResult<Var>,
    tol: f32,
) -> OxitorchResult<bool> {
    let x = Var::leaf(input.clone(), true);
    let out = f(&x)?;
    if out.value().numel() != 1 {
        return Ok(false);
    }
    backward(&out, None)?;

    let analytic = x.grad().map_or_else(Vec::new, |g| g.to_vec());
    let base = input.to_vec();
    let h = 1e-2_f32;
    let mut ok = true;
    for i in 0..base.len() {
        let mut xp = base.clone();
        xp[i] += h;
        let mut xm = base.clone();
        xm[i] -= h;
        let fp = f(&Var::leaf(Tensor::from_vec(&input.shape_i64(), xp)?, false))?
            .value()
            .to_vec()[0];
        let fm = f(&Var::leaf(Tensor::from_vec(&input.shape_i64(), xm)?, false))?
            .value()
            .to_vec()[0];
        let numeric = (fp - fm) / (2.0 * h);
        let scale = tol * analytic[i].abs().max(1.0);
        if (analytic[i] - numeric).abs() > scale {
            ok = false;
        }
    }
    Ok(ok)
}

/// Non-scalar variant: reduces `f(input)` with a fixed random seed (sum of
/// `output * weights` with deterministic weights) before checking.
///
/// # Errors
/// Propagates any op error.
pub fn check_with_weights(
    input: &Tensor,
    weights: &Tensor,
    f: impl Fn(&Var) -> OxitorchResult<Var>,
    tol: f32,
) -> OxitorchResult<bool> {
    let x = Var::leaf(input.clone(), true);
    let out = f(&x)?;
    let dot = ops::mul(&out, &Var::leaf(weights.clone(), false))?;
    let loss = sum_to_scalar(&dot)?;
    backward(&loss, None)?;

    let analytic = x.grad().map_or_else(Vec::new, |g| g.to_vec());
    let base = input.to_vec();
    let h = 1e-2_f32;
    let mut ok = true;
    for i in 0..base.len() {
        let mut xp = base.clone();
        xp[i] += h;
        let mut xm = base.clone();
        xm[i] -= h;
        let run = |v: Vec<f32>| -> OxitorchResult<f32> {
            let xv = Var::leaf(Tensor::from_vec(&input.shape_i64(), v)?, false);
            let out = f(&xv)?;
            let dot = ops::mul(&out, &Var::leaf(weights.clone(), false))?;
            let loss = sum_to_scalar(&dot)?;
            Ok(loss.value().to_vec()[0])
        };
        let numeric = (run(xp)? - run(xm)?) / (2.0 * h);
        let scale = tol * analytic[i].abs().max(1.0);
        if (analytic[i] - numeric).abs() > scale {
            ok = false;
        }
    }
    Ok(ok)
}
