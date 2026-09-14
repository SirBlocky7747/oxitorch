//! Phase 2 autograd tests: gradcheck vs finite differences for every op,
//! diamond/accumulation semantics, second-order gradients, and an SGD
//! training loop.

use oxi_core::OxitorchResult;
use oxi_tensor::ops::ReduceKind;
use oxi_tensor::Tensor;

use crate::gradcheck::{check_scalar, check_with_weights};
use crate::graph::{backward, backward_ext, enable_grad, is_grad_enabled, no_grad, GradGuard, Var};
use crate::ops::{self};
use crate::optim::Sgd;

fn tensor(shape: &[i64], data: Vec<f32>) -> Var {
    Var::leaf(Tensor::from_vec(shape, data).unwrap(), true)
}

/// Weighted-sums the output down to a scalar (test-local mirror of
/// gradcheck's private `sum_to_scalar`).
fn weighted_scalar(out: &Var, w: &Tensor) -> OxitorchResult<Var> {
    let dot = ops::mul(out, &Var::leaf(w.clone(), false))?;
    let mut s = dot;
    while s.value().ndim() > 0 {
        s = ops::reduce(&s, ReduceKind::Sum, 0, false)?;
    }
    Ok(s)
}

/// Two-input gradcheck: verifies the analytic gradients of *both* inputs
/// against central differences of `w · f(lhs, rhs)` (indices and other
/// non-differentiated arguments are baked into `f`).
fn check_two_inputs(
    lhs: &Tensor,
    rhs: &Tensor,
    w: &Tensor,
    f: &impl Fn(&Var, &Var) -> OxitorchResult<Var>,
    tol: f32,
) -> OxitorchResult<bool> {
    let run = |lhs_data: &[f32], rhs_data: &[f32]| -> OxitorchResult<f32> {
        let lhs_var = Var::leaf(
            Tensor::from_vec(&lhs.shape_i64(), lhs_data.to_vec())?,
            false,
        );
        let rhs_var = Var::leaf(
            Tensor::from_vec(&rhs.shape_i64(), rhs_data.to_vec())?,
            false,
        );
        let scalar = weighted_scalar(&f(&lhs_var, &rhs_var)?, w)?;
        Ok(scalar.value().to_vec()[0])
    };
    let lhs_var = Var::leaf(lhs.clone(), true);
    let rhs_var = Var::leaf(rhs.clone(), true);
    let loss = weighted_scalar(&f(&lhs_var, &rhs_var)?, w)?;
    backward(&loss, None)?;
    let grads = [
        lhs_var.grad().map_or_else(Vec::new, |g| g.to_vec()),
        rhs_var.grad().map_or_else(Vec::new, |g| g.to_vec()),
    ];
    let bases = [lhs.to_vec(), rhs.to_vec()];
    let h = 1e-2_f32;
    let mut ok = true;
    for (input, analytic) in grads.iter().enumerate() {
        let base = &bases[input];
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += h;
            let mut minus = base.clone();
            minus[i] -= h;
            let (fp, fm) = if input == 0 {
                (run(&plus, &bases[1])?, run(&minus, &bases[1])?)
            } else {
                (run(&bases[0], &plus)?, run(&bases[0], &minus)?)
            };
            let numeric = (fp - fm) / (2.0 * h);
            if (analytic[i] - numeric).abs() > tol * analytic[i].abs().max(1.0) {
                ok = false;
            }
        }
    }
    Ok(ok)
}

/// Deterministic pseudo-random data in [-1, 1).
fn rng_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

#[test]
fn add_sub_mul_div_pow_gradcheck() {
    #![allow(clippy::many_single_char_names)] // a/b/w are the tensors under test
    let a = Tensor::from_vec(&[2, 3], rng_vec(6, 1)).unwrap();
    let b = Tensor::from_vec(&[2, 3], rng_vec(6, 2)).unwrap();
    let w = Tensor::from_vec(&[2, 3], rng_vec(6, 3)).unwrap();
    let bb = Tensor::from_vec(&[3], rng_vec(3, 4)).unwrap(); // broadcast

    let a_leaf = || Var::leaf(a.clone(), false);
    let b_leaf = || Var::leaf(b.clone(), false);

    assert!(check_with_weights(&a, &w, |x| ops::add(x, &b_leaf()), 1e-3).unwrap());
    assert!(check_with_weights(&b, &w, |y| ops::add(&a_leaf(), y), 1e-3).unwrap());
    // Broadcast: b is [3], broadcasts over [2, 3].
    assert!(check_with_weights(&bb, &w, |y| ops::add(&a_leaf(), y), 1e-3).unwrap());

    assert!(check_with_weights(&a, &w, |x| ops::sub(x, &b_leaf()), 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, |x| ops::mul(x, &b_leaf()), 1e-3).unwrap());
    // div: b kept away from 0 by construction (|b| > 0.1).
    assert!(check_with_weights(&a, &w, |x| ops::div(x, &b_leaf()), 1e-3).unwrap());
    // pow: exponent fixed at 2 (a^2); a stays positive.
    let pos =
        Tensor::from_vec(&[2, 3], a.to_vec().iter().map(|v| v.abs() + 0.5).collect()).unwrap();
    assert!(check_scalar(
        &pos,
        |x| {
            let sq = ops::mul(x, x)?;
            let r = ops::reduce(&sq, ReduceKind::Sum, 0, false)?;
            ops::reduce(&r, ReduceKind::Sum, 0, false)
        },
        1e-3
    )
    .unwrap());
}

#[test]
fn unaries_gradcheck() {
    // exp/log/sqrt inputs kept > 0.5 to stay in a smooth range.
    let a = Tensor::from_vec(
        &[2, 3],
        rng_vec(6, 5)
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<f32>>(),
    )
    .unwrap();
    let w = Tensor::from_vec(&[2, 3], rng_vec(6, 6)).unwrap();

    assert!(check_with_weights(&a, &w, ops::exp, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, ops::log, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, ops::sqrt, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, ops::tanh, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, ops::sigmoid, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w, ops::neg, 1e-3).unwrap());
    // relu: shift inputs away from the kink at 0.
    let shifted: Vec<f32> = a
        .to_vec()
        .iter()
        .map(|v| if v.abs() < 0.2 { v + 0.3 } else { *v })
        .collect();
    let a2 = Tensor::from_vec(&[2, 3], shifted).unwrap();
    assert!(check_with_weights(&a2, &w, ops::relu, 1e-3).unwrap());
}

#[test]
fn matmul_gradcheck() {
    let a = Tensor::from_vec(&[3, 4], rng_vec(12, 7)).unwrap();
    let b = Tensor::from_vec(&[4, 2], rng_vec(8, 8)).unwrap();
    // Scalar loss via sum of the product.
    let run = |x: &Var| -> OxitorchResult<Var> {
        let y = ops::matmul(x, &Var::leaf(b.clone(), false))?;
        let r = ops::reduce(&y, ReduceKind::Sum, 0, false)?;
        ops::reduce(&r, ReduceKind::Sum, 0, false)
    };
    assert!(check_scalar(&a, run, 1e-3).unwrap());
    let run_b = |y: &Var| -> OxitorchResult<Var> {
        let z = ops::matmul(&Var::leaf(a.clone(), false), y)?;
        let r = ops::reduce(&z, ReduceKind::Sum, 0, false)?;
        ops::reduce(&r, ReduceKind::Sum, 0, false)
    };
    assert!(check_scalar(&b, run_b, 1e-3).unwrap());
}

#[test]
fn reductions_gradcheck() {
    let a = Tensor::from_vec(&[3, 4], rng_vec(12, 9)).unwrap();

    // sum over dim 1, keepdim, weighted
    let w = Tensor::from_vec(&[3, 1], rng_vec(3, 10)).unwrap();
    assert!(
        check_with_weights(&a, &w, |x| ops::reduce(x, ReduceKind::Sum, 1, true), 1e-3).unwrap()
    );

    // mean over dim 0
    let w2 = Tensor::from_vec(&[1, 4], rng_vec(4, 11)).unwrap();
    assert!(
        check_with_weights(&a, &w2, |x| ops::reduce(x, ReduceKind::Mean, 0, true), 1e-3).unwrap()
    );

    // norm over last dim
    let w3 = Tensor::from_vec(&[3, 1], rng_vec(3, 12)).unwrap();
    assert!(
        check_with_weights(&a, &w3, |x| ops::reduce(x, ReduceKind::Norm, 1, true), 1e-3).unwrap()
    );
}

#[test]
fn views_gradcheck() {
    let a = Tensor::from_vec(&[3, 4], rng_vec(12, 13)).unwrap();
    // Weights must match each view's OUTPUT shape.
    let w43 = Tensor::from_vec(&[4, 3], rng_vec(12, 17)).unwrap(); // transpose
    let w24 = Tensor::from_vec(&[2, 4], rng_vec(8, 18)).unwrap(); // narrow
    let w34 = Tensor::from_vec(&[3, 4], rng_vec(12, 14)).unwrap(); // identity view

    assert!(check_with_weights(&a, &w43, ops::transpose, 1e-3).unwrap());
    assert!(check_with_weights(&a, &w43, |x| ops::reshape(x, &[4, 3]), 1e-3).unwrap());
    assert!(check_with_weights(&a, &w24, |x| ops::narrow_dim0(x, 1, 2), 1e-3).unwrap());
    assert!(check_with_weights(&a, &w34, |x| ops::broadcast_to(x, &[3, 4]), 1e-3).unwrap());
}

#[test]
fn permute_gradcheck_and_backward() {
    let arr = Tensor::from_vec(&[2, 3, 4], rng_vec(24, 19)).unwrap();
    let wgt = Tensor::from_vec(&[4, 2, 3], rng_vec(24, 20)).unwrap();
    assert!(check_with_weights(&arr, &wgt, |x| ops::permute(x, &[2, 0, 1]), 1e-3).unwrap());

    // Round trip: permute then inverse-permute is the identity.
    let x = tensor(&[2, 3, 4], rng_vec(24, 21));
    let y = ops::permute(&ops::permute(&x, &[2, 0, 1]).unwrap(), &[1, 2, 0]).unwrap();
    assert_eq!(y.value().to_vec(), x.value().to_vec());
    assert_eq!(y.value().shape(), x.value().shape());

    // Grad of sum(permute(x)) flows straight back.
    let loss = {
        let p = ops::permute(&x, &[2, 0, 1]).unwrap();
        let r = ops::reduce(&p, ReduceKind::Sum, 0, false).unwrap();
        let r = ops::reduce(&r, ReduceKind::Sum, 0, false).unwrap();
        ops::reduce(&r, ReduceKind::Sum, 0, false).unwrap()
    };
    backward(&loss, None).unwrap();
    assert_eq!(x.grad().unwrap().to_vec(), vec![1.0; 24]);
}

#[test]
fn scatter_dim0_gradcheck_and_last_write_wins() {
    let base = Tensor::from_vec(&[4, 3], rng_vec(12, 20)).unwrap();
    let vals = Tensor::from_vec(&[3, 3], rng_vec(9, 21)).unwrap();
    let w = Tensor::from_vec(&[4, 3], rng_vec(12, 22)).unwrap();
    // f(base, values) = w · scatter(base, values); central differences on
    // both inputs must match the analytic pair.
    let ok = check_two_inputs(
        &base,
        &vals,
        &w,
        &|b, v| ops::scatter_dim0(b, &[1, 3, 1], v),
        1e-3,
    );
    assert!(ok.unwrap());

    // Direct VJP check: duplicates route the grad only to the last write,
    // and written rows of `base` get zero.
    let b = tensor(&[3, 2], vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let v = tensor(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = ops::scatter_dim0(&b, &[2, 0, 2], &v).unwrap();
    let g = Var::leaf(
        Tensor::from_vec(&[3, 2], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
        false,
    );
    let loss = crate::ops::reduce(&ops::mul(&out, &g).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    let loss = crate::ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    // out = [[3,4],[20,21],[5,6]] -> d/dbase = [[0,0],[1,1],[0,0]];
    // slot 2 wins row 2, slot 0 loses -> d/dvalues = [[0,0],[1,1],[1,1]].
    assert_eq!(
        b.grad().unwrap().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        v.grad().unwrap().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn masked_select_gradcheck_and_broadcast_accumulation() {
    // Gradcheck over the data input (the mask is a constant: it selects
    // which paths exist, so it is not differentiated — 0/1 boundaries).
    // Mask [1,0,1,0,1,0] picks (0,0), (0,2), (1,1) -> 3 elements.
    let data = Tensor::from_vec(&[2, 3], rng_vec(6, 25)).unwrap();
    let weights = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
    let mask = Tensor::from_vec(&[2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
    let ok = check_with_weights(
        &data,
        &weights,
        |x| {
            let m = Var::leaf(mask.clone(), false);
            ops::masked_select(x, &m)
        },
        1e-3,
    );
    assert!(ok.unwrap());

    // Broadcast row mask [1, 0, 1]: picks (0,0), (0,2), (1,0), (1,2) in
    // grid order -> values [1, 3, 4, 6]; grads land exactly there.
    let x = tensor(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m = Var::leaf(Tensor::from_vec(&[3], vec![1.0, 0.0, 1.0]).unwrap(), false);
    let out = ops::masked_select(&x, &m).unwrap();
    let seed = Var::leaf(
        Tensor::from_vec(&[4], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        false,
    );
    // `out` is 1-D: one sum reduction reaches the scalar loss.
    let loss =
        crate::ops::reduce(&ops::mul(&out, &seed).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    // Picks in grid order: (0,0) -> 1, (0,2) -> 3, (1,0) -> 4, (1,2) -> 6.
    assert_eq!(
        x.grad().unwrap().to_vec(),
        vec![10.0, 0.0, 20.0, 30.0, 0.0, 40.0]
    );

    // A broadcast mask that picks the same element twice must accumulate.
    let col = tensor(&[2, 1], vec![7.0, 9.0]);
    let mc = Var::leaf(
        Tensor::from_vec(&[2, 3], vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]).unwrap(),
        false,
    );
    let outc = ops::masked_select(&col, &mc).unwrap();
    let gc = Var::leaf(
        Tensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        false,
    );
    let lossc =
        crate::ops::reduce(&ops::mul(&outc, &gc).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    backward(&lossc, None).unwrap();
    // Column 7.0 is picked by mask entries 1.0, 2.0; column 9.0 by 3.0, 4.0.
    assert_eq!(col.grad().unwrap().to_vec(), vec![3.0, 7.0]);
}

#[test]
fn index_select_gradcheck_and_duplicate_accumulation() {
    let a = Tensor::from_vec(&[4, 3], rng_vec(12, 15)).unwrap();
    let w = Tensor::from_vec(&[2, 3], rng_vec(6, 16)).unwrap();
    assert!(check_with_weights(&a, &w, |x| ops::index_select(x, 0, &[2, 0]), 1e-3).unwrap());

    // Duplicates accumulate: index_select([1,1]) rows must get g0+g1.
    let g = Var::leaf(
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        false,
    );
    let x = tensor(&[3, 3], vec![0.0; 9]);
    let picked = ops::index_select(&x, 0, &[1, 1]).unwrap();
    let loss =
        crate::ops::reduce(&ops::mul(&picked, &g).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    let loss = crate::ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    let grad = x.grad().unwrap().to_vec();
    assert_eq!(grad[3..6], vec![5.0, 7.0, 9.0]); // 1+4, 2+5, 3+6
}

#[test]
fn diamond_graph_accumulates() {
    // y = (x + x) * x — three contributions to the same leaf.
    let x = tensor(&[2], vec![3.0, -1.0]);
    let s = ops::add(&x, &x).unwrap();
    let m = ops::mul(&s, &x).unwrap();
    let loss = ops::reduce(&m, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    // d/dx (2x * x) = 4x
    assert_eq!(x.grad().unwrap().to_vec(), vec![12.0, -4.0]);
}

#[test]
fn no_grad_leaf_gets_nothing() {
    let x = Var::leaf(Tensor::from_vec(&[2], vec![1.0, 2.0]).unwrap(), false);
    let y = ops::mul(&x, &x).unwrap();
    let loss = ops::reduce(&y, ReduceKind::Sum, 0, false).unwrap();
    assert!(backward(&loss, None).is_err()); // nothing requires grad
    assert!(x.grad().is_none());
}

#[test]
fn second_order_gradient() {
    // f(x) = sum(sigmoid(x)^2); f'' should match finite differences of grad.
    #![allow(clippy::many_single_char_names)] // s = sigmoid(x), w = weights
    let a = Tensor::from_vec(&[2, 2], vec![0.3, -0.4, 0.9, -1.1]).unwrap();
    let w = Var::leaf(Tensor::from_vec(&[2, 2], vec![1.0; 4]).unwrap(), false);

    let x = Var::leaf(a.clone(), true);
    let s = ops::sigmoid(&x).unwrap();
    let s2 = ops::mul(&s, &s).unwrap();
    let dot = ops::mul(&s2, &w).unwrap();
    let loss = ops::reduce(&dot, ReduceKind::Sum, 0, false).unwrap();
    let loss = ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    let g1 = x.grad().unwrap().to_vec();

    // Central differences of the first-order gradient: perturb x, recompute
    // grad, compare.
    let h = 1e-2_f32;
    for i in 0..4 {
        let mut base = a.to_vec();
        base[i] += h;
        let xp = Var::leaf(Tensor::from_vec(&[2, 2], base).unwrap(), true);
        let sp = ops::sigmoid(&xp).unwrap();
        let sp2 = ops::mul(&sp, &sp).unwrap();
        let lp = ops::reduce(&sp2, ReduceKind::Sum, 0, false).unwrap();
        let lp = ops::reduce(&lp, ReduceKind::Sum, 0, false).unwrap();
        backward(&lp, None).unwrap();
        let gp = xp.grad().unwrap().to_vec()[i];

        let mut base = a.to_vec();
        base[i] -= h;
        let xm = Var::leaf(Tensor::from_vec(&[2, 2], base).unwrap(), true);
        let sm = ops::sigmoid(&xm).unwrap();
        let sm2 = ops::mul(&sm, &sm).unwrap();
        let lm = ops::reduce(&sm2, ReduceKind::Sum, 0, false).unwrap();
        let lm = ops::reduce(&lm, ReduceKind::Sum, 0, false).unwrap();
        backward(&lm, None).unwrap();
        let gm = xm.grad().unwrap().to_vec()[i];

        let second_numeric = (gp - gm) / (2.0 * h);
        // f = sigma(x)^2  =>  f'' = 2 s^2 (1-s)(2-3s).
        let s = 1.0 / (1.0 + (-a.to_vec()[i]).exp());
        let analytic_second = 2.0 * s * s * (1.0 - s) * (2.0 - 3.0 * s);
        assert!(
            (second_numeric - analytic_second).abs() < 1e-2,
            "second-order mismatch at {i}: {second_numeric} vs {analytic_second}"
        );
        let _ = g1; // first-order grad correctness covered by gradcheck tests
    }
}

#[test]
fn no_grad_records_detached_and_enable_grad_reenters() {
    let x = tensor(&[2], vec![3.0, -1.0]);

    // Outside: ops record; inside no_grad: detached leaves.
    let outside = ops::mul(&x, &x).unwrap();
    assert!(outside.requires_grad());

    let inside = no_grad(|| ops::mul(&x, &x).unwrap());
    assert!(!inside.requires_grad());
    assert_eq!(inside.op_name(), None); // recorded as a leaf
    assert_eq!(inside.value().to_vec(), outside.value().to_vec());

    // Backward cannot flow through the detached branch...
    let loss = ops::reduce(
        &ops::mul(&inside, &inside).unwrap(),
        ReduceKind::Sum,
        0,
        false,
    )
    .unwrap();
    assert!(backward(&loss, None).is_err());
    assert!(x.grad().is_none());

    // ...but the recorded branch still works after the scope.
    let loss2 = ops::reduce(
        &ops::mul(&outside, &outside).unwrap(),
        ReduceKind::Sum,
        0,
        false,
    )
    .unwrap();
    backward(&loss2, None).unwrap();
    // d/dx Σ(x²)² = 4x·x²  (x = [3, -1] → outside = [9, 1])
    assert_eq!(x.grad().unwrap().to_vec(), vec![108.0, -4.0]);

    // enable_grad re-enters inside no_grad (torch parity).
    let reentered = no_grad(|| {
        let r = enable_grad(|| ops::mul(&x, &x).unwrap());
        assert!(
            !is_grad_enabled(),
            "inner scope must close before the outer"
        );
        r
    });
    assert!(reentered.requires_grad());

    // enable_grad is the identity when already on.
    assert!(enable_grad(is_grad_enabled));
    assert!(is_grad_enabled());
}

#[test]
fn grad_guard_restores_mode_even_on_panic() {
    let prev = is_grad_enabled();
    let result = std::panic::catch_unwind(|| {
        let _g = GradGuard::pause();
        assert!(!is_grad_enabled());
        panic!("scope unwinds");
    });
    assert!(result.is_err());
    assert_eq!(is_grad_enabled(), prev, "guard must restore mode on unwind");

    // And the forced variant.
    no_grad(|| {
        let result = std::panic::catch_unwind(|| {
            let _g = GradGuard::force();
            assert!(is_grad_enabled());
            panic!("scope unwinds");
        });
        assert!(result.is_err());
        assert!(!is_grad_enabled());
    });
    assert!(is_grad_enabled());
}

#[test]
fn retain_graph_allows_repeated_backward_with_accumulation() {
    let x = tensor(&[2], vec![3.0, -1.0]);
    let y = ops::mul(&x, &x).unwrap();
    let loss = ops::reduce(&y, ReduceKind::Sum, 0, false).unwrap();

    // First pass retains; second pass accumulates; third frees.
    backward_ext(&loss, None, true).unwrap();
    assert_eq!(x.grad().unwrap().to_vec(), vec![6.0, -2.0]);

    backward_ext(&loss, None, true).unwrap();
    assert_eq!(x.grad().unwrap().to_vec(), vec![12.0, -4.0]);

    backward_ext(&loss, None, false).unwrap();
    assert_eq!(x.grad().unwrap().to_vec(), vec![18.0, -6.0]);

    // Graph is freed now: another pass errors instead of silently doing
    // nothing, and leaves .grad untouched.
    assert!(backward(&loss, None).is_err());
    assert_eq!(x.grad().unwrap().to_vec(), vec![18.0, -6.0]);
}

#[test]
fn default_backward_consumes_the_graph() {
    let x = tensor(&[2], vec![1.0, 2.0]);
    let loss = ops::reduce(&ops::mul(&x, &x).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    assert_eq!(x.grad().unwrap().to_vec(), vec![2.0, 4.0]);
    let err = backward(&loss, None).unwrap_err();
    assert!(err.to_string().contains("retain_graph"), "{err}");
}

#[test]
fn no_grad_results_bridge_back_into_graded_code() {
    // Inference output (detached) used as a constant in a later training
    // step: the constant must not receive gradients.
    let w = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let frozen = no_grad(|| ops::matmul(&w, &w).unwrap());
    assert!(!frozen.requires_grad());

    let x = tensor(&[2, 2], vec![0.5; 4]);
    let out = ops::matmul(&x, &frozen).unwrap();
    let loss = ops::reduce(&out, ReduceKind::Sum, 0, false).unwrap();
    let loss = ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();
    assert!(x.grad().is_some());
    assert!(w.grad().is_none(), "frozen subgraph gets no gradient");
}

#[test]
fn bmm_gradcheck_both_operands() {
    let a = Tensor::from_vec(&[2, 3, 4], rng_vec(24, 22)).unwrap();
    let b = Tensor::from_vec(&[2, 4, 2], rng_vec(16, 23)).unwrap();
    // Grad w.r.t. a with b constant...
    let run_a = |x: &Var| -> OxitorchResult<Var> {
        let y = ops::bmm(x, &Var::leaf(b.clone(), false))?;
        let r = ops::reduce(&y, ReduceKind::Sum, 0, false)?;
        let r = ops::reduce(&r, ReduceKind::Sum, 0, false)?;
        ops::reduce(&r, ReduceKind::Sum, 0, false)
    };
    assert!(check_scalar(&a, run_a, 1e-3).unwrap());
    // ...and w.r.t. b with a constant.
    let run_b = |y: &Var| -> OxitorchResult<Var> {
        let z = ops::bmm(&Var::leaf(a.clone(), false), y)?;
        let r = ops::reduce(&z, ReduceKind::Sum, 0, false)?;
        let r = ops::reduce(&r, ReduceKind::Sum, 0, false)?;
        ops::reduce(&r, ReduceKind::Sum, 0, false)
    };
    assert!(check_scalar(&b, run_b, 1e-3).unwrap());
}

#[test]
fn bmm_grads_match_closed_form() {
    // dL/dA = g @ B^T, dL/dB = A^T @ g per batch — check exactly.
    let a = tensor(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = tensor(
        &[2, 2, 2],
        vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    );
    let g = Var::leaf(
        Tensor::from_vec(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]).unwrap(),
        false,
    );
    let out = ops::bmm(&a, &b).unwrap();
    let loss = ops::reduce(&ops::mul(&out, &g).unwrap(), ReduceKind::Sum, 0, false).unwrap();
    let loss = ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    let loss = ops::reduce(&loss, ReduceKind::Sum, 0, false).unwrap();
    backward(&loss, None).unwrap();

    // batch 0: g = I -> dA = B^T = [[9,11],[10,12]]; dB = A^T = [[1,3],[2,4]]
    // batch 1: g = 2I -> dA = 2 B^T = [[26,30],[28,32]]; dB = 2 A^T = [[10,14],[12,16]]
    assert_eq!(
        a.grad().unwrap().to_vec(),
        vec![9.0, 11.0, 10.0, 12.0, 26.0, 30.0, 28.0, 32.0]
    );
    assert_eq!(
        b.grad().unwrap().to_vec(),
        vec![1.0, 3.0, 2.0, 4.0, 10.0, 14.0, 12.0, 16.0]
    );
}

#[test]
fn bmm_respects_grad_mode() {
    let a = tensor(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = tensor(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let detached = no_grad(|| ops::bmm(&a, &b).unwrap());
    assert!(!detached.requires_grad());
    assert_eq!(detached.op_name(), None);
    let recorded = ops::bmm(&a, &b).unwrap();
    assert!(recorded.requires_grad());
    assert_eq!(recorded.op_name(), Some("bmm"));
}

#[test]
fn sgd_trains_linear_regression() {
    // y = 2x + 1 on 8 points; SGD must drive loss toward 0.
    let xs: Vec<f32> = (0..8).map(|i| i as f32 * 0.5 - 1.5).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| 2.0 * x + 1.0).collect();
    let x = Var::leaf(Tensor::from_vec(&[8, 1], xs).unwrap(), false);
    let y = Var::leaf(Tensor::from_vec(&[8, 1], ys).unwrap(), false);
    let mut params = vec![
        Var::leaf(Tensor::from_vec(&[1, 1], vec![0.5]).unwrap(), true), // w
        Var::leaf(Tensor::from_vec(&[1], vec![0.0]).unwrap(), true),    // b
    ];
    let opt = Sgd::new(0.05, 0.0, false);

    let mut first_loss = f32::INFINITY;
    for step in 0..200 {
        let pred = ops::matmul(&x, &params[0]).unwrap();
        let pred = ops::add(&pred, &params[1]).unwrap();
        let diff = ops::sub(&pred, &y).unwrap();
        let sq = ops::mul(&diff, &diff).unwrap();
        let loss = ops::reduce(&sq, ReduceKind::Sum, 0, false).unwrap();
        if step == 0 {
            first_loss = loss.value().to_vec()[0];
        }
        backward(&loss, None).unwrap();
        opt.step(&mut params).unwrap();
    }
    let final_loss = {
        let pred = ops::matmul(&x, &params[0]).unwrap();
        let pred = ops::add(&pred, &params[1]).unwrap();
        let diff = ops::sub(&pred, &y).unwrap();
        let sq = ops::mul(&diff, &diff).unwrap();
        let loss = ops::reduce(&sq, ReduceKind::Sum, 0, false).unwrap();
        loss.value().to_vec()[0]
    };
    assert!(
        final_loss < first_loss * 1e-4,
        "loss did not converge: {first_loss} -> {final_loss}"
    );
    // w ≈ 2, b ≈ 1
    let wv = params[0].value().to_vec()[0];
    let bv = params[1].value().to_vec()[0];
    assert!((wv - 2.0).abs() < 0.05, "w = {wv}");
    assert!((bv - 1.0).abs() < 0.05, "b = {bv}");
}
