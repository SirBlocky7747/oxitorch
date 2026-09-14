//! Optimizers over the autograd `Var` graph (plan.md Phase 2/4).
//!
//! The optimizers here own **all** optimizer state (momentum buffers, Adam
//! moments, per-parameter step counters) and perform every update on flat
//! `f32` slices — no tensor ops, no numpy round-trip. Rust-side so a
//! training loop's `step()` never marshals parameters across the Python
//! boundary: the pyo3 wrappers (see `oxi-bindings`) hand over the parameter
//! list each step and swap every `Var` in place, keeping Python object
//! identity stable across the whole run.
//!
//! State is lazily allocated per **parameter index**, mirroring torch's
//! per-parameter `state`: a parameter that first produces a gradient on step
//! k initializes its buffers (and bias-correction counter) on that step, and
//! a parameter skipped on some step keeps its buffers untouched.

use std::cell::{Cell, RefCell};

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

use crate::graph::Var;

/// Wraps updated values as a fresh `requires_grad` leaf with the same shape
/// (parameters are rebuilt as leaves each step; the graph re-records on the
/// next forward pass).
fn rewrap_leaf(param: &Var, data: Vec<f32>) -> OxitorchResult<Var> {
    let shape: Vec<i64> = param.value().shape().iter().map(|&d| d as i64).collect();
    Ok(Var::leaf(Tensor::from_vec(&shape, data)?, true))
}

/// Lazy per-index slot access: grows the slot vec to cover `index` and
/// returns the slot (initializing with `init` if empty).
fn slot(
    slots: &mut Vec<Option<Vec<f32>>>,
    index: usize,
    len: usize,
    init: impl Fn() -> Vec<f32>,
) -> OxitorchResult<&mut Vec<f32>> {
    if slots.len() <= index {
        slots.resize(index + 1, None);
    }
    let buf = slots[index].get_or_insert_with(init);
    if buf.len() != len {
        return Err(OxitorchError::InvalidArgument(format!(
            "optimizer state for parameter {index} has {buf_len} elements but the \
             parameter now has {len}; the parameter set changed mid-run",
            buf_len = buf.len(),
        )));
    }
    Ok(buf)
}

/// Lazy per-index scalar counter (torch's per-parameter `state["step"]`).
fn bump_step(steps: &mut Vec<u64>, index: usize) -> u64 {
    if steps.len() <= index {
        steps.resize(index + 1, 0);
    }
    steps[index] += 1;
    steps[index]
}

/// Vanilla stochastic gradient descent with optional (Nesterov) momentum.
///
/// Update (torch semantics, velocity initialized from the first gradient):
/// `buf = momentum * buf + g`; `p -= lr * (nesterov ? g + momentum * buf : buf)`.
pub struct Sgd {
    lr: Cell<f32>,
    momentum: f32,
    nesterov: bool,
    velocities: RefCell<Vec<Option<Vec<f32>>>>,
}

impl Sgd {
    /// Creates an SGD optimizer.
    ///
    /// (Nesterov-without-momentum is rejected at the API boundary; the
    /// engine trusts its caller here.)
    #[must_use]
    pub fn new(lr: f32, momentum: f64, nesterov: bool) -> Self {
        Self {
            lr: Cell::new(lr),
            momentum: momentum as f32,
            nesterov,
            velocities: RefCell::new(Vec::new()),
        }
    }

    /// Current learning rate.
    #[must_use]
    pub fn lr(&self) -> f32 {
        self.lr.get()
    }

    /// Sets the learning rate (schedulers mutate this between steps).
    pub fn set_lr(&self, lr: f32) {
        self.lr.set(lr);
    }

    /// Advances one parameter in place. No-op when the parameter has no
    /// gradient (its velocity buffer stays untouched for next time).
    ///
    /// # Errors
    /// Propagates tensor construction errors and state-size mismatches.
    pub fn step_param(&self, index: usize, param: &mut Var) -> OxitorchResult<()> {
        let Some(grad) = param.grad() else {
            return Ok(());
        };
        let grad_vec = grad.to_vec();
        let lr = self.lr.get();
        let update: Vec<f32> = if self.momentum > 0.0 {
            let mut slots = self.velocities.borrow_mut();
            let buf = slot(&mut slots, index, grad_vec.len(), || {
                vec![0.0; grad_vec.len()]
            })?;
            for (entry, &grad_val) in buf.iter_mut().zip(&grad_vec) {
                *entry = self.momentum * *entry + grad_val;
            }
            if self.nesterov {
                grad_vec
                    .iter()
                    .zip(buf.iter())
                    .map(|(&grad_val, &buf_val)| grad_val + self.momentum * buf_val)
                    .collect()
            } else {
                buf.clone()
            }
        } else {
            grad_vec
        };
        let value = param.value().to_vec();
        let updated: Vec<f32> = value
            .iter()
            .zip(update)
            .map(|(&val, upd)| val - lr * upd)
            .collect();
        *param = rewrap_leaf(param, updated)?;
        Ok(())
    }

    /// Applies one step over a whole parameter list (index = list position).
    ///
    /// # Errors
    /// Propagates [`Sgd::step_param`].
    pub fn step(&self, params: &mut [Var]) -> OxitorchResult<()> {
        for (index, param) in params.iter_mut().enumerate() {
            self.step_param(index, param)?;
        }
        Ok(())
    }

    /// Zeroes gradients for the next iteration.
    pub fn zero_grad(&self, params: &[Var]) {
        for param in params {
            let _ = param.zero_grad();
        }
    }
}

/// Adam with torch's two decay semantics, selected by [`AdamW::decoupled`]:
/// decoupled decay on the parameter (**AdamW**, Loshchilov & Hutter 2019) or
/// decay folded into the gradient before the moments (classic **Adam**,
/// Kingma & Ba 2015).
///
/// Update (per parameter, with its own step counter `t`):
/// `m = β1 m + (1-β1) g_eff`; `v = β2 v + (1-β2) g_eff²`;
/// `p -= lr · (m/(1-β1^t)) / (√(v/(1-β2^t)) + ε)` plus, when decoupled,
/// `lr · weight_decay · p` applied directly to the parameter.
pub struct AdamW {
    lr: Cell<f32>,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    /// `true`: decay applied to the parameter (AdamW); `false`: added to the
    /// gradient before the moment updates (classic Adam).
    decoupled: bool,
    moments: RefCell<Vec<Option<Vec<f32>>>>,
    sq_moments: RefCell<Vec<Option<Vec<f32>>>>,
    steps: RefCell<Vec<u64>>,
}

impl AdamW {
    /// Creates the optimizer.
    #[must_use]
    #[allow(clippy::fn_params_excessive_bools)]
    pub fn new(
        lr: f32,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        decoupled: bool,
    ) -> Self {
        Self {
            lr: Cell::new(lr),
            beta1: beta1 as f32,
            beta2: beta2 as f32,
            eps: eps as f32,
            weight_decay: weight_decay as f32,
            decoupled,
            moments: RefCell::new(Vec::new()),
            sq_moments: RefCell::new(Vec::new()),
            steps: RefCell::new(Vec::new()),
        }
    }

    /// Current learning rate.
    #[must_use]
    pub fn lr(&self) -> f32 {
        self.lr.get()
    }

    /// Sets the learning rate (schedulers mutate this between steps).
    pub fn set_lr(&self, lr: f32) {
        self.lr.set(lr);
    }

    /// Advances one parameter in place. No-op when the parameter has no
    /// gradient (its moments and step counter stay untouched).
    ///
    /// # Errors
    /// Propagates tensor construction errors and state-size mismatches.
    pub fn step_param(&self, index: usize, param: &mut Var) -> OxitorchResult<()> {
        let Some(grad) = param.grad() else {
            return Ok(());
        };
        let grad_vec = grad.to_vec();
        let len = grad_vec.len();
        let value = param.value().to_vec();
        let lr = self.lr.get();
        let t = bump_step(&mut self.steps.borrow_mut(), index);
        let bc1 = 1.0 - self.beta1.powf(t as f32);
        let bc2 = 1.0 - self.beta2.powf(t as f32);
        let mut m_slots = self.moments.borrow_mut();
        let mut v_slots = self.sq_moments.borrow_mut();
        let moment = slot(&mut m_slots, index, len, || vec![0.0; len])?;
        let velocity = slot(&mut v_slots, index, len, || vec![0.0; len])?;
        let updated: Vec<f32> = value
            .iter()
            .zip(&grad_vec)
            .enumerate()
            .map(|(elem, (&val, &grad_val))| {
                let g_eff = if !self.decoupled && self.weight_decay != 0.0 {
                    grad_val + self.weight_decay * val
                } else {
                    grad_val
                };
                moment[elem] = self.beta1 * moment[elem] + (1.0 - self.beta1) * g_eff;
                velocity[elem] = self.beta2 * velocity[elem] + (1.0 - self.beta2) * g_eff * g_eff;
                let m_hat = moment[elem] / bc1;
                let v_hat = velocity[elem] / bc2;
                let mut next = val - lr * m_hat / (v_hat.sqrt() + self.eps);
                if self.decoupled && self.weight_decay != 0.0 {
                    next -= lr * self.weight_decay * val;
                }
                next
            })
            .collect();
        *param = rewrap_leaf(param, updated)?;
        Ok(())
    }

    /// Applies one step over a whole parameter list (index = list position).
    ///
    /// # Errors
    /// Propagates [`AdamW::step_param`].
    pub fn step(&self, params: &mut [Var]) -> OxitorchResult<()> {
        for (index, param) in params.iter_mut().enumerate() {
            self.step_param(index, param)?;
        }
        Ok(())
    }

    /// Zeroes gradients for the next iteration.
    pub fn zero_grad(&self, params: &[Var]) {
        for param in params {
            let _ = param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::backward;
    use oxi_tensor::ops::ReduceKind;

    /// A scalar leaf trained against a fixed gradient, returning its value
    /// after `step_param`.
    fn step_scalar(opt: &impl OptimStep, index: usize, start: f32, grad: f32) -> f32 {
        let mut param = Var::leaf(Tensor::from_vec(&[1], vec![start]).unwrap(), true);
        param
            .accumulate_grad(Tensor::from_vec(&[1], vec![grad]).unwrap())
            .unwrap();
        opt.step_param(index, &mut param).unwrap();
        param.value().to_vec()[0]
    }

    /// Object-safe step entry for the helper above.
    trait OptimStep {
        fn step_param(&self, index: usize, param: &mut Var) -> OxitorchResult<()>;
    }
    impl OptimStep for Sgd {
        fn step_param(&self, index: usize, param: &mut Var) -> OxitorchResult<()> {
            Sgd::step_param(self, index, param)
        }
    }
    impl OptimStep for AdamW {
        fn step_param(&self, index: usize, param: &mut Var) -> OxitorchResult<()> {
            AdamW::step_param(self, index, param)
        }
    }

    #[test]
    fn sgd_momentum_matches_reference() {
        let opt = Sgd::new(0.1, 0.9, false);
        // val=1, three steps of constant grad 0.5, lr 0.1, momentum 0.9:
        // buf = [0.5, 0.95, 1.355]; val decreases by lr*buf each step.
        let want = [1.0 - 0.05, 0.95 - 0.095, 0.855 - 0.1355];
        let mut val = 1.0;
        for &expected in &want {
            val = step_scalar(&opt, 0, val, 0.5);
            assert!((val - expected).abs() < 1e-6, "{val} vs {expected}");
        }
    }

    #[test]
    fn sgd_nesterov_uses_lookahead() {
        let opt = Sgd::new(0.1, 0.9, true);
        // Step 1: buf=0.5, update = g + 0.9*buf = 0.5 + 0.45 = 0.95.
        let val = step_scalar(&opt, 0, 1.0, 0.5);
        assert!((val - (1.0 - 0.1 * 0.95)).abs() < 1e-6);
    }

    #[test]
    fn sgd_skipped_param_keeps_velocity() {
        let opt = Sgd::new(0.1, 0.9, false);
        let first = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        first
            .accumulate_grad(Tensor::from_vec(&[1], vec![0.5]).unwrap())
            .unwrap();
        let second = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        let mut params = [first, second];
        opt.step(&mut params).unwrap(); // second has no grad yet
                                        // Next step: second's velocity initializes fresh (no stale momentum).
        let first_next = Var::leaf(Tensor::from_vec(&[1], vec![0.95]).unwrap(), true);
        first_next
            .accumulate_grad(Tensor::from_vec(&[1], vec![0.5]).unwrap())
            .unwrap();
        let second_next = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        second_next
            .accumulate_grad(Tensor::from_vec(&[1], vec![0.5]).unwrap())
            .unwrap();
        let mut params = [first_next, second_next];
        opt.step(&mut params).unwrap();
        // second: buf = 0.5 (fresh), val = 1 - 0.05.
        assert!((params[1].value().to_vec()[0] - 0.95).abs() < 1e-6);
        // first: buf = 0.9*0.5 + 0.5 = 0.95, val = 0.95 - 0.095.
        assert!((params[0].value().to_vec()[0] - (0.95 - 0.095)).abs() < 1e-6);
    }

    #[test]
    fn adamw_two_steps_match_closed_form() {
        let opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1, true);
        let (lr, b1, b2, eps, wd, grad) = (0.1f64, 0.9, 0.999, 1e-8, 0.1, 0.5f64);
        let (mut val, mut moment, mut vel) = (1.0f64, 0.0, 0.0);
        for step in 1..=2 {
            moment = b1 * moment + (1.0 - b1) * grad;
            vel = b2 * vel + (1.0 - b2) * grad * grad;
            let m_hat = moment / (1.0 - b1.powi(step));
            let v_hat = vel / (1.0 - b2.powi(step));
            let want = val - lr * m_hat / (v_hat.sqrt() + eps) - lr * wd * val;
            let got = step_scalar(&opt, 0, val as f32, grad as f32);
            assert!(
                (f64::from(got) - want).abs() < 1e-5,
                "step {step}: engine {got} vs closed form {want}"
            );
            val = want;
        }
    }

    #[test]
    fn adam_classic_decay_folds_into_gradient() {
        // decoupled=false: g_eff = g + wd*w feeds the moments; no param term.
        let opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1, false);
        let start = 1.0f32;
        let grad = 0.5f32;
        let wd = 0.1f32;
        let g_eff = grad + wd * start;
        let moment = (1.0 - 0.9) * g_eff;
        let vel = (1.0 - 0.999) * g_eff * g_eff;
        let m_hat = moment / (1.0 - 0.9);
        let v_hat = vel / (1.0 - 0.999);
        let want = start - 0.1 * m_hat / (v_hat.sqrt() + 1e-8);
        let got = step_scalar(&opt, 0, start, grad);
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    #[test]
    fn per_param_step_counters_track_skips() {
        // Torch semantics: a parameter that skips a step does not advance its
        // own bias-correction counter.
        let opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0, true);
        let first = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        first
            .accumulate_grad(Tensor::from_vec(&[1], vec![0.5]).unwrap())
            .unwrap();
        let second = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        let mut params = [first, second];
        opt.step(&mut params).unwrap(); // second skipped
                                        // second's first real step must use t=1 (bc = 0.1), not t=2.
        let second_next = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        second_next
            .accumulate_grad(Tensor::from_vec(&[1], vec![0.5]).unwrap())
            .unwrap();
        let first_again = Var::leaf(Tensor::from_vec(&[1], vec![1.0]).unwrap(), true);
        let mut params = [first_again, second_next];
        opt.step(&mut params).unwrap();
        let grad = 0.5f64;
        let moment = 0.1 * grad;
        let vel = 0.001 * grad * grad;
        let want = 1.0 - 0.1 * (moment / 0.1) / ((vel / 0.001).sqrt() + 1e-8);
        assert!(
            (f64::from(params[1].value().to_vec()[0]) - want).abs() < 1e-5,
            "{} vs {want}",
            params[1].value().to_vec()[0]
        );
    }

    #[test]
    fn optimizer_trains_linear_regression_end_to_end() {
        // AdamW over the classic linear fit — the full engine path
        // (forward → backward → step) with a two-parameter model.
        let inputs: Vec<f32> = (0..8).map(|idx| idx as f32 * 0.5 - 1.5).collect();
        let targets: Vec<f32> = inputs.iter().map(|&xv| 2.0 * xv + 1.0).collect();
        let x = Var::leaf(Tensor::from_vec(&[8, 1], inputs).unwrap(), false);
        let y = Var::leaf(Tensor::from_vec(&[8, 1], targets).unwrap(), false);
        let mut params = vec![
            Var::leaf(Tensor::from_vec(&[1, 1], vec![0.5]).unwrap(), true),
            Var::leaf(Tensor::from_vec(&[1], vec![0.0]).unwrap(), true),
        ];
        let opt = AdamW::new(0.05, 0.9, 0.999, 1e-8, 0.0, true);
        for _ in 0..200 {
            let pred = crate::ops::matmul(&x, &params[0]).unwrap();
            let pred = crate::ops::add(&pred, &params[1]).unwrap();
            let diff = crate::ops::sub(&pred, &y).unwrap();
            let sq = crate::ops::mul(&diff, &diff).unwrap();
            let loss = crate::ops::reduce(&sq, ReduceKind::Sum, 0, false).unwrap();
            backward(&loss, None).unwrap();
            opt.step(&mut params).unwrap();
        }
        let weight = params[0].value().to_vec()[0];
        let bias = params[1].value().to_vec()[0];
        assert!((weight - 2.0).abs() < 0.05, "w = {weight}");
        assert!((bias - 1.0).abs() < 0.05, "b = {bias}");
    }

    #[test]
    fn state_mismatch_is_an_error() {
        let opt = Sgd::new(0.1, 0.9, false);
        let first = Var::leaf(Tensor::from_vec(&[2], vec![1.0, 2.0]).unwrap(), true);
        first
            .accumulate_grad(Tensor::from_vec(&[2], vec![0.5, 0.5]).unwrap())
            .unwrap();
        let mut params = [first];
        opt.step(&mut params).unwrap();
        // Same index, different element count: state no longer fits.
        let reshaped = Var::leaf(Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap(), true);
        reshaped
            .accumulate_grad(Tensor::from_vec(&[3], vec![0.5, 0.5, 0.5]).unwrap())
            .unwrap();
        let mut params = [reshaped];
        assert!(opt.step(&mut params).is_err());
    }
}
