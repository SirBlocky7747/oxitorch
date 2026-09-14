//! Computation graph and reverse-mode automatic differentiation (plan.md
//! Phase 2).
//!
//! - [`graph::Var`] — a differentiable tensor handle; ops record VJP
//!   closures that are themselves built from differentiable `Var`
//!   primitives, so second-order gradients work through the same API.
//! - [`graph::backward`] — iterative topological backward with gradient
//!   accumulation and broadcast-aware grad reduction.
//! - [`ops`] — the differentiable op set (elementwise, matmul, reductions,
//!   views, indexing).
//! - [`gradcheck`] — finite-difference verification used by tests and
//!   available for user debugging.
//! - [`optim::Sgd`] — the first optimizer, exercising the engine end to end.

pub mod gradcheck;
pub mod graph;
pub mod ops;
pub mod optim;

#[cfg(test)]
mod tests;

use oxi_core::OxitorchResult;
use oxi_tensor::Tensor;

/// Runs `backward()` from `output` with an implicit all-ones seed.
///
/// # Errors
/// Propagates [`graph::backward`].
pub fn backward(output: &graph::Var) -> OxitorchResult<()> {
    graph::backward(output, None)
}

/// Runs `backward()` from `output` with an explicit gradient seed.
///
/// # Errors
/// Propagates [`graph::backward`].
pub fn backward_with_grad(output: &graph::Var, seed: Tensor) -> OxitorchResult<()> {
    graph::backward(output, Some(seed))
}

/// Convenience: a `requires_grad` leaf from a tensor.
#[must_use]
pub fn param(value: oxi_tensor::Tensor) -> graph::Var {
    graph::Var::leaf(value, true)
}

/// Stub kept for compatibility: reports phase readiness.
///
/// # Errors
/// [`OxitorchError::NotImplemented`] only for ops not yet covered.
pub fn engine_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
