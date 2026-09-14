//! Neural-network modules, parameters, and losses.
//!
//! Placeholder crate (plan.md Phase 3). `Module` composition will live on the
//! Python side in v1 (faster to ship); this crate holds the Rust-side layers,
//! init schemes, and loss kernels.

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

/// Stub for a `Linear` layer forward pass; always fails until Phase 3.
///
/// # Errors
/// Always returns [`OxitorchError::NotImplemented`] in Phase 0.
pub fn linear(_input: &Tensor, _weight: &Tensor, _bias: Option<&Tensor>) -> OxitorchResult<Tensor> {
    Err(OxitorchError::NotImplemented(
        "nn layers land in Phase 3 (plan.md weeks 9–12)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_stub_reports_not_implemented() {
        let x = Tensor::zeros(&[1, 1]).unwrap();
        let w = Tensor::zeros(&[1, 1]).unwrap();
        let err = linear(&x, &w, None).unwrap_err();
        assert!(matches!(err, OxitorchError::NotImplemented(_)));
    }
}
