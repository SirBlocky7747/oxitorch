//! Optimizers and learning-rate schedulers.
//!
//! Placeholder crate (plan.md Phase 4): SGD/Adam/AdamW/RMSprop/Adagrad plus
//! StepLR/CosineAnnealing/OneCycle/warmup schedulers.

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

/// Stub for one SGD step; always fails until Phase 4.
///
/// # Errors
/// Always returns [`OxitorchError::NotImplemented`] in Phase 0.
pub fn sgd_step(_param: &mut Tensor, _grad: &Tensor, _lr: f64) -> OxitorchResult<()> {
    Err(OxitorchError::NotImplemented(
        "optimizers land in Phase 4 (plan.md week 13)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_stub_reports_not_implemented() {
        let mut p = Tensor::zeros(&[1]).unwrap();
        let g = Tensor::zeros(&[1]).unwrap();
        let err = sgd_step(&mut p, &g, 0.1).unwrap_err();
        assert!(matches!(err, OxitorchError::NotImplemented(_)));
    }
}
