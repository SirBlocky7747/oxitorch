//! Serialization and interop.
//!
//! Placeholder crate (plan.md Phase 6): native safetensors read/write,
//! PyTorch-compatible `state_dict()` key handling, optimizer checkpoints, and
//! (stretch) `.pt` zip+pickle import so PyTorch-trained weights load directly.

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

/// Stub for saving a state dict to safetensors; always fails until Phase 6.
///
/// # Errors
/// Always returns [`OxitorchError::NotImplemented`] in Phase 0.
pub fn save_safetensors(_state: &[(String, Tensor)], _path: &str) -> OxitorchResult<()> {
    Err(OxitorchError::NotImplemented(
        "safetensors IO lands in Phase 6 (plan.md week 15)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_stub_reports_not_implemented() {
        let t = Tensor::zeros(&[1]).unwrap();
        let err = save_safetensors(&[("w".into(), t)], "out.safetensors").unwrap_err();
        assert!(matches!(err, OxitorchError::NotImplemented(_)));
    }
}
