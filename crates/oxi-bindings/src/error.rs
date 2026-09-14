//! Mapping of Rust errors onto Python exceptions.
//!
//! Convention (decided in Phase 0, plan.md):
//!
//! | Rust variant                        | Python exception       |
//! |-------------------------------------|------------------------|
//! | `InvalidArgument` / `ShapeMismatch` | `ValueError`           |
//! | `NotImplemented`                    | `NotImplementedError`  |
//! | everything else                     | `RuntimeError`         |

use oxi_core::OxitorchError;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Converts an [`OxitorchError`] into the Python exception per the table
/// above, preserving the `Display` message.
///
/// Takes ownership by design so call sites read `result.map_err(to_pyerr)?`;
/// a blanket `From` impl is not possible here (orphan rule: both types are
/// foreign to this crate).
#[allow(clippy::needless_pass_by_value)] // ownership is the call-site ergonomics
pub fn to_pyerr(err: OxitorchError) -> PyErr {
    match &err {
        OxitorchError::InvalidArgument(_) | OxitorchError::ShapeMismatch { .. } => {
            PyValueError::new_err(err.to_string())
        }
        OxitorchError::NotImplemented(_) => PyNotImplementedError::new_err(err.to_string()),
        OxitorchError::OutOfBounds(_) | OxitorchError::Other(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_covers_every_variant() {
        // PyErr formatting touches interpreter state; pyo3 requires an
        // initialized interpreter (idempotent, cheap after first call).
        pyo3::Python::initialize();
        // Compile-time exhaustiveness: adding a variant to OxitorchError
        // without updating the match above must break this test build.
        let cases = [
            OxitorchError::InvalidArgument("x".into()),
            OxitorchError::ShapeMismatch {
                lhs: "[1]".into(),
                rhs: "[2]".into(),
            },
            OxitorchError::NotImplemented("x".into()),
            OxitorchError::OutOfBounds("x".into()),
            OxitorchError::Other("x".into()),
        ];
        for case in cases {
            // PyErr construction is GIL-free since pyo3 0.22 (errors are
            // stored lazily and materialized when raised in Python); Debug
            // formatting must work without a live GIL handle.
            let pyerr = to_pyerr(case);
            assert!(!format!("{pyerr:?}").is_empty());
        }
    }
}
