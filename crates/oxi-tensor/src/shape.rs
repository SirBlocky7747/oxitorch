//! Validated shape handling: dimension checks plus row-major stride math.

use oxi_core::{OxitorchError, OxitorchResult};

/// A validated tensor shape with row-major strides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Validates a shape given as `i64` dims (mirrors the Python-facing API,
    /// where negative dims are user errors, not Python-style negatives).
    pub fn new(dims: &[i64]) -> OxitorchResult<Self> {
        if dims.iter().any(|&d| d < 0) {
            return Err(OxitorchError::InvalidArgument(format!(
                "dimensions must be non-negative, got {dims:?}"
            )));
        }
        let dims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        // Guard against element counts overflowing usize on 64-bit targets.
        dims.iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                OxitorchError::InvalidArgument(format!(
                    "number of elements for shape {dims:?} overflows usize"
                ))
            })?;
        Ok(Self { dims })
    }

    /// Row-major strides in elements (C-contiguous), `1` for the last dim.
    ///
    /// For 0-d shapes this returns an empty slice.
    pub fn row_major_strides(&self) -> Vec<usize> {
        let mut strides = vec![0usize; self.dims.len()];
        let mut acc = 1usize;
        for (i, &d) in self.dims.iter().enumerate().rev() {
            strides[i] = acc;
            acc = acc.saturating_mul(d);
        }
        strides
    }

    /// Total element count (0-d shapes hold exactly one element).
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Borrowed view of the dims.
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Dims as `i64`, for handing back across the Python boundary.
    pub fn as_i64(&self) -> Vec<i64> {
        self.dims.iter().map(|&d| d as i64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_for_common_ranks() {
        assert_eq!(
            Shape::new(&[2, 3, 4]).unwrap().row_major_strides(),
            [12, 4, 1]
        );
        assert_eq!(Shape::new(&[5]).unwrap().row_major_strides(), [1]);
        assert_eq!(
            Shape::new(&[]).unwrap().row_major_strides(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn element_count() {
        assert_eq!(Shape::new(&[2, 3, 4]).unwrap().element_count(), 24);
        assert_eq!(Shape::new(&[]).unwrap().element_count(), 1);
        assert_eq!(Shape::new(&[0, 7]).unwrap().element_count(), 0);
    }

    #[test]
    fn rejects_bad_dims() {
        assert!(Shape::new(&[-2, 3]).is_err());
        assert!(Shape::new(&[i64::MAX; 3]).is_err());
    }
}
