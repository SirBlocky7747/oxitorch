//! Shape and stride algebra (plan.md Phase 1: "shape algebra + property
//! tests").
//!
//! All functions are pure math over `&[usize]`; `Tensor` owns the state.

use oxi_core::{OxitorchError, OxitorchResult};

/// Row-major (C-contiguous) strides in elements for `shape`.
///
/// A 0-d shape has no strides; all sizes (including 0) are legal.
#[must_use]
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for (i, &d) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(d);
    }
    strides
}

/// Total element count (`1` for 0-d shapes).
#[must_use]
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Whether `strides` is exactly the canonical row-major layout for `shape`.
///
/// This is the definition of "contiguous" used throughout: a tensor is
/// contiguous iff `strides == row_major_strides(shape)`. (An offset into a
/// shared storage does not affect contiguity.)
#[must_use]
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    shape.len() == strides.len() && strides == row_major_strides(shape)
}

/// Validates non-negative dims and that the element count fits in `usize`.
pub fn validate_shape(dims: &[i64]) -> OxitorchResult<Vec<usize>> {
    if dims.iter().any(|&d| d < 0) {
        return Err(OxitorchError::InvalidArgument(format!(
            "dimensions must be non-negative, got {dims:?}"
        )));
    }
    let dims: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    element_count_checked(&dims)?;
    Ok(dims)
}

/// Element count, erroring on `usize` overflow instead of wrapping.
pub fn element_count_checked(shape: &[usize]) -> OxitorchResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            OxitorchError::InvalidArgument(format!(
                "number of elements for shape {shape:?} overflows usize"
            ))
        })
}

/// NumPy/PyTorch broadcasting: dims align right; each pair must be equal or
/// one of them 1. Returns the broadcast shape.
///
/// # Errors
/// [`OxitorchError::ShapeMismatch`] if the shapes cannot broadcast.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> OxitorchResult<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for (i, out_dim) in out.iter_mut().enumerate() {
        let da = a
            .get(a.len().wrapping_sub(rank).wrapping_add(i))
            .copied()
            .unwrap_or(1);
        let db = b
            .get(b.len().wrapping_sub(rank).wrapping_add(i))
            .copied()
            .unwrap_or(1);
        if da != db && da != 1 && db != 1 {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{a:?}"),
                rhs: format!("{b:?}"),
            });
        }
        *out_dim = da.max(db);
    }
    Ok(out)
}

/// Maps `strides` of a tensor with `shape` onto `out_shape` for broadcasting:
/// leading missing dims get stride 0, and any dim where the tensor is 1 (but
/// the output is larger) gets stride 0.
///
/// # Errors
/// [`OxitorchError::ShapeMismatch`] if `shape` cannot broadcast to
/// `out_shape` (see [`broadcast_shapes`]).
pub fn broadcast_strides(
    shape: &[usize],
    strides: &[usize],
    out_shape: &[usize],
) -> OxitorchResult<Vec<usize>> {
    broadcast_shapes(shape, out_shape)?;
    let mut out = vec![0usize; out_shape.len()];
    let offset = out_shape.len() - shape.len();
    for (i, &_dim) in out_shape.iter().enumerate() {
        if i >= offset {
            let j = i - offset;
            if shape[j] != 1 {
                out[i] = strides[j];
            }
        }
    }
    Ok(out)
}

/// Normalized slice bounds for one dim: `(lo, hi, by)` with negative
/// bounds resolved against `len`, `hi` clamped, matching Python's `range`
/// semantics. `by` must be positive (reverse slicing lands with the
/// dtype-generic pass).
///
/// # Errors
/// [`OxitorchError::NotImplemented`] for `by <= 0`.
pub fn normalize_slice(
    len: usize,
    lo: i64,
    hi: i64,
    by: i64,
) -> OxitorchResult<(usize, usize, usize)> {
    if by <= 0 {
        return Err(OxitorchError::NotImplemented(
            "negative slice steps are not supported yet".into(),
        ));
    }
    let by = by as usize;
    let resolve = |v: i64| -> usize {
        if v < 0 {
            usize::try_from(v.unsigned_abs()).map_or(0, |u| len.saturating_sub(u))
        } else {
            (v as usize).min(len)
        }
    };
    let lo = resolve(lo);
    let hi = resolve(hi).max(lo);
    Ok((lo, hi, by))
}

/// Output dim length for a normalized `(lo, hi, by)` range.
#[must_use]
pub fn slice_len(lo: usize, hi: usize, by: usize) -> usize {
    if hi <= lo {
        0
    } else {
        (hi - lo).div_ceil(by)
    }
}

/// Normalizes a dim index against `ndim`, accepting negatives
/// (e.g. `-1` = last dim).
///
/// # Errors
/// [`OxitorchError::InvalidArgument`] if out of range.
pub fn normalize_dim(dim: i64, ndim: usize) -> OxitorchResult<usize> {
    let dim = if dim < 0 {
        let d = ndim as i64 + dim;
        if d < 0 {
            return Err(OxitorchError::InvalidArgument(format!(
                "dim {dim} out of range for {ndim}-D tensor"
            )));
        }
        d as usize
    } else if (dim as usize) < ndim {
        dim as usize
    } else {
        return Err(OxitorchError::InvalidArgument(format!(
            "dim {dim} out of range for {ndim}-D tensor"
        )));
    };
    Ok(dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides_basic() {
        assert_eq!(row_major_strides(&[2, 3, 4]), [12, 4, 1]);
        assert_eq!(row_major_strides(&[5]), [1]);
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
        assert_eq!(row_major_strides(&[0, 7]), [7, 1]);
    }

    #[test]
    fn contiguity() {
        assert!(is_contiguous(&[2, 3], &[3, 1]));
        assert!(!is_contiguous(&[2, 3], &[1, 2]));
        assert!(!is_contiguous(&[2, 3], &[3]));
    }

    #[test]
    fn broadcast_shape_cases() {
        assert_eq!(broadcast_shapes(&[2, 3], &[2, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_shapes(&[2, 1], &[1, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_shapes(&[5, 1, 4], &[3, 1]).unwrap(), [5, 3, 4]);
        assert_eq!(broadcast_shapes(&[3], &[2, 3]).unwrap(), [2, 3]);
        assert_eq!(broadcast_shapes(&[], &[2, 3]).unwrap(), [2, 3]);
        // A vector broadcasts against any leading dims.
        assert_eq!(broadcast_shapes(&[3], &[3, 3]).unwrap(), [3, 3]);
        assert!(broadcast_shapes(&[2, 3], &[4, 3]).is_err());
    }

    #[test]
    fn broadcast_strides_map_to_zero() {
        // [2,1] strides [3,1] -> [2,3]: last dim broadcasts (stride 0).
        assert_eq!(
            broadcast_strides(&[2, 1], &[3, 1], &[2, 3]).unwrap(),
            [3, 0]
        );
        // Leading missing dims get stride 0: [3] -> [2,3].
        assert_eq!(broadcast_strides(&[3], &[1], &[2, 3]).unwrap(), [0, 1]);
    }

    #[test]
    fn slice_math() {
        assert_eq!(normalize_slice(5, 1, 4, 1).unwrap(), (1, 4, 1));
        assert_eq!(normalize_slice(5, -2, 5, 1).unwrap(), (3, 5, 1));
        assert_eq!(normalize_slice(5, 0, -1, 2).unwrap(), (0, 4, 2));
        assert_eq!(normalize_slice(5, 0, 10, 1).unwrap(), (0, 5, 1));
        assert!(normalize_slice(5, 0, 5, -1).is_err());
        assert_eq!(slice_len(1, 4, 1), 3);
        assert_eq!(slice_len(0, 5, 2), 3);
        assert_eq!(slice_len(3, 3, 1), 0);
    }

    #[test]
    fn dim_normalization() {
        assert_eq!(normalize_dim(-1, 3).unwrap(), 2);
        assert_eq!(normalize_dim(0, 3).unwrap(), 0);
        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
    }

    #[test]
    fn validation() {
        assert!(validate_shape(&[2, 3]).is_ok());
        assert!(validate_shape(&[-1]).is_err());
        assert!(validate_shape(&[i64::MAX, i64::MAX]).is_err());
    }
}
