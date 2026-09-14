//! Stride-aware iteration over one or more buffers (plan.md Phase 1:
//! "stride-aware elementwise kernels").
//!
//! One generic iterator drives every elementwise op: each input contributes a
//! `(slice, stride)` pair, and the output is written through the caller's
//! closure. When *all* operands are contiguous the iterator collapses to a
//! flat slice walk — the fast path the dispatch table rides on.

/// One operand: the buffer to read and its stride in elements.
pub struct Operand<'a, T> {
    /// Flattened storage of the operand.
    pub data: &'a [T],
    /// Stride in elements along each output dim.
    pub strides: &'a [usize],
}

/// Visits every element of the output grid `shape`, resolving each operand
/// through its strides and feeding `(operand values...)` to `f`.
///
/// Returns the collected results in row-major order.
pub fn map_n<T: Copy, R, F>(shape: &[usize], operands: &[Operand<'_, T>], mut f: F) -> Vec<R>
where
    F: FnMut(&[T]) -> R,
{
    let count = crate::layout::element_count(shape);
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }

    // Fast path: every operand contiguous (stride 1 in the last dim, and
    // trailing stride equals the product of later dims) -> flat zip.
    let all_contiguous = shape.iter().enumerate().rev().all(|(i, &d)| {
        d == 1
            || d == 0
            || operands
                .iter()
                .all(|op| op.strides[i] == row_stride(op.strides, shape, i))
    });
    let flat_ok = all_contiguous
        && operands
            .iter()
            .all(|op| crate::layout::element_count(shape) <= op.data.len());

    if flat_ok && !shape.is_empty() {
        for pos in 0..count {
            let vals: Vec<T> = operands.iter().map(|op| op.data[pos]).collect();
            out.push(f(&vals));
        }
        return out;
    }

    // General path: walk the multi-index grid.
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..count {
        let vals: Vec<T> = operands
            .iter()
            .map(|op| {
                let off: usize = idx.iter().zip(op.strides).map(|(&i, &s)| i * s).sum();
                op.data[off]
            })
            .collect();
        out.push(f(&vals));

        // Increment the multi-index (odometer, last dim fastest).
        for d in (0..idx.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Stride that makes operand `i` contiguous within the output grid: the
/// product of the dims after `i`.
fn row_stride(strides: &[usize], shape: &[usize], i: usize) -> usize {
    let _ = strides;
    shape[i + 1..].iter().product::<usize>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_n_contiguous_fast_path() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let strides = &[2usize, 1]; // canonical row-major for [2, 2]
        let out = map_n(
            &[2, 2],
            &[Operand { data: &a, strides }, Operand { data: &b, strides }],
            |v| v[0] + v[1],
        );
        assert_eq!(out, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn map_n_broadcast_row_vector() {
        // a: [1, 3] broadcast over [2, 3]; a contiguous strides [3, 1] with
        // dim0 broadcast to 0.
        let a = [1.0f32, 2.0, 3.0];
        let a_strides = [0usize, 1];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let b_strides = [3usize, 1];
        let out = map_n(
            &[2, 3],
            &[
                Operand {
                    data: &a,
                    strides: &a_strides,
                },
                Operand {
                    data: &b,
                    strides: &b_strides,
                },
            ],
            |v| v[0] * v[1],
        );
        assert_eq!(out, vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    }

    #[test]
    fn map_n_transposed_operand() {
        // A [2,2] tensor stored transposed: strides [1, 2] reads column-major.
        let a = [1.0f32, 3.0, 2.0, 4.0]; // logical [1,2;3,4] via strides [1,2]
        let strides = [1usize, 2];
        let out = map_n(
            &[2, 2],
            &[Operand {
                data: &a,
                strides: &strides,
            }],
            |v| v[0],
        );
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn map_n_scalar_broadcast() {
        // 0-d operand broadcasts everywhere (stride entry unused).
        let s = [5.0f32];
        let a = [1.0f32, 2.0];
        let a_strides = [1usize];
        // Shape [2]: 0-d operand's strides array is shorter; use zeros.
        let zero_strides = [0usize];
        let out = map_n(
            &[2],
            &[
                Operand {
                    data: &s,
                    strides: &zero_strides,
                },
                Operand {
                    data: &a,
                    strides: &a_strides,
                },
            ],
            |v| v[0] + v[1],
        );
        assert_eq!(out, vec![6.0, 7.0]);
    }
}
