//! Unit and property tests for the Phase 1 tensor core.

use super::*;
use proptest::prelude::*;

/// Reference implementation for broadcast indexing: `idx` is the multi-index
/// into an `out_rank`-D output grid; the operand (of rank <= out_rank, right-
/// aligned) contributes 0 for broadcast dims and its row-major offset
/// otherwise. Mirrors the semantics of `layout::broadcast_strides`.
fn ref_offset(operand_shape: &[usize], idx: &[usize], out_rank: usize) -> usize {
    let mut off = 0usize;
    let strides = crate::layout::row_major_strides(operand_shape);
    let offset = out_rank - operand_shape.len();
    for (i, &ix) in idx.iter().enumerate() {
        if i >= offset {
            let j = i - offset;
            if operand_shape[j] != 1 {
                off += ix * strides[j];
            }
        }
    }
    off
}

#[test]
fn allocation_reports_layout() {
    let t = Tensor::zeros(&[2, 3]).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.strides(), &[3, 1]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.ndim(), 2);
    assert!(t.is_contiguous());
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(t.device(), Device::cpu());
}

#[test]
fn from_vec_roundtrips_values() {
    let t = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(t.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn from_vec_rejects_length_mismatch() {
    let err = Tensor::from_vec(&[2, 2], vec![1.0, 2.0]).unwrap_err();
    assert!(matches!(err, OxitorchError::InvalidArgument(_)));
}

#[test]
fn rejects_negative_dims_and_overflow() {
    assert!(Tensor::zeros(&[-1]).is_err());
    assert!(Tensor::zeros(&[i64::MAX, i64::MAX]).is_err());
}

#[test]
fn scalar_tensors_have_zero_dims() {
    let t = Tensor::full(&[], 7.0).unwrap();
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.numel(), 1);
    assert_eq!(t.to_string(), "7");
}

#[test]
fn eye_works() {
    let i3 = Tensor::eye(3).unwrap();
    assert_eq!(
        i3.as_slice().unwrap()[..7],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    );
}

// ---- broadcasting binary ops -------------------------------------------

#[test]
fn add_broadcasts_row_vector() {
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let row = Tensor::from_vec(&[3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = a.add(&row).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.as_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn add_broadcasts_column_vector_and_scalar() {
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let col = Tensor::from_vec(&[2, 1], vec![100.0, 200.0]).unwrap();
    assert_eq!(
        a.add(&col).unwrap().as_slice().unwrap(),
        &[101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );

    let s = Tensor::from_vec(&[], vec![1.0]).unwrap();
    assert_eq!(
        a.add(&s).unwrap().as_slice().unwrap(),
        &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(s.add(&a).unwrap().shape(), &[2, 3], "commutes");
}

#[test]
fn add_broadcast_incompatible_shapes_error() {
    let a = Tensor::zeros(&[2, 3]).unwrap();
    let b = Tensor::zeros(&[4, 3]).unwrap();
    let err = a.add(&b).unwrap_err();
    assert!(matches!(err, OxitorchError::ShapeMismatch { .. }));
}

#[test]
fn binary_ops_and_registry() {
    let a = Tensor::from_vec(&[2], vec![8.0, 9.0]).unwrap();
    let b = Tensor::from_vec(&[2], vec![2.0, 3.0]).unwrap();
    assert_eq!(a.sub(&b).unwrap().as_slice().unwrap(), &[6.0, 6.0]);
    assert_eq!(a.mul(&b).unwrap().as_slice().unwrap(), &[16.0, 27.0]);
    assert_eq!(a.div(&b).unwrap().as_slice().unwrap(), &[4.0, 3.0]);
    assert_eq!(a.pow(&b).unwrap().as_slice().unwrap(), &[64.0, 729.0]);
    assert_eq!(
        a.binary_op("max", &b).unwrap().as_slice().unwrap(),
        &[8.0, 9.0]
    );
    assert!(a.binary_op("frob", &b).is_err());
}

#[test]
fn comparison_ops_produce_0_or_1() {
    let a = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(&[3], vec![2.0, 2.0, 1.0]).unwrap();
    assert_eq!(a.lt(&b).unwrap().as_slice().unwrap(), &[1.0, 0.0, 0.0]);
    assert_eq!(a.le(&b).unwrap().as_slice().unwrap(), &[1.0, 1.0, 0.0]);
    assert_eq!(a.eq(&b).unwrap().as_slice().unwrap(), &[0.0, 1.0, 0.0]);
    assert_eq!(a.ne(&b).unwrap().as_slice().unwrap(), &[1.0, 0.0, 1.0]);
    assert_eq!(a.gt(&b).unwrap().as_slice().unwrap(), &[0.0, 0.0, 1.0]);
    assert_eq!(a.ge(&b).unwrap().as_slice().unwrap(), &[0.0, 1.0, 1.0]);
}

#[test]
fn unary_ops_fast_and_strided_paths_agree() {
    let base = Tensor::from_vec(&[2, 3], (0..6).map(|i| i as f32).collect()).unwrap();
    // Strided view: transpose.
    let t = base.transpose().unwrap();
    assert!(!t.is_contiguous());
    for name in ["neg", "abs", "sqrt", "exp", "relu", "sigmoid", "square"] {
        let fast = base.unary_op(name).unwrap();
        let slow = t.contiguous().transpose().unwrap().unary_op(name).unwrap();
        // same logical values in, same logical values out
        assert_eq!(
            fast.to_vec(),
            slow.to_vec(),
            "unary {name} disagrees between contiguous and strided paths"
        );
    }
}

#[test]
fn scalar_helpers() {
    let a = Tensor::from_vec(&[2], vec![1.0, 2.0]).unwrap();
    assert_eq!(
        a.add_scalar(10.0).unwrap().as_slice().unwrap(),
        &[11.0, 12.0]
    );
    assert_eq!(a.mul_scalar(3.0).unwrap().as_slice().unwrap(), &[3.0, 6.0]);
}

// ---- reductions ----------------------------------------------------------

#[test]
fn reduce_last_dim_rows() {
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
    let s = a.reduce(ops::ReduceKind::Sum, -1, false).unwrap();
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s.as_slice().unwrap(), &[6.0, 60.0]);
    let m = a.reduce(ops::ReduceKind::Max, -1, true).unwrap();
    assert_eq!(m.shape(), &[2, 1]);
    assert_eq!(m.as_slice().unwrap(), &[3.0, 30.0]);
}

#[test]
fn reduce_inner_dim_columns() {
    let a = Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    // Sum over dim 0 (columns): [1+3+5, 2+4+6] = [9, 12]
    let s = a.reduce(ops::ReduceKind::Sum, 0, false).unwrap();
    assert_eq!(s.shape(), &[2]);
    assert_eq!(s.as_slice().unwrap(), &[9.0, 12.0]);
}

#[test]
fn reduce_argmax_argmin_first_tie_and_negative() {
    let a = Tensor::from_vec(&[2, 3], vec![5.0, 7.0, 7.0, 1.0, 9.0, 2.0]).unwrap();
    let am = a.reduce(ops::ReduceKind::ArgMax, -1, false).unwrap();
    // 7 appears twice in row 0 -> first occurrence (index 1).
    assert_eq!(am.as_slice().unwrap(), &[1.0, 1.0]);
    let an = a.reduce(ops::ReduceKind::ArgMin, 1, false).unwrap();
    assert_eq!(an.as_slice().unwrap(), &[0.0, 0.0]);
}

#[test]
fn reduce_3d_middle_dim() {
    let a = Tensor::from_vec(&[2, 2, 2], (1..=8).map(|x| x as f32).collect()).unwrap();
    // Sum over dim 1: [[1+3, 2+4], [5+7, 6+8]] = [[4, 6], [12, 14]]
    let s = a.reduce(ops::ReduceKind::Sum, 1, false).unwrap();
    assert_eq!(s.shape(), &[2, 2]);
    assert_eq!(s.as_slice().unwrap(), &[4.0, 6.0, 12.0, 14.0]);
}

#[test]
fn reduce_mean_norm_and_all() {
    let a = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(a.mean_all().unwrap().as_slice().unwrap(), &[2.5]);
    assert_eq!(a.sum_all().unwrap().as_slice().unwrap(), &[10.0]);
    // Norm over dim 1: sqrt(1+4)=√5, sqrt(9+16)=5.
    let n = a.norm(-1, false).unwrap();
    assert!((n.as_slice().unwrap()[0] - 5f32.sqrt()).abs() < 1e-6);
    assert!((n.as_slice().unwrap()[1] - 5.0).abs() < 1e-6);
}

#[test]
fn reduce_errors() {
    let e = Tensor::zeros(&[0, 3]).unwrap();
    assert!(e.reduce(ops::ReduceKind::Max, 0, false).is_err());
    assert!(e.reduce(ops::ReduceKind::Sum, 0, false).is_ok()); // sum of empty = 0
    let a = Tensor::zeros(&[2, 2]).unwrap();
    assert!(a.reduce(ops::ReduceKind::Sum, 5, false).is_err());
}

// ---- views ----------------------------------------------------------------

#[test]
fn views_share_storage() {
    let base = Tensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let v = base.narrow(0, 1, 3, 1).unwrap();
    assert_eq!(v.shape(), &[2]);
    assert_eq!(v.to_vec(), vec![2.0, 3.0]);
    // Mutating through one view must be visible through the other (same Arc).
    let ptr1 = Arc::as_ptr(&base.clone().storage).cast::<u8>();
    let ptr2 = Arc::as_ptr(&v.storage).cast::<u8>();
    assert_eq!(ptr1, ptr2);
}

#[test]
fn narrow_with_step_and_negatives() {
    let base = Tensor::from_vec(&[6], (1..=6).map(|x| x as f32).collect()).unwrap();
    assert_eq!(
        base.narrow(0, 0, 6, 2).unwrap().to_vec(),
        vec![1.0, 3.0, 5.0]
    );
    assert_eq!(
        base.narrow(0, -3, 6, 1).unwrap().to_vec(),
        vec![4.0, 5.0, 6.0]
    );
    assert!(base.narrow(0, 0, 6, 0).is_err());
}

#[test]
fn select_drops_dim() {
    let base = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let r1 = base.select(0, 1).unwrap();
    assert_eq!(r1.shape(), &[3]);
    assert_eq!(r1.to_vec(), vec![4.0, 5.0, 6.0]);
    let c2 = base.select(1, -2).unwrap();
    assert_eq!(c2.to_vec(), vec![2.0, 5.0]);
    assert!(base.select(0, 2).is_err());
}

#[test]
fn reshape_views_and_copies() {
    let base = Tensor::from_vec(&[2, 3], (1..=6).map(|x| x as f32).collect()).unwrap();
    assert_eq!(base.reshape(&[3, 2]).unwrap().to_vec(), base.to_vec());
    assert_eq!(base.reshape(&[-1]).unwrap().shape(), &[6]);
    assert_eq!(base.reshape(&[6, -1]).unwrap().shape(), &[6, 1]);
    // torch semantics: reshaping N > 1 elements to [] is an error.
    assert!(base.reshape(&[]).is_err());
    assert_eq!(
        Tensor::full(&[], 7.0)
            .unwrap()
            .reshape(&[])
            .unwrap()
            .numel(),
        1
    );
    assert!(base.reshape(&[4, 2]).is_err());
    assert!(base.reshape(&[-1, -1]).is_err());

    // Non-contiguous reshape falls back to a copy with correct values.
    let t = base.transpose().unwrap(); // [3, 2] non-contiguous
    let r = t.reshape(&[6]).unwrap();
    assert_eq!(r.to_vec(), t.to_vec());
}

#[test]
#[allow(clippy::float_cmp)] // integer-valued floats, bitwise equality is exact
fn permute_and_transpose_are_views() {
    let base = Tensor::from_vec(&[2, 3, 4], (0..24).map(|x| x as f32).collect()).unwrap();
    let p = base.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);
    assert!(!p.is_contiguous());
    // Logical read-back must match manual indexing: p[i, j, k] = base[j, k, i].
    let pv = p.to_vec();
    for i in 0..4 {
        for j in 0..2 {
            for k in 0..3 {
                assert_eq!(pv[i * 6 + j * 3 + k], (j * 12 + k * 4 + i) as f32);
            }
        }
    }
    assert!(base.permute(&[0, 0, 1]).is_err());
    assert!(base.permute(&[0, 1]).is_err());

    let m = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(m.transpose().unwrap().to_vec(), vec![1.0, 3.0, 2.0, 4.0]);
    assert!(Tensor::zeros(&[2, 2, 2]).unwrap().transpose().is_err());
}

#[test]
fn broadcast_to_zero_strides() {
    let row = Tensor::from_vec(&[1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let big = row.broadcast_to(&[2, 3]).unwrap();
    assert_eq!(big.shape(), &[2, 3]);
    assert_eq!(big.to_vec(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    assert_eq!(row.broadcast_to(&[4, 3]).unwrap().numel(), 12);
    assert!(row.broadcast_to(&[2, 4]).is_err());
}

#[test]
fn contiguous_materializes() {
    let base = Tensor::from_vec(&[2, 3], (1..=6).map(|x| x as f32).collect()).unwrap();
    let t = base.transpose().unwrap();
    let c = t.contiguous();
    assert!(c.is_contiguous());
    assert_eq!(c.to_vec(), t.to_vec());
    // contiguous() on an already-contiguous tensor is a cheap clone.
    let base2 = base.contiguous();
    assert!(Arc::ptr_eq(&base.storage, &base2.storage));
}

#[test]
fn as_slice_requires_contiguity() {
    let base = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(base.as_slice().is_ok());
    assert!(base.transpose().unwrap().as_slice().is_err());
}

// ---- fancy indexing -------------------------------------------------------

#[test]
fn index_select_rows_and_negative() {
    let base = Tensor::from_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let picked = base.index_select(0, &[2, 0]).unwrap();
    assert_eq!(picked.shape(), &[2, 2]);
    assert_eq!(picked.to_vec(), vec![5.0, 6.0, 1.0, 2.0]);
    let neg = base.index_select(0, &[-1]).unwrap();
    assert_eq!(neg.to_vec(), vec![5.0, 6.0]);
    assert!(base.index_select(0, &[3]).is_err());
}

#[test]
fn gather_flat_and_dim0() {
    let base = Tensor::from_vec(&[2, 3], (1..=6).map(|x| x as f32).collect()).unwrap();
    assert_eq!(base.gather_flat(&[0, 5, 1]).unwrap().shape(), &[3]);
    assert_eq!(
        base.gather_flat(&[0, 5, 1]).unwrap().to_vec(),
        vec![1.0, 6.0, 2.0]
    );
    assert!(base.gather_flat(&[6]).is_err());
    assert!(base.gather_flat(&[-7]).is_err());

    let rows = base.gather_dim0(&[1, 1, 0]).unwrap();
    assert_eq!(rows.shape(), &[3, 3]);
    assert_eq!(
        rows.to_vec(),
        vec![4.0, 5.0, 6.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
    );
    assert!(base.gather_dim0(&[2]).is_err());
}

#[test]
fn scatter_dim0_roundtrip_with_gather() {
    let base = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let vals = Tensor::from_vec(&[1, 2], vec![9.0, 8.0]).unwrap();
    let out = base.scatter_dim0(&[1], &vals).unwrap();
    assert_eq!(out.to_vec(), vec![1.0, 2.0, 9.0, 8.0]);
    // Roundtrip: gather what we scattered.
    let back = out.gather_dim0(&[1]).unwrap();
    assert_eq!(back.to_vec(), vec![9.0, 8.0]);
    assert!(base.scatter_dim0(&[5], &vals).is_err());
    assert!(base
        .scatter_dim0(&[0], &Tensor::zeros(&[2, 2]).unwrap())
        .is_err());
}

#[test]
fn masked_select_basic_and_broadcast() {
    let base = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask = Tensor::from_vec(&[2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).unwrap();
    assert_eq!(
        base.masked_select(&mask).unwrap().to_vec(),
        vec![1.0, 3.0, 5.0]
    );

    // Broadcast a row mask.
    let row_mask = Tensor::from_vec(&[3], vec![1.0, 0.0, 0.0]).unwrap();
    assert_eq!(
        base.masked_select(&row_mask).unwrap().to_vec(),
        vec![1.0, 4.0]
    );
}

// ---- matmul -----------------------------------------------------------------

#[test]
fn matmul_identity_and_values() {
    let a = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let i = Tensor::eye(2).unwrap();
    assert_eq!(
        a.matmul(&i).unwrap().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0]
    );

    let b = Tensor::from_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(
        a.matmul(&b).unwrap().as_slice().unwrap(),
        &[19.0, 22.0, 43.0, 50.0]
    );
}

#[test]
fn matmul_rectangular_and_transposed_inputs() {
    let a = Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    assert_eq!(
        a.matmul(&b).unwrap().as_slice().unwrap(),
        &[58.0, 64.0, 139.0, 154.0]
    );

    // Feeding transposed views must work (materialized internally):
    // a^T [3,2] @ b^T [2,3] -> [3,3].
    let at = a.transpose().unwrap();
    let bt = b.transpose().unwrap();
    let out = at.matmul(&bt).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    assert_eq!(
        out.as_slice().unwrap(),
        &[39.0, 49.0, 59.0, 54.0, 68.0, 82.0, 69.0, 87.0, 105.0]
    );
    // Inner-dim mismatch still errors through views.
    assert!(at.matmul(&b).is_err());
}

#[test]
fn matmul_parallel_path_matches_sequential() {
    // 64x64x64 = 262144 >= threshold -> exercises the rayon row-block path.
    let a = Tensor::from_vec(
        &[64, 64],
        (0..64 * 64).map(|i| ((i * 7) % 13) as f32 - 6.0).collect(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        &[64, 64],
        (0..64 * 64).map(|i| ((i * 5) % 11) as f32 - 5.0).collect(),
    )
    .unwrap();
    let out = a.matmul(&b).unwrap();
    // Spot-check several entries against a naive dot product.
    let af = a.as_slice().unwrap();
    let bf = b.as_slice().unwrap();
    for &(i, j) in &[(0usize, 0usize), (17, 42), (63, 63), (31, 5)] {
        let expect: f32 = (0..64).map(|p| af[i * 64 + p] * bf[p * 64 + j]).sum();
        assert!((out.as_slice().unwrap()[i * 64 + j] - expect).abs() <= expect.abs() * 1e-5);
    }
}

// ---- bmm ---------------------------------------------------------------------

#[test]
fn bmm_values_and_shapes() {
    // Two batched 2x2 products, hand-checked.
    let a = Tensor::from_vec(&[2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let b = Tensor::from_vec(
        &[2, 2, 2],
        vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    )
    .unwrap();
    let out = a.bmm(&b).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    // batch 0: [[1,2],[3,4]] @ [[9,10],[11,12]] = [[31,34],[71,78]]
    // batch 1: [[5,6],[7,8]] @ [[13,14],[15,16]] = [[155,166],[211,226]]
    assert_eq!(
        out.as_slice().unwrap(),
        &[31.0, 34.0, 71.0, 78.0, 155.0, 166.0, 211.0, 226.0]
    );
}

#[test]
fn bmm_matches_per_batch_matmul() {
    // Property: bmm(a, b)[i] == a[i].matmul(b[i]) for deterministic data.
    let batch = 3usize;
    let (num_rows, inner, num_cols) = (4usize, 5usize, 2usize);
    let mut state = 43u64; // odd seed (42 | 1) for the xorshift below
    let mut gen = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    };
    let a = Tensor::from_vec(
        &[batch as i64, num_rows as i64, inner as i64],
        gen(batch * num_rows * inner),
    )
    .unwrap();
    let b = Tensor::from_vec(
        &[batch as i64, inner as i64, num_cols as i64],
        gen(batch * inner * num_cols),
    )
    .unwrap();
    let out = a.bmm(&b).unwrap();
    let of = out.as_slice().unwrap();
    let af = a.as_slice().unwrap();
    let bf = b.as_slice().unwrap();
    for bi in 0..batch {
        for row in 0..num_rows {
            for col in 0..num_cols {
                let expect: f32 = (0..inner)
                    .map(|p| {
                        af[bi * num_rows * inner + row * inner + p]
                            * bf[bi * inner * num_cols + p * num_cols + col]
                    })
                    .sum();
                let got = of[bi * num_rows * num_cols + row * num_cols + col];
                assert!(
                    (got - expect).abs() <= expect.abs().max(1.0) * 1e-5,
                    "batch {bi} [{row},{col}]: {got} vs {expect}"
                );
            }
        }
    }
}

#[test]
fn bmm_accepts_transposed_views_and_errors() {
    let a = Tensor::from_vec(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(&[1, 2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    // Non-contiguous operand must be materialized internally.
    let out = a.bmm(&b.permute(&[0, 2, 1]).unwrap()).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[17.0, 23.0, 39.0, 53.0]);

    // Batch mismatch and inner-dim mismatch are errors.
    let two = Tensor::zeros(&[2, 2, 2]).unwrap();
    assert!(a.bmm(&two).is_err());
    assert!(two.bmm(&b).is_err());
    // Rank guard.
    let m2 = Tensor::zeros(&[2, 2]).unwrap();
    assert!(a.bmm(&m2).is_err());
}

#[test]
fn bmm_parallel_path_matches_sequential() {
    // 8 planes of 64x64x64 = 2M elements >= threshold -> rayon batch path.
    let batch = 8usize;
    let dim = 64usize;
    let a = Tensor::from_vec(
        &[batch as i64, dim as i64, dim as i64],
        (0..batch * dim * dim)
            .map(|i| ((i * 7) % 13) as f32 - 6.0)
            .collect(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        &[batch as i64, dim as i64, dim as i64],
        (0..batch * dim * dim)
            .map(|i| ((i * 5) % 11) as f32 - 5.0)
            .collect(),
    )
    .unwrap();
    let out = a.bmm(&b).unwrap();
    let af = a.as_slice().unwrap();
    let bf = b.as_slice().unwrap();
    let of = out.as_slice().unwrap();
    // Spot-check one entry per batch plane against a naive dot product.
    for bi in 0..batch {
        let (row, col) = (17usize, 42usize);
        let expect: f32 = (0..dim)
            .map(|p| af[bi * dim * dim + row * dim + p] * bf[bi * dim * dim + p * dim + col])
            .sum();
        assert!((of[bi * dim * dim + row * dim + col] - expect).abs() <= expect.abs() * 1e-5);
    }
}

#[test]
fn matmul_errors() {
    let a = Tensor::zeros(&[2, 3]).unwrap();
    let b = Tensor::zeros(&[2, 2]).unwrap();
    assert!(a
        .matmul(&b)
        .unwrap_err()
        .to_string()
        .contains("shape mismatch"));
    let c = Tensor::zeros(&[2, 2, 2]).unwrap();
    assert!(c.matmul(&b).is_err());
}

// ---- property tests --------------------------------------------------------

proptest! {
    #[test]
    fn broadcast_add_matches_elementwise_reference(
        _a_shape in prop::collection::vec(1usize..4, 1..4),
        b_shape in prop::collection::vec(1usize..4, 1..4),
        seed in any::<u64>(),
    ) {
        // Construct a guaranteed-broadcastable pair: b keeps its random
        // shape; a takes b's trailing `a_rank` dims, replacing every other
        // one with 1 (some/all of a's dims broadcast).
        let a_rank = 1 + (seed as usize) % b_shape.len();
        let a_shape: Vec<usize> = b_shape[b_shape.len() - a_rank..]
            .iter()
            .enumerate()
            .map(|(i, &d)| if (seed as usize + i * 7) % 2 == 0 { 1 } else { d })
            .collect();
        let n_a = crate::layout::element_count(&a_shape);
        let n_b = crate::layout::element_count(&b_shape);
        let a_data: Vec<f32> = (0..n_a).map(|i| ((seed as usize + i * 31) % 19) as f32 - 9.0).collect();
        let b_data: Vec<f32> = (0..n_b).map(|i| ((seed as usize + i * 17) % 7) as f32 - 3.0).collect();
        let a = Tensor::from_vec(
            &a_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            a_data,
        )
        .unwrap();
        let b = Tensor::from_vec(
            &b_shape.iter().map(|&d| d as i64).collect::<Vec<_>>(),
            b_data,
        )
        .unwrap();

        let out = a.add(&b).unwrap();
        let out_shape = crate::layout::broadcast_shapes(&a_shape, &b_shape).unwrap();
        prop_assert_eq!(out.shape(), &out_shape[..]);
        // Symmetry: the reversed add must agree element-for-element.
        let out_rev = b.add(&a).unwrap();
        prop_assert_eq!(out.to_vec(), out_rev.to_vec());

        // Reference: walk the broadcast grid manually.
        let count = crate::layout::element_count(&out_shape);
        for pos in 0..count {
            let mut idx = vec![0usize; out_shape.len()];
            let mut rem = pos;
            for (i, &d) in out_shape.iter().enumerate().rev() {
                idx[i] = rem % d;
                rem /= d;
            }
            let av = a.to_vec()[ref_offset(&a_shape, &idx, out_shape.len())];
            let bv = b.to_vec()[ref_offset(&b_shape, &idx, out_shape.len())];
            prop_assert!((out.to_vec()[pos] - (av + bv)).abs() < 1e-5);
        }
    }

    #[test]
    fn reshape_permute_roundtrip_preserves_values(
        rows in 1usize..6,
        cols in 1usize..6,
    ) {
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let t = Tensor::from_vec(&[rows as i64, cols as i64], data.clone()).unwrap();
        // reshape to flat and back must preserve everything.
        let flat = t.reshape(&[-1]).unwrap();
        prop_assert_eq!(flat.shape(), &[rows * cols]);
        let back = flat.reshape(&[cols as i64, rows as i64]).unwrap();
        prop_assert_eq!(back.to_vec(), data.clone());
        // permute then contiguous gives the transposed data.
        let tr = t.permute(&[1, 0]).unwrap().contiguous();
        let data_ref = &data;
        let expect: Vec<f32> = (0..cols)
            .flat_map(move |c| (0..rows).map(move |r| data_ref[r * cols + c]))
            .collect();
        prop_assert_eq!(tr.to_vec(), expect);
    }

    #[test]
    fn matmul_matches_naive_reference(
        m in 1usize..17,
        k in 1usize..17,
        n in 1usize..17,
        seed in any::<u64>(),
    ) {
        let mut state = seed;
        let mut next = move |_: usize| -> f32 {
            // PCG-style LCG; constants from the PCG paper reference table.
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as f32 / 16_777_216.0 - 64.0
        };
        let a: Vec<f32> = (0..m * k).map(&mut next).collect();
        let b: Vec<f32> = (0..k * n).map(&mut next).collect();

        let ta = Tensor::from_vec(&[m as i64, k as i64], a).unwrap();
        let tb = Tensor::from_vec(&[k as i64, n as i64], b).unwrap();
        let got = ta.matmul(&tb).unwrap();

        let a = ta.as_slice().unwrap();
        let b = tb.as_slice().unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                let expect = got.as_slice().unwrap()[i * n + j];
                prop_assert!(
                    (acc - expect).abs() <= acc.abs().max(expect.abs()) * 1e-5,
                    "mismatch at ({i}, {j}): naive {acc} vs kernel {expect}"
                );
            }
        }
    }
}
