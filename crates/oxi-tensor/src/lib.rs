//! The oxitorch tensor crate.
//!
//! Phase 1 (plan.md): dtype-aware `Tensor` (f32 compute dtype), broadcasting
//! rules identical to NumPy/PyTorch, stride-aware elementwise kernels, thread
//! parallel reductions, `matmul`, indexing (basic/slice/fancy/boolean/
//! gather/scatter), and the `DeviceBackend` seam for future GPU backends.

use std::sync::Arc;

use oxi_core::{DType, Device, OxitorchError, OxitorchResult};
use rayon::prelude::*;

mod iterate;
pub mod layout;
pub mod ops;

pub use iterate::Operand;

/// The f32 element type for Phase 1 compute (bf16/f16 storage dtypes arrive
/// with the dtype-generic promotion pass).
pub type Elem = f32;

/// View metadata: shape plus strides (elements). Cheap to clone.
#[derive(Debug, Clone, PartialEq, Eq)]
struct View {
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl View {
    fn new(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        Self { shape, strides }
    }

    fn row_major(shape: Vec<usize>) -> Self {
        let strides = layout::row_major_strides(&shape);
        Self { shape, strides }
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn is_contiguous(&self) -> bool {
        layout::is_contiguous(&self.shape, &self.strides)
    }
}

/// A dense, reference-counted, strided tensor.
///
/// Storage is shared (`Arc<Vec<Elem>>` + offset); clones are views.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Vec<Elem>>,
    /// Offset of element (0, 0, ...) into the storage, in elements.
    offset: usize,
    view: View,
}

impl Tensor {
    // ---- constructors ---------------------------------------------------

    /// Allocates a zero-filled tensor of the given shape.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for negative dims or overflow.
    pub fn zeros(shape: &[i64]) -> OxitorchResult<Self> {
        let dims = layout::validate_shape(shape)?;
        Ok(Self {
            storage: Arc::new(vec![0.0; layout::element_count(&dims)]),
            offset: 0,
            view: View::row_major(dims),
        })
    }

    /// Allocates a tensor filled with `value`.
    ///
    /// # Errors
    /// Same as [`Tensor::zeros`].
    pub fn full(shape: &[i64], value: Elem) -> OxitorchResult<Self> {
        let dims = layout::validate_shape(shape)?;
        Ok(Self {
            storage: Arc::new(vec![value; layout::element_count(&dims)]),
            offset: 0,
            view: View::row_major(dims),
        })
    }

    /// Builds a tensor from a flat row-major buffer (no copy beyond moving
    /// into shared storage).
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if the buffer length does not match
    /// the shape.
    pub fn from_vec(shape: &[i64], data: Vec<Elem>) -> OxitorchResult<Self> {
        let dims = layout::validate_shape(shape)?;
        let count = layout::element_count(&dims);
        if count != data.len() {
            return Err(OxitorchError::InvalidArgument(format!(
                "shape {shape:?} implies {count} elements but the buffer holds {}",
                data.len()
            )));
        }
        Ok(Self {
            storage: Arc::new(data),
            offset: 0,
            view: View::row_major(dims),
        })
    }

    /// Wraps an existing `Arc` buffer with the given shape without copying.
    ///
    /// # Errors
    /// Same length check as [`Tensor::from_vec`].
    pub fn from_shared(shape: &[i64], data: Arc<Vec<Elem>>) -> OxitorchResult<Self> {
        let dims = layout::validate_shape(shape)?;
        let count = layout::element_count(&dims);
        if count != data.len() {
            return Err(OxitorchError::InvalidArgument(format!(
                "shape {shape:?} implies {count} elements but the buffer holds {}",
                data.len()
            )));
        }
        Ok(Self {
            storage: data,
            offset: 0,
            view: View::row_major(dims),
        })
    }

    /// Allocates a tensor of ones with the same shape as `other`.
    ///
    /// # Errors
    /// Propagates [`Tensor::full`].
    pub fn ones_like(other: &Self) -> OxitorchResult<Self> {
        Self::full(
            &other.shape().iter().map(|&d| d as i64).collect::<Vec<_>>(),
            1.0,
        )
    }

    /// The shape as `i64` dims (autograd/Python convenience).
    #[must_use]
    pub fn shape_i64(&self) -> Vec<i64> {
        self.view.shape.iter().map(|&d| d as i64).collect()
    }

    /// The identity tensor (useful in tests and examples).
    ///
    /// # Errors
    /// Propagates [`Tensor::zeros`]/`from_vec` errors.
    pub fn eye(n: usize) -> OxitorchResult<Self> {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_vec(&[n as i64, n as i64], data)
    }

    // ---- layout accessors ------------------------------------------------

    /// The tensor's shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.view.shape
    }

    /// The tensor's strides, in elements.
    #[must_use]
    pub fn strides(&self) -> &[usize] {
        &self.view.strides
    }

    /// Number of dimensions (0 for scalars).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.view.ndim()
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        layout::element_count(&self.view.shape)
    }

    /// The dtype of this tensor (f32 across Phase 1).
    #[must_use]
    pub const fn dtype(&self) -> DType {
        DType::F32
    }

    /// The device this tensor lives on (CPU across Phase 1).
    #[must_use]
    pub const fn device(&self) -> Device {
        Device::cpu()
    }

    /// Whether the tensor is stored contiguously in row-major order.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.view.is_contiguous()
    }

    /// Flat contiguous view of the elements, row-major.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if the tensor is not contiguous
    /// (call [`Tensor::contiguous`] first).
    pub fn as_slice(&self) -> OxitorchResult<&[Elem]> {
        if !self.is_contiguous() {
            return Err(OxitorchError::InvalidArgument(
                "as_slice requires a contiguous tensor; call .contiguous() first".into(),
            ));
        }
        Ok(&self.storage[self.offset..self.offset + self.numel()])
    }

    /// Materializes elements in row-major order, regardless of layout.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Elem> {
        if self.is_contiguous() {
            self.storage[self.offset..self.offset + self.numel()].to_vec()
        } else {
            self.read_grid()
        }
    }

    /// Returns a contiguous copy if needed; otherwise a cheap clone.
    #[must_use]
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            self.clone()
        } else {
            Self {
                storage: Arc::new(self.to_vec()),
                offset: 0,
                view: View::row_major(self.view.shape.clone()),
            }
        }
    }

    /// Reads elements in row-major logical order through the strides.
    fn read_grid(&self) -> Vec<Elem> {
        let ops = [Operand {
            data: &self.storage[self.offset..],
            strides: &self.view.strides,
        }];
        iterate::map_n(&self.view.shape, &ops, |v| v[0])
    }

    // ---- elementwise ops (dispatch table) --------------------------------

    /// Builds an output tensor from per-position results.
    fn from_results(shape: &[usize], results: Vec<Elem>) -> Self {
        debug_assert_eq!(results.len(), layout::element_count(shape));
        Self {
            storage: Arc::new(results),
            offset: 0,
            view: View::row_major(shape.to_vec()),
        }
    }

    fn binary(&self, rhs: &Self, f: ops::BinaryFn<Elem>) -> OxitorchResult<Self> {
        let out_shape = layout::broadcast_shapes(self.shape(), rhs.shape())?;
        let a_strides = layout::broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let b_strides = layout::broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let results = iterate::map_n(
            &out_shape,
            &[
                Operand {
                    data: &self.storage[self.offset..],
                    strides: &a_strides,
                },
                Operand {
                    data: &rhs.storage[rhs.offset..],
                    strides: &b_strides,
                },
            ],
            f,
        );
        Ok(Self::from_results(&out_shape, results))
    }

    fn cmp(&self, rhs: &Self, f: ops::CmpFn<Elem>) -> OxitorchResult<Self> {
        let out_shape = layout::broadcast_shapes(self.shape(), rhs.shape())?;
        let a_strides = layout::broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let b_strides = layout::broadcast_strides(rhs.shape(), rhs.strides(), &out_shape)?;
        let results: Vec<bool> = iterate::map_n(
            &out_shape,
            &[
                Operand {
                    data: &self.storage[self.offset..],
                    strides: &a_strides,
                },
                Operand {
                    data: &rhs.storage[rhs.offset..],
                    strides: &b_strides,
                },
            ],
            f,
        );
        let data: Vec<Elem> = results.into_iter().map(Elem::from).collect();
        Ok(Self::from_results(&out_shape, data))
    }

    fn unary(&self, f: ops::UnaryFn<Elem>) -> Self {
        if self.is_contiguous() {
            // Fast path: flat map over the slice.
            let data = self.storage[self.offset..self.offset + self.numel()]
                .iter()
                .map(|&a| f(a))
                .collect();
            Self::from_results(self.shape(), data)
        } else {
            let data = iterate::map_n(
                &self.view.shape,
                &[Operand {
                    data: &self.storage[self.offset..],
                    strides: &self.view.strides,
                }],
                |v| f(v[0]),
            );
            Self::from_results(&self.view.shape, data)
        }
    }

    /// Dispatches a named binary op through the registry.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for unknown op names; shape errors
    /// from broadcasting.
    pub fn binary_op(&self, name: &str, rhs: &Self) -> OxitorchResult<Self> {
        let spec = ops::find_binary_op(name)
            .ok_or_else(|| OxitorchError::InvalidArgument(format!("unknown binary op {name:?}")))?;
        self.binary(rhs, spec.f)
    }

    /// Dispatches a named comparison op (bool output).
    ///
    /// # Errors
    /// Same as [`Tensor::binary_op`].
    pub fn cmp_op(&self, name: &str, rhs: &Self) -> OxitorchResult<Self> {
        let spec = ops::find_cmp_op(name).ok_or_else(|| {
            OxitorchError::InvalidArgument(format!("unknown comparison op {name:?}"))
        })?;
        self.cmp(rhs, spec.f)
    }

    /// Dispatches a named unary op through the registry.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for unknown op names.
    pub fn unary_op(&self, name: &str) -> OxitorchResult<Self> {
        let spec = ops::find_unary_op(name)
            .ok_or_else(|| OxitorchError::InvalidArgument(format!("unknown unary op {name:?}")))?;
        Ok(self.unary(spec.f))
    }

    /// Elementwise `(a + b)` with broadcasting.
    ///
    /// # Errors
    /// Shape mismatch when the shapes cannot broadcast.
    pub fn add(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.binary_op("add", rhs)
    }

    /// Elementwise `(a - b)` with broadcasting.
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn sub(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.binary_op("sub", rhs)
    }

    /// Elementwise `(a * b)` with broadcasting.
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn mul(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.binary_op("mul", rhs)
    }

    /// Elementwise `(a / b)` with broadcasting.
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn div(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.binary_op("div", rhs)
    }

    /// Elementwise `a^b` with broadcasting.
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn pow(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.binary_op("pow", rhs)
    }

    /// Elementwise equality (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn eq(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("eq", rhs)
    }

    /// Elementwise `<` (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn lt(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("lt", rhs)
    }

    /// Elementwise `<=` (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn le(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("le", rhs)
    }

    /// Elementwise `>` (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn gt(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("gt", rhs)
    }

    /// Elementwise `>=` (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn ge(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("ge", rhs)
    }

    /// Elementwise `!=` (bool output, broadcast).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn ne(&self, rhs: &Self) -> OxitorchResult<Self> {
        self.cmp_op("ne", rhs)
    }

    /// Scalar add: `self + scalar` (broadcast of a 0-d right operand).
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn add_scalar(&self, scalar: Elem) -> OxitorchResult<Self> {
        let s = Tensor::from_vec(&[], vec![scalar])?;
        self.add(&s)
    }

    /// Scalar multiplication: `self * scalar`.
    ///
    /// # Errors
    /// Same as [`Tensor::add`].
    pub fn mul_scalar(&self, scalar: Elem) -> OxitorchResult<Self> {
        let s = Tensor::from_vec(&[], vec![scalar])?;
        self.mul(&s)
    }

    /// Applies a named unary op.
    ///
    /// # Errors
    /// Unknown op name.
    pub fn apply_unary(&self, name: &str) -> OxitorchResult<Self> {
        self.unary_op(name)
    }

    // ---- reductions ------------------------------------------------------

    /// Reduces `data` (a strip of `strip_len` elements repeated `strips`
    /// times, `stride` apart) into per-strip results, parallelized for large
    /// workloads.
    fn reduce_strips(
        data: &[Elem],
        strips: usize,
        strip_len: usize,
        stride: usize,
        kind: ops::ReduceKind,
    ) -> Vec<f64> {
        if strips == 0 {
            return Vec::new();
        }
        if strips == 1 {
            return vec![kind
                .reduce_strip(&data[..strip_len])
                .expect("strip_len > 0 checked by reduce_strip")];
        }
        if strip_len * strips >= ops::PARALLEL_THRESHOLD_ELEMENTS {
            (0..strips)
                .into_par_iter()
                .map(|s| {
                    kind.reduce_strip(&data[s * stride..s * stride + strip_len])
                        .expect("same")
                })
                .collect()
        } else {
            (0..strips)
                .map(|s| {
                    kind.reduce_strip(&data[s * stride..s * stride + strip_len])
                        .expect("same")
                })
                .collect()
        }
    }

    /// Reduces along `dim` (negative dims count from the end).
    ///
    /// With `keepdim=false` the reduced dim is removed; with `keepdim=true`
    /// it becomes 1. The result dtype is i64 for argmax/argmin, f32 otherwise.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for out-of-range dims or reducing
    /// min/max/arg* over a zero-length dim; shape errors otherwise.
    pub fn reduce(&self, kind: ops::ReduceKind, dim: i64, keepdim: bool) -> OxitorchResult<Self> {
        let dim = layout::normalize_dim(dim, self.ndim())?;
        let strip_len = self.view.shape[dim];
        if strip_len == 0 && kind.is_index_output() {
            return Err(OxitorchError::InvalidArgument(
                "cannot compute argmax/argmin over a zero-length dim".into(),
            ));
        }
        if strip_len == 0 && matches!(kind, ops::ReduceKind::Max | ops::ReduceKind::Min) {
            return Err(OxitorchError::InvalidArgument(
                "cannot compute max/min over a zero-length dim".into(),
            ));
        }

        // Reduce the last dim: strips are contiguous slices of the flat
        // (possibly non-contiguous) grid read in row-major order.
        let inner: usize = self.view.shape[dim + 1..].iter().product();
        let outer: usize = self.view.shape[..dim].iter().product();
        let flat = self.to_vec();
        let mut results: Vec<f64> = Vec::with_capacity(outer * inner);
        if dim == self.ndim() - 1 {
            // Strips are contiguous rows of length `strip_len`.
            results.extend(Self::reduce_strips(
                &flat,
                outer,
                strip_len,
                strip_len.max(1),
                kind,
            ));
        } else {
            // Strips are strided columns; gather them first (simple, correct).
            for o in 0..outer {
                for i in 0..inner {
                    let mut strip = Vec::with_capacity(strip_len);
                    for s in 0..strip_len {
                        strip.push(flat[(o * strip_len + s) * inner + i]);
                    }
                    results.push(kind.reduce_strip(&strip)?);
                }
            }
        }

        let mut out_shape = self.view.shape.clone();
        if keepdim {
            out_shape[dim] = 1;
        } else {
            out_shape.remove(dim);
        }

        if kind.is_index_output() {
            let data: Vec<Elem> = results.into_iter().map(|v| v as Elem).collect();
            // Index outputs keep i64 semantics on the Python side; the Rust
            // f32 carrier is exact for indices < 2^24 (Phase 1 sizes).
            return Ok(Self::from_results(&out_shape, data));
        }
        let data: Vec<Elem> = results.into_iter().map(|v| v as Elem).collect();
        Ok(Self::from_results(&out_shape, data))
    }

    /// Full-tensor sum (scalar tensor). Zero for an empty tensor.
    ///
    /// # Errors
    /// Propagates [`Tensor::from_results`] shape validation.
    pub fn sum_all(&self) -> OxitorchResult<Self> {
        let sum: Elem = self.to_vec().iter().sum();
        Ok(Self::from_results(&[], vec![sum]))
    }

    /// Full-tensor mean (scalar tensor). NaN for an empty tensor, mirroring
    /// `torch.mean`.
    ///
    /// # Errors
    /// Propagates [`Tensor::from_results`] shape validation.
    pub fn mean_all(&self) -> OxitorchResult<Self> {
        let v = self.to_vec();
        let mean = v.iter().sum::<Elem>() / v.len() as Elem;
        Ok(Self::from_results(&[], vec![mean]))
    }

    /// Euclidean (L2) norm along `dim`.
    ///
    /// # Errors
    /// Propagates [`Tensor::reduce`].
    pub fn norm(&self, dim: i64, keepdim: bool) -> OxitorchResult<Self> {
        self.reduce(ops::ReduceKind::Norm, dim, keepdim)
    }

    // ---- views -----------------------------------------------------------

    /// Element count of dims before `dim` and after `dim`.
    fn outer_inner(&self, dim: usize) -> (usize, usize) {
        let outer: usize = self.view.shape[..dim].iter().product();
        let inner: usize = self.view.shape[dim + 1..].iter().product();
        (outer, inner)
    }

    /// Narrows one dim: `start`, `stop`, `step` follow Python slice rules
    /// (negative values count from the end). Always returns a view — no copy.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for step <= 0 or bad dims.
    pub fn narrow(&self, dim: i64, lo: i64, hi: i64, by: i64) -> OxitorchResult<Self> {
        let dim = layout::normalize_dim(dim, self.ndim())?;
        let len = self.view.shape[dim];
        let (lo, hi, by) = layout::normalize_slice(len, lo, hi, by)?;
        let new_len = layout::slice_len(lo, hi, by);

        let mut shape = self.view.shape.clone();
        let mut strides = self.view.strides.clone();
        shape[dim] = new_len;
        strides[dim] *= by;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset + lo * self.view.strides[dim],
            view: View::new(shape, strides),
        })
    }

    /// Selects one index along `dim`, dropping the dim (view, no copy).
    ///
    /// # Errors
    /// [`OxitorchError::OutOfBounds`] if the index is out of range.
    pub fn select(&self, dim: i64, index: i64) -> OxitorchResult<Self> {
        let dim = layout::normalize_dim(dim, self.ndim())?;
        let len = self.view.shape[dim];
        let index = if index < 0 { len as i64 + index } else { index };
        if index < 0 || index as usize >= len {
            return Err(OxitorchError::OutOfBounds(format!(
                "index {index} out of range for dim {dim} of size {len}"
            )));
        }
        let mut shape = self.view.shape.clone();
        let mut strides = self.view.strides.clone();
        shape.remove(dim);
        strides.remove(dim);
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset + index as usize * self.view.strides[dim],
            view: View::new(shape, strides),
        })
    }

    /// Gathers a list of indices along `dim` (copy; generalizes `select`).
    ///
    /// # Errors
    /// [`OxitorchError::OutOfBounds`] for any out-of-range index.
    pub fn index_select(&self, dim: i64, indices: &[i64]) -> OxitorchResult<Self> {
        let dim = layout::normalize_dim(dim, self.ndim())?;
        let len = self.view.shape[dim];
        for &i in indices {
            let idx = if i < 0 { len as i64 + i } else { i };
            if idx < 0 || idx as usize >= len {
                return Err(OxitorchError::OutOfBounds(format!(
                    "index {i} out of range for dim {dim} of size {len}"
                )));
            }
        }
        let (outer, inner) = self.outer_inner(dim);
        // Read straight from the (offset-applied) slice when contiguous —
        // this kernel must not materialize the whole tensor per call.
        let materialized;
        let src: &[Elem] = if self.is_contiguous() {
            self.as_slice()
                .expect("contiguity checked immediately above")
        } else {
            materialized = self.contiguous();
            materialized.as_slice().expect("contiguous() output")
        };
        let row = len as usize * inner;
        let mut out = Vec::with_capacity(indices.len() * outer * inner);
        for o in 0..outer {
            for &i in indices {
                let idx = if i < 0 {
                    (len as i64 + i) as usize
                } else {
                    i as usize
                };
                let start = o * row + idx * inner;
                out.extend_from_slice(&src[start..start + inner]);
            }
        }
        let mut shape = self.view.shape.clone();
        shape[dim] = indices.len();
        Ok(Self::from_results(&shape, out))
    }

    /// Reshapes to `new_shape` (`-1` infers the missing dim). Returns a view
    /// when possible, otherwise a contiguous copy.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if the element counts disagree or
    /// more than one `-1` is given.
    pub fn reshape(&self, new_shape: &[i64]) -> OxitorchResult<Self> {
        let infer_count = new_shape.iter().filter(|&&d| d == -1).count();
        if infer_count > 1 {
            return Err(OxitorchError::InvalidArgument(
                "can only infer one dimension with -1".into(),
            ));
        }
        let numel = self.numel();
        let mut dims: Vec<usize> = Vec::with_capacity(new_shape.len());
        let mut inferred_at: Option<usize> = None;
        let mut known: usize = 1;
        for (i, &d) in new_shape.iter().enumerate() {
            if d == -1 {
                inferred_at = Some(i);
                dims.push(1);
            } else if d < 0 {
                return Err(OxitorchError::InvalidArgument(format!(
                    "invalid dimension {d} in reshape"
                )));
            } else {
                dims.push(d as usize);
                known = known.saturating_mul(d as usize);
            }
        }
        if let Some(i) = inferred_at {
            if numel % known != 0 {
                return Err(OxitorchError::InvalidArgument(format!(
                    "cannot reshape {numel} elements into shape {new_shape:?}"
                )));
            }
            dims[i] = numel / known;
        }
        if layout::element_count(&dims) != numel {
            return Err(OxitorchError::InvalidArgument(format!(
                "cannot reshape {numel} elements into shape {new_shape:?}"
            )));
        }

        if self.is_contiguous() {
            return Ok(Self {
                storage: Arc::clone(&self.storage),
                offset: self.offset,
                view: View::row_major(dims),
            });
        }
        // Non-contiguous fallback: materialize then re-view.
        let data = self.to_vec();
        Ok(Self {
            storage: Arc::new(data),
            offset: 0,
            view: View::row_major(dims),
        })
    }

    /// Permutes dims according to `order` (a view, never a copy).
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if `order` is not a permutation of
    /// the tensor's dims.
    pub fn permute(&self, order: &[i64]) -> OxitorchResult<Self> {
        if order.len() != self.ndim() {
            return Err(OxitorchError::InvalidArgument(format!(
                "permute expects {} dims, got {}",
                self.ndim(),
                order.len()
            )));
        }
        let mut seen = vec![false; self.ndim()];
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        for &d in order {
            let dim = layout::normalize_dim(d, self.ndim())?;
            if seen[dim] {
                return Err(OxitorchError::InvalidArgument(format!(
                    "repeated dim {d} in permute"
                )));
            }
            seen[dim] = true;
            shape.push(self.view.shape[dim]);
            strides.push(self.view.strides[dim]);
        }
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            view: View::new(shape, strides),
        })
    }

    /// 2-D transpose (view).
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if the tensor is not 2-D.
    pub fn transpose(&self) -> OxitorchResult<Self> {
        if self.ndim() != 2 {
            return Err(OxitorchError::InvalidArgument(
                "transpose() expects a 2-D tensor; use permute for other ranks".into(),
            ));
        }
        self.permute(&[1, 0])
    }

    /// Expands the tensor to `out_shape` following broadcast rules. Returns a
    /// view with zero strides where broadcasting occurs.
    ///
    /// # Errors
    /// [`OxitorchError::ShapeMismatch`] if the current shape cannot broadcast
    /// to `out_shape`.
    pub fn broadcast_to(&self, out_shape: &[usize]) -> OxitorchResult<Self> {
        let strides = layout::broadcast_strides(self.shape(), self.strides(), out_shape)?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            view: View::new(out_shape.to_vec(), strides),
        })
    }

    // ---- fancy indexing ---------------------------------------------------

    /// Gathers elements at `indices` (per-flat-index, `i64`) into a 1-D
    /// tensor of the same length — the "fancy indexing" building block.
    ///
    /// # Errors
    /// [`OxitorchError::OutOfBounds`] for any out-of-range index.
    pub fn gather_flat(&self, indices: &[i64]) -> OxitorchResult<Self> {
        let flat = self.to_vec();
        let mut out = Vec::with_capacity(indices.len());
        for &i in indices {
            let idx = if i < 0 { flat.len() as i64 + i } else { i };
            if idx < 0 || idx as usize >= flat.len() {
                return Err(OxitorchError::OutOfBounds(format!(
                    "flat index {i} out of range for {} elements",
                    flat.len()
                )));
            }
            out.push(flat[idx as usize]);
        }
        Ok(Self::from_results(&[indices.len()], out))
    }

    /// Gathers `self[indices[i], j...]` for a 1-D `indices` over dim 0.
    ///
    /// General N-d gather (arbitrary index tensor shapes) lands with the
    /// dtype-generic pass; this covers the hot `embedding`-style case.
    ///
    /// # Errors
    /// [`OxitorchError::OutOfBounds`] for out-of-range indices.
    pub fn gather_dim0(&self, indices: &[i64]) -> OxitorchResult<Self> {
        if self.ndim() == 0 {
            return Err(OxitorchError::InvalidArgument(
                "gather_dim0 requires at least a 1-D tensor".into(),
            ));
        }
        let rows = self.view.shape[0];
        for &i in indices {
            let idx = if i < 0 { rows as i64 + i } else { i };
            if idx < 0 || idx as usize >= rows {
                return Err(OxitorchError::OutOfBounds(format!(
                    "index {i} out of range for dim 0 of size {rows}"
                )));
            }
        }
        let inner: usize = self.view.shape[1..].iter().product();
        let flat = self.to_vec();
        let mut out = Vec::with_capacity(indices.len() * inner);
        for &i in indices {
            let idx = if i < 0 {
                (rows as i64 + i) as usize
            } else {
                i as usize
            };
            out.extend_from_slice(&flat[idx * inner..(idx + 1) * inner]);
        }
        let mut shape = self.view.shape.clone();
        shape[0] = indices.len();
        Ok(Self::from_results(&shape, out))
    }

    /// Writes `values` into `self[indices[i], ...]` (dim 0), returning a new
    /// contiguous tensor — the "scatter" building block.
    ///
    /// # Errors
    /// [`OxitorchError::OutOfBounds`] for out-of-range indices, or
    /// [`OxitorchError::ShapeMismatch`] if `values` does not match the
    /// selected rows.
    pub fn scatter_dim0(&self, indices: &[i64], values: &Self) -> OxitorchResult<Self> {
        if self.ndim() == 0 {
            return Err(OxitorchError::InvalidArgument(
                "scatter_dim0 requires at least a 1-D tensor".into(),
            ));
        }
        let inner: usize = self.view.shape[1..].iter().product();
        let mut shape = self.view.shape.clone();
        shape[0] = indices.len();
        let expected = Self::from_results(&shape, vec![0.0; indices.len() * inner]);
        if values.shape() != expected.shape() {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{:?}", values.shape()),
                rhs: format!("{shape:?}"),
            });
        }
        let rows = self.view.shape[0];
        for &i in indices {
            let idx = if i < 0 { rows as i64 + i } else { i };
            if idx < 0 || idx as usize >= rows {
                return Err(OxitorchError::OutOfBounds(format!(
                    "index {i} out of range for dim 0 of size {rows}"
                )));
            }
        }
        let mut data = self.to_vec();
        let vflat = values.to_vec();
        for (pos, &i) in indices.iter().enumerate() {
            let idx = if i < 0 {
                (rows as i64 + i) as usize
            } else {
                i as usize
            };
            data[idx * inner..(idx + 1) * inner]
                .copy_from_slice(&vflat[pos * inner..(pos + 1) * inner]);
        }
        Ok(Self::from_results(self.shape(), data))
    }

    /// Selects elements where `mask` (broadcast against `self`) is nonzero,
    /// returning a 1-D tensor — boolean-mask indexing.
    ///
    /// # Errors
    /// Shape errors when `mask` cannot broadcast against `self`.
    pub fn masked_select(&self, mask: &Self) -> OxitorchResult<Self> {
        let out_shape = layout::broadcast_shapes(self.shape(), mask.shape())?;
        let a_strides = layout::broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let m_strides = layout::broadcast_strides(mask.shape(), mask.strides(), &out_shape)?;
        let picked: Vec<Elem> = iterate::map_n(
            &out_shape,
            &[
                Operand {
                    data: &self.storage[self.offset..],
                    strides: &a_strides,
                },
                Operand {
                    data: &mask.storage[mask.offset..],
                    strides: &m_strides,
                },
            ],
            |v| if v[1] == 0.0 { f32::NAN } else { v[0] },
        )
        .into_iter()
        .filter(|v| !v.is_nan())
        .collect();
        Ok(Self::from_results(&[picked.len()], picked))
    }

    // ---- matmul ------------------------------------------------------------

    /// Matrix multiplication for 2-D tensors: `(m, k) @ (k, n) -> (m, n)`.
    ///
    /// Rows are processed in parallel blocks for large matrices. Non
    /// contiguous inputs are materialized first (Phase 1 tradeoff; strided
    /// GEMM arrives with the dtype-generic pass).
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if either input is not 2-D,
    /// [`OxitorchError::ShapeMismatch`] if the inner dimensions disagree.
    #[allow(unsafe_code, clippy::many_single_char_names)] // sanctioned unsafe GEMM below; m/k/n are the convention
    pub fn matmul(&self, rhs: &Self) -> OxitorchResult<Self> {
        if self.ndim() != 2 || rhs.ndim() != 2 {
            return Err(OxitorchError::InvalidArgument(format!(
                "matmul expects 2-D tensors, got ndim {} and {}",
                self.ndim(),
                rhs.ndim()
            )));
        }
        let (m, k) = (self.shape()[0], self.shape()[1]);
        let (k2, n) = (rhs.shape()[0], rhs.shape()[1]);
        if k != k2 {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("[{m}, {k}]"),
                rhs: format!("[{k2}, {n}]"),
            });
        }
        let a = self.contiguous();
        let b = rhs.contiguous();
        let a = a.as_slice().expect("contiguous");
        let b = b.as_slice().expect("contiguous");

        let mut c = vec![0.0f32; m * n];
        if m * k * n < ops::PARALLEL_GEMM_THRESHOLD_ELEMENTS {
            // SAFETY: `sgemm` reads `m * k` elements from `a` and `k * n`
            // from `b`, and writes `m * n` to `c`. All three are live,
            // non-aliasing, contiguous row-major f32 buffers; the row/column
            // strides below are in elements and match that layout.
            // `beta = 0.0` means `c` is never read.
            unsafe {
                matrixmultiply::sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    a.as_ptr(),
                    k as isize,
                    1,
                    b.as_ptr(),
                    n as isize,
                    1,
                    0.0,
                    c.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        } else {
            // Parallel row-block GEMM: each block owns a disjoint slice of c.
            // A fixed split (rows = 2 x threads) amortizes better than one
            // chunk per thread and keeps every thread busy per bench data.
            let rows = a.len() / k;
            let block = rows.div_ceil(rayon::current_num_threads().max(1) * 2);
            c.par_chunks_mut(block * n)
                .enumerate()
                .for_each(|(bi, chunk)| {
                    let start_row = bi * block;
                    let end_row = (start_row + chunk.len() / n).min(rows);
                    // SAFETY: same buffer contract as the sequential call
                    // above; each parallel chunk writes a disjoint row range.
                    unsafe {
                        matrixmultiply::sgemm(
                            end_row - start_row,
                            k,
                            n,
                            1.0,
                            a[start_row * k..].as_ptr(),
                            k as isize,
                            1,
                            b.as_ptr(),
                            n as isize,
                            1,
                            0.0,
                            chunk.as_mut_ptr(),
                            n as isize,
                            1,
                        );
                    }
                });
        }
        Self::from_vec(&[m as i64, n as i64], c)
    }

    /// Batched matrix multiply: `(b, m, k) @ (b, k, n) -> (b, m, n)`, batch
    /// element `i` being `a[i] @ b[i]`. Inputs are materialized contiguous
    /// (views are copied); each batch runs the same SGEMM kernel as
    /// [`Tensor::matmul`], large batches fanning out across threads.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] unless both inputs are 3-D, and
    /// [`OxitorchError::ShapeMismatch`] when the inner dims disagree or the
    /// batch sizes differ.
    pub fn bmm(&self, rhs: &Self) -> OxitorchResult<Self> {
        if self.ndim() != 3 || rhs.ndim() != 3 {
            return Err(OxitorchError::InvalidArgument(format!(
                "bmm expects 3-D tensors, got ndim {} and {}",
                self.ndim(),
                rhs.ndim()
            )));
        }
        let (rows_a, cols_a, inner_a) = (self.shape()[0], self.shape()[1], self.shape()[2]);
        let (rows_b, inner_b, cols_b) = (rhs.shape()[0], rhs.shape()[1], rhs.shape()[2]);
        if rows_a != rows_b {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{rows_a} batches"),
                rhs: format!("{rows_b} batches"),
            });
        }
        if inner_a != inner_b {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("[{rows_a}, {cols_a}, {inner_a}]"),
                rhs: format!("[{rows_b}, {inner_b}, {cols_b}]"),
            });
        }
        let a = self.contiguous();
        let b = rhs.contiguous();
        let a = a.as_slice().expect("contiguous");
        let b = b.as_slice().expect("contiguous");

        let (rows, inner, cols) = (cols_a, inner_a, cols_b);
        let plane = rows * cols;
        let mut c = vec![0.0f32; rows_a * plane];
        let gemm_elems = rows * inner * cols;
        if rows_a * gemm_elems < ops::PARALLEL_GEMM_THRESHOLD_ELEMENTS {
            // SAFETY: per batch `i`, `sgemm` reads `rows * inner` elements
            // from `a[i*rows*inner..]` and `inner * cols` from
            // `b[i*inner*cols..]`, writing `rows * cols` into `c[i*plane..]`;
            // all buffers are live, non-aliasing, contiguous row-major f32
            // with element-unit strides, and `beta = 0.0` means `c` is never
            // read.
            for bi in 0..rows_a {
                #[allow(unsafe_code)] // sanctioned GEMM call, see workspace lints
                unsafe {
                    matrixmultiply::sgemm(
                        rows,
                        inner,
                        cols,
                        1.0,
                        a[bi * rows * inner..].as_ptr(),
                        inner as isize,
                        1,
                        b[bi * inner * cols..].as_ptr(),
                        cols as isize,
                        1,
                        0.0,
                        c[bi * plane..].as_mut_ptr(),
                        cols as isize,
                        1,
                    );
                }
            }
        } else {
            // Large workload: one batch plane per rayon task.
            c.par_chunks_mut(plane).enumerate().for_each(|(bi, chunk)| {
                // SAFETY: as above; each task writes one disjoint batch
                // plane of `c`.
                #[allow(unsafe_code)] // sanctioned GEMM call, see workspace lints
                unsafe {
                    matrixmultiply::sgemm(
                        rows,
                        inner,
                        cols,
                        1.0,
                        a[bi * rows * inner..].as_ptr(),
                        inner as isize,
                        1,
                        b[bi * inner * cols..].as_ptr(),
                        cols as isize,
                        1,
                        0.0,
                        chunk.as_mut_ptr(),
                        cols as isize,
                        1,
                    );
                }
            });
        }
        Self::from_vec(&[rows_a as i64, rows as i64, cols as i64], c)
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("dtype", &self.dtype().name())
            .field("device", &self.device().to_string())
            .finish()
    }
}

/// Human-readable print in the spirit of `torch.Tensor.__repr__`, kept
/// deliberately small: shape header plus up to 6 rows/values.
impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.ndim() {
            0 => write!(f, "{}", self.to_vec()[0]),
            1 => {
                writeln!(
                    f,
                    "Tensor(shape={:?}, dtype={})",
                    self.shape(),
                    self.dtype()
                )?;
                for (i, v) in self.to_vec().into_iter().take(6).enumerate() {
                    writeln!(f, "  [{i}] {v}")?;
                }
                if self.numel() > 6 {
                    writeln!(f, "  ... ({} more)", self.numel() - 6)?;
                }
                Ok(())
            }
            _ => {
                writeln!(
                    f,
                    "Tensor(shape={:?}, dtype={})",
                    self.shape(),
                    self.dtype()
                )?;
                let data = self.to_vec();
                let rows = self.shape()[0].min(6);
                let cols = self.shape().last().copied().unwrap_or(0);
                for r in 0..rows {
                    writeln!(f, "  {:?}", &data[r * cols..(r + 1) * cols])?;
                }
                if self.shape()[0] > 6 {
                    writeln!(f, "  ... ({} more rows)", self.shape()[0] - 6)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests;
