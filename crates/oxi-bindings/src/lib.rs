//! PyO3 binding layer for oxitorch.
//!
//! Thin by design (plan.md): no kernel or engine logic lives here. The
//! user-facing `Tensor` wraps an `oxi_autograd::Var`, so every op records
//! into the computation graph and `backward()` works from Python. Ops on
//! detached tensors (comparisons, argmax, printing, numpy conversion) work
//! on the underlying value directly.

// pyo3 extracts owned collections (`Vec<i64>`) into `#[pyfunction]`/method
// params by value; passing them by ref would require cloning at the call
// site anyway, so the pedantic lint is silenced at the crate boundary.
#![allow(clippy::needless_pass_by_value)]

pub mod data;
mod error;

use oxi_autograd::graph::Var;
use oxi_autograd::ops as aops;
use oxi_tensor::ops::ReduceKind;
use oxi_tensor::Tensor as RTensor;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PySlice};

use crate::error::to_pyerr;
use oxi_core::OxitorchResult;

/// Extracts a contiguous numpy array (or any f32 buffer object) into a flat
/// vec + shape.
fn buffer_to_f32(py: Python<'_>, buffer: &PyBuffer<f32>) -> PyResult<(Vec<f32>, Vec<i64>)> {
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "only contiguous numpy arrays are accepted at the boundary; call .copy() first",
        ));
    }
    let shape: Vec<i64> = buffer.shape().iter().map(|&d| d as i64).collect();
    let data = buffer
        .to_vec(py)
        .map_err(|_| PyValueError::new_err("buffer could not be read as f32"))?;
    Ok((data, shape))
}

/// The user-facing tensor object (`oxitorch.Tensor`).
#[pyclass(name = "Tensor")]
#[derive(Clone)]
pub struct PyTensor {
    inner: Var,
}

/// Binary-dunder operand: another tensor or a Python scalar.
enum PyOperand {
    /// A tensor operand.
    Tensor(PyTensor),
    /// A Python number (int or float), promoted to f32.
    Scalar(f32),
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyOperand {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(t) = obj.extract::<PyTensor>() {
            return Ok(Self::Tensor(t));
        }
        if let Ok(c) = obj.extract::<f64>() {
            return Ok(Self::Scalar(c as f32));
        }
        Err(PyTypeError::new_err(
            "operands must be oxitorch.Tensor or a Python number",
        ))
    }
}

impl PyTensor {
    fn new(inner: Var) -> Self {
        Self { inner }
    }

    fn value(&self) -> &RTensor {
        self.inner.value()
    }

    /// A constant full tensor of `c` with this tensor's shape (for promoting
    /// Python scalars to constant graph leaves).
    fn const_full(&self, c: f32) -> RTensor {
        let shape: Vec<i64> = self.value().shape().iter().map(|&d| d as i64).collect();
        RTensor::full(&shape, c).expect("full with a valid shape cannot fail")
    }

    /// Binary op with a tensor or scalar operand; scalars are promoted to
    /// constant leaves so autograd flows through unchanged.
    fn operand_binary(
        &self,
        other: PyOperand,
        f: impl Fn(&Var, &Var) -> OxitorchResult<Var>,
    ) -> PyResult<PyTensor> {
        match other {
            PyOperand::Tensor(t) => f(&self.inner, &t.inner),
            PyOperand::Scalar(c) => f(&self.inner, &Var::leaf(self.const_full(c), false)),
        }
        .map(PyTensor::new)
        .map_err(to_pyerr)
    }

    fn binary(
        &self,
        f: impl Fn(&Var, &Var) -> OxitorchResult<Var>,
        other: &PyTensor,
    ) -> PyResult<PyTensor> {
        f(&self.inner, &other.inner)
            .map(PyTensor::new)
            .map_err(to_pyerr)
    }

    fn cmp(&self, name: &str, other: &PyTensor) -> PyResult<PyTensor> {
        self.value()
            .cmp_op(name, other.value())
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    /// Full-tensor reduction to a scalar tensor (torch's `t.sum()` without
    /// a `dim`). Differentiable for sum/mean by reducing each dim in turn
    /// (keepdim, so each dim index stays valid across the loop).
    fn reduce_all(&self, kind: ReduceKind) -> PyResult<PyTensor> {
        if self.inner.requires_grad() && matches!(kind, ReduceKind::Sum | ReduceKind::Mean) {
            let ndim = self.value().ndim();
            if ndim == 0 {
                return Ok(PyTensor::new(self.inner.clone()));
            }
            let mut v = self.inner.clone();
            for d in 0..ndim {
                v = aops::reduce(&v, kind, d as i64, true).map_err(to_pyerr)?;
            }
            return aops::reshape(&v, &[]).map(PyTensor::new).map_err(to_pyerr);
        }
        let v = self.value().to_vec();
        let r = kind.reduce_strip(&v).map_err(to_pyerr)?;
        RTensor::from_vec(&[], vec![r as f32])
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }
}

#[pymethods]
impl PyTensor {
    /// Builds from a numpy array, any f32 buffer object, or a sequence of
    /// numbers (1-D).
    #[new]
    #[pyo3(signature = (buffer, requires_grad=false))]
    fn py_new(buffer: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<Self> {
        if let Ok(buf) = PyBuffer::<f32>::get(buffer) {
            let (data, shape) = buffer_to_f32(buffer.py(), &buf)?;
            return RTensor::from_vec(&shape, data)
                .map(|t| PyTensor::new(Var::leaf(t, requires_grad)))
                .map_err(to_pyerr);
        }
        let data: Vec<f32> = buffer.extract().map_err(|_| {
            PyTypeError::new_err("expected a numpy array, f32 buffer, or sequence of numbers")
        })?;
        RTensor::from_vec(&[data.len() as i64], data)
            .map(|t| PyTensor::new(Var::leaf(t, requires_grad)))
            .map_err(to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("{}", self.value())
    }

    fn __str__(&self) -> String {
        format!("{}", self.value())
    }

    fn __len__(&self) -> PyResult<usize> {
        if self.value().ndim() == 0 {
            return Err(PyTypeError::new_err("len() of a 0-d tensor"));
        }
        Ok(self.value().shape()[0])
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        // Integer keys route through the differentiable `select`; slice
        // keys through the differentiable `narrow` (step 1).
        if let Ok(i) = key.extract::<i64>() {
            return self.select(0, i);
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let lo = slice
                .getattr("start")?
                .extract::<Option<i64>>()?
                .unwrap_or(0);
            let hi = slice
                .getattr("stop")?
                .extract::<Option<i64>>()?
                .unwrap_or(i64::MAX);
            let by = slice
                .getattr("step")?
                .extract::<Option<i64>>()?
                .unwrap_or(1);
            return self.narrow(0, lo, hi, by);
        }
        if let Ok(indices) = key.extract::<Vec<i64>>() {
            // Fancy indexing (integer array) along dim 0 == index_select.
            return self.index_select(0, indices);
        }
        Err(PyTypeError::new_err(
            "tensor indices must be integers, slices, or integer lists",
        ))
    }

    fn __add__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::add)
    }

    fn __radd__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::add)
    }

    fn __sub__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::sub)
    }

    fn __rsub__(&self, other: PyOperand) -> PyResult<PyTensor> {
        match other {
            PyOperand::Scalar(c) => {
                // c - t = (-t) + c
                let neg = aops::neg(&self.inner).map_err(to_pyerr)?;
                aops::add(&neg, &Var::leaf(self.const_full(c), false))
            }
            PyOperand::Tensor(t) => aops::sub(&t.inner, &self.inner),
        }
        .map(PyTensor::new)
        .map_err(to_pyerr)
    }

    fn __mul__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::mul)
    }

    fn __rmul__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::mul)
    }

    fn __truediv__(&self, other: PyOperand) -> PyResult<PyTensor> {
        self.operand_binary(other, aops::div)
    }

    fn __rtruediv__(&self, other: PyOperand) -> PyResult<PyTensor> {
        match other {
            PyOperand::Scalar(c) => {
                // c / t = c * t^(-1); pow and mul are both differentiable.
                let inv = aops::pow(&self.inner, &Var::leaf(self.const_full(-1.0), false))
                    .map_err(to_pyerr)?;
                aops::mul(&inv, &Var::leaf(self.const_full(c), false))
            }
            PyOperand::Tensor(t) => aops::div(&t.inner, &self.inner),
        }
        .map(PyTensor::new)
        .map_err(to_pyerr)
    }

    fn __pow__(&self, other: PyOperand, _modulo: Option<&PyTensor>) -> PyResult<PyTensor> {
        match other {
            PyOperand::Tensor(t) => self.binary(aops::pow, &t),
            PyOperand::Scalar(c) => {
                if self.inner.requires_grad() {
                    let exp = Var::leaf(self.const_full(c), false);
                    aops::pow(&self.inner, &exp)
                        .map(PyTensor::new)
                        .map_err(to_pyerr)
                } else {
                    self.value()
                        .pow(&self.const_full(c))
                        .map(|t| PyTensor::new(Var::leaf(t, false)))
                        .map_err(to_pyerr)
                }
            }
        }
    }

    fn __neg__(&self) -> PyResult<PyTensor> {
        aops::neg(&self.inner).map(PyTensor::new).map_err(to_pyerr)
    }

    fn __abs__(&self) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            aops::abs(&self.inner).map(PyTensor::new).map_err(to_pyerr)
        } else {
            self.value()
                .unary_op("abs")
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    // Comparison dunders return bool tensors (detached), mirroring torch.
    fn __eq__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("eq", other)
    }

    fn __ne__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("ne", other)
    }

    fn __lt__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("lt", other)
    }

    fn __le__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("le", other)
    }

    fn __gt__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("gt", other)
    }

    fn __ge__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.cmp("ge", other)
    }

    fn __matmul__(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::matmul, other)
    }

    /// `float(t)` for one-element tensors.
    fn __float__(&self) -> PyResult<f64> {
        let v = self.value().to_vec();
        if v.len() != 1 {
            return Err(PyTypeError::new_err(format!(
                "only one-element tensors can be converted to float, got {} elements",
                v.len()
            )));
        }
        Ok(f64::from(v[0]))
    }

    // ---- autograd -----------------------------------------------------------

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    /// The accumulated gradient, if any.
    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner
            .grad()
            .map(|g| PyTensor::new(Var::leaf(g, false)))
    }

    /// Overwrites the accumulated gradient (`t.grad = other`); used by
    /// optimizers/clipping for in-place gradient surgery.
    #[setter]
    fn set_grad(&self, value: &PyTensor) -> PyResult<()> {
        self.inner.set_grad(value.value().clone()).map_err(to_pyerr)
    }

    /// Detaches from the graph (returns a new non-differentiable Tensor).
    fn detach(&self) -> PyTensor {
        PyTensor::new(Var::leaf(self.inner.detach(), false))
    }

    /// Backward pass from this tensor (must be scalar unless `grad` given).
    ///
    /// By default the recorded graph is freed by the pass; pass
    /// `retain_graph=True` to backward through the same graph again
    /// (contributions accumulate into `.grad`).
    #[pyo3(signature = (grad=None, *, retain_graph=false))]
    fn backward(&self, grad: Option<&PyTensor>, retain_graph: bool) -> PyResult<()> {
        let seed = grad.map(|g| g.value().clone());
        oxi_autograd::graph::backward_ext(&self.inner, seed, retain_graph).map_err(to_pyerr)
    }

    fn zero_grad(&self) -> PyResult<()> {
        self.inner.zero_grad().map_err(to_pyerr)
    }

    /// Numpy array (copy via `frombuffer`, then reshaped). Non-contiguous
    /// tensors are materialized first.
    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let flat = self.value().contiguous().to_vec();
        let bytes: &[u8] = bytemuck::cast_slice(&flat);
        let np = py.import("numpy")?;
        let dtype = np.getattr("float32")?;
        let buf = PyBytes::new(py, bytes);
        let arr = np.call_method1("frombuffer", (buf, dtype))?;
        let shape: Vec<isize> = self.value().shape().iter().map(|&d| d as isize).collect();
        arr.call_method1("reshape", (shape,))
    }

    // ---- metadata -----------------------------------------------------------

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.value().shape().to_vec()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.value().dtype().name().to_string()
    }

    #[getter]
    fn device(&self) -> String {
        self.value().device().to_string()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.value().ndim()
    }

    fn numel(&self) -> usize {
        self.value().numel()
    }

    fn is_contiguous(&self) -> bool {
        self.value().is_contiguous()
    }

    // ---- elementwise & reductions --------------------------------------------

    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::add, other)
    }

    fn sub(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::sub, other)
    }

    fn mul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::mul, other)
    }

    fn div(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::div, other)
    }

    fn pow(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::pow, other)
    }

    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::matmul, other)
    }

    /// Batched matmul `(b,m,k) @ (b,k,n) -> (b,m,n)` (torch parity).
    fn bmm(&self, other: &PyTensor) -> PyResult<PyTensor> {
        self.binary(aops::bmm, other)
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn sum(&self, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
        match dim {
            Some(d) => aops::reduce(&self.inner, ReduceKind::Sum, d, keepdim)
                .map(PyTensor::new)
                .map_err(to_pyerr),
            None => self.reduce_all(ReduceKind::Sum),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
        match dim {
            Some(d) => aops::reduce(&self.inner, ReduceKind::Mean, d, keepdim)
                .map(PyTensor::new)
                .map_err(to_pyerr),
            None => self.reduce_all(ReduceKind::Mean),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn max(&self, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
        match dim {
            Some(d) => aops::reduce(&self.inner, ReduceKind::Max, d, keepdim)
                .map(PyTensor::new)
                .map_err(to_pyerr),
            None => self.reduce_all(ReduceKind::Max),
        }
    }

    #[pyo3(signature = (dim=None, keepdim=false))]
    fn min(&self, dim: Option<i64>, keepdim: bool) -> PyResult<PyTensor> {
        match dim {
            Some(d) => aops::reduce(&self.inner, ReduceKind::Min, d, keepdim)
                .map(PyTensor::new)
                .map_err(to_pyerr),
            None => self.reduce_all(ReduceKind::Min),
        }
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmax(&self, dim: i64, keepdim: bool) -> PyResult<PyTensor> {
        self.value()
            .reduce(ReduceKind::ArgMax, dim, keepdim)
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn argmin(&self, dim: i64, keepdim: bool) -> PyResult<PyTensor> {
        self.value()
            .reduce(ReduceKind::ArgMin, dim, keepdim)
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn norm(&self, dim: i64, keepdim: bool) -> PyResult<PyTensor> {
        aops::reduce(&self.inner, ReduceKind::Norm, dim, keepdim)
            .map(PyTensor::new)
            .map_err(to_pyerr)
    }

    // ---- views & indexing (autograd where Phase 2 covers them) --------------

    fn reshape(&self, shape: Vec<i64>) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            aops::reshape(&self.inner, &shape)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .reshape(&shape)
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn view(&self, shape: Vec<i64>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }

    fn transpose(&self) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            aops::transpose(&self.inner)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .transpose()
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn permute(&self, dims: Vec<i64>) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            aops::permute(&self.inner, &dims)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .permute(&dims)
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn narrow(&self, dim: i64, lo: i64, hi: i64, by: i64) -> PyResult<PyTensor> {
        // `narrow` with step 1 is a contiguous window: express via the
        // differentiable dim-0 narrow (permute the window dim to 0 first
        // when needed). Stepped slices stay detached for now.
        if by == 1 && self.inner.requires_grad() {
            let dim_len = self.value().shape()[dim as usize] as i64;
            let offset = if lo < 0 { dim_len + lo } else { lo };
            let len = self
                .value()
                .narrow(dim, lo, hi, by)
                .map_err(to_pyerr)?
                .shape()[dim as usize] as i64;
            let windowed = if dim == 0 {
                aops::narrow_dim0(&self.inner, offset, len)
            } else {
                let ndim = self.value().ndim() as i64;
                let mut order: Vec<i64> = (0..ndim).collect();
                order.swap(0, dim as usize);
                let moved = aops::permute(&self.inner, &order).map_err(to_pyerr)?;
                aops::narrow_dim0(&moved, offset, len).and_then(|v| {
                    let mut inv: Vec<i64> = (0..ndim).collect();
                    inv.swap(0, dim as usize);
                    aops::permute(&v, &inv)
                })
            };
            return windowed.map(PyTensor::new).map_err(to_pyerr);
        }
        self.value()
            .narrow(dim, lo, hi, by)
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    fn select(&self, dim: i64, index: i64) -> PyResult<PyTensor> {
        // `select` = narrow to one row along `dim` + drop the dim via
        // reshape — both differentiable through the narrow path above.
        if self.inner.requires_grad() {
            let d = dim;
            let idx = if index < 0 {
                self.value().shape()[dim as usize] as i64 + index
            } else {
                index
            };
            let out_shape: Vec<i64> = self
                .value()
                .select(d, idx)
                .map_err(to_pyerr)?
                .shape()
                .iter()
                .map(|&x| x as i64)
                .collect();
            let ndim = self.value().ndim() as i64;
            let windowed = if d == 0 {
                aops::narrow_dim0(&self.inner, idx, 1)
            } else {
                let mut order: Vec<i64> = (0..ndim).collect();
                order.swap(0, d as usize);
                let moved = aops::permute(&self.inner, &order).map_err(to_pyerr)?;
                aops::narrow_dim0(&moved, idx, 1)
            };
            return aops::reshape(&windowed.map_err(to_pyerr)?, &out_shape)
                .map(PyTensor::new)
                .map_err(to_pyerr);
        }
        self.value()
            .select(dim, index)
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    fn index_select(&self, dim: i64, indices: Vec<i64>) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            aops::index_select(&self.inner, dim, &indices)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .index_select(dim, &indices)
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn gather(&self, indices: Vec<i64>) -> PyResult<PyTensor> {
        // gather along dim 0 == index_select; keep it in the graph.
        self.index_select(0, indices)
    }

    fn scatter(&self, indices: Vec<i64>, values: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.requires_grad() || values.inner.requires_grad() {
            aops::scatter_dim0(&self.inner, &indices, &values.inner)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .scatter_dim0(&indices, values.value())
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn masked_select(&self, mask: &PyTensor) -> PyResult<PyTensor> {
        if self.inner.requires_grad() || mask.inner.requires_grad() {
            aops::masked_select(&self.inner, &mask.inner)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .masked_select(mask.value())
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    fn contiguous(&self) -> PyTensor {
        PyTensor::new(Var::leaf(self.value().contiguous(), false))
    }

    /// Applies a named unary op (`t.apply("relu")`). Autograd-connected for
    /// the ops Phase 2 differentiates; others detach.
    fn apply(&self, name: &str) -> PyResult<PyTensor> {
        if self.inner.requires_grad() {
            let v = match name {
                "relu" => aops::relu(&self.inner),
                "sigmoid" => aops::sigmoid(&self.inner),
                "tanh" => aops::tanh(&self.inner),
                "exp" => aops::exp(&self.inner),
                "log" => aops::log(&self.inner),
                "sqrt" => aops::sqrt(&self.inner),
                "neg" => aops::neg(&self.inner),
                "abs" => aops::abs(&self.inner),
                // Unimplemented differentiable unaries: return an honest
                // detached leaf (via the plain fallthrough below) rather
                // than a lying "requires grad" leaf that eats gradients.
                _ => return self.apply_detached(name),
            };
            return v.map(PyTensor::new).map_err(to_pyerr);
        }
        self.apply_detached(name)
    }

    /// `apply` for tensors that do not require grad (and the honest fallback
    /// for not-yet-differentiable unaries).
    fn apply_detached(&self, name: &str) -> PyResult<PyTensor> {
        self.value()
            .unary_op(name)
            .map(|t| PyTensor::new(Var::leaf(t, false)))
            .map_err(to_pyerr)
    }

    /// Expands to `shape` following broadcast rules (zero-stride view).
    fn broadcast_to(&self, shape: Vec<i64>) -> PyResult<PyTensor> {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if self.inner.requires_grad() {
            aops::broadcast_to(&self.inner, &dims)
                .map(PyTensor::new)
                .map_err(to_pyerr)
        } else {
            self.value()
                .broadcast_to(&dims)
                .map(|t| PyTensor::new(Var::leaf(t, false)))
                .map_err(to_pyerr)
        }
    }

    /// Flat row-major values as a Python list.
    fn tolist(&self) -> Vec<f32> {
        self.value().to_vec()
    }

    /// The scalar value of a 1-element tensor (torch parity).
    fn item(&self) -> PyResult<f32> {
        if self.value().numel() != 1 {
            return Err(PyValueError::new_err(format!(
                "only one-element tensors can be converted to scalars (got {})",
                self.value().numel()
            )));
        }
        Ok(self.value().contiguous().to_vec()[0])
    }
}

/// `oxitorch.zeros(shape)`.
#[pyfunction]
fn zeros(shape: Vec<i64>) -> PyResult<PyTensor> {
    RTensor::zeros(&shape)
        .map(|t| PyTensor::new(Var::leaf(t, false)))
        .map_err(to_pyerr)
}

/// `oxitorch.full(shape, value)`.
#[pyfunction]
#[pyo3(signature = (shape, value, requires_grad=false))]
fn full(shape: Vec<i64>, value: f32, requires_grad: bool) -> PyResult<PyTensor> {
    RTensor::full(&shape, value)
        .map(|t| PyTensor::new(Var::leaf(t, requires_grad)))
        .map_err(to_pyerr)
}

/// `oxitorch.eye(n)`.
#[pyfunction]
fn eye(n: usize) -> PyResult<PyTensor> {
    RTensor::eye(n)
        .map(|t| PyTensor::new(Var::leaf(t, false)))
        .map_err(to_pyerr)
}

/// `oxitorch.from_numpy(array)` — accepts any contiguous f32 buffer object.
#[pyfunction]
#[pyo3(signature = (array, requires_grad=false))]
fn from_numpy(array: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<PyTensor> {
    let buf = PyBuffer::<f32>::get(array)
        .map_err(|_| PyTypeError::new_err("expected a numpy array or f32 buffer object"))?;
    let (data, shape) = buffer_to_f32(array.py(), &buf)?;
    RTensor::from_vec(&shape, data)
        .map(|t| PyTensor::new(Var::leaf(t, requires_grad)))
        .map_err(to_pyerr)
}

/// Number of available devices (1: CPU).
#[pyfunction]
fn device_count() -> u32 {
    1
}

/// Whether ops currently record into the autograd graph on this thread.
#[pyfunction]
fn is_grad_enabled() -> bool {
    oxi_autograd::graph::is_grad_enabled()
}

/// Enables or disables graph recording for the *current scope* on this
/// thread. Prefer the `oxitorch.no_grad()` / `oxitorch.enable_grad()`
/// context managers, which restore the previous mode automatically.
#[pyfunction]
fn set_grad_enabled(enabled: bool) -> bool {
    oxi_autograd::graph::set_grad_enabled(enabled)
}

/// Optimizer plumbing shared by the Rust-backed optimizer wrappers.
///
/// The wrappers own *no tensor state*: the engine optimizer holds all
/// buffers, and each call hands over the current parameter tensors. A step
/// rewrites the `Var` inside each `PyTensor` in place — Python tensor
/// objects keep their identity across the whole training run and no values
/// ever round-trip through numpy.
mod optim_wrappers {
    use super::{to_pyerr, PyTensor};
    use oxi_autograd::optim as oopt;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::sync::Mutex;

    /// Steps every parameter, writing each updated `Var` back into the same
    /// `PyTensor` object.
    fn step_all<T>(opt: &T, params: Vec<PyRefMut<'_, PyTensor>>) -> PyResult<()>
    where
        T: Fn(usize, &mut oxi_autograd::graph::Var) -> oxi_core::OxitorchResult<()>,
    {
        for (index, mut p) in params.into_iter().enumerate() {
            let mut var = p.inner.clone();
            opt(index, &mut var).map_err(to_pyerr)?;
            p.inner = var;
        }
        Ok(())
    }

    /// Rust-backed SGD (velocity state lives in the engine; the engine
    /// optimizer is `Cell`/`RefCell`-based, so the pyclass holds it behind a
    /// `Mutex` to satisfy pyo3's `Sync` requirement).
    #[pyclass]
    pub struct RustSgd {
        opt: Mutex<oopt::Sgd>,
    }

    #[pymethods]
    impl RustSgd {
        #[new]
        #[pyo3(signature = (lr = 0.01, momentum = 0.0, nesterov = false))]
        fn new(lr: f64, momentum: f64, nesterov: bool) -> PyResult<Self> {
            if nesterov && momentum <= 0.0 {
                return Err(PyValueError::new_err(
                    "nesterov momentum requires momentum > 0",
                ));
            }
            Ok(Self {
                opt: Mutex::new(oopt::Sgd::new(lr as f32, momentum, nesterov)),
            })
        }
        #[getter]
        fn lr(&self) -> f64 {
            f64::from(self.opt.lock().expect("optimizer lock poisoned").lr())
        }

        #[setter]
        fn set_lr(&self, lr: f64) {
            self.opt
                .lock()
                .expect("optimizer lock poisoned")
                .set_lr(lr as f32);
        }

        /// Zeroes gradients (stateless: gradient storage lives on the tensors).
        #[staticmethod]
        fn zero_grad(params: Vec<PyRef<'_, PyTensor>>) -> PyResult<()> {
            for p in &params {
                p.inner.zero_grad().map_err(to_pyerr)?;
            }
            Ok(())
        }

        fn step(&self, params: Vec<PyRefMut<'_, PyTensor>>) -> PyResult<()> {
            let opt = self.opt.lock().expect("optimizer lock poisoned");
            step_all(&|i, v| opt.step_param(i, v), params)
        }
    }

    /// Rust-backed Adam/AdamW (moment + step-counter state in the engine;
    /// `decoupled` selects AdamW vs classic-Adam decay semantics).
    #[pyclass]
    pub struct RustAdamW {
        opt: Mutex<oopt::AdamW>,
    }

    #[pymethods]
    impl RustAdamW {
        #[new]
        #[pyo3(signature = (lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2, decoupled = true))]
        fn new(lr: f64, betas: (f64, f64), eps: f64, weight_decay: f64, decoupled: bool) -> Self {
            Self {
                opt: Mutex::new(oopt::AdamW::new(
                    lr as f32,
                    betas.0,
                    betas.1,
                    eps,
                    weight_decay,
                    decoupled,
                )),
            }
        }

        #[getter]
        fn lr(&self) -> f64 {
            f64::from(self.opt.lock().expect("optimizer lock poisoned").lr())
        }

        #[setter]
        fn set_lr(&self, lr: f64) {
            self.opt
                .lock()
                .expect("optimizer lock poisoned")
                .set_lr(lr as f32);
        }

        /// Zeroes gradients (stateless: gradient storage lives on the tensors).
        #[staticmethod]
        fn zero_grad(params: Vec<PyRef<'_, PyTensor>>) -> PyResult<()> {
            for p in &params {
                p.inner.zero_grad().map_err(to_pyerr)?;
            }
            Ok(())
        }

        fn step(&self, params: Vec<PyRefMut<'_, PyTensor>>) -> PyResult<()> {
            let opt = self.opt.lock().expect("optimizer lock poisoned");
            step_all(&|i, v| opt.step_param(i, v), params)
        }
    }
}

use optim_wrappers::{RustAdamW, RustSgd};

/// Native library version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The oxitorch native module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<PyTensor>()?;
    module.add_class::<RustSgd>()?;
    module.add_class::<RustAdamW>()?;
    // Registers the DataLoader/LoaderIter classes and load_mnist_dir into
    // this same module (the pymodule fn exists so wrap_pyfunction! resolves
    // inside the defining module).
    data::oxitorch_data(module)?;
    module.add_function(wrap_pyfunction!(zeros, module)?)?;
    module.add_function(wrap_pyfunction!(full, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(from_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(device_count, module)?)?;
    module.add_function(wrap_pyfunction!(is_grad_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(set_grad_enabled, module)?)?;
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
