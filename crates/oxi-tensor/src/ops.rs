//! Elementwise kernels and the op registry (plan.md Phase 1: "Dispatch table
//! keyed on (op × dtype × device); fast paths for contiguous tensors").
//!
//! Phase 1 scope: f32 only. The dtype axis of the dispatch table becomes
//! load-bearing when generic dtypes land.

use rayon::prelude::*;

use oxi_core::{DType, Device, OxitorchError, OxitorchResult};

/// Element counts at or above this run on the rayon pool.
pub const PARALLEL_THRESHOLD_ELEMENTS: usize = 4096;

/// `m * k * n` work estimates at or above this use parallel row-block GEMM.
/// Below it, thread-fanout overhead exceeds the SGEMM savings (measured on
/// the Phase 1 bench: 64³/128³ are ~2-6x SLOWER parallel; 256³+ ~2x faster).
pub const PARALLEL_GEMM_THRESHOLD_ELEMENTS: usize = 8_388_608; // 2^23: between 128³ and 256³

/// A binary elementwise op: `(a, b) -> a op b`.
pub type BinaryFn<T> = fn(&[T]) -> T;

/// A comparison elementwise op: `(a, b) -> bool`.
pub type CmpFn<T> = fn(&[T]) -> bool;

/// A unary elementwise op: `a -> f(a)`.
pub type UnaryFn<T> = fn(T) -> T;

/// Registry entry for a binary op.
pub struct BinaryOpSpec<T> {
    /// Registry name (e.g. `"add"`).
    pub name: &'static str,
    /// The kernel.
    pub f: BinaryFn<T>,
    /// Reduction identity (used by future op-fusion; documented per op).
    pub identity: T,
}

/// Registry entry for a comparison op.
pub struct CmpOpSpec<T> {
    /// Registry name (e.g. `"eq"`).
    pub name: &'static str,
    /// The kernel.
    pub f: CmpFn<T>,
}

/// Registry entry for a unary op.
pub struct UnaryOpSpec<T> {
    /// Registry name (e.g. `"sqrt"`).
    pub name: &'static str,
    /// The kernel.
    pub f: UnaryFn<T>,
}

/// Binary ops for f32.
pub const F32_BINARY_OPS: &[BinaryOpSpec<f32>] = &[
    BinaryOpSpec {
        name: "add",
        f: |v| v[0] + v[1],
        identity: 0.0,
    },
    BinaryOpSpec {
        name: "sub",
        f: |v| v[0] - v[1],
        identity: 0.0,
    },
    BinaryOpSpec {
        name: "mul",
        f: |v| v[0] * v[1],
        identity: 1.0,
    },
    BinaryOpSpec {
        name: "div",
        f: |v| v[0] / v[1],
        identity: 1.0,
    },
    BinaryOpSpec {
        name: "pow",
        f: |v| v[0].powf(v[1]),
        identity: 1.0,
    },
    BinaryOpSpec {
        name: "min",
        f: |v| v[0].min(v[1]),
        identity: f32::INFINITY,
    },
    BinaryOpSpec {
        name: "max",
        f: |v| v[0].max(v[1]),
        identity: f32::NEG_INFINITY,
    },
];

/// Comparison ops for f32 (produce bool tensors).
// Exact bitwise equality IS the semantics of `eq`/`ne` — that's the point.
#[allow(clippy::float_cmp)]
pub const F32_CMP_OPS: &[CmpOpSpec<f32>] = &[
    CmpOpSpec {
        name: "eq",
        f: |v| v[0] == v[1],
    },
    CmpOpSpec {
        name: "ne",
        f: |v| v[0] != v[1],
    },
    CmpOpSpec {
        name: "lt",
        f: |v| v[0] < v[1],
    },
    CmpOpSpec {
        name: "le",
        f: |v| v[0] <= v[1],
    },
    CmpOpSpec {
        name: "gt",
        f: |v| v[0] > v[1],
    },
    CmpOpSpec {
        name: "ge",
        f: |v| v[0] >= v[1],
    },
];

/// Unary ops for f32.
#[allow(clippy::missing_docs_in_private_items)] // names are the documentation
pub const F32_UNARY_OPS: &[UnaryOpSpec<f32>] = &[
    UnaryOpSpec {
        name: "neg",
        f: |a| -a,
    },
    UnaryOpSpec {
        name: "abs",
        f: f32::abs,
    },
    UnaryOpSpec {
        name: "sqrt",
        f: f32::sqrt,
    },
    UnaryOpSpec {
        name: "rsqrt",
        f: |a| 1.0 / a.sqrt(),
    },
    UnaryOpSpec {
        name: "exp",
        f: f32::exp,
    },
    UnaryOpSpec {
        name: "log",
        f: f32::ln,
    },
    UnaryOpSpec {
        name: "log2",
        f: f32::log2,
    },
    UnaryOpSpec {
        name: "log10",
        f: f32::log10,
    },
    UnaryOpSpec {
        name: "sin",
        f: f32::sin,
    },
    UnaryOpSpec {
        name: "cos",
        f: f32::cos,
    },
    UnaryOpSpec {
        name: "tanh",
        f: f32::tanh,
    },
    UnaryOpSpec {
        name: "relu",
        f: |a| a.max(0.0),
    },
    UnaryOpSpec {
        name: "sigmoid",
        f: |a| 1.0 / (1.0 + (-a).exp()),
    },
    UnaryOpSpec {
        name: "floor",
        f: f32::floor,
    },
    UnaryOpSpec {
        name: "ceil",
        f: f32::ceil,
    },
    UnaryOpSpec {
        name: "round",
        f: f32::round,
    },
    UnaryOpSpec {
        name: "sign",
        f: f32::signum,
    },
    UnaryOpSpec {
        name: "square",
        f: |a| a * a,
    },
    UnaryOpSpec {
        name: "reciprocal",
        f: |a| 1.0 / a,
    },
];

/// Looks up a binary op by name.
#[must_use]
pub fn find_binary_op(name: &str) -> Option<&'static BinaryOpSpec<f32>> {
    F32_BINARY_OPS.iter().find(|spec| spec.name == name)
}

/// Looks up a comparison op by name.
#[must_use]
pub fn find_cmp_op(name: &str) -> Option<&'static CmpOpSpec<f32>> {
    F32_CMP_OPS.iter().find(|spec| spec.name == name)
}

/// Looks up a unary op by name.
#[must_use]
pub fn find_unary_op(name: &str) -> Option<&'static UnaryOpSpec<f32>> {
    F32_UNARY_OPS.iter().find(|spec| spec.name == name)
}

/// Reduction kinds supported by [`crate::Tensor`] methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    /// Sum of elements.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
    /// Index of the maximum element (i64 output).
    ArgMax,
    /// Index of the minimum element (i64 output).
    ArgMin,
    /// Euclidean (L2) norm.
    Norm,
}

impl ReduceKind {
    /// Reduces one contiguous strip `data` to its scalar result.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] for min/max/arg* over empty data.
    pub fn reduce_strip(self, data: &[f32]) -> OxitorchResult<f64> {
        if data.is_empty() {
            return match self {
                Self::Sum | Self::Mean | Self::Norm => Ok(0.0),
                _ => Err(OxitorchError::InvalidArgument(
                    "cannot compute min/max/argmax/argmin over an empty reduction".into(),
                )),
            };
        }
        Ok(match self {
            Self::Sum => f64::from(data.iter().sum::<f32>()),
            Self::Mean => f64::from(data.iter().sum::<f32>() / data.len() as f32),
            Self::Max => f64::from(data.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
            Self::Min => f64::from(data.iter().copied().fold(f32::INFINITY, f32::min)),
            // torch semantics: ties resolve to the FIRST occurrence.
            Self::ArgMax => {
                let mut best = 0usize;
                let mut best_v = data[0];
                for (i, &v) in data.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i;
                    }
                }
                best as f64
            }
            Self::ArgMin => {
                let mut best = 0usize;
                let mut best_v = data[0];
                for (i, &v) in data.iter().enumerate() {
                    if v < best_v {
                        best_v = v;
                        best = i;
                    }
                }
                best as f64
            }
            Self::Norm => f64::from(data.iter().map(|v| v * v).sum::<f32>().sqrt()),
        })
    }

    /// Whether the output dtype is i64 (argmax/argmin) instead of f32.
    #[must_use]
    pub const fn is_index_output(self) -> bool {
        matches!(self, Self::ArgMax | Self::ArgMin)
    }
}

/// The device-backend seam (plan.md Phase 1: "Device abstraction trait
/// (`DeviceBackend`) — CPU first, but the trait defined so GPU backends slot
/// in later").
///
/// Phase 1 keeps this deliberately minimal: identity + a couple of
/// capability probes. Phase 7 grows it into the real kernel dispatch vtable
/// once wgpu/cuda prove what needs to be virtual.
pub trait DeviceBackend: std::fmt::Debug + Send + Sync {
    /// Backend name (e.g. `"cpu"`).
    fn name(&self) -> &'static str;
    /// The device this backend serves.
    fn device(&self) -> Device;
    /// Whether this backend stores `dtype` natively.
    fn supports_dtype(&self, dtype: DType) -> bool;
}

/// The CPU backend: the only Phase 1 implementation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl DeviceBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device(&self) -> Device {
        Device::cpu()
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        // f16/bf16 are storage dtypes; everything else is native.
        !matches!(dtype, DType::F16 | DType::Bf16)
    }
}

/// Runs `f` over `0..len`, parallelized with rayon when the workload is
/// large enough to amortize the pool handshake.
pub fn maybe_parallel<R>(len: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R>
where
    R: Send,
{
    const PARALLEL_THRESHOLD: usize = 4096;
    if len >= PARALLEL_THRESHOLD {
        (0..len).into_par_iter().map(&f).collect()
    } else {
        (0..len).map(f).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookups() {
        assert_eq!(find_binary_op("add").unwrap().name, "add");
        assert!(find_binary_op("frobnicate").is_none());
        assert_eq!(find_cmp_op("lt").unwrap().name, "lt");
        assert_eq!(find_unary_op("sqrt").unwrap().name, "sqrt");
    }

    #[test]
    #[allow(clippy::float_cmp)] // test values are exactly representable
    fn registry_kernels_smoke() {
        assert_eq!((find_binary_op("pow").unwrap().f)(&[2.0, 10.0]), 1024.0);
        assert!((find_cmp_op("ge").unwrap().f)(&[2.0, 2.0]));
        assert_eq!((find_unary_op("relu").unwrap().f)(-3.0), 0.0);
        assert_eq!((find_unary_op("sigmoid").unwrap().f)(0.0), 0.5);
    }

    #[test]
    #[allow(clippy::float_cmp)] // test values are exactly representable
    fn reduce_strips() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(ReduceKind::Sum.reduce_strip(&data).unwrap(), 10.0);
        assert_eq!(ReduceKind::Mean.reduce_strip(&data).unwrap(), 2.5);
        assert_eq!(ReduceKind::Max.reduce_strip(&data).unwrap(), 4.0);
        assert_eq!(ReduceKind::Min.reduce_strip(&data).unwrap(), 1.0);
        assert_eq!(ReduceKind::ArgMax.reduce_strip(&data).unwrap(), 3.0);
        assert_eq!(ReduceKind::ArgMin.reduce_strip(&data).unwrap(), 0.0);
        // 3-4-5 triangle: norm = 5.
        assert_eq!(ReduceKind::Norm.reduce_strip(&[3.0, 4.0]).unwrap(), 5.0);
        assert_eq!(ReduceKind::Sum.reduce_strip(&[]).unwrap(), 0.0);
        assert!(ReduceKind::Max.reduce_strip(&[]).is_err());
    }

    #[test]
    fn cpu_backend_capabilities() {
        let backend = CpuBackend;
        assert_eq!(backend.name(), "cpu");
        assert_eq!(backend.device(), Device::cpu());
        assert!(backend.supports_dtype(DType::F32));
        assert!(!backend.supports_dtype(DType::F16));
    }
}
