//! Shared types for the oxitorch tensor engine.
//!
//! This crate deliberately contains no kernel or autograd logic: it exists so
//! sibling crates (`oxi-tensor`, `oxi-autograd`, `oxi-nn`, ...) can depend on
//! the same [`DType`], [`DeviceType`], and error types without circular
//! dependencies.
//!
//! Error-handling convention: Rust code returns `Result<_, OxitorchError>`;
//! the `oxi-bindings` crate maps error variants onto Python exceptions at the
//! binding layer (`RuntimeError`, `ValueError`, ...). See
//! `crates/oxi-bindings/src/error.rs`.

/// Element types supported by oxitorch tensors.
///
/// Mirrors the Phase 1 plan: `bf16`/`f16` are storage dtypes first; compute is
/// promoted to `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8,
    /// Signed 8-bit integer.
    I8,
    /// Signed 16-bit integer.
    I16,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// IEEE 754 half-precision float (storage; compute promotes to f32).
    F16,
    /// IEEE 754 single-precision float (the default compute dtype).
    F32,
    /// IEEE 754 double-precision float.
    F64,
    /// Brain floating point (storage; compute promotes to f32).
    Bf16,
    /// Boolean (stored as one byte, values 0 or 1).
    Bool,
}

impl DType {
    /// Size of one element of this dtype in bytes.
    #[must_use]
    pub const fn item_size(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::I16 | Self::F16 => 2,
            Self::I32 | Self::F32 | Self::Bf16 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Whether this dtype represents floating-point values.
    #[must_use]
    pub const fn is_floating_point(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::F64 | Self::Bf16)
    }

    /// Whether this dtype represents signed or unsigned integers.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// The dtype used for index tensors (mirrors `torch.int64`).
    #[must_use]
    pub const fn index_dtype() -> Self {
        Self::I64
    }

    /// The dtype `oxitorch.empty(...)`/`oxitorch.zeros(...)` default to
    /// (mirrors `torch.get_default_dtype()`, pinned to f32 for now).
    #[must_use]
    pub const fn default_dtype() -> Self {
        Self::F32
    }

    /// Short name, matching the `torch.*` naming (`float32`, `int64`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Bf16 => "bfloat16",
            Self::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Device families a tensor can live on.
///
/// Phase 0 only implements [`DeviceType::Cpu`]; the enum is defined now so
/// GPU backends (wgpu, cuda) slot in later without an API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DeviceType {
    /// Host CPU (the only implemented device in Phase 0).
    Cpu,
    /// wgpu-backed GPU (Vulkan/Metal/DX12/WebGPU) — Phase 7.
    Wgpu,
    /// NVIDIA CUDA GPU — Phase 7.
    Cuda,
}

impl DeviceType {
    /// Short name matching `torch.device` (`"cpu"`, `"cuda"`, ...).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A specific device, e.g. `cpu` or `cuda:0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    /// Device family.
    pub device_type: DeviceType,
    /// Device index (always 0 for CPU in Phase 0).
    pub index: usize,
}

impl Device {
    /// The default device: `cpu:0`.
    #[must_use]
    pub const fn cpu() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// Whether this device is the host CPU.
    #[must_use]
    pub const fn is_cpu(self) -> bool {
        matches!(self.device_type, DeviceType::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_cpu() {
            f.write_str("cpu")
        } else {
            write!(f, "{}:{}", self.device_type.name(), self.index)
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu()
    }
}

/// The error type for all oxitorch crates.
///
/// Convention: Rust-side errors are typed variants; the Python binding layer
/// maps them onto exceptions (`ValueError` for [`OxitorchError::ShapeMismatch`]
/// and [`OxitorchError::InvalidArgument`], `RuntimeError` otherwise).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum OxitorchError {
    /// Tensor shapes do not agree (e.g. for a binary op).
    #[error("shape mismatch: {lhs} vs {rhs}")]
    ShapeMismatch {
        /// Shape of the left-hand operand.
        lhs: String,
        /// Shape of the right-hand operand.
        rhs: String,
    },
    /// An argument was structurally wrong (bad dtype name, negative dim...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// The feature is planned but not implemented yet (e.g. GPU devices).
    #[error("not implemented: {0}")]
    NotImplemented(String),
    /// The op would access memory outside a tensor's storage.
    #[error("out of bounds: {0}")]
    OutOfBounds(String),
    /// Anything else that does not fit a specific variant.
    #[error("{0}")]
    Other(String),
}

/// Convenient `Result` alias used across all oxitorch crates.
pub type OxitorchResult<T> = Result<T, OxitorchError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_sizes_match_spec() {
        assert_eq!(DType::U8.item_size(), 1);
        assert_eq!(DType::Bool.item_size(), 1);
        assert_eq!(DType::I16.item_size(), 2);
        assert_eq!(DType::F16.item_size(), 2);
        assert_eq!(DType::I32.item_size(), 4);
        assert_eq!(DType::F32.item_size(), 4);
        assert_eq!(DType::Bf16.item_size(), 4);
        assert_eq!(DType::I64.item_size(), 8);
        assert_eq!(DType::F64.item_size(), 8);
    }

    #[test]
    fn dtype_kinds_and_names() {
        assert!(DType::F32.is_floating_point());
        assert!(DType::Bf16.is_floating_point());
        assert!(!DType::I64.is_floating_point());
        assert!(DType::I8.is_integer());
        assert!(!DType::Bool.is_integer());
        assert_eq!(DType::Bf16.name(), "bfloat16");
        assert_eq!(DType::default_dtype(), DType::F32);
        assert_eq!(DType::index_dtype(), DType::I64);
    }

    #[test]
    fn device_display_matches_torch_style() {
        assert_eq!(Device::cpu().to_string(), "cpu");
        let cuda = Device {
            device_type: DeviceType::Cuda,
            index: 1,
        };
        assert_eq!(cuda.to_string(), "cuda:1");
        assert!(Device::cpu().is_cpu());
        assert_eq!(Device::default(), Device::cpu());
    }

    #[test]
    fn errors_display_readably() {
        let err = OxitorchError::ShapeMismatch {
            lhs: "[2, 3]".into(),
            rhs: "[4, 5]".into(),
        };
        assert_eq!(err.to_string(), "shape mismatch: [2, 3] vs [4, 5]");
        assert_eq!(
            OxitorchError::NotImplemented("cuda backend".into()).to_string(),
            "not implemented: cuda backend"
        );
    }
}
