//! Datasets, data loaders, and transforms (plan.md Phase 5).
//!
//! The distinguishing design decision lives here: batching workers are
//! **Rust-side threads**, not Python multiprocessing. The [`batch`] module's
//! [`batch::BatchChannel`] carries requests/responses between a GIL-free
//! dispatcher and worker threads; datasets implement [`BatchSource`] so
//! workers never need the Python GIL.
//!
//! Layout:
//! - [`rng::Mt19937`] — seeded MT19937 RNG (NumPy-compatible streams for the
//!   shuffle order, verified against `numpy.random.RandomState`).
//! - [`mnist::MnistData`] — raw IDX1/IDX3 file parsing (gzip or plain).
//! - [`batch`] — the worker/dispenser channel and batch-index algebra.
//! - [`transform`] — batch augmentation kernels run *inside* the workers
//!   (crop/flip/normalize over flat slices), seeded per batch.
//! - [`BatchSource`] — the trait worker threads consume.

pub mod batch;
pub mod mnist;
pub mod rng;
pub mod transform;

pub use batch::{
    epoch_plan, gather_columns, spawn_workers, BatchDispatcher, BatchRequest, BatchResponse,
    CompletedBatch, ResponseReorderer, TensorBatchSource,
};
pub use mnist::MnistData;
pub use rng::Mt19937;
pub use transform::{batch_rng, batch_seed, NativeTransform, TransformPipeline};

use oxi_core::OxitorchResult;
use oxi_tensor::Tensor;

/// A dataset that worker threads can read without the Python GIL.
///
/// `get_batch` gathers the rows named by `indices` into one batch of tensors
/// (e.g. `[inputs, labels]`). Implementations must be `Send + Sync`; treat
/// the data as immutable — the loader never mutates a source.
pub trait BatchSource: Send + Sync {
    /// Builds the batch of tensors for the given row indices.
    ///
    /// # Errors
    /// If any index is out of range or the source is internally inconsistent.
    fn get_batch(&self, indices: &[u64]) -> OxitorchResult<Vec<Tensor>>;
}
