//! Batching transport and epoch plans.
//!
//! [`spawn_workers`] starts a pool of worker threads over a [`BatchSource`]
//! and returns a [`BatchDispatcher`]. Workers pull batch requests, build the
//! batch tensors **on their own threads** (no GIL, no Python — this is the
//! "Rust-side workers" selling point from plan.md), and push responses; the
//! dispatcher's [`ResponseReorderer`] yields them in submission order so
//! iteration is deterministic regardless of worker completion order.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

use crate::transform::{batch_rng, batch_seed, TransformPipeline};
use crate::BatchSource;

/// A work request for a single batch.
#[derive(Debug)]
pub struct BatchRequest {
    /// Submission sequence within the stream (responses echo it back).
    pub seq: u64,
    /// The row indices of this batch.
    pub indices: Vec<u64>,
}

/// A worker's reply, tagged with the request's sequence.
#[derive(Debug)]
pub enum BatchResponse {
    /// A completed batch of tensors (sources build them off the GIL).
    Tensors {
        /// Echo of the request's `seq`.
        seq: u64,
        /// The batch tensors (one per dataset column).
        tensors: Vec<Tensor>,
    },
    /// The worker failed to build the batch (bad indices, source error...).
    Error {
        /// Echo of the request's `seq`.
        seq: u64,
        /// Human-readable failure reason.
        message: String,
    },
}

/// A completed batch of tensors, plus any error that must be raised in plan
/// order (a failure at batch k stops iteration there, not earlier).
#[derive(Debug)]
pub struct CompletedBatch {
    /// The tensors of the batch (one per dataset column).
    pub tensors: Vec<Tensor>,
    /// Set when the worker failed; the dispatcher raises it at this batch.
    pub error: Option<String>,
}

/// Reorders completed responses into submission order.
pub struct ResponseReorderer {
    next_seq: u64,
    ready: BTreeMap<u64, CompletedBatch>,
}

impl ResponseReorderer {
    /// A reorderer expecting batches starting at `seq` 0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_seq: 0,
            ready: BTreeMap::new(),
        }
    }

    /// Offers a worker response. Returns the batch if it fills the next
    /// expected slot (draining any now-adjacent buffered ones); `None` when
    /// it was buffered out of order.
    pub fn accept(&mut self, resp: BatchResponse) -> Option<CompletedBatch> {
        match resp {
            BatchResponse::Tensors { seq, tensors } => {
                self.ready.insert(
                    seq,
                    CompletedBatch {
                        tensors,
                        error: None,
                    },
                );
            }
            BatchResponse::Error { seq, message } => {
                self.ready.insert(
                    seq,
                    CompletedBatch {
                        tensors: Vec::new(),
                        error: Some(message),
                    },
                );
            }
        }
        self.try_pop()
    }

    /// Returns the next in-order batch if it has already arrived.
    pub fn try_pop(&mut self) -> Option<CompletedBatch> {
        let batch = self.ready.remove(&self.next_seq)?;
        self.next_seq += 1;
        Some(batch)
    }

    /// Number of completed-but-not-yet-consumed batches.
    pub fn buffered(&self) -> usize {
        self.ready.len()
    }
}

impl Default for ResponseReorderer {
    fn default() -> Self {
        Self::new()
    }
}

/// The work queue shared by the worker threads: a Mutex-protected request
/// deque plus condition-free signal handling via an mpsc fallback.
struct WorkQueue {
    inner: Mutex<QueueState>,
}

struct QueueState {
    pending: VecDeque<BatchRequest>,
    /// Set when the dispatcher shuts the queue down.
    closed: bool,
}

impl WorkQueue {
    /// Pops the next request; `None` only once the queue is closed *and*
    /// drained. An empty-but-open queue parks briefly (a 1 ms poll keeps
    /// workers responsive to shutdown without a full condvar dance).
    fn pop(&self) -> Option<BatchRequest> {
        loop {
            let mut st = self.inner.lock().unwrap();
            if let Some(req) = st.pending.pop_front() {
                return Some(req);
            }
            if st.closed {
                return None;
            }
            drop(st);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    fn push(&self, req: BatchRequest) {
        self.inner.lock().unwrap().pending.push_back(req);
    }

    fn close(&self) {
        self.inner.lock().unwrap().closed = true;
    }
}

/// The dispatcher end of a running worker pool: submit batch requests and
/// receive completed batches in submission order.
pub struct BatchDispatcher {
    queue: Arc<WorkQueue>,
    responses_rx: mpsc::Receiver<BatchResponse>,
    reorder: ResponseReorderer,
    workers: Vec<std::thread::JoinHandle<()>>,
    shutdown: bool,
}

impl BatchDispatcher {
    /// Submits one batch request (non-blocking; the work queue grows with
    /// the plan, workers drain it across the epoch).
    pub fn submit(&self, seq: u64, indices: Vec<u64>) -> bool {
        if self.shutdown {
            return false;
        }
        self.queue.push(BatchRequest { seq, indices });
        true
    }

    /// Blocks for the next in-order batch; `None` once the pool is shut
    /// down. Worker failures are carried as `CompletedBatch::error` (they
    /// surface at their plan position, not early).
    pub fn next_batch(&mut self) -> Option<CompletedBatch> {
        if let Some(b) = self.reorder.try_pop() {
            return Some(b);
        }
        loop {
            let resp = self.responses_rx.recv().ok()?;
            if let Some(b) = self.reorder.accept(resp) {
                return Some(b);
            }
        }
    }

    /// Number of completed batches buffered ahead of the consumer.
    pub fn buffered(&self) -> usize {
        self.reorder.buffered()
    }

    /// Shuts the pool down (closes the work queue so workers exit once
    /// drained) and joins the worker threads. Called on drop; safe to call
    /// manually.
    pub fn shutdown(&mut self) {
        if self.shutdown {
            return;
        }
        self.shutdown = true;
        self.queue.close();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for BatchDispatcher {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Spawns a worker pool over `source` and returns its dispatcher.
/// `workers` is the thread count (at least 1).
pub fn spawn_workers(source: &Arc<dyn BatchSource>, workers: usize) -> BatchDispatcher {
    let workers = workers.max(1);
    let queue = Arc::new(WorkQueue {
        inner: Mutex::new(QueueState {
            pending: VecDeque::new(),
            closed: false,
        }),
    });
    let (responses_tx, responses_rx) = mpsc::channel::<BatchResponse>();
    let handles = (0..workers)
        .map(|_| {
            let queue = Arc::clone(&queue);
            let source = Arc::clone(source);
            let responses_tx = responses_tx.clone();
            std::thread::Builder::new()
                .name("oxitorch-loader".into())
                .spawn(move || {
                    while let Some(req) = queue.pop() {
                        let resp = match source.get_batch(&req.indices) {
                            Ok(tensors) => BatchResponse::Tensors {
                                seq: req.seq,
                                tensors,
                            },
                            Err(e) => BatchResponse::Error {
                                seq: req.seq,
                                message: e.to_string(),
                            },
                        };
                        // A failed send means the dispatcher is gone; stop.
                        if responses_tx.send(resp).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawn loader worker")
        })
        .collect();
    BatchDispatcher {
        queue,
        responses_rx,
        reorder: ResponseReorderer::new(),
        workers: handles,
        shutdown: false,
    }
}

/// Per-epoch batch plan: the (possibly shuffled) row order split into
/// batches, honouring `drop_last`.
pub fn epoch_plan(
    order: &[u64],
    batch_size: usize,
    drop_last: bool,
) -> OxitorchResult<Vec<Vec<u64>>> {
    if batch_size == 0 {
        return Err(OxitorchError::InvalidArgument(
            "batch_size must be >= 1".into(),
        ));
    }
    let mut batches = Vec::new();
    let mut chunks = order.chunks(batch_size).peekable();
    while let Some(chunk) = chunks.next() {
        let last = chunks.peek().is_none();
        if last && drop_last && chunk.len() < batch_size && !batches.is_empty() {
            break;
        }
        batches.push(chunk.to_vec());
    }
    Ok(batches)
}

/// An in-memory, thread-safe dataset view over *column tensors* — the
/// standard tensor-dataset case: one tensor per dataset column, each with
/// `n` rows along dim 0 (e.g. images `(n, 1, 28, 28)` + labels `(n,)`).
///
/// `get_batch` gathers the requested rows *on the calling (worker) thread*
/// with no GIL involved: each column becomes `index_select(0, indices)`,
/// which keeps every trailing dimension (batches of images stay 4-D).
pub struct TensorBatchSource {
    columns: Vec<Tensor>,
    n: usize,
    /// Augmentations applied to column 0 (the image column) per batch,
    /// seeded from `(epoch_seed, batch indices)`.
    transforms: TransformPipeline,
    /// The epoch's seed (0 when unseeded); folds into per-batch transform
    /// seeds so identical plans across epochs still draw fresh crops.
    epoch_seed: u64,
}

impl TensorBatchSource {
    /// Builds the source from column tensors (all sharing the same row
    /// count along dim 0), applying `transforms` to the first (image)
    /// column inside `get_batch`, seeded by `(epoch_seed, batch indices)`.
    ///
    /// # Errors
    /// With no columns, mismatched row counts, or zero rows.
    pub fn with_transforms(
        columns: Vec<Tensor>,
        transforms: TransformPipeline,
        epoch_seed: u64,
    ) -> OxitorchResult<Self> {
        let source = Self::new(columns)?;
        Ok(Self {
            transforms,
            epoch_seed,
            ..source
        })
    }

    /// Builds the source from column tensors (all sharing the same row
    /// count along dim 0).
    ///
    /// # Errors
    /// With no columns, mismatched row counts, or zero rows.
    pub fn new(columns: Vec<Tensor>) -> OxitorchResult<Self> {
        if columns.is_empty() {
            return Err(OxitorchError::InvalidArgument(
                "a tensor source needs at least one column".into(),
            ));
        }
        let n = columns[0].shape()[0];
        if n == 0 {
            return Err(OxitorchError::InvalidArgument(
                "a tensor source needs at least one row".into(),
            ));
        }
        for (i, col) in columns.iter().enumerate() {
            if col.shape()[0] != n {
                return Err(OxitorchError::ShapeMismatch {
                    lhs: format!("column {i} has {} rows", col.shape()[0]),
                    rhs: format!("{n} rows"),
                });
            }
        }
        Ok(Self {
            columns,
            n,
            transforms: TransformPipeline::default(),
            epoch_seed: 0,
        })
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the source is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl BatchSource for TensorBatchSource {
    fn get_batch(&self, indices: &[u64]) -> OxitorchResult<Vec<Tensor>> {
        let mut cols = gather_columns(&self.columns, indices)?;
        if !self.transforms.is_empty() {
            let seed = batch_seed(self.epoch_seed, indices_seed(indices));
            let mut rng = batch_rng(seed);
            cols[0] = self.transforms.apply(&cols[0], &mut rng)?;
        }
        Ok(cols)
    }
}

/// Folds a batch's row indices into the per-batch transform seed: batches of
/// the same epoch use the (epoch, seq) pair, and inline callers without a
/// dispatcher derive their seed from the indices themselves so a given
/// `(plan, indices)` sequence is always reproducible.
#[must_use]
fn indices_seed(indices: &[u64]) -> u64 {
    // FNV-1a over the little-endian index bytes.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &i in indices {
        for b in i.to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Gathers rows from column tensors: `index_select(0, indices)` per column.
/// The shared kernel for worker batches and inline (single-thread) batches.
///
/// # Errors
/// If any index is out of range.
pub fn gather_columns(columns: &[Tensor], indices: &[u64]) -> OxitorchResult<Vec<Tensor>> {
    let n = columns.first().map_or(0, |c| c.shape()[0]) as u64;
    if indices.iter().any(|&i| i >= n) {
        return Err(OxitorchError::OutOfBounds(format!(
            "index out of range (n={n})"
        )));
    }
    let idx: Vec<i64> = indices.iter().map(|&i| i as i64).collect();
    columns
        .iter()
        .map(|col| col.index_select(0, &idx))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_sequential_and_remainder() {
        let order: Vec<u64> = (0..7).collect();
        let plan = epoch_plan(&order, 3, false).unwrap();
        assert_eq!(plan, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);

        let plan = epoch_plan(&order, 3, true).unwrap();
        assert_eq!(plan, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn plan_drop_last_single_batch_keeps_short_batch() {
        // A single short batch with drop_last still yields it (else the
        // loader would be infinitely empty).
        let plan = epoch_plan(&[9, 8], 4, true).unwrap();
        assert_eq!(plan, vec![vec![9, 8]]);
    }

    #[test]
    fn plan_rejects_zero_batch_size() {
        assert!(epoch_plan(&[1], 0, false).is_err());
    }

    #[test]
    fn source_gathers_rows_keeping_dims() {
        // 6 images of (1, 2, 2) + labels.
        let imgs = Tensor::from_vec(&[6, 1, 2, 2], (0..24).map(|v| v as f32).collect()).unwrap();
        let labels = Tensor::from_vec(&[6], (0..6).map(|v| v as f32).collect()).unwrap();
        let src = TensorBatchSource::new(vec![imgs, labels]).unwrap();
        let batch = src.get_batch(&[5, 0]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].shape(), &[2, 1, 2, 2]);
        assert_eq!(
            batch[0].to_vec(),
            vec![20.0, 21.0, 22.0, 23.0, 0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(batch[1].to_vec(), vec![5.0, 0.0]);
    }

    #[test]
    fn source_rejects_out_of_range() {
        let col = Tensor::from_vec(&[1, 2], vec![1.0, 2.0]).unwrap();
        let src = TensorBatchSource::new(vec![col]).unwrap();
        assert!(src.get_batch(&[1]).is_err());
    }

    #[test]
    fn source_rejects_mismatched_columns() {
        let a = Tensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
        let b = Tensor::from_vec(&[3], vec![0.0, 1.0, 2.0]).unwrap();
        assert!(TensorBatchSource::new(vec![a, b]).is_err());
    }

    #[test]
    fn reorder_yields_in_submission_order() {
        let mut r = ResponseReorderer::new();
        assert!(r.try_pop().is_none());
        // Out-of-order arrival: 1, then 0.
        let resp1 = BatchResponse::Tensors {
            seq: 1,
            tensors: vec![],
        };
        assert!(r.accept(resp1).is_none());
        assert_eq!(r.buffered(), 1);
        let resp0 = BatchResponse::Tensors {
            seq: 0,
            tensors: vec![],
        };
        let b = r
            .accept(resp0)
            .expect("seq 0 completes after seq 1 buffered");
        assert!(b.error.is_none());
        // seq 1 was buffered while waiting for seq 0; it is now at the head.
        let b1 = r.try_pop().expect("buffered seq 1 ready");
        assert!(b1.error.is_none());
        // Next slot is 2; truly nothing buffered now.
        assert!(r.try_pop().is_none());
    }

    #[test]
    fn reorder_carries_errors_at_their_position() {
        let mut r = ResponseReorderer::new();
        // accept returns the completed batch directly when it fills seq 0.
        let b = r
            .accept(BatchResponse::Error {
                seq: 0,
                message: "boom".into(),
            })
            .expect("seq 0 returned by accept");
        assert_eq!(b.error.as_deref(), Some("boom"));
        assert!(r.try_pop().is_none());
    }

    /// A slow, deliberately racy source: sleeps per row so workers finish
    /// out of order; the dispatcher must still yield batches in order.
    struct RacySource {
        base: TensorBatchSource,
    }

    impl BatchSource for RacySource {
        fn get_batch(&self, indices: &[u64]) -> OxitorchResult<Vec<Tensor>> {
            // Later batches sleep longer so early submissions finish last
            // sometimes; the reorderer must mask all of this.
            std::thread::sleep(std::time::Duration::from_millis(2 + (indices[0] % 4)));
            self.base.get_batch(indices)
        }
    }

    #[test]
    fn dispatcher_returns_batches_in_plan_order() {
        let feats = Tensor::from_vec(&[10, 4], (0..40).map(|v| v as f32).collect()).unwrap();
        let labels = Tensor::from_vec(&[10], (0..10).map(|v| v as f32).collect()).unwrap();
        let base = TensorBatchSource::new(vec![feats, labels]).unwrap();
        let src: Arc<dyn BatchSource> = Arc::new(RacySource { base });

        let mut d = spawn_workers(&src, 4);
        let order: Vec<u64> = (0..10).collect();
        let plan = epoch_plan(&order, 3, false).unwrap();
        for (seq, batch) in plan.iter().enumerate() {
            assert!(d.submit(seq as u64, batch.clone()));
        }
        let mut got = Vec::new();
        for _ in 0..plan.len() {
            let b = d.next_batch().expect("batch per plan entry");
            assert!(b.error.is_none());
            got.push(b.tensors[1].to_vec()); // labels column
        }
        // In-order labels: 0-2, 3-5, 6-8, 9.
        assert_eq!(
            got,
            vec![
                vec![0.0, 1.0, 2.0],
                vec![3.0, 4.0, 5.0],
                vec![6.0, 7.0, 8.0],
                vec![9.0]
            ]
        );
        d.shutdown();
    }

    #[test]
    fn dispatcher_survives_worker_errors() {
        let feats = Tensor::from_vec(&[2, 4], (0..8).map(|v| v as f32).collect()).unwrap();
        let labels = Tensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
        let src: Arc<dyn BatchSource> =
            Arc::new(TensorBatchSource::new(vec![feats, labels]).unwrap());
        let mut d = spawn_workers(&src, 2);
        assert!(d.submit(0, vec![0]));
        assert!(d.submit(1, vec![9])); // out of range -> worker error
        let b0 = d.next_batch().unwrap();
        assert!(b0.error.is_none());
        let b1 = d.next_batch().unwrap();
        assert!(b1.error.is_some());
        d.shutdown();
    }

    #[test]
    fn shutdown_joins_workers() {
        let feats = Tensor::from_vec(&[2, 4], (0..8).map(|v| v as f32).collect()).unwrap();
        let labels = Tensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
        let src: Arc<dyn BatchSource> =
            Arc::new(TensorBatchSource::new(vec![feats, labels]).unwrap());
        let mut d = spawn_workers(&src, 3);
        assert!(d.submit(0, vec![0]));
        d.shutdown(); // joins even with queued work outstanding
        assert!(!d.submit(1, vec![1])); // after shutdown, submit refuses
    }
}
