//! Python-facing data pipeline: a native `DataLoader` over Rust worker
//! threads and an MNIST loader.
//!
//! The `DataLoader` pyclass owns *tensor columns* (the `TensorDataset`
//! case), so batches are built entirely off the GIL on the worker pool —
//! the dispatcher is spawned per `__iter__` call and joined when the
//! iterator is exhausted or dropped. `NativeTransform` marks the Python
//! transform composition as runnable natively, so augmented loading also
//! happens inside the workers. `load_mnist_dir` parses IDX files in pure
//! Rust (gzip or plain).

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use oxi_autograd::graph::Var;
use oxi_data::{
    epoch_plan, spawn_workers, NativeTransform as NTransform, TensorBatchSource, TransformPipeline,
};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use crate::error::to_pyerr;
use crate::PyTensor;

/// Turns engine tensors into Python tensors; all loader batches are
/// constants (data never requires grad).
fn wrap_columns(cols: Vec<oxi_tensor::Tensor>) -> Vec<PyTensor> {
    cols.into_iter()
        .map(|t| PyTensor::new(Var::leaf(t, false)))
        .collect()
}

/// Seed sequence for shuffled epochs: each `__iter__` call derives a fresh
/// seed from the shared epoch counter, so different epochs see different
/// shuffle orders while staying fully reproducible for a given construction.
struct EpochSeeds {
    base: u64,
    counter: AtomicU64,
}

impl EpochSeeds {
    fn next(&self) -> u64 {
        let epoch = self.counter.fetch_add(1, Ordering::Relaxed);
        // SplitMix64 finalizer mixes the base seed and epoch counter into a
        // well-distributed per-epoch seed.
        let mut z = self
            .base
            .wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A batch-level transform the Python layer can request by name.
///
/// Python's `RandomCrop`/`RandomHorizontalFlip`/`Normalize` classes expose a
/// `native_repr()` — a list of `(name, args)` tuples — which this pyclass
/// turns into engine kernels. An empty pipeline means no augmentation; the
/// loader then skips the transform machinery entirely.
///
/// Wire format for `steps`: `("crop", [size])`, `("flip", [])`, and
/// `("normalize", [mean..., std...])` — the normalize vector packs the
/// per-channel means followed by the stds (always equal lengths, so the
/// split point is unambiguous).
#[pyclass]
#[derive(Clone)]
pub struct NativeTransform {
    steps: Vec<NTransform>,
}

#[pymethods]
impl NativeTransform {
    #[new]
    fn new(steps: Vec<(String, Vec<f64>)>) -> PyResult<Self> {
        let steps = steps
            .into_iter()
            .map(|(name, args)| match (name.as_str(), args.as_slice()) {
                ("crop", [size]) if *size >= 0.0 => Ok(NTransform::RandomCrop(*size as usize)),
                ("crop", _) => Err(PyValueError::new_err(
                    "crop expects a single non-negative size argument",
                )),
                ("flip", []) => Ok(NTransform::RandomHorizontalFlip),
                ("flip", _) => Err(PyValueError::new_err("flip expects no arguments")),
                ("normalize", packed) if packed.len() >= 2 && packed.len() % 2 == 0 => {
                    let mid = packed.len() / 2;
                    let (mean, std): (Vec<f32>, Vec<f32>) = (
                        packed[..mid].iter().map(|&v| v as f32).collect(),
                        packed[mid..].iter().map(|&v| v as f32).collect(),
                    );
                    Ok(NTransform::Normalize(mean, std))
                }
                ("normalize", _) => Err(PyValueError::new_err(
                    "normalize expects mean... followed by std... of equal length",
                )),
                _ => Err(PyValueError::new_err(format!(
                    "unknown native transform {name:?}"
                ))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self { steps })
    }
}

impl NativeTransform {
    /// The engine pipeline (empty = no augmentation).
    fn pipeline(&self) -> TransformPipeline {
        TransformPipeline::new(self.steps.clone())
    }
}

/// A native DataLoader over tensor columns, batched by Rust worker threads.
#[pyclass]
pub struct DataLoader {
    columns: Vec<oxi_tensor::Tensor>,
    n: usize,
    batch_size: usize,
    drop_last: bool,
    shuffle: bool,
    seeds: Option<EpochSeeds>,
    num_workers: usize,
    transforms: Option<TransformPipeline>,
    /// Per-epoch transform seed generator (base = `transform_seed`).
    tseeds: EpochSeeds,
}

#[pymethods]
impl DataLoader {
    /// `DataLoader(columns, batch_size=1, shuffle=False, drop_last=False,
    /// seed=None, num_workers=0, transforms=None, transform_seed=0)` —
    /// `num_workers` is the Rust thread count (0 means batches are built on
    /// the calling thread); `transforms` is a `NativeTransform` applied to
    /// the first column inside the workers (or inline path), seeded
    /// reproducibly from `transform_seed` per epoch.
    #[new]
    #[allow(clippy::too_many_arguments)] // torch-parity constructor surface
    #[pyo3(signature = (columns, batch_size = 1, shuffle = false, drop_last = false, seed = None, num_workers = 0, transforms = None, transform_seed = 0))]
    fn new(
        columns: Vec<PyTensor>,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        seed: Option<u64>,
        num_workers: usize,
        transforms: Option<NativeTransform>,
        transform_seed: u64,
    ) -> PyResult<Self> {
        if columns.is_empty() {
            return Err(PyValueError::new_err(
                "DataLoader needs at least one tensor column",
            ));
        }
        let cols: Vec<oxi_tensor::Tensor> = columns
            .into_iter()
            .map(|p| p.inner.value().clone())
            .collect();
        let n = cols[0].shape()[0];
        if n == 0 {
            return Err(PyValueError::new_err("DataLoader dataset is empty"));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        for (i, c) in cols.iter().enumerate() {
            if c.shape()[0] != n {
                return Err(PyValueError::new_err(format!(
                    "column {i} has {} rows, expected {n}",
                    c.shape()[0]
                )));
            }
        }
        let seeds = seed.map(|base| EpochSeeds {
            base,
            counter: AtomicU64::new(0),
        });
        let pipeline = transforms.map(|t| t.pipeline());
        if let Some(p) = &pipeline {
            if p.is_empty() {
                return Err(PyValueError::new_err(
                    "transforms pipeline is empty; pass None instead",
                ));
            }
        }
        Ok(Self {
            columns: cols,
            n,
            batch_size,
            drop_last,
            shuffle,
            seeds,
            num_workers,
            transforms: pipeline,
            tseeds: EpochSeeds {
                base: transform_seed,
                counter: AtomicU64::new(0),
            },
        })
    }

    /// Number of samples.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    /// Samples per batch.
    #[getter]
    fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// `__iter__` — a fresh iterator over batched columns. Each epoch gets
    /// its own worker pool and (when seeded) its own shuffle order.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<LoaderIter>> {
        let order: Vec<u64> = if slf.shuffle {
            match &slf.seeds {
                Some(seeds) => {
                    let mut rng = oxi_data::Mt19937::new(seeds.next() as u32);
                    rng.permutation(slf.n)
                }
                None => (0..slf.n as u64).collect(),
            }
        } else {
            (0..slf.n as u64).collect()
        };
        let plan = epoch_plan(&order, slf.batch_size, slf.drop_last).map_err(to_pyerr)?;
        // One source per epoch: it bakes in this epoch's transform seed, so
        // inline and worker paths draw identical augmentation streams (both
        // derive each batch's seed from (epoch_seed, batch indices)).
        let tseed = slf.tseeds.next();
        let source: Arc<dyn oxi_data::BatchSource> = match &slf.transforms {
            Some(pipeline) => Arc::new(
                TensorBatchSource::with_transforms(slf.columns.clone(), pipeline.clone(), tseed)
                    .map_err(to_pyerr)?,
            ),
            None => Arc::new(TensorBatchSource::new(slf.columns.clone()).map_err(to_pyerr)?),
        };
        let dispatcher = if slf.num_workers == 0 {
            None
        } else {
            let disp = spawn_workers(&source, slf.num_workers);
            Some(Mutex::new(disp))
        };
        let window = if slf.num_workers == 0 {
            0
        } else {
            slf.num_workers + 2
        };
        let mut iter = LoaderIter {
            source,
            plan: plan.into_iter().map(Vec::into_boxed_slice).collect(),
            next_seq: 0,
            next_submit: 0,
            outstanding: 0,
            window,
            dispatcher,
        };
        iter.top_up();
        Py::new(slf.py(), iter)
    }
}

/// One pass over the loader: pulls batches from the worker pool (or builds
/// them inline when `num_workers == 0`) in deterministic plan order.
///
/// The worker path keeps a **prefetch window** of `num_workers + 2`
/// submitted-but-unconsumed batches so workers compute *while* the Python
/// side trains on the previous batch; memory stays bounded by the window.
#[pyclass]
pub struct LoaderIter {
    /// The epoch's batch source: gathers rows and applies transforms with
    /// this epoch's seed (shared verbatim by the worker threads).
    source: Arc<dyn oxi_data::BatchSource>,
    plan: Vec<Box<[u64]>>,
    /// Cursor for the inline (single-thread) path.
    next_seq: usize,
    /// Next plan slot to submit on the worker path.
    next_submit: usize,
    /// Submitted-but-not-yet-consumed batches (worker path).
    outstanding: usize,
    window: usize,
    dispatcher: Option<Mutex<oxi_data::BatchDispatcher>>,
}

impl LoaderIter {
    /// Submits more plan slots up to the prefetch window.
    fn top_up(&mut self) {
        let Some(disp) = &self.dispatcher else {
            return;
        };
        while self.outstanding < self.window && self.next_submit < self.plan.len() {
            let indices = self.plan[self.next_submit].clone();
            let seq = self.next_submit as u64;
            self.next_submit += 1;
            self.outstanding += 1;
            disp.lock().unwrap().submit(seq, indices.into_vec());
        }
    }
}

#[pymethods]
impl LoaderIter {
    /// Iterators return themselves (the Python iteration protocol).
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next batch as a list of column tensors.
    ///
    /// Raises `StopIteration` when the epoch is exhausted (Python iteration
    /// protocol); worker failures surface as `RuntimeError` at their plan
    /// position and end the epoch.
    fn __next__(&mut self) -> PyResult<Vec<PyTensor>> {
        let Some(disp) = &self.dispatcher else {
            // Inline path: build the batch on the calling thread via the
            // same source the workers would use (gather + transforms).
            if self.next_seq >= self.plan.len() {
                return Err(PyStopIteration::new_err(()));
            }
            let indices = self.plan[self.next_seq].clone();
            self.next_seq += 1;
            return match self.source.get_batch(&indices) {
                Ok(cols) => Ok(wrap_columns(cols)),
                Err(e) => Err(PyRuntimeError::new_err(format!(
                    "DataLoader failed at batch {}: {e}",
                    self.next_seq - 1
                ))),
            };
        };
        if self.outstanding == 0 {
            return Err(PyStopIteration::new_err(()));
        }
        let batch = {
            let mut disp = disp.lock().unwrap();
            match disp.next_batch() {
                Some(b) => b,
                None => {
                    return Err(PyRuntimeError::new_err(
                        "DataLoader worker pool shut down mid-epoch",
                    ))
                }
            }
        };
        self.outstanding -= 1;
        if let Some(msg) = batch.error {
            self.outstanding = 0; // end the epoch after a worker failure
            return Err(PyRuntimeError::new_err(format!(
                "DataLoader worker failed: {msg}"
            )));
        }
        self.top_up();
        Ok(wrap_columns(batch.tensors))
    }
}

/// Loads an MNIST split from a directory containing IDX files (gzip or
/// plain) and returns `(images, labels)` tensors — images `(n, 1, 28, 28)`
/// normalized to `[0, 1]`, labels `(n,)` class indices.
///
/// # Errors
/// Rises `OxitorchError`s mapped to Python exceptions for missing files,
/// bad magic, truncation, or length mismatches.
#[pyfunction]
#[pyo3(signature = (root, train))]
fn load_mnist_dir(py: Python<'_>, root: PathBuf, train: bool) -> PyResult<(PyTensor, PyTensor)> {
    let data = py
        .detach(move || oxi_data::MnistData::load(&root, train))
        .map_err(to_pyerr)?;
    Ok((
        PyTensor::new(Var::leaf(data.images, false)),
        PyTensor::new(Var::leaf(data.labels, false)),
    ))
}

/// Composite submodule so `wrap_pyfunction!` resolves inside its defining
/// module; the parent copies the function object up for flat access.
#[pymodule]
pub fn oxitorch_data(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<DataLoader>()?;
    module.add_class::<LoaderIter>()?;
    module.add_class::<NativeTransform>()?;
    module.add_function(wrap_pyfunction!(load_mnist_dir, module)?)?;
    Ok(())
}
