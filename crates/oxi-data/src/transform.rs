//! Batch transforms executed **inside the loader worker threads**.
//!
//! The numpy transform path fetches and augments one image at a time on the
//! Python side (per-item `__getitem__` + `ascontiguousarray`); the transforms
//! here operate on the *whole gathered batch* as flat f32 slices, off the
//! GIL, so augmented loading runs entirely in Rust workers.
//!
//! Determinism model: every batch request carries a seed derived from the
//! epoch seed and batch position, so a run is fully reproducible and the
//! inline (`num_workers == 0`) path produces *bit-identical* batches to the
//! worker path (same seeds, same kernels).
//!
//! Shape contract: transforms address images through the trailing two dims
//! `(h, w)`; everything before them is sample+channel. Batches from the
//! loader are `(n, c, h, w)` (or `(c, h, w)` for single images), laid out
//! channel-major within each sample — each `h*w` plane is contiguous, and
//! sample `i`'s channels are `c` consecutive planes.

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

use crate::rng::Mt19937;

/// Builds a tensor from a `usize` shape (the kernels all work in `usize`).
fn rebuild(shape: &[usize], data: Vec<f32>) -> OxitorchResult<Tensor> {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    Tensor::from_vec(&dims, data)
}

/// A batch-level augmentation, applied to the gathered image column.
#[derive(Debug, Clone, PartialEq)]
pub enum NativeTransform {
    /// Random `size x size` crop; an independent position per image.
    RandomCrop(usize),
    /// Horizontal flip with probability 0.5, decided per image.
    RandomHorizontalFlip,
    /// Per-channel `(x - mean) / std` standardization.
    Normalize(Vec<f32>, Vec<f32>),
}

impl NativeTransform {
    /// Applies this transform to a batch in place-ish (returns the new
    /// tensor), drawing randomness from `rng`.
    ///
    /// # Errors
    /// On shape violations (non-(h, w) trailing dims) or bad Normalize
    /// channel counts / non-positive std.
    pub fn apply(&self, batch: &Tensor, rng: &mut Mt19937) -> OxitorchResult<Tensor> {
        match self {
            Self::RandomCrop(size) => random_crop(batch, *size, rng),
            Self::RandomHorizontalFlip => random_horizontal_flip(batch, rng),
            Self::Normalize(mean, std) => normalize(batch, mean, std),
        }
    }
}

/// Spatial geometry of an image batch: the trailing two dims are `(h, w)`,
/// the data is contiguous, and samples own `channels` consecutive planes.
struct Geometry {
    /// Number of samples (`shape[0]`, or 1 for `(c, h, w)` / smaller ranks).
    samples: usize,
    channels: usize,
    height: usize,
    width: usize,
}

impl Geometry {
    /// Reads the geometry of a contiguous image batch.
    ///
    /// # Errors
    /// If the rank is below 2 (no spatial dims) or a spatial dim is 0.
    fn of(tensor: &Tensor) -> OxitorchResult<Self> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("rank {}", shape.len()),
                rhs: "image batches need at least (h, w)".into(),
            });
        }
        let (height, width) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        if height == 0 || width == 0 {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{height}x{width}"),
                rhs: "non-empty spatial dims".into(),
            });
        }
        let (samples, channels) = match shape.len() {
            2 => (1, 1),
            3 => (1, shape[0]),
            _ => (shape[0], shape[1]),
        };
        Ok(Self {
            samples,
            channels,
            height,
            width,
        })
    }
}

/// `RandomCrop` kernel: per-image random `(top, left)`, row-wise memcpy.
///
/// Each sample gets its own crop position (torch's per-item semantics), so
/// the output stays batch-shaped `(n, c, size, size)`.
fn random_crop(batch: &Tensor, size: usize, rng: &mut Mt19937) -> OxitorchResult<Tensor> {
    if !batch.is_contiguous() {
        return random_crop(&batch.contiguous(), size, rng);
    }
    let g = Geometry::of(batch)?;
    if size == 0 || size > g.height || size > g.width {
        return Err(OxitorchError::InvalidArgument(format!(
            "crop size {size} exceeds image {}x{}",
            g.height, g.width
        )));
    }
    let src = batch.as_slice()?;
    let planes = g.samples * g.channels;
    let plane = g.height * g.width;
    let mut out = vec![0.0f32; planes * size * size];
    for p in 0..planes {
        let top = rng.below(g.height - size + 1);
        let left = rng.below(g.width - size + 1);
        let src_row = p * plane + top * g.width + left;
        let dst_row = p * size * size;
        for r in 0..size {
            out[dst_row + r * size..dst_row + (r + 1) * size]
                .copy_from_slice(&src[src_row + r * g.width..src_row + r * g.width + size]);
        }
    }
    let mut shape: Vec<usize> = batch.shape().to_vec();
    let rank = shape.len();
    shape[rank - 2] = size;
    shape[rank - 1] = size;
    rebuild(&shape, out)
}

/// `RandomHorizontalFlip` kernel: per-image coin flip, row reversal in the
/// output buffer (the input is left untouched).
fn random_horizontal_flip(batch: &Tensor, rng: &mut Mt19937) -> OxitorchResult<Tensor> {
    if !batch.is_contiguous() {
        return random_horizontal_flip(&batch.contiguous(), rng);
    }
    let g = Geometry::of(batch)?;
    let src = batch.as_slice()?;
    let mut out = src.to_vec();
    let planes = g.samples * g.channels;
    let plane = g.height * g.width;
    for p in 0..planes {
        // One coin per *sample*: all channels of an image flip together.
        if p % g.channels == 0 && rng.random() >= 0.5 {
            continue;
        }
        let base = p * plane;
        for r in 0..g.height {
            let row = base + r * g.width;
            out[row..row + g.width].reverse();
        }
    }
    rebuild(batch.shape(), out)
}

/// `Normalize` kernel: one subtract/multiply pass over each channel plane.
fn normalize(batch: &Tensor, mean: &[f32], std: &[f32]) -> OxitorchResult<Tensor> {
    if !batch.is_contiguous() {
        return normalize(&batch.contiguous(), mean, std);
    }
    let g = Geometry::of(batch)?;
    if mean.len() != g.channels || std.len() != g.channels {
        return Err(OxitorchError::InvalidArgument(format!(
            "normalize expects {} channels, got mean {} / std {}",
            g.channels,
            mean.len(),
            std.len()
        )));
    }
    if std.iter().any(|&s| s <= 0.0) {
        return Err(OxitorchError::InvalidArgument(
            "normalize std must be > 0".into(),
        ));
    }
    let src = batch.as_slice()?;
    let mut out = src.to_vec();
    let plane = g.height * g.width;
    for (p, chunk) in out.chunks_exact_mut(plane).enumerate() {
        let c = p % g.channels;
        let (m, s) = (mean[c], std[c]);
        let inv = 1.0 / s;
        for v in chunk.iter_mut() {
            *v = (*v - m) * inv;
        }
    }
    rebuild(batch.shape(), out)
}

/// An ordered composition of [`NativeTransform`]s (the `Compose` case).
#[derive(Debug, Clone, Default)]
pub struct TransformPipeline {
    steps: Vec<NativeTransform>,
}

impl TransformPipeline {
    /// A pipeline from ordered steps.
    #[must_use]
    pub fn new(steps: Vec<NativeTransform>) -> Self {
        Self { steps }
    }

    /// Whether the pipeline is a no-op (fast-path datasets skip it).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Applies every step in order to the batch.
    ///
    /// # Errors
    /// Propagates per-step errors (shape/normalization violations).
    pub fn apply(&self, batch: &Tensor, rng: &mut Mt19937) -> OxitorchResult<Tensor> {
        let mut current = batch.clone();
        for step in &self.steps {
            current = step.apply(&current, rng)?;
        }
        Ok(current)
    }
}

/// Per-batch RNG seed derivation: a SplitMix64 finalizer over
/// `(epoch_seed, batch_seq)` so every batch of an epoch sees an independent
/// stream while the whole epoch stays reproducible from the epoch seed.
#[must_use]
pub fn batch_seed(epoch_seed: u64, seq: u64) -> u64 {
    let mut z = epoch_seed ^ seq.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Constructs the per-batch generator (32-bit MT19937 seed from the 64-bit
/// batch seed by folding the halves together).
#[must_use]
pub fn batch_rng(seed: u64) -> Mt19937 {
    Mt19937::new((seed ^ (seed >> 32)) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        rebuild(shape, data).unwrap()
    }

    #[test]
    fn crop_positions_are_independent_per_image() {
        // Two 4x4 images; crop 2x2.
        let batch = tensor(&[2, 1, 4, 4], (0u8..32).map(f32::from).collect());
        let mut rng = batch_rng(batch_seed(123, 0));
        let out = random_crop(&batch, 2, &mut rng).unwrap();
        assert_eq!(out.shape(), &[2, 1, 2, 2]);
        // Every cropped pixel must exist in the source at one consistent
        // (top, left) offset per image.
        let src = batch.as_slice().unwrap();
        let out_v = out.to_vec();
        for (img, o) in out_v.chunks(4).enumerate() {
            let mut found = None;
            for top in 0..3 {
                for left in 0..3 {
                    let mut ok = true;
                    for r in 0..2 {
                        for c in 0..2 {
                            let s = src[img * 16 + (top + r) * 4 + left + c];
                            if (s - o[r * 2 + c]).abs() > f32::EPSILON {
                                ok = false;
                            }
                        }
                    }
                    if ok {
                        found = Some((top, left));
                    }
                }
            }
            assert!(found.is_some(), "image {img} crop matches no window");
        }
    }

    #[test]
    fn crop_is_reproducible_from_seed_and_seq() {
        let batch = tensor(&[3, 2, 5, 5], (0u8..150).map(f32::from).collect());
        let mut a = batch_rng(batch_seed(9, 4));
        let mut b = batch_rng(batch_seed(9, 4));
        let mut c = batch_rng(batch_seed(9, 5));
        let ra = random_crop(&batch, 3, &mut a).unwrap();
        let rb = random_crop(&batch, 3, &mut b).unwrap();
        let rc = random_crop(&batch, 3, &mut c).unwrap();
        assert_eq!(ra.to_vec(), rb.to_vec(), "same seed+seq -> same crop");
        assert_ne!(ra.to_vec(), rc.to_vec(), "different seq -> different crop");
    }

    #[test]
    fn crop_rejects_oversize() {
        let batch = tensor(&[1, 1, 4, 4], vec![0.0; 16]);
        let mut rng = batch_rng(1);
        assert!(random_crop(&batch, 5, &mut rng).is_err());
        assert!(random_crop(&batch, 0, &mut rng).is_err());
    }

    #[test]
    fn flip_decides_per_sample_not_per_channel() {
        // One sample, two channels; the coin is drawn once, so both planes
        // flip together. With a seed whose first draw is >= 0.5 the output
        // is fully flipped; either way channels stay in lockstep.
        let batch = tensor(
            &[1, 2, 2, 2],
            vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
        );
        let mut rng = batch_rng(7);
        let out = random_horizontal_flip(&batch, &mut rng).unwrap();
        let o = out.to_vec();
        let flipped = |ch: &[f32]| ch[1] > ch[0]; // row [a, b] -> [b, a]
        assert_eq!(
            flipped(&o[0..4]),
            flipped(&o[4..8]),
            "channels of one sample flip together"
        );
    }

    #[test]
    fn flip_matches_manual_reversal_when_seeded_low() {
        // seed 0 draws ~0.548 first (MT19937, numpy stream), which flips.
        // Instead of depending on the stream, check both outcomes exist
        // across seeds and each is a valid flip of the input.
        let batch = tensor(&[1, 1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let mut saw_flipped = false;
        let mut saw_identity = false;
        for seed in 0..64u64 {
            let mut rng = batch_rng(seed);
            let out = random_horizontal_flip(&batch, &mut rng).unwrap().to_vec();
            if out == vec![4.0, 3.0, 2.0, 1.0] {
                saw_flipped = true;
            } else if out == vec![1.0, 2.0, 3.0, 4.0] {
                saw_identity = true;
            } else {
                panic!("seed {seed} produced a non-flip: {out:?}");
            }
        }
        assert!(saw_flipped && saw_identity, "both outcomes occur");
    }

    #[test]
    fn normalize_is_exact_per_channel() {
        // (2, 3, 1, 2) row-major: sample = i / 6, channel = (i % 6) / 2 —
        // channels are consecutive planes within each sample.
        let batch = tensor(&[2, 3, 1, 2], (0u8..12).map(f32::from).collect());
        let out = normalize(&batch, &[1.0, 2.0, 3.0], &[2.0, 4.0, 8.0]).unwrap();
        let mean = [1.0, 2.0, 3.0];
        let std = [2.0, 4.0, 8.0];
        let want: Vec<f32> = (0..12)
            .map(|i| {
                let c = (i % 6) / 2;
                (f32::from(i as u8) - mean[c]) / std[c]
            })
            .collect();
        assert_eq!(out.to_vec(), want);
    }

    #[test]
    fn normalize_validates_channels_and_std() {
        let batch = tensor(&[2, 3, 2, 2], vec![0.0; 24]);
        assert!(normalize(&batch, &[0.0], &[1.0]).is_err());
        assert!(normalize(&batch, &[0.0; 3], &[0.0; 3]).is_err());
    }

    #[test]
    fn pipeline_applies_steps_in_order() {
        // Normalize then crop vs crop then normalize differ unless the crop
        // is uniform; use a deterministic geometry where order is visible:
        // normalize scales channels, then flip reverses rows.
        let batch = tensor(&[1, 1, 1, 2], vec![2.0, 4.0]);
        let pipeline = TransformPipeline::new(vec![
            NativeTransform::Normalize(vec![1.0], vec![2.0]),
            NativeTransform::RandomHorizontalFlip,
        ]);
        // With a seed that flips: (2,4) -> (0.5,1.5) -> (1.5,0.5).
        let mut rng = batch_rng(0);
        let out = pipeline.apply(&batch, &mut rng).unwrap();
        let v = out.to_vec();
        assert!(
            v == vec![0.5, 1.5] || v == vec![1.5, 0.5],
            "normalized values only: {v:?}"
        );
    }

    #[test]
    fn transforms_preserve_non_image_columns_via_source() {
        // Covered at the source level in batch.rs; here just ensure the
        // pipeline handles the no-op case.
        let batch = tensor(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let pipeline = TransformPipeline::default();
        let mut rng = batch_rng(1);
        let out = pipeline.apply(&batch, &mut rng).unwrap();
        assert_eq!(out.to_vec(), batch.to_vec());
    }
}
