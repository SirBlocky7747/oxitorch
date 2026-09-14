//! IDX file format parsing (the format of the MNIST database) in pure Rust.
//!
//! IDX3 (images): magic `0x00000803` then dims (big-endian u32) then u8 data.
//! IDX1 (labels): magic `0x00000801` then count then u8 labels. Both may be
//! gzip-compressed (the usual distribution) — detected by the `1f 8b` magic.
//!
//! Data is normalized to `f32` in `[0, 1]` at load time (torchvision's
//! `transforms.ToTensor()` semantics) and the images tensor is shaped
//! `(n, 1, 28, 28)`; labels stay a 1-D `f32` tensor of class indices.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

/// A parsed MNIST-style dataset: images `(n, 1, 28, 28)` in `[0, 1]` and
/// labels `(n,)` class indices as f32.
pub struct MnistData {
    /// Normalized images, row-major, shape `(n, 1, 28, 28)`.
    pub images: Tensor,
    /// Class labels, shape `(n,)`, f32 (matches the engine's f32 compute).
    pub labels: Tensor,
}

impl MnistData {
    /// Loads the four MNIST files from `root` (`train` selects the train or
    /// test split). Gzip and plain IDX are both accepted.
    ///
    /// # Errors
    /// Missing files, bad magic numbers, truncation, or length mismatches.
    pub fn load(root: &Path, train: bool) -> OxitorchResult<Self> {
        let prefix = if train { "train" } else { "t10k" };
        let images_path = root.join(format!("{prefix}-images-idx3-ubyte"));
        let images_path_gz = root.join(format!("{prefix}-images-idx3-ubyte.gz"));
        let labels_path = root.join(format!("{prefix}-labels-idx1-ubyte"));
        let labels_path_gz = root.join(format!("{prefix}-labels-idx1-ubyte.gz"));

        let pick = |plain: &Path, gz: &Path| -> OxitorchResult<PathBuf> {
            if plain.exists() {
                Ok(plain.to_path_buf())
            } else if gz.exists() {
                Ok(gz.to_path_buf())
            } else {
                Err(OxitorchError::Other(format!(
                    "MNIST file not found: {} (or .gz)",
                    plain.display()
                )))
            }
        };

        let images_raw = read_file(&pick(&images_path, &images_path_gz)?)?;
        let labels_raw = read_file(&pick(&labels_path, &labels_path_gz)?)?;
        Self::from_idx(&images_raw, &labels_raw)
    }

    /// Parses raw IDX3 + IDX1 bytes into tensors.
    ///
    /// # Errors
    /// On bad magic, truncation, dimension mismatches, or length mismatches.
    pub fn from_idx(images_bytes: &[u8], labels_bytes: &[u8]) -> OxitorchResult<Self> {
        let images = parse_idx3(images_bytes)?;
        let labels = parse_idx1(labels_bytes)?;
        let n_images = images.shape()[0];
        let n_labels = labels.shape()[0];
        if n_images != n_labels {
            return Err(OxitorchError::Other(format!(
                "MNIST images/labels length mismatch: {n_images} vs {n_labels}"
            )));
        }
        Ok(Self { images, labels })
    }
}

use std::path::PathBuf;

fn read_file(path: &Path) -> OxitorchResult<Vec<u8>> {
    let mut f = File::open(path)
        .map_err(|e| OxitorchError::Other(format!("open {}: {e}", path.display())))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)
        .map_err(|e| OxitorchError::Other(format!("read {}: {e}", path.display())))?;
    // Gzip magic: 1f 8b.
    if buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b {
        let mut decoder = flate2::read::GzDecoder::new(&buf[..]);
        let mut out = Vec::new();
        decoder
            .read_to_end(&mut out)
            .map_err(|e| OxitorchError::Other(format!("gunzip {}: {e}", path.display())))?;
        return Ok(out);
    }
    Ok(buf)
}

fn be_u32(b: &[u8], at: usize) -> OxitorchResult<u32> {
    if at + 4 > b.len() {
        return Err(OxitorchError::OutOfBounds(format!(
            "IDX header truncated at byte {at}"
        )));
    }
    Ok(u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]))
}

/// Parses IDX3 images into an `(n, 1, 28, 28)` f32 tensor in `[0, 1]`.
fn parse_idx3(b: &[u8]) -> OxitorchResult<Tensor> {
    let magic = be_u32(b, 0)?;
    if magic != 0x0000_0803 {
        return Err(OxitorchError::Other(format!(
            "bad IDX3 magic: {magic:#010x}"
        )));
    }
    let n = be_u32(b, 4)? as usize;
    let rows = be_u32(b, 8)? as usize;
    let cols = be_u32(b, 12)? as usize;
    let count = n.checked_mul(rows).and_then(|v| v.checked_mul(cols));
    let count = count.ok_or_else(|| OxitorchError::Other("IDX3 size overflow".into()))?;
    if b.len() < 16 + count {
        return Err(OxitorchError::OutOfBounds(format!(
            "IDX3 truncated: have {} bytes, need {}",
            b.len(),
            16 + count
        )));
    }
    let data: Vec<f32> = b[16..16 + count]
        .iter()
        .map(|&v| f32::from(v) / 255.0)
        .collect();
    let shape = [n as i64, 1, rows as i64, cols as i64];
    Tensor::from_vec(&shape, data)
}

/// Parses IDX1 labels into an `(n,)` f32 tensor.
fn parse_idx1(b: &[u8]) -> OxitorchResult<Tensor> {
    let magic = be_u32(b, 0)?;
    if magic != 0x0000_0801 {
        return Err(OxitorchError::Other(format!(
            "bad IDX1 magic: {magic:#010x}"
        )));
    }
    let n = be_u32(b, 4)? as usize;
    if b.len() < 8 + n {
        return Err(OxitorchError::OutOfBounds(format!(
            "IDX1 truncated: have {} bytes, need {}",
            b.len(),
            8 + n
        )));
    }
    let data: Vec<f32> = b[8..8 + n].iter().map(|&v| f32::from(v)).collect();
    Tensor::from_vec(&[n as i64], data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal in-memory IDX3/IDX1 pair (2 images, 2x3 px).
    fn sample_idx3() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x0000_0803u32.to_be_bytes());
        b.extend_from_slice(&2u32.to_be_bytes());
        b.extend_from_slice(&2u32.to_be_bytes());
        b.extend_from_slice(&3u32.to_be_bytes());
        b.extend_from_slice(&[0, 127, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
        b
    }

    fn sample_idx1() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&0x0000_0801u32.to_be_bytes());
        b.extend_from_slice(&2u32.to_be_bytes());
        b.extend_from_slice(&[7, 3]);
        b
    }

    #[test]
    fn parses_sample_idx() {
        let d = MnistData::from_idx(&sample_idx3(), &sample_idx1()).unwrap();
        assert_eq!(d.images.shape(), &[2, 1, 2, 3]);
        assert_eq!(d.labels.shape(), &[2]);
        let px = d.images.to_vec();
        assert!((px[0] - 0.0).abs() < 1e-7);
        assert!((px[1] - 127.0 / 255.0).abs() < 1e-6);
        assert!((px[2] - 1.0).abs() < 1e-7);
        assert_eq!(d.labels.to_vec(), vec![7.0, 3.0]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bad = sample_idx3();
        bad[2] = 0x09; // corrupt magic
        assert!(MnistData::from_idx(&bad, &sample_idx1()).is_err());
    }

    #[test]
    fn rejects_truncated_payload() {
        let full = sample_idx3();
        let truncated = &full[..full.len() - 1];
        assert!(MnistData::from_idx(truncated, &sample_idx1()).is_err());
    }

    #[test]
    fn rejects_length_mismatch() {
        let mut labels = sample_idx1();
        labels[7] = 5; // claim 5 labels, only 2 bytes follow
        let mut padded = labels;
        padded.extend_from_slice(&[0, 0, 0]);
        assert!(MnistData::from_idx(&sample_idx3(), &padded).is_err());
    }

    #[test]
    fn gzip_roundtrip() {
        // Gzip-compress the sample and ensure the reader sees through it.
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write as _;
        let mut enc = GzEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&sample_idx3()).unwrap();
        let gz = enc.finish().unwrap();

        let labels = sample_idx1();
        let dir = std::env::temp_dir().join(format!("oxi-data-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("train-images-idx3-ubyte.gz"), &gz).unwrap();
        std::fs::write(dir.join("train-labels-idx1-ubyte"), &labels).unwrap();

        let d = MnistData::load(&dir, true).unwrap();
        assert_eq!(d.images.shape(), &[2, 1, 2, 3]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
