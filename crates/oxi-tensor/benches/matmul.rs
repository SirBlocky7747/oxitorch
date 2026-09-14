//! Day-1 benchmark harness for oxitorch (plan.md Phase 0).
//!
//! Measures `Tensor::matmul` across representative sizes. The naive loop is
//! included as a reference floor; once Phase 1 lands better kernels these
//! groups become the regression baseline tracked by the nightly CI job
//! (`.github/workflows/nightly-bench.yml`).

// criterion_group! generates entry-point functions without doc comments,
// which the workspace-wide `missing_docs` lint would otherwise flag.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use oxi_tensor::Tensor;

/// Fills a tensor deterministically (xorshift) so benches are reproducible.
fn fill(t: &Tensor) -> Tensor {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let data: Vec<f32> = (0..t.numel())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2000) as f32 / 1000.0 - 1.0
        })
        .collect();
    Tensor::from_vec(
        &t.shape().iter().map(|&d| d as i64).collect::<Vec<_>>(),
        data,
    )
    .expect("bench shapes are valid")
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[64usize, 128, 256, 512] {
        let a = fill(&Tensor::zeros(&[n as i64, n as i64]).unwrap());
        let b = fill(&Tensor::zeros(&[n as i64, n as i64]).unwrap());
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_function(format!("{n}x{n}x{n}"), |bencher| {
            bencher.iter(|| black_box(&a).matmul(black_box(&b)).unwrap());
        });
    }
    group.finish();
}

/// Reference floor: the naive triple loop, so we can see what the packed
/// SGEMM path actually buys us.
#[allow(clippy::many_single_char_names)] // m, k, n are the GEMM convention
fn bench_matmul_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_naive");
    for &n in &[64usize, 128] {
        let a = fill(&Tensor::zeros(&[n as i64, n as i64]).unwrap());
        let b = fill(&Tensor::zeros(&[n as i64, n as i64]).unwrap());
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_function(format!("{n}x{n}x{n}"), |bencher| {
            bencher.iter_batched(
                || vec![0.0f32; n * n],
                |mut out| {
                    let (m, k, nn) = (n, n, n);
                    let a = a.as_slice().expect("contiguous");
                    let b = b.as_slice().expect("contiguous");
                    for i in 0..m {
                        for p in 0..k {
                            let av = a[i * k + p];
                            for j in 0..nn {
                                out[i * nn + j] += av * b[p * nn + j];
                            }
                        }
                    }
                    black_box(out)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_matmul_naive);
criterion_main!(benches);
