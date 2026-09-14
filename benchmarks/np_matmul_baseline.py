#!/usr/bin/env python3
"""NumPy matmul baseline for oxitorch benchmark comparisons.

plan.md Phase 0: "Benchmark harness from day 1: simple matmul/conv
micro-bench vs. NumPy and (optional) PyTorch, tracked in CI with a nightly
job." This script is the NumPy side of that comparison. It uses the same
sizes as the criterion groups in `crates/oxi-tensor/benches/matmul.rs`.

Usage:
    python benchmarks/np_matmul_baseline.py [--sizes 64 128 256 512]
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def bench_matmul(n: int, iters: int = 20) -> float:
    """Returns best-of-iters wall time in seconds for an (n, n) @ (n, n)."""
    rng = np.random.default_rng(0x5EED)
    a = rng.standard_normal((n, n), dtype=np.float64).astype(np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    a @ b  # warm up BLAS
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
        help="square matrix sizes to benchmark",
    )
    args = parser.parse_args()

    print(f"numpy {np.__version__} matmul baseline (best of 20, f32)")
    print(f"{'size':>6} {'seconds':>12} {'gflops':>10}")
    for n in args.sizes:
        seconds = bench_matmul(n)
        gflops = (2 * n**3) / seconds / 1e9
        print(f"{n:>6} {seconds:>12.6f} {gflops:>10.2f}")


if __name__ == "__main__":
    main()
