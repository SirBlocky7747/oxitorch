# Oxitorch

[![CI](https://github.com/oxitorch/oxitorch/actions/workflows/ci.yml/badge.svg)](https://github.com/oxitorch/oxitorch/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/oxitorch)](https://pypi.org/project/oxitorch/)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)

> Working codename: **oxitorch** (rename freely).

A PyTorch-equivalent deep-learning framework whose **tensor engine, autograd,
and NN layers are implemented in Rust** and exposed to Python via PyO3
bindings — aiming for PyTorch-compatible semantics so users can port models
with minimal friction.

## Status: Phase 3 — NN module system

The full plan lives in [`plan.md`](plan.md). Currently:

- ✅ Cargo workspace: `oxi-core` (types) + real `oxi-tensor` engine +
  `oxi-autograd` (reverse-mode AD); `oxi-nn`, `oxi-optim`, `oxi-data`,
  `oxi-io` are placeholders
- ✅ `Tensor` with `Arc` storage + zero-copy views: slice/narrow, select,
  reshape, transpose/permute, broadcast_to, index_select, gather/scatter,
  masked_select, contiguous
- ✅ NumPy-compatible broadcasting, stride-aware elementwise kernels (registry
  of ~30 ops), thread-parallel reductions (sum/mean/max/min/argmax/argmin/norm
  with dim/keepdim), parallel SGEMM `matmul`, and batched `bmm` with
  per-batch VJPs — `Linear` accepts `(in,)`, `(batch, in)`, and any
  higher-rank `(..., in)` minibatch, torch parity
- ✅ Autograd: `Var` computation graph, topological `backward()` with grad
  accumulation, broadcast-aware grad reduction (`unbroadcast`), VJPs built
  from differentiable primitives so **second-order gradients work**, and
  finite-difference `gradcheck`
- ✅ **Indexing is graph-connected end to end**: `gather`/`index_select`/
  fancy-index `t[[i, j]]`, `scatter` (last-write-wins VJP: grads route only
to the final writer, overwritten base rows get zero), and `masked_select`
(broadcast-grid scatter so duplicate picks accumulate) — embedding-style
duplicate-token-id grad accumulation is exact vs NumPy
- ✅ Grad mode & memory: `no_grad()`/`enable_grad()` context managers
  (thread-local, torch-composable nesting), `set_grad_enabled`/
  `is_grad_enabled`, and `backward(retain_graph=True)` — VJPs are freed by
  default so double backward errors like torch, retained graphs accumulate
  leaf grads, and ops under `no_grad` record no graph at all
- ✅ Python: `oxitorch.Tensor` with `requires_grad=`, `.backward()`, `.grad`,
  `.detach()`, `.zero_grad()`; scalar arithmetic (`t * 2 + 1`, `2 - t`, `t**3`)
  is fully autograd-connected; grads verified against NumPy
- ✅ **NN module system (Python-side per plan decision): `Module`/`Parameter`/
  `Sequential` with torch-style `named_parameters()` paths and
  `state_dict()` roundtrips; recursive `train()`/`eval()`**
- ✅ **Layers: `Linear` (kaiming-uniform init), `Conv2d` + `MaxPool2d`
  (torch receptive-field fan-in init, `state_dict` roundtrips), `LayerNorm`,
  `RMSNorm`, `Embedding` (duplicate-index grad accumulation), `Dropout`
  (inverted, train/eval aware), activation wrappers**
- ✅ **Convolutions & pooling (Phase 4 pre-work): `F.conv2d` via im2col —
  window gather (`index_select`) + per-image `bmm` against the unfolded
  weights, zero padding built with `scatter` so its VJP lands gradients in
  the center region only — and `F.max_pool2d` as the same gather + Max
  reduce. Values match a direct NumPy loop across stride/padding configs,
  x/weight/bias gradients pass finite-difference checks, and a
  conv→relu→pool→linear net trains in lockstep with a NumPy reference**
- ✅ **Functional API: linear, conv2d, max_pool2d, softmax/log_softmax
  (numerically stable), gelu/silu/relu/sigmoid/tanh/leaky_relu,
  layer_norm/rms_norm, embedding, dropout, mse_loss, cross_entropy — every
  value & gradient numpy-parity tested**
- ✅ **Optimizers: SGD (+momentum/nesterov), Adam (L2-in-grad), and AdamW
  (decoupled decay, torch defaults) over `Module.parameters()` —
  closed-form verified step by step. Optimizer math and state run in Rust
  (`oxi-autograd::optim`): no numpy round-trip per step, tensor identity
  preserved across the training run, lazy per-parameter state (torch
  semantics); ~2x faster `step()` than the earlier numpy implementation.**
- ✅ **LR schedulers: `StepLR`, `CosineAnnealingLR`, `LambdaLR`,
  `LinearWarmup`, and `SequentialLR` (composes warmup → cosine;
  construction-order-safe; torch constructor-steps-once semantics)**
- ✅ **Gradient clipping: `clip_grad_norm_` (any p-norm incl. inf, returns
  the pre-clip norm) and `clip_grad_value_`, in place via a Rust-side
  `Tensor.grad=` setter**
- ✅ **End-to-end: XOR MLP trains to zero loss via `cross_entropy` + SGD;
  state_dict cloning reproduces predictions exactly; a conv→relu→pool→
  linear CNN trains with AdamW + warmup/cosine + clipping, matching a
  from-scratch NumPy reference run epoch-for-epoch (Phase 4 exit criterion)**
- ✅ **Data pipeline (Phase 5): `oxitorch.utils.data.DataLoader` with torch's
  constructor surface (`batch_size`, `shuffle`, `drop_last`, `num_workers`,
  `sampler`, `collate_fn`, `seed`). Tensor-backed datasets take the native
  fast path: batches are gathered by Rust worker threads (no GIL, no
  multiprocessing) with a bounded prefetch window and deterministic plan
  ordering. `oxitorch.datasets.MNIST` downloads + parses the official IDX
  files in pure Rust; `CIFAR10` reads the python-pickle distribution;
  numpy-domain transforms (`RandomCrop`, `RandomHorizontalFlip`,
  `Normalize`, `Compose`) cover torchvision-lite needs — and native-repr
  chains run as batch kernels *inside the worker threads* (augmented
  loading: ~31x faster than the per-item numpy path on 60k MNIST images).
  The engine's MT19937 shuffle RNG is NumPy-`RandomState`-exact.**
- ✅ **MNIST exit criterion: `python examples/train_mnist.py` goes from raw
  download to **98.09% test accuracy** in 8 epochs (MLP, AdamW + cosine,
  all batches fetched by Rust workers; ~4.4M rows/s loader throughput)**
- ✅ CI: fmt + clippy + cargo test + pytest matrix + wheel builds
- ✅ Benchmarks: criterion harness + NumPy baseline (nightly CI job)

Next: safetensors/state-dict serialization (Phase 6), GPU backends (Phase 7).

## Architecture

```
┌────────────────────────────────────────────────┐
│  Python package: oxitorch                      │
│  (torch-like API, type stubs, docs)            │
├────────────────────────────────────────────────┤
│  PyO3 binding layer (crate: oxi-bindings)      │
├────────────────────────────────────────────────┤
│  oxi-nn │ oxi-optim │ oxi-data │ oxi-io        │
├──────────────── depends on                   ──┤
│  oxi-autograd (graph, reverse-mode AD)         │
├────────────────────────────────────────────────┤
│  oxi-tensor (dtype, device, layout, dispatch)  │
├────────────────────────────────────────────────┤
│  Backends: cpu (SIMD/threads) │ wgpu │ cuda    │
└────────────────────────────────────────────────┴
```

## Development

### Requirements

- Rust 1.83+ (`rustup update stable`)
- Python 3.9–3.14
- [maturin](https://www.maturin.rs) (build backend) and pytest for tests:
  `pip install maturin pytest numpy`

### Build & test

```bash
# Rust side
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# Python side — editable install builds the native extension via maturin
pip install -e ".[test]"
pytest
```

### Benchmarks

```bash
cargo bench -p oxi-tensor                 # criterion benches (Rust)
python benchmarks/np_matmul_baseline.py   # NumPy baseline (tracked in CI)
```

## License

Dual-licensed under MIT or Apache-2.0, at your option — see
[LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
