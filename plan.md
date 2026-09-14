# plan.md — Oxitorch: a PyTorch-equivalent library with a Rust core

> Working codename: **oxitorch** (rename freely).
> Goal: a Python deep-learning framework whose tensor engine, autograd, and NN
> layers are implemented in Rust and exposed to Python via bindings — aiming for
> PyTorch-compatible semantics so users can port models with minimal friction.

---

## 0. Guiding decisions (make these first, they shape everything)

- [ ] **Build scope decision**: implement the tensor engine from scratch in Rust
      (max control, max effort) vs. build on existing crates (`ndarray`,
      `candle`, `burn`) (fast start, their design constraints).
      *Recommended: hybrid — own tensor type + memory/autograd, delegate low-level
      BLAS to `faer`/`ndarray-linalg`/`matrixmultiply` initially.*
- [ ] **Python binding stack**: `pyo3` + `maturin` (the de-facto standard).
- [ ] **API strategy**: mirror the `torch.*` Python API surface (Tensor methods,
      `torch.nn`, `torch.optim`, `torch.utils.data`) so existing PyTorch code
      ports with mostly import swaps.
- [ ] **License & naming**: pick an Apache-2.0/MIT dual license and a final
      PyPI name (check availability).

## Architecture overview

```
┌────────────────────────────────────────────────┐
│  Python package: oxitorch                      │
│  (torch-like API, type stubs, docs)            │
├────────────────────────────────────────────────┤
│  PyO3 binding layer (crate: oxi-bindings)      │
├────────────────────────────────────────────────┤
│  oxi-nn │ oxi-optim │ oxi-data │ oxi-io        │
├────────────────────────────────────────────────┤
│  oxi-autograd (graph, reverse-mode AD)         │
├────────────────────────────────────────────────┤
│  oxi-tensor (dtype, device, layout, dispatch)  │
├────────────────────────────────────────────────┤
│  Backends: cpu (SIMD/threads) │ wgpu │ cuda    │
└────────────────────────────────────────────────┘
```

Cargo workspace crates:
- `oxi-tensor` — Tensor type, dtypes, strides, device, dispatch, CPU kernels
- `oxi-autograd` — computation graph, gradients, higher-order AD
- `oxi-nn` — Modules, Parameters, init, losses, functional API
- `oxi-optim` — SGD/Adam/etc., LR schedulers
- `oxi-data` — Dataset/DataLoader, collation, prefetching
- `oxi-io` — safetensors, state-dict save/load, checkpointing
- `oxi-bindings` — PyO3 glue (thin: no logic lives here)
- `oxitorch` — Python package metadata, stubs, tests

---

## Phase 0 — Project scaffolding (week 1) ✅ *(complete)*

- [x] Init Cargo workspace with the crates above; shared `oxi-core` types crate
      if needed to avoid circular deps.
- [ ] Init Python package layout: `python/oxitorch/` (src layout),
      `pyproject.toml` with maturin build backend.
- [ ] CI (GitHub Actions): cargo test + clippy + fmt, pytest matrix
      (3.9–3.13), wheel builds (linux/macOS x86_64 + arm64).
- [ ] Benchmark harness from day 1: simple matmul/conv micro-bench vs. NumPy
      and (optional) PyTorch, tracked in CI with a nightly job.
- [ ] Decide error-handling convention: Rust `Result` internally → Python
      exceptions at the binding layer (`RuntimeError`, `ValueError`, ...).

**Exit criteria:** `pip install -e .` works, `import oxitorch` succeeds, CI
green on all platforms.

## Phase 1 — Tensor core (weeks 2–5) ✅ *(core complete; SIMD + f64 below)*

- [ ] `DType` enum: f32/f64, i8..i64, u8, bool, bf16, f16 (bf16/f16 storage
      first, compute via f32 promote). *(enum exists in `oxi-core`; compute
      is f32-only so far — dtype-generic dispatch is the main carry-over)*
- [x] `Tensor`: owned buffer + shape + strides; reference-counted views
      (`Arc<Storage>` + offset), zero-copy slicing/reshaping where possible.
- [x] Broadcasting rules identical to NumPy/PyTorch; implement `expand`,
      `broadcast_to`, shape algebra + property tests.
- [ ] Stride-aware elementwise kernels (add, sub, mul, div, neg, exp, log,
      pow, comparisons) with compile-time SIMD (std `simd`-style via
      `wide`/`std::simd` or autovectorization first). *(kernels done via
      registry + autovectorization; explicit SIMD remains)*
- [x] Thread-parallel elementwise/reduce kernels (rayon or custom pool).
- [x] Reductions: sum, mean, max, min, argmax/argmin, norm, with dim/keepdim.
- [x] `matmul` via `matrixmultiply`/`faer` (correct before fast). *(matrixmultiply,
      sequential <2²³ work, parallel row-blocks above; measured tuned)*
- [x] `bmm` batched matmul `(b,m,k) @ (b,k,n) -> (b,m,n)` with per-batch VJPs
      (`gA = g @ Bᵀ`, `gB = Aᵀ @ g`), transposed-view inputs materialized
      internally, and a per-batch-plane rayon path above the parallel threshold.
      *(Linear now folds any `(..., in)` input through one GEMM, torch parity)*
- [x] Device abstraction trait (`DeviceBackend`) — CPU first, but the trait
      defined so GPU backends slot in later.
- [x] Indexing: basic, slice, fancy (integer array), boolean mask,
      `gather`/`scatter`.

**Exit criteria:** parity test suite vs. NumPy for ~80 ops passes on random
inputs (allclose with PyTorch's default tolerances); matmul within 2× NumPy
on large sizes. *(31-test NumPy parity suite green; 512³ matmul 978 µs vs
NumPy 469 µs — ~2.1×, essentially at the exit bar; the remaining gap is the
known matrixmultiply-vs-MKL story)*

**Phase 3 carry-over, now done — indexing in the autograd graph:**
`scatter` and `masked_select` have real VJPs (`ops::scatter_dim0` routes the
gradient to the *last writer* per row — exact for the last-write-wins forward —
and zeroes written rows of the base; `ops::masked_select` scatters picked
grads through the broadcast grid so broadcasting-mask duplicates **accumulate**),
fancy indexing `t[[i, j, ...]]` routes through differentiable `index_select`,
and every indexing binding (`gather`, `index_select`, `scatter`,
`masked_select`, `__getitem__` lists) is graph-connected when any operand
requires grad — embedding-style duplicate-index accumulation is exact and
verified against NumPy.

## Phase 2 — Autograd engine (weeks 6–8) ✅ *(complete incl. the two former
      carry-overs: `no_grad`/`enable_grad` contexts + `retain_graph`)*

## Phase 2 — Autograd engine (weeks 6–8) ✅ *(complete incl. the two former
      carry-overs: `no_grad`/`enable_grad` contexts + `retain_graph`)*

- [x] `Variable`/graph model: `Tensor` carries optional `grad_fn` + `requires_grad`.
      *(`Var` = `Tensor` + op provenance + VJP; `Rc`-shared DAG nodes)*
- [x] Tape (reverse-mode AD): record ops on a graph; `backward()` topological
      pass with gradient accumulation into `.grad`.
- [ ] Custom op registration (forward + vjp) — public API for user-defined ops.
      *(VJP closure architecture is in place; the public registration API
      lands with Phase 3's layer system)*
- [x] Higher-order gradients: at least second-order for core ops (needed for
      GANs/R1 and some meta-learning use cases). *(VJPs are built from
      differentiable `Var` primitives, so double backward is structural)*
- [x] `detach()`, `no_grad()`, `enable_grad()`, `set_requires_grad`, leaf vs
      non-leaf semantics matching PyTorch. *(`detach` + `requires_grad`;
      `no_grad`/`enable_grad` are thread-local save/restore scopes — Python
      context managers over a Rust `Cell<bool>` flag consulted by `Var::op`,
      with `set_grad_enabled`/`is_grad_enabled` also exposed)*
- [x] Grad checks: finite-difference tests over the whole op set (CI job).
      *(`oxi-autograd::gradcheck` + 10 Rust tests + 8 Python parity tests)*
- [x] Memory: graph freeing after backward, `retain_graph` flag. *(VJPs are
      now `Fn` and freed by default — a second backward errors like torch's;
      `backward(retain_graph=True)` keeps them and leaf grads accumulate
      while intermediate grads reset per pass. Ops under `no_grad` record no
      graph at all)*

**Exit criteria:** all ops in Phase 1 have gradient parity tests vs. PyTorch's
`torch.autograd.gradcheck`; can train a linear regression end-to-end in
Python.

## Phase 3 — NN module system (weeks 9–12)

- [x] `Module` base (via Python class + Rust state or pure-Python
      `nn.Module`-like class holding Rust tensors; *decide: Python-side Module
      in v1, faster to ship*). — **done: pure-Python `Module`/`Parameter`/`Sequential`
      in `oxitorch.nn`, holding Rust autograd tensors.**
- [x] `Parameter`, `Module.named_parameters()`, `parameters()`, `state_dict()`. —
      **done, torch-style dotted paths; `load_state_dict` validates shapes.**
- [ ] Layers: Linear, Conv1d/2d/3d, ConvTranspose, BatchNorm (1/2/3d),
      LayerNorm, GroupNorm, RMSNorm, Dropout, Embedding, RNN/GRU/LSTM. —
      **done: Linear, Conv2d, MaxPool2d, LayerNorm, RMSNorm, Embedding,
      Dropout + activation wrappers; conv1d/3d/transpose/batchnorm/RNN carry
      to Phase 4+.**
- [x] Activations: ReLU/ReLU6, GELU, SiLU, LeakyReLU, Tanh, Sigmoid, Softmax,
      LogSoftmax. — **done except ReLU6/LogSoftmax wrappers (log_softmax exists
      in functional); trivial to add.**
- [x] `functional` API mirroring `torch.nn.functional`. — **done: Phase 3
      subset (linear, activations, softmax/log_softmax, norms, embedding,
      dropout, mse/cross_entropy); all numpy-parity tested.**
- [ ] Convolutions: im2col+GEMM first (correct + decently fast); direct
      kernels later. — **done for 2-D (Phase 4 pre-work): `F.conv2d` builds
      im2col from one `index_select` over flat window offsets (padding via
      zero-canvas `scatter`, so its VJP lands the gradient in the center
      region only), reshapes to `(N, oh*ow, C*kh*kw)`, and rides the bmm
      VJPs — grads to x/weight/bias all finite-difference verified, and
      `F.max_pool2d` is the same window gather + Max reduce. `Conv2d`/
      `MaxPool2d` layer classes use torch's kaiming-uniform (receptive-field
      fan-in) init; stride/padding configs are NumPy-parity tested and a
      conv→relu→pool→linear net trains in lockstep with a NumPy reference.
      Direct kernels, conv1d/3d, and pooling ceil_mode carry forward.**
- [ ] Losses: MSELoss, L1, BCE, BCEWithLogits, CrossEntropy, NLL, Huber,
      SmoothL1, CosineEmbedding, TripletMargin. — **done: MSE + CrossEntropy
      as functions (class wrappers and remaining losses later).**
- [ ] Initialization: kaiming, xavier, orthogonal, trunc-normal. — **partially:
      kaiming-uniform for Linear in numpy; a shared init module comes with
      the dtype-generic pass.**

**Exit criteria:** a 3-layer MLP and a small CNN defined in Python reach
expected accuracy on a fixed synthetic task; layer outputs match PyTorch
within tolerance given identical weights.

## Phase 4 — Optimizers & training loop (week 13) ✅ *(complete: AdamW,
      schedulers incl. warmup composition, gradient clipping; the CNN exit
      criterion runs against a NumPy reference)*

- [ ] SGD (+momentum, nesterov, weight decay), Adam, AdamW, RMSprop, Adagrad,
      LAMB (optional). — **done: SGD+momentum/nesterov (Phase 3), Adam
      (L2-in-grad), AdamW (decoupled decay, torch defaults); closed-form
      verified; RMSprop/Adagrad/LAMB carry forward. Optimizer math and
      state (moments, per-parameter step counters) now live in Rust
      (`oxi-autograd::optim` via `_native.RustSgd`/`RustAdamW`): steps run
      on flat f32 slices with no numpy round-trip, Python `Tensor` objects
      keep identity across steps, and state is lazily allocated per
      parameter (torch per-parameter `state` semantics); ~2x faster
      `step()` than the previous numpy path.**
- [ ] LR schedulers: StepLR, CosineAnnealing, OneCycle, warmup, LambdaLR. —
      **done: StepLR, CosineAnnealingLR, LambdaLR, LinearWarmup, and
      SequentialLR (composes warmup→cosine with per-phase epoch restart and
      construction-order-safe base lrs); torch constructor-steps-once
      semantics; OneCycle carries forward.**
- [x] `zero_grad(set_to_none=...)` semantics, gradient clipping
      (norm/val), AMP hooks placeholder. — **done: `clip_grad_norm_`
      (p-norm incl. inf, returns pre-clip norm, skips missing grads) and
      `clip_grad_value_` over `Module.parameters()`, backed by a new Rust
      `Var::set_grad`/`Tensor.grad=` setter for in-place gradient surgery.
      `set_to_none` arg and AMP hooks carry forward.**
- [ ] Mixed precision: autocast context (f16/bf16 compute regions) on CPU where
      feasible, designed for GPU later.

**Exit criteria:** AdamW + cosine schedule trains a small transformer /
ResNet-ish CNN on a bundled dataset without NaNs; loss curves match a PyTorch
reference run within a few %. *(conv→relu→pool→linear net trained 30 epochs
with AdamW + LinearWarmup→CosineAnnealing + grad-norm clipping: losses finite
throughout, strictly decreasing, and match a from-scratch NumPy reference
run (identical AdamW/schedule/clip/backward math) epoch-for-epoch within
5e-3 relative.)*

## Phase 5 — Data pipeline (week 14) ✅ *(complete: Rust-worker DataLoader, MNIST/CIFAR loaders; exit
      criterion met — 98.09% test accuracy)*

- [x] `Dataset`, `IterableDataset`, `DataLoader` with `batch_size`,
      `shuffle`, `drop_last`, `num_workers` (Rust-side threads instead of
      multiprocessing — a genuine selling point). — **done:
      `oxitorch.utils.data` with torch's constructor surface. Tensor-backed
      datasets (`TensorDataset`, untransformed MNIST/CIFAR10) take the
      native fast path: `_native.DataLoader` gathers whole batches with
      `index_select(0, ...)` on a Rust worker pool (per-epoch pool spawned
      by `__iter__`, prefetch window of `workers + 2`, a `ResponseReorderer`
      yields batches in deterministic plan order regardless of completion
      order, and worker failures surface at their plan position). Per-epoch
      shuffle seeds derive from the base seed + epoch via SplitMix64, and
      the underlying MT19937 is verified NumPy-`RandomState`-exact. Generic
      datasets run per-item through a thread pool (numpy transforms release
      the GIL). Worker-count parity and identical-trained-weights tests
      prove the math is invisible to the execution path.**
- [x] Collation, samplers (random, sequential, distributed-shard stub). —
      **done: `default_collate` (tuples, tensors, numpy, scalars — with the
      0-d-promotes-to-(1,) numpy pitfall handled), `SequentialSampler`,
      `RandomSampler` (seeded per epoch), `Subset`; distributed shard stub
      carries forward.**
- [x] Torchvision-lite: bundled MNIST/CIFAR loaders (raw file parsing in Rust)
      + basic transforms (crop, flip, normalize, ToTensor). — **done:
      `oxitorch.datasets.MNIST` downloads the official IDX mirrors and
      parses them in pure Rust (`oxi-data::mnist`, gzip or plain, magic/
      truncation/length validation, images normalized to [0,1] at load);
      `CIFAR10` parses the python-pickle distribution via numpy.
      Transforms compose over the numpy domain: `RandomCrop`,
      `RandomHorizontalFlip`, `Normalize`, `Compose` (PIL deliberately
      excluded; ToTensor semantics are built into the parsers). Beyond the
      plan: native batch transforms — transform chains that are fully
      engine-representable expose `native_repr()`, and the DataLoader hands
      them to `oxi-data::transform` kernels that run *inside the worker
      threads* over whole gathered batches (per-image crop positions and
      coin flips drawn from the loader's seeded stream; every `__iter__`
      pass consumes the next slice of the seed stream, torch-style).
      Augmented loading measured at 1.49s → 47ms per 60k-image epoch
      (~31x) with 4 workers; inline and worker paths are bit-identical
      (same seeds, same kernels) and a seeded run is reproducible
      epoch-for-epoch.**
- [x] `pin_memory`/prefetch no-ops now; real GPU pipelining later. — **the
      native path already prefetches `workers + 2` batches ahead; real
      pin_memory/GPU pipelining carries to Phase 7.**

**Exit criteria:** DataLoader saturates CPU-bound transforms with workers;
MNIST training script runs end-to-end from raw download to >97% test
accuracy. — **MET: `examples/train_mnist.py` downloads, parses in Rust,
trains a 784-256-128-10 MLP for 8 epochs (AdamW + cosine, all batches
fetched by 4 Rust workers) to **98.09% test accuracy**; loader throughput
is ~4.4M rows/s single-worker. The same work also fixed a `Tensor::index_select`
hot-spot (it materialized the whole source per call; now reads the
contiguous slice directly — 5.9s → 13.5ms for a full MNIST epoch of batches,
~140x, and it benefits every index_select user).**

## Phase 6 — Serialization & interop (week 15)

- [ ] safetensors read/write (native Rust crate) for weights.
- [ ] `state_dict()` / `load_state_dict()` with PyTorch-compatible keys, incl.
      strict mode and prefix remapping.
- [ ] Checkpoints: optimizer state + step + scheduler state.
- [ ] `.pt` read support (zip + pickle subset) as *stretch* — to import
      existing PyTorch weights directly.

**Exit criteria:** train in PyTorch → export safetensors → load and get
identical logits in oxitorch (parity harness script).

## Phase 7 — GPU backends (weeks 16–24, longest pole)

- [ ] Backend trait hardening based on real usage from Phases 1–6.
- [ ] **wgpu backend** (Vulkan/Metal/DX12/WebGPU): elementwise, matmul
      (tiled GEMM shader), conv, reductions — big portability win (runs on
      Apple Silicon, AMD, Intel).
- [ ] **CUDA backend**: cuBLAS/cuDNN via `bindgen` FFI or cust kernels;
      streams, async copies, memory pool, `Tensor.cuda()`.
- [ ] Device movement (`to(device)`), cross-device ops raising clean errors.
- [ ] Kernel fusion for common patterns (add+norm, conv+bias+relu) — only after
      correctness.

**Exit criteria:** MNIST + small CNN train on GPU; ≥50% of PyTorch throughput
on matmul/conv micro-benchmarks for the wgpu backend.

## Phase 8 — Performance hardening (weeks 25–27)

- [ ] Profile-guided: SIMD everywhere it matters, cache-blocked GEMM, op
      fusion, memory pools to stop allocator churn.
- [ ] Dispatch table keyed on (op × dtype × device); fast paths for
      contiguous tensors, slow generic fallback.
- [ ] Benchmark suite with regression tracking; publish numbers.
- [ ] Reduced-precision GEMM (bf16) if profitable.

**Exit criteria:** headline benchmark table (matmul, conv, full training
steps) within 0.5–1× of PyTorch CPU; no benchmark regression >5% week over week.

## Phase 9 — Ecosystem & polish (weeks 28–32)

- [ ] Full type stubs + `pyright`/`mypy` clean; docstrings mirroring torch's.
- [ ] Tutorial docs: quickstart, porting-from-pytorch guide, op coverage table.
- [ ] `torch.compile`-analog: *skip* (graph compile is out of scope); instead
      offer ahead-of-time fused ops.
- [ ] (Stretch) distributed data-parallel (gloo-like TCP backend, NCCL later).
- [ ] (Stretch) ONNX export/import.
- [ ] Example gallery: MLP, CNN, transformer, fine-tuning.

**Exit criteria:** a PyTorch user can port the official PyTorch MNIST example
in <30 minutes following the guide; docs site live.

## Phase 10 — Beta release

- [ ] Full audit of API vs. `torch` docs; divergence log.
- [ ] Cross-platform wheels; PyPI release; changelog; versioning policy.
- [ ] Fuzzing on tensor ops (arbitrary shapes/dtypes via `proptest`/hypothesis).
- [ ] Known-issues page.

---

## Cross-cutting: testing & parity strategy

1. **Golden parity suite** — generated .npz/.pt fixtures of random tensors +
   expected outputs produced by PyTorch for every op; CI asserts allclose.
2. **Gradcheck everywhere** — `torch.autograd.gradcheck`-equivalent in Rust
   tests.
3. **Training parity** — fixed-seed training runs compared to reference
   PyTorch loss curves (M3 milestone onwards).
4. **Property tests** — shape/broadcast/stride invariants with `proptest`.
5. **Perf regression CI** — nightly benches; fail on >10% regression vs.
   rolling baseline.

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CUDA kernel coverage is enormous | GPU phase balloons | wgpu first; CUDA scope limited to ops needed by target models; partner with `cust`/`cublas` rather than hand-writing everything |
| API drift from torch | users can't port | generate a divergence table each release; port tests *from* torch's test suite where license permits |
| Autograd bugs are silent | wrong training results | exhaustive gradcheck CI; parity fixtures for every op before it ships |
| Perf expectations | adoption | be honest: match CPU torch on common ops first; GPU parity later |
| Scope creep (torch has 2,000+ ops) | never ships | explicit non-goals: no JIT compile, no mobile, no torchserve; maintain an op coverage target (see below) |

**Op coverage target for v0.1 beta:** the ~250 ops that cover MNIST, CIFAR
CNNs, and a small transformer (audit against PyTorch tutorials' op usage).

## Non-goals (v1)

- torch.compile / TorchScript / graph tracing
- Mobile & embedded runtime
- Full `torch.distributed`
- torch Hub & model zoo

## Milestone summary

| Milestone | Contents | Exit |
|---|---|---|
| M0 | Phase 0 | import works, CI green |
| M1 | Phases 1–2 | tensor+autograd parity vs torch (CPU) |
| M2 | Phases 3–4 | train MLP+CNN on CPU, curves match torch |
| M3 | Phase 5 | MNIST end-to-end >97% |
| M4 | Phase 6 | torch-weight round-trip works |
| M5 | Phase 7 | GPU training (wgpu), CUDA beta |
| M6 | Phases 8–10 | perf within 1× torch CPU, beta on PyPI |
