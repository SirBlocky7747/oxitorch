# Error-handling convention

This is the convention decided in Phase 0 (plan.md: *"Decide error-handling
convention: Rust `Result` internally → Python exceptions at the binding
layer"*) and applies to every crate in the workspace.

## Rust side

All engine crates (`oxi-core`, `oxi-tensor`, `oxi-autograd`, `oxi-nn`,
`oxi-optim`, `oxi-data`, `oxi-io`) return
`oxi_core::OxitorchResult<T> = Result<T, OxitorchError>`. Panics are reserved
for internal invariants only; any condition a user can trigger through the
Python API must be an `OxitorchError`.

The variants live in `crates/oxi-core/src/lib.rs`:

| Variant            | Meaning                                                        |
|--------------------|----------------------------------------------------------------|
| `ShapeMismatch`    | Operand shapes do not agree (e.g. for a binary op).             |
| `InvalidArgument`  | Argument structurally wrong (bad dtype name, negative dim...).  |
| `NotImplemented`   | Planned feature not yet implemented (e.g. GPU devices).         |
| `OutOfBounds`      | Op would access memory outside a tensor's storage.              |
| `Other`            | Anything that does not fit a specific variant.                  |

`unsafe_code = "deny"` is set workspace-wide: unsafe code is a compile error
unless a targeted `#[allow(unsafe_code)]` opts out with a `SAFETY:` comment.
The single sanctioned block today is the GEMM call in `oxi-tensor`'s
`Tensor::matmul` (third-party BLAS kernels are inherently unsafe); any new
unsafe block needs the same treatment and review.

## Python side

`oxi-bindings` performs the translation exactly once, at the boundary, in
`crates/oxi-bindings/src/error.rs`:

| Rust variant                        | Python exception       |
|-------------------------------------|------------------------|
| `InvalidArgument` / `ShapeMismatch` | `ValueError`           |
| `NotImplemented`                    | `NotImplementedError`  |
| everything else                     | `RuntimeError`         |

The error's `Display` message is preserved verbatim as the Python exception
message, and mirrors `torch`'s behavior where user errors are `ValueError` /
`RuntimeError` (matching the table in plan.md's Phase 0 checklist).

Rules:

1. **No panics across FFI.** Binding functions convert results with
   `.map_err(error::to_pyerr)?` before anything can unwind.
2. **No logic in bindings.** `oxi-bindings` never invents error text or
   semantics; it only maps variants to exception classes.
3. **Exhaustiveness is enforced.** `error.rs` contains a test constructing
   every variant; adding a new variant without updating the mapping breaks
   the build.
4. **Messages are user-facing.** Error `Display` strings are shown to Python
   users directly — write them as complete sentences with the offending
   values included (shapes, dtypes, indices).
