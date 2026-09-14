//! The `Var` computation graph: nodes, edges, and the backward driver.
//!
//! Design (plan.md Phase 2):
//! - A [`Var`] is a `Tensor` plus provenance: which op made it and which
//!   `Var`s it consumed. Cloning a `Var` clones a graph *reference*, not
//!   the graph — diamonds in the DAG are shared, and gradient contributions
//!   accumulate into the same node.
//! - Each op records a **VJP closure** (vector–Jacobian product). Backward
//!   for a node is `vjp(grad_of_output) -> [grads for inputs]`, where the
//!   closures build new `Var` ops, so **double backward works for any op
//!   whose VJP is written from differentiable primitives**.
//! - `backward()` runs a topological sort (iterative, recursion-free) and
//!   accumulates `node.grad += upstream` for every reachable node.
//! - Gradients flowing into broadcast operands are summed back down to the
//!   operand's shape ([`unbroadcast`]) — the reverse of `broadcast_to`.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use oxi_core::{OxitorchError, OxitorchResult};
use oxi_tensor::Tensor;

pub(crate) type Vjp = Box<dyn Fn(&Var) -> OxitorchResult<Vec<Var>> + Send>;

// Thread-local grad mode (torch parity): when `false`, ops record no graph
// — their results come back detached and `backward()` cannot run through
// them. Each [`GradGuard`] saves the previous mode and restores it on drop,
// so nested `no_grad`/`enable_grad` scopes compose like torch's.
thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether ops executed on this thread currently record into the graph.
#[must_use]
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

/// Sets the grad mode for this thread, returning the previous mode so
/// callers can restore it (manual save/restore; prefer [`GradGuard`],
/// [`no_grad`], or [`enable_grad`]).
#[must_use]
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|c| c.replace(enabled))
}

/// Runs `f` with graph recording disabled: every op inside returns detached
/// values (torch's `no_grad`). Nesting composes by depth counting.
///
/// ```
/// # use oxi_autograd::graph::{no_grad, is_grad_enabled};
/// no_grad(|| {
///     assert!(!is_grad_enabled());
/// });
/// assert!(is_grad_enabled());
/// ```
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = GradGuard::pause();
    f()
}

/// Runs `f` with graph recording force-enabled — useful inside a `no_grad`
/// scope (torch's `enable_grad`). Nesting composes by depth counting.
///
/// ```
/// # use oxi_autograd::graph::{no_grad, enable_grad, is_grad_enabled};
/// no_grad(|| {
///     let _ = enable_grad(|| assert!(is_grad_enabled()));
///     assert!(!is_grad_enabled());
/// });
/// ```
pub fn enable_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = GradGuard::force();
    f()
}

/// RAII toggle for grad mode. Construction sets the mode; `Drop` restores
/// the previous one. Prefer [`no_grad`] / [`enable_grad`] unless the mode
/// must span unrelated calls (the Python bindings use this directly).
pub struct GradGuard {
    prev: bool,
}

impl GradGuard {
    /// Pauses graph recording until the guard is dropped.
    #[must_use = "the guard restores grad mode on drop"]
    pub fn pause() -> Self {
        Self::set(false)
    }

    /// Forces graph recording on until the guard is dropped.
    #[must_use = "the guard restores grad mode on drop"]
    pub fn force() -> Self {
        Self::set(true)
    }

    fn set(on: bool) -> Self {
        Self {
            prev: GRAD_ENABLED.with(|c| c.replace(on)),
        }
    }
}

impl Drop for GradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|c| c.set(self.prev));
    }
}

/// Shared, interior-mutable node payload.
///
/// `Arc` + `Mutex` (not `Rc` + `RefCell`) so `Var` is `Send + Sync` and can
/// live inside pyo3 `#[pyclass]` objects and cross the GIL boundary.
struct Node {
    /// The forward value.
    value: Tensor,
    /// Op name for error messages and debugging (`None` for leaves).
    op: Option<&'static str>,
    /// Inputs, in the order the VJP returns gradients.
    inputs: Vec<Var>,
    /// The VJP, if this node was produced by a differentiable op.
    vjp: Mutex<Option<Vjp>>,
    /// Accumulated gradient (`requires_grad` leaves and intermediates).
    grad: Mutex<Option<Tensor>>,
    /// Whether gradients should accumulate here (mirrors torch's leaf flag).
    requires_grad: bool,
}

/// A differentiable tensor handle. Cheap to clone (shared graph node).
#[derive(Clone)]
pub struct Var {
    node: Arc<Node>,
}

impl Var {
    /// Wraps a `Tensor` as a graph leaf.
    #[must_use]
    pub fn leaf(value: Tensor, requires_grad: bool) -> Self {
        Self {
            node: Arc::new(Node {
                value,
                op: None,
                inputs: Vec::new(),
                vjp: Mutex::new(None),
                grad: Mutex::new(None),
                requires_grad,
            }),
        }
    }

    /// Records an op node: `value = op(inputs...)`, with its VJP.
    ///
    /// When grad mode is off on this thread ([`no_grad`]), the result is a
    /// detached leaf instead: no inputs are stored, no VJP is kept, and the
    /// node does not require grad — matching torch.
    pub fn op(
        value: Tensor,
        op: &'static str,
        inputs: Vec<Var>,
        vjp: impl Fn(&Var) -> OxitorchResult<Vec<Var>> + Send + 'static,
    ) -> Self {
        if !is_grad_enabled() {
            return Self::leaf(value, false);
        }
        let requires_grad = inputs.iter().any(Var::requires_grad);
        Self {
            node: Arc::new(Node {
                value,
                op: Some(op),
                inputs,
                vjp: Mutex::new(Some(Box::new(vjp))),
                grad: Mutex::new(None),
                requires_grad,
            }),
        }
    }

    /// The forward value.
    #[must_use]
    pub fn value(&self) -> &Tensor {
        &self.node.value
    }

    /// Whether this node participates in autograd (leaf flag or any input
    /// requires grad).
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.node.requires_grad
    }

    /// The op name, or `None` for leaves.
    #[must_use]
    pub fn op_name(&self) -> Option<&'static str> {
        self.node.op
    }
    /// The accumulated gradient, if `backward()` has run and reached here.
    #[must_use]
    pub fn grad(&self) -> Option<Tensor> {
        self.node.grad.lock().expect("grad lock poisoned").clone()
    }

    /// Overwrites the accumulated gradient (gradient surgery: clipping,
    /// scaled/normalised seeds). Shape must match the node's value.
    ///
    /// # Errors
    /// Propagates shape validation of the replacement tensor.
    pub fn set_grad(&self, g: Tensor) -> OxitorchResult<()> {
        if g.shape() != self.node.value.shape() {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{:?}", g.shape()),
                rhs: format!("{:?}", self.node.value.shape()),
            });
        }
        *self.node.grad.lock().expect("grad lock poisoned") = Some(g);
        Ok(())
    }

    /// Discards the graph: returns the forward value and drops provenance.
    #[must_use]
    pub fn detach(&self) -> Tensor {
        self.value().clone()
    }

    /// Zeroes this node's gradient (a fresh `Tensor::zeros` of its shape).
    ///
    /// # Errors
    /// Propagates [`Tensor::zeros`].
    pub fn zero_grad(&self) -> OxitorchResult<()> {
        let shape: Vec<i64> = self.value().shape().iter().map(|&d| d as i64).collect();
        *self.node.grad.lock().expect("grad lock poisoned") = Some(Tensor::zeros(&shape)?);
        Ok(())
    }

    pub(crate) fn accumulate_grad(&self, g: Tensor) -> OxitorchResult<()> {
        let mut slot = self.node.grad.lock().expect("grad lock poisoned");
        match slot.take() {
            None => *slot = Some(g),
            Some(existing) => *slot = Some(existing.add(&g)?),
        }
        Ok(())
    }

    /// Runs VJP for this node given the upstream grad.
    ///
    /// # Errors
    /// [`OxitorchError::InvalidArgument`] if the VJP was already freed by a
    /// previous `backward()` without `retain_graph` — i.e. the caller tried
    /// to backward through a graph a second time.
    fn vjp_apply(&self, upstream: &Var) -> OxitorchResult<Vec<Var>> {
        let f = self
            .node
            .vjp
            .lock()
            .expect("vjp lock poisoned")
            .take()
            .ok_or_else(|| {
                OxitorchError::InvalidArgument(
                    "trying to backward through the graph a second time (or along an already freed path): pass retain_graph=True on the first backward call".into(),
                )
            })?;
        f(upstream)
    }
}

/// Sums `g` down to `shape` — the inverse of broadcasting.
///
/// Aligns `g`'s trailing dims with `shape` (prepending size-1 dims as
/// needed), then sums every axis where the target is 1.
///
/// # Errors
/// [`OxitorchError::ShapeMismatch`] if `g` cannot broadcast to the original
/// output shape implied by `shape`.
pub fn unbroadcast(g: &Tensor, shape: &[usize]) -> OxitorchResult<Tensor> {
    let g_shape = g.shape();
    // Trailing alignment: g may be longer (broadcast up) or equal; it must
    // be broadcast-compatible when right-aligned.
    if g_shape.len() < shape.len() {
        return Err(OxitorchError::ShapeMismatch {
            lhs: format!("{g_shape:?}"),
            rhs: format!("{shape:?}"),
        });
    }
    let extra = g_shape.len() - shape.len();
    for (i, &s) in shape.iter().enumerate() {
        let gi = g_shape[extra + i];
        if gi != s && s != 1 {
            return Err(OxitorchError::ShapeMismatch {
                lhs: format!("{g_shape:?}"),
                rhs: format!("{shape:?}"),
            });
        }
    }

    // Sum every axis the forward broadcast expanded: prepended axes first
    // (each contributes one reduction to a size-1 head dim), then size-1
    // target dims whose g dim is larger.
    let mut t = g.clone();
    for i in 0..extra {
        t = t.reduce(oxi_tensor::ops::ReduceKind::Sum, i as i64, true)?;
    }
    for (i, &s) in shape.iter().enumerate() {
        if s == 1 && t.shape()[i] != 1 {
            t = t.reduce(oxi_tensor::ops::ReduceKind::Sum, i as i64, true)?;
        }
    }
    // Reshape from keepdim form to the exact target shape (drops the
    // prepended size-1 dims).
    if t.shape() != shape {
        t = t.reshape(&shape.iter().map(|&d| d as i64).collect::<Vec<_>>())?;
    }
    Ok(t)
}

/// Iterative DFS frame: first visit (push inputs), then emit (post-order).
enum Frame {
    /// First visit of a node.
    Enter(Var),
    /// All inputs processed; emit in post-order.
    Exit(Var),
}

/// Iterative topological sort from `root` following input edges.
fn topo_order(root: &Var) -> Vec<Var> {
    let mut order = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack = vec![Frame::Enter(root.clone())];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(v) => {
                let key = Arc::as_ptr(&v.node) as usize;
                if !visited.insert(key) {
                    continue;
                }
                stack.push(Frame::Exit(v.clone()));
                let inputs = v.node.inputs.clone();
                for input in &inputs {
                    let ik = Arc::as_ptr(&input.node) as usize;
                    if !visited.contains(&ik) {
                        stack.push(Frame::Enter(input.clone()));
                    }
                }
            }
            Frame::Exit(v) => order.push(v),
        }
    }
    order
}

/// Backward pass: seeds `root`'s gradient with `seed` (default all-ones) and
/// accumulates gradients into every reachable `requires_grad` node.
///
/// By default VJPs are consumed as they run, so a second `backward` through
/// the same graph fails (torch's behavior); pass `retain_graph` via
/// [`backward_ext`] to keep them.
///
/// # Errors
/// Propagates VJP errors; [`OxitorchError::InvalidArgument`] if the root
/// does not require grad or is not a scalar when seeded implicitly.
pub fn backward(root: &Var, seed: Option<Tensor>) -> OxitorchResult<()> {
    backward_ext(root, seed, false)
}

/// `backward` with graph retention control. With `retain_graph = true` the
/// recorded VJPs survive the pass, so further `backward()` calls through the
/// same graph work (contributions accumulate into `.grad`); the graph is
/// then freed by the first backward *without* retention.
///
/// # Errors
/// Propagates VJP errors; [`OxitorchError::InvalidArgument`] for a
/// non-requiring-grad root, a non-scalar implicit seed, or a second backward
/// over an already-freed graph.
pub fn backward_ext(root: &Var, seed: Option<Tensor>, retain_graph: bool) -> OxitorchResult<()> {
    if !root.requires_grad() {
        return Err(OxitorchError::InvalidArgument(
            "element 0 of tensors does not require grad and does not have a grad_fn".into(),
        ));
    }
    let seed = match seed {
        Some(s) => s,
        None if root.value().numel() == 1 => Tensor::ones_like(&root.value().clone())?,
        None => {
            return Err(OxitorchError::InvalidArgument(
                "grad can be implicitly created only for scalar outputs; pass a seed tensor".into(),
            ))
        }
    };
    let order = topo_order(root);
    // Intermediate grads are transient (torch parity): clear anything left
    // over from earlier passes so seeds and contributions don't compound.
    // Leaf grads persist — they accumulate across `backward()` calls.
    for node in &order {
        if node.node.op.is_some() {
            *node.node.grad.lock().expect("grad lock poisoned") = None;
        }
    }
    root.accumulate_grad(seed)?;
    for node in order.into_iter().rev() {
        if node.node.op.is_none() {
            continue; // true leaf: nothing to propagate
        }
        let upstream = node
            .node
            .grad
            .lock()
            .expect("grad lock poisoned")
            .clone()
            .ok_or_else(|| {
                OxitorchError::InvalidArgument(
                    "internal: node reached in backward without a gradient".into(),
                )
            })?;
        let upstream_var = Var::leaf(upstream, false);
        let input_grads = if retain_graph {
            // Borrow instead of take: the VJP stays available for further
            // backward passes. Only called for op nodes (leaves skipped).
            // The guard must outlive the call, hence the explicit binding.
            let guard = node.node.vjp.lock().expect("vjp lock poisoned");
            let f = guard.as_ref().ok_or_else(|| {
                OxitorchError::InvalidArgument(
                    "internal: op node reached in backward without a VJP".into(),
                )
            })?;
            f(&upstream_var)?
        } else {
            node.vjp_apply(&upstream_var)?
        };
        let inputs = node.node.inputs.clone();
        for (input, g) in inputs.iter().zip(input_grads) {
            if !input.requires_grad() {
                continue;
            }
            let g = unbroadcast(g.value(), input.value().shape())?;
            input.accumulate_grad(g)?;
        }
    }
    Ok(())
}

// The nodes are shared via `Arc` and mutated only under short-lived locks;
// safe Send/Sync for use inside pyo3 classes.
const _: () = {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    let _ = assert_send::<Var>;
    let _ = assert_sync::<Var>;
};

/// A leaf that gradients flow into; the usual model-parameter wrapper.
impl From<Tensor> for Var {
    fn from(value: Tensor) -> Self {
        Var::leaf(value, true)
    }
}

impl std::fmt::Debug for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = self.node.op;
        let shape = self.node.value.shape().to_vec();
        match op {
            Some(op) => write!(f, "Var({op}, shape={shape:?})"),
            None => write!(f, "Var(leaf, shape={shape:?})"),
        }
    }
}
