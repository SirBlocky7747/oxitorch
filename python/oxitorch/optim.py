"""Optimizers & training-loop machinery (Phase 4).

The optimizer *math and state* (velocity/momentum buffers, Adam moments,
per-parameter step counters) live in Rust (`oxi-autograd::optim`, exposed
through `_native.RustSgd` / `_native.RustAdamW`): each `step()` hands over
the current parameter tensors and every update happens on flat f32 slices
in the engine — no numpy round-trip, and the `Tensor` objects keep their
identity across the whole run. The Python classes below are thin delegates
mirroring torch's API (`zero_grad()` / `step()`, `lr` mutable for
schedulers). Schedulers and gradient clipping are Python-side and compose
with the Rust optimizers unchanged.
"""

import numpy as np

from . import _native


class _RustOptimizer:
    """Shared plumbing for optimizers whose math lives in Rust.

    Owns no tensor state: the native optimizer holds all buffers. Each call
    re-collects the *current* parameter tensors (they may have been rebound
    by `load_state_dict` or user code since construction) and hands them to
    the engine, which rewrites each tensor's value in place.
    """

    def __init__(self, params, native):
        self.params = list(params)
        self._native = native

    @staticmethod
    def _tensors(params):
        """Accepts `Parameter` wrappers and bare tensors (duck-typed)."""
        return [p.tensor if hasattr(p, "tensor") else p for p in params]

    @property
    def lr(self):
        return self._native.lr

    @lr.setter
    def lr(self, value):
        self._native.lr = value

    def zero_grad(self):
        self._native.zero_grad(self._tensors(self.params))

    def step(self):
        self._native.step(self._tensors(self.params))


class Sgd(_RustOptimizer):
    """Vanilla SGD with optional (Nesterov) momentum — Rust-backed.

    Velocity buffers are initialized lazily from each parameter's first
    gradient (torch semantics), so a parameter that starts with `grad=None`
    gets no stale momentum.
    """

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False):
        if nesterov and momentum <= 0.0:
            raise ValueError("nesterov momentum requires momentum > 0")
        self.momentum = momentum
        self.nesterov = nesterov
        super().__init__(params, _native.RustSgd(lr, momentum, nesterov))


class AdamW(_RustOptimizer):
    """AdamW (Loshchilov & Hutter, 2019): Adam with decoupled weight decay.

    The decay term `weight_decay * param` is applied *directly* to the
    parameter (not through the adaptive moments), which is what modern
    training loops expect and what `torch.optim.AdamW` does. Hyperparameter
    defaults match torch: `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`,
    `weight_decay=1e-2`. Step counters are per-parameter: a parameter that
    skips steps (grad stays `None`) does not advance its bias correction.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(params, _native.RustAdamW(lr, self.betas, eps, weight_decay, True))


class Adam(_RustOptimizer):
    """Adam (Kingma & Ba, 2015) with torch's default hyperparameters.

    L2-style weight decay folded into the gradient (`weight_decay * param`
    is *added to the grad* before the moment updates) — the classic Adam
    formulation. Use `AdamW` for the modern decoupled variant. Defaults:
    `weight_decay=0.0` (decay is opt-in here, unlike AdamW).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        super().__init__(params, _native.RustAdamW(lr, self.betas, eps, weight_decay, False))


# ---- learning-rate schedulers (Phase 4) ---------------------------------------

class _LRScheduler:
    """Base scheduler: owns the optimizer, tracks `last_epoch`, and updates
    `optimizer.lr` on each `step()`. Subclasses implement `_lr_factor`
    (multiplier on the optimizer's *initial* lr at epoch `t`).

    Composes like torch's `SequentialLR`-style chaining: a warmup scheduler
    can be followed by a cosine one because each computes from the base lr
    and this base class just applies the newest factor.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = last_epoch
        # torch semantics: constructing the scheduler already performs one
        # `step()`, setting the lr for epoch 0.
        self.step()

    def get_last_lr(self):
        return [self.optimizer.lr]

    def _lr_factor(self, t):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.base_lr * self._lr_factor(self.last_epoch)


class StepLR(_LRScheduler):
    """Multiply the lr by `gamma` every `step_size` epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _lr_factor(self, t):
        return self.gamma ** (t // self.step_size)


class CosineAnnealingLR(_LRScheduler):
    """Cosine anneal from the base lr to `eta_min` over `T_max` epochs
    (Loshchilov & Hutter's SGDR schedule, without restarts).
    """

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _lr_factor(self, t):
        if self.T_max <= 0:
            return 1.0
        cos = (1.0 + np.cos(np.pi * t / self.T_max)) / 2.0
        return (self.eta_min + (self.base_lr - self.eta_min) * cos) / self.base_lr


class LinearWarmup(_LRScheduler):
    """Linearly ramp the lr from 0 to the base lr over `warmup_epochs`, then
    hold it at the base lr (compose with a decay scheduler via chaining).
    """

    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def _lr_factor(self, t):
        if self.warmup_epochs == 0:
            return 1.0
        return min(1.0, (t + 1) / self.warmup_epochs)


class SequentialLR(_LRScheduler):
    """Chain schedulers over epoch ranges, torch-style: `milestones` are the
    epochs at which each successive scheduler takes over. Each sub-scheduler
    restarts at its own epoch 0 on takeover (`local_t = t - boundary`).

    Sub-scheduler `base_lr`s are rebased to the first scheduler's base at
    construction, so construction order never pollutes the schedule (the
    warmup's ctor step would otherwise shrink the cosine's base).
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        if len(schedulers) != len(milestones) + 1:
            raise ValueError(
                f"{len(schedulers)} schedulers need {len(schedulers) - 1} "
                f"milestones, got {len(milestones)}"
            )
        self.optimizer = optimizer
        self.base_lr = schedulers[0].base_lr
        self.schedulers = schedulers
        for s in schedulers:
            s.base_lr = self.base_lr
        self.milestones = list(milestones)
        self.last_epoch = last_epoch
        self.step()

    def step(self):
        self.last_epoch += 1
        t = self.last_epoch
        idx = 0
        for m in self.milestones:
            if t >= m:
                idx += 1
        local_t = t if idx == 0 else t - self.milestones[idx - 1]
        self.optimizer.lr = self.base_lr * self.schedulers[idx]._lr_factor(local_t)


class LambdaLR(_LRScheduler):
    """Arbitrary schedule: lr = base_lr * `lr_lambda(epoch)`."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def _lr_factor(self, t):
        return self.lr_lambda(t)


# ---- gradient clipping (Phase 4) ----------------------------------------------

def _grads(params):
    for p in params:
        if p.grad is not None:
            yield p, p.grad.numpy()


def _set_grad(p, new_np):
    """Replace a parameter's gradient in place (tensor-level rebinding;
    `Parameter.grad` is a read-only view over `p.tensor.grad`)."""
    p.tensor.grad = _native.from_numpy(
        np.ascontiguousarray(new_np, dtype=np.float32), requires_grad=False
    )


def clip_grad_norm_(params, max_norm, norm_type=2.0):
    """Clip gradient norm in place; returns the *pre-clip* total norm (float).

    torch semantics: if the total norm exceeds `max_norm`, all grads are
    scaled by `max_norm / (total_norm + 1e-6)`; otherwise they are left
    untouched. `norm_type` may be any p >= 1 (inf for max-norm).
    """
    params = list(params)
    if norm_type == float("inf"):
        total = max(
            (float(np.max(np.abs(g))) for _, g in _grads(params)), default=0.0
        )
        scale = max_norm / (total + 1e-6)
    else:
        total = sum(
            float(np.sum(np.abs(g) ** norm_type)) for _, g in _grads(params)
        ) ** (1.0 / norm_type)
        scale = max_norm / (total + 1e-6)
    if scale < 1.0:
        for p, g in _grads(params):
            _set_grad(p, g * scale)
    return total


def clip_grad_value_(params, clip_value):
    """Clamp every gradient element to `[-clip_value, clip_value]` in place."""
    clip_value = float(clip_value)
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")
    for p, g in _grads(list(params)):
        _set_grad(p, np.clip(g, -clip_value, clip_value))
