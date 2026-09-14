"""Module base classes: Parameter, Module, Sequential.

plan.md Phase 3 decision: a *Python-side* Module system in v1 — faster to
ship — holding Rust `Tensor`s as state and composing the differentiable ops
exposed by `_native`. `Module.named_parameters()` mirrors torch's ordering
(recursion over registered submodules, insertion order preserved).
"""

from . import functional as F


class Parameter:
    """A named, trainable tensor (thin wrapper marking `requires_grad`)."""

    def __init__(self, tensor):
        self.tensor = tensor  # oxitorch.Tensor with requires_grad=True

    @property
    def grad(self):
        return self.tensor.grad

    def zero_grad(self):
        self.tensor.zero_grad()

    def detach_numpy(self):
        return self.tensor.detach().numpy()


class Module:
    """Base class: collect Parameters from attributes and submodules."""

    def parameters(self):
        """All parameters of this module and its submodules, in order."""
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Parameter):
                        params.append(item)
        return params

    def named_parameters(self, prefix=""):
        """Yield `(name, parameter)` pairs, torch-style dotted paths."""
        for name, v in self.__dict__.items():
            if isinstance(v, Parameter):
                yield prefix + name, v
            elif isinstance(v, Module):
                yield from v.named_parameters(prefix + name + ".")
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")
                    elif isinstance(item, Parameter):
                        yield f"{prefix}{name}.{i}", item

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        """Sets the training flag on this module and all submodules."""
        self.training = mode
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train(mode)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        """Flat `{name: numpy array}` snapshot of all parameters."""
        return {name: p.tensor.detach().numpy().copy() for name, p in self.named_parameters()}

    def load_state_dict(self, sd):
        """Copy values from a `state_dict()` snapshot into the parameters."""
        for name, p in self.named_parameters():
            if name not in sd:
                raise KeyError(f"missing key in state_dict: {name!r}")
            arr = np_asarray(sd[name])
            if list(arr.shape) != list(p.tensor.shape):
                raise ValueError(
                    f"shape mismatch for {name!r}: state {list(arr.shape)} "
                    f"vs parameter {list(p.tensor.shape)}"
                )
            # Replace via from_numpy keeps requires_grad; values are copied.
            new_t = F._native_from_numpy(arr, requires_grad=True)
            p.tensor = new_t

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def np_asarray(arr):
    import numpy as np

    return np.asarray(arr)


class Sequential(Module):
    """Chains modules: `Sequential(Linear(...), ReLU(), ...)`."""

    def __init__(self, *modules):
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, idx):
        return self.layers[idx]
