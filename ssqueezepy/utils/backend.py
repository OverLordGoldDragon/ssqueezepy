# -*- coding: utf-8 -*-
import numpy as np
# torch & cupy imported at bottom


def allclose(a, b, device='cuda'):
    """`numpy.allclose` or `torch.allclose`, latter if input(s) are Tensor."""
    if is_tensor(a, b, mode='any'):
        a, b = asarray(a, device=device), asarray(b, device=device)
        return torch.allclose(a, b)
    return np.allclose(a, b)


def astype(x, dtype, device='cuda'):
    if is_tensor(x):
        return x.to(dtype=_torch_dtype(dtype))
    return x.astype(dtype)


def array(x, dtype=None, device='cuda'):
    if USE_GPU():
        return torch.tensor(x, dtype=_torch_dtype(dtype), device=device)
    return np.array(x)


def asarray(x, dtype=None, device='cuda'):
    if USE_GPU():
        return torch.as_tensor(x, dtype=_torch_dtype(dtype), device=device)
    return np.asarray(x, dtype=dtype)


def zeros(shape, dtype=None, device='cuda'):
    if USE_GPU():
        return torch.zeros(shape, dtype=_torch_dtype(dtype), device=device)
    return np.zeros(shape, dtype=dtype)


def ones(shape, dtype=None, device='cuda'):
    if USE_GPU():
        return torch.ones(shape, dtype=_torch_dtype(dtype), device=device)
    return np.ones(shape, dtype=dtype)


def is_tensor(*args, mode='all'):
    cond = all if mode == 'all' else any
    return cond(isinstance(x, torch.Tensor) for x in args)


def is_array_or_tensor(*args, mode='all'):
    cond = all if mode == 'all' else any
    return cond(isinstance(x, (torch.Tensor, np.ndarray)) for x in args)


def is_dtype(x, str_dtype):
    return (str_dtype in str(x.dtype) if isinstance(str_dtype, str) else
            any(sd in str(x.dtype) for sd in str_dtype))


def atleast_1d(x, dtype=None, device='cuda'):
    return Q.atleast_1d(asarray(x, dtype=dtype, device=device))


def asnumpy(x):
    if is_tensor(x):
        return x.cpu().numpy()
    return x


def arange(a, b=None, dtype=None, device='cuda'):
    if b is None:
        a, b = 0, a
    if USE_GPU():
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        return torch.arange(a, b, dtype=dtype, device=device)
    return np.arange(a, b, dtype=dtype)


def vstack(x):
    if is_tensor(x) or (isinstance(x, list) and is_tensor(x[0])):
        if isinstance(x, list):
            # stack arrays as elements in extended dim0
            return torch.vstack([_x[None] for _x in x])
        return torch.vstack(x)
    return np.vstack([x])


#### misc + dummies ##########################################################
def warn_if_tensor_and_par(x, parallel):
    if parallel and is_tensor(x):
        from .common import WARN
        WARN("`parallel` ignored with tensor input.")


def _torch_dtype(dtype):
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    elif isinstance(dtype, np.dtype):
        return getattr(torch, str(dtype).split('.')[-1])
    return dtype  # assume torch.dtype


class _TensorDummy():
    pass


class TorchDummy():
    """Dummy class with dummy attributes."""
    def __init__(self):
        self.Tensor = _TensorDummy
        self.dtype = _TensorDummy


class CupyDummy():
    """Dummy class with dummy attributes."""
    def memoize(self, *args, **kwargs):
        def wrap(fn):
            return fn
        return wrap


class _Q():
    """Class for accessing `numpy` or `torch` attributes according to `USE_GPU()`.
    """
    def __getattr__(self, name):
        if USE_GPU():
            return getattr(torch, name)
        return getattr(np, name)


##############################################################################
Q = _Q()

try:
    import torch
    import cupy as cp
except:
    torch = TorchDummy()
    cp    = CupyDummy()

from ..configs import USE_GPU
