# -*- coding: utf-8 -*-
import numpy as np
try:
    import torch
    import cupy as cp
except:
    get_torch_dummy = lambda: TorchDummy()
    get_cupy_dummy  = lambda: CupyDummy()
    torch = get_torch_dummy()
    cp    = get_cupy_dummy()


def allclose(a, b, device='cuda'):
    """`numpy.allclose` or `torch.allclose`, latter if input(s) are Tensor."""
    if is_tensor(a, b, mode='any'):
        a, b = asarray(a, device=device), asarray(b, device=device)
        return torch.allclose(a, b)
    return np.allclose(a, b)


def astype(x, dtype):
    if is_tensor(x):
        return x.type(_torch_dtype(dtype))
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


def is_dtype(x, str_dtype):
    if not isinstance(str_dtype, (list, tuple)):
        str_dtype = [str_dtype]
    return any(str(x).split('.')[-1] == dtype for dtype in str_dtype)


def atleast_1d(x, device='cuda'):
    return Q.atleast_1d(asarray(x, device=device))


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


def _torch_dtype(dtype):
    return dtype if not isinstance(dtype, str) else getattr(torch, dtype)


class TorchDummy():
    """Dummy class with dummy attributes."""
    def __init__(self):
        self.Tensor = __TensorDummy


class __TensorDummy():
    pass

class _Util():
    """For wrapper: `@cp._util.memoize`."""
    def memoize(self, fn):
        return fn


class CupyDummy():
    """Dummy class with dummy attributes."""
    def __init__(self):
        self._util = _Util()


class _Q():
    """Class for accessing `numpy` or `torch` attributes according to `USE_GPU()`.
    """
    def __getattr__(self, name):
        if USE_GPU():
            return getattr(torch, name)
        return getattr(np, name)

Q = _Q()

##############################################################################
from ..configs import USE_GPU
