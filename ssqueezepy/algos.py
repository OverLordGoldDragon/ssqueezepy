import numpy as np
from numba import njit
from functools import reduce


def find_closest(a, v):
    """Equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D.
    Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    """
    sidx = v.argsort()
    v_s = v[sidx]
    idx = np.searchsorted(v_s, a)
    idx[idx==len(v)] = len(v)-1
    idx0 = (idx-1).clip(min=0)

    m = np.abs(a-v_s[idx]) >= np.abs(v_s[idx0]-a)
    m[idx==0] = 0
    idx[m] -= 1
    out = sidx[idx]
    return out


@njit
def indexed_sum(a, k):
    out = np.zeros(a.shape, dtype=np.cfloat)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[k[i, j], j] += a[i, j]
    return out


def nCk(n, k):
    """Efficient n-Choose-k"""
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom

#### Replacers ###############################################################
def _process_replace_fn_args(x, ref):
    if ref is None:
        ref = x
    xndim = x.ndim  # store original ndim to undo expansion later
    if not (isinstance(x, np.ndarray) and isinstance(ref, np.ndarray)):
        raise TypeError("inputs must be numpy arrays "
                        "(got %s, %s)" % (type(x), type(ref)))
    while x.ndim < 3:
        x = np.expand_dims(x, -1)
    while ref.ndim < 3:
        ref = np.expand_dims(ref, -1)
    if x.ndim > 3 or ref.ndim > 3:
        raise ValueError("inputs must be 1D, 2D, or 3D numpy arrays "
                         "(got x.ndim==%d, ref.ndim==%d)" % (x.ndim, ref.ndim))
    return x, ref, xndim


def replace_at_inf_or_nan(x, ref=None, replacement=0.):
    x, ref, xndim = _process_replace_fn_args(x, ref)
    x = _replace_at_inf_or_nan(x, ref, replacement)
    while x.ndim > xndim:
        x = x.squeeze(axis=-1)
    return x

def replace_at_inf(x, ref=None, replacement=0.):
    x, ref, xndim = _process_replace_fn_args(x, ref)
    x = _replace_at_inf(x, ref, replacement)
    while x.ndim > xndim:
        x = x.squeeze(axis=-1)
    return x

def replace_at_nan(x, ref=None, replacement=0.):
    x, ref, xndim = _process_replace_fn_args(x, ref)
    x = _replace_at_nan(x, ref, replacement)
    while x.ndim > xndim:
        x = x.squeeze(axis=-1)
    return x

def replace_at_value(x, ref=None, value=0., replacement=0.):
    """Note: `value=np.nan` won't work (but np.inf will, separate from -np.inf).
    """
    x, ref, xndim = _process_replace_fn_args(x, ref)
    x = _replace_at_value(x, ref, value, replacement)
    while x.ndim > xndim:
        x = x.squeeze(axis=-1)
    return x

@njit
def _replace_at_inf_or_nan(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isinf(ref[i, j, k]) or np.isnan(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@njit
def _replace_at_inf(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isinf(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@njit
def _replace_at_nan(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isnan(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@njit
def _replace_at_value(x, ref, value=0., replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if ref[i, j, k] == value:
                    x[i, j, k] = replacement
    return x

#############################################################################
