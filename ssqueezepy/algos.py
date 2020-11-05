import numpy as np
from numba import jit


def find_closest(a, v):
    # equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D
    # credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
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


@jit(nopython=True)
def indexed_sum(a, k):
    out = np.zeros(a.shape, dtype=np.cfloat)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[k[i, j], j] += a[i, j]
    return out


@jit(nopython=True)
def replace_at_inf_or_nan(x, ref, replacement=0):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if np.isinf(ref[i, j]) or np.isnan(ref[i, j]):
                x[i, j] = replacement
    return x
