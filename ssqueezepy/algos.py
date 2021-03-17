# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange
from functools import reduce
from collections import namedtuple
from string import Template
from .utils.backend import asnumpy, cp, torch
from .utils import backend as S


def nCk(n, k):
    """n-Choose-k"""
    mul = lambda a, b: a * b
    r = min(k, n - k)
    numer = reduce(mul, range(n, n - r, -1), 1)
    denom = reduce(mul, range(1, r + 1), 1)
    return numer / denom

#### `indexed_sum` ###########################################################
def indexed_sum(a, k, parallel=True):
    """Sum `a` into rows of 2D array according to indices given by 2D `k`"""
    out = np.zeros(a.shape, dtype=a.dtype)
    if parallel:
        return _indexed_sum_par(a, k, out)
    return _indexed_sum(a, k, out)


@jit(nopython=True, cache=True)
def _indexed_sum(a, k, out):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[k[i, j], j] += a[i, j]
    return out

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_par(a, k, out):
    for j in prange(a.shape[1]):
        for i in range(a.shape[0]):
            out[k[i, j], j] += a[i, j]
    return out


def _process_ssq_params(Wx, w_or_dWx, ssq_freqs, const, logscale, gamma,
                        parallel, flipud, out, complex_out):
    gpu = S.is_tensor(Wx)

    # process `Wx`, `w_or_dWx`, `out`
    if out is None:
        out_shape = (*Wx.shape, 2) if complex_out else Wx.shape
        if gpu:
            out = torch.zeros(out_shape, dtype=Wx.dtype, device=Wx.device)
        else:
            out = np.zeros(out_shape, dtype=Wx.dtype)
    elif complex_out:
        out = torch.view_as_real(out)
    if gpu:
        Wx = torch.view_as_real(Wx)
        w_or_dWx = torch.view_as_real(w_or_dWx)

    # process `const`
    if not isinstance(const, (np.ndarray, torch.Tensor)) or len(const) != len(Wx):
        if gpu:
            const_arr = torch.tensor(len(Wx) * [const], dtype=Wx.dtype,
                                     device=Wx.dtype)
        else:
            const_arr = np.array(len(Wx) * [const])
    elif gpu and isinstance(const, np.ndarray):
        const_arr = torch.as_tensor(const, dtype=Wx.dtype, device=Wx.device)
    const_arr = const_arr.squeeze()

    # process other constants
    if logscale:
        _, params = _get_params_find_closest_log(ssq_freqs)
    else:
        params = dict(vmin=ssq_freqs[0], dv=(ssq_freqs[1] - ssq_freqs[0]))

    if gpu:
        # process kernel params
        (blockspergrid, threadsperblock, kernel_kw, str_dtype
         ) = _get_kernel_params(Wx, dim=1)
        M = kernel_kw['M']
        kernel_kw.update(dict(f='f' if kernel_kw['dtype'] == 'float' else '',
                              extra=f"k = {M} - 1 - k;" if flipud else ""))

        # collect tensors & constants
        if 'idx1' in params:
            params['idx1'] = int(params['idx1'])
        kernel_args = [Wx.data_ptr(), w_or_dWx.data_ptr(), out.data_ptr(),
                       const_arr.data_ptr(), *list(params.values())]
        if gamma is not None:
            kernel_args.insert(4, cp.asarray(gamma, dtype=str_dtype))
    else:
        params.update(dict(const=const_arr, flipud=flipud, omax=len(out) - 1))

    if gpu:
        return Wx, w_or_dWx, out, kernel_args, kernel_kw
    else:
        return Wx, w_or_dWx, out, params


def indexed_sum_onfly(Wx, w, v, const=1, logscale=False, parallel=None,
                      gpu=None, flipud=False, out=None):
    """`indexed_sum` and `find_closest` within same loop, sparing an array;
    see `help(algos.find_closest)` on how `k` is computed.
    """
    if gpu and parallel:
        WARN("`gpu` overrides `parallel`.")
        parallel = False
    fn_num = 2 if gpu else (1 if parallel else 0)

    if out is None:
        if gpu:
            out = torch.zeros(Wx.shape, dtype=Wx.dtype, device='cuda')
        else:
            out = np.zeros(Wx.shape, dtype=Wx.dtype)

    if not isinstance(const, (np.ndarray, torch.Tensor)) or len(const) != len(Wx):
        const = np.array(len(Wx) * [const]).squeeze()
    const = const.squeeze()
    if not gpu:
        const = const.astype(Wx.dtype)

    if logscale:  # TODO move
        idx, params = _get_params_find_closest_log(v)
        if idx is None:
            fns = (_indexed_sum_log, _indexed_sum_log_par, _indexed_sum_log_gpu)
        else:
            fns = (_indexed_sum_log_piecewise, _indexed_sum_log_piecewise_par,
                   _indexed_sum_log_piecewise_gpu)
    else:
        params = dict(vmin=v[0], dv=(v[1] - v[0]))
        fns = (_indexed_sum_lin, _indexed_sum_lin_par, _indexed_sum_lin_gpu)
    params['const'] = const
    params['flipud'] = flipud
    if not gpu:
        params['omax'] = len(out) - 1
    elif 'idx1' in params:
        params['idx1'] = int(params['idx1'])

    fn = fns[fn_num]
    fn(Wx, w, out, **params)
    return out


@jit(nopython=True, cache=True)
def _indexed_sum_log(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_par(Wx, w, out, const, vlmin, dvl, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = int(min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

def _indexed_sum_log_gpu(Wx, w, out, const, vlmin, dvl, flipud=False):
    kernel = '''
    extern "C" __global__
    void indexed_sum_log(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vlmin, double dvl)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)log2${f}(w[i][j]) - vlmin) / dvl);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    '''
    return _gpu_run_indexed_sum(kernel, Wx, w, out, const, vlmin, dvl,
                                flipud=flipud)


@jit(nopython=True, cache=True)
def _indexed_sum_log_piecewise(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                               idx1, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = min(round((wl - vlmin1) / dvl1) + idx1, omax)
            else:
                k = round(max((wl - vlmin0) / dvl0, 0))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_piecewise_par(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                                   idx1, omax, flipud=False):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = min(round((wl - vlmin1) / dvl1) + idx1, omax)
            else:
                k = round(max((wl - vlmin0) / dvl0, 0))
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

def _indexed_sum_log_piecewise_gpu(Wx, w, out, const, vlmin0, vlmin1,
                                   dvl0, dvl1, idx1, flipud=False):
    kernel = '''
    extern "C" __global__
    void indexed_sum_log_piecewise(${dtype} Wx[${M}][${N}][2],
                                   ${dtype} w[${M}][${N}],
                                   ${dtype} out[${M}][${N}][2],
                                   ${dtype} const_arr[${M}],
                                   double vlmin0, double vlmin1,
                                   double dvl0, double dvl1,
                                   int idx1)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      double wl;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          wl = (double)log2${f}(w[i][j]);

          if (wl > vlmin1){
              k = (int)round((wl - vlmin1) / dvl1) + idx1;
              if (k >= ${M})
                  k = ${M} - 1;
          } else {
              k = (int)round((wl - vlmin0) / dvl0);
              if (k < 0)
                  k = 0;
          }
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    '''
    return _gpu_run_indexed_sum(kernel, Wx, w, out, const, vlmin0, vlmin1,
                                dvl0, dvl1, idx1, flipud=flipud)


@jit(nopython=True, cache=True)
def _indexed_sum_lin(Wx, w, out, const, vmin, dv, omax, flipud=False):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((w[i, j] - vmin) / dv, 0)), omax)
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_lin_par(Wx, w, out, const, vmin, dv, omax, flipud=False):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((w[i, j] - vmin) / dv, 0)), omax)
            if flipud:
                k = omax - k
            out[k, j] += Wx[i, j] * const[i]

def _indexed_sum_lin_gpu(Wx, w, out, const, vmin, dv, flipud=False):
    kernel = '''
    extern "C" __global__
    void indexed_sum_lin(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vmin, double dv)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)(w[i][j]) - vmin) / dv);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    '''
    return _gpu_run_indexed_sum(kernel, Wx, w, out, const, vmin, dv,
                                flipud=flipud)

#### `find_closest` algorithms ###############################################
def find_closest(a, v, logscale=False, parallel=None, smart=None):
    """`argmin(abs(a[i, j] - v)) for all `i, j`; `a` is 2D, `v` is 1D.

    # Arguments:
        a: np.ndarray
            2D array.

        v: np.ndarray
            1D array.

        logscale: bool (default False)
            Whether "closest" is taken in linear or logarithmic space.

        parallel: bool (default True) / None
            Whether to use algorithms with `numba.jit(parallel=True)`

        smart: bool (default False) / None
            Whether to use a very fast smart algorithm (but still the slowest
            for ssqueezing; see usage guide below).
            Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    ____________________________________________________________________________
    **Usage guide**

    If 100% accuracy is desired, or `v` is not linearly or logarithmically
    distributed, use `find_closest_smart` (`smart=True`) or `find_closest_brute`
    (not callable from here).
        `_smart` is faster on single CPU thread, but `_brute` can win
        via parallelism.

    Else, `find_closest_lin` and `find_closest_log` do the trick (the special
    case of log-piecewise is handled), and are much faster.
        - Relative to "exact", they differ only by 0% to 0.0001%, purely per
        float precision limitations, and never by more than one index in `out`
        (where  whether e.g. `w=0.500000001` belongs to 0 or 1 isn't statistically
        meaningful to begin with).

    ____________________________________________________________________________
    **How it works:** `find_closest_log`, `find_closest_lin`

    The root assumption is that `v` is uniformly (in linear or log space)
    distributed, and we calculate analytically in which bin `w` will land as:
        `(w - bin_min) / bin_step_size`
    Above is forced to bound in [0, len(v) - 1].
    """
    if smart is None and parallel is None:
        parallel = True
    elif parallel and smart:
        WARN("find_closest: `smart` overrides `parallel`")

    if smart:
        if logscale:
            out = find_closest_smart(np.log2(a), np.log2(v))
        else:
            out = find_closest_smart(a, v)
    elif logscale:
        out = find_closest_log(a, v, parallel=parallel)
    else:
        out = find_closest_lin(a, v, parallel=parallel)
    return out


@jit(nopython=True, cache=True, parallel=True)
def find_closest_brute(a, v):
    """Computes exactly but exhaustively."""
    out = np.zeros(a.shape, dtype=np.int32)
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            out[i, j] = np.argmin(np.abs(a[i, j] - v))
    return out


def find_closest_smart(a, v):
    """Equivalent to argmin(abs(a[i, j] - v)) for all i, j; a is 2D, v is 1D.
    Credit: Divakar -- https://stackoverflow.com/a/64526158/10133797
    """
    sidx = v.argsort()
    v_s = v[sidx]
    idx = np.searchsorted(v_s, a)
    idx[idx == len(v)] = len(v) - 1
    idx0 = (idx - 1).clip(min=0)

    m = np.abs(a - v_s[idx]) >= np.abs(v_s[idx0] - a)
    m[idx == 0] = 0
    idx[m] -= 1
    out = sidx[idx]
    return out


def _get_params_find_closest_log(v):
    idx = logscale_transition_idx(v)
    vlmin = np.log2(v[0])

    if idx is None:
        dvl = np.log2(v[1]) - np.log2(v[0])
        params = dict(vlmin=vlmin, dvl=dvl)
    else:
        vlmin0, vlmin1 = vlmin, np.log2(v[idx - 1])
        dvl0 = np.log2(v[1])   - np.log2(v[0])
        dvl1 = np.log2(v[idx]) - np.log2(v[idx - 1])
        idx1 = np.asarray(idx - 1, dtype=np.int32)
        params = dict(vlmin0=vlmin0, vlmin1=vlmin1, dvl0=dvl0, dvl1=dvl1,
                      idx1=idx1)
    return idx, params

def find_closest_log(a, v, parallel=True):
    idx, params = _get_params_find_closest_log(v)
    out = np.zeros(a.shape, dtype=np.int32)
    params['omax'] = len(out) - 1

    if idx is None:
        fn = _find_closest_log_par if parallel else _find_closest_log
    else:
        fn = (_find_closest_log_piecewise_par if parallel else
              _find_closest_log_piecewise)

    fn(a, out, **params)
    return out

@jit(nopython=True, cache=True)
def _find_closest_log(a, out, vlmin, dvl, omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = min(round(max((np.log2(a[i, j]) - vlmin) / dvl, 0)), omax)

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_log_par(a, out, vlmin, dvl, omax):
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            out[i, j] = min(round(max((np.log2(a[i, j]) - vlmin) / dvl, 0)), omax)


@jit(nopython=True, cache=True)
def _find_closest_log_piecewise(a, out, vlmin0, vlmin1, dvl0, dvl1, idx1,
                                omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            al = np.log2(a[i, j])
            if al > vlmin1:
                out[i, j] = min(round((al - vlmin1) / dvl1) + idx1, omax)
            else:
                out[i, j] = round(max((al - vlmin0) / dvl0, 0))

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_log_piecewise_par(a, out, vlmin0, vlmin1, dvl0, dvl1, idx1,
                                    omax):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            if np.isinf(a[i, j]):
                continue
            al = np.log2(a[i, j])
            if al > vlmin1:
                out[i, j] = min(round((al - vlmin1) / dvl1) + idx1, omax)
            else:
                out[i, j] = round(max((al - vlmin0) / dvl0, 0))


def find_closest_lin(a, v, parallel=True):
    vmin = v[0]
    dv = v[1] - v[0]
    out = np.zeros(a.shape, dtype=np.int32)

    fn = _find_closest_lin_par if parallel else _find_closest_lin
    fn(a, out, vmin, dv, omax=len(out) - 1)
    return out

@jit(nopython=True, cache=True)
def _find_closest_lin(a, out, vmin, dv, omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = min(round(max((a[i, j] - vmin) / dv, 0)), omax)

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_lin_par(a, out, vmin, dv, omax):
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            out[i, j] = min(round(max((a[i, j] - vmin) / dv, 0)), omax)

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
    """Note: `value=np.nan` won't work (but np.inf will, separate from -np.inf)"""
    x, ref, xndim = _process_replace_fn_args(x, ref)
    x = _replace_at_value(x, ref, value, replacement)
    while x.ndim > xndim:
        x = x.squeeze(axis=-1)
    return x

def replace_under_abs(x, ref=None, value=0., replacement=0., parallel=None,
                      gpu=None):
    if gpu:
        _replace_under_abs_gpu(x, ref, value, replacement)
    elif parallel:
        _replace_under_abs_par(x, ref, value, replacement)
    else:
        _replace_under_abs(x, ref, value, replacement)

# TODO make parallel?
# TODO return None?
@jit(nopython=True, cache=True)
def _replace_at_inf_or_nan(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isinf(ref[i, j, k]) or np.isnan(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@jit(nopython=True, cache=True)
def _replace_at_inf(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isinf(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@jit(nopython=True, cache=True)
def _replace_at_nan(x, ref, replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if np.isnan(ref[i, j, k]):
                    x[i, j, k] = replacement
    return x

@jit(nopython=True, cache=True)
def _replace_at_value(x, ref, value=0., replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                if ref[i, j, k] == value:
                    x[i, j, k] = replacement
    return x


@jit(nopython=True, cache=True)
def _replace_under_abs(x, ref, value=0., replacement=0.):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
                if abs(ref[i, j]) < value:
                    x[i, j] = replacement

@jit(nopython=True, cache=True, parallel=True)
def _replace_under_abs_par(x, ref, value=0., replacement=0.):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
                if abs(ref[i, j]) < value:
                    x[i, j] = replacement


def _replace_under_abs_gpu(w, Wx, value=0., replacement=0.):
    """Not as general as CPU variants (namely `w` must be real and `Wx`
    must be complex).
    """
    kernel = '''
    extern "C" __global__
    void replace_under_abs(${dtype} w[${M}][${N}],
                           ${dtype} Wx[${M}][${N}][2],
                           ${dtype} *value, ${dtype} *replacement)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;

      if (i >= ${M} || j >= ${N})
        return;

      if (norm${f}(2, Wx[i][j]) < *value)
        w[i][j] = *replacement;
    }
    '''
    (blockspergrid, threadsperblock, kernel_kw, str_dtype
     ) = _get_kernel_params(Wx, dim=2)
    kernel_kw['f'] = 'f' if kernel_kw['dtype'] == 'float' else ''

    Wx = torch.view_as_real(Wx)
    kernel_args = [w.data_ptr(), Wx.data_ptr(),
                   cp.asarray(value,       dtype=str_dtype),
                   cp.asarray(replacement, dtype=str_dtype)]

    _run_on_gpu(kernel, blockspergrid, threadsperblock,
                *kernel_args, **kernel_kw)

#### misc (short) ############################################################
@jit(nopython=True, cache=True)
def _min_neglect_idx(arr, th=1e-12):
    """Used in utils.integrate_analytic and ._integrate_bounded."""
    for i, x in enumerate(arr):
        if x < th:
            return i
    return i

#### misc (long) #############################################################
def find_maximum(fn, step_size=1e-3, steps_per_search=1e4, step_start=0,
                 step_limit=1000, min_value=-1):
    """Finds max of any function with a single maximum, and input value
    at which the maximum occurs. Inputs and outputs must be 1D.

    Must be strictly non-decreasing from step_start up to maximum of interest.
    Takes absolute value of fn's outputs.
    """
    steps_per_search = int(steps_per_search)
    largest_max = min_value
    increment = int(steps_per_search * step_size)

    input_values = np.linspace(step_start, increment)
    output_values = -1 * np.ones(steps_per_search)

    search_idx = 0
    while True:
        start = step_start + increment * search_idx
        end   = start + increment
        input_values = np.linspace(start, end, steps_per_search, endpoint=False)

        output_values[:] = np.abs(asnumpy(fn(input_values)))

        output_max = output_values.max()
        if output_max > largest_max:
            largest_max = output_max
            input_value = input_values[np.argmax(output_values)]
        elif output_max < largest_max:
            break
        search_idx += 1

        if input_values.max() > step_limit:
            raise ValueError(("could not find function maximum with given "
                              "(step_size, steps_per_search, step_start, "
                              "step_limit, min_value)=({}, {}, {}, {}, {})"
                              ).format(step_size, steps_per_search, step_start,
                                       step_limit, min_value))
    return input_value, largest_max


def find_first_occurrence(fn, value, step_size=1e-3, steps_per_search=1e4,
                          step_start=0, step_limit=1000):
    """Finds earliest input value for which `fn(input_value) == value`, searching
    from `step_start` to `step_limit` in `step_size` increments.
    Takes absolute value of fn's outputs.
    """
    steps_per_search = int(steps_per_search)
    increment = int(steps_per_search * step_size)
    output_values = -1 * np.ones(steps_per_search)

    step_limit_exceeded = False
    search_idx = 0
    while True:
        start = step_start + increment * search_idx
        end   = start + increment
        input_values = np.linspace(start, end, steps_per_search, endpoint=False)
        if input_values.max() > step_limit:
            step_limit_exceeded = True
            input_values = np.clip(input_values, None, step_limit)

        output_values[:] = np.abs(asnumpy(fn(input_values)))
        mxdiff = np.abs(np.diff(output_values)).max()

        # more reliable than `argmin not in (0, len - 1)` for smooth `fn`
        if np.any(np.abs(output_values - value) <= mxdiff):
            idx = np.argmin(np.abs(output_values - value))
            break
        search_idx += 1

        if step_limit_exceeded:
            raise ValueError(("could not find input value to yield function "
                              f"output value={value} with given "
                              "(step_size, steps_per_search, step_start, "
                              "step_limit, min_value)=({}, {}, {}, {})"
                              ).format(step_size, steps_per_search,
                                       step_start, step_limit))
    input_value = input_values[idx]
    output_value = output_values[idx]
    return input_value, output_value


def phase_cwt_cpu(Wx, dWx, gamma, parallel=False):
    dtype = 'float32' if Wx.dtype == np.complex64 else 'float64'
    out = np.zeros(Wx.shape, dtype=dtype)
    gamma = np.asarray(gamma, dtype=dtype)

    fn = _phase_cwt_par if parallel else _phase_cwt
    fn(Wx, dWx, out, gamma)
    return out

@jit(nopython=True)
def _phase_cwt(Wx, dWx, out, gamma):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))

@jit(nopython=True, parallel=True)  # TODO cache
def _phase_cwt_par(Wx, dWx, out, gamma):
    for i in prange(Wx.shape[0]):
        for j in prange(Wx.shape[1]):
            if abs(Wx[i, j]) < gamma:
                out[i, j] = np.inf
            else:
                A, B = dWx[i, j].real, dWx[i, j].imag
                C, D = Wx[i, j].real,  Wx[i, j].imag
                out[i, j] = abs((B*C - A*D) / ((C**2 + D**2) * 6.283185307179586))


def ssq_cwt_log_gpu(Wx, dWx, ssq_freqs, const, gamma, flipud=False, out=None):
    """Complete synchrosqueezing pipeline without intermediate assignment to `w`.
    """
    kernel = '''
    extern "C" __global__
    void ssq_cwt_log(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vlmin, double dvl) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)log2${f}(w_ij) - vlmin) / dvl);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    '''
    return _gpu_run_ssq(kernel, Wx, dWx, ssq_freqs, const, gamma, flipud, out)


def ssq_cwt_log_piecewise_gpu(Wx, dWx, ssq_freqs, const, gamma, flipud=False,
                              out=None):
    kernel = '''
    extern "C" __global__
    void ssq_cwt_log_piecewise(${dtype} Wx[${M}][${N}][2],
                               ${dtype} dWx[${M}][${N}][2],
                               ${dtype} out[${M}][${N}][2],
                               ${dtype} const_arr[${M}],
                               ${dtype} *gamma,
                               double vlmin0, double vlmin1,
                               double dvl0, double dvl1,
                               int idx1) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        double wl;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            wl = log2${f}(w_ij);
            if (wl > vlmin1){
                k = (int)round((wl - vlmin1) / dvl1) + idx1;
                if (k >= ${M})
                    k = ${M} - 1;
            } else {
                k = (int)round((wl - vlmin0) / dvl0);
                if (k < 0)
                    k = 0;
            }
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    '''
    return _gpu_run_ssq(kernel, Wx, dWx, ssq_freqs, const, gamma, flipud, out)


def ssq_cwt_lin_gpu(Wx, dWx, ssq_freqs, const, gamma, flipud=False, out=None):
    kernel = '''
    extern "C" __global__
    void ssq_cwt_log(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vmin, double dv) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)w_ij - vmin) / dv);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    '''
    return _gpu_run_ssq(kernel, Wx, dWx, ssq_freqs, const, gamma, flipud, out)


def phase_cwt_gpu(Wx, dWx, gamma):
    """Computes only the imaginary part of `dWx / Wx` while dividing by 2*pi
    in same operation; doesn't compute division at all if `abs(Wx) < gamma`.
    Less memory & less computation.
    """
    # TODO they all do ^; update main docs instead
    kernel = '''
    extern "C" __global__
    void phase_cwt(${dtype} Wx[${M}][${N}][2],
                   ${dtype} dWx[${M}][${N}][2],
                   ${dtype} out[${M}][${N}],
                   ${dtype} *gamma) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= ${M} || j >= ${N})
            return;

        if (norm${f}(2, Wx[i][j]) < *gamma){
          out[i][j] = 1.0/0.0;
          return;
        }

        ${dtype} A = dWx[i][j][0];
        ${dtype} B = dWx[i][j][1];
        ${dtype} C = Wx[i][j][0];
        ${dtype} D = Wx[i][j][1];

        out[i][j] = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));
    }
    '''
    (blockspergrid, threadsperblock, kernel_kw, str_dtype
     ) = _get_kernel_params(Wx, dim=2)
    kernel_kw['f'] = 'f' if kernel_kw['dtype'] == 'float' else ''

    out = torch.zeros(Wx.shape, device=Wx.device, dtype=getattr(torch, str_dtype))
    Wx  = torch.view_as_real(Wx)
    dWx = torch.view_as_real(dWx)

    kernel_args = [Wx.data_ptr(), dWx.data_ptr(), out.data_ptr(),
                   cp.asarray(gamma, dtype=str_dtype)]
    _run_on_gpu(kernel, blockspergrid, threadsperblock,
                *kernel_args, **kernel_kw)
    return out


#### gpu utils ###############################################################
@cp._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


def _get_kernel_params(x, dim=2, threadsperblock=None):
    M, N = x.shape

    if dim == 1:
        threadsperblock = threadsperblock or (1024,)
        blockspergrid = (int(np.ceil(M * N / threadsperblock[0])),)  # TODO M * N?
    elif dim == 2:
        threadsperblock = threadsperblock or (32, 32)
        blockspergrid_x = int(np.ceil(M / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(N / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    dtype = ('double' if x.dtype in (torch.float64, torch.complex128) else
             'float')
    kernel_kw = dict(dtype=dtype, M=M, N=N)
    str_dtype = 'float32' if dtype == 'float' else 'float64'
    return blockspergrid, threadsperblock, kernel_kw, str_dtype


def _gpu_run_indexed_sum(kernel, Wx, w, out, const, *more_consts, flipud=False):
    blockspergrid, threadsperblock, kernel_kw, _ = _get_kernel_params(Wx, dim=1)
    M = kernel_kw['M']
    kernel_kw.update(dict(f='f' if kernel_kw['dtype'] == 'float' else '',
                          extra=f"k = {M} - 1 - k;" if flipud else ""))

    # (M, N) -> (M, N, 2); treat real & imag as separate arrays for CUDA
    Wx = torch.view_as_real(Wx)
    if out is None:
        out = torch.zeros(Wx.shape, dtype=Wx.dtype, device=Wx.device)
    else:
        out = torch.view_as_real(out)
    print(const)
    const_arr = torch.as_tensor(const, dtype=Wx.dtype, device=Wx.device)

    kernel_args = [Wx.data_ptr(), w.data_ptr(), out.data_ptr(),
                   const_arr.data_ptr(), *more_consts]
    _run_on_gpu(kernel, blockspergrid, threadsperblock,
                *kernel_args, **kernel_kw)

    out = torch.view_as_complex(out)
    return out


def _gpu_run_ssq(kernel, Wx, dWx, ssq_freqs, const, gamma, flipud=False,
                 out=None):
    (blockspergrid, threadsperblock, kernel_kw, str_dtype
     ) = _get_kernel_params(Wx, dim=1)
    M = kernel_kw['M']
    kernel_kw.update(dict(f='f' if kernel_kw['dtype'] == 'float' else '',
                          extra=f"k = {M} - 1 - k;" if flipud else ""))

    # process `Wx, dWx, out`
    Wx  = torch.view_as_real(Wx)
    dWx = torch.view_as_real(dWx)
    if out is None:
        out = torch.zeros(Wx.shape, dtype=Wx.dtype, device=Wx.device)
    else:
        out = torch.view_as_real(out)

    # process `const`
    if not isinstance(const, (np.ndarray, torch.Tensor)) or len(const) != len(Wx):
        const = np.array(len(Wx) * [const]).squeeze()
    const = const.squeeze()
    const_arr = torch.as_tensor(const, dtype=Wx.dtype, device=Wx.device)

    # process other constants
    _, params = _get_params_find_closest_log(ssq_freqs)
    # params = dict(vmin=ssq_freqs[0], dv=ssq_freqs[1] - ssq_freqs[0])
    if 'idx1' in params:
        params['idx1'] = int(params['idx1'])

    # collect tensors & constants, run
    kernel_args = [Wx.data_ptr(), dWx.data_ptr(), out.data_ptr(),
                   const_arr.data_ptr(), cp.asarray(gamma, dtype=str_dtype),
                   *list(params.values())]
    _run_on_gpu(kernel, blockspergrid, threadsperblock, *kernel_args, **kernel_kw)

    out = torch.view_as_complex(out)
    return out


Stream = namedtuple('Stream', ['ptr'])

def _run_on_gpu(kernel, grid, block, *args, **kwargs):
    kernel_name = kernel.split('void ')[1].split('(')[0]
    fn = load_kernel(kernel_name, kernel, **kwargs)
    fn(grid=grid, block=block, args=args,
       stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))


_kernel_codes = dict(
    ssq_cwt_log_piecewise = '''
    extern "C" __global__
    void ssq_cwt_log_piecewise(${dtype} Wx[${M}][${N}][2],
                               ${dtype} dWx[${M}][${N}][2],
                               ${dtype} out[${M}][${N}][2],
                               ${dtype} const_arr[${M}],
                               ${dtype} *gamma,
                               double vlmin0, double vlmin1,
                               double dvl0, double dvl1,
                               int idx1) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        double wl;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            wl = log2${f}(w_ij);
            if (wl > vlmin1){
                k = (int)round((wl - vlmin1) / dvl1) + idx1;
                if (k >= ${M})
                    k = ${M} - 1;
            } else {
                k = (int)round((wl - vlmin0) / dvl0);
                if (k < 0)
                    k = 0;
            }
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    ssq_cwt_log = '''
    extern "C" __global__
    void ssq_cwt_log(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vlmin, double dvl) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)log2${f}(w_ij) - vlmin) / dvl);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    ssq_cwt_lin = '''
    extern "C" __global__
    void ssq_cwt_lin(${dtype} Wx[${M}][${N}][2],
                     ${dtype} dWx[${M}][${N}][2],
                     ${dtype} out[${M}][${N}][2],
                     ${dtype} const_arr[${M}],
                     ${dtype} *gamma,
                     double vmin, double dv) {
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= ${N})
            return;

        int k;
        ${dtype} w_ij, A, B, C, D;

        for (int i=0; i < ${M}; ++i){
          if (norm${f}(2, Wx[i][j]) > *gamma){

            A = dWx[i][j][0];
            B = dWx[i][j][1];
            C = Wx[i][j][0];
            D = Wx[i][j][1];
            w_ij = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));

            k = (int)round(((double)w_ij - vmin) / dv);
            if (k >= ${M})
                k = ${M} - 1;
            else if (k < 0)
                k = 0;
            ${extra}

            out[k][j][0] += Wx[i][j][0] * const_arr[i];
            out[k][j][1] += Wx[i][j][1] * const_arr[i];
          }
        }
    }
    ''',

    indexed_sum_log_piecewise = '''
    extern "C" __global__
    void indexed_sum_log_piecewise(${dtype} Wx[${M}][${N}][2],
                                   ${dtype} w[${M}][${N}],
                                   ${dtype} out[${M}][${N}][2],
                                   ${dtype} const_arr[${M}],
                                   double vlmin0, double vlmin1,
                                   double dvl0, double dvl1,
                                   int idx1)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      double wl;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          wl = (double)log2${f}(w[i][j]);

          if (wl > vlmin1){
              k = (int)round((wl - vlmin1) / dvl1) + idx1;
              if (k >= ${M})
                  k = ${M} - 1;
          } else {
              k = (int)round((wl - vlmin0) / dvl0);
              if (k < 0)
                  k = 0;
          }
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',

    indexed_sum_log = '''
    extern "C" __global__
    void indexed_sum_log(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vlmin, double dvl)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)log2${f}(w[i][j]) - vlmin) / dvl);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',


    indexed_sum_lin = '''
    extern "C" __global__
    void indexed_sum_lin(${dtype} Wx[${M}][${N}][2],
                         ${dtype} w[${M}][${N}],
                         ${dtype} out[${M}][${N}][2],
                         ${dtype} const_arr[${M}],
                         double vmin, double dv)
    {
      int j = blockIdx.x * blockDim.x + threadIdx.x;

      if (j >= ${N})
        return;

      int k;
      for (int i=0; i < ${M}; ++i){
        if (!isinf(w[i][j])){
          k = (int)round(((double)(w[i][j]) - vmin) / dv);

          if (k >= ${M})
              k = ${M} - 1;
          else if (k < 0)
              k = 0;
          ${extra}

          out[k][j][0] += Wx[i][j][0] * const_arr[i];
          out[k][j][1] += Wx[i][j][1] * const_arr[i];
        }
      }
    }
    ''',

    phase_cwt = '''
    extern "C" __global__
    void phase_cwt(${dtype} Wx[${M}][${N}][2],
                   ${dtype} dWx[${M}][${N}][2],
                   ${dtype} out[${M}][${N}],
                   ${dtype} *gamma) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= ${M} || j >= ${N})
            return;

        if (norm${f}(2, Wx[i][j]) < *gamma){
          out[i][j] = 1.0/0.0;
          return;
        }

        ${dtype} A = dWx[i][j][0];
        ${dtype} B = dWx[i][j][1];
        ${dtype} C = Wx[i][j][0];
        ${dtype} D = Wx[i][j][1];

        out[i][j] = abs((B*C - A*D) / ((C*C + D*D) * 6.283185307179586));
    }
    ''',



)

###############################################################################
from .utils.common import WARN
from .utils.cwt_utils import logscale_transition_idx
