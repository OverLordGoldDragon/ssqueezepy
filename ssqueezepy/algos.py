# -*- coding: utf-8 -*-
import numpy as np
from numba import jit, prange
from functools import reduce


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


def indexed_sum_onfly(Wx, w, v, const=1, logscale=False, parallel=True):
    """`indexed_sum` and `find_closest` within same loop, sparing an array;
    see `help(algos.find_closest)` on how `k` is computed.
    """
    out = np.zeros(Wx.shape, dtype=Wx.dtype)
    if logscale:
        idx, params = _get_params_find_closest_log(v)
        params['const'] = const
        if idx is None:
            fn = (_indexed_sum_log_par if parallel else
                  _indexed_sum_log)
        else:
            fn = (_indexed_sum_log_piecewise_par if parallel else
                  _indexed_sum_log_piecewise)
    else:
        params = dict(vmin=v[0], dv=(v[1] - v[0]))
        fn = (_indexed_sum_lin_par if parallel else
              _indexed_sum_lin)
    params['omax'] = len(out) - 1

    fn(Wx, w, out, **params)
    return out


@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_par(Wx, w, out, const, vlmin, dvl, omax):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax)
            out[k, j] += Wx[i, j] * const

@jit(nopython=True, cache=True)
def _indexed_sum_log(Wx, w, out, const, vlmin, dvl, omax):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((np.log2(w[i, j]) - vlmin) / dvl, 0)), omax)
            out[k, j] += Wx[i, j] * const


@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_log_piecewise_par(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                                   idx1, omax):
    # it's also possible to construct the if-else logic in terms of mappables
    # of `vlmin`, `dvl`, and `idx`, which generalizes to any number of transitions
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = min(round((wl - vlmin1) / dvl1) + idx1, omax)
                out[k, j] += Wx[i, j] * const[i]
            else:
                k = min(round(max((wl - vlmin0) / dvl0, 0)), omax)
                out[k, j] += Wx[i, j] * const[i]

@jit(nopython=True, cache=True)
def _indexed_sum_log_piecewise(Wx, w, out, const, vlmin0, vlmin1, dvl0, dvl1,
                               idx1, omax):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            wl = np.log2(w[i, j])
            if wl > vlmin1:
                k = min(round((wl - vlmin1) / dvl1) + idx1, omax)
                out[k, j] += Wx[i, j] * const[i]
            else:
                k = min(round(max((wl - vlmin0) / dvl0, 0)), omax)
                out[k, j] += Wx[i, j] * const[i]


@jit(nopython=True, cache=True, parallel=True)
def _indexed_sum_lin_par(Wx, w, out, const, vmin, dv, omax):
    for j in prange(Wx.shape[1]):
        for i in range(Wx.shape[0]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((w[i, j] - vmin) / dv, 0)), omax)
            out[k, j] += Wx[i, j] * const

@jit(nopython=True, cache=True)
def _indexed_sum_lin(Wx, w, out, const, vmin, dv, omax):
    for i in range(Wx.shape[0]):
        for j in range(Wx.shape[1]):
            if np.isinf(w[i, j]):
                continue
            k = min(round(max((w[i, j] - vmin) / dv, 0)), omax)
            out[k, j] += Wx[i, j] * const

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
        params = (vlmin, dvl)
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

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_log_par(a, out, vlmin, dvl, omax):
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            out[i, j] = min(round(max((np.log2(a[i, j]) - vlmin) / dvl, 0)), omax)

@jit(nopython=True, cache=True)
def _find_closest_log(a, out, vlmin, dvl, omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = min(round(max((np.log2(a[i, j]) - vlmin) / dvl, 0)), omax)

@jit(nopython=True, parallel=True)  # TODO cache
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
                # TODO shouldn't this be min(, idx)
                out[i, j] = min(round(max((al - vlmin0) / dvl0, 0)), omax)

@jit(nopython=True, cache=True)
def _find_closest_log_piecewise(a, out, vlmin0, vlmin1, dvl0, dvl1, idx1,
                                omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            al = np.log2(a[i, j])
            if al > vlmin1:
                out[i, j] = min(round((al - vlmin1) / dvl1) + idx1, omax)
            else:
                out[i, j] = min(round(max((al - vlmin0) / dvl0, 0)), omax)


# TODO pass `out` from outside? (reusables)
def find_closest_lin(a, v, parallel=True):
    vmin = v[0]
    dv = v[1] - v[0]
    out = np.zeros(a.shape, dtype=np.int32)

    fn = _find_closest_lin_par if parallel else _find_closest_lin
    fn(a, out, vmin, dv, omax=len(out) - 1)
    return out

@jit(nopython=True, cache=True, parallel=True)
def _find_closest_lin_par(a, out, vmin, dv, omax):
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            out[i, j] = min(round(max((a[i, j] - vmin) / dv, 0)), omax)

@jit(nopython=True, cache=True)
def _find_closest_lin(a, out, vmin, dv, omax):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
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

# TODO make parallel?
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

        output_values[:] = np.abs(fn(input_values))

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

        output_values[:] = np.abs(fn(input_values))
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


###############################################################################
from .utils.common import WARN
from .utils.cwt_utils import logscale_transition_idx
