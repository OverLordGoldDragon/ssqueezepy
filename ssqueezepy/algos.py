# -*- coding: utf-8 -*-
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
    """Sum `a` into rows of 2D array according to indices given by 2D `k`"""
    out = np.zeros(a.shape, dtype=np.cfloat)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[k[i, j], j] += a[i, j]
    return out


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

#### misc (short) ############################################################
@njit
def _min_neglect_idx(arr, th=1e-12):
    """Used in utils._integrate_analytic and ._integrate_bounded."""
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
