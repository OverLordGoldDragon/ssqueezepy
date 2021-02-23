# -*- coding: utf-8 -*-
import numpy as np
import pyfftw
from scipy.fft import fftshift, ifftshift

pyfftw.config.NUM_THREADS = 8
pyfftw.config.PLANNER_EFFORT = 'FFTW_' + ('MEASURE', 'PATIENT')[0]
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(600)

__all__ = [
    'fft',
    'rfft',
    'fft2',
    'fftshift',
    'ifftshift',
]


def fft(x, axis=-1):
    fft_object = _get_fft_object(x, axis, real=False)
    fft_object.input_array[:] = x
    return fft_object()


def rfft(x, axis=-1):
    fft_object = _get_fft_object(x, axis, real=True)
    fft_object.input_array[:] = x
    return fft_object()


def fft2(x, axis=-1):
    fft_object = _get_fft_object(x, axis, real=True)
    fft_object.input_array[:] = x

    n_fft = x.shape[axis]
    xf = np.zeros(x.shape, dtype='complex64')

    if axis == 0:
        xf[:n_fft//2 + 1, :] = fft_object()
        slc = xf[1:n_fft//2][::-1]
        xf.real[n_fft//2 + 1:, :] = slc.real
        xf.imag[n_fft//2 + 1:, :] = -slc.imag
    elif axis in (1, -1):
        xf[:, :n_fft//2 + 1] = fft_object()
        slc = xf[:, 1:n_fft//2][:, ::-1]
        xf.real[:, n_fft//2 + 1:] = slc.real
        xf.imag[:, n_fft//2 + 1:] = -slc.imag
    return xf


def _get_fft_object(x, axis, real=False, patience=0):
    if isinstance(patience, tuple):
        patience, planning_timelimit = patience
    else:
        planning_timelimit = None
    _validate_input(x, axis, real=real)
    cdtype = 'complex128' if x.dtype in (np.float64, np.cfloat) else 'complex64'

    shape = _get_shape(x, axis, real=real)
    flags = [('FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE')[patience]]
    a = pyfftw.empty_aligned(x.shape, dtype=x.dtype if real else cdtype)
    b = pyfftw.empty_aligned(shape,   dtype=cdtype)
    fft_object = pyfftw.FFTW(a, b, axes=(axis,), flags=['FFTW_WISDOM_ONLY'])

    return fft_object


def _get_shape(x, axis, real=False):
    N = len(x)
    if x.ndim == 2:
        M = x.shape[1]
    n_fft = x.shape[axis]

    fft_out_len = (n_fft//2 + 1) if real else n_fft
    if x.ndim == 2:
        shape = ((fft_out_len, M) if axis == 0 else
                 (N, fft_out_len))
    else:
        shape = (fft_out_len,)
    return shape


def _validate_input(x, axis, real=False):
    """Assert is single/double precision and is 1D/2D."""
    if x.ndim not in (1, 2):
        raise ValueError("unsupported `x.ndim`: %s " % x.ndim
                         + "; must be 1D or 2D")

    supported = ('float32', 'float64', 'complex64', 'complex128')
    dtype = str(x.dtype)
    if dtype not in supported:
        raise TypeError("unsupported `x.dtype`: %s " % dtype
                        + "(must be one of: %s)" % ', '.join(supported))
    if real and dtype.startswith('complex'):
        raise TypeError("`x` cannot be complex for `rfft`")

    if axis not in (0, 1, -1):
        raise ValueError("unsupported `axis`: %s " % axis
                         + "; must be 0, 1, or -1")
