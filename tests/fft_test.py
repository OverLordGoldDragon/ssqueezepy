# -*- coding: utf-8 -*-
"""Fast Fourier Transform tests.

Tests that `ssqueezepy.FFT` outputs match `scipy`'s.
Also test that parallelized implementations output same as non-parallelized.
"""
import os
import pytest
import numpy as np
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft

import ssqueezepy
from ssqueezepy import fft, rfft, ifft, irfft, cwt, extract_ridges
from ssqueezepy.algos import indexed_sum
from ssqueezepy.configs import gdefaults

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0


def test_1D():
    for N in (128, 129):
        x = np.random.randn(N)
        xf = x[:N//2 + 1] * (1 + 2j)

        souts = dict(fft=sfft(x), rfft=srfft(x), ifft=sifft(xf),
                     irfft1=sirfft(xf), irfft2=sirfft(xf, n=len(x)))

        for patience in (0, (1, 1), (2, 1)):
            qouts = dict(
                fft=fft(x,       patience=patience),
                rfft=rfft(x,     patience=patience),
                ifft=ifft(xf,    patience=patience),
                irfft1=irfft(xf, patience=patience),
                irfft2=irfft(xf, patience=patience, n=len(x)),
            )
            for name, qout in qouts.items():
                assert np.allclose(qout, souts[name]), (
                    "{}: N={}, patience={}".format(name, N, patience))


def test_2D():
    for N in (128, 129):
        for M in (64, 65):
            for axis in (0, 1):
                x = np.random.randn(N, M)
                if axis == 0:
                    xf = x[:N//2 + 1] * (1 + 2j)
                else:
                    xf = x[:, :M//2 + 1] * (1 + 2j)

                souts = dict(
                    fft=sfft(x,       axis=axis),
                    rfft=srfft(x,     axis=axis),
                    ifft=sifft(xf,    axis=axis),
                    irfft1=sirfft(xf, axis=axis),
                    irfft2=sirfft(xf, axis=axis, n=x.shape[axis]),
                )

                for patience in (0, (1, .5), (2, .5)):
                    kw = dict(axis=axis, patience=patience)
                    qouts = dict(
                        fft=fft(x,       **kw),
                        rfft=rfft(x,     **kw),
                        ifft=ifft(xf,    **kw),
                        irfft1=irfft(xf, **kw),
                        irfft2=irfft(xf, **kw, n=x.shape[axis]),
                    )
                    for name, qout in qouts.items():
                        assert np.allclose(qout, souts[name]), (
                            "{}: (N, M)=({}, {}), patience={}".format(
                                name, N, M, patience))


def test_exhaustive():
    """Ensure exhaustive case works."""
    fft(np.random.randn(4), patience=(2, None))


def test_ridge_extraction():
    x = np.random.randn(512)
    Wx, scales = cwt(x)

    out1 = extract_ridges(Wx, scales, parallel=False)
    out2 = extract_ridges(Wx, scales, parallel=True)

    assert np.allclose(out1, out2), "MAE: %s" % np.mean(np.abs(out1 - out2))


def test_indexed_sum():
    Wx = np.random.randn(1000, 1000).astype('complex64')
    k = np.random.randint(0, len(Wx), Wx.shape)

    out1 = indexed_sum(Wx, k, parallel=False)
    out2 = indexed_sum(Wx, k, parallel=True)

    assert np.allclose(out1, out2), "MAE: %s" % np.mean(np.abs(out1 - out2))


def test_parallel_setting():
    """Assert
        1. ssqueezepy is parallel by default
        2. `configs.ini` includes parallel config
        3. os.environ flag overrides `configs.ini`
    """
    assert ssqueezepy.is_parallel()

    parallel = gdefaults('configs.is_parallel', parallel=None)
    assert parallel is not None
    assert parallel == 1

    os.environ['SSQ_PARALLEL'] = '0'
    try:
        assert not ssqueezepy.is_parallel()
    except AssertionError:
        raise AssertionError()
    finally:
        # ensure `os.environ` is cleaned even if assert fails
        os.environ.pop('SSQ_PARALLEL')


if __name__ == '__main__':
    if VIZ:
        test_1D()
        test_2D()
        test_ridge_extraction()
        test_indexed_sum()
        test_parallel_setting()
    else:
        pytest.main([__file__, "-s"])
