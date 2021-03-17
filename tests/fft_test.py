# -*- coding: utf-8 -*-
"""Fast Fourier Transform tests.

Tests that `ssqueezepy.FFT` outputs match `scipy`'s.
Also test that parallelized implementations output same as non-parallelized.
"""
# TODO rename? gpu vs cpu etc
import os
import pytest
import numpy as np
import torch
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft

import ssqueezepy
from ssqueezepy import fft, rfft, ifft, irfft, cwt, extract_ridges
from ssqueezepy.algos import indexed_sum, indexed_sum_onfly
from ssqueezepy.configs import gdefaults
from ssqueezepy import TestSignals, Wavelet
from ssqueezepy.utils import process_scales

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 1
try:
    torch.tensor(1, device='cuda')
    CAN_GPU = True
except:
    CAN_GPU = False


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
    assert ssqueezepy.IS_PARALLEL()

    parallel = gdefaults('configs.IS_PARALLEL', parallel=None)
    assert parallel is not None
    assert parallel == 1

    os.environ['SSQ_PARALLEL'] = '0'
    try:
        assert not ssqueezepy.IS_PARALLEL()
    except AssertionError:
        raise AssertionError()
    finally:
        # ensure `os.environ` is cleaned even if assert fails
        os.environ.pop('SSQ_PARALLEL')

from ssqueezepy.algos import phase_cwt_cpu, phase_cwt_gpu, replace_under_abs

def test_phase_cwt():
    x = TestSignals(N=10000).par_lchirp()[0]
    x += x[::-1]
    wavelet = Wavelet()
    scales = process_scales('log', len(x), wavelet, nv=32)[:240]

    os.environ['SSQ_GPU'] = '1'
    Wx, _, dWx = cwt(x, wavelet, scales=scales, derivative=True, cache_wavelet=1)
    Wx, dWx = Wx.cpu().numpy(), dWx.cpu().numpy()

    # for dtype in ('float32', 'float64'):
    for dtype in ('complex128', 'complex64'):
        # Wx  = np.random.randn(100, 8192).astype(dtype) * (1 + 2j)
        # dWx = np.random.randn(100, 8192).astype(dtype) * (2 - 1j)
        Wx, dWx = Wx.astype(dtype), dWx.astype(dtype)
        if CAN_GPU:
            Wxt  = torch.tensor(Wx,  device='cuda')
            dWxt = torch.tensor(dWx, device='cuda')
        gamma = 1e-2

        _out = (dWx / Wx).imag / (2 * np.pi)
        _out[np.abs(Wx) < gamma] = np.inf
        _out = np.abs(_out)

        out0 = phase_cwt_cpu(Wx, dWx, gamma, parallel=False)
        out1 = phase_cwt_cpu(Wx, dWx, gamma, parallel=True)
        if CAN_GPU:
            out2 = phase_cwt_gpu(Wxt, dWxt, gamma).cpu().numpy()

        with np.errstate(invalid='ignore'):
            mape0_ = (np.abs(_out - out0) / np.abs(_out)).mean()
            mape01 = (np.abs(out0 - out1) / np.abs(out0)).mean()
            if CAN_GPU:
                mape02 = (np.abs(out0 - out2) / np.abs(out0)).mean()
        assert np.allclose(out0, _out), ("base",     dtype, mape0_)
        assert np.allclose(out0, out1), ("parallel", dtype, mape01)
        if CAN_GPU:
            assert np.allclose(out0, out2), ("gpu", dtype, mape02)


def test_replace_under_abs():
    for dtype in ('float32', 'float64'):
        w0 = np.random.randn(100, 200).astype(dtype)
        Wx = np.random.randn(100, 200).astype(dtype) * (2 - 1j)
        w1 = w0.copy()
        if CAN_GPU:
            wt  = torch.tensor(w0, device='cuda')
            Wxt = torch.tensor(Wx, device='cuda')
        gamma = 1e-2

        replace_under_abs(w0, Wx, gamma, np.inf, parallel=False, gpu=False)
        replace_under_abs(w1, Wx, gamma, np.inf, parallel=True,  gpu=False)
        if CAN_GPU:
            replace_under_abs(wt, Wxt, gamma, np.inf, parallel=False, gpu=True)
            wt = wt.cpu().numpy()

        assert np.allclose(w0, w1), ("parallel", dtype)
        if CAN_GPU:
            assert np.allclose(w0, wt), ("gpu", dtype)


def _make_ssq_freqs(M, scaletype):
    if scaletype == 'log-piecewise':
        sf = np.logspace(0, np.log10(M), 2*M)
        sf1 = sf[:M//2]
        sf2 = sf[M//2 + 3 - 1::3]
        ssq_freqs = np.hstack([sf1, sf2])
    elif scaletype == 'log':
        ssq_freqs = np.logspace(0, np.log10(M), M)
    elif scaletype == 'linear':
        ssq_freqs = np.linspace(0, M, M)
    return ssq_freqs


def test_indexed_sum_onfly():
    for scaletype in ('log-piecewise', 'log', 'linear'):
      for dtype in ('float32', 'float64'):
        for flipud in (False, True):
          Wx = np.random.randn(100, 2000).astype(dtype) * (1 + 2j)
          w  = np.abs(np.random.randn(*Wx.shape).astype(dtype))
          w *= (2*len(Wx) / w.max())
          Wxt, wt = [torch.tensor(g, device='cuda') for g in (Wx, w)]

          ssq_freqs = _make_ssq_freqs(len(Wx), scaletype)
          ssq_logscale = scaletype.startswith('log')
          const = (np.log(2) / 32 if 1 else
                   ssq_freqs)

          out0 = indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale,
                                   parallel=False, gpu=False, flipud=flipud)
          out1 = indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale,
                                   parallel=True,  gpu=False, flipud=flipud)
          out2 = indexed_sum_onfly(Wxt, wt, ssq_freqs, const, ssq_logscale,
                                   parallel=False, gpu=True,  flipud=flipud)
          out2 = out2.cpu().numpy()

          adiff01 = np.abs(out0 - out1).mean()
          adiff02 = np.abs(out0 - out2).mean()
          # this is due to `const` varying rather than 'linear'
          th = ((1e-16 if dtype == 'float64' else 1e-8) if ssq_logscale else
                (1e-13 if dtype == 'float64' else 1e-5))
          assert adiff01 < th, (scaletype, dtype, flipud, adiff01)
          assert adiff02 < th, (scaletype, dtype, flipud, adiff02)


if __name__ == '__main__':
    if VIZ:
        # test_1D()
        # test_2D()
        # test_ridge_extraction()
        # test_indexed_sum()
        # test_parallel_setting()
        test_phase_cwt()
        test_replace_under_abs()
        test_indexed_sum_onfly()
    else:
        pytest.main([__file__, "-s"])
