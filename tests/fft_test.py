# -*- coding: utf-8 -*-
"""Fast Fourier Transform, CPU parallelization, and GPU execution tests:
   - multi-thread CPU & GPU outputs match that of single-thread CPU
   - batched (multi-input) outputs match single for-looped
   - `ssqueezepy.FFT` outputs match `scipy`'s
   - unified synchrosqueezing pipelines outputs match that of v0.6.0

Note that GPU tests are skipped in CI (Travis), and are instead done locally.
"""
import os
import pytest
import numpy as np
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft
from scipy.fft import ifftshift

import ssqueezepy
from ssqueezepy import TestSignals, Wavelet, ssq_stft, ssq_cwt
from ssqueezepy import fft, rfft, ifft, irfft, cwt
from ssqueezepy.algos import indexed_sum, indexed_sum_onfly, ssqueeze_fast
from ssqueezepy.algos import phase_cwt_cpu, phase_cwt_gpu, replace_under_abs
from ssqueezepy.algos import phase_stft_gpu, phase_stft_cpu
from ssqueezepy.configs import gdefaults
from ssqueezepy.utils import process_scales, buffer

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0
try:
    import torch
    torch.tensor(1, device='cuda')
    CAN_GPU = True
except:
    CAN_GPU = False


def _wavelet(name='gmw', **kw):
    return Wavelet((name, kw))


def test_1D():
    os.environ['SSQ_GPU'] = '0'
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
    os.environ['SSQ_GPU'] = '0'
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
    os.environ['SSQ_GPU'] = '0'
    fft(np.random.randn(4), patience=(2, None))


def test_indexed_sum():
    os.environ['SSQ_GPU'] = '0'
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
    os.environ['SSQ_GPU'] = '0'
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


def _noninf_mean(x):
    x[np.isinf(x) | np.isnan(x)] = 0
    return x.mean()

def test_phase_cwt():
    os.environ['SSQ_GPU'] = '0'
    x = TestSignals(N=1000).par_lchirp()[0]
    x += x[::-1]
    wavelet = Wavelet()
    scales = process_scales('log', len(x), wavelet, nv=32)[:240]

    Wx, _, dWx = cwt(x, wavelet, scales=scales, derivative=True, cache_wavelet=1)

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
            mape0_ = _noninf_mean(np.abs(_out - out0) / np.abs(_out))
            mape01 = _noninf_mean(np.abs(out0 - out1) / np.abs(out0))
            if CAN_GPU:
                mape02 = _noninf_mean(np.abs(out0 - out2) / np.abs(out0))

        assert np.allclose(out0, _out), ("base",     dtype, mape0_)
        assert np.allclose(out0, out1), ("parallel", dtype, mape01)
        if CAN_GPU:
            assert np.allclose(out0, out2), ("gpu", dtype, mape02)


def test_phase_stft():
    atol = 1e-7
    np.random.seed(0)
    for dtype in ('float64', 'float32'):
        Wx  = np.random.randn(100, 1028).astype(dtype) * (1 + 2j)
        dWx = np.random.randn(100, 1028).astype(dtype) * (2 - 1j)
        Sfs = np.linspace(0, .5, len(Wx)).astype(dtype)
        if CAN_GPU:
            Wxt  = torch.as_tensor(Wx,  device='cuda')
            dWxt = torch.as_tensor(dWx, device='cuda')
            Sfst = torch.as_tensor(Sfs, device='cuda')
        gamma = 1e-2

        _out = Sfs[:, None] - (dWx / Wx).imag / (2*np.pi)
        _out[np.abs(Wx) < gamma] = np.inf
        _out = np.abs(_out)

        out0 = phase_stft_cpu(Wx, dWx, Sfs, gamma, parallel=False)
        out1 = phase_stft_cpu(Wx, dWx, Sfs, gamma, parallel=True)
        if CAN_GPU:
            out2 = phase_stft_gpu(Wxt, dWxt, Sfst, gamma).cpu().numpy()

        with np.errstate(invalid='ignore'):
            mape0_ = _noninf_mean(np.abs(_out - out0) / np.abs(_out))
            mape01 = _noninf_mean(np.abs(out0 - out1) / np.abs(out0))
            if CAN_GPU:
                mape02 = _noninf_mean(np.abs(out0 - out2) / np.abs(out0))

        assert np.allclose(out0, _out, atol=atol), ("base",     dtype, mape0_)
        assert np.allclose(out0, out1, atol=atol), ("parallel", dtype, mape01)
        if CAN_GPU:
            assert np.allclose(out0, out2, atol=atol), ("gpu", dtype, mape02)


def test_replace_under_abs():
    np.random.seed(0)
    gamma = 1e-2
    for dtype in ('float32', 'float64'):
        w0 = np.random.randn(100, 200).astype(dtype)
        Wx = np.random.randn(100, 200).astype(dtype) * (2 - 1j)
        w1 = w0.copy()
        if CAN_GPU:
            wt  = torch.tensor(w0, device='cuda')
            Wxt = torch.tensor(Wx, device='cuda')

        replace_under_abs(w0, Wx, gamma, np.inf, parallel=False)
        replace_under_abs(w1, Wx, gamma, np.inf, parallel=True)
        if CAN_GPU:
            replace_under_abs(wt, Wxt, gamma, np.inf)
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
    np.random.seed(0)
    for dtype in ('float32', 'float64'):
      Wx = np.random.randn(100, 512).astype(dtype) * (1 + 2j)
      w  = np.abs(np.random.randn(*Wx.shape).astype(dtype))
      w *= (2*len(Wx) / w.max())
      if CAN_GPU:
          Wxt, wt = [torch.tensor(g, device='cuda') for g in (Wx, w)]

      for scaletype in ('log-piecewise', 'log', 'linear'):
        for flipud in (False, True):
          ssq_freqs = _make_ssq_freqs(len(Wx), scaletype)
          ssq_logscale = scaletype.startswith('log')
          const = (np.log(2) / 32 if 1 else
                   ssq_freqs)

          out0 = indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale,
                                   flipud=flipud, parallel=False)
          out1 = indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale,
                                   flipud=flipud, parallel=True)
          if CAN_GPU:
              out2 = indexed_sum_onfly(Wxt, wt, ssq_freqs, const, ssq_logscale,
                                       flipud=flipud).cpu().numpy()

          adiff01 = np.abs(out0 - out1).mean()
          if CAN_GPU:
              adiff02 = np.abs(out0 - out2).mean()
          # this is due to `const` varying rather than 'linear'
          th = ((1e-16 if dtype == 'float64' else 1e-8) if ssq_logscale else
                (1e-13 if dtype == 'float64' else 1e-5))
          assert adiff01 < th, (scaletype, dtype, flipud, adiff01)
          if CAN_GPU:
              assert adiff02 < th, (scaletype, dtype, flipud, adiff02)


def test_ssqueeze_cwt():
    np.random.seed(0)
    gamma = 1e-2
    for dtype in ('float32', 'float64'):
      Wx  = np.random.randn(100, 512).astype(dtype) * (1 + 2j)
      dWx = np.random.randn(100, 512).astype(dtype) * (2 - 1j)
      if CAN_GPU:
          Wxt, dWxt = [torch.tensor(g, device='cuda') for g in (Wx, dWx)]

      for scaletype in ('log-piecewise', 'log', 'linear'):
        for flipud in (False, True):
          ssq_freqs = _make_ssq_freqs(len(Wx), scaletype)
          ssq_logscale = scaletype.startswith('log')
          const = (np.log(2) / 32 if ssq_logscale else
                   ssq_freqs)

          args = (ssq_freqs, const, ssq_logscale)
          kw = dict(flipud=flipud, gamma=gamma)
          out0 = ssqueeze_fast(Wx, dWx, *args, **kw, parallel=False)
          out1 = ssqueeze_fast(Wx, dWx, *args, **kw, parallel=True)
          if CAN_GPU:
              out2 = ssqueeze_fast(Wxt, dWxt, *args, **kw).cpu().numpy()

          adiff01 = np.abs(out0 - out1).mean()
          if CAN_GPU:
              adiff02 = np.abs(out0 - out2).mean()
          # this is due to `const` varying rather than 'linear'
          th = ((1e-16 if dtype == 'float64' else 1e-8) if ssq_logscale else
                (1e-13 if dtype == 'float64' else 1e-5))
          assert adiff01 < th, (scaletype, dtype, flipud, adiff01)
          if CAN_GPU:
              assert adiff02 < th, (scaletype, dtype, flipud, adiff02)


def test_ssqueeze_stft():
    np.random.seed(0)
    scaletype = 'linear'
    ssq_logscale = False
    gamma = 1e-2
    const = np.log(2) / 32

    for dtype in ('float32', 'float64'):
      Sx  = np.random.randn(100, 512).astype(dtype) * (1 + 2j)
      dSx = np.random.randn(100, 512).astype(dtype) * (2 - 1j)
      if CAN_GPU:
          Sxt, dSxt = [torch.tensor(g, device='cuda') for g in (Sx, dSx)]

      for flipud in (False, True):
        ssq_freqs = _make_ssq_freqs(len(Sx), scaletype)

        args = (ssq_freqs, const, ssq_logscale)
        kw = dict(flipud=flipud, gamma=gamma)
        out0 = ssqueeze_fast(Sx, dSx, *args, **kw, parallel=False)
        out1 = ssqueeze_fast(Sx, dSx, *args, **kw, parallel=True)
        if CAN_GPU:
          out2 = ssqueeze_fast(Sxt, dSxt, *args, **kw).cpu().numpy()

        adiff01 = np.abs(out0 - out1).mean()
        if CAN_GPU:
          adiff02 = np.abs(out0 - out2).mean()
        # this is due to `const` varying rather than 'linear'
        th = (1e-16 if dtype == 'float64' else 1e-8)
        assert adiff01 < th, (scaletype, dtype, flipud, adiff01)
        if CAN_GPU:
          assert adiff02 < th, (scaletype, dtype, flipud, adiff02)


def test_ssqueeze_vs_indexed_sum():
    """Computing `Tx` in one loop vs. first computing `w` then summing."""
    np.random.seed(0)
    gamma = 1e-2
    for dtype in ('float32', 'float64'):
      Wx  = np.random.randn(100, 512).astype(dtype) * (1 + 2j)
      dWx = np.random.randn(100, 512).astype(dtype) * (2 - 1j)
      w = np.abs((dWx / Wx).imag / (2*np.pi))
      w[np.abs(Wx) < gamma] = np.inf

      for scaletype in ('log-piecewise', 'log', 'linear'):
        for flipud in (False, True):
          ssq_freqs = _make_ssq_freqs(len(Wx), scaletype)
          ssq_logscale = scaletype.startswith('log')
          const = (np.log(2) / 32 if ssq_logscale else
                   ssq_freqs)

          args = (ssq_freqs, const, ssq_logscale)
          kw = dict(parallel=False, flipud=flipud)
          out0 = indexed_sum_onfly(Wx, w, *args, **kw)
          out1 = ssqueeze_fast(Wx, dWx, *args, **kw, gamma=gamma)

          adiff01 = np.abs(out0 - out1).mean()
          # this is due to `const` varying rather than 'linear'
          th = ((1e-16 if dtype == 'float64' else 1e-8) if ssq_logscale else
                (1e-13 if dtype == 'float64' else 1e-5))
          assert adiff01 < th, (scaletype, dtype, flipud, adiff01)


def test_buffer():
    """Test that CPU & GPU outputs match for `modulated=True` & `=False`,
    and that `modulated=True` matches `ifftshift(buffer(modulated=False))`.
    Also that single- & multi-thread CPU outputs agree.

    Test both single and batched input.
    """
    N = 128
    tsigs = TestSignals(N=N)

    for dtype in ('float64', 'float32'):
      for ndim in (1, 2):
        x = (tsigs.cosine()[0].astype(dtype) if ndim == 1 else
             np.random.randn(4, N))
        xt = torch.as_tensor(x, device='cuda') if CAN_GPU else 0
        for modulated in (False, True):
          for seg_len in (N//2, N//2 - 1):
            for n_overlap in (N//2 - 1, N//2 - 2, N//2 - 3):
              if seg_len == n_overlap:
                continue

              out0 = buffer(x, seg_len, n_overlap, modulated, parallel=True)
              if modulated:
                  out00 = buffer(x, seg_len, n_overlap, modulated=False,
                                 parallel=False)
                  out00 = ifftshift(out00, axes=0 if ndim == 1 else 1)
              if CAN_GPU:
                  out1 = buffer(xt, seg_len, n_overlap, modulated).cpu().numpy()

              assert_params = (dtype, modulated, seg_len, n_overlap)
              if modulated:
                  adiff000 = np.abs(out0 - out00).mean()
                  assert adiff000 == 0, (*assert_params, adiff000)
              if CAN_GPU:
                  adiff01 = np.abs(out0 - out1).mean()
                  assert adiff01 == 0, (*assert_params, adiff01)


def test_ssq_stft():
    N = 256
    tsigs = TestSignals(N=N)
    gpu_atol = 1e-5

    for dtype in ('float64', 'float32'):
      x = tsigs.par_lchirp()[0].astype(dtype)
      kw = dict(modulated=1, n_fft=128, dtype=dtype, astensor=False)

      os.environ['SSQ_GPU'] = '0'
      Tx00 = ssq_stft(x, **kw, get_w=1)[0]
      Tx01 = ssq_stft(x, **kw, get_w=0)[0]
      if CAN_GPU:
          os.environ['SSQ_GPU'] = '1'
          Tx10 = ssq_stft(x, **kw, get_w=1)[0]
          Tx11 = ssq_stft(x, **kw, get_w=0)[0]

      adiff0001 = np.abs(Tx00 - Tx01).mean()
      assert np.allclose(Tx00, Tx01), (dtype, adiff0001)
      if CAN_GPU:
          adiff0010 = np.abs(Tx00 - Tx10).mean()
          adiff0011 = np.abs(Tx00 - Tx11).mean()
          assert np.allclose(Tx00, Tx10, atol=gpu_atol), (dtype, adiff0010)
          assert np.allclose(Tx00, Tx11, atol=gpu_atol), (dtype, adiff0011)


def test_ssq_cwt():
    N = 256
    tsigs = TestSignals(N=N)

    for dtype in ('float64', 'float32'):
      gpu_atol = 1e-8 if dtype == 'float64' else 2e-4
      x = tsigs.par_lchirp()[0].astype(dtype)
      kw = dict(astensor=False)

      os.environ['SSQ_GPU'] = '0'
      Tx00 = ssq_cwt(x, _wavelet(dtype=dtype), **kw, get_w=1)[0]
      Tx01 = ssq_cwt(x, _wavelet(dtype=dtype), **kw, get_w=0)[0]
      if CAN_GPU:
          os.environ['SSQ_GPU'] = '1'
          Tx10 = ssq_cwt(x, _wavelet(dtype=dtype), **kw, get_w=1)[0]
          Tx11 = ssq_cwt(x, _wavelet(dtype=dtype), **kw, get_w=0)[0]

      adiff0001 = np.abs(Tx00 - Tx01).mean()
      assert np.allclose(Tx00, Tx01), (dtype, adiff0001)
      if CAN_GPU:
          adiff0010 = np.abs(Tx00 - Tx10).mean()
          adiff0011 = np.abs(Tx00 - Tx11).mean()
          assert np.allclose(Tx00, Tx10, atol=gpu_atol), (dtype, adiff0010)
          assert np.allclose(Tx00, Tx11, atol=gpu_atol), (dtype, adiff0011)
    os.environ['SSQ_GPU'] = '0'


def test_wavelet_dtype_gmw():
    """Ensure `Wavelet.fn` output is of specified `dtype` for GMW wavelet,
    and that `.info()` is computable.
    """
    for SSQ_GPU in ('0', '1'):
      if SSQ_GPU == '1' and not CAN_GPU:
        continue
      for order in (0, 1):
        for norm in ('bandpass', 'energy'):
          for dtype in ('float64', 'float32'):
            os.environ['SSQ_GPU'] = SSQ_GPU
            kw = dict(order=order, norm=norm, dtype=dtype)
            wavelet = _wavelet('gmw', **kw)
            if norm == 'energy':
                dtype = 'float64'
            assert wavelet.dtype == dtype, (
                "GPU={}, order={}, norm={}, dtype={}, wavelet.dtype={}".format(
                    SSQ_GPU, order, norm, dtype, wavelet.dtype))
            wavelet.info()
    os.environ['SSQ_GPU'] = '0'


def test_wavelet_dtype():
    """Ensure `Wavelet.fn` output is of specified `dtype` for non-GMW wavelets,
    and that `.info()` is computable.
    """
    for SSQ_GPU in ('0', '1'):
      if SSQ_GPU == '1' and not CAN_GPU:
        continue
      for name in ('morlet', 'bump', 'cmhat', 'hhhat'):
        for dtype in ('float64', 'float32'):
          os.environ['SSQ_GPU'] = SSQ_GPU
          wavelet = _wavelet(name, dtype=dtype)
          assert wavelet.dtype == dtype, (
              "GPU={}, name={}, dtype={}, wavelet.dtype={}".format(
                  SSQ_GPU, name, dtype, wavelet.dtype))
          wavelet.info()
    os.environ['SSQ_GPU'] = '0'


def test_higher_order():
    """`cwt` & `ssq_cwt` CPU & GPU outputs agreement."""
    if not CAN_GPU:
        return

    tsigs = TestSignals(N=256)
    x = tsigs.par_lchirp()[0]
    x += x[::-1]

    kw = dict(order=range(3), astensor=False)
    for dtype in ('float32', 'float64'):
        os.environ['SSQ_GPU'] = '0'
        Tx0, Wx0, *_ = ssq_cwt(x, _wavelet(dtype=dtype), **kw)
        os.environ['SSQ_GPU'] = '1'
        Tx1, Wx1, *_ = ssq_cwt(x, _wavelet(dtype=dtype), **kw)

        adiff_Tx = np.abs(Tx0 - Tx1).mean()
        adiff_Wx = np.abs(Wx0 - Wx1).mean()
        th = 1e-6  # less should be possible for float64, but didn't investigate
        assert adiff_Tx < th, (dtype, th)
        assert adiff_Wx < th, (dtype, th)
    os.environ['SSQ_GPU'] = '0'


def test_cwt_for_loop():
    """Ensure `vectorized=False` runs on GPU and outputs match `=True`."""
    if not CAN_GPU:
        return
    np.random.seed(0)
    x = np.random.randn(256)
    kw = dict(derivative=True, astensor=False)

    os.environ['SSQ_GPU'] = '1'
    for dtype in ('float64', 'float32'):
        Wx0, _, dWx0 = cwt(x, _wavelet(dtype=dtype), vectorized=False, **kw)
        Wx1, _, dWx1 = cwt(x, _wavelet(dtype=dtype), vectorized=True,  **kw)

        adiff_Wx  = np.abs(Wx0 - Wx1)
        adiff_dWx = np.abs(dWx0 - dWx1)
        atol = 1e-12 if dtype == 'float64' else 1e-6
        assert np.allclose(Wx0,  Wx1,  atol=atol), (dtype, adiff_Wx.mean())
        assert np.allclose(dWx0, dWx1, atol=atol), (dtype, adiff_dWx.mean())


def test_ssq_cwt_batched():
    """Ensure batched (2D `x`) inputs output same as if samples fed separately,
    and agreement between CPU & GPU.
    """
    np.random.seed(0)
    x = np.random.randn(4, 256)
    kw = dict(astensor=False)

    for dtype in ('float64', 'float32'):
        os.environ['SSQ_GPU'] = '0'
        Tx0, Wx0, *_ = ssq_cwt(x, _wavelet(dtype=dtype), **kw)

        Tx00 = np.zeros(Tx0.shape, dtype=Tx0.dtype)
        Wx00 = Tx00.copy()
        for i, _x in enumerate(x):
            out = ssq_cwt(_x, _wavelet(dtype=dtype), **kw)
            Tx00[i], Wx00[i] = out[0], out[1]

        if CAN_GPU:
            os.environ['SSQ_GPU'] = '1'
            Tx1, Wx1, *_ = ssq_cwt(x, _wavelet(dtype=dtype), **kw)

        atol = 1e-12 if dtype == 'float64' else 1e-2
        adiff_Tx000 = np.abs(Tx00 - Tx0).mean()
        adiff_Wx000 = np.abs(Wx00 - Wx0).mean()
        assert np.allclose(Wx00, Wx0), (dtype, adiff_Wx000)
        assert np.allclose(Tx00, Tx0), (dtype, adiff_Tx000)
        if CAN_GPU:
            adiff_Tx01  = np.abs(Tx0 - Tx1).mean()
            adiff_Wx01  = np.abs(Wx0 - Wx1).mean()
            assert np.allclose(Wx0, Wx1, atol=atol), (dtype, adiff_Wx01)
            assert np.allclose(Tx0, Tx1, atol=atol), (dtype, adiff_Tx01)

            # didn't investigate float32, and `allclose` threshold is pretty bad,
            # so check MAE
            if dtype == 'float32':
                assert adiff_Tx01 < 1e-6, (dtype, adiff_Tx01)


def test_ssq_stft_batched():
    """Ensure batched (2D `x`) inputs output same as if samples fed separately,
    and agreement between CPU & GPU.
    """
    np.random.seed(0)
    x = np.random.randn(4, 256)

    for dtype in ('float64', 'float32'):
        os.environ['SSQ_GPU'] = '0'
        kw = dict(astensor=False, dtype=dtype)
        Tx0, Sx0, *_ = ssq_stft(x, **kw)

        Tx00 = np.zeros(Tx0.shape, dtype=Tx0.dtype)
        Sx00 = Tx00.copy()
        for i, _x in enumerate(x):
            out = ssq_stft(_x, **kw)
            Tx00[i], Sx00[i] = out[0], out[1]

        if CAN_GPU:
            os.environ['SSQ_GPU'] = '1'
            Tx1, Sx1, *_ = ssq_stft(x, **kw)

        atol = 1e-12 if dtype == 'float64' else 1e-6
        adiff_Tx000 = np.abs(Tx00 - Tx0).mean()
        adiff_Sx000 = np.abs(Sx00 - Sx0).mean()
        assert np.allclose(Sx00, Sx0), (dtype, adiff_Sx000)
        assert np.allclose(Tx00, Tx0), (dtype, adiff_Tx000)
        if CAN_GPU:
            adiff_Tx01  = np.abs(Tx0 - Tx1)
            adiff_Sx01  = np.abs(Sx0 - Sx1)
            assert np.allclose(Sx0, Sx1, atol=atol), (dtype, adiff_Sx01)
            assert np.allclose(Tx0, Tx1, atol=atol), (dtype, adiff_Tx01)


def test_cwt_batched_for_loop():
    """Ensure basic batched cwt works with both `vectorized`."""
    os.environ['SSQ_GPU'] = '0'
    np.random.seed(0)
    x = np.random.randn(4, 256)

    for dtype in ('float64', 'float32'):
        Wx0, *_ = cwt(x, _wavelet(dtype=dtype), vectorized=True)
        Wx1, *_ = cwt(x, _wavelet(dtype=dtype), vectorized=False)

        adiff_Wx01  = np.abs(Wx0 - Wx1)
        assert np.allclose(Wx0, Wx1), (dtype, adiff_Wx01.mean())


if __name__ == '__main__':
    if VIZ:
        test_1D()
        test_2D()
        test_indexed_sum()
        test_parallel_setting()
        test_phase_cwt()
        test_phase_stft()
        test_replace_under_abs()
        test_indexed_sum_onfly()
        test_ssqueeze_cwt()
        test_ssqueeze_stft()
        test_ssqueeze_vs_indexed_sum()
        test_buffer()
        test_ssq_stft()
        test_ssq_cwt()
        test_wavelet_dtype_gmw()
        test_wavelet_dtype()
        test_higher_order()
        test_cwt_for_loop()
        test_ssq_cwt_batched()
        test_ssq_stft_batched()
        test_cwt_batched_for_loop()
    else:
        pytest.main([__file__, "-s"])
