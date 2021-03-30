# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
from ssqueezepy import cwt, icwt, stft, istft
from ssqueezepy._stft import get_window
from ssqueezepy.toolkit import lin_band
try:
    from librosa import stft as lstft
except Exception as e:
    import logging
    logging.warn("librosa import failed with:\n%s" % str(e))


VIZ = 0  # set to 1 to enable various visuals and run without pytest
os.environ['SSQ_GPU'] = '0'  # in case concurrent tests set it to '1'


#### Helper methods ##########################################################
def _t(min, max, N):
    return np.linspace(min, max, N, endpoint=False)

def cos_f(freqs, N=128, phi=0):
    return np.concatenate([np.cos(2 * np.pi * f * (_t(i, i + 1, N) + phi))
                           for i, f in enumerate(freqs)])

def mad_rms(x, xrec):
    """Reconstruction error metric; scale-invariant, robust to outliers
    and partly sparsity. https://stats.stackexchange.com/q/495242/239063"""
    return np.mean(np.abs(x - xrec)) / np.sqrt(np.mean(x**2))

#### Test signals ############################################################
def echirp(N):
    t = _t(0, 10, N)
    return np.cos(2 * np.pi * 3 * np.exp(t / 3)), t

def lchirp(N):
    t = _t(0, 10, N)
    return np.cos(np.pi * t**2), t


def _freqs(N, freqs):
    x = cos_f(freqs, N // len(freqs))
    ts = _t(0, len(x) / N, len(x))
    return x, ts

def fast_transitions(N):
    return _freqs(N, np.array([N/100, N/200, N/3, N/20,
                               N/3-1, N/50, N/4, N/150]) / 8)

def low_freqs(N):
    return _freqs(N, [.3, .3, 1, 1, 2, 2])

def high_freqs(N):
    return _freqs(N, np.array([N/2, N/2-1, N/4, N/3]) / 4)


#### Tests ###################################################################
test_fns = (echirp, lchirp, fast_transitions, low_freqs, high_freqs)
wavelet = ('gmw', {'beta': 8, 'dtype': 'float64'})
th = .1


def test_ssq_cwt():
    errs = []
    for fn in test_fns:
        x, ts = fn(2048)
        for scales in ('log', 'log-piecewise', 'linear'):
            if fn.__name__ == 'low_freqs':
                if scales == 'linear':
                    # 'linear' default can't handle low frequencies for large N
                    # 'log-piecewise' maps it too sparsely
                    continue
                else:
                    scales = f'{scales}:maximal'

            Tx, *_ = ssq_cwt(x, wavelet, scales=scales, nv=32, t=ts)
            xrec = issq_cwt(Tx, wavelet)

            errs.append(round(mad_rms(x, xrec), 5))
            title = "abs(SSQ_CWT) | {}, scales='{}'".format(fn.__qualname__,
                                                            scales)
            _maybe_viz(Tx, x, xrec, title, errs[-1])
            assert errs[-1] < th, (errs[-1], fn.__name__, scales)
    print("\nssq_cwt PASSED\nerrs:", ', '.join(map(str, errs)))


def test_cwt():
    errs = []
    for fn in test_fns:
        x, ts = fn(2048)
        for l1_norm in (True, False):
            scales = ('log:maximal' if fn.__name__ in ('low_freqs', 'high_freqs')
                      else 'log')
            # 'linear' default can't handle low frequencies for large N
            kw = dict(wavelet=wavelet, scales=scales, l1_norm=l1_norm, nv=32)

            Wx, *_ = cwt(x, t=ts, **kw)
            xrec = icwt(Wx, one_int=True, **kw)

            errs.append(round(mad_rms(x, xrec), 5))
            title = f"abs(CWT) | l1_norm={l1_norm}"
            title = "abs(CWT) | {}, l1_norm={}".format(fn.__qualname__,
                                                       l1_norm)
            _maybe_viz(Wx, x, xrec, title, errs[-1])
            assert errs[-1] < th, (errs[-1], fn.__name__, f"l1_norm: {l1_norm}")
    print("\ncwt PASSED\nerrs:", ', '.join(map(str, errs)))


def test_cwt_log_piecewise():
    x, ts = echirp(1024)

    wavelet = 'gmw'
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(x, wavelet, scales='log-piecewise',
                                            t=ts, preserve_transform=True)
    xrec_ssq_cwt = issq_cwt(Tx, 'gmw')
    xrec_cwt = icwt(Wx, wavelet, scales=scales)

    err_ssq_cwt = round(mad_rms(x, xrec_ssq_cwt), 5)
    err_cwt = round(mad_rms(x, xrec_cwt), 5)
    assert err_ssq_cwt < .02, err_ssq_cwt
    assert err_cwt < .02, err_cwt


def test_component_inversion():
    def echirp(N):
        t = np.linspace(0, 10, N, False)
        return np.cos(2 * np.pi * np.exp(t / 3)), t

    N = 2048
    noise_var = 6

    x, ts = echirp(N)
    x *= (1 + .3 * cos_f([1], N))  # amplitude modulation
    xo = x.copy()
    np.random.seed(4)
    x += np.sqrt(noise_var) * np.random.randn(len(x))

    wavelet = ('gmw', {'beta': 6})
    Tx, *_ = ssq_cwt(x, wavelet, scales='log:maximal', nv=32, t=ts, flipud=0)

    # hand-coded, subject to failure
    bw, slope, offset = .035, .44, .58
    Cs, freqband = lin_band(Tx, slope, offset, bw, norm=(0, 2e-1))

    xrec = issq_cwt(Tx, wavelet, Cs, freqband)[0]

    axof   = np.abs(np.fft.rfft(xo))
    axrecf = np.abs(np.fft.rfft(xrec))

    err_sig = mad_rms(xo, xrec)
    err_spc = mad_rms(axof, axrecf)
    print("signal   MAD/RMS: %.6f" % err_sig)
    print("spectrum MAD/RMS: %.6f" % err_spc)
    assert err_sig <= .40, f"{err_sig} > .40"
    assert err_spc <= .15, f"{err_spc} > .15"


def test_stft():
    """Ensure every combination of even & odd configs can be handled;
    leave window length unspecified to ensure unspecified inverts unspecified.
    """
    th = 1e-14
    for N in (128, 129):
      x = np.random.randn(N)
      for n_fft in (120, 121):
        for hop_len in (1, 2, 3):
          for modulated in (True, False):
            kw = dict(hop_len=hop_len, n_fft=n_fft, modulated=modulated)

            Sx = stft(x, dtype='float64', **kw)
            xr = istft(Sx, N=len(x), **kw)

            txt = ("\nSTFT: (N, n_fft, hop_len, modulated) = ({}, {}, "
                   "{}, {})").format(N, n_fft, hop_len, modulated)
            assert len(x) == len(xr), "%s != %s %s" % (N, len(xr), txt)
            mae = np.abs(x - xr).mean()
            assert mae < th, "MAE = %.2e > %.2e %s" % (mae, th, txt)


def test_ssq_stft():
    """Same as `test_stft` except don't test `hop_len` or `modulated` since
    only `1` and `True` are invertible (by the library, and maybe theoretically).

    `window_scaling=.5` has >x2 greater MAE for some reason. May look into.
    """
    th = 1e-1
    for N in (128, 129):
      x = np.random.randn(N)
      for n_fft in (120, 121):
        for window_scaling in (1., .5):
          if window_scaling == 1:
              window = None
          else:
              window = get_window(window, win_len=n_fft//1, n_fft=n_fft)
              window *= window_scaling

          Sx, *_ = ssq_stft(x, window=window, n_fft=n_fft)
          xr = issq_stft(  Sx, window=window, n_fft=n_fft)

          txt = ("\nSSQ_STFT: (N, n_fft, window_scaling) = ({}, {}, {})"
                 ).format(N, n_fft, window_scaling)
          assert len(x) == len(xr), "%s != %s %s" % (N, len(xr), txt)
          mae = np.abs(x - xr).mean()
          assert mae < th, "MAE = %.2e > %.2e %s" % (mae, th, txt)


def test_stft_vs_librosa():
    try:
        lstft
    except:
        return

    np.random.seed(0)
    # try all even/odd combos
    for N in (512, 513):
      for hop_len in (1, 2, 3):
        for n_fft in (512, 513):
          for win_len in (N//8, N//8 - 1):
             x = np.random.randn(N)
             Sx  = stft( x, n_fft=n_fft, hop_len=hop_len,    win_len=win_len,
                         window='hann', modulated=False, dtype='float64')
             lSx = lstft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                         window='hann')

             if n_fft % 2 == 0:
                 if hop_len == 1:
                     lSx = lSx[:, :-1]
                 elif (((N % 2 == 0) and hop_len == 2) or
                       ((N % 2 == 1) and hop_len == 3)):
                     lSx = lSx[:, :-1]
             mae = np.abs(Sx - lSx).mean()
             assert np.allclose(Sx, lSx), (
                 "N={}, hop_len={}, n_fft={}, win_len={}, MAE={}"
                 ).format(N, hop_len, n_fft, win_len, mae)


def _maybe_viz(Wx, xo, xrec, title, err):
    if not VIZ:
        return
    mx = np.abs(Wx).max()
    if 'SSQ' in title:
        Wx = np.pad(np.flipud(Wx), [[5], [0]])
        mx = .1*mx
    else:
        mx = .9*mx

    imshow(Wx, abs=1, norm=(0, mx), cmap='jet', show=1, title=title)
    plot(xo, title="Original vs reconstructed | MAD/RMS=%.4f" % err)
    plot(xrec, show=1)


if __name__ == '__main__':
    if VIZ:
        from ssqueezepy.visuals import plot, imshow
        test_ssq_cwt()
        test_cwt()
        test_cwt_log_piecewise()
        test_component_inversion()
        test_stft()
        test_ssq_stft()
        test_stft_vs_librosa()
    else:
        pytest.main([__file__, "-s"])
