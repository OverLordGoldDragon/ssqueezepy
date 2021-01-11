# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt, ssq_stft, issq_stft
from ssqueezepy import cwt, icwt, stft, istft
from ssqueezepy._stft import _get_window
from ssqueezepy.toolkit import lin_band

VIZ = 0  # set to 1 to enable various visuals and run without pytest


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
    return _freqs(N, np.array([N/100, N/200, N/2, N/20,
                               N/2-1, N/50, N/3, N/150]) / 8)

def low_freqs(N):
    return _freqs(N, [.3, .3, 1, 1, 2, 2])

def high_freqs(N):
    return _freqs(N, np.array([N/2, N/2-1, N/20, N/4]) / 4)


#### Tests ###################################################################
test_fns = (echirp, lchirp, fast_transitions, low_freqs, high_freqs)
wavelet = ('morlet', {'mu': 5})
th = .1


def test_ssq_cwt():
    errs = []
    for fn in test_fns:
        x, ts = fn(2048)
        for scales in ('log', 'linear'):
            # 'linear' default can't handle low frequencies for large N
            if scales == 'linear' and fn.__name__ == 'fast_transitions':
                continue

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
            # 'linear' default can't handle low frequencies for large N
            kw = dict(wavelet=wavelet, scales='log', l1_norm=l1_norm, nv=32)

            Wx, *_ = cwt(x, t=ts, **kw)
            xrec = icwt(Wx, one_int=True, **kw)

            errs.append(round(mad_rms(x, xrec), 5))
            title = f"abs(CWT) | l1_norm={l1_norm}"
            title = "abs(SSQ_CWT) | {}, l1_norm={}".format(fn.__qualname__,
                                                           l1_norm)
            _maybe_viz(Wx, x, xrec, title, errs[-1])
            assert errs[-1] < th, (errs[-1], fn.__name__, f"l1_norm: {l1_norm}")
    print("\ncwt PASSED\nerrs:", ', '.join(map(str, errs)))


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

    wavelet = ('morlet', {'mu': 4.5})
    Tx, *_ = ssq_cwt(x, wavelet, scales='log:maximal', nv=32, t=ts)

    # hand-coded, subject to failure
    bw, slope, offset = .035, .45, .45
    Cs, freqband = lin_band(Tx, slope, offset, bw, norm=(0, 2e-1))

    xrec = issq_cwt(Tx, wavelet, Cs, freqband)[0]

    axof   = np.abs(np.fft.rfft(xo))
    axrecf = np.abs(np.fft.rfft(xrec))

    err_sig = mad_rms(xo, xrec)
    err_spc = mad_rms(axof, axrecf)
    print("signal   MAD/RMS: %.6f" % err_sig)
    print("spectrum MAD/RMS: %.6f" % err_spc)
    assert err_sig <= .42, f"{err_sig} > .42"
    assert err_spc <= .14, f"{err_spc} > .14"


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

                    Sx, *_ = stft(x, **kw)
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
                    window = _get_window(window, win_len=n_fft//1, n_fft=n_fft)
                    window *= window_scaling

                Sx, *_ = ssq_stft(x, window=window, n_fft=n_fft)
                xr = issq_stft(Sx, window=window, n_fft=n_fft, N=N)

                txt = ("\nSSQ_STFT: (N, n_fft, window_scaling) = ({}, {}, {})"
                       ).format(N, n_fft, window_scaling)
                assert len(x) == len(xr), "%s != %s %s" % (N, len(xr), txt)
                mae = np.abs(x - xr).mean()
                assert mae < th, "MAE = %.2e > %.2e %s" % (mae, th, txt)


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
    else:
        pytest.main([__file__, "-s"])
