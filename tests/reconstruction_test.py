# -*- coding: utf-8 -*-
import pytest
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt
from ssqueezepy import cwt, icwt

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
