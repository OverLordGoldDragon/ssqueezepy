import pytest
import numpy as np
from ssqueezepy import ssq_cwt, issq_cwt
from ssqueezepy import cwt, icwt


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
def fast_transitions(N):
    x = cos_f([.3, .3, 127, 40, 20, 4, 2, 1], N)
    ts = _t(0, len(x) / N, len(x))
    return x, ts

def echirp(N):
    t = _t(0, 10, N)
    return np.cos(2 * np.pi * 3 * np.exp(t / 3)), t

def lchirp(N):
    t = _t(0, 10, N)
    return np.cos(np.pi * t**2), t

#### Tests ###################################################################
test_fns = (echirp, lchirp, fast_transitions)
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
            assert errs[-1] < th, (errs[-1], fn.__name__, scales)
    print("ssq_cwt PASSED\nerrs:", ', '.join(map(str, errs)))


def test_cwt():
    errs = []
    for fn in test_fns:
        x, ts = fn(2048)
        for l1_norm in (True, False):
            # 'linear' default can't handle low frequencies for large N
            kw = dict(wavelet=wavelet, scales='log', l1_norm=l1_norm)

            Wx, *_ = cwt(x, t=ts, **kw)
            xrec = icwt(Wx, one_int=True, **kw)

            errs.append(round(mad_rms(x, xrec), 5))
            assert errs[-1] < th, (errs[-1], fn.__name__, f"l1_norm: {l1_norm}")
    print("cwt PASSED\nerrs:", ', '.join(map(str, errs)))


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
