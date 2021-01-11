# -*- coding: utf-8 -*-
"""Lazy tests just to ensure nothing breaks
"""
#### Disable Numba JIT during testing, as pytest can't measure its coverage ##
# TODO find shorter way to do this
print("numba.njit is now monkey")
def njit(fn):
    def decor(*args, **kw):
        return fn(*args, **kw)
    return decor

import numba
njit_orig = numba.njit
numba.njit = njit
##############################################################################
import pytest
import numpy as np
from ssqueezepy.wavelets import Wavelet, center_frequency, freq_resolution
from ssqueezepy.wavelets import time_resolution, _xifn
from ssqueezepy.wavelets import _aifftshift_even, _afftshift_even
from ssqueezepy.utils import cwt_scalebounds, buffer, est_riskshrink_thresh
from ssqueezepy._cwt import _icwt_norm
from ssqueezepy.visuals import hist, plot, scat, imshow
from ssqueezepy.toolkit import lin_band, cos_f, mad_rms
from ssqueezepy import ssq_cwt, issq_cwt, cwt, icwt, ssqueeze

#### Ensure cached imports reloaded ##########################################
from types import ModuleType
from imp import reload
import ssqueezepy

reload(numba)
numba.njit = njit
reload(ssqueezepy)
for name in dir(ssqueezepy):
    obj = getattr(ssqueezepy, name)
    if isinstance(obj, ModuleType):
        reload(obj)
##############################################################################

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0


def test_ssq_cwt():
    x = np.random.randn(64)
    for wavelet in ('morlet', ('morlet', {'mu': 20}), 'bump'):
        Tx, *_ = ssq_cwt(x, wavelet)
        issq_cwt(Tx, wavelet)

    kw = dict(x=x, wavelet='morlet')
    params = dict(
        squeezing=('lebesgue',),
        scales=('linear', 'log:minimal', 'linear:naive',
                np.power(2**(1/16), np.arange(1, 32))),
        difftype=('phase', 'numeric'),
        padtype=('zero', 'replicate'),
        mapkind=('energy', 'peak'),
    )

    for name in params:
        for value in params[name]:
            try:
                _ = ssq_cwt(**kw, **{name: value})
            except Exception as e:
                raise Exception(f"{name}={value} failed with:\n{e}")

    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=2)
    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=1)


def test_cwt():
    x = np.random.randn(64)
    Wx, *_ = cwt(x, 'morlet', vectorized=True)
    _ = icwt(Wx, 'morlet', one_int=True)
    _ = icwt(Wx, 'morlet', one_int=False)

    Wx2, *_ = cwt(x, 'morlet', vectorized=False)
    mae = np.mean(np.abs(Wx - Wx2))
    assert mae <= 1e-16, f"MAE = {mae} > 1e-16 for for-loop vs vectorized `cwt`"

    _ = est_riskshrink_thresh(Wx, nv=32)
    _ = _icwt_norm(scaletype='linear', l1_norm=False)

    x[0] = np.nan
    x[1] = np.inf
    x[2] = -np.inf
    _ = cwt(x, 'morlet', vectorized=False, derivative=True, l1_norm=False)


def test_wavelets():
    for wavelet in ('morlet', ('morlet', {'mu': 4}), 'bump'):
        wavelet = Wavelet(wavelet)

    wavelet = Wavelet(('morlet', {'mu': 5}))
    wavelet.viz(name='overview')
    wavelet.info(nondim=1)
    wavelet.info(nondim=0)

    #### Visuals #############################################################
    for name in wavelet.VISUALS:
        if 'anim:' in name:  # heavy-duty computations, skip animating
            kw = {'testing': True}
        else:
            kw = {}
        try:
            wavelet.viz(name, N=256, **kw)
        except TypeError as e:
            if "positional argument" not in str(e):
                raise TypeError(e)
            try:
                wavelet.viz(name, scale=10, N=256, **kw)
            except TypeError as e:
                if "positional argument" not in str(e):
                    raise TypeError(e)
                wavelet.viz(name, scales='log', N=256, **kw)

    _ = cwt_scalebounds(wavelet, N=512, viz=3)

    #### misc ################################################################
    wavelet = Wavelet(lambda x: x)
    _ = _xifn(scale=10, N=128)


def test_toolkit():
    Tx = np.random.randn(20, 20)
    Cs, freqband = lin_band(Tx, slope=1, offset=.1, bw=.025)

    _ = cos_f([1], N=64)
    _ = mad_rms(np.random.randn(10), np.random.randn(10))


def test_visuals():
    x = np.random.randn(10)
    hist(x, show=1, stats=1)

    y = x * (1 + 1j)
    plot(y, complex=1, c_annot=1, vlines=1, ax_equal=1,
         xticks=np.arange(len(y)), yticks=y)

    scat(x, vlines=1, hlines=1)
    imshow(np.random.randn(4, 4), complex=1)


def test_utils():
    _ = buffer(np.random.randn(20), 4, 1)

    wavelet = Wavelet(('morlet', {'mu': 6}))
    _ = center_frequency(wavelet, viz=1)
    _ = freq_resolution( wavelet, viz=1, scale=3, force_int=0)
    _ = time_resolution( wavelet, viz=1)

    xh = np.random.randn(128)
    xhs = np.zeros(xh.size)
    _aifftshift_even(xh, xhs)
    _afftshift_even(xh, xhs)


def test_anim():
    # bare minimally (still takes long, but covers many lines of code)
    wavelet = Wavelet(('morlet', {'mu': 6}))
    wavelet.viz('anim:time-frequency', N=8, scales=np.linspace(10, 20, 3))


def test_ssqueezing():
    def _pass_on_error(fn, *args, **kw):
        try: fn(*args, **kw)
        except: pass

    Wx = np.random.randn(4, 4)
    w = np.abs(Wx)

    _pass_on_error(ssqueeze, Wx, w, transform='greenland')
    _pass_on_error(ssqueeze, Wx, w, transform='cwt', scales=None)
    _pass_on_error(ssqueeze, Wx, w, transform='cwt', wavelet=None,
                   mapkind='maximal')
    _pass_on_error(ssqueeze, Wx, w, transform='stft', mapkind='minimal')
    _pass_on_error(ssqueeze, Wx, w, transform='abs')
    _pass_on_error(ssqueeze, Wx, w, squeezing='big_bird')


if __name__ == '__main__':
    if VIZ:
        test_ssq_cwt()
        test_cwt()
        test_wavelets()
        test_toolkit()
        test_visuals()
        test_utils()
        test_anim()
    else:
        pytest.main([__file__, "-s"])

        # restore original in case it matters for future testing
        reload(numba)
        numba.njit = njit_orig
        reload(ssqueezepy)
        for name in dir(ssqueezepy):
            obj = getattr(ssqueezepy, name)
            if isinstance(obj, ModuleType):
                reload(obj)
        print("numba.njit is no longer monkey")
