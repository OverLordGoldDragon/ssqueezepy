# -*- coding: utf-8 -*-
"""Lazy tests just to ensure nothing breaks
"""
import pytest
import numpy as np
from ssqueezepy.wavelets import Wavelet
from ssqueezepy.utils import cwt_scalebounds, buffer, est_riskshrink_thresh
from ssqueezepy.visuals import hist, plot, scat
from ssqueezepy.toolkit import lin_band, cos_f, mad_rms
from ssqueezepy import ssq_cwt, issq_cwt, cwt, icwt

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
        difftype=('phase', 'numerical'),
        padtype=('zero', 'replicate'),
        mapkind=('energy', 'peak'),
    )

    for name in params:
        for value in params[name]:
            try:
                ssq_cwt(**kw, **{name: value})
            except Exception as e:
                raise Exception(f"{name}={value} failed with:\n{e}")


def test_cwt():
    x = np.random.randn(64)
    Wx, *_ = cwt(x, 'morlet', vectorized=True)
    _ = icwt(Wx, 'morlet', one_int=True)
    _ = icwt(Wx, 'morlet', one_int=False)

    Wx2, *_ = cwt(x, 'morlet', vectorized=False)
    mae = np.mean(np.abs(Wx - Wx2))
    assert mae <= 1e-16, f"MAE = {mae} > 1e-16 for for-loop vs vectorized `cwt`"

    _ = est_riskshrink_thresh(Wx, nv=32)


def test_wavelets():
    for wavelet in ('morlet', ('morlet', {'mu': 4}), 'bump'):
        wavelet = Wavelet(wavelet)

    wavelet = Wavelet('morlet')
    wavelet.info()

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



def test_toolkit():
    Tx = np.random.randn(20, 20)
    Cs, freqband = lin_band(Tx, slope=1, offset=.1, bw=.025)

    _ = cos_f([1], N=64)
    _ = mad_rms(np.random.randn(10), np.random.randn(10))


def test_visuals():
    x = np.random.randn(10)
    hist(x, show=1, stats=1)

    y = x * (1 + 1j)
    plot(y, complex=1, c_annot=1, vlines=1, ax_equal=1)

    scat(x, vlines=1, hlines=1)


def test_utils():
    _ = buffer(np.random.randn(20, 20), 5, 1)


if __name__ == '__main__':
    if VIZ:
        test_ssq_cwt()
        test_cwt()
        test_wavelets()
    else:
        pytest.main([__file__, "-s"])
