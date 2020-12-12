# -*- coding: utf-8 -*-
"""Lazy tests just to ensure nothing breaks
"""
import pytest
import numpy as np
from ssqueezepy.wavelets import Wavelet
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
        # minbounds=(True,),
        squeezing=('lebesgue',),
        scales=('linear', np.power(2**(1/16), np.arange(1, 32))),
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
    Wx, *_ = cwt(x, 'morlet')
    icwt(Wx, 'morlet')


def test_wavelets():
    for wavelet in ('morlet', ('morlet', {'mu': 4}), 'bump'):
        psihfn = Wavelet(wavelet)

    psihfn = Wavelet('morlet')
    psihfn.info()
    psihfn.viz()


if VIZ:
    test_ssq_cwt()
    test_cwt()
    test_wavelets()

elif __name__ == '__main__':
    pytest.main([__file__, "-s"])
