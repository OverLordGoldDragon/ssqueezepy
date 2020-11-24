import pytest
import numpy as np
from ssqueezepy.utils import adm_cwt, adm_ssq

VIZ = 0  # set to 1 to visualize wavelet adm-coef dependence on their params


def _test(make_wavelet, params, th=1e-3):
    acwt = np.zeros(len(params))
    assq = acwt.copy()

    for i, param in enumerate(params):
        wavelet = make_wavelet(param)
        acwt[i] = adm_cwt(wavelet)
        assq[i] = adm_ssq(wavelet)

    _maybe_viz(acwt, assq, params)
    if not np.all(acwt > th):
        raise AssertionError(f"th={th}")
    if not np.all(assq > th):
        raise AssertionError(f"th={th}")


def test_morlet():
    mus = np.linspace(4, 30, 200)
    make_wavelet = lambda mu: ('morlet', {'mu': mu})
    _test(make_wavelet, params=mus)


def test_bump():
    mus = np.linspace(4, 30, 200)
    make_wavelet = lambda mu: ('bump', {'mu': mu})
    _test(make_wavelet, params=mus)


def test_cmhat():
    mus = np.linspace(4, 30, 200)
    make_wavelet = lambda mu: ('cmhat', {'mu': mu})
    _test(make_wavelet, params=mus)


def test_hhhat():
    mus = np.linspace(4, 30, 200)
    make_wavelet = lambda mu: ('hhhat', {'mu': mu})
    _test(make_wavelet, params=mus)


def _maybe_viz(acwt, assq, params):
    if VIZ:
        import matplotlib.pyplot as plt
        plt.plot(params, acwt)
        plt.plot(params, assq)
        mx = max(acwt.max(), assq.max())
        plt.ylim(-.05 * mx, None)
        plt.show()


if __name__ == '__main__':
    pytest.main([__file__, "-s"])
