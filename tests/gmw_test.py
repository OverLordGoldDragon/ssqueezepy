# -*- coding: utf-8 -*-
"""Generalized Morse Wavelets.

Tests:
    - Implementations are `Wavelet`-compatible
    - Consistency of `Wavelet`-compatible implems with that of full `morsewave`
    - GMW L1 & L2 norms work as expected
"""
import os
import pytest
import numpy as np
from ssqueezepy.wavelets import Wavelet
from ssqueezepy._gmw import compute_gmw, morsewave

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0
os.environ['SSQ_GPU'] = '0'  # in case concurrent tests set it to '1'


def test_api_vs_full():
    os.environ['SSQ_GPU'] = '0'
    for gamma, beta in [(3, 60), (4, 80)]:
      for norm in ('bandpass', 'energy'):
        for scale in (1, 2):
          for N in (512, 513):
            kw = dict(N=N, gamma=gamma, beta=beta, norm=norm)
            kw2 = dict(scale=scale, time=True, centered_scale=True,
                       norm_scale=True, dtype='float64')
            psih_s, psi_s = compute_gmw(**kw, **kw2)
            psih_f, psi_f = morsewave(**kw, freqs=1 / scale)

            mad_t = np.mean(np.abs(psi_s - psi_f))
            mad_f = np.mean(np.abs(psih_s - psih_f))
            assert np.allclose(psi_s, psi_f),   errmsg(mad_t, **kw, **kw2)
            assert np.allclose(psih_s, psih_f), errmsg(mad_f, **kw, **kw2)


def test_api_vs_full_higher_order():
    os.environ['SSQ_GPU'] = '0'
    for gamma, beta in [(3, 60), (4, 80)]:
      for order in (1, 2):
        for norm in ('bandpass', 'energy'):
          for scale in (1, 2):
            for N in (512, 513):
              kw = dict(N=N, gamma=gamma, beta=beta, norm=norm)
              kw2 = dict(scale=scale, time=True, centered_scale=True,
                         norm_scale=True, dtype='float64')
              psih_s, psi_s = compute_gmw(**kw, **kw2, order=order)
              psih_f, psi_f = morsewave(**kw, freqs=1/scale, K=order + 1)

              psih_f, psi_f = psih_f[:, -1], psi_f[:, -1]

              mad_t = np.mean(np.abs(psi_s - psi_f))
              mad_f = np.mean(np.abs(psih_s - psih_f))
              assert np.allclose(psi_s, psi_f),   errmsg(mad_t, **kw, **kw2)
              assert np.allclose(psih_s, psih_f), errmsg(mad_f, **kw, **kw2)


def test_norm():
    """Test that L1-normed time-domain wavelet's L1 norm is fixed at 2,
             and L2-normed freq-domain wavelet's L2 norm is fixed at `N`.
    """
    os.environ['SSQ_GPU'] = '0'
    th = 1e-3

    for gamma, beta in [(3, 60), (4, 80)]:
      for norm in ('bandpass', 'energy'):
        for scale in (2, 3):
          for N in (512, 513):
            for centered_scale in (True, False):
              kw = dict(N=N, scale=scale, gamma=gamma, beta=beta, norm=norm,
                        centered_scale=centered_scale, time=True,
                        norm_scale=True, dtype='float64')
              psih, psi = compute_gmw(**kw)

              if norm == 'bandpass':
                  l1_t = np.sum(np.abs(psi))
                  assert abs(l1_t - 2) < th, errmsg(abs(l1_t - 2), **kw)
              elif norm == 'energy':
                  l2_f = np.sum(np.abs(psih)**2)
                  assert abs(l2_f - N) < th, errmsg(abs(l2_f - N), **kw)


def test_wavelet():
    """Test that `gmw` is a valid `Wavelet`."""
    os.environ['SSQ_GPU'] = '0'
    wavelet = Wavelet('gmw')
    wavelet.info()
    wavelet.viz()

    wavelet = Wavelet(('gmw', {'gamma': 3, 'beta': 60, 'norm': 'energy',
                               'centered_scale': True}))
    wavelet.info()
    wavelet.viz()


def errmsg(err, scale, gamma, beta, N, norm, centered_scale, **other):
    return ("err={:.2e} (gamma, beta, scale, N, norm, centered_scale) = "
            "({}, {}, {}, {}, {}, {})"
            ).format(err, gamma, beta, scale, N, norm, centered_scale)


if __name__ == '__main__':
    if VIZ:
        test_api_vs_full()
        test_api_vs_full_higher_order()
        test_norm()
        test_wavelet()
    else:
        pytest.main([__file__, "-s"])
