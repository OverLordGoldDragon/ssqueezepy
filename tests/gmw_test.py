# -*- coding: utf-8 -*-
"""Generalized Morse Wavelets"""
import pytest
import numpy as np
from ssqueezepy._gmw import gmw, morsewave

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0


def test_simplified_vs_full():
    for gamma, beta in [(3, 60), (4, 80)]:
      for norm in ('bandpass', 'energy'):
        for scale in (1, 2):
          for N in (512, 513):
            kw = dict(N=N, gamma=gamma, beta=beta, norm=norm)
            psih_s, psi_s = gmw(**kw, f=scale, time=True)
            psih_f, psi_f = morsewave(**kw, freqs=scale)
            # print(np.sum(np.abs(psih_s)**2), kw, scale, flush=True)
            # print(np.sum(np.abs(psi_s)), kw, scale, flush=True)

            mad_t = np.mean(np.abs(psi_s - psi_f))
            mad_f = np.mean(np.abs(psih_s - psih_f))
            assert np.allclose(psi_s, psi_f),   errmsg(mad_t, scale, **kw)
            assert np.allclose(psih_s, psih_f), errmsg(mad_f, scale, **kw)


def test_norm():
    """Test that L1-normed time-domain wavelet's L1 norm is fixed at 2,
             and L2-normed freq-domain wavelet's L2 norm is fixed at `N`.
    """
    th = 1e-3

    for gamma, beta in [(3, 60), (4, 80)]:
      for norm in ('bandpass', 'energy'):
        for scale in (1, 2):
          for N in (512, 513):
            kw = dict(N=N, gamma=gamma, beta=beta, norm=norm)
            psih, psi = gmw(**kw, f=scale, time=True)

            if norm == 'bandpass':
                l1_t = np.sum(np.abs(psi))
                assert abs(l1_t - 2) < th, errmsg(abs(l1_t - 2), scale, **kw)
            elif norm == 'energy':
                l2_f = np.sum(np.abs(psih)**2)
                assert abs(l2_f - N) < th, errmsg(abs(l2_f - N), scale, **kw)


def errmsg(err, scale, gamma, beta, N, norm):
    return ("err={:.2e} (gamma, beta, scale, N, norm) = ({}, {}, {}, {}, {})"
            ).format(err, gamma, beta, scale, N, norm)


if __name__ == '__main__':
    if VIZ:
        test_simplified_vs_full()
        test_norm()
    else:
        pytest.main([__file__, "-s"])
