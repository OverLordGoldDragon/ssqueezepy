# -*- coding: utf-8 -*-
"""Utilities & others
"""
import pytest
from ssqueezepy.wavelets import Wavelet
from ssqueezepy.utils import cwt_scalebounds

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0


def test_bounds():
    wavelet = Wavelet(('morlet', {'mu': 6}))

    for N in (4096, 2048, 1024, 512, 256, 128, 64):
        try:
            cwt_scalebounds(wavelet, N=N)
        except Exception as e:
            raise Exception(f"N={N} failed; errmsg:\n{e}")


if __name__ == '__main__':
    if VIZ:
        test_bounds()
    else:
        pytest.main([__file__, "-s"])
