# -*- coding: utf-8 -*-
"""Test ssqueezepy/_test_signals.py"""
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import Wavelet, TestSignals
from ssqueezepy.utils import window_resolution

VIZ = 0


def test_demo():
    tsigs = TestSignals(N=256)
    dft = (None, 'rows', 'cols')[0]
    tsigs.demo(dft=dft)

    signals = [
        'am-cosine',
        ('hchirp', dict(fmin=.2)),
        ('sine:am-cosine', (dict(f=32, phi0=1), dict(amin=.3))),
    ]
    tsigs.demo(signals, N=256)
    tsigs.demo(signals, dft='rows')
    tsigs.demo(signals, dft='cols')


def test_wavcomp():
    tsigs = TestSignals(N=256)
    wavelets = [Wavelet(('gmw', {'beta': 5})),
                Wavelet(('gmw', {'beta': 22})),
                ]
    tsigs.wavcomp(wavelets)

    # test name-param pair, and ability to auto-set `N`
    N_all = [256, None]
    signals_all = [[('#echirp', dict(fmin=.1))],
                   [('lchirp',  dict(fmin=1, fmax=60, tmin=0, tmax=5))]]
    for N, signals in zip(N_all, signals_all):
        tsigs.wavcomp(wavelets, signals=signals, N=N)


def test_cwt_vs_stft():
    # (N, beta, NW): (512, 42.5, 255); (256, 21.5, 255)
    N = 256
    n_fft = N
    win_len = n_fft
    tsigs = TestSignals(N=N)
    wavelet = Wavelet(('GMW', {'beta': 21.5}))

    NW = win_len//2 - 1
    window = np.abs(sig.windows.dpss(win_len, NW))
    window_name = 'DPSS'
    config_str = '\nNW=%s' % NW

    # ensure `wavelet` and `window` have ~same time & frequency resolutions
    # TODO make function to auto-find matching wavelet given window & vice versa
    print("std_w, std_t, harea\nwavelet: {:.4f}, {:.4f}, {:.8f}"
          "\nwindow:  {:.4f}, {:.4f}, {:.8f}".format(
              wavelet.std_w, wavelet.std_t, wavelet.harea,
              *window_resolution(window)))

    tsigs.cwt_vs_stft(wavelet, window, signals='all', N=N, win_len=win_len,
                      n_fft=n_fft, window_name=window_name, config_str=config_str)


if __name__ == '__main__':
    if VIZ:
        test_demo()
        test_wavcomp()
        test_cwt_vs_stft()
    else:
        pytest.main([__file__, "-s"])
