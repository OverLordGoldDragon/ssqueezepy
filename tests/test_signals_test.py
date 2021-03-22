# -*- coding: utf-8 -*-
"""Test ssqueezepy/_test_signals.py"""
import os
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import Wavelet, TestSignals
from ssqueezepy.utils import window_resolution

VIZ = 0
os.environ['SSQ_GPU'] = '0'  # in case concurrent tests set it to '1'


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
    os.environ['SSQ_GPU'] = '0'
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
    os.environ['SSQ_GPU'] = '0'
    # (N, beta, NW): (512, 42.5, 255); (256, 21.5, 255)
    N = 256#512
    signals = 'all'
    snr = 5
    n_fft = N
    win_len = n_fft#//2
    tsigs = TestSignals(N=N, snr=snr)
    wavelet = Wavelet(('GMW', {'beta': 21.5}))

    NW = win_len//2 - 1
    window = np.abs(sig.windows.dpss(win_len, NW))
    # window = np.pad(window, win_len//2)
    window_name = 'DPSS'
    config_str = '\nNW=%s' % NW

    # ensure `wavelet` and `window` have ~same time & frequency resolutions
    # TODO make function to auto-find matching wavelet given window & vice versa
    print("std_w, std_t, harea\nwavelet: {:.4f}, {:.4f}, {:.8f}"
          "\nwindow:  {:.4f}, {:.4f}, {:.8f}".format(
              wavelet.std_w, wavelet.std_t, wavelet.harea,
              *window_resolution(window)))

    tsigs.cwt_vs_stft(wavelet, window, signals=signals, N=N, win_len=win_len,
                      n_fft=n_fft, window_name=window_name, config_str=config_str)


def test_ridgecomp():
    os.environ['SSQ_GPU'] = '0'
    N = 256
    n_ridges = 3
    penalty = 25
    signals = 'poly-cubic'

    tsigs = TestSignals(N=N)
    kw = dict(N=N, signals=signals, n_ridges=n_ridges, penalty=penalty)
    tsigs.ridgecomp(transform='cwt',  **kw)
    tsigs.ridgecomp(transform='stft', **kw)


def test_gpu():
    """Test that TestSignals can run on GPU."""
    try:
        import torch
        torch.tensor(1., device='cuda')
    except:
        return

    N = 256
    tsigs = TestSignals(N=N)
    window = np.abs(sig.windows.dpss(N, N//2 - 1))
    signals = 'par-lchirp'

    os.environ['SSQ_GPU'] = '1'
    wavelet = Wavelet()
    tsigs.cwt_vs_stft(wavelet, window, signals=signals, N=N)
    os.environ['SSQ_GPU'] = '0'


if __name__ == '__main__':
    if VIZ:
        test_demo()
        test_wavcomp()
        test_cwt_vs_stft()
        test_ridgecomp()
        test_gpu()
    else:
        pytest.main([__file__, "-s"])
