# -*- coding: utf-8 -*-
"""Coverage-mainly tests; see examples/extracting_ridges.py for more test cases.
"""
import os
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges, cwt
from ssqueezepy.visuals import plot, imshow

# set to 1 to run tests as functions, showing plots
VIZ = 0
os.environ['SSQ_GPU'] = '0'  # in case concurrent tests set it to '1'



def test_basic():
    """Example ridge from similar example as can be found at MATLAB:
    https://www.mathworks.com/help/wavelet/ref/wsstridge.html#bu6we25-penalty
    """
    test_matrix = np.array([[1, 4, 4], [2, 2, 2], [5, 5, 4]])
    fs_test = np.exp([1, 2, 3])

    ridge_idxs, *_ = extract_ridges(test_matrix, fs_test, penalty=2.0,
                                    get_params=True)
    assert np.allclose(ridge_idxs, np.array([[2, 2, 2]]))


def test_poly():
    """Cubic polynomial frequency variation + pure tone."""
    N, f = 257, 0.5
    padtype = 'wrap'
    penalty = 2.0

    t  = np.linspace(0, 10, N, endpoint=True)
    p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
    x1 = sig.sweep_poly(t, p1)
    x2 = np.sin(2*np.pi * f * t)
    x = x1 + x2

    tf_transforms(x, t, padtype=padtype, stft_bw=4, penalty=penalty)


def viz(x, Tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=True):
    if not VIZ:
        return
    if show_x:
        plot(x, title="x(t)", show=1,
             xlabel="Time [samples]", ylabel="Signal Amplitude [A.U.]")

    if transform == 'cwt' and not ssq:
        Tf = np.flipud(Tf)
        ridge_idxs = len(Tf) - ridge_idxs

    ylabel = ("Frequency scales [1/Hz]" if (transform == 'cwt' and not ssq) else
              "Frequencies [Hz]")
    title = "abs({}{}) w/ ridge_idxs".format("SSQ_" if ssq else "",
                                             transform.upper())

    ikw = dict(abs=1, cmap='jet', yticks=yticks, title=title)
    pkw = dict(linestyle='--', color='k', xlabel="Time [samples]", ylabel=ylabel,
               xlims=(0, Tf.shape[1]), ylims=(0, len(Tf)))

    imshow(Tf, **ikw, show=0)
    plot(ridge_idxs, **pkw, show=1)


def tf_transforms(x, t, wavelet='morlet', window=None, padtype='wrap',
                  penalty=.5, n_ridges=2, cwt_bw=15, stft_bw=15,
                  ssq_cwt_bw=4, ssq_stft_bw=4):
    os.environ['SSQ_GPU'] = '0'
    kw_cwt  = dict(t=t, padtype=padtype)
    kw_stft = dict(fs=1/(t[1] - t[0]), padtype=padtype)
    Twx, Wx, ssq_freqs_c, scales, *_ = ssq_cwt(x,  wavelet, **kw_cwt)
    Tsx, Sx, ssq_freqs_s, Sfs, *_    = ssq_stft(x, window,  **kw_stft)

    ckw = dict(penalty=penalty, n_ridges=n_ridges, transform='cwt')
    skw = dict(penalty=penalty, n_ridges=n_ridges, transform='stft')
    cwt_ridges      = extract_ridges(Wx,  scales,      bw=cwt_bw,      **ckw)
    ssq_cwt_ridges  = extract_ridges(Twx, ssq_freqs_c, bw=ssq_cwt_bw,  **ckw)
    stft_ridges     = extract_ridges(Sx,  Sfs,         bw=stft_bw,     **skw)
    ssq_stft_ridges = extract_ridges(Tsx, ssq_freqs_s, bw=ssq_stft_bw, **skw)

    viz(x, Wx,  cwt_ridges,      scales,      ssq=0, transform='cwt',  show_x=1)
    viz(x, Twx, ssq_cwt_ridges,  ssq_freqs_c, ssq=1, transform='cwt',  show_x=0)
    viz(x, Sx,  stft_ridges,     Sfs,         ssq=0, transform='stft', show_x=0)
    viz(x, Tsx, ssq_stft_ridges, ssq_freqs_s, ssq=1, transform='stft', show_x=0)


def test_parallel():
    """Ensure `parallel=True` output matches that of `=False`."""
    for N in (255, 512):
        x = np.random.randn(N)
        Wx, scales = cwt(x)

        out0 = extract_ridges(Wx, scales, parallel=False)
        out1 = extract_ridges(Wx, scales, parallel=True)

        adiff = np.abs(out0 - out1)
        assert np.allclose(out0, out1), "N=%s, Max err: %s" % (N, adiff.max())


if __name__ == '__main__':
    if VIZ:
        test_basic()
        test_poly()
        test_parallel()
    else:
        pytest.main([__file__, "-s"])
