# -*- coding: utf-8 -*-
import pytest
import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, extract_ridges
from ssqueezepy.visuals import plot, imshow

# set to 1 to run tests as functions, showing plots
VIZ = 1


def viz(signal, Tf, ridge, yticks=None, ssq=False):
    if not VIZ:
        return
    plot(signal, title="Time signal", show=1,
         xlabel="Time axis [A.U.]",
         ylabel="Signal Amplitude [A.U.]")

    ikw = dict(abs=1, cmap='jet', yticks=yticks, show=0)
    pkw = dict(linestyle='--', color='k', show=1,
               xlabel="Time axis [A.U.]")
    if ssq:
        imshow(np.flipud(Tf), title="abs(SSQ_CWT) w/ ridge", **ikw)
        plot(len(Tf) - ridge, **pkw, ylabel="Frequencies [A.U.]")
    else:
        imshow(Tf, title="abs(CWT) w/ ridge", **ikw)
        plot(ridge, **pkw, ylabel="Frequency scales [A.U.]")


def test_basic():
    """Example ridge from similar example as can be found at MATLAB:
    https://www.mathworks.com/help/wavelet/ref/wsstridge.html#bu6we25-penalty
    """
    test_matrix = np.array([[1, 4, 4],[2, 2, 2],[5,5,4]])
    fs_test = np.exp([1,2,3])

    ridge_idxs, *_ = extract_ridges(test_matrix, fs_test, penalty=2.0)
    print('ridge follows indexes:', ridge_idxs)


def test_sine():
    """Sine + cosine."""
    sig_len, f1, f2 = 500, 0.5, 2.0
    padtype = 'wrap'
    penalty = 2.0

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    x1 = np.sin(2*np.pi * f1 * t_vec)
    x2 = np.cos(2*np.pi * f2 * t_vec)
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    ridge_idxs, _,max_energy = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs, scales)

    # SSQ_CWT example
    ridge_idxs, _,max_energy = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=4)
    viz(x, Tx, ridge_idxs, ssq_freq, ssq=True)


def test_chirp():
    """Linear + quadratic chirp."""
    sig_len = 500
    penalty = 0.1
    padtype = 'reflect'

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    x1 = sig.chirp(t_vec, f0=2,  f1=8, t1=20, method='linear')
    x2 = sig.chirp(t_vec, f0=.4, f1=4, t1=20, method='quadratic')
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # SSQ_CWT example
    ridge_idxs, *_ = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=2)
    viz(x, Tx, ridge_idxs, ssq=True)


def test_poly():
    """Cubic polynomial frequency variation + pure tone."""
    sig_len, f = 500, 0.5
    padtype = 'wrap'

    t_vec = np.linspace(0, 10, sig_len, endpoint=True)
    p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
    x1 = sig.sweep_poly(t_vec, p1)
    x2 = np.sin(2*np.pi * f * t_vec)
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example
    penalty = 2.0
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # SSQ_CWT example
    ridge_idxs, *_ = extract_ridges(Tx, ssq_freq, penalty, n_ridges=2, BW=2)
    viz(x, Tx, ridge_idxs, ssq=True)

def test_failed_chirp_wsst():
    """Linear + quadratic chirp."""
    sig_len = 600
    padtype = 'symmetric'
    t_vec = np.linspace(0, 3, sig_len, endpoint=True)
  
    x1 = sig.chirp(t_vec-1.5, f0=30,  t1=1.1,f1=40,  method='quadratic')
    x2 = sig.chirp(t_vec-1.5,f0=10,  t1=1.1,f1=5, method='quadratic')
    x = x1 + x2

    Tx, ssq_freq, Wx, scales, _ = ssq_cwt(x, t=t_vec, padtype=padtype)

    # CWT example no penalty
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty=0.0, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)

    # CWT example with penalty
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty=0.5, n_ridges=2, BW=25)
    viz(x, Wx, ridge_idxs)
    
    

if __name__ == '__main__':
    if VIZ:
        test_basic()
        test_sine()
        test_chirp()
        test_poly()
        test_failed_chirp_wsst()
    else:
        pytest.main([__file__, "-s"])