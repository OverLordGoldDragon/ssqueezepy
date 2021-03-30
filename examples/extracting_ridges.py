# -*- coding: utf-8 -*-
"""Authors: David Bondesson, OverLordGoldDragon

Ridge extraction on signals with varying time-frequency characteristics.
"""
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import numpy as np
import scipy.signal as sig
from ssqueezepy import ssq_cwt, ssq_stft, extract_ridges, TestSignals
from ssqueezepy.visuals import plot, imshow

#%%## Visual methods #########################################################
def viz(x, Tf, ridge_idxs, yticks=None, ssq=False, transform='cwt', show_x=True):
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

#%%# Basic example ###########################################################
# Example ridge from similar example as can be found at MATLAB:
# https://www.mathworks.com/help/wavelet/ref/wsstridge.html#bu6we25-penalty
test_matrix = np.array([[1, 4, 4], [2, 2, 2], [5, 5, 4]])
fs_test = np.exp([1, 2, 3])

ridge_idxs, *_ = extract_ridges(test_matrix, fs_test, penalty=2.0,
                                get_params=True)
print("Ridge follows indexes:", ridge_idxs)
assert np.allclose(ridge_idxs, np.array([[2, 2, 2]]))

#%%# sin + cos ###############################################################
N, f1, f2 = 513, 5, 20
padtype = 'wrap'
penalty = 20

t  = np.linspace(0, 1, N, endpoint=True)
x1 = np.sin(2*np.pi * f1 * t)
x2 = np.cos(2*np.pi * f2 * t)
x = x1 + x2

tf_transforms(x, t, padtype=padtype, penalty=penalty)

#%%# Linear + quadratic chirp ################################################
N = 513
penalty = 20
padtype = 'reflect'

t  = np.linspace(0, 20, N, endpoint=True)
x1 = sig.chirp(t, f0=2,  f1=8, t1=20, method='linear')
x2 = sig.chirp(t, f0=.4, f1=4, t1=20, method='quadratic')
x = x1 + x2

tf_transforms(x, t, padtype=padtype, stft_bw=4, penalty=penalty)

#%%# Cubic polynomial frequency variation + pure tone ########################
N, f = 257, 0.5
padtype = 'wrap'
penalty = 20

t  = np.linspace(0, 10, N, endpoint=True)
p1 = np.poly1d([0.025, -0.36, 1.25, 2.0])
p3 = np.poly1d([0.01, -0.25, 1.5, 4.0])
x1 = sig.sweep_poly(t, p1)
x3 = sig.sweep_poly(t, p3)
x2 = np.sin(2*np.pi * f * t)
x = x1 + x2 + x3
# x += np.sqrt(1) * np.random.randn(len(x))

tf_transforms(x, t, n_ridges=3, padtype=padtype, stft_bw=4, ssq_stft_bw=4,
              penalty=penalty)

#%%# Reflect-added linear chirps #############################################
N = 512
penalty = 2

tsigs = TestSignals(N)
x, t = tsigs.lchirp(N)
x += x[::-1]

tf_transforms(x, t, penalty=penalty, cwt_bw=10)

#%%# Parallel F.M. linear chirps  ############################################
N = 512
penalty = 2
padtype = 'reflect'

tsigs = TestSignals(N)
x, t = tsigs.par_lchirp(N)

tf_transforms(x, t, penalty=penalty, padtype=padtype)
