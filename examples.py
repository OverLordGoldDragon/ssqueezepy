# -*- coding: utf-8 -*-
# Papers:
# [1] https://sci-hub.se/https://doi.org/10.1016/j.sigpro.2012.11.029
# [2] https://arxiv.org/pdf/0912.2437.pdf
import numpy as np
import matplotlib.pyplot as plt

from ssqueezepy import synsq_cwt_fwd, synsq_stft_fwd
from ssqueezepy import cwt_fwd, stft_fwd

#%%
def viz_y(y, y1, y2):
    _, axes = plt.subplots(1, 2, sharey=True, figsize=(11, 3))
    axes[0].plot(y1)
    axes[1].plot(y2)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.show()
    
    plt.plot(y)
    plt.gcf().set_size_inches(14, 4)
    plt.show()
    
def viz_s(s, sN, s1, s2, s3):
    _, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 3))
    axes[0].plot(t, s1)
    axes[1].plot(t, s2)
    axes[2].plot(t, s3)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.show()
    
    _, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    axes[0].plot(t, s)
    axes[1].plot(t, sN)
    plt.show()

def _get_norm(data, norm_rel, norm_abs):
    if norm_abs is not None and norm_rel != 1:
        raise ValueError("specify only one of `norm_rel`, `norm_abs`")

    if norm_abs is None:
        vmax = np.max(np.abs(data)) * norm_rel
        vmin = -vmax
    else:
        vmin, vmax = norm_abs
    return vmin, vmax

def _make_plots(*data, cmap='bwr', norm=None, titles=None):
    vmin, vmax = norm or None, None
    
    _, axes = plt.subplots(1, len(data), sharey=True, figsize=(11, 4))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=.1, hspace=0)

    for i, x in enumerate(data):
        axes[i].imshow(x, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[i].invert_yaxis()
        axes[i].set_title(titles[i], fontsize=14, weight='bold')
    plt.show()    

def viz_gen(x, cmap='bwr', norm_rel=1, norm_abs=None):
    vmin, vmax = _get_norm(np.real(x), norm_rel, norm_abs)
    _make_plots(np.real(x), np.imag(x), cmap=cmap, 
                norm=(vmin, vmax), titles=("Real", "Imag"))

def viz_gen2(x1, x2, cmap='bwr', norm_rel=1, norm_abs=None):
    vmin, vmax = _get_norm(np.real(x1), norm_rel, norm_abs)
    _make_plots(np.real(x1), np.real(x2), cmap=cmap,
                norm=(vmin, vmax), titles=("s", "sN"))    
    
OPTS = {'type': 'bump', 'mu':1}
#%%
"""Paper [2], pg. 18"""
t = np.linspace(0, 12, 1000)
y1 = np.cos(8*t)
y2 = np.cos(t**2 + t + np.cos(t))
y  = y1 + y2
viz_y(y, y1, y2)
#%%
Tx_yc, *_ = synsq_cwt_fwd(y, fs=len(t)/t[-1], nv=32, opts=OPTS)
Tx_ys, *_ = synsq_stft_fwd(t, y, opts=OPTS)
#%%
viz_gen(Tx_yc, cmap='bwr', norm_abs=(-5e-5, 5e-5))
viz_gen(Tx_ys, cmap='bwr', norm_abs=(-5e-5, 5e-5))
#%%
"""Paper [1], pg. 1086"""
t = np.linspace(0, 10, 2048)
s1 = (1 + .2*np.cos(t)) * np.cos(2*np.pi*(2*t   + .3*np.cos(t)))
s2 = (1 + .3*np.cos(t)) * np.cos(2*np.pi*(2.4*t + .3*np.sin(t) + .5*t**1.2)
                                 ) * np.exp(-t/15)
s3 = np.cos(2*np.pi*(5.3*t + 0.2*t**1.3))
N = np.random.randn(len(t)) * np.sqrt(2.4)
s  = s1 + s2 + s3
sN = s + N
viz_s(s, sN, s1, s2, s3)
#%%
# feed "denoised" (actually noiseless) signal as noted on pg. 1086 of [1]
Tx_sc,  *_ = synsq_cwt_fwd(s,  fs=len(t)/t[-1], nv=32, opts=OPTS)
Tx_sNc, *_ = synsq_cwt_fwd(sN, fs=len(t)/t[-1], nv=32, opts=OPTS)
Tx_ss,  *_ = synsq_stft_fwd(t, s,  opts=OPTS)
Tx_sNs, *_ = synsq_stft_fwd(t, sN, opts=OPTS)
#%%
viz_gen2(Tx_sc, Tx_sNc, cmap='bwr', norm_abs=(-5e-5, 5e-5))
viz_gen2(Tx_ss, Tx_sNs, cmap='bwr', norm_abs=(-5e-5, 5e-5))
#%%
Wx_s,  *_ = cwt_fwd(s,  'bump',        opts=OPTS)
Wx_sN, *_ = cwt_fwd(sN, 'bump',        opts=OPTS)
Sx_s,  *_ = stft_fwd(s,  dt=t[1]-t[0], opts=OPTS)
Sx_sN, *_ = stft_fwd(sN, dt=t[1]-t[0], opts=OPTS)
#%%
viz_gen2(Wx_s, Wx_sN, cmap='bwr', norm_abs=(-1.3, 1.3))
viz_gen2(Sx_s, Sx_sN, cmap='bwr', norm_abs=(-1.3, 1.3))
