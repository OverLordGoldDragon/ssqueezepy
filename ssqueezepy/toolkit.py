# -*- coding: utf-8 -*-
import numpy as np
from .visuals import imshow, plot


#### Synchrosqueezing ########################################################
def lin_band(Tx, slope, offset, bw=.025, **kw):
    """Visually estimate a linear band to invert over in time-frequency(/scale)
    plane.
    """
    na, N = Tx.shape
    tcs = np.linspace(0, 1, N)
    Cs       = slope*(tcs + offset) * na
    freqband = bw * na * np.ones(N)
    Cs, freqband = Cs.astype('int32'), freqband.astype('int32')

    imshow(Tx, abs=1, aspect='auto', show=0, **kw)
    plot(Cs + freqband, color='r')
    plot(Cs - freqband, color='r', show=1)
    return Cs, freqband


#### Signals #################################################################
def _t(min, max, N, endpoint=False):
    return np.linspace(min, max, N, endpoint=endpoint)

def cos_f(freqs, N=128, phi=0, endpoint=False):
    """Adjacent different frequency cosines."""
    return np.concatenate([np.cos(2*np.pi * f * (_t(i, i + 1, N, endpoint) + phi))
                           for i, f in enumerate(freqs)])

def sin_f(freqs, N=128, phi=0, endpoint=False):
    """Adjacent different frequency sines."""
    return np.concatenate([np.sin(2*np.pi * f * (_t(i, i + 1, N, endpoint) + phi))
                           for i, f in enumerate(freqs)])

#### Misc ####################################################################
def mad_rms(x, xrec):
    """Reconstruction error metric; scale-invariant, robust to outliers
    and partly sparsity. https://stats.stackexchange.com/q/495242/239063"""
    return np.mean(np.abs(x - xrec)) / np.sqrt(np.mean(x**2))


def where_amax(x):
    """Return N-dimensional indices of where `abs(x) == max(abs(x))`."""
    return np.where(np.abs(x) == np.abs(x).max())
