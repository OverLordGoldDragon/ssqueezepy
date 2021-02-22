# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, fftshift
from numba import jit
from scipy import integrate

__all__ = [
    "buffer",
    "unbuffer",
    "window_norm",
    "window_resolution",
    "window_area",
]


def buffer(x, seg_len, n_overlap):
    """Build 2D array where each column is a successive slice of `x` of length
    `seg_len` and overlapping by `n_overlap` (or equivalently incrementing
    starting index of each slice by `hop_len = seg_len - n_overlap`).

    Mimics MATLAB's `buffer`, with less functionality.

    Ex:
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xb = buffer(x, seg_len=5, n_overlap=3)
        xb == [[0, 1, 2, 3, 4],
               [2, 3, 4, 5, 6],
               [4, 5, 6, 7, 8]].T
    """
    hop_len = seg_len - n_overlap
    n_segs = (len(x) - seg_len) // hop_len + 1
    out = np.zeros((seg_len, n_segs), dtype=x.dtype)

    for i in range(n_segs):
        start = i * hop_len
        end   = start + seg_len
        out[:, i] = x[start:end]
    return out


def unbuffer(xbuf, window, hop_len, n_fft, N, win_exp=1):
    """Undoes `buffer`, per padding logic used in `stft`:
        (N, n_fft) : logic
         even, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (128, 120, 247, 60, 59)
          odd,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (129, 121, 249, 60, 60)
         even,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (128, 121, 248, 60, 60)
          odd, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (129, 120, 248, 60, 59)
    """
    if N is None:
        # assume greatest possible len(x) (unpadded)
        N = xbuf.shape[1] * hop_len + len(window) - 1
    if len(window) != n_fft:
        raise ValueError("Must have `len(window) == n_fft` "
                         "(got %s != %s)" % (len(window), n_fft))
    if win_exp == 0:
        window = 1
    elif win_exp != 1:
        window = window ** win_exp
    x = np.zeros(N + n_fft - 1)

    _overlap_add(x, xbuf, window, hop_len, n_fft)
    x = x[n_fft//2 : -((n_fft - 1)//2)]
    return x


def window_norm(window, hop_len, n_fft, N, win_exp=1):
    """Computes window modulation array for use in `stft` and `istft`."""
    wn = np.zeros(N + n_fft - 1)

    _window_norm(wn, window, hop_len, n_fft, win_exp)
    return wn[n_fft//2 : -((n_fft - 1)//2)]


@jit(nopython=True, cache=True)
def _overlap_add(x, xbuf, window, hop_len, n_fft):
    for i in range(xbuf.shape[1]):
        n = i * hop_len
        x[n:n + n_fft] += xbuf[:, i] * window


@jit(nopython=True, cache=True)
def _window_norm(wn, window, hop_len, n_fft, win_exp=1):
    max_hops = (len(wn) - n_fft) // hop_len + 1
    wpow = window ** (win_exp + 1)

    for i in range(max_hops):
        n = i * hop_len
        wn[n:n + n_fft] += wpow


def window_resolution(window):
    """Minimal function to compute a window's time & frequency widths, assuming
    Fourier spectrum centered about dc (else use `ssqueezepy.wavelets` methods).

    Returns std_w, std_t, harea. `window` must be np.ndarray and >=0.
    """
    from ..wavelets import _xifn
    assert window.min() >= 0, "`window` must be >= 0 (got min=%s)" % window.min()
    N = len(window)

    t  = np.arange(-N/2, N/2, step=1)
    ws = fftshift(_xifn(1, N))

    psihs   = fftshift(fft(window))
    apsi2   = np.abs(window)**2
    apsih2s = np.abs(psihs)**2

    var_w = integrate.trapz(ws**2 * apsih2s, ws) / integrate.trapz(apsih2s, ws)
    var_t = integrate.trapz(t**2  * apsi2, t)    / integrate.trapz(apsi2, t)

    std_w, std_t = np.sqrt(var_w), np.sqrt(var_t)
    harea = std_w * std_t
    return std_w, std_t, harea


def window_area(window, time=True, frequency=False):
    """Minimal function to compute a window's time or frequency 'area' as area
    under curve of `abs(window)**2`. `window` must be np.ndarray.
    """
    from ..wavelets import _xifn
    if not time and not frequency:
        raise ValueError("must compute something")

    if time:
        t = np.arange(-len(window)/2, len(window)/2, step=1)
        at = integrate.trapz(np.abs(window)**2, t)
    if frequency:
        ws = fftshift(_xifn(1, len(window)))
        apsih2s = np.abs(fftshift(fft(window)))**2
        aw = integrate.trapz(apsih2s, ws)

    if time and frequency:
        return at, aw
    elif time:
        return at
    return aw
