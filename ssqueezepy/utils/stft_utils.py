# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, fftshift
from numba import jit, prange
from scipy import integrate
from .gpu_utils import _run_on_gpu, _get_kernel_params
from .backend import torch

__all__ = [
    "buffer",
    "unbuffer",
    "window_norm",
    "window_resolution",
    "window_area",
]


def buffer(x, seg_len, n_overlap, modulated=False, gpu=False, parallel=True):
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
    s20 = int(np.ceil(seg_len / 2))
    s21 = s20 - 1 if (seg_len % 2 == 1) else s20

    if gpu:
        return _buffer_gpu(x, seg_len, n_segs, hop_len, s20, s21, modulated)

    out = np.zeros((seg_len, n_segs), dtype=x.dtype)
    fn = _buffer_par2 if parallel else _buffer
    fn(x, out, seg_len, n_segs, hop_len, s20, s21, modulated)
    return out


@jit(nopython=True, cache=True)
def _buffer(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    for i in range(n_segs):
        start = hop_len * i
        if not modulated:
            start = hop_len * i
            end   = start + seg_len
            out[:, i] = x[start:end]
        else:
            start0 = hop_len * i
            end0   = start0 + s21
            start1 = end0
            end1   = start1 + s20
            out[:s20, i] = x[start1:end1]
            out[s20:, i] = x[start0:end0]


@jit(nopython=True, cache=True, parallel=True)
def _buffer_par2(x, out, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    for i in prange(n_segs):
        start = hop_len * i
        if not modulated:
            start = hop_len * i
            end   = start + seg_len
            out[:, i] = x[start:end]
        else:
            start0 = hop_len * i
            end0   = start0 + s21
            start1 = end0
            end1   = start1 + s20
            out[:s20, i] = x[start1:end1]
            out[s20:, i] = x[start0:end0]


def _buffer_gpu(x, seg_len, n_segs, hop_len, s20, s21, modulated=False):
    kernel = '''
    extern "C" __global__
    void buffer(${dtype} x[${N}],
                ${dtype} out[${L}][${W}],
                bool modulated,
                int hop_len, int seg_len,
                int s20, int s21)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i >= ${W})
        return;

      int start = hop_len * i;
      for (int j=start; j < start + seg_len; ++j){
        if (!modulated){
          out[j - start][i] = x[j];
        } else {
          if (j < start + s20){
            out[j - start][i] = x[j + s21];
          } else{
            out[j - start][i] = x[j - s20];
          }
        }
      }
    }
    '''
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, device='cuda')
    out = x.new_zeros((seg_len, n_segs))

    blockspergrid, threadsperblock, kernel_kw, _ = _get_kernel_params(out, dim=1)
    kernel_kw.update(dict(N=len(x), L=len(out), W=out.shape[1]))
    kernel_args = [x.data_ptr(), out.data_ptr(), bool(modulated), hop_len,
                   seg_len, s20, s21]
    _run_on_gpu(kernel, blockspergrid, threadsperblock, *kernel_args, **kernel_kw)
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
