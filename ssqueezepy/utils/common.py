# -*- coding: utf-8 -*-
import numpy as np
import logging
from numpy.fft import fft, ifft, fftshift, ifftshift
from textwrap import wrap


logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored
pi = np.pi
EPS32 = np.finfo(np.float32).eps  # machine epsilon
EPS64 = np.finfo(np.float64).eps

__all__ = [
    "WARN",
    "NOTE",
    "pi",
    "EPS32",
    "EPS64",
    "p2up",
    "padsignal",
    "trigdiff",
    "mad",
    "est_riskshrink_thresh",
    "assert_is_one_of",
    "_textwrap",
]


def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int
            Length of original (unpadded) signal.

    # Returns:
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.
    """
    eps = np.finfo(np.float64).eps  # machine epsilon for float64
    up = 2 ** (1 + np.round(np.log2(n + eps)))

    n2 = np.floor((up - n) / 2)
    n1 = n2 + n % 2           # if n is odd, left-pad by (n2 + 1), else n1=n2
    assert n1 + n + n2 == up  # [left_pad, original, right_pad]
    return int(up), int(n1), int(n2)


def padsignal(x, padtype='reflect', padlength=None, get_params=False):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray
            Input vector, 1D or 2D. 2D has time in dim1, e.g. `(n_signals, time)`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See [1] & [2].

        padlength: int / None
            Number of samples to pad input to (i.e. len(x_padded) == padlength).
            Even: left = right, Odd: left = right + 1.
            Defaults to next highest power of 2 w.r.t. `len(x)`.

    # Returns:
        xp: np.ndarray
            Padded signal.
        n_up: int
            Next power of 2, or `padlength` if provided.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.

    # References:
        1. Signal extension modes. PyWavelets contributors
        https://pywavelets.readthedocs.io/en/latest/ref/
        signal-extension-modes.html

        2. Wavelet Bases and Lifting Wavelets. H. Xiong.
        http://min.sjtu.edu.cn/files/wavelet/
        6-lifting%20wavelet%20and%20filterbank.pdf
    """
    def _process_args(x, padtype):
        assert_is_one_of(padtype, 'padtype',
                         ('reflect', 'symmetric', 'replicate', 'wrap', 'zero'))
        if not isinstance(x, np.ndarray):
            raise TypeError("`x` must be a numpy array (got %s)" % type(x))
        elif x.ndim not in (1, 2):
            raise ValueError("`x` must be 1D or 2D (got x.ndim == %s)" % x.ndim)

    _process_args(x, padtype)
    N = x.shape[-1]

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(N)
    else:
        n_up = padlength
        if abs(padlength - N) % 2 == 0:
            n1 = n2 = (n_up - N) // 2
        else:
            n2 = (n_up - N) // 2
            n1 = n2 + 1
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    if x.ndim == 1:
        pad_width = (n1, n2)
    elif x.ndim == 2:
        pad_width = [(0, 0), (n1, n2)]

    # comments use (n=4, n1=4, n2=3) as example, but this combination can't occur
    if padtype == 'zero':
        # [1,2,3,4] -> [0,0,0,0, 1,2,3,4, 0,0,0]
        xp = np.pad(x, pad_width)
    elif padtype == 'reflect':
        # [1,2,3,4] -> [3,4,3,2, 1,2,3,4, 3,2,1]
        xp = np.pad(x, pad_width, mode='reflect')
    elif padtype == 'replicate':
        # [1,2,3,4] -> [1,1,1,1, 1,2,3,4, 4,4,4]
        xp = np.pad(x, pad_width, mode='edge')
    elif padtype == 'wrap':
        # [1,2,3,4] -> [1,2,3,4, 1,2,3,4, 1,2,3]
        xp = np.pad(x, pad_width, mode='wrap')
    elif padtype == 'symmetric':
        # [1,2,3,4] -> [4,3,2,1, 1,2,3,4, 4,3,2]
        if x.ndim == 1:
            xp = np.hstack([x[::-1][-n1:], x, x[::-1][:n2]])
        elif x.ndim == 2:
            xp = np.hstack([x[:, ::-1][:, -n1:], x, x[:, ::-1][:, :n2]])

    Npad = xp.shape[-1]
    _ = (Npad, n_up, n1, N, n2)
    assert (Npad == n_up == n1 + N + n2), "%s ?= %s ?= %s + %s + %s" % _
    return (xp, n_up, n1, n2) if get_params else xp


def trigdiff(A, fs=1, padtype=None, rpadded=None, N=None, n1=None):
    """Trigonometric / frequency-domain differentiation; see `difftype` in
    `help(ssq_cwt)`. Used internally by `ssq_cwt` with `order > 0`.
    """
    from ..wavelets import _xifn

    assert isinstance(A, np.ndarray)
    assert A.ndim == 2

    rpadded = rpadded or False
    padtype = padtype or ('reflect' if not rpadded else None)
    if rpadded and (n1 is None or N is None):
        raise ValueError("must pass `n1` and `N` if `rpadded`")

    if padtype is not None:
        A, _, n1, *_ = padsignal(A, padtype, get_params=True)

    xi = _xifn(1, A.shape[-1])[None]

    A_freqdom = fft(fftshift(A, axes=-1), axis=-1)
    A_diff = ifftshift(ifft(A_freqdom * 1j * xi * fs, axis=-1), axes=-1)

    if rpadded:
        A_diff = A_diff[:, n1:n1+N]
    return A_diff


def est_riskshrink_thresh(Wx, nv):
    """Estimate the RiskShrink hard thresholding level, based on [1].
    This has a denoising effect, but risks losing much of the signal; it's larger
    the more high-frequency content there is, even if not noise.

    # Arguments:
        Wx: np.ndarray
            CWT of a signal (see `cwt`).
        nv: int
            Number of voices used in CWT (see `cwt`).

    # Returns:
        gamma: float
            The RiskShrink hard thresholding estimate.

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        est_riskshrink_thresh.m
    """
    N = Wx.shape[1]
    Wx_fine = np.abs(Wx[:nv])
    gamma = 1.4826 * np.sqrt(2 * np.log(N)) * mad(Wx_fine)
    return gamma


def mad(data, axis=None):
    """Mean absolute deviation"""
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def assert_is_one_of(x, name, supported, e=ValueError):
    if x not in supported:
        raise e("`{}` must be one of: {} (got {})".format(
            name, ', '.join(supported), x))


def _textwrap(txt, wrap_len=50):
    """Preserves line breaks and includes `'\n'.join()` step."""
    return '\n'.join(['\n'.join(
        wrap(line, wrap_len, break_long_words=False, replace_whitespace=False))
        for line in txt.splitlines() if line.strip() != ''])
