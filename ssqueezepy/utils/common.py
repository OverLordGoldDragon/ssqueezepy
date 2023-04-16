# -*- coding: utf-8 -*-
import numpy as np
import logging
from textwrap import wrap
from .fft_utils import fft, ifft


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
    "find_closest_parallel_is_faster",
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
    up = int(2**(1 + np.round(np.log2(n))))
    n2 = int((up - n) // 2)
    n1 = int(up - n - n2)
    return up, n1, n2


def padsignal(x, padtype='reflect', padlength=None, get_params=False):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray / torch.Tensor
            Input vector, 1D or 2D. 2D has time in dim1, e.g. `(n_inputs, time)`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See [1] & [2].

            Torch doesn't support all padding schemes, but `cwt` will still
            pad it via NumPy.

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
        is_numpy = bool(isinstance(x, np.ndarray))
        supported = (('zero', 'reflect', 'symmetric', 'replicate', 'wrap')
                     if is_numpy else
                     ('zero', 'reflect'))
        assert_is_one_of(padtype, 'padtype', supported)

        if not hasattr(x, 'ndim'):
            raise TypeError("`x` must be a numpy array or torch Tensor "
                            "(got %s)" % type(x))
        elif x.ndim not in (1, 2):
            raise ValueError("`x` must be 1D or 2D (got x.ndim == %s)" % x.ndim)
        return is_numpy

    is_numpy = _process_args(x, padtype)
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

    # set functional spec
    if x.ndim == 1:
        pad_width = (n1, n2)
    elif x.ndim == 2:
        pad_width = ([(0, 0), (n1, n2)] if is_numpy else
                     (n1, n2))

    # comments use (n=4, n1=4, n2=3) as example, but this combination can't occur
    if is_numpy:
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
    else:
        import torch
        mode = 'constant' if padtype == 'zero' else 'reflect'
        if x.ndim == 1:
            xp = torch.nn.functional.pad(x[None], pad_width, mode)[0]
        else:
            xp = torch.nn.functional.pad(x, pad_width, mode)

    return (xp, n_up, n1, n2) if get_params else xp


def trigdiff(A, fs=1., padtype=None, rpadded=None, N=None, n1=None, window=None,
             transform='cwt'):
    """Trigonometric / frequency-domain differentiation; see `difftype` in
    `help(ssq_cwt)`. Used internally by `ssq_cwt` with `order > 0`.

    Un-transforms `A`, then transforms differentiated.

    # Arguments:
        A: np.ndarray
            2D array to differentiate (or 3D, batched).

        fs: float
            Sampling frequency, used to scale derivative to physical units.

        padtype: str / None
            Whether to pad `A` (along dim1) before differentiating.

        rpadded: bool (default None)
            Whether `A` is already padded. Defaults to True if `padtype` is None.
            Must pass `N` if True.

        N: int
            Length of unpadded signal (i.e. `A.shape[1]`).

        n1: int
            Will trim differentiated array as `A_diff[:, n1:n1+N]` (un-padding).

        transform: str['cwt', 'stft']
            Whether `A` stems from CWT or STFT, which changes how differentiation
            is done. `'stft'` currently not supported.

    """
    from ..wavelets import _xifn
    from . import backend as S

    def _process_args(A, rpadded, padtype, N, transform, window):
        if transform == 'stft':
            raise NotImplementedError("`transform='stft'` is currently not "
                                      "supported.")
        assert isinstance(A, np.ndarray) or S.is_tensor(A), type(A)
        assert A.ndim in (2, 3)

        if rpadded and N is None:
            raise ValueError("must pass `N` if `rpadded`")
        if transform == 'stft' and window is None:
            raise ValueError("`transform='stft'` requires `window`")

        rpadded = rpadded or False
        padtype = padtype or ('reflect' if not rpadded else None)
        return rpadded, padtype

    rpadded, padtype = _process_args(A, rpadded, padtype, N, transform, window)

    if padtype is not None:
        A, _, n1, *_ = padsignal(A, padtype, get_params=True)

    if transform == 'cwt':
        xi = S.asarray(_xifn(1, A.shape[-1]), A.dtype)

        A_freqdom = fft(A, axis=-1, astensor=True)
        A_diff = ifft(A_freqdom * 1j * xi * fs, axis=-1, astensor=True)
    else:
        # this requires us to first fully invert STFT(x), then `buffer(x)`,
        # then compute `diff_window`, which isn't hard to implement;
        # last of these is done

        # wf = fft(S.asarray(window, A.dtype))
        # xi = S.asarray(_xifn(1, len(window))[None], A.dtype)
        # if len(window) % 2 == 0:
        #     xi[len(window) // 2] = 0
        # reshape = (-1, 1) if A.ndim == 2 else (1, -1, 1)
        # diff_window = ifft(wf * 1j * xi).real.reshape(*reshape)
        pass

    if rpadded or padtype is not None:
        if N is None:
            N = A.shape[-1]
        if n1 is None:
            _, n1, _ = p2up(N)
        idx = ((slice(None), slice(n1, n1 + N)) if A.ndim == 2 else
               (slice(None), slice(None), slice(n1, n1 + N)))
        A_diff = A_diff[idx]
    if S.is_tensor(A_diff):
        A_diff = A_diff.contiguous()
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


def find_closest_parallel_is_faster(shape, dtype='float32', trials=7, verbose=1):
    """Returns True if `find_closest(, parallel=True)` is faster, as averaged
    over `trials` trials on dummy data.
    """
    from timeit import timeit
    from ..algos import find_closest

    a = np.abs(np.random.randn(*shape).astype(dtype))
    v = np.random.uniform(0, len(a), len(a)).astype(dtype)

    t0 = timeit(lambda: find_closest(a, v, parallel=False), number=trials)
    t1 = timeit(lambda: find_closest(a, v, parallel=True),  number=trials)
    if verbose:
        print("Parallel avg.:     {} sec\nNon-parallel avg.: {} sec".format(
            t1 / trials, t0 / trials))
    return t1 > t0


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
