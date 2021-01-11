# -*- coding: utf-8 -*-
"""NOT FOR USE; will be ready for v0.6.0"""
import numpy as np
import numpy.matlib
from numpy.fft import fft, ifft, rfft, irfft, fftshift, ifftshift
import scipy.signal as sig
from .utils import WARN, padsignal, buffer, unbuffer, window_norm
from .wavelets import _xifn


pi = np.pi
EPS = np.finfo(np.float64).eps  # machine epsilon for float64


def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, dt=1,
         padtype='reflect', modulated=True, derivative=False):
    """Compute the short-time Fourier transform and modified short-time
    Fourier transform from [1].

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        window: str / np.ndarray / None
            STFT windowing kernel. If string, will fetch per
            `scipy.signal.get_window(window, win_len, fftbins=True)`.
            Defaults to `scipy.signal.windows.dpss`.

        n_fft: int / None
            FFT length, or STFT column length. If `win_len < n_fft`, will
            pad `window`. Every STFT column is `fft(window * x_slice)`.
            Defaults to `len(x)`.

        win_len: int / None
            Length of `window` to use. Used to generate a window if `window`
            is string, and ignored if it's np.ndarray.
            Defaults to `n_fft//8` or `len(window)` (if `window` is np.ndarray).

        hop_len: int
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`.
            Must be 1 for invertible synchrosqueezed STFT.

        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'reflect', 'symmetric', 'replicate', 'wrap').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See `help(utils.padsignal)`.

        modulated: bool (default True)
            Whether to use "modified" variant as in [1], which centers DFT
            cisoids at the window for each shift `u`. `False` will not invert
            once synchrosqueezed.
            Recommended to use `True`; see "Modulation" below.


    Modulation:
        `True` will center DFT cisoids at the window for each shift `u`:
            Sm(u, k) = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
        as opposed to usual STFT:
            S(u, k)  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)

        Most implementations (including `scipy`, `librosa`) compute *neither*,
        but rather center the window for each slice, thus shifting DFT bases
        relative to n=0 (t=0). These create spectra that, viewed as signals, are
        of high frequency, making inversion and synchrosqueezing very unstable.
        https://github.com/OverLordGoldDragon/ssqueezepy/pull/25

    # Returns:
        Sx: (na x n) size matrix (rows = scales, cols = times) containing
            samples of the CWT of `x`.
        Sfs: vector containign the associated frequencies.
        dSx: (na x n) size matrix containing samples of the time-derivatives
             of the STFT of `x`.

    Recommended:
        - odd win_len with odd n_fft and even with even, not vice versa
        These make the ('periodic') window's left=right pad len which gives
        it zero phase, desired in some applications

    # References:
        1. G. Thakur and H.-T. Wu,
        "Synchrosqueezing-based Recovery of Instantaneous Frequency
        from Nonuniform Samples",
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
    """
    def _stft(xp, window, n_fft, hop_len, dt, modulated, derivative):
        Sx  = buffer(xp, n_fft, n_fft - hop_len)
        dSx = Sx.copy()
        Sx  *= window.reshape(-1, 1)
        dSx *= diff_window.reshape(-1, 1)

        if modulated:
            # shift windowed slices so they're always DFT-centered (about n=0),
            # thus shifting bases (cisoids) along the window: e^(-j*(w - u))
            Sx = ifftshift(Sx,  axes=0)
            if derivative:
                dSx = ifftshift(dSx, axes=0)

        # keep only positive frequencies (Hermitian symmetry assuming real `x`)
        Sx = rfft(Sx,  axis=0)
        if derivative:
            dSx = rfft(dSx, axis=0) / dt
        return (Sx, dSx) if derivative else (Sx, None)

    n_fft = n_fft or len(x)
    if win_len is None:
        win_len = (len(window) if isinstance(window, np.ndarray) else
                   n_fft//8)
    window, diff_window = _get_window(window, win_len, n_fft, derivative=True)
    _check_NOLA(window, hop_len)

    padlength = len(x) + n_fft - 1  # pad `x` to length `padlength`
    xp = padsignal(x, padtype, padlength=padlength)

    Sx, dSx = _stft(xp, window, n_fft, hop_len, dt, modulated, derivative)

    return (Sx, dSx) if derivative else Sx


def istft(Sx, window=None, win_len=None, hop_len=None, n_fft=None, N=None,
          modulated=True, win_exp=1):
    """Inverse short-time Fourier transform.

    Nice visuals and explanations on istft:
        https://www.mathworks.com/help/signal/ref/iscola.html

    # Arguments:
        Sx: np.ndarray. Wavelet transform of a signal (see `stft_fwd`).
        opts: dict. Options:
            'type': str. Wavelet type. See `stft_fwd`, and `wfiltfn`.
            Others; see `stft_fwd` and source code.

    # Returns:
        x: the signal, as reconstructed from `Sx`.
    """
    ### process args #####################################
    n_fft = n_fft or (Sx.shape[0] - 1) * 2
    win_len = win_len or n_fft // 8
    hop_len = hop_len or 1
    N = N or (hop_len * Sx.shape[1] - 1)  # assume largest possible N if not given

    window = _get_window(window, win_len, n_fft=n_fft)
    _check_NOLA(window, hop_len)

    xbuf = irfft(Sx, n=n_fft, axis=0).real
    if modulated:
        xbuf = fftshift(xbuf, axes=0)

    # overlap-add the columns
    x  = unbuffer(xbuf, window, hop_len, n_fft, N, win_exp)
    wn = window_norm(   window, hop_len, n_fft, N, win_exp)
    x /= wn

    return x


def phase_stft(Sx, dSx, Sfs, gamma=None):
    """Calculate the phase transform of modified STFT at each (freq, time) pair:
        w[a, b] = Im( eta - d/dt(Sx[t, eta]) / Sx[t, eta] / (2*pi*j))
    Uses direct differentiation by calculating dSx/dt in frequency domain
    (the secondary output of `stft_fwd`, see `stft_fwd`).

    # Arguments:
        Sx: np.ndarray. Wavelet transform of `x` (see `stft_fwd`).
        dSx: np.ndarray. Samples of time-derivative of STFT of `x`
             (see `stft_fwd`).
        opts: dict. Options:
            'gamma': float. Wavelet threshold (default: sqrt(machine epsilon))

    # Returns:
        w: phase transform, w.shape == Sx.shape

    # References:
        1. G. Thakur and H.-T. Wu,
        "Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples",
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.

        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.
    """
    gamma = gamma or np.sqrt(EPS)

    w = Sfs.reshape(-1, 1) - np.imag(dSx / Sx / (2*pi))
    # threshold out small points
    w[(np.abs(Sx) < gamma)] = np.inf
    # reassign negative phase (should be minority) as if positive
    w = np.abs(w)  # TODO also for CWT?

    return w


def _get_window(window, win_len, n_fft=None, derivative=False):
    if n_fft is None:
        pl, pr = 0, 0
    else:
        if win_len > n_fft:
            raise ValueError("Can't have `win_len > n_fft` ({} > {})".format(
                win_len, n_fft))
        pl = (n_fft - win_len) // 2
        pr = (n_fft - win_len - pl)

    if window is not None:
        if isinstance(window, str):
            window = sig.get_window(window, win_len, fftbins=True)
        elif isinstance(window, np.ndarray):
            if len(window) != win_len:
                WARN("len(window) != win_len (%s != %s)" % (len(window), win_len))
            if len(window) < (win_len + pl + pr):
                window = np.pad(window, [pl, pr])
        else:
            raise ValueError("`window` must be string or np.ndarray "
                             "(got %s)" % window)
    else:
        # fftbins=True -> 'periodic' window -> narrower main side-lobe and
        # closer to zero-phase in left=right padded case
        # for windows edging at 0
        # window = sig.get_window('hann', win_len, fftbins=True)
        window = sig.windows.dpss(win_len, 4, sym=False)
        window = np.pad(window, [pl, pr])

    if derivative:
        wf = fft(window)
        Nw = len(window)
        xi = _xifn(1, Nw)
        if Nw % 2 == 0:
            xi[Nw // 2] = 0
        diff_window = ifft(wf * 1j * xi).real

    return (window, diff_window) if derivative else window


def _check_NOLA(window, hop_len):
    """https://gauss256.github.io/blog/cola.html"""
    if not sig.check_NOLA(window, len(window), len(window) - hop_len):
        WARN("`window` fails Non-zero Overlap Add (NOLA) criterion; STFT "
             "not invertible")
