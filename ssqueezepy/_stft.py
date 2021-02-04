# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as sig
from numpy.fft import fft, ifft, rfft, irfft, fftshift, ifftshift
from .utils import WARN, padsignal, buffer, unbuffer, window_norm
from .wavelets import _xifn


def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=1.,
         padtype='reflect', modulated=True, derivative=False):
    """Compute the short-time Fourier transform and modified short-time
    Fourier transform from [1].

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        window: str / np.ndarray / None
            STFT windowing kernel. If string, will fetch per
            `scipy.signal.get_window(window, win_len, fftbins=True)`.
            Defaults to `scipy.signal.windows.dpss(win_len, 4)`; the DPSS
            window provides the best time-frequency resolution.

            Always padded to `n_fft`, so for accurate filter characteristics
            (side lobe decay, etc), best to pass in pre-designed `window`
            with `win_len == n_fft`.

        n_fft: int >= 0 / None
            FFT length, or STFT column length. If `win_len < n_fft`, will
            pad `window`. Every STFT column is `fft(window * x_slice)`.
            Defaults to `len(x)`.

        win_len: int >= 0 / None
            Length of `window` to use. Used to generate a window if `window`
            is string, and ignored if it's np.ndarray.
            Defaults to `n_fft//8` or `len(window)` (if `window` is np.ndarray).

        hop_len: int > 0
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`.
            Must be 1 for invertible synchrosqueezed STFT.

        fs: float
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 0 to 0.5*fs, i.e. as fraction of reference
            sampling rate up to Nyquist limit. Used to compute `dSx` and
            `ssq_freqs`.

        padtype: str
            Pad scheme to apply on input. See `help(utils.padsignal)`.

        modulated: bool (default True)
            Whether to use "modified" variant as in [1], which centers DFT
            cisoids at the window for each shift `u`. `False` will not invert
            once synchrosqueezed.
            Recommended to use `True`; see "Modulation" below.

        derivative: bool (default False)
            Whether to compute and return `dSx`. Requires `fs`.

    **Modulation:**
        `True` will center DFT cisoids at the window for each shift `u`:
            Sm[u, k] = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
        as opposed to usual STFT:
            S[u, k]  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)

        Most implementations (including `scipy`, `librosa`) compute *neither*,
        but rather center the window for each slice, thus shifting DFT bases
        relative to n=0 (t=0). These create spectra that, viewed as signals, are
        of high frequency, making inversion and synchrosqueezing very unstable.
        Details & visuals: https://dsp.stackexchange.com/a/72590/50076

    # Returns:
        Sx: [(n_fft//2 + 1) x n_hops] np.ndarray
            STFT of `x`. Positive frequencies only (+dc), via `rfft`.
            (n_hops = (len(x) - 1)//hop_len + 1)
            (rows=scales, cols=timeshifts)

        dWx: [(n_fft//2 + 1) x n_hops] np.ndarray
            Returned only if `derivative=True`.
            Time-derivative of the STFT of `x`, computed via STFT done with
            time-differentiated `window`, as in [1]. This differs from CWT's,
            where its (and Sx's) DFTs are taken along columns rather than rows.
            d/dt(window) obtained via freq-domain differentiation (help(cwt)).

    Recommended:
        - odd win_len with odd n_fft and even with even, not vice versa
        These make the ('periodic') window's left=right pad len which gives
        it zero phase, desired in some applications

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_fw.m
    """
    def _stft(xp, window, n_fft, hop_len, fs, modulated, derivative):
        Sx  = buffer(xp, n_fft, n_fft - hop_len)
        dSx = Sx.copy()
        Sx  *= window.reshape(-1, 1)
        dSx *= diff_window.reshape(-1, 1)

        if modulated:
            # shift windowed slices so they're always DFT-centered (about n=0),
            # thus shifting bases (cisoids) along the window: e^(-j*(w - u))
            Sx = ifftshift(Sx, axes=0)
            if derivative:
                dSx = ifftshift(dSx, axes=0)

        # keep only positive frequencies (Hermitian symmetry assuming real `x`)
        Sx = rfft(Sx, axis=0)
        if derivative:
            dSx = rfft(dSx, axis=0) * fs
        return (Sx, dSx) if derivative else (Sx, None)

    n_fft = n_fft or len(x)
    if win_len is None:
        win_len = (len(window) if isinstance(window, np.ndarray) else
                   n_fft//8)
    window, diff_window = get_window(window, win_len, n_fft, derivative=True)
    _check_NOLA(window, hop_len)

    padlength = len(x) + n_fft - 1  # pad `x` to length `padlength`
    xp = padsignal(x, padtype, padlength=padlength)

    Sx, dSx = _stft(xp, window, n_fft, hop_len, fs, modulated, derivative)

    return (Sx, dSx) if derivative else Sx


def istft(Sx, window=None, n_fft=None, win_len=None, hop_len=1, N=None,
          modulated=True, win_exp=1):
    """Inverse Short-Time Fourier transform. Computed with least-squares
    estimate for `win_exp`=1 per Griffin-Lim [1], recommended for STFT with
    modifications, else simple inversion with `win_exp`=0:

        x[n] = sum(y_t[n] * w^a[n - tH]) / sum(w^{a+1}[n - tH]),
        y_t = ifft(Sx), H = hop_len, a = win_exp, t = hop index, n = sample index

    Warns if `window` NOLA constraint isn't met (see [2]), invalidating inversion.
    Nice visuals and explanations on istft:
        https://www.mathworks.com/help/signal/ref/istft.html

    # Arguments:
        Sx: np.ndarray
            STFT of 1D `x`.

        window, n_fft, win_len, hop_len, modulated
            Should be same as used in forward STFT. See `help(stft)`.

        N: int > 0 / None
            `len(x)` of original `x`, used in inversion padding and windowing.
            If None, assumes longest possible `x` given `hop_len`, `Sx.shape[1]`.

        win_exp: int >= 0
            Window power used in inversion per:

    # Returns:
        x: np.ndarray, 1D
            Signal as reconstructed from `Sx`.

    # References:
        1. Signal Estimation from Modified Short-Time Fourier Transform.
        D. W. Griffin, J. S. Lim.
        https://citeseerx.ist.psu.edu/viewdoc/
        download?doi=10.1.1.306.7858&rep=rep1&type=pdf

        2. Invertibility of overlap-add processing. B. Sharpe.
        https://gauss256.github.io/blog/cola.html
    """
    ### process args #####################################
    n_fft = n_fft or (Sx.shape[0] - 1) * 2
    win_len = win_len or n_fft // 8
    N = N or (hop_len * Sx.shape[1] - 1)  # assume largest possible N if not given

    window = get_window(window, win_len, n_fft=n_fft)
    _check_NOLA(window, hop_len)

    xbuf = irfft(Sx, n=n_fft, axis=0).real
    if modulated:
        xbuf = fftshift(xbuf, axes=0)

    # overlap-add the columns
    x  = unbuffer(xbuf, window, hop_len, n_fft, N, win_exp)
    wn = window_norm(   window, hop_len, n_fft, N, win_exp)
    x /= wn

    return x


def get_window(window, win_len, n_fft=None, derivative=False):
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
            # fftbins=True -> 'periodic' window -> narrower main side-lobe and
            # closer to zero-phase in left=right padded case
            # for windows edging at 0
            window = sig.get_window(window, win_len, fftbins=True)

        elif isinstance(window, np.ndarray):
            if len(window) != win_len:
                WARN("len(window) != win_len (%s != %s)" % (len(window), win_len))

        else:
            raise ValueError("`window` must be string or np.ndarray "
                             "(got %s)" % window)
    else:
        # sym=False <-> fftbins=True (see above)
        window = sig.windows.dpss(win_len, 4, sym=False)

    if len(window) < (win_len + pl + pr):
        window = np.pad(window, [pl, pr])

    if derivative:
        wf = fft(window)
        Nw = len(window)
        xi = _xifn(1, Nw)
        if Nw % 2 == 0:
            xi[Nw // 2] = 0
        # frequency-domain differentiation; see `dWx` return docs in `help(cwt)`
        diff_window = ifft(wf * 1j * xi).real

    return (window, diff_window) if derivative else window


def _check_NOLA(window, hop_len):
    """https://gauss256.github.io/blog/cola.html"""
    if not sig.check_NOLA(window, len(window), len(window) - hop_len):
        WARN("`window` fails Non-zero Overlap Add (NOLA) criterion; STFT "
             "not invertible")
