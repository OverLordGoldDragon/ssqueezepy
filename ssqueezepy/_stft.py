# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as sig
from .utils import WARN, padsignal, buffer, unbuffer, window_norm
from .utils import _process_fs_and_t
from .utils.fft_utils import fft, ifft, rfft, irfft, fftshift, ifftshift
from .utils.backend import torch, is_tensor
from .algos import zero_denormals
from .wavelets import _xifn, _process_params_dtype
from .configs import gdefaults, USE_GPU


def stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
         padtype='reflect', modulated=True, derivative=False, dtype=None):
    """Short-Time Fourier Transform.

    `modulated=True` computes "modified" variant from [1] which is advantageous
    to reconstruction & synchrosqueezing (see "Modulation" below).

    # Arguments:
        x: np.ndarray
            Input vector(s), 1D or 2D. See `help(cwt)`.

        window: str / np.ndarray / None
            STFT windowing kernel. If string, will fetch per
            `scipy.signal.get_window(window, win_len, fftbins=True)`.
            Defaults to `scipy.signal.windows.dpss(win_len, win_len//8)`;
            the DPSS window provides the best time-frequency resolution.

            Always padded to `n_fft`, so for accurate filter characteristics
            (side lobe decay, etc), best to pass in pre-designed `window`
            with `win_len == n_fft`.

        n_fft: int >= 0 / None
            FFT length, or `(STFT column length) // 2 + 1`.
            If `win_len < n_fft`, will pad `window`. Every STFT column is
            `fft(window * x_slice)`.
            Defaults to `len(x)//hop_len`, up to 512.

        win_len: int >= 0 / None
            Length of `window` to use. Used to generate a window if `window`
            is string, and ignored if it's np.ndarray.
            Defaults to `n_fft//8` or `len(window)` (if `window` is np.ndarray).

        hop_len: int > 0
            STFT stride, or number of samples to skip/hop over between subsequent
            windowings. Relates to 'overlap' as `overlap = n_fft - hop_len`.
            Must be 1 for invertible synchrosqueezed STFT.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq frequencies
            range from 0 to 0.5*fs, i.e. as fraction of reference sampling rate
            up to Nyquist limit. Used to compute `dSx` and `ssq_freqs`.

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        padtype: str
            Pad scheme to apply on input. See `help(utils.padsignal)`.

        modulated: bool (default True)
            Whether to use "modified" variant as in [1], which centers DFT
            cisoids at the window for each shift `u`. `False` will not invert
            once synchrosqueezed.
            Recommended `True`. See "Modulation" and [2] below.

        derivative: bool (default False)
            Whether to compute and return `dSx`. Uses `fs`.

        dtype: str['float32', 'float64'] / None
            Compute precision; use 'float32` for speed & memory at expense of
            accuracy (negligible for most purposes).
            If None, uses value from `configs.ini`.

            To be safe with `'float32'`, time-localized `window`, and large
            `hop_len`, use

                from ssqueezepy._stft import _check_NOLA
                _check_NOLA(window, hop_len, 'float32', imprecision_strict=True)

    **Modulation**
        `True` will center DFT cisoids at the window for each shift `u`:
            Sm[u, k] = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*(n - u)/N)
        as opposed to usual STFT:
            S[u, k]  = sum_{0}^{N-1} f[n] * g[n - u] * exp(-j*2pi*k*n/N)

        Most implementations (including `scipy`, `librosa`) compute *neither*,
        but rather center the window for each slice, thus shifting DFT bases
        relative to n=0 (t=0). These create spectra that, viewed as signals, are
        of high frequency, making inversion and synchrosqueezing very unstable.
        Details & visuals: https://dsp.stackexchange.com/a/72590/50076

        Better explanation in ref [2].

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

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Equivalence between "windowed Fourier transform" and STFT as
        convolutions/filtering. John Muradeli.
        https://dsp.stackexchange.com/a/86938/50076

        3. STFT: why overlapping the window? John Muradeli.
        https://dsp.stackexchange.com/a/88124/50076

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_fw.m
    """
    def _stft(xp, window, diff_window, n_fft, hop_len, fs, modulated, derivative):
        Sx = buffer(xp, n_fft, n_fft - hop_len, modulated)
        if derivative:
            dSx = buffer(xp, n_fft, n_fft - hop_len, modulated)

        if modulated:
            window = ifftshift(window, astensor=True)
            if derivative:
                diff_window = ifftshift(diff_window, astensor=True) * fs

        reshape = (-1, 1) if xp.ndim == 1 else (1, -1, 1)
        Sx *= window.reshape(*reshape)
        if derivative:
            dSx *= (diff_window.reshape(*reshape))

        # keep only positive frequencies (Hermitian symmetry assuming real `x`)
        axis = 0 if xp.ndim == 1 else 1
        Sx = rfft(Sx, axis=axis, astensor=True)
        if derivative:
            dSx = rfft(dSx, axis=axis, astensor=True)
        return (Sx, dSx) if derivative else (Sx, None)

    # process args
    assert x.ndim in (1, 2)
    N = x.shape[-1]
    _, fs, _ = _process_fs_and_t(fs, t, N)
    n_fft = n_fft or min(N//hop_len, 512)

    # process `window`, make `diff_window`, check NOLA, enforce `dtype`
    if win_len is None:
        win_len = (len(window) if isinstance(window, np.ndarray) else
                   n_fft)
    dtype = gdefaults('_stft.stft', dtype=dtype)
    window, diff_window = get_window(window, win_len, n_fft, derivative=True,
                                     dtype=dtype)
    _check_NOLA(window, hop_len, dtype)
    x = _process_params_dtype(x, dtype=dtype, auto_gpu=False)

    # pad `x` to length `padlength`
    padlength = N + n_fft - 1
    xp = padsignal(x, padtype, padlength=padlength)

    # arrays -> tensors if using GPU
    if USE_GPU():
        xp, window, diff_window = [torch.as_tensor(g, device='cuda') for g in
                                   (xp, window, diff_window)]
    # take STFT
    Sx, dSx = _stft(xp, window, diff_window, n_fft, hop_len, fs, modulated,
                    derivative)

    # ensure indexing works as expected downstream (cupy)
    Sx  = Sx.contiguous()  if is_tensor(Sx)  else Sx
    dSx = dSx.contiguous() if is_tensor(dSx) else dSx

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
            Window power used in inversion (see [1], [2], or equation above).

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

        3. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        stft_iw.m
    """
    ### process args #####################################
    n_fft = n_fft or (Sx.shape[0] - 1) * 2
    win_len = win_len or n_fft
    N = N or hop_len * Sx.shape[1]  # assume largest possible N if not given
    dtype = 'float32' if str(Sx.dtype) == 'complex64' else 'float64'

    window = get_window(window, win_len, n_fft=n_fft, dtype=dtype)
    _check_NOLA(window, hop_len, dtype=dtype)

    xbuf = irfft(Sx, n=n_fft, axis=0).real
    if modulated:
        xbuf = fftshift(xbuf, axes=0)

    # overlap-add the columns
    x = unbuffer(xbuf, window, hop_len, n_fft, N, win_exp)

    # window norm, control for float precision
    wn = window_norm(window, hop_len, n_fft, N, win_exp)
    th = np.finfo(x.dtype).tiny
    if wn.min() < th:
        approx_nonzero_idxs = wn > th
        x[approx_nonzero_idxs] /= wn[approx_nonzero_idxs]
    else:
        x /= wn

    # unpad
    x = x[n_fft//2 : -((n_fft - 1)//2)]

    return x


def get_window(window, win_len, n_fft=None, derivative=False, dtype=None):
    """See `window` in `help(stft)`. Will return window of length `n_fft`,
    regardless of `win_len` (will pad if needed).
    """
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
        window = sig.windows.dpss(win_len, max(4, win_len//8), sym=False)

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

    # cast `dtype`, zero denormals (extremely small numbers that slow down CPU)
    window = _process_params_dtype(window, dtype=dtype, auto_gpu=False)
    zero_denormals(window)

    if derivative:
        diff_window = _process_params_dtype(diff_window, dtype=dtype,
                                            auto_gpu=False)
        zero_denormals(diff_window)
    return (window, diff_window) if derivative else window


def _check_NOLA(window, hop_len, dtype=None, imprecision_strict=False):
    """https://gauss256.github.io/blog/cola.html"""
    # basic NOLA
    if hop_len > len(window):
        WARN("`hop_len > len(window)`; STFT not invertible")
    elif not sig.check_NOLA(window, len(window), len(window) - hop_len):
        WARN("`window` fails Non-zero Overlap Add (NOLA) criterion; "
             "STFT not invertible")

    # handle `dtype`; note this is just a guess, what matters is `Sx.dtype`
    if dtype is None:
        dtype = str(window.dtype)

    # check for right boundary effect: as ssqueezepy's number of output frames
    # is critically sampled (not more than needed), it creates an issue with
    # float32 and time-localized windows, which struggle to invert the last frame
    tol = 0.15 if imprecision_strict else 1e-3
    if dtype == 'float32' and not sig.check_NOLA(
            window, len(window), len(window) - hop_len, tol=tol):
        # 1e-3 can still have imprecision detectable by eye, but only upon few
        # samples, so avoid paranoia. Use 1e-2 to be safe, and 0.15 for ~exact
        WARN("Imprecision expected at right-most hop of signal, in inversion. "
             "Lower `hop_len`, choose wider `window`, or use `dtype='float64'`.")
