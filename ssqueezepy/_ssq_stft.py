# -*- coding: utf-8 -*-
import numpy as np
from ._stft import stft, phase_stft, get_window, _check_NOLA
from ._ssq_cwt import _invert_components, _process_component_inversion_args
from .utils import WARN
from .ssqueezing import ssqueeze

pi = np.pi


def ssq_stft(x, window=None, n_fft=None, win_len=None, hop_len=1,
             fs=1., padtype='reflect', modulated=True, squeezing='sum'):
    """Synchrosqueezed Short-Time Fourier Transform.
    Implements the algorithm described in Sec. III of [1].

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        window, n_fft, win_len, hop_len, fs, padtype, modulated
            See `help(stft)`.

        squeezing: str
            Synchrosqueezing scheme to use. See `help(ssqueeze)`.

    # Returns:
        Tx: np.ndarray
            Synchrosqueezed STFT of `x`, of same shape as `Sx` (see `stft`).
        ssq_freqs: np.ndarray
            Frequencies associated with rows of `Tx` and `Sx`.
        Sx: np.ndarray
            STFT of `x`. See `help(stft)`.
        dSx: np.ndarray
            Time-derivative of STFT of `x`. See `help(stft)`.
        w: np.ndarray
            Phase transform of STFT of `x`. See `help(phase_stft)`.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533
    """
    n_fft = n_fft or len(x)

    Sx, dSx = stft(x, window, n_fft=n_fft, win_len=win_len,
                   hop_len=hop_len, fs=fs, padtype=padtype,
                   modulated=modulated, derivative=True)

    Sfs = np.linspace(0, .5, Sx.shape[0]) * fs
    w = phase_stft(Sx, dSx, Sfs)

    Tx, ssq_freqs = ssqueeze(Sx, w, transform='stft', squeezing=squeezing,
                             ssq_freqs=Sfs)

    return Tx, ssq_freqs, Sx, dSx, w


def issq_stft(Tx, window=None, cc=None, cw=None, n_fft=None, win_len=None,
              hop_len=1, N=None, modulated=True):
    """Inverse synchrosqueezed STFT.

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        window, n_fft, win_len, hop_len, modulated
            See `help(stft)`. Must match those used in `ssq_stft`.

        cc, cw: np.ndarray
            See `help(issq_cwt)`.

    # Returns:
        x: np.ndarray
            Signal as reconstructed from `Tx`.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. Fourier synchrosqueezed transform MATLAB docs.
        https://www.mathworks.com/help/signal/ref/fsst.html
    """
    def _process_args(Tx, window, cc, cw, win_len, hop_len, n_fft, modulated):
        if not modulated:
            raise ValueError("inversion with `modulated == False` "
                             "is unsupported.")
        if hop_len != 1:
            raise ValueError("inversion with `hop_len != 1` is unsupported.")

        cc, cw, full_inverse = _process_component_inversion_args(cc, cw)

        n_fft = n_fft or (Tx.shape[0] - 1) * 2
        win_len = win_len or n_fft

        window = get_window(window, win_len, n_fft=n_fft)
        _check_NOLA(window, hop_len)
        if abs(np.argmax(window) - len(window)//2) > 1:
            WARN("`window` maximum not centered; results may be unreliable.")

        return window, cc, cw, win_len, hop_len, n_fft, full_inverse

    (window, cc, cw, win_len, hop_len, n_fft, full_inverse
     ) = _process_args(Tx, window, cc, cw, win_len, hop_len, n_fft, modulated)

    if full_inverse:
        # Integration over all frequencies recovers original signal
        x = Tx.real.sum(axis=0)
    else:
        x = _invert_components(Tx, cc, cw)

    x *= (2 / window[len(window)//2])
    return x
