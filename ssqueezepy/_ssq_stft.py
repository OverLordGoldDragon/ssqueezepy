# -*- coding: utf-8 -*-
"""NOT FOR USE; will be ready for v0.6.0"""
import numpy as np
from ._stft import stft, phase_stft, _get_window, _check_NOLA
from ._ssq_cwt import _invert_components, _process_component_inversion_args
from .utils import WARN
from .ssqueezing import ssqueeze

pi = np.pi


def ssq_stft(x, window=None, n_fft=None, win_len=None, hop_len=1,
             dt=1, padtype='reflect', modulated=True, squeezing='sum'):
    """Synchrosqueezed Short-Time Fourier Transform.
    Implements the algorithm described in Sec. III of [1].

    # Arguments:
        t: np.ndarray. Vector of times samples are taken
           (e.g. np.linspace(0, 1, n))
        x: np.ndarray. Vector of signal samples (e.g. x = np.cos(20 * pi * t))
        opts: dict. Options:
            'type': str. Type of wavelet (see `wfiltfn`)
            's', 'mu': float. Wavelet parameters (see `wfiltfn`)
            'gamma': float. Wavelet hard thresholding value
                     (see `cwt_freq_direct`)

    # Returns:
        Tx: synchrosqueezed output of `x` (columns associated with time `t`)
        fs: frequencies associated with rows of `Tx`
        Sx: STFT of `x` (see `stft_fw`)
        Sfs: frequencies associated with rows of `Sx`
        w: phase transform of `Sx`

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010
    """
    n_fft = n_fft or len(x)
    # Calculate the modified STFT, using window of opts['winlen']
    # in frequency domain
    Sx, dSx = stft(x, window, n_fft=n_fft, win_len=win_len,
                   hop_len=hop_len, dt=dt, padtype=padtype,
                   modulated=modulated, derivative=True)

    Sfs = np.linspace(0, .5, n_fft // 2 + 1) / dt
    w = phase_stft(Sx, dSx, Sfs)

    # Calculate the synchrosqueezed frequency decomposition
    # The parameter alpha from reference [2] is given by Sfs[1] - Sfs[0]
    Tx, fs = ssqueeze(Sx, w, transform='stft', scales='linear',
                      squeezing=squeezing)

    return Tx, fs, Sx, Sfs, w, dSx


def issq_stft(Tx, window=None, cc=None, cw=None, win_len=None, hop_len=1,
              n_fft=None, N=None, modulated=True, win_exp=0):
    """Docstring
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

        window = _get_window(window, win_len, n_fft=n_fft)
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
