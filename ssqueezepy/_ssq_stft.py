# -*- coding: utf-8 -*-
import numpy as np
from ._stft import stft, get_window, _check_NOLA
from ._ssq_cwt import _invert_components, _process_component_inversion_args
from .utils import WARN, EPS, pi
from .ssqueezing import ssqueeze, _check_squeezing_arg


def ssq_stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=1.,
             modulated=True, ssq_freqs=None, padtype='reflect', squeezing='sum',
             gamma=None):
    """Synchrosqueezed Short-Time Fourier Transform.
    Implements the algorithm described in Sec. III of [1].

    Good MATLAB docs: https://www.mathworks.com/help/signal/ref/fsst.html

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        window, n_fft, win_len, hop_len, fs, padtype, modulated
            See `help(stft)`.

        ssq_freqs, squeezing
            See `help(ssqueezing.ssqueeze)`.

        gamma: float / None
            STFT phase threshold. Sets `w=inf` for small values of `Sx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Sx) < beta] = inf
            This is used to zero `Sx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = sqrt(machine epsilon) = np.sqrt(np.finfo(np.float64).eps)

    # Returns:
        Tx: np.ndarray
            Synchrosqueezed STFT of `x`, of same shape as `Sx`.
        ssq_freqs: np.ndarray
            Frequencies associated with rows of `Tx`.
        Sx: np.ndarray
            STFT of `x`. See `help(stft)`.
        Sfs: np.ndarray
            Frequencies associated with rows of `Sx` (by default == `ssq_freqs`).
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
    _check_squeezing_arg(squeezing)

    Sx, dSx = stft(x, window, n_fft=n_fft, win_len=win_len, hop_len=hop_len,
                   fs=fs, padtype=padtype, modulated=modulated, derivative=True)

    Sfs = np.linspace(0, .5, Sx.shape[0]) * fs
    w = phase_stft(Sx, dSx, Sfs, gamma)

    if ssq_freqs is None:
        ssq_freqs = Sfs
    Tx, ssq_freqs = ssqueeze(Sx, w, transform='stft', squeezing=squeezing,
                             ssq_freqs=ssq_freqs)

    return Tx, ssq_freqs, Sx, Sfs, dSx, w


def issq_stft(Tx, window=None, cc=None, cw=None, n_fft=None, win_len=None,
              hop_len=1, modulated=True):
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


def phase_stft(Sx, dSx, Sfs, gamma=None):
    """Phase transform of STFT:
        w[u, k] = Im( k - d/dt(Sx[u, k]) / Sx[u, k] / (j*2pi) )

    Defined in Sec. 3 of [1]. Additionally explained in:
        https://dsp.stackexchange.com/a/72589/50076

    # Arguments:
        Sx: np.ndarray
            STFT of `x`, where `x` is 1D.

        dSx: np.ndarray
            Time-derivative of STFT of `x`

        Sfs: np.ndarray
            Associated physical frequencies, according to `dt` used in `stft`.
            Spans 0 to fs/2, linearly.

        gamma: float / None
            STFT phase threshold. Sets `w=inf` for small values of `Sx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Sx) < beta] = inf
            This is used to zero `Wx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = sqrt(machine epsilon) = np.sqrt(np.finfo(np.float64).eps)

    # Returns:
        w: np.ndarray
            Phase transform for each element of `Sx`. w.shape == Sx.shape.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010
    """
    w = Sfs.reshape(-1, 1) - np.imag(dSx / Sx) / (2*pi)

    # treat negative phases as positive; these are in small minority, and
    # slightly aid invertibility (as less of `Wx` is zeroed in ssqueezing)
    w = np.abs(w)

    gamma = gamma or np.sqrt(EPS)
    # threshold out small points
    w[np.abs(Sx) < gamma] = np.inf
    return w
