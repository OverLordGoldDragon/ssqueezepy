# -*- coding: utf-8 -*-
import numpy as np
from ._stft import stft, get_window, _check_NOLA
from ._ssq_cwt import _invert_components, _process_component_inversion_args
from .utils.cwt_utils import _process_fs_and_t
from .utils.common import WARN, EPS32, EPS64
from .utils import backend as S
from .utils.backend import torch
from .algos import phase_stft_cpu, phase_stft_gpu
from .ssqueezing import ssqueeze, _check_ssqueezing_args


def ssq_stft(x, window=None, n_fft=None, win_len=None, hop_len=1, fs=None, t=None,
             modulated=True, ssq_freqs=None, padtype='reflect', squeezing='sum',
             gamma=None, preserve_transform=None, dtype=None, astensor=True,
             flipud=False, get_w=False, get_dWx=False):
    """Synchrosqueezed Short-Time Fourier Transform.
    Implements the algorithm described in Sec. III of [1].

    MATLAB docs: https://www.mathworks.com/help/signal/ref/fsst.html

    # Arguments:
        x: np.ndarray
            Input vector(s), 1D or 2D. See `help(cwt)`.

        window, n_fft, win_len, hop_len, fs, t, padtype, modulated
            See `help(stft)`.

        ssq_freqs, squeezing
            See `help(ssqueezing.ssqueeze)`.

        gamma: float / None
            See `help(_ssq_cwt.phase_cwt)`.

        preserve_transform: bool (default True)
            Whether to return `Sx` as directly output from `stft` (it might be
            altered by `ssqueeze` or `phase_transform`). Uses more memory
            per storing extra copy of `Sx`.

        dtype: str['float32', 'float64'] / None
            See `help(stft)`.

        astensor: bool (default True)
            If `'SSQ_GPU' == '1'`, whether to return arrays as on-GPU tensors
            or move them back to CPU & convert to Numpy arrays.

        flipud: bool (default False)
            See `help(ssqueeze)`.

        get_w, get_dWx
            See `help(ssq_cwt)`.
            (Named `_dWx` instead of `_dSx` for consistency.)

    # Returns:
        Tx: np.ndarray
            Synchrosqueezed STFT of `x`, of same shape as `Sx`.
        Sx: np.ndarray
            STFT of `x`. See `help(stft)`.
        ssq_freqs: np.ndarray
            Frequencies associated with rows of `Tx`.
        Sfs: np.ndarray
            Frequencies associated with rows of `Sx` (by default == `ssq_freqs`).
        w: np.ndarray (if `get_w=True`)
            Phase transform of STFT of `x`. See `help(phase_stft)`.
        dSx: np.ndarray (if `get_dWx=True`)
            Time-derivative of STFT of `x`. See `help(stft)`.

    # References:
        1. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533
    """
    if x.ndim == 2 and get_w:
        raise NotImplementedError("`get_w=True` unsupported with batched input.")
    _, fs, _ = _process_fs_and_t(fs, t, x.shape[-1])
    _check_ssqueezing_args(squeezing)

    Sx, dSx = stft(x, window, n_fft=n_fft, win_len=win_len, hop_len=hop_len,
                   fs=fs, padtype=padtype, modulated=modulated, derivative=True,
                   dtype=dtype)

    # preserve original `Sx` or not
    if preserve_transform is None:
        preserve_transform = not S.is_tensor(Sx)
    if preserve_transform:
        _Sx = (Sx.copy() if not S.is_tensor(Sx) else
               Sx.detach().clone())
    else:
        _Sx = Sx

    # make `Sfs`
    Sfs = _make_Sfs(Sx, fs)
    # gamma
    if gamma is None:
        gamma = np.sqrt(EPS64 if S.is_dtype(Sx, 'complex128') else EPS32)

    # compute `w` if `get_w` and free `dWx` from memory if `not get_dWx`
    if get_w:
        w = phase_stft(_Sx, dSx, Sfs, gamma)
        _dSx = None  # don't use in `ssqueeze`
        if not get_dWx:
            dSx = None
    else:
        w = None
        _dSx = dSx

    # synchrosqueeze
    if ssq_freqs is None:
        ssq_freqs = Sfs
    Tx, ssq_freqs = ssqueeze(_Sx, w, squeezing=squeezing, ssq_freqs=ssq_freqs,
                             Sfs=Sfs, flipud=flipud, gamma=gamma, dWx=_dSx,
                             maprange='maximal', transform='stft')
    # return
    if not astensor and S.is_tensor(Tx):
        Tx, Sx, w, dSx = [g.cpu().numpy() if S.is_tensor(g) else g
                          for g in (Tx, Sx, w, dSx)]

    if get_w and get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, w, dSx
    elif get_w:
        return Tx, Sx, ssq_freqs, Sfs, w
    elif get_dWx:
        return Tx, Sx, ssq_freqs, Sfs, dSx
    else:
        return Tx, Sx, ssq_freqs, Sfs


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
            WARN("`window` maximum not centered; results may be inaccurate.")

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


def phase_stft(Sx, dSx, Sfs, gamma=None, parallel=None):
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
    S.warn_if_tensor_and_par(Sx, parallel)
    if gamma is None:
        gamma = np.sqrt(EPS64 if S.is_dtype(Sx, 'complex128') else EPS32)

    if S.is_tensor(Sx):
        return phase_stft_gpu(Sx, dSx, Sfs, gamma)
    return phase_stft_cpu(Sx, dSx, Sfs, gamma, parallel)


def _make_Sfs(Sx, fs):
    dtype = 'float32' if 'complex64' in str(Sx.dtype) else 'float64'
    n_rows = len(Sx) if Sx.ndim == 2 else Sx.shape[1]
    if S.is_tensor(Sx):
        Sfs = torch.linspace(0, .5*fs, n_rows, device=Sx.device,
                             dtype=getattr(torch, dtype))
    else:
        Sfs = np.linspace(0, .5*fs, n_rows, dtype=dtype)
    return Sfs
