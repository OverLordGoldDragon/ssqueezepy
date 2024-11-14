# -*- coding: utf-8 -*-
import warnings
import numpy as np
from .wavelets import Wavelet, center_frequency
from .utils import backend as S, cwt_scalebounds, p2up
from .utils.common import EPS32, EPS64, p2up, trigdiff
from .ssqueezing import ssqueeze
from ._ssq_cwt import phase_cwt, phase_cwt_num
from ._ssq_stft import phase_stft, _make_Sfs


__all__ = ['freq_to_scale', 'scale_to_freq', 'phase_ssqueeze', 'phase_transform']


def freq_to_scale(freqs, wavelet, N, fs=1, n_search_scales=None, kind='peak',
                  base=2):
    """Convert frequencies to scales.

    # Arguments:
        freqs: np.ndarray
            1D array of frequencies. Must range between 0 and `N/fs/2` (Nyquist).

        wavelet: wavelets.Wavelet / str / tupe[str, dict]
            Wavelet.

        N: int
            `len(x)` of interest.

        fs: int
            Sampling rate in Hz.

        n_search_scales: int / None
            This method approximates the conversion of scales. Higher = better
            accuracy, but takes longer. Defaults to `10 * len(freqs)`.

        kind: str
            Mapping to use, one of: 'peak', 'energy', 'peak-ct'.
            See `help(ssqueezepy.center_frequency)`.

        base: int
            Base of exponent of `freqs`. Defaults to 2.
            `freqs` can be any distributed in any way, including mix of log
            and linear, so the base only helps improve the search if it matches
            that of `freqs`. If `freqs` is purely exponential, then
            `base = np.diff(np.log(freqs))[0] * 2.718281828`.

    # Returns:
        scales: np.ndarray
            1D arrays of scales.
    """
    def logb(x, base=2):
        return np.log(x) / np.log(base)

    def log(x):
        return logb(x, base)

    freqs = freqs / fs  # convert to unitless, [0., 0.5)
    assert np.all(freqs >= 0),       "frequencies must be positive"
    assert freqs.max() <= 0.5,       "max frequency must be 0.5"
    assert freqs.max() == freqs[-1], "max frequency must be last sample"
    assert freqs.min() == freqs[0],  "min frequency must be first sample"

    M = len(freqs)
    if n_search_scales is None:
        n_search_scales = 10 * M
    smin, smax = cwt_scalebounds(wavelet, N, preset='maximal', use_padded_N=False)
    search_scales = np.logspace(log(smin), log(smax), n_search_scales, base=base)

    w_from_scales = []
    for scale in search_scales:
        w = center_frequency(wavelet, scale, N, kind=kind)
        w_from_scales.append(min(max(w, 0), np.pi))
    f_from_scales = np.array(w_from_scales) / (2*np.pi)

    # pick closest match
    fmin, fmax = freqs.min(), freqs.max()
    smax = search_scales[np.argmin(np.abs(f_from_scales - fmin))]
    smin = search_scales[np.argmin(np.abs(f_from_scales - fmax))]
    # make scales between found min and max
    scales = np.logspace(log(smax), log(smin), M, base=base)

    return scales


def scale_to_freq(scales, wavelet, N, fs=1, padtype='reflect'):
    """Convert scales to frequencies.

    # Arguments:
        freqs: np.ndarray
            1D array of frequencies. Must range between 0 and `N/fs/2` (Nyquist).

        wavelet: wavelets.Wavelet / str / tupe[str, dict]
            Wavelet.

        N: int
            `len(x)` of interest.

        fs: int
            Sampling rate in Hz.

        padtype: str / None
            `padtype` used in the transform. Used to determine the length
            of wavelets used in the transform: `None` uses half the length
            relative to `not None`.
            The exact value doesn't matter, only whether it's `None` or not.

    # Returns:
        freqs: np.ndarray
            1D arrays of frequencies.
    """
    # process args
    if isinstance(scales, float):
        scales = np.array([scales])
    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    # evaluate wavelet at `scales`
    Npad = p2up(N)[0] if padtype is not None else N
    psis = wavelet(scale=scales, N=Npad)
    if hasattr(psis, 'cpu'):
        psis = psis.cpu().numpy()
    # find peak indices
    idxs = np.argmax(psis, axis=-1)

    # check
    # https://github.com/OverLordGoldDragon/ssqueezepy/issues/41
    if np.any(idxs > Npad//2) or 0 in idxs:
        warnings.warn("found potentially ill-behaved wavelets (peak indices at "
                      "negative freqs or at dc); will round idxs to 1 or N/2")
        n_psis = len(psis)
        for i, ix in enumerate(idxs):
            if ix > Npad//2 or ix == 0:
                if i > n_psis // 2:  # low freq
                    idxs[i] = 1
                else:  # high freq
                    idxs[i] = Npad//2
    # convert
    freqs = idxs / Npad  # [0, ..., .5]
    assert freqs.min() >= 0,   freqs.min()
    assert freqs.max() <= 0.5, freqs.max()

    freqs *= fs   # [0, ..., fs/2]
    return freqs


def phase_ssqueeze(Wx, dWx=None, ssq_freqs=None, scales=None, Sfs=None, fs=1.,
                   t=None, squeezing='sum', maprange=None, wavelet=None,
                   gamma=None, was_padded=True, flipud=False,
                   rpadded=False, padtype=None, N=None, n1=None,
                   difftype=None, difforder=None,
                   get_w=False, get_dWx=False, transform='cwt'):
    """Take `phase_transform` then `ssqueeze`. Can be used on an arbitrary
    CWT/STFT-like time-frequency transform `Wx`.
    Experimental; prefer `ssq_cwt` & `ssq_stft`.
    # Arguments:
        Wx, dWx (see w), ssq_freqs, scales, Sfs, fs, t, squeezing, maprange,
        wavelet, gamma, was_padded, flipud:
            See `help(ssqueezing.ssqueeze)`.
        rpadded: bool (default None) / None
            Whether `Wx` (and `dWx`) is passed in padded. `True` will unpad
            `Wx` and `dWx`  before SSQ. Also, if `dWx` is None:
                - `rpadded==False`: will pad `Wx` in computing `dWx` if
                `padtype!=None`, then unpad both before SSQ
                - `rpadded==True`: won't pad `Wx` regardless of `padtype`
        padtype: str / None
            Used if `rpadded==False`. See `help(utils.padsignal)`. Note that
            padding `Wx` isn't same as passing padded `Wx` from `cwt`, but it
            can get close.
        N, n1: int / None
            Needed if `rpadded==True` to unpad `Wx` & `dWx` as `Wx[:, n1:n1 + N]`.
        difftype, difforder: str
            Used if `dWx = None` and `transform == 'cwt'`; see `help(ssq_cwt)`.
        get_w, get_dWx: bool
            See `help(ssq_cwt)`.
    # Returns:
        Tx, Wx, ssq_freqs, scales, Sfs, w, dWx
    """
    w, Wx, dWx, Sfs, gamma = phase_transform(
        Wx, dWx, difftype, difforder=difforder, gamma=gamma, rpadded=rpadded,
        padtype=padtype, N=N, n1=n1, get_w=get_w, fs=fs, transform=transform)

    if w is not None and not get_dWx:
        dWx = None

    if maprange is None:
        maprange = 'peak' if transform == 'cwt' else 'maximal'
    Tx, ssq_freqs = ssqueeze(Wx, w, ssq_freqs, scales, Sfs, fs=fs, t=t,
                             squeezing=squeezing, maprange=maprange,
                             wavelet=wavelet, gamma=gamma, was_padded=was_padded,
                             flipud=flipud, dWx=dWx, transform=transform)
    return Tx, Wx, ssq_freqs, scales, Sfs, w, dWx


def phase_transform(Wx, dWx=None, difftype='trig', difforder=4, gamma=None,
                    fs=1., Sfs=None, rpadded=False, padtype='reflect', N=None,
                    n1=None, get_w=False, transform='cwt'):
    """Unified method for CWT & STFT SSQ phase transforms.
    See `help(_ssq_cwt.phase_cwt)` and `help(_ssq_stft.phase_stft)`.
    """
    def _cwt(Wx, dWx, fs, gamma, N, n1, difftype, difforder, rpadded, padtype,
             get_w):
        # infer `N` and/or `n1`
        if N is None and not rpadded:
            N = Wx.shape[-1]
        if n1 is None:
            _, n1, _ = p2up(N)
        # compute `dWx` if not supplied
        if dWx is None:
            dWx = trigdiff(Wx, fs, padtype, rpadded, N=N, n1=n1, transform='cwt')

        if get_w:
            if difftype == 'trig':
                # calculate instantaneous frequency directly from the
                # frequency-domain derivative
                w = phase_cwt(Wx, dWx, difftype, gamma)
            elif difftype == 'phase':
                # !!! bad; yields negatives, and forcing abs(w) doesn't help
                # calculate inst. freq. from unwrapped phase of CWT
                w = phase_cwt(Wx, None, difftype, gamma)
            elif difftype == 'numeric':
                # !!! tested to be very inaccurate for small scales
                # calculate derivative numericly
                Wx = Wx[:, (n1 - 4):(n1 + N + 4)]
                dt = 1 / fs
                w = phase_cwt_num(Wx, dt, difforder, gamma)
        else:
            w = None
        return w, Wx, dWx

    def _stft(Wx, dWx, fs, gamma, Sfs, get_w):
        if Sfs is None:
            Sfs = _make_Sfs(Wx, fs)
        if get_w:
            w = phase_stft(Wx, dWx, Sfs, gamma)
        else:
            w = None
        return w, Wx, dWx, Sfs

    # validate args
    if transform == 'stft' and dWx is None:
        raise NotImplementedError("`phase_transform` without `dWx` for "
                                  "STFT is not currently supported.")
    if rpadded and N is None:
        raise ValueError("`rpadded=True` requires `N`")
    if Wx.ndim > 2 and get_w:
        raise NotImplementedError("`get_w=True` unsupported with batched input.")

    # gamma
    if gamma is None:
        gamma = 10 * (EPS64 if S.is_dtype(Wx, 'complex128') else EPS32)

    # take phase transform if `get_w` else only compute `dWx` (if None)
    if transform == 'cwt':
        w, Wx, dWx = _cwt(Wx, dWx, fs, gamma, N, n1, difftype, difforder,
                          rpadded, padtype, get_w)
        Sfs = None
    elif transform == 'stft':
        w, Wx, dWx, Sfs = _stft(Wx, dWx, fs, gamma, Sfs, get_w)

    return w, Wx, dWx, Sfs, gamma
