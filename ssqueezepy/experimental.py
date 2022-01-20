# -*- coding: utf-8 -*-
import warnings
import numpy as np
from .wavelets import Wavelet, center_frequency
from .utils import cwt_scalebounds


__all__ = ['freq_to_scale', 'scale_to_freq']


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
    Npad = int(2**np.ceil(np.log2(N))) if padtype is not None else N
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
