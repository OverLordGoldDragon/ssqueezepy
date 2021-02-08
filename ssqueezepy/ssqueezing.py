# -*- coding: utf-8 -*-
import numpy as np
from types import FunctionType
from .algos import find_closest, indexed_sum, replace_at_inf
from .utils import process_scales, _infer_scaletype, _process_fs_and_t
from .utils import NOTE, pi
from .wavelets import center_frequency


def ssqueeze(Wx, w, ssq_freqs=None, scales=None, fs=None, t=None, transform='cwt',
             squeezing='sum', maprange='maximal', wavelet=None):
    """Synchrosqueezes the CWT or STFT of `x`.

    # Arguments:
        Wx or Sx: np.ndarray
            CWT or STFT of `x`. `Wx` is assumed L1-normed.

        w: np.ndarray
            Phase transform of `Wx` or `Sx`. Must be >=0.

        ssq_freqs: str['log', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.

        scales: str['log', 'linear', 'log:maximal', ...] / np.ndarray
            See help(cwt).

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5*fs, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        transform: str['cwt', 'stft']
            Whether `Wx` is from CWT or STFT (`Sx`).

        squeezing: str['sum', 'lebesgue'] / function
            - 'sum': summing `Wx` according to `w`. Standard synchrosqueezing.
            Invertible.
            - 'lebesgue': as in [3], summing `Wx=ones()/len(Wx)`. Effectively,
            raw `Wx` phase is synchrosqueezed, independent of `Wx` values. Not
            recommended with CWT or STFT with `modulated=True`. Not invertible.
            For `modulated=False`, provides a more stable and accurate
            representation.
            - 'abs': summing `abs(Wx)` according to `w`. Not invertible
            (but theoretically possible to get close with least-squares estimate,
             so much "more invertible" than 'lebesgue'). Alt to 'lebesgue',
            providing same benefits while losing much less information.

            Custom function can be used to transform `Wx` arbitrarily for
            summation, e.g. `Wx**2` via `lambda x: x**2`. Output shape
            must match `Wx.shape`.

        maprange: str['maximal', 'peak', 'energy'] / tuple(float, float)
            See `help(ssq_cwt)`.

        wavelet: wavelets.Wavelet
            Only used if maprange != 'maximal' to compute center frequencies.
            See `help(cwt)`.

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.

    # References:
        1. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_squeeze.m
    """
    def _ssqueeze(w, Wx, nv, ssq_freqs, transform, ssq_scaletype, cwt_scaletype):
        # incorporate threshold by zeroing out Inf values, so they get ignored
        Wx = replace_at_inf(Wx, ref=w, replacement=0)

        # do squeezing by finding which frequency bin each phase transform point
        # w[a, b] lands in (i.e. to which f in ssq_freqs each w[a, b] is closest)
        # equivalent to argmin(abs(w[a, b] - ssq_freqs)) for every a, b
        with np.errstate(divide='ignore', invalid='ignore'):
            k = (find_closest(w, ssq_freqs) if ssq_scaletype != 'log' else
                 find_closest(np.log2(w), np.log2(ssq_freqs)))

        # Tx[k[i, j], j] += Wx[i, j] * norm
        if transform == 'cwt':
            # Eq 14 [2]; Eq 2.3 [1]
            if cwt_scaletype == 'log':
                # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
                Tx = indexed_sum(Wx * np.log(2) / nv, k)
            elif cwt_scaletype == 'linear':
                # omit /dw since it's cancelled by *dw in inversion anyway
                da = (scales[1] - scales[0])
                Tx = indexed_sum(Wx / scales * da, k)
        elif transform == 'stft':
            df = (ssq_freqs[1] - ssq_freqs[0])  # 'alpha' from [3]
            Tx = indexed_sum(Wx * df, k)
        return Tx

    def _ssq_freqrange(maprange, dt, N, wavelet, scales):
        if isinstance(maprange, tuple):
            fm, fM = maprange
        elif maprange == 'maximal':
            dT = dt * N
            # normalized frequencies to map discrete-domain to physical:
            #     f[[cycles/samples]] -> f[[cycles/second]]
            # minimum measurable (fundamental) frequency of data
            fm = 1 / dT
            # maximum measurable (Nyquist) frequency of data
            fM = 1 / (2 * dt)
        elif maprange in ('peak', 'energy'):
            kw = dict(wavelet=wavelet, N=N, kind=maprange)
            if maprange == 'energy':
                kw['force_int'] = True
            wm = center_frequency(scale=scales[-1], **kw)
            wM = center_frequency(scale=scales[0],  **kw)
            fm = wm / (2*pi) / dt
            fM = wM / (2*pi) / dt
        return fm, fM

    def _compute_associated_frequencies(dt, na, N, transform, ssq_scaletype,
                                        maprange, wavelet, scales):
        fm, fM = _ssq_freqrange(maprange, dt, N, wavelet, scales)

        # frequency divisions `w_l` to reassign to in Synchrosqueezing
        if ssq_scaletype == 'log':
            # [fm, ..., fM]
            ssq_freqs = fm * np.power(fM / fm, np.arange(na) / (na - 1))
        else:
            if transform == 'cwt':
                ssq_freqs = np.linspace(fm, fM, na)
            elif transform == 'stft':
                ssq_freqs = np.linspace(0, .5, na) / dt
        return ssq_freqs

    def _process_args(w, fs, t, N, transform, squeezing, scales, maprange,
                      wavelet):
        if w.min() < 0:
            raise ValueError("found negatives in `w`")
        if transform not in ('cwt', 'stft'):
            raise ValueError("`transform` must be one of: cwt, stft "
                             "(got %s)" % squeezing)
        _check_ssqueezing_args(squeezing, maprange, transform)

        if scales is None and transform == 'cwt':
            raise ValueError("`scales` can't be None if `transform == 'cwt'`")

        dt, *_ = _process_fs_and_t(fs, t, N)
        return dt

    na, N = Wx.shape
    dt = _process_args(w, fs, t, N, transform, squeezing, scales,
                       maprange, wavelet)

    if transform == 'cwt':
        scales, cwt_scaletype, _, nv = process_scales(scales, N, get_params=True)
    else:
        cwt_scaletype, nv = None, None

    if not isinstance(ssq_freqs, np.ndarray):
        if isinstance(ssq_freqs, str):
            ssq_scaletype = ssq_freqs
        else:
            # default to same scheme used by `scales`
            ssq_scaletype = cwt_scaletype
        ssq_freqs = _compute_associated_frequencies(
            dt, na, N, transform, ssq_scaletype, maprange, wavelet, scales)
    else:
        ssq_scaletype = _infer_scaletype(ssq_freqs)

    if isinstance(squeezing, FunctionType):
        Wx = squeezing(Wx)
    elif squeezing == 'lebesgue':  # from reference [3]
        Wx = np.ones(Wx.shape) / len(Wx)
    elif squeezing == 'abs':
        Wx = np.abs(Wx)

    Tx = _ssqueeze(w, Wx, nv, ssq_freqs, transform, ssq_scaletype, cwt_scaletype)
    return Tx, ssq_freqs


def _check_ssqueezing_args(squeezing, maprange=None, wavelet=None,
                           transform='cwt'):
    supported = ('sum', 'lebesgue', 'abs')
    if not isinstance(squeezing, (str, FunctionType)):
        raise TypeError("`squeezing` must be string or function "
                        "(got %s)" % type(squeezing))
    elif isinstance(squeezing, str) and squeezing not in supported:
        raise ValueError("`squeezing` must be one of: {} (got {})".format(
            ', '.join(supported), squeezing))

    if maprange is None:
        return
    if isinstance(maprange, (tuple, list)):
        if not all(isinstance(m, (float, int)) for m in maprange):
            raise ValueError("all elements of `maprange` must be float or int")
    elif isinstance(maprange, str):
        supported = ('maximal', 'peak', 'energy')
        if maprange not in supported:
            raise ValueError("`maprange` must be one of {} (got {})".format(
                ', '.join(supported), maprange))
    else:
        raise TypeError("`maprange` must be str, tuple, or list "
                        "(got %s)" % type(maprange))

    if isinstance(maprange, str) and maprange != 'maximal':
        if transform != 'cwt':
            NOTE("`maprange` currently only functional with `transform='cwt'`")
        elif wavelet is None:
            raise ValueError(f"must pass `wavelet` with maprange='{maprange}'")
