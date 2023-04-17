# -*- coding: utf-8 -*-
import numpy as np
from types import FunctionType
from .algos import indexed_sum_onfly, ssqueeze_fast
from .utils import p2up, process_scales, infer_scaletype, _process_fs_and_t
from .utils import NOTE, pi, logscale_transition_idx, assert_is_one_of
from .utils.backend import Q
from .utils.common import WARN
from .utils import backend as S
from .wavelets import center_frequency


def ssqueeze(Wx, w=None, ssq_freqs=None, scales=None, Sfs=None, fs=None, t=None,
             squeezing='sum', maprange='maximal', wavelet=None, gamma=None,
             was_padded=True, flipud=False, dWx=None, transform='cwt'):
    """Synchrosqueezes the CWT or STFT of `x`.

    # Arguments:
        Wx or Sx: np.ndarray
            CWT or STFT of `x`. CWT is assumed L1-normed, and STFT with
            `modulated=True`. If 3D, will treat elements along dim0 as independent
            inputs, synchrosqueezing one-by-one (but memory-efficiently).

        w: np.ndarray / None
            Phase transform of `Wx` or `Sx`. Must be >=0.
            If None, `gamma` & `dWx` must be supplied (and `Sfs` for SSQ_STFT).

        ssq_freqs: str['log', 'log-piecewise', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.
            See `help(cwt)` on `'log-piecewise'`.

        scales: str['log', 'log-piecewise', 'linear', ...] / np.ndarray
            See `help(cwt)`.

        Sfs: np.ndarray
            Needed if `transform='stft'` and `dWx=None`. See `help(ssq_stft)`.

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
            See `help(ssq_cwt)`. Only `'maximal'` supported with STFT.

        wavelet: wavelets.Wavelet
            Only used if maprange != 'maximal' to compute center frequencies.
            See `help(cwt)`.

        gamma: float
            See `help(ssq_cwt)`.

        was_padded: bool (default `rpadded`)
            Whether `x` was padded to next power of 2 in `cwt`, in which case
            `maprange` is computed differently.
              - Used only with `transform=='cwt'`.
              - Ignored if `maprange` is tuple.

        flipud: bool (default False)
            Whether to fill `Tx` equivalently to `flipud(Tx)` (faster & less
            memory than calling `Tx = np.flipud(Tx)` afterwards).

        dWx: np.ndarray,
            Used internally by `ssq_cwt` / `ssq_stft`; must pass when `w` is None.

        transform: str['cwt', 'stft']
            Whether `Wx` is from CWT or STFT (`Sx`).

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
    def _ssqueeze(Tx, w, Wx, dWx, nv, ssq_freqs, scales, transform, ssq_scaletype,
                  cwt_scaletype, flipud, gamma, Sfs):
        if transform == 'cwt':
            # Eq 14 [2]; Eq 2.3 [1]
            if cwt_scaletype[:3] == 'log':
                # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
                const = np.log(2) / nv

            elif cwt_scaletype == 'linear':
                # omit /dw since it's cancelled by *dw in inversion anyway
                const = ((scales[1] - scales[0]) / scales).squeeze()
        elif transform == 'stft':
            const = (ssq_freqs[1] - ssq_freqs[0])  # 'alpha' from [3]

        ssq_logscale = ssq_scaletype.startswith('log')
        # do squeezing by finding which frequency bin each phase transform point
        # w[a, b] lands in (i.e. to which f in ssq_freqs each w[a, b] is closest)
        # equivalent to argmin(abs(w[a, b] - ssq_freqs)) for every a, b
        # Tx[k[i, j], j] += Wx[i, j] * norm -- (see below method's docstring)
        if w is None:
            ssqueeze_fast(Wx, dWx, ssq_freqs, const, ssq_logscale, flipud,
                          gamma, out=Tx, Sfs=Sfs)
        else:
            indexed_sum_onfly(Wx, w, ssq_freqs, const, ssq_logscale, flipud,
                              out=Tx)

    def _process_args(Wx, w, fs, t, transform, squeezing, scales, maprange,
                      wavelet, dWx):
        if w is None and (dWx is None or gamma is None):
            raise ValueError("if `w` is None, `dWx` and `gamma` must not be.")
        elif w is not None and w.min() < 0:
            raise ValueError("found negatives in `w`")

        _check_ssqueezing_args(squeezing, maprange, transform=transform,
                               wavelet=wavelet)

        if scales is None and transform == 'cwt':
            raise ValueError("`scales` can't be None if `transform == 'cwt'`")

        N = Wx.shape[-1]
        dt, *_ = _process_fs_and_t(fs, t, N)
        return N, dt

    N, dt = _process_args(Wx, w, fs, t, transform, squeezing, scales,
                          maprange, wavelet, dWx)

    if transform == 'cwt':
        scales, cwt_scaletype, _, nv = process_scales(scales, N, get_params=True)
    else:
        cwt_scaletype, nv = None, None

    # handle `ssq_freqs` & `ssq_scaletype`
    if not (isinstance(ssq_freqs, np.ndarray) or S.is_tensor(ssq_freqs)):
        if isinstance(ssq_freqs, str):
            ssq_scaletype = ssq_freqs
        else:
            # default to same scheme used by `scales`
            ssq_scaletype = cwt_scaletype

        if ((maprange == 'maximal' or isinstance(maprange, tuple)) and
                ssq_scaletype == 'log-piecewise'):
            raise ValueError("can't have `ssq_scaletype = log-piecewise` or "
                             "tuple with `maprange = 'maximal'` "
                             "(got %s)" % str(maprange))
        ssq_freqs = _compute_associated_frequencies(
            scales, N, wavelet, ssq_scaletype, maprange, was_padded, dt,
            transform)
    elif transform == 'stft':
        # removes warning per issue with `infer_scaletype`
        # future TODO: shouldn't need this
        ssq_scaletype = 'linear'
    else:
        ssq_scaletype, _ = infer_scaletype(ssq_freqs)

    # transform `Wx` if needed
    if isinstance(squeezing, FunctionType):
        Wx = squeezing(Wx)
    elif squeezing == 'lebesgue':  # from reference [3]
        Wx = S.ones(Wx.shape, dtype=Wx.dtype) / len(Wx)
    elif squeezing == 'abs':
        Wx = Q.abs(Wx)

    # synchrosqueeze
    Tx = S.zeros(Wx.shape, dtype=Wx.dtype)
    args = (nv, ssq_freqs, scales, transform, ssq_scaletype,
            cwt_scaletype, flipud, gamma, Sfs)
    if Wx.ndim == 2:
        _ssqueeze(Tx, w, Wx, dWx, *args)
    elif Wx.ndim == 3:
        w, dWx = [(g if g is not None else [None]*len(Tx))
                  for g in (w, dWx)]
        for _Tx, _w, _Wx, _dWx in zip(Tx, w, Wx, dWx):
            _ssqueeze(_Tx, _w, _Wx, _dWx, *args)

    # `scales` go high -> low
    if (transform == 'cwt' and not flipud) or flipud:
        if not isinstance(ssq_freqs, np.ndarray):
            import torch
            ssq_freqs = torch.flip(ssq_freqs, (0,))
        else:
            ssq_freqs = ssq_freqs[::-1]

    return Tx, ssq_freqs


#### `ssqueeze` utils ########################################################
def _ssq_freqrange(maprange, dt, N, wavelet, scales, was_padded):
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
        kw = dict(wavelet=wavelet, N=N, maprange=maprange, dt=dt,
                  was_padded=was_padded)
        fm = _get_center_frequency(**kw, scale=scales[-1])
        fM = _get_center_frequency(**kw, scale=scales[0])
    return fm, fM


def _compute_associated_frequencies(scales, N, wavelet, ssq_scaletype, maprange,
                                    was_padded=True, dt=1, transform='cwt'):
    fm, fM = _ssq_freqrange(maprange, dt, N, wavelet, scales, was_padded)

    na = len(scales)
    # frequency divisions `w_l` to reassign to in Synchrosqueezing
    if ssq_scaletype == 'log':
        # [fm, ..., fM]
        ssq_freqs = fm * np.power(fM / fm, np.arange(na)/(na - 1))

    elif ssq_scaletype == 'log-piecewise':
        idx = logscale_transition_idx(scales)
        if idx is None:
            ssq_freqs = fm * np.power(fM / fm, np.arange(na)/(na - 1))
        else:
            f0, f2 = fm, fM
            # note that it's possible for f1 == f0 per discretization limitations,
            # in which case `sqf1` will contain the same value repeated
            f1 = _get_center_frequency(wavelet, N, maprange, dt, scales[idx],
                                       was_padded)

            # here we don't know what the pre-downsampled `len(scales)` was,
            # so we take a longer route by piecewising respective center freqs
            t1 = np.arange(0,  na - idx - 1)/(na - 1)
            t2 = np.arange(na - idx - 1, na)/(na - 1)
            # simulates effect of "endpoint" since we'd need to know `f2`
            # with `endpoint=False`
            t1 = np.hstack([t1, t2[0]])

            sqf1 = _exp_fm(t1, f0, f1)[:-1]
            sqf2 = _exp_fm(t2, f1, f2)
            ssq_freqs = np.hstack([sqf1, sqf2])

            ssq_idx = logscale_transition_idx(ssq_freqs)
            if ssq_idx is None:
                raise Exception("couldn't find logscale transition index of "
                                "generated `ssq_freqs`; something went wrong")
            assert (na - ssq_idx) == idx, "{} != {}".format(na - ssq_idx, idx)

    else:
        if transform == 'cwt':
            ssq_freqs = np.linspace(fm, fM, na)
        elif transform == 'stft':
            ssq_freqs = np.linspace(0, .5, na) / dt
    return ssq_freqs


def _exp_fm(t, fmin, fmax):
    tmin, tmax = t.min(), t.max()
    a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
    b = fmax**(1/tmax) * (1/a)**(1/tmax)
    return a*b**t


def _get_center_frequency(wavelet, N, maprange, dt, scale, was_padded):
    if was_padded:
        N = p2up(N)[0]
    kw = dict(wavelet=wavelet, N=N, scale=scale, kind=maprange)
    if maprange == 'energy':
        kw['force_int'] = True

    wc = center_frequency(**kw)
    fc = wc / (2*pi) / dt
    return fc


#### misc ####################################################################
def _check_ssqueezing_args(squeezing, maprange=None, wavelet=None, difftype=None,
                           difforder=None, get_w=None, transform='cwt'):
    if transform not in ('cwt', 'stft'):
        raise ValueError("`transform` must be one of: cwt, stft "
                         "(got %s)" % squeezing)

    if not isinstance(squeezing, (str, FunctionType)):
        raise TypeError("`squeezing` must be string or function "
                        "(got %s)" % type(squeezing))
    elif isinstance(squeezing, str):
        assert_is_one_of(squeezing, 'squeezing', ('sum', 'lebesgue', 'abs'))

    # maprange
    if maprange is not None:
        if isinstance(maprange, (tuple, list)):
            if not all(isinstance(m, (float, int)) for m in maprange):
                raise ValueError("all elements of `maprange` must be "
                                 "float or int")
        elif isinstance(maprange, str):
            assert_is_one_of(maprange, 'maprange', ('maximal', 'peak', 'energy'))
        else:
            raise TypeError("`maprange` must be str, tuple, or list "
                            "(got %s)" % type(maprange))

        if isinstance(maprange, str) and maprange != 'maximal':
            if transform != 'cwt':
                NOTE("string `maprange` currently only functional with "
                     "`transform='cwt'`")
            elif wavelet is None:
                raise ValueError(f"maprange='{maprange}' requires `wavelet`")

    # difftype
    if difftype is not None:
        if difftype not in ('trig', 'phase', 'numeric'):
            raise ValueError("`difftype` must be one of: direct, phase, numeric"
                             " (got %s)" % difftype)
        elif difftype != 'trig':
            from .configs import USE_GPU
            if USE_GPU():
                raise ValueError("GPU computation only supports "
                                 "`difftype = 'trig'`")
            elif not get_w:
                raise ValueError("`difftype != 'trig'` requires `get_w = True`")

    # difforder
    if difforder is not None:
        if difftype != 'numeric':
            WARN("`difforder` is ignored if `difftype != 'numeric'")
        elif difforder not in (1, 2, 4):
            raise ValueError("`difforder` must be one of: 1, 2, 4 "
                             "(got %s)" % difforder)
    elif difftype == 'numeric':
        difforder = 4

    return difforder
