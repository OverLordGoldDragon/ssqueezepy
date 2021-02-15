# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, ifft, ifftshift
from .utils import WARN, adm_cwt, adm_ssq, _process_fs_and_t
from .utils import padsignal, process_scales, logscale_transition_idx
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


def cwt(x, wavelet='gmw', scales='log-piecewise', fs=None, t=None, nv=32,
        l1_norm=True, derivative=False, padtype='reflect', rpadded=False,
        vectorized=True, order=0, average=None):
    """Continuous Wavelet Transform, discretized, as described in
    Sec. 4.3.3 of [1] and Sec. IIIA of [2]. Uses a form of discretized
    convolution theorem via wavelets in the Fourier domain and FFT of input.

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.
                  See `help(wavelets.Wavelet)`.

        scales: str['log', 'log-piecewise', 'linear', 'log:maximal', ...]
                / np.ndarray
            CWT scales.
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'log-piecewise': 'log' except very high `scales` are downsampled
                to prevent redundancy. This is recommended. See
                https://github.com/OverLordGoldDragon/ssqueezepy/issues/
                29#issuecomment-776792726
                - 'linear': linearly distributed scales.
                  !!! EXPERIMENTAL; default scheme for len(x)>2048 performs
                  poorly (and there may not be a good non-piecewise scheme).

            str assumes default `preset='minimal-low'`, which can be changed via
            e.g. 'log:maximal', 'linear:minimal'.
            See `preset` in `help(utils.cwt_scalebounds)`.

        nv: int
            Number of voices (wavelets per octave). Suggested >= 16.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which for
            `maprange='maximal'` makes ssq frequencies range from 1/dT to 0.5*fs,
            i.e. as fraction of reference sampling rate up to Nyquist limit;
            dT = total duration (N/fs).
            Used to compute `dt`, which is only used if `derivative=True`.
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Used to compute `dt`, which is only used if `derivative=True`.
            Overrides `fs` if not None.

        l1_norm: bool (default True)
            Whether to L1-normalize the CWT, which yields a more representative
            distribution of energies and component amplitudes than L2 (see [3]).
            If False (default True), uses L2 norm.

        derivative: bool (default False)
            Whether to compute and return `dWx`. Requires `fs` or `t`.

        padtype: str / None
            Pad scheme to apply on input. See `help(utils.padsignal)`.
            `None` -> no padding.

        rpadded: bool (default False)
             Whether to return padded Wx and dWx.
             `False` drops the added padding per `padtype` to return Wx and dWx
             of .shape[1] == len(x).

        vectorized: bool (default True)
            Whether to compute quantities for all scales at once, which is
            faster but uses more memory.

        order: int (default 0) / tuple[int] / range
            > 0 computes `cwt` with higher-order GMWs. If tuple, computes
            `cwt` at each specified order. See `help(_cwt.cwt_higher_order)`.

        average: bool / None
            Only used for tuple `order`; see `help(_cwt.cwt_higher_order)`.

    # Returns:
        Wx: [na x n] np.ndarray (na = number of scales; n = len(x))
            CWT of `x`. (rows=scales, cols=timeshifts)
        scales: [na] np.ndarray
            Scales at which CWT was computed.
        dWx: [na x n] np.ndarray
            Returned only if `derivative=True`.
            Time-derivative of the CWT of `x`, computed via frequency-domain
            differentiation (effectively, derivative of trigonometric
            interpolation; see [4]). Implements as described in Sec IIIB of [2].

    # References:
        1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Rectification of the Bias in the Wavelet Power Spectrum.
        Y. Liu, X. S. Liang, R. H. Weisberg.
        http://ocg6.marine.usf.edu/~liu/Papers/Liu_etal_2007_JAOT_wavelet.pdf

        4. The Exponential Accuracy of Fourier and Chebyshev Differencing Methods.
        E. Tadmor.
        http://webhome.auburn.edu/~jzl0097/teaching/math_8970/Tadmor_86.pdf

        5. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        cwt_fw.m
    """
    def _vectorized(xh, scales, wavelet, pn, derivative):
        Wx = (wavelet(scale=scales, nohalf=False) * pn
              ).astype('complex128')
        if derivative:
            dWx = (1j * wavelet.xi / dt) * Wx

        Wx = ifftshift(ifft(Wx * xh, axis=-1), axes=-1)
        if derivative:
            dWx = ifftshift(ifft(dWx * xh, axis=-1), axes=-1)
        return (Wx, dWx) if derivative else (Wx, None)

    def _for_loop(xh, scales, wavelet, pn, derivative):
        Wx = np.zeros((len(scales), len(xh))).astype('complex128')
        if derivative:
            dWx = Wx.copy()

        for i, a in enumerate(scales):
            # sample FT of wavelet at scale `a`
            # * pn = freq-domain spectral reversal to center time-domain wavelet
            psih = wavelet(scale=a, nohalf=False) * pn
            Wx[i] = ifftshift(ifft(psih * xh))

            if derivative:
                dpsih = (1j * wavelet.xi / dt) * psih
                dWx[i] = ifftshift(ifft(dpsih * xh))
        return (Wx, dWx) if derivative else (Wx, None)

    def _process_args(x, scales, nv, fs, t):
        if not isinstance(x, np.ndarray):
            raise TypeError("`x` must be a numpy array (got %s)" % type(x))
        elif x.ndim != 1:
            raise ValueError("`x` must be 1D (got x.ndim == %s)" % x.ndim)

        if np.isnan(x.max()) or np.isinf(x.max()) or np.isinf(x.min()):
            WARN("found NaN or inf values in `x`; will zero")
            replace_at_inf_or_nan(x, replacement=0.)

        if isinstance(scales, np.ndarray):
            nv = None
        N = x.shape[-1]
        dt, *_ = _process_fs_and_t(fs, t, N=N)
        return N, nv, dt

    if isinstance(order, (tuple, list, range)) or order > 0:
        kw = dict(wavelet=wavelet, scales=scales, fs=fs, t=t, nv=nv,
                  l1_norm=l1_norm, derivative=derivative, padtype=padtype,
                  rpadded=rpadded, vectorized=vectorized)
        return cwt_higher_order(x, order=order, average=average, **kw)

    N, nv, dt = _process_args(x, scales, nv, fs, t)

    if padtype is not None:
        xp, _, n1, _ = padsignal(x, padtype, get_params=True)
    else:
        xp = x

    # zero-mean `xp`, take to freq-domain
    xp -= xp.mean()
    xh = fft(xp, axis=-1)

    # validate `wavelet`, process `scales`, define `pn` for later
    wavelet = _process_gmw_wavelet(wavelet, l1_norm)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet, nv=nv)
    pn = (-1)**np.arange(xp.shape[-1])

    # temporarily adjust `wavlet.N`, take CWT
    wavelet_N_orig = wavelet.N
    wavelet.N = len(xp)
    Wx, dWx = (_vectorized(xh, scales, wavelet, pn, derivative) if vectorized else
               _for_loop(  xh, scales, wavelet, pn, derivative))
    wavelet.N = wavelet_N_orig  # restore

    # handle unpadding, normalization
    if not rpadded and padtype is not None:
        # shorten to pre-padded size
        Wx = Wx[:, n1:n1 + N]
        if derivative:
            dWx = dWx[:, n1:n1 + N]
    if not l1_norm:
        # normalize energy per L2 wavelet norm, else already L1-normalized
        Wx *= np.sqrt(scales)
        if derivative:
            dWx *= np.sqrt(scales)

    return ((Wx, scales, dWx) if derivative else
            (Wx, scales))


def icwt(Wx, wavelet='gmw', scales='log', nv=None, one_int=True, x_len=None,
         x_mean=0, padtype='zero', rpadded=False, l1_norm=True):
    """The inverse Continuous Wavelet Transform of `Wx`, via double or
    single integral.

    # Arguments:
        Wx: np.ndarray
            CWT computed via `ssqueezepy.cwt`.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.

        scales: str['log', 'linear', 'log:maximal', ...] / np.ndarray
            See help(cwt).

        nv: int / None
            Number of voices. Suggested >= 32. Needed if `scales` isn't array
            (will default to `cwt`'s).

        one_int: bool (default True)
            Whether to use one-integral iCWT or double.
            Current one-integral implementation performs best.
                - True: Eq 2.6, modified, of [3]. Explained in [4].
                - False: Eq 4.67 of [1]. Explained in [5].

        x_len: int / None. Length of `x` used in forward CWT, if different
            from Wx.shape[1] (default if None).

        x_mean: float. mean of original `x` (not picked up in CWT since it's an
            infinite scale component). Default 0.

        padtype: str
            Pad scheme to apply on input, in case of `one_int=False`.
            See `help(utils.padsignal)`.

        rpadded: bool (default False)
            True if Wx is padded (e.g. if used `cwt(, rpadded=True)`).

        l1_norm: bool (default True)
            True if Wx was obtained via `cwt(, l1_norm=True)`.

    # Returns:
        x: np.ndarray
            The signal, as reconstructed from Wx.

    # References:
        1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        4. One integral inverse CWT. OverLordGoldDragon.
        https://dsp.stackexchange.com/a/71274/50076

        5. Inverse CWT derivation. OverLordGoldDragon.
        https://dsp.stackexchange.com/a/71148/50076

        6. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_fw.m
    """
    #### Prepare for inversion ###############################################
    na, n = Wx.shape
    x_len = x_len or n
    if not isinstance(scales, np.ndarray) and nv is None:
        nv = 32  # must match forward's; default to `cwt`'s

    wavelet = _process_gmw_wavelet(wavelet, l1_norm)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    # will override `nv` to match `scales`'s
    scales, scaletype, _, nv = process_scales(scales, x_len, wavelet, nv=nv,
                                              get_params=True)
    assert (len(scales) == na), "%s != %s" % (len(scales), na)

    #### Handle piecewise scales case ########################################
    # `nv` must be left unspecified so it's inferred automatically from `scales`
    # in `process_scales` for each piecewise case
    if scaletype == 'log-piecewise':
        kw = dict(wavelet=wavelet, one_int=one_int, x_len=x_len, x_mean=x_mean,
                  padtype=padtype, rpadded=rpadded, l1_norm=l1_norm)

        idx = logscale_transition_idx(scales)
        x  = icwt(Wx[:idx], scales=scales[:idx], **kw)
        x += icwt(Wx[idx:], scales=scales[idx:], **kw)
        return x
    ##########################################################################

    #### Invert ##############################################################
    if one_int:
        x = _icwt_1int(Wx, scales, scaletype, l1_norm)
    else:
        x = _icwt_2int(Wx, scales, scaletype, l1_norm,
                       wavelet, x_len, padtype, rpadded)

    # admissibility coefficient
    Cpsi = (adm_ssq(wavelet) if one_int else
            adm_cwt(wavelet))
    if scaletype == 'log':
        # Eq 4.67 in [1]; Theorem 4.5 in [1]; below Eq 14 in [2]
        # ln(2**(1/nv)) == ln(2)/nv == diff(ln(scales))[0]
        x *= (2 / Cpsi) * np.log(2 ** (1 / nv))
    else:
        x *= (2 / Cpsi)

    x += x_mean  # CWT doesn't capture mean (infinite scale)
    return x


def _icwt_2int(Wx, scales, scaletype, l1_norm, wavelet, x_len,
               padtype='zero', rpadded=False):
    """Double-integral iCWT; works with any(?) wavelet."""
    # add CWT padding if it doesn't exist
    if not rpadded:
        Wx, n_up, n1, _ = padsignal(Wx, padtype=padtype, get_params=True)

    # see help(cwt) on `norm` and `pn`
    norm = _icwt_norm(scaletype, l1_norm)
    pn = (-1)**np.arange(n_up)
    x = np.zeros(n_up)

    for a, Wxa in zip(scales, Wx):  # TODO vectorize?
        psih = wavelet(scale=a, N=n_up) * pn
        xa = ifftshift(ifft(fft(Wxa) * psih))  # convolution theorem
        x += xa.real / norm(a)

    x = x[n1:n1 + x_len]  # keep the unpadded part
    return x


def _icwt_1int(Wx, scales, scaletype, l1_norm):
    """One-integral iCWT; assumes analytic wavelet."""
    norm = _icwt_norm(scaletype, l1_norm)
    return (Wx.real / norm(scales)).sum(axis=0)


def _icwt_norm(scaletype, l1_norm):
    if l1_norm:
        norm = ((lambda a: 1) if scaletype == 'log' else
                (lambda a: a))
    else:
        if scaletype == 'log':
            norm = lambda a: a**.5
        elif scaletype == 'linear':
            norm = lambda a: a**1.5
    return norm


def _process_gmw_wavelet(wavelet, l1_norm):
    """Ensure `norm` for GMW is consistent with `l1_norm`."""
    norm = 'bandpass' if l1_norm else 'energy'

    if isinstance(wavelet, str) and wavelet.lower()[:3] == 'gmw':
        wavelet = ('gmw', {'norm': norm})

    elif isinstance(wavelet, tuple) and wavelet[0].lower()[:3] == 'gmw':
        wavelet, wavopts = wavelet
        wavopts['norm'] = wavopts.get('norm', norm)
        wavelet = (wavelet, wavopts)

    elif isinstance(wavelet, Wavelet):
        if wavelet.name == 'GMW L2' and l1_norm:
            raise ValueError("using GMW L2 wavelet with `l1_norm=True`")
        elif wavelet.name == 'GMW L1' and not l1_norm:
            raise ValueError("using GMW L1 wavelet with `l1_norm=False`")
    return wavelet


def cwt_higher_order(x, wavelet='gmw', order=1, average=None, **kw):
    """Compute `cwt` with GMW wavelets of order 0 to `order`. See `help(cwt)`.

    Yields lower variance and more noise robust representation. See ref[1].

    # Arguments:
        x: np.ndarray
            Input, 1D.

        wavelet: str / wavelets.Wavelet

        order: int / tuple[int] / range
            Order of GMW to use for CWT. If tuple, will compute for each
            order specified in tuple, subject to `average`.

        average: bool (default True if `order` is tuple)
            If True, will take arithmetic mean of resulting `Wx` (and `dWx`
            if `derivative=True`), else return as list. Note for phase transform,
            one should compute derivative of averaged `Wx` rather than take
            average of individual `dWx`s.
            Ignored with non-tuple `order.

        kw: dict / kwargs
            Arguments to `cwt`.
            If `scales` is string, will reuse zeroth-order's; zeroth order
            isn't included in `order`, will set from wavelet at `order=0`.

    # References
        [1] Generalized Morse Wavelets. S. C. Olhede, A. T. Walden. 2002.
        https://spiral.imperial.ac.uk/bitstream/10044/1/1150/1/
        OlhedeWaldenGenMorse.pdf
    """
    def _process_wavelet(wavelet, order):
        wavelet = Wavelet._init_if_not_isinstance(wavelet)
        if not wavelet.name.lower().startswith('gmw'):
            raise ValueError("`wavelet` must be GMW for higher-order transforms "
                             "(got %s)" % wavelet.name)

        wavopts = wavelet.config.copy()
        wavopts.pop('order')
        wavelets = [Wavelet(('gmw', dict(order=k, **wavopts))) for k in order]
        return wavelets, wavopts

    def _process_args(wavelet, order, average, kw):
        if isinstance(order, (list, range)):
            order = tuple(order)
        if not isinstance(order, (list, tuple)):
            order = [order]
        if len(order) == 1 and average:
            WARN("`average` ignored with single `order`")
            average = False

        wavelets, wavopts = _process_wavelet(wavelet, order)

        scales = kw.get('scales', 'log-piecewise')
        if isinstance(scales, str):
            wav = Wavelet(('gmw', dict(order=0, **wavopts)))
            scales = process_scales(scales, len(x), wavelet=wav,
                                    nv=kw.get('nv', 32))
        kw['scales'] = scales

        return wavelets, order, average, scales

    wavelets, order, average, scales = _process_args(wavelet, order, average, kw)

    Wx_all = []
    derivative = kw.get('derivative', False)
    if derivative:
        dWx_all = []

    # take the CWTs
    for k in range(len(order)):
        out = cwt(x, wavelets[k], order=0, **kw)
        Wx_all.append(out[0])
        if derivative:
            dWx_all.append(out[-1])

    # handle averaging; strip `Wx_all` of list container if only one array
    if average or (average is None and isinstance(order, tuple)):
        Wx_all = np.mean(np.vstack([Wx_all]), axis=0)
        if derivative:
            dWx_all = np.mean(np.vstack([dWx_all]), axis=0)
    elif len(Wx_all) == 1:
        Wx_all = Wx_all[0]
        if derivative:
            dWx_all = dWx_all[0]

    return ((Wx_all, kw['scales'], dWx_all) if derivative else
            (Wx_all, kw['scales']))
