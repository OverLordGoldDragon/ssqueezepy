import numpy as np
from numpy.fft import fft, ifft, ifftshift
from .utils import WARN, p2up, adm_cwt, adm_ssq, wfilth, _process_fs_and_t
from .utils import padsignal, process_scales
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


def cwt(x, wavelet, scales='log', fs=None, t=None, nv=32, l1_norm=True,
        derivative=False, padtype='reflect', minbounds=False, rpadded=False,
        vectorized=True):
    """Continuous Wavelet Transform, discretized, as described in
    Sec. 4.3.3 of [1] and Sec. IIIA of [2]. Uses a form of discretized
    convolution theorem via wavelets in the Fourier domain and FFT of input.

    # Arguments:
        x: np.ndarray
            Input signal vector.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.

        scales: str['log', 'linear'] / np.ndarray
            CWT scales vector.
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'linear': linearly distributed scales.
                  !!! EXPERIMENTAL; default scheme for len(x)>2048 performs
                  poorly (and there may not be a good non-piecewise scheme).

        nv: int
            Number of voices. Suggested >= 32.

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
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

        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'reflect', 'symmetric', 'replicate').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See `padsignal`.

        minbounds: bool (default False)
            True will mimic MATLAB's setting of min and max CWT `scale`, min set
            such that time-domain wavelet's one stddev spans the N-point signal,
            and max set such that freq-domain wavelet peaks at Nyquist. These
            differ a bit with MATLAB's thresholding, favoring more scales
            (https://www.mathworks.com/help/wavelet/ref/cwtfreqbounds.html)
            Default is False since low frequencies # TODO

        rpadded: bool (default False)
             Whether to return padded Wx and dWx.
             `False` drops the added padding per `padtype` to return Wx and dWx
             of .shape[1] == len(x).

        vectorized: bool (default True)
            Whether to compute quantities for all scales at once, which is
            faster but uses more memory.

    # Returns:
        Wx: [na x n] np.ndarray (na = number of scales; n = len(x))
            The CWT of `x`. (rows=scales, cols=timeshifts)
        scales: [na] np.ndarray
            Scales at which CWT was computed.
        x_mean: float
            mean of `x` to use in inversion (CWT needs scale=inf to capture).
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
    def _vectorized(xh, scales, psihfn, pn, derivative):
        Wx = (psihfn(scale=scales) * pn).astype('complex128')
        if derivative:
            dWx = (1j * psihfn.xi / dt) * Wx

        Wx = ifftshift(ifft(Wx * xh, axis=-1), axes=-1)
        if derivative:
            dWx = ifftshift(ifft(dWx * xh, axis=-1), axes=-1)
        return (Wx, dWx) if derivative else (Wx, None)

    def _for_loop(xh, scales, psihfn, pn, derivative):
        Wx = np.zeros((len(scales), psihfn.N)).astype('complex128')
        if derivative:
            dWx = Wx.copy()

        for i, a in enumerate(scales):
            # sample FT of wavelet at scale `a`
            # * pn = freq-domain spectral reversal to center time-domain wavelet
            psih = psihfn(scale=a) * pn
            Wx[i] = ifftshift(ifft(psih * xh))

            if derivative:
                dpsih = (1j * psihfn.xi / dt) * psih
                dWx[i] = ifftshift(ifft(dpsih * xh))
        return (Wx, dWx) if derivative else (Wx, None)

    def _process_args(x, scales, fs, t):
        if np.isnan(x.max()) or np.isinf(x.max()) or np.isinf(x.min()):
            print(WARN, "found NaN or inf values in `x`; will zero")
            replace_at_inf_or_nan(x, replacement=0.)
        dt, *_ = _process_fs_and_t(fs, t, N=len(x))
        return dt  # == 1 / fs

    dt = _process_args(x, scales, fs, t)

    x_mean = x.mean()  # store original mean
    n = len(x)         # store original length
    x, nup, n1, n2 = padsignal(x, padtype)

    x -= x.mean()
    xh = fft(x)
    scales = process_scales(scales, n, wavelet, nv=nv, minbounds=minbounds)
    psihfn = (Wavelet(wavelet, N=nup) if not isinstance(wavelet, Wavelet) else
              wavelet)
    pn = (-1) ** np.arange(nup)

    Wx, dWx = (_vectorized(xh, scales, psihfn, pn, derivative) if vectorized else
               _for_loop(  xh, scales, psihfn, pn, derivative))

    if not rpadded:
        # shorten to pre-padded size
        Wx = Wx[:, n1:n1 + n]
        if derivative:
            dWx = dWx[:, n1:n1 + n]
    if not l1_norm:
        # normalize energy per L2 wavelet norm, else already L1-normalized
        Wx *= np.sqrt(scales)
        if derivative:
            dWx *= np.sqrt(scales)

    return ((Wx, scales, x_mean, dWx) if derivative else
            (Wx, scales, x_mean))


# TODO `scales` aren't needed with `l1_norm=True`
def icwt(Wx, wavelet, scales='log', nv=None, one_int=True, x_len=None, x_mean=0,
         padtype='zero', minbounds=False, rpadded=False, l1_norm=True):
    """The inverse continuous wavelet transform of Wx, via double or
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

        scales: str['log', 'linear'] / np.ndarray
            CWT scales vector used in forward-CWT. Alternatively pass
            same set of kwargs (e.g. scales='log', minbounds=False)
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'linear': linearly distributed scales.
                  !!! EXPERIMENTAL; default scheme for len(x)>2048 performs
                  poorly (and there may not be a good non-piecewise scheme).

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
            Pad scheme to apply on input. One of:
                ('zero', 'symmetric', 'replicate').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See `padsignal`.
            !!! currently uses only 'zero'

        minbounds: bool (default False)
            See `help(cwt)`.

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

    scales, scaletype, _, nv = process_scales(
        scales, x_len, wavelet, nv=nv, minbounds=minbounds, get_params=True)
    assert (len(scales) == na), "%s != %s" % (len(scales), na)

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
    nup, n1, n2 = p2up(x_len)
    # add CWT padding if it doesn't exist  # TODO symmetric & other?
    if not rpadded:
        Wx = np.pad(Wx, [[0, 0], [n1, n2]])  # pad time axis, left=n1, right=n2

    norm = _icwt_norm(scaletype, l1_norm, one_int=False)
    x = np.zeros(nup)
    for a, Wxa in zip(scales, Wx):  # TODO vectorize?
        psih = wfilth(wavelet, nup, a, l1_norm=l1_norm)
        xa = ifftshift(ifft(fft(Wxa) * psih))  # convolution theorem
        x += xa.real / norm(a)

    x = x[n1:n1 + x_len]  # keep the unpadded part
    return x


def _icwt_1int(Wx, scales, scaletype, l1_norm):
    """One-integral iCWT; assumes analytic wavelet."""
    norm = _icwt_norm(scaletype, l1_norm, one_int=True)
    return (Wx.real / norm(scales)).sum(axis=0)


def _icwt_norm(scaletype, l1_norm, one_int):
    if l1_norm:
        norm = ((lambda a: 1) if scaletype == 'log' else
                (lambda a: a))
    else:
        if scaletype == 'log':
            norm = ((lambda a: a**.5) if one_int else
                    (lambda a: a))
        elif scaletype == 'linear':
            norm = ((lambda a: a**1.5) if one_int else
                    (lambda a: a**2))
    return norm
