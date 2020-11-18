import numpy as np
from numpy.fft import fft, ifft, ifftshift
from .utils import WARN, p2up, adm_cwt, adm_ssq, wfilth
from .utils import padsignal, process_scales
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


def cwt(x, wavelet, scales='log', dt=1, nv=None, l1_norm=True,
        padtype='symmetric', rpadded=False):
    """Forward continuous wavelet transform, discretized, as described in
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
        dt: float
            Sampling period. t[1] - t[0], if `t` = vector of sampling times.
        l1_norm: bool (default True)
            Whether to L1-normalize the CWT, which yields a more representative
            distribution of energies and component amplitudes than L2 (see [3]).
            If False (default True), uses L2 norm.
        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'symmetric', 'replicate').
            'zero' is most naive, while 'symmetric' (default) partly mitigates
            boundary effects. See `padsignal`.
        rpadded: bool (default False)
             Whether to return padded Wx and dWx.
             `False` drops the added padding per `padtype` to return Wx and dWx
             of .shape[1] == len(x).

    # Returns:
        Wx: [na x n] np.ndarray (na = number of scales; n = len(x))
            The CWT of `x`. (rows=scales, cols=timeshifts)
        scales: [na] np.ndarray.
            Scales at which CWT was computed.
        dWx: [na x n] np.ndarray.
            Time-derivative of the CWT of `x`, computed via frequency-domain
            differentiation (effectively, derivative of trigonometric
            interpolation; see [4]). Implements as described in Sec IIIB of [2].
        x_mean: float.
            mean of `x` to use in inversion (CWT needs scale=inf to capture).

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
    def _process_args(x, scales, dt):
        if not dt > 0:
            raise ValueError("`dt` must be > 0")
        if np.isnan(x.max()) or np.isinf(x.max()) or np.isinf(x.min()):
            print(WARN, "found NaN or inf values in `x`; will zero")
            replace_at_inf_or_nan(x, replacement=0)
        return

    _process_args(x, scales, dt)
    nv = nv or 32

    x_mean = x.mean()  # store original mean
    n = len(x)         # store original length
    x, N, n1, n2 = padsignal(x, padtype)

    scales = process_scales(scales, n, nv=nv)

    # must cast to complex else value assignment discards imaginary component
    Wx = np.zeros((len(scales), N)).astype('complex128')  # N == len(x) (padded)
    dWx = Wx.copy()

    x -= x.mean()
    xh = fft(x)

    pn = (-1) ** np.arange(N)
    psihfn = Wavelet(wavelet, N=N)

    # TODO redefine xi and remove fftshift / ifftshift?
    # TODO vectorize? can FFT all at once if all `psih` are precomputed
    # but keep loop option in case of OOM
    for i, a in enumerate(scales):
        # sample FT of wavelet at scale `a`
        # `* pn` = freq-domain spectral reversal to center time-domain wavelet
        psih = psihfn(a) * pn

        xcpsi = ifftshift(ifft(xh * psih))
        Wx[i] = xcpsi

        dpsih = (1j * psihfn.xi / dt) * psih
        dxcpsi = ifftshift(ifft(dpsih * xh))
        dWx[i] = dxcpsi

    if not rpadded:
        # shorten to pre-padded size
        Wx  = Wx[ :, n1:n1 + n]
        dWx = dWx[:, n1:n1 + n]
    if not l1_norm:
        # normalize energy per L2 wavelet norm, else already L1-normalized
        Wx *= np.sqrt(scales)
        dWx *= np.sqrt(scales)

    return Wx, scales, dWx, x_mean


def icwt(Wx, wavelet, scales='log', one_int=True, x_len=None, x_mean=0,
         padtype='symmetric', rpadded=False, l1_norm=True):
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
            'zero' is most naive, while 'symmetric' (default) partly mitigates
            boundary effects. See `padsignal`.
            !!! currently uses only 'zero'
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
    N, n1, n2 = p2up(x_len)

    scales, ssq_freqs, _, nv = process_scales(scales, x_len, na=na,
                                              get_params=True)
    # add CWT padding if it doesn't exist  # TODO symmetric & other?
    if not rpadded:
        Wx = np.pad(Wx, [[0, 0], [n1, n2]])  # pad time axis, left=n1, right=n2
    else:
        n = x_len

    #### Invert ##############################################################
    if one_int:
        x = _icwt_1int(Wx, scales, ssq_freqs, l1_norm)
    else:
        x = _icwt_2int(Wx, scales, wavelet, N, ssq_freqs, l1_norm)

    # admissibility coefficient
    Cpsi = (adm_ssq(wavelet) if one_int else
            adm_cwt(wavelet))
    if ssq_freqs == 'log':
        # Eq 4.67 in [1]; Theorem 4.5 in [1]; below Eq 14 in [2]
        # ln(2**(1/nv)) == ln(2)/nv == diff(ln(scales))[0]
        x *= (2 / Cpsi) * np.log(2 ** (1 / nv))
    else:
        x *= (2 / Cpsi)

    x += x_mean       # CWT doesn't capture mean (infinite scale)
    x = x[n1:n1 + n]  # keep the unpadded part
    return x


def _icwt_2int(Wx, scales, wavelet, N, ssq_freqs, l1_norm):
    """Double-integral iCWT; works with any(?) wavelet."""
    norm = _icwt_norm(ssq_freqs, l1_norm, one_int=False)
    x = np.zeros(N)
    for a, Wxa in zip(scales, Wx):  # TODO vectorize?
        psih = wfilth(wavelet, N, a, l1_norm=l1_norm)
        xa = ifftshift(ifft(fft(Wxa) * psih)).real  # convolution theorem
        x += xa / norm(a)
    return x


def _icwt_1int(Wx, scales, ssq_freqs, l1_norm):
    """One-integral iCWT; assumes analytic wavelet."""
    norm = _icwt_norm(ssq_freqs, l1_norm, one_int=True)
    return (Wx.real / (norm(scales))).sum(axis=0)


def _icwt_norm(ssq_freqs, l1_norm, one_int):
    if l1_norm:
        norm = ((lambda a: 1) if ssq_freqs == 'log' else
                (lambda a: a))
    else:
        if ssq_freqs == 'log':
            norm = ((lambda a: a**.5) if one_int else
                    (lambda a: a))
        elif ssq_freqs == 'linear':
            norm = ((lambda a: a**1.5) if one_int else
                    (lambda a: a**2))
    return norm
