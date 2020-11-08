import numpy as np
from numpy.fft import fft, ifft, ifftshift
from .utils import WARN, p2up, adm_cwt, adm_ssq, wfilth
from .utils import padsignal, process_scales
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


def cwt(x, wavelet, scales='log', dt=1, nv=None, l1_norm=True,
        padtype='symmetric', rpadded=False):
    """Forward continuous wavelet transform, discretized, as described in
    Sec. 4.3.3 of [1] and Sec. IIIA for [2]. This algorithm uses the FFT and
    samples the wavelet atoms in the Fourier domain. Options such as padding
    of the original signal are allowed. Returns the vector of scales and, if
    requested, the analytic time-derivative of the wavelet transform (as
    described in Sec. IIIB of [2]).

    # Arguments:
        x: np.ndarray. Input signal vector, length `n` (need not be dyadic).
        wavelet_type: str. See `wfiltfn`.
        scales: CWT scales. np.ndarray or ('log', 'linear')
                !!! beware of scales='linear'; bad current default scheme for
                capturing low frequencies for sequences longer than 2048.
        nv: int. Number of voices. Suggested >= 32.
        dt: int. sampling period.
        opts: dict. Options:
            'padtype': ('symmetric', 'replicate', 'circular'). Type of padding.
                       (default = 'symmetric')
            'rpadded': bool. Whether to return padded Wx and dWx.
                       (default = False)
            'type', 's', 'mu', ...: str. Wavelet options (see `wfiltfn`).

    # Returns:
        Wx: (na x n) size matrix (rows = scales, cols = times), containing
            samples of the CWT of `x`.
        scales: `na` length vector containing the associated scales.
        dWx: (na x n) size matrix containing samples of the time-derivatives
              of the CWT of `x`.
        x_mean: mean of `x` to use in inversion (CWT needs scale=inf to capture)

    # References:
        1. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.

        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.
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
    """The inverse continuous wavelet transform of signal Wx via double integral.
    Implements Eq. (4.67) of [1].

    Inputs:
       Wx: wavelet transform of a signal, see help cwt_fw
       type: wavelet used to take the wavelet transform,
             see help cwt_fw and help wfiltfn
       opt: options structure used for forward wavelet transform.
       len_x: length of original x, before padding
       x_mean: mean of original x (not picked up in CWT since it's an
                                  infinite scale component)

    Output:
     x: the signal, as reconstructed from Wx

    Explained in: https://dsp.stackexchange.com/a/71148/50076

    References:
        1. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.

        2. G. Thakur, E. Brevdo, N.-S. Fuckar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
         robustness properties and new paleoclimate applications",
         Signal Processing, 93:1079-1094, 2013.
    """
    #### Prepare for inversion ###############################################
    na, n = Wx.shape
    x_len = x_len or n
    N, n1, n2 = p2up(x_len)

    scales, freqscale, _, nv = process_scales(scales, x_len, na=na,
                                              get_params=True)
    # add CWT padding if it doesn't exist  # TODO symmetric & other?
    if not rpadded:
        Wx = np.pad(Wx, [[0, 0], [n1, n2]])  # pad time axis, left=n1, right=n2
    else:
        n = x_len

    #### Invert ##############################################################
    if one_int:
        x = _icwt_1int(Wx, scales, freqscale, l1_norm)
    else:
        x = _icwt_2int(Wx, scales, wavelet, N, freqscale, l1_norm)

    # admissibility coefficient
    Cpsi = (adm_ssq(wavelet) if one_int else
            adm_cwt(wavelet))
    if freqscale == 'log':
        # Eq 4.67 in [1]; Theorem 4.5 in [1]; below Eq 14 in [2]
        # ln(2**(1/nv)) == ln(2)/nv == diff(ln(scales))[0]
        x *= (2 / Cpsi) * np.log(2 ** (1 / nv))
    else:
        x *= (2 / Cpsi)

    x += x_mean       # CWT doesn't capture mean (infinite scale)
    x = x[n1:n1 + n]  # keep the unpadded part
    return x


def _icwt_2int(Wx, scales, wavelet, N, freqscale, l1_norm):
    """Double-integral iCWT; works with any(?) wavelet.
    Explanation: https://dsp.stackexchange.com/a/71148/50076
    """
    norm = _icwt_norm(freqscale, l1_norm, one_int=False)
    x = np.zeros(N)
    for a, Wxa in zip(scales, Wx):  # TODO vectorize?
        psih = wfilth(wavelet, N, a, l1_norm=l1_norm)
        xa = ifftshift(ifft(fft(Wxa) * psih)).real  # convolution theorem
        x += xa / norm(a)
    return x


def _icwt_1int(Wx, scales, freqscale, l1_norm):
    """One-integral iCWT; assumes analytic wavelet.
    Explanation: https://dsp.stackexchange.com/a/71274/50076
    """
    norm = _icwt_norm(freqscale, l1_norm, one_int=True)
    return (Wx.real / (norm(scales))).sum(axis=0)


def _icwt_norm(freqscale, l1_norm, one_int):
    if l1_norm:
        norm = ((lambda a: 1) if freqscale == 'log' else
                (lambda a: a))
    else:
        if freqscale == 'log':
            norm = ((lambda a: a**.5) if one_int else
                    (lambda a: a))
        elif freqscale == 'linear':
            norm = ((lambda a: a**1.5) if one_int else
                    (lambda a: a**2))
    return norm
