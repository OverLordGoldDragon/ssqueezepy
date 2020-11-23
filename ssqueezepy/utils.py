# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import logging
from quadpy import quad as quadgk
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.info("NOTE: %s" % msg)
pi = np.pi
EPS = np.finfo(np.float64).eps  # machine epsilon for float64  # TODO float32?



def mad(data, axis=None):
    """Mean absolute deviation"""
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def est_riskshrink_thresh(Wx, nv):
    """Estimate the RiskShrink hard thresholding level, based on [1].
    This has a denoising effect, but risks losing much of the signal; it's larger
    the more high-frequency content there is, even if not noise.

    # Arguments:
        Wx: np.ndarray
            CWT of a signal (see `cwt`).
        nv: int
            Number of voices used in CWT (see `cwt`).

    # Returns:
        gamma: float
            The RiskShrink hard thresholding estimate.

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        est_riskshrink_thresh.m
    """
    N = Wx.shape[1]
    Wx_fine = np.abs(Wx[:nv])
    gamma = 1.4826 * np.sqrt(2 * np.log(N)) * mad(Wx_fine)
    return gamma


def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int
            Length of original (unpadded) signal.

    # Returns:
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.
    """
    eps = np.finfo(np.float64).eps  # machine epsilon for float64
    up = 2 ** (1 + np.round(np.log2(n + eps)))

    n1 = np.floor((up - n) / 2)
    n2 = n1 + n % 2           # if n is odd, right-pad by (n1 + 1), else n2=n1
    assert n1 + n + n2 == up  # [left_pad, original, right_pad]
    return int(up), int(n1), int(n2)


def padsignal(x, padtype='symmetric', padlength=None):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray. Original signal.
        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'symmetric', 'replicate').
            'zero' is most naive, while 'symmetric' (default) partly mitigates
            boundary effects. See [1] & [2].
        padlength: int / None
            Number of samples to pad on each side. Default is for padded signal
            to have total length that's next power of 2.

    # Returns:
        xpad: np.ndarray
            Padded signal.
        n_up: int
            Next power of 2.
        n1: int
            Left  pad length.
        n2: int
            Right pad length.

    # References:
        1. Signal extension modes. PyWavelets contributors
        https://pywavelets.readthedocs.io/en/latest/ref/
        signal-extension-modes.html

        2. Wavelet Bases and Lifting Wavelets. H. Xiong.
        http://min.sjtu.edu.cn/files/wavelet/
        6-lifting%20wavelet%20and%20filterbank.pdf
    """
    padtypes = ('symmetric', 'replicate', 'zero')
    if padtype not in padtypes:
        raise ValueError(("Unsupported `padtype` {}; must be one of: {}"
                          ).format(padtype, ", ".join(padtypes)))
    n = len(x)

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(n)
    else:
        n_up = n + 2 * padlength
        n1 = padlength + 1  # TODO why n1 >= n2 here but n2 >= n1 in p2up
        n2 = padlength
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    # comments use (n=4, n1=3, n2=4) as example, but this combination can't occur
    if padtype == 'symmetric':
        # [1,2,3,4] -> [3,2,1, 1,2,3,4, 4,3,2,1]
        xpad = np.hstack([x[::-1][-n1:], x, x[::-1][:n2]])
    elif padtype == 'replicate':
        # [1,2,3,4] -> [1,1,1, 1,2,3,4, 4,4,4,4]
        x = np.pad(x, [n1, n2], mode='edge')
    elif padtype == 'zero':
        # [1,2,3,4] -> [0,0,0, 1,2,3,4, 0,0,0,0]
        x = np.pad(x, [n1, n2])

    assert len(xpad) == n_up == n1 + n + n2
    return xpad, n_up, n1, n2


# TODO default dt = 1/N? one sample per sec unlikely
def wfilth(wavelet, N, a=1, dt=1, derivative=False, l1_norm=True):
    """Computes the discretized (sampled) wavelets in Fourier frequency domain.
    Used in CWT for discretized convolution theorem via FFT.

    # Arguments:
        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.
        N: int
            Number of samples to calculate.
        a: float
            Wavelet scale parameter (default=1). Higher -> lower frequency.
        dt: float
            Sampling period (of input signal `x`), used to scale `dpsih`.
        derivative: bool (default False)
            Whether to compute and return derivative of same wavelet.
            Computed via frequency-domain differentiation (effectively,
            derivative of trigonometric interpolation; see [1]).
        l1_norm: bool (default True)
            Whether to L1-normalize the wvelet, which yields a CWT with more
            representative distribution of energies and component amplitudes
            than L2 (see [2]). If False (default True), uses L2 norm.

    # Returns:
        psih: np.ndarray
            Discretized (sampled) wavelets in Fourier frequency domain.
        dpsih: np.ndarray
            Derivative of same wavelet, used in CWT for computing `dWx`.

    # References:
        1. The Exponential Accuracy of Fourier and Chebyshev Differencing Methods.
        E. Tadmor.
        http://webhome.auburn.edu/~jzl0097/teaching/math_8970/Tadmor_86.pdf

        2. Rectification of the Bias in the Wavelet Power Spectrum.
        Y. Liu, X. S. Liang, R. H. Weisberg.
        http://ocg6.marine.usf.edu/~liu/Papers/Liu_etal_2007_JAOT_wavelet.pdf
    """
    if not np.log2(N).is_integer():
        raise ValueError(f"`N` must be a power of 2 (got {N})")

    psihfn = Wavelet(wavelet, N=N)

    # sample FT of wavelet at scale `a`, normalize energy
    # `* (-1)^[0,1,...]` = frequency-domain spectral reversal
    #                      to center time-domain wavelet
    norm = 1 if l1_norm else np.sqrt(a)
    psih = psihfn(a) * norm * (-1)**np.arange(N)

    # Sometimes bump gives a NaN when it means 0
    if 'bump' in wavelet:
        psih = replace_at_inf_or_nan(psih, 0)

    if derivative:
        dpsih = (1j * psihfn.xi / dt) * psih  # `dt` relevant for phase transform
        return psih, dpsih
    else:
        return psih


def adm_ssq(wavelet):
    """Calculates the synchrosqueezing admissibility constant, the term
    R_psi in Eq 15 of [1] (also see Eq 2.5 of [2]). Uses numerical intergration.

        integral(conj(psihfn(w)) / w, w=0..inf)

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf
    """
    psihfn = Wavelet(wavelet)
    Css = quadgk(lambda w: np.conj(psihfn(w)) / w, 0., np.inf)[0]
    return Css


def adm_cwt(wavelet):
    """Calculates the cwt admissibility constant as per Eq. (4.67) of [1].
    Uses numerical integration.

        integral(|psihfn(w)|^2/w, w=0..inf)

    1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
    https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
	"""
    wavelet = wavelet if isinstance(wavelet, tuple) else (wavelet, {})
    wavelet, opts = wavelet

    if wavelet == 'sombrero':
        s = opts.get('s', 1)
        Cpsi = (4/3) * s * np.sqrt(pi)
    elif wavelet == 'shannon':
        Cpsi = np.log(2)
    else:
        psihfn = Wavelet(wavelet)
        Cpsi = quadgk(lambda w: np.conj(psihfn(w)) * psihfn(w) / w,
                      0., np.inf)[0]
    return Cpsi


# TODO never reviewed
def buffer(x, n, p=0, opt=None):
    """Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    # Arguments:
        x: np.ndarray. Signal array.
        n: int. Number of data segments.
        p: int. Number of values to overlap
        opt: str.  Initial condition options. Default sets the first `p`
        values to zero, while 'nodelay' begins filling the buffer immediately.

    # Returns:
        result : (n,n) ndarray
            Buffer array created from x.

    # References:
        ryanjdillon: https://stackoverflow.com/a/40105995/10133797
    """
    if opt not in ('nodelay', None):
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    if opt == 'nodelay':
        # No zeros at array start
        result = x[:n]
        i = n
    else:
        # Start with `p` zeros
        result = np.hstack([np.zeros(p), x[:n-p]])
        i = n - p

    # Make 2D array
    result = np.expand_dims(result, axis=0)
    result = list(result)

    while i < len(x):
        # Create next column, add `p` results from last col if given
        col = x[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[-1][-p:], col])

        # Append zeros if last row and not length `n`
        if len(col):
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result.append(np.array(col))
        i += (n - p)

    return np.vstack(result).T


def _assert_positive_integer(g, name=''):
    if not (g > 0 and float(g).is_integer()):
        raise ValueError(f"'{name}' must be a positive integer (got {g})")


def process_scales(scales, len_x, nv=None, na=None, get_params=False):
    """Makes scales if `scales` is a string, else validates the array,
    and returns relevant parameters if requested.

        - Ensures, if array,  `scales` is 1D, or 2D with last dim == 1
        - Ensures, if string, `scales` is one of ('log', 'linear')
        - If `get_params`, also returns (`scaletype`, `nv`, `na`)
           - `scaletype`: inferred from `scales` ('linear' or 'log') if array
           - `nv`, `na`: computed newly only if not already passed
    """
    def _process_args(scales, nv, na):
        if isinstance(scales, str):
            if scales not in ('log', 'linear'):
                raise ValueError("`scales`, if string, must be one of: log, "
                                 "linear (got %s)" % scales)
            elif (na is None and nv is None):
                raise ValueError("must pass one of `na`, `nv`, if `scales` "
                                 "isn't array")
            scaletype = scales
        elif isinstance(scales, np.ndarray):
            if scales.squeeze().ndim != 1:
                raise ValueError("`scales`, if array, must be 1D "
                                 "(got shape %s)" % scales.shape)

            scaletype = _infer_scaletype(scales)
            if na is not None:
                print(WARN, "`na` is ignored if `scales` is an array")
            na = len(scales)
        else:
            raise TypeError("`scales` must be a string or Numpy array "
                            "(got %s)" % type(scales))
        return scaletype, na

    scaletype, na = _process_args(scales, nv, na)

    # compute params
    n_up, *_ = p2up(len_x)
    noct = np.log2(n_up) - 1
    if nv is None:
        nv = na / noct
    elif na is None:
        na = int(noct * nv)
    _assert_positive_integer(noct, 'noct')
    _assert_positive_integer(nv, 'nv')

    # make `scales` if passed string
    if isinstance(scales, str):
        if scaletype == 'log':
            scales = np.power(2 ** (1 / nv), np.arange(1, na + 1))
        elif scaletype == 'linear':
            scales = np.linspace(1, na, na)  # ??? should `1` be included?
    scales = scales.reshape(-1, 1)  # ensure 2D for mult / div later

    return (scales if not get_params else
            (scales, scaletype, na, nv))


def _infer_scaletype(ipt):
    """Infer whether `ipt` is linearly or exponentially distributed.
    Used internally on `scales` and `ssq_freqs`.
    """
    th = 1e-15 if ipt.dtype == np.float64 else 2e-7
    if np.mean(np.abs(np.diff(ipt, 2, axis=0))) < th:
        scaletype = 'linear'
    elif np.mean(np.abs(np.diff(np.log(ipt), 2, axis=0))) < th:
        scaletype = 'log'
    else:
        raise ValueError("could not infer `scaletype` from `ipt`; "
                         "`ipt` array must be linear or exponential.")
    return scaletype


def _process_fs_and_t(fs, t, N):
    """Ensures `t` is uniformly-spaced and of same length as `x` (==N)
    and returns `fs` and `dt` based on it, or from defaults if `t` is None.
    """
    if t is not None:
        if len(t) != N:
            # not explicitly used anywhere but ensures wrong `t` wasn't supplied
            raise Exception("`t` must be of same length as `x`.")
        elif not np.mean(np.abs(np.diff(t, 2, axis=0))) < 1e-7:  # float32 thr.
            raise Exception("Time vector `t` must be uniformly sampled.")
        fs = 1 / (t[1] - t[0])
    else:
        if fs is None:
            fs = 1
        elif fs <= 0:
            raise ValueError("`fs` must be > 0")
    dt = 1 / fs
    return dt, fs, t
