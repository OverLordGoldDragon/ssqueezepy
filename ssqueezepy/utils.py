# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
import numpy.matlib
from termcolor import colored
from quadpy import quad as quadgk
from .algos import replace_at_inf_or_nan
from .wavelets import Wavelet


WARN = colored('WARNING:', 'red')
NOTE = colored('NOTE:', 'blue')
pi = np.pi


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def est_riskshrink_thresh(Wx, nv):
    """Estimate the RiskShrink hard thresholding level.

    # Arguments:
        Wx:  np.ndarray. Wavelet transform of a signal.
        opt: dict. Options structure used for forward wavelet transform.

    # Returns:
        gamma: float. The RiskShrink hard thresholding estimate.
    """
    N = Wx.shape[1]
    Wx_fine = np.abs(Wx[:nv])
    gamma = 1.4826 * np.sqrt(2 * np.log(N)) * mad(Wx_fine)
    return gamma


def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.

    # Arguments:
        n: int. Non-dyadic integer.

    # Returns:
        up: next power of 2
        n1: length on left
        n2: length on right
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
        padtype: str ('symmetric', 'replicate').
        padlength: int. Number of samples to pad on each side. Default is
                   nearest power of 2.

    # Returns:
        x: padded signal.
        n_up: next power of 2.
        n1: length on left.
        n2: length on right.
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
    """Outputs the FFT of the wavelet of family and options in `wavelet`,
    of length N at scale a.

    Note that the output is made so that the inverse fft of the
    result is zero-centered in time.  This is important for
    convolving with the derivative(dpsih).  To get the correct
    output, perform an ifftshift.  That is,
        psi   = ifftshift(ifft(psih))
        xfilt = ifftshift(ifft(fft(x) * psih))

    Inputs:
        type: wavelet type (see help wfiltfn)
        N: number of samples to calculate
        a: wavelet scale parameter (default = 1)
        opt: wavelet options (see help wfiltfn)
          opt.dt: delta t (sampling period, default = 1)
                  important for properly scaling dpsih

    Outputs:
        psih: wavelet sampling in frequency domain (for use in fft)
        dpsih: derivative of same wavelet, sampled in frequency domain (for fft)
        xi: associated fourier domain frequencies of the samples.
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
        # discretized freq-domain derivative of trigonometric interpolant of psih
        # http://wavelets.ens.fr/ENSEIGNEMENT/COURS/UCSB/farge_ann_rev_1992.pdf
        dpsih = (1j * psihfn.xi / dt) * psih  # `dt` relevant for phase transform
        return psih, dpsih
    else:
        return psih


def adm_ssq(wavelet):
    """Calculate the synchrosqueezing admissibility constant, the term
    R_\psi in Eq. 3 of [1].

    Uses numerical intergration.

    # Arguments:
        wavelet: str. See `wfiltfn`.
        opts: dict. Options. See `wfiltfn`.

    # Returns:
        Css: integral(conj(wavelet_fn(w)) / w, w=0..inf)

    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.

        2. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms:
        an empricial mode decomposition-like tool",
        Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
    """
    psihfn = Wavelet(wavelet)
    Css = quadgk(lambda w: np.conj(psihfn(w)) / w, 0., np.inf)[0]
    return Css


def adm_cwt(wavelet):
    """Calculate cwt admissibility constant int(|f(w)|^2/w, w=0..inf) as
    per Eq. (4.67) of [1].

    1. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.
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
        - If `get_params`, also returns (`freqscale`, `nv`, `na`)
           - `freqscale`: inferred from `scales` if it's an array
           - `nv`, `na`: computed newly only if not already passed
    """
    def _infer_freqscale(scales):
        th = 1e-15 if scales.dtype == np.float64 else 2e-7
        if np.mean(np.abs(np.diff(scales, 2, axis=0))) < th:
            freqscale = 'linear'
        elif np.mean(np.abs(np.diff(np.log(scales), 2, axis=0))) < th:
            freqscale = 'log'
        else:
            raise ValueError("could not infer `freqscale` from `scales`; "
                             "`scales` array must be linear or logarithmic.")
        return freqscale

    def _process_args(scales, nv, na):
        if isinstance(scales, str):
            if scales not in ('log', 'linear'):
                raise ValueError("`scales`, if string, must be one of: log, "
                                 "linear (got %s)" % scales)
            elif (na is None and nv is None):
                raise ValueError("must pass one of `na`, `nv`, if `scales` "
                                 "isn't array")
            freqscale = scales
        elif isinstance(scales, np.ndarray):
            if scales.squeeze().ndim != 1:
                raise ValueError("`scales`, if array, must be 1D "
                                 "(got shape %s)" % scales.shape)

            freqscale = _infer_freqscale(scales)
            if na is not None:
                print(WARN, "`na` is ignored if `scales` is an array")
            na = len(scales)
        else:
            raise TypeError("`scales` must be a string or Numpy array "
                            "(got %s)" % type(scales))
        return freqscale, na

    freqscale, na = _process_args(scales, nv, na)

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
        if freqscale == 'log':
            scales = np.power(2 ** (1 / nv), np.arange(1, na + 1))
        elif freqscale == 'linear':
            scales = np.linspace(1, na, na)  # ??? should `1` be included?
    scales = scales.reshape(-1, 1)  # ensure 2D for mult / div later

    return (scales if not get_params else
            (scales, freqscale, na, nv))
