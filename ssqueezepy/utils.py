# -*- coding: utf-8 -*-
import numpy as np
import logging
from scipy import integrate
from .algos import replace_at_inf_or_nan, _min_neglect_idx
from .wavelets import Wavelet, _xi
from .viz_toolkit import _viz_cwt_scalebounds


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


def padsignal(x, padtype='reflect', padlength=None):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray. Original signal.
        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'symmetric', 'replicate').
            'zero' is most naive, while 'reflect' (default) partly mitigates
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
    # TODO @padlength: change to denote *total* pad length?
    padtypes = ('reflect', 'symmetric', 'replicate', 'zero')
    if padtype not in padtypes:
        raise ValueError(("Unsupported `padtype` {}; must be one of: {}"
                          ).format(padtype, ", ".join(padtypes)))
    n = len(x)

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(n)
    else:
        n_up = n + 2 * padlength
        n1 = n2 = padlength
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    # comments use (n=4, n1=3, n2=4) as example, but this combination can't occur
    if padtype == 'reflect':
        # [1,2,3,4] -> [4,3,2, 1,2,3,4, 3,2,1,2]
        xpad = np.pad(x, [n1, n2], mode='reflect')
    elif padtype == 'symmetric':
        # [1,2,3,4] -> [3,2,1, 1,2,3,4, 4,3,2,1]
        xpad = np.hstack([x[::-1][-n1:], x, x[::-1][:n2]])
    elif padtype == 'replicate':
        # [1,2,3,4] -> [1,1,1, 1,2,3,4, 4,4,4,4]
        xpad = np.pad(x, [n1, n2], mode='edge')
    elif padtype == 'zero':
        # [1,2,3,4] -> [0,0,0, 1,2,3,4, 0,0,0,0]
        xpad = np.pad(x, [n1, n2])

    _ = (len(xpad), n_up, n1, n, n2)
    assert (len(xpad) == n_up == n1 + n + n2), "%s ?= %s ?= %s + %s + %s" % _
    return xpad, n_up, n1, n2


def wfilth(wavelet, N, a=1, fs=1, derivative=False, l1_norm=True):
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

        fs: float
            Sampling frequency (of input signal `x`), used to scale `dpsih`.

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

    psihfn = Wavelet._init_if_not_isinstance(wavelet)

    # sample FT of wavelet at scale `a`, normalize energy
    # `* (-1)^[0,1,...]` = frequency-domain spectral reversal
    #                      to center time-domain wavelet
    norm = 1 if l1_norm else np.sqrt(a)
    psih = psihfn(scale=a) * norm * (-1)**np.arange(N)

    # Sometimes bump gives a NaN when it means 0
    if 'bump' in wavelet:
        psih = replace_at_inf_or_nan(psih, 0)

    if derivative:
        dpsih = (1j * psihfn.xi * fs) * psih  # `dt` relevant for phase transform
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
    psihfn = Wavelet._init_if_not_isinstance(wavelet).fn
    Css = _integrate_analytic(lambda w: np.conj(psihfn(w)) / w)
    Css = Css.real if abs(Css.imag) < 1e-15 else Css
    return Css


def adm_cwt(wavelet):
    """Calculates the cwt admissibility constant as per Eq. (4.67) of [1].
    Uses numerical integration.

        integral(psihfn(w) * conj(psihfn(w)) / w, w=0..inf)

    1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
    https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
	"""
    psihfn = Wavelet._init_if_not_isinstance(wavelet).fn
    Cpsi = _integrate_analytic(lambda w: np.conj(psihfn(w)) * psihfn(w) / w)
    Cpsi = Cpsi.real if abs(Cpsi.imag) < 1e-15 else Cpsi
    return Cpsi


def cwt_scalebounds(wavelet, N=None, cutoff=1, stdevs=1,
                    min_decay=1e5, max_mult=2, viz=False):
    """
    Lower  stdevs -> higher max_scale
    Higher cutoff -> lower  min_scale

    Approx linear time; benched (N, t) = (260000, 1.38s), (16400, 0.038s).
    Main compute expense is in finding max scale.
    """
    from .wavelets import time_resolution
    def _find_scale(param, name, cond_fn, psihfn, N, scale0, inc, **kw):
        scale = scale0 + inc  # TODO add cutoff
        while not cond_fn(psihfn, scale, N, param, **kw):
            # print("_find_scale at:", scale)
            scale += inc
            if scale < 0:
                raise ValueError("Failed to find `scale` for "
                                 "`%s`=%s" % (name, param))
        return scale

    def _meets_stdevs(psihfn, scale, N, stdevs, min_decay, max_mult):
        """Ensure `stdevs` number of standard deviations span the length of
        the signal (half of it = all of it, per unilateral std)
        """
        # TODO wtf
        # print('_meets_stdevs', scale, N, stdevs)
        # print(time_resolution(psihfn, scale, N, nondim=False) * stdevs,
        #       scale, N, stdevs)
        return time_resolution(psihfn, scale, N, nondim=False, max_mult=max_mult,
                               min_decay=min_decay) * stdevs > N / 2

    def _meets_cutoff_or_overshoots(psihfn, scale, N, cutoff):
        """'Overshoots' = less than cutoff at last index (cutoff > 0)
                        = more than cutoff at last index (cutoff < 0)

        'True' return will stop loop scale incrementing in _find_scale.
        """
        def _cutoff_idx(psihfn, mx_idx, mx, cutoff):
            if cutoff > 0:
                return np.where(psih[mx_idx:] < cutoff * mx)[0]
            else:
                return np.where(psih > abs(cutoff * mx))[0]

        # drop negative freqs
        psih = psihfn(_xi(scale=scale, N=N), nohalf=True)[:N//2 + 1]
        last_idx = len(psih) - 1
        mx, mx_idx = _find_peak(psihfn, scale, N, cutoff)

        #### DEBUG ##########################################################
        if 0:
            m = 32
            psih = psihfn(_xi(scale=scale, N=N), nohalf=True)[:N//2 + 1]
            last_idx = len(psih) - 1
            mx, mx_idx = _find_peak(psihfn, scale, N, cutoff)

            print(scale, cutoff, mx, mx_idx, last_idx, mx_idx > last_idx)
            M = m * N
            w1 = _xi(m, M)[:M//2 + 1]
            psih1 = psihfn(scale * w1, nohalf=True)

            plot(psih1)
            plot(psih, show=1)
            print(_cutoff_idx(psihfn, mx_idx, mx, cutoff))

            if input("woot etc") != 'y':
                1/0
        ######################################################################

        if mx_idx > last_idx:
            if cutoff > 0:
                # peak can't lie in extension
                return True
        elif cutoff < 0:
            print("EXECUTED")
            # peak must lie in extension; decrease scale to move peak right
            return True

        cut_idx = _cutoff_idx(psihfn, mx_idx, mx, cutoff)
        if abs(cutoff) == 1:
            idx = mx_idx

        elif cutoff > 0:
            # earliest index where psih is beneath cutoff and right of peak
            # (or above cutoff and left of peak, for cutoff < 0)
            if cut_idx.size == 0:
                # no match means we've overshot; return True to move peak right
                idx = last_idx
            else:
                idx = mx_idx + cut_idx[0]

        elif cutoff < 0:
            if cut_idx.size == 0:
                # no match means we've undershot; return False to
                # increase scale and move peak left
                idx = -1
            else:
                idx = cut_idx[-1]

        # cutoff met at edge
        return idx == last_idx

    # TODO mx isn't necessarily THE peak
    psihfn = Wavelet._init_if_not_isinstance(wavelet)
    N = N or psihfn.N

    if stdevs <= 0:
        raise ValueError("`stdevs` must be >0 (got %s)" % stdevs)
    # might never be exactly zero
    cutoff = max(abs(cutoff), 1e-18) * np.sign(cutoff)

    from .viz_toolkit import plot

    # start where freq-dom wavelet peak is likely left of Nyquist
    # start with large increments to save computation, then refine w/ small inc
    # do opposite if cutoff is negative
    scale0 = 6 if cutoff > 0 else 1.1
    for inc in (-1, -.1, -.01, -.001):
        if cutoff < 0:
            inc = -inc
        # plot(psihfn(scale=scale0, nohalf=True)[:N//2 + 1], show=1)
        # print("scale0:", scale0, "cutoff", cutoff)
        min_scale = _find_scale(cutoff, 'cutoff', _meets_cutoff_or_overshoots,
                                psihfn, N, scale0, inc)
        # decrement approximation and repeat to refine
        scale0 = min_scale - inc
        # plot(psihfn(scale=min_scale, nohalf=True)[:N//2 + 1], show=1)
        # print("min_scale", min_scale)

    # lowballing Morlet max_scale at N=2048
    scale0 = (400 / 2048) * N
    # won't need longer arange per large integer exp steps (2^(5...), 2^(6...))
    incs = (scale0 / 5) * np.power(5., -np.arange(4))
    for inc in incs:
        max_scale = _find_scale(stdevs, 'stdevs', _meets_stdevs, psihfn, N,
                                scale0, inc, min_decay=min_decay,
                                max_mult=max_mult)
        scale0 = max_scale - inc

    if viz:
        _viz_cwt_scalebounds(psihfn, N, min_scale, max_scale,
                             stdevs=stdevs, cutoff=cutoff)
    return min_scale, max_scale


def _find_peak(psihfn, scale, N, cutoff):
    """Helper method for cwt_scalebounds and viz_cwt_scalebounds.
    Will extend psihfn from selected scale beyond Nyquist (pi) to find
    true peak value, to then take its cutoff fraction.
    """
    mults = [4, 8, 16, 32]
    if cutoff < 0:
        # desired to have peak in extension
        mults = mults[::-1]

    for m in mults:
        M = m*N
        wm = _xi(m, M)[:M//2 + 1]  # drop negative freqs
        psih = psihfn(scale * wm, nohalf=True)

        mx = psih.max()
        if psih[-1] < mx:
            # if there's a point toward right that's smaller, peak is found
            mx_idx = np.argmax(psih)
            break
    else:
        raise Exception(("Couldn't find peak after extending 32-fold for "
                         "scale={}, N={}; `scale` might be too low").format(
                             scale, N))
    return mx, mx_idx


def buffer(x, seg_len, n_overlap):
    # TODO docstr
    hop_len = seg_len - n_overlap
    n_segs = (len(x) - seg_len) // hop_len + 1
    out = np.zeros((seg_len, n_segs))

    for i in range(n_segs):
        start = i * hop_len
        end   = start + seg_len
        out[:, i] = x[start:end]
    return out


def _assert_positive_integer(g, name=''):
    if not (g > 0 and float(g).is_integer()):
        raise ValueError(f"'{name}' must be a positive integer (got {g})")


def process_scales(scales, len_x, wavelet=None, nv=None, get_params=False,
                   minbounds=False):
    """Makes scales if `scales` is a string, else validates the array,
    and returns relevant parameters if requested.

        - Ensures, if array,  `scales` is 1D, or 2D with last dim == 1
        - Ensures, if string, `scales` is one of ('log', 'linear')
        - If `get_params`, also returns (`scaletype`, `nv`, `na`)
           - `scaletype`: inferred from `scales` ('linear' or 'log') if array
           - `nv`, `na`: computed newly only if not already passed

    `minbounds=True` will mimic MATLAB's setting of min and max CWT `scale`,
    min set where time-domain wavelet's one standard deviation spans the N-point
    signal, and max set such that freq-domain wavelet peaks at Nyquist. These
    differ a bit with MATLAB's thresholding, favoring more scales
    (https://www.mathworks.com/help/wavelet/ref/cwtfreqbounds.html)
    """
    def _process_args(scales, nv, wavelet):
        if isinstance(scales, str):
            if scales not in ('log', 'linear'):
                raise ValueError("`scales`, if string, must be one of: log, "
                                 "linear (got %s)" % scales)
            if nv is None and scales == 'log':
                raise ValueError("must set `nv` if `scales`=='log'")
            if wavelet is None:
                raise ValueError("must set `wavelet` if `scales` isn't array")
            scaletype = scales

        elif isinstance(scales, np.ndarray):
            if scales.squeeze().ndim != 1:
                raise ValueError("`scales`, if array, must be 1D "
                                 "(got shape %s)" % scales.shape)
            scaletype, _nv = _infer_scaletype(scales)
            if scaletype == 'log':
                if nv is not None and _nv != nv:
                    raise Exception("`nv` used in `scales` differs from "
                                    "`nv` passed (%s != %s)" % (_nv, nv))
                nv = _nv
            scales = scales.reshape(-1, 1)  # ensure 2D for broadcast ops later

        else:
            raise TypeError("`scales` must be a string or Numpy array "
                            "(got %s)" % type(scales))

        if nv is not None:
            _assert_positive_integer(nv, 'nv')
            nv = int(nv)
        return scaletype, nv

    scaletype, nv = _process_args(scales, nv, wavelet)
    if isinstance(scales, np.ndarray):
        scales = scales.reshape(-1, 1)
        return (scales if not get_params else
                (scales, scaletype, len(scales), nv))

    #### Compute scales & params #############################################
    if not minbounds:
        nup, *_ = p2up(len_x)
        noct = np.log2(nup) - 1
        _assert_positive_integer(noct, 'noct')
        na = int(noct * nv)
        mn_pow, mx_pow = 1, na + 1

    else:
        min_scale, max_scale = cwt_scalebounds(wavelet, len_x)

        # number of 2^-distributed scales spanning min to max
        na = int(np.ceil(nv * np.log2(max_scale / min_scale)))
        # floor to keep freq-domain peak at or to right of Nyquist
        # min must be more precise, if need integer rounding do on max
        mn_pow = int(np.floor(nv * np.log2(min_scale)))
        mx_pow = mn_pow + na

    if scaletype == 'log':
        scales = 2 ** (np.arange(mn_pow, mx_pow) / nv)
    elif scaletype == 'linear':
        # TODO poor scheme
        min_scale, max_scale = 2**(mn_pow/nv), 2**(mx_pow/nv)
        na = int(np.ceil(2 * max_scale / min_scale))
        scales = np.linspace(min_scale, max_scale, na)

    scales = scales.reshape(-1, 1)  # ensure 2D for broadcast ops later
    return (scales if not get_params else
            (scales, scaletype, na, nv))


def _infer_scaletype(ipt):
    """Infer whether `ipt` is linearly or exponentially distributed (if latter,
    also infers `nv`). Used internally on `scales` and `ssq_freqs`.
    """
    if not isinstance(ipt, np.ndarray):
        raise TypeError("`ipt` must be a numpy array (got %s)" % type(ipt))
    elif ipt.dtype not in (np.float32, np.float64):
        raise TypeError("`ipt.dtype` must be np.float32 or np.float64 "
                        "(got %s)" % ipt.dtype)

    th_log = 1e-15 if ipt.dtype == np.float64 else 4e-7
    th_lin = th_log * 100  # less accurate for some reason

    if np.mean(np.abs(np.diff(ipt, 2, axis=0))) < th_lin:
        scaletype = 'linear'
        nv = None
    elif np.mean(np.abs(np.diff(np.log(ipt), 2, axis=0))) < th_log:
        scaletype = 'log'
        # ceil to avoid faulty float-int ROUNDOFFS
        nv = int(np.round(1 / np.diff(np.log2(ipt), axis=0)[0]))
    else:
        raise ValueError("could not infer `scaletype` from `ipt`; "
                         "`ipt` array must be linear or exponential.")
    return scaletype, nv


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


def _integrate_analytic(int_fn, nowarn=False):
    """Assumes function that's zero for negative inputs (e.g. analytic wavelet),
    decays toward right, and is unimodal: int_fn(t<0)=0, int_fn(t->inf)->0.
    Integrates using trapezoidal rule, from 0 to inf (equivalently).

    Integrates near zero separately in log space (useful for e.g. 1/x).
    """
    def _est_arr(mxlim, N):
        t = np.linspace(mxlim, .1, N, endpoint=False)[::-1]
        arr = int_fn(t)

        max_idx = np.argmax(arr)
        min_neglect_idx = _min_neglect_idx(np.abs(arr[max_idx:]),
                                           th=1e-15) + max_idx
        return arr, t, min_neglect_idx

    def _find_convergent_array():
        mxlims = [1, 20, 80, 160]
        for m, mxlim in zip([1, 1, 4, 8], mxlims):
            arr, t, min_neglect_idx = _est_arr(mxlim, N=10000*m)
            # ensure sufficient decay between peak and right endpoint, and
            # that `arr` isn't a flatline (contains wavelet peak)
            if ((len(t) - min_neglect_idx > 1000 * m) and
                np.sum(np.abs(arr)) > 1e-5):
                break
        else:
            if int_nz < 1e-5:
                raise Exception("Could not find converging or non-negligibly"
                                "-valued bounds of integration for `int_fn`")
            elif not nowarn:
                WARN("Integrated only from 1e-15 to 0.1 in logspace")
        return arr[:min_neglect_idx], t[:min_neglect_idx]

    def _integrate_near_zero():
        # sample `intfn` more finely as it might be extremely narrow near zero.
        # this still doesn't work well as float64 zeros the numerator before /w,
        # but the true integral will be negligibly small most of the time anyway
        # (.001 to .1 may not be negligible, however; better captured by logspace)
        t = np.logspace(-15, -1, 1000)
        arr = int_fn(t)
        return integrate.trapz(arr, t)

    int_nz = _integrate_near_zero()
    arr, t = _find_convergent_array()
    return integrate.trapz(arr, t) + int_nz


def _integrate_bounded(int_fn, center=0, get_data=False):
    """Assumes function is bounded on left and right (i.e. decays to zero).

    Currently unused.
    """
    def _est_arr(mxlim, N):
        t = np.linspace(center - mxlim, center + mxlim, N)
        arr = int_fn(t)
        cidx = N // 2

        left_idx  = cidx - _min_neglect_idx(np.abs(arr[:cidx][::-1]), th=1e-15)
        right_idx = cidx + _min_neglect_idx(np.abs(arr[cidx:]), th=1e-15)
        return arr, t, cidx, left_idx, right_idx

    def _find_convergent_array():
        for i in (1, 4, 8):
            arr, t, cidx, left_idx, right_idx = _est_arr(mxlim=20*i, N=10000*i)
            # ensure sufficient decay between center and endpoints, and
            # that `arr` isn't a flatline via normalized stdev
            arr_nstd = np.std(np.abs(arr) / np.max(np.abs(arr)))
            if ((len(t) - right_idx > 1000 * i) and
                (left_idx - 0       > 1000 * i) and
                arr_nstd > .1):
                break
        else:
            raise Exception("Could not force function to converge, or to find "
                            + ("non-(almost)zero values (nstd={:2e}, left_idx={},"
                               " right_idx={}, center_idx={}").format(
                                   arr_nstd, left_idx, right_idx, cidx))
        return arr[left_idx:right_idx], t[left_idx:right_idx]

    arr, t = _find_convergent_array()
    _int = integrate.trapz(arr, t)
    if get_data:
        return _int, arr, t
    return _int
