# -*- coding: utf-8 -*-
import numpy as np
import logging
from numpy.fft import fft, fftshift
from scipy import integrate
from numba import jit
from textwrap import wrap
from .algos import _min_neglect_idx, find_maximum, find_first_occurrence
from .wavelets import Wavelet

logging.basicConfig(format='')
WARN = lambda msg: logging.warning("WARNING: %s" % msg)
NOTE = lambda msg: logging.warning("NOTE: %s" % msg)  # else it's mostly ignored
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

    n2 = np.floor((up - n) / 2)
    n1 = n2 + n % 2           # if n is odd, left-pad by (n2 + 1), else n1=n2
    assert n1 + n + n2 == up  # [left_pad, original, right_pad]
    return int(up), int(n1), int(n2)


def padsignal(x, padtype='reflect', padlength=None, get_params=False):
    """Pads signal and returns trim indices to recover original.

    # Arguments:
        x: np.ndarray
            Input vector, 1D or 2D. 2D has time in dim1, e.g. `(n_signals, time)`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See [1] & [2].

        padlength: int / None
            Number of samples to pad input to (i.e. len(x_padded) == padlength).
            Even: left = right, Odd: left = right + 1.
            Defaults to next highest power of 2 w.r.t. `len(x)`.

    # Returns:
        xp: np.ndarray
            Padded signal.
        n_up: int
            Next power of 2, or `padlength` if provided.
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
    def _process_args(x, padtype):
        # TODO: padlength -> padded_len ?
        supported = ('reflect', 'symmetric', 'replicate', 'wrap', 'zero')
        if padtype not in supported:
            raise ValueError(("Unsupported `padtype` {}; must be one of: {}"
                              ).format(padtype, ", ".join(supported)))
        if not isinstance(x, np.ndarray):
            raise TypeError("`x` must be a numpy array (got %s)" % type(x))
        elif x.ndim not in (1, 2):
            raise ValueError("`x` must be 1D or 2D (got x.ndim == %s)" % x.ndim)

    _process_args(x, padtype)
    N = x.shape[-1]

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(N)
    else:
        n_up = padlength
        if abs(padlength - N) % 2 == 0:
            n1 = n2 = (n_up - N) // 2
        else:
            n2 = (n_up - N) // 2
            n1 = n2 + 1
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    if x.ndim == 1:
        pad_width = (n1, n2)
    elif x.ndim == 2:
        pad_width = [(0, 0), (n1, n2)]

    # comments use (n=4, n1=4, n2=3) as example, but this combination can't occur
    if padtype == 'zero':
        # [1,2,3,4] -> [0,0,0,0, 1,2,3,4, 0,0,0]
        xp = np.pad(x, pad_width)
    elif padtype == 'reflect':
        # [1,2,3,4] -> [3,4,3,2, 1,2,3,4, 3,2,1]
        xp = np.pad(x, pad_width, mode='reflect')
    elif padtype == 'replicate':
        # [1,2,3,4] -> [1,1,1,1, 1,2,3,4, 4,4,4]
        xp = np.pad(x, pad_width, mode='edge')
    elif padtype == 'wrap':
        # [1,2,3,4] -> [1,2,3,4, 1,2,3,4, 1,2,3]
        xp = np.pad(x, pad_width, mode='wrap')
    elif padtype == 'symmetric':
        # [1,2,3,4] -> [4,3,2,1, 1,2,3,4, 4,3,2]
        if x.ndim == 1:
            xp = np.hstack([x[::-1][-n1:], x, x[::-1][:n2]])
        elif x.ndim == 2:
            xp = np.hstack([x[:, ::-1][:, -n1:], x, x[:, ::-1][:, :n2]])

    Npad = xp.shape[-1]
    _ = (Npad, n_up, n1, N, n2)
    assert (Npad == n_up == n1 + N + n2), "%s ?= %s ?= %s + %s + %s" % _
    return (xp, n_up, n1, n2) if get_params else xp


#### CWT utils ################################################################
def adm_ssq(wavelet):
    """Calculates the synchrosqueezing admissibility constant, the term
    R_psi in Eq 15 of [1] (also see Eq 2.5 of [2]). Uses numeric intergration.

        integral(conj(wavelet(w)) / w, w=0..inf)

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet).fn
    Css = _integrate_analytic(lambda w: np.conj(wavelet(w)) / w)
    Css = Css.real if abs(Css.imag) < 1e-15 else Css
    return Css


def adm_cwt(wavelet):
    """Calculates the cwt admissibility constant as per Eq. (4.67) of [1].
    Uses numeric integration.

        integral(wavelet(w) * conj(wavelet(w)) / w, w=0..inf)

    1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
    https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet).fn
    Cpsi = _integrate_analytic(lambda w: np.conj(wavelet(w)) * wavelet(w) / w)
    Cpsi = Cpsi.real if abs(Cpsi.imag) < 1e-15 else Cpsi
    return Cpsi


def find_min_scale(wavelet, cutoff=1):
    """
    Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that its last (largest) positive value evaluates
    `wavelet` to `cutoff * max(psih)`. If cutoff > 0, it lands to right of peak,
    else to left (i.e. peak excluded).
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    w_peak, peak = find_maximum(wavelet.fn)
    if cutoff > 0:
        # search to right of peak
        step_start, step_limit = w_peak, 10*w_peak
    else:
        # search to left of peak
        step_start, step_limit = 0, w_peak

    w_cutoff, _ = find_first_occurrence(wavelet.fn, value=abs(cutoff) * peak,
                                        step_start=step_start,
                                        step_limit=step_limit)
    min_scale = w_cutoff / np.pi
    return min_scale


def find_max_scale(wavelet, N, min_cutoff=.1, max_cutoff=.8):
    """
    Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that two of its consecutive values land
    symmetrically about the peak of `psih` (i.e. none *at* peak), while
    still yielding `wavelet(w)` to fall between `min_cutoff`* and `max_cutoff`*
    `max(psih)`. `scale` is selected such that the symmetry is attained
    using smallest possible bins (closest to dc). Steps:

        1. Find `w` (input value to `wavelet`) for which `wavelet` is maximized
        (i.e. peak of `psih`).
        2. Find two `w` such that `wavelet` attains `min_cutoff` and `max_cutoff`
        times its maximum value, using `w` in previous step as upper bound.
        3. Find `div_size` such that `xi` lands at both points of symmetry;
        `div_size` == increment between successive values of
        `xi = scale * xi(scale=1)`.
            - `xi` begins at zero; along the cutoff bounds, and us selecting
            the smallest number of divisions/increments to reach points of
            symmetry, we guarantee a unique `scale`.

    This yields a max `scale` that'll generally lie in 'nicely-behaved' region
    of std_t; value can be used to fine-tune further.
    See `visuals.sweep_std_t`.
    """
    if max_cutoff <= 0 or min_cutoff <= 0:
        raise ValueError("`max_cutoff` and `min_cutoff` must be positive "
                         "(got %s, %s)" % (max_cutoff, min_cutoff))
    elif max_cutoff <= min_cutoff:
        raise ValueError("must have `max_cutoff > min_cutoff` "
                         "(got %s, %s)" % (max_cutoff, min_cutoff))

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    w_peak, peak = find_maximum(wavelet.fn)

    # we solve the inverse problem; instead of looking for spacing of xi
    # that'd land symmetrically about psih's peak, we pick such points
    # above a set ratio of peak's value and ensure they divide the line
    # from left symmetry point to zero an integer number of times

    # define all points of wavelet from cutoff to peak, left half
    w_cutoff, _ = find_first_occurrence(wavelet.fn, value=min_cutoff * peak,
                                        step_start=0, step_limit=w_peak)

    w_ltp = np.arange(w_cutoff, w_peak, step=1/N)  # left-to-peak

    # consider every point on wavelet(w_ltp) (except peak) as candidate cutoff
    # point, and pick earliest one that yields integer number of increments
    # from left point of symmetry to origin
    div_size = (w_peak - w_ltp[:-1]) * 2  # doubled so peak is skipped
    n_divs = w_ltp[:-1] / div_size
    # diff of modulus; first drop in n_divs is like [.98, .99, 0, .01], so at 0
    # we've hit an integer, and n_divs grows ~linearly so behavior guaranteed
    # -.8 arbitrary to be ~1 but <1
    try:
        idx = np.where(np.diff(n_divs % 1) < -.8)[0][0]
    except:
        raise Exception("Failed to find suffciently-integer xi divisions; try "
                        "widening (min_cutoff, max_cutoff)")
    # the div to base the scale on (angular bin spacing of scale*xi)
    div_scale = div_size[idx + 1]

    # div size of scale=1 (spacing between angular bins at scale=1)
    w_1div = np.pi / (N / 2)

    max_scale = div_scale / w_1div
    return max_scale


def cwt_scalebounds(wavelet, N, preset=None, min_cutoff=None, max_cutoff=None,
                    cutoff=None, double_N=True, viz=False):
    """`min_cutoff, max_cutoff` used to find max scale, `cutoff` to find min.
    viz==2 for more visuals, ==3 for even more.

      - Lesser  `cutoff`     -> lesser `min_scale`, always
      - Greater `min_cutoff` -> lesser `max_scale`, generally

    # Arguments:
        wavelet: `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain. See help(cwt).

        N: int
            Length of wavelet to use.

        min_cutoff, max_cutoff: float > 0 / None
            Used to find max scale. See help(utils.find_max_scale)

        cutoff: float / None
            Used to find min scale. See help(utils.find_min_scale)

        preset: str['maximal', 'minimal', 'naive'] / None
            - 'maximal': yields a larger max and smaller min.
            - 'minimal': strives to keep wavelet in "well-behaved" range of std_t
            and std_w, but very high or very low frequencies' energies will be
            under-represented. Is closer to MATLAB's default `cwtfreqbounds`.
            - 'minimal-low': `'minimal'`s lower bound, `'maximal'`'s upper
            - 'naive': returns (1, N), which is per original MATLAB Toolbox,
            but a poor choice for most wavelet options.
            - None: will use `min_cutoff, max_cutoff, cutoff` values, else
            override all of them. If it and they are all None, will default to
            `preset='minimal-low'`.

            preset: (min_cutoff, max_cutoff, cutoff)
                - 'maximal':     0.001, 0.8, -0.5
                - 'minimal':     0.6,   0.8, 1
                - 'minimal-low': 0.6,   0.8, -0.5

        double_N: bool (default True)
            Whether to use 2*N in computations. Typically `N == len(x)`,
            but CWT pads to 2*N, which is the actual wavelet length used,
            which typically behaves significantly differently at scale extrema,
            thus recommended default True. Differs from passing N=2*N and False
            only for first visual if `viz`, see code.

    """
    def _process_args(preset, min_cutoff, max_cutoff, cutoff):
        defaults = {'minimal': dict(min_cutoff=.6,   max_cutoff=.8, cutoff=1),
                    'maximal': dict(min_cutoff=1e-5, max_cutoff=.8, cutoff=-.5)}
        defaults['minimal-low'] = {k: defaults[d][k] for d, k in
                                   [('minimal', 'min_cutoff'),
                                    ('minimal', 'max_cutoff'),
                                    ('maximal', 'cutoff')]}
        none_default = 'minimal-low'

        if preset is not None:
            if any((min_cutoff, max_cutoff, cutoff)):
                WARN("`preset` will override `min_cutoff, max_cutoff, cutoff`")

            supported = ('maximal', 'minimal', 'minimal-low', 'naive')
            if preset not in supported:
                raise ValueError("`preset` must be one of: {} (got {})".format(
                    ', '.join(supported), preset))
            elif preset == 'naive':
                pass  # handle later
            else:
                min_cutoff, max_cutoff, cutoff = defaults[preset].values()
        else:
            if min_cutoff is None:
                min_cutoff = defaults[none_default]['min_cutoff']
            elif min_cutoff <= 0:
                raise ValueError("`min_cutoff` must be >0 (got %s)" % min_cutoff)

            if max_cutoff is None:
                max_cutoff = defaults[none_default]['max_cutoff']
            elif max_cutoff < min_cutoff:
                raise ValueError("must have `max_cutoff > min_cutoff` "
                                 "(got %s, %s)" % (max_cutoff, min_cutoff))

            if cutoff is None:
                cutoff = defaults[none_default]['cutoff']
            elif cutoff == 0:
                NOTE("`cutoff==0` might never be attained; setting to 1e-14")
                cutoff = 1e-14

        return min_cutoff, max_cutoff, cutoff

    def _viz():
        from .visuals import _viz_cwt_scalebounds, wavelet_waveforms

        _viz_cwt_scalebounds(wavelet, N, min_scale=min_scale,
                             max_scale=max_scale, cutoff=cutoff)
        if viz >= 2:
            wavelet_waveforms(wavelet, M, min_scale)
            wavelet_waveforms(wavelet, M, max_scale)
        if viz == 3:
            from .visuals import sweep_harea
            scales = make_scales(M, min_scale, max_scale)
            sweep_harea(wavelet, M, scales)

    min_cutoff, max_cutoff, cutoff = _process_args(preset, min_cutoff,
                                                   max_cutoff, cutoff)
    if preset == 'naive':  # still _process_args for the NOTE
        return 1, N

    M = 2*N if double_N else N
    min_scale = find_min_scale(wavelet, cutoff=cutoff)
    max_scale = find_max_scale(wavelet, M, min_cutoff=min_cutoff,
                               max_cutoff=max_cutoff)
    if viz:
        _viz()
    return min_scale, max_scale


def _assert_positive_integer(g, name=''):
    if not (g > 0 and float(g).is_integer()):
        raise ValueError(f"'{name}' must be a positive integer (got {g})")


def process_scales(scales, len_x, wavelet=None, nv=None, get_params=False,
                   double_N=True):
    """Makes scales if `scales` is a string, else validates the array,
    and returns relevant parameters if requested.

        - Ensures, if array,  `scales` is 1D, or 2D with last dim == 1
        - Ensures, if string, `scales` is one of ('log', 'linear')
        - If `get_params`, also returns (`scaletype`, `nv`, `na`)
           - `scaletype`: inferred from `scales` ('linear' or 'log') if array
           - `nv`, `na`: computed newly only if not already passed
    """
    def _process_args(scales, nv, wavelet):
        preset = None
        if isinstance(scales, str):
            if ':' in scales:
                scales, preset = scales.split(':')
            if scales not in ('log', 'linear'):
                raise ValueError("`scales`, if string, must be one of: log, "
                                 "linear (got %s)" % scales)
            if nv is None:
                nv = 32
            if wavelet is None:
                raise ValueError("must set `wavelet` if `scales` isn't array")
            scaletype = scales

        elif isinstance(scales, np.ndarray):
            if scales.squeeze().ndim != 1:
                raise ValueError("`scales`, if array, must be 1D "
                                 "(got shape %s)" % str(scales.shape))
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
        return scaletype, nv, preset

    scaletype, nv, preset = _process_args(scales, nv, wavelet)
    if isinstance(scales, np.ndarray):
        scales = scales.reshape(-1, 1)
        return (scales if not get_params else
                (scales, scaletype, len(scales), nv))

    #### Compute scales & params #############################################
    min_scale, max_scale = cwt_scalebounds(wavelet, N=len_x, preset=preset,
                                           double_N=double_N)
    scales = make_scales(len_x, min_scale, max_scale, nv=nv, scaletype=scaletype)
    na = len(scales)

    return (scales if not get_params else
            (scales, scaletype, na, nv))


def _infer_scaletype(ipt):
    """Infer whether `ipt` is linearly or exponentially distributed (if latter,
    also infers `nv`). Used internally on `scales` and `ssq_freqs`.

    Returns one of: 'linear', 'log', 'log-piecewise'
    """
    if not isinstance(ipt, np.ndarray):
        raise TypeError("`ipt` must be a numpy array (got %s)" % type(ipt))
    elif ipt.dtype not in (np.float32, np.float64):
        raise TypeError("`ipt.dtype` must be np.float32 or np.float64 "
                        "(got %s)" % ipt.dtype)

    th_log = 1e-15 if ipt.dtype == np.float64 else 4e-7
    th_lin = th_log * 1e3  # less accurate for some reason

    if np.mean(np.abs(np.diff(ipt, 2, axis=0))) < th_lin:
        scaletype = 'linear'
        nv = None
    elif np.mean(np.abs(np.diff(np.log(ipt), 2, axis=0))) < th_log:
        scaletype = 'log'
        # ceil to avoid faulty float-int ROUNDOFFS
        nv = int(np.round(1 / np.diff(np.log2(ipt), axis=0)[0]))
    elif logscale_transition_idx(ipt) is None:
        raise ValueError("could not infer `scaletype` from `ipt`; "
                         "`ipt` array must be linear or exponential. "
                         "(got diff(ipt)=%s..." % np.diff(ipt, axis=0)[:4])
    else:
        scaletype = 'log-piecewise'
        nv = None
    return scaletype, nv


def logscale_transition_idx(scales):
    scales_diff2 = np.diff(np.log(scales), 2, axis=0)
    idx = np.argmax(scales_diff2) + 1

    # every other value must be zero, assert it is so
    scales_diff2[idx - 1] = 0
    if not np.all(np.abs(scales_diff2) < 1e-15):
        return None
    else:
        return idx


def _process_fs_and_t(fs, t, N):
    """Ensures `t` is uniformly-spaced and of same length as `x` (==N)
    and returns `fs` and `dt` based on it, or from defaults if `t` is None.
    """
    if t is not None:
        if len(t) != N:
            # not explicitly used anywhere but ensures wrong `t` wasn't supplied
            raise Exception("`t` must be of same length as `x` "
                            "(%s != %s)" % (len(t), N))
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


def make_scales(N, min_scale=None, max_scale=None, nv=32, scaletype='log'):
    min_scale = min_scale or 1
    max_scale = max_scale or N

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
        na = int(np.ceil(max_scale / min_scale))
        scales = np.linspace(min_scale, max_scale, na)

    else:
        raise ValueError("`scaletype` must be 'log' or 'linear'; "
                         "got: %s" % scaletype)
    scales = scales.reshape(-1, 1)  # ensure 2D for broadcast ops later
    return scales


#### STFT utils ###############################################################
def buffer(x, seg_len, n_overlap):
    """Build 2D array where each column is a successive slice of `x` of length
    `seg_len` and overlapping by `n_overlap` (or equivalently incrementing
    starting index of each slice by `hop_len = seg_len - n_overlap`).

    Mimics MATLAB's `buffer`, with less functionality.

    Ex:
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        xb = buffer(x, seg_len=5, n_overlap=3)
        xb == [[0, 1, 2, 3, 4],
               [2, 3, 4, 5, 6],
               [4, 5, 6, 7, 8]].T
    """
    hop_len = seg_len - n_overlap
    n_segs = (len(x) - seg_len) // hop_len + 1
    out = np.zeros((seg_len, n_segs))

    for i in range(n_segs):
        start = i * hop_len
        end   = start + seg_len
        out[:, i] = x[start:end]
    return out


def unbuffer(xbuf, window, hop_len, n_fft, N, win_exp=1):
    """Undoes `buffer`, per padding logic used in `stft`:
        (N, n_fft) : logic
         even, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (128, 120, 247, 60, 59)
          odd,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (129, 121, 249, 60, 60)
         even,  odd: left = right
             (N, n_fft, len(xp), pl, pr) -> (128, 121, 248, 60, 60)
          odd, even: left = right + 1
             (N, n_fft, len(xp), pl, pr) -> (129, 120, 248, 60, 59)
    """
    if N is None:
        # assume greatest possible len(x) (unpadded)
        N = xbuf.shape[1] * hop_len + len(window) - 1
    if len(window) != n_fft:
        raise ValueError("Must have `len(window) == n_fft` "
                         "(got %s != %s)" % (len(window), n_fft))
    if win_exp == 0:
        window = 1
    elif win_exp != 1:
        window = window ** win_exp
    x = np.zeros(N + n_fft - 1)

    _overlap_add(x, xbuf, window, hop_len, n_fft)
    x = x[n_fft//2 : -((n_fft - 1)//2)]
    return x


def window_norm(window, hop_len, n_fft, N, win_exp=1):
    """Computes window modulation array for use in `stft` and `istft`."""
    wn = np.zeros(N + n_fft - 1)

    _window_norm(wn, window, hop_len, n_fft, win_exp)
    return wn[n_fft//2 : -((n_fft - 1)//2)]


@jit(nopython=True, cache=True)
def _overlap_add(x, xbuf, window, hop_len, n_fft):
    for i in range(xbuf.shape[1]):
        n = i * hop_len
        x[n:n + n_fft] += xbuf[:, i] * window


@jit(nopython=True, cache=True)
def _window_norm(wn, window, hop_len, n_fft, win_exp=1):
    max_hops = (len(wn) - n_fft) // hop_len + 1
    wpow = window ** (win_exp + 1)

    for i in range(max_hops):
        n = i * hop_len
        wn[n:n + n_fft] += wpow


def window_resolution(window):
    """Minimal function to compute a window's time & frequency widths, assuming
    Fourier spectrum centered about dc (else use `ssqueezepy.wavelets` methods).

    Returns std_w, std_t, harea. `window` must be np.ndarray and >=0.
    """
    from .wavelets import _xifn
    assert window.min() >= 0, "`window` must be >= 0 (got min=%s)" % window.min()
    N = len(window)

    t  = np.arange(-N/2, N/2, step=1)
    ws = fftshift(_xifn(1, N))

    psihs   = fftshift(fft(window))
    apsi2   = np.abs(window)**2
    apsih2s = np.abs(psihs)**2

    var_w = integrate.trapz(ws**2 * apsih2s, ws) / integrate.trapz(apsih2s, ws)
    var_t = integrate.trapz(t**2  * apsi2, t)    / integrate.trapz(apsi2, t)

    std_w, std_t = np.sqrt(var_w), np.sqrt(var_t)
    harea = std_w * std_t
    return std_w, std_t, harea


def window_area(window, time=True, frequency=False):
    """Minimal function to compute a window's time or frequency 'area' as area
    under curve of `abs(window)**2`. `window` must be np.ndarray.
    """
    from .wavelets import _xifn
    if not time and not frequency:
        raise ValueError("must compute something")

    if time:
        t = np.arange(-len(window)/2, len(window)/2, step=1)
        at = integrate.trapz(np.abs(window)**2, t)
    if frequency:
        ws = fftshift(_xifn(1, len(window)))
        apsih2s = np.abs(fftshift(fft(window)))**2
        aw = integrate.trapz(apsih2s, ws)

    if time and frequency:
        return at, aw
    elif time:
        return at
    return aw


#### misc utils ###############################################################
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


def _textwrap(txt, wrap_len=50):
    """Preserves line breaks and includes `'\n'.join()` step."""
    return '\n'.join(['\n'.join(
        wrap(line, wrap_len, break_long_words=False, replace_whitespace=False))
        for line in txt.splitlines() if line.strip() != ''])
