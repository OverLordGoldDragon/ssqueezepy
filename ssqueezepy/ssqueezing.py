import numpy as np
from .algos import find_closest, indexed_sum, replace_at_inf_or_nan
from .utils import process_scales


EPS = np.finfo(np.float64).eps  # machine epsilon for float64  # TODO float32?
pi = np.pi


def ssqueeze(Wx, w, scales, t, freqscale=None, transform='cwt', squeezing='full'):
    """Calculates the synchrosqueezed CWT or STFT of `x`. Used internally by
    `synsq_cwt` and `synsq_stft_fwd`.

    # Arguments:
        Wx or Sx: np.ndarray. CWT or STFT of `x`.
        w: np.ndarray. Phase transform at same locations in T-F plane.
        scales: CWT scales. np.ndarray or ('log', 'linear')
                !!! beware of scales='linear'; bad current default scheme for
                capturing low frequencies for sequences longer than 2048.
                Recommended scales='log' with freqscale='linear' instead.
        t: np.ndarray. Time vector.
        freqscale: synchrosqueezing plane scales. np.ndarray or ('log', 'linear')
        opts: dict. Options:
            'transform': ('CWT', 'STFT'). Underlying time-frequency transform.
            'squeezing': ('full', 'measure'). Latter corresponds to approach
                         in [3], which is not invertible but has better
                         robustness properties in some cases; not recommended
                         unless you know what you're doing.

    # Returns:
        Tx: synchrosqueezed output.
        fs: associated frequencies.

    Note the multiplicative correction term x in `synsq_cwt_squeeze_mex`,
    required due to the fact that the squeezing integral of Eq. (2.7), in,
    [1], is taken w.r.t. dlog(a). This correction term needs to be included
    as a factor of Eq. (2.3), which we implement here.

    A more detailed explanation is available in Sec. III of [2].
    Note the constant multiplier log(2)/nv has been moved to the
    inverse of the normalization constant, as calculated in `adm_ssq`.

    # References:
        1. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms:
        an empricial mode decomposition-like tool",
        Applied and Computational Harmonic Analysis, 30(2):243-261, 2011.

        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.

        3. G. Thakur and H.-T. Wu,  "Synchrosqueezing-based Recovery of
        Instantaneous Frequency from Nonuniform Samples",
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
    """
    def _ssqueeze(w, Wx, fs, transform, freqscale):
        # incorporate threshold by zeroing out Inf values, so they get ignored
        Wx = replace_at_inf_or_nan(Wx, ref=w, replacement=0)
        # reassign indeterminate (ignored per above anyway) to avoid warnings
        w  = replace_at_inf_or_nan(w, ref=w, replacement=fs[-1])

        # do squeezing by finding which frequency bin each phase transform point
        # w[a, b] lands in (i.e. to which f in fs each w[a, b] is closest to)
        # equivalent to argmin(abs(w[a, b] - fs)) for every a, b
        k = (find_closest(w, fs) if freqscale != 'log' else
             find_closest(np.log2(w), np.log2(fs)))

        # Tx[k[i, j], j] += Wx[i, j] * norm
        if transform == 'cwt':
            # Eq 14 [2]; Eq 2.3 [1]
            if freqscale == 'log':
                # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
                Tx = indexed_sum(Wx / scales**(1/2) * np.log(2) / nv, k)
            elif freqscale == 'linear':
                # omit /dw since it's cancelled by *dw in inversion anyway
                da = (scales[1] - scales[0])
                Tx = indexed_sum(Wx / scales**(3/2) * da, k)
        else:  # 'stft'
            Tx = indexed_sum(Wx * (fs[1] - fs[0]), k)  # TODO validate
        return Tx

    def _compute_associated_frequencies(t, na, N, transform, freqscale):
        dT = t[-1] - t[0]
        dt = t[1]  - t[0]
        # normalized frequencies to map discrete-domain to physical:
        #     f[[cycles/samples]] -> f[[cycles/second]]
        # maximum measurable (Nyquist) frequency of data
        fM = 1 / (2 * dt)
        # minimum measurable (fundamental) frequency of data
        fm = 1 / dT

        # frequency divisions `w_l` to search over in Synchrosqueezing
        if freqscale == 'log':
            fs = fm * np.power(fM / fm, np.arange(na) / (na - 1))  # [fm,...,fM]
        else:
            if transform == 'cwt':
                fs = np.linspace(fm, fM, na)
            else:  # 'stft'
                # ??? seems to be 0 to f_sampling/2, but why use N?
                # what about fm and fM?
                fs = np.linspace(0, 1, N) / dt
                fs = fs[:N // 2]
        return fs

    def _process_args(w, transform, squeezing):
        if w.min() < 0:
            raise ValueError("found negatives in `w`")
        if transform not in ('cwt', 'stft'):
            raise ValueError("`transform` must be one of: cwt, stft "
                             "(got %s)" % squeezing)
        if squeezing not in ('full', 'measure'):
            raise ValueError("`squeezing` must be one of: full, measure "
                             "(got %s)" % squeezing)

    _process_args(w, transform, squeezing)

    na, N = Wx.shape
    scales, _freqscale, _, nv = process_scales(scales, N, get_params=True)
    if freqscale is None:
        # default to same scheme used by `scales`
        # !!! scales='linear' not recommended for len(x)>2048; see docstr
        freqscale = _freqscale
    fs = _compute_associated_frequencies(t, na, N, transform, freqscale)

    if squeezing == 'measure':  # from reference [3]
        # !!! not recommended unless having specific reason;
        # no reconstruction; not validated
        Wx = np.ones(Wx.shape) / len(Wx)

    Tx = _ssqueeze(w, Wx, fs, transform, freqscale)
    return Tx, fs


def phase_cwt(Wx, dWx, difftype='direct', gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses direct differentiation by calculating dWx/db in frequency domain
    (the secondary output of `cwt`, see `cwt`)

    This is the analytic implementation of Eq. (7) of [1].

    # Arguments:
        Wx: np.ndarray. wavelet transform of `x` (see `cwt`).
        dWx: np.ndarray. Samples of time derivative of wavelet transform of `x`
             (see `cwt`).
        opts. dict. Options:
            'gamma': wavelet threshold (default: sqrt(machine epsilon))

    # Returns:
        w: phase transform, w.shape == Wx.shape.

    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.

        2. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms:
        an empricial mode decomposition-like tool",
        Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
    """
    # Calculate phase transform for each `ai`, normalize by 2pi
    if difftype == 'phase':
        # TODO gives bad results; shouldn't we divide by Wx? also gives w < 0
        u = np.unwrap(np.angle(Wx)).T
        w = np.vstack([np.diff(u, axis=0), u[-1] - u[0]]).T
        w = np.abs(w)  # !!! <- forced step, messes up values
    else:
        w = np.abs(np.imag(dWx / Wx))
    w /= (2 * pi)

    gamma = gamma or np.sqrt(EPS)
    w[np.abs(Wx) < gamma] = np.inf
    return w


def phase_cwt_num(Wx, dt, difforder=4, gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses numerical differentiation (1st, 2nd, or 4th order).

    This is a numerical differentiation implementation of Eq. (7) of [1].

    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x` (see `cwt`).
        dt: int. Sampling period (e.g. t[1] - t[0]).
        opts. dict. Options:
            'dorder': int (1, 2, 4). Differences order. (default = 4)
            'gamma': float. Wavelet threshold. (default = sqrt(machine epsilon))

    # Returns:
        w: demodulated FM-estimates, w.shape == Wx.shape.

    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.
    """
    # unreliable; bad results on high freq pure tones
    def _differentiate(Wx, dt):
        if difforder in (2, 4):
            # append for differentiating
            Wxr = np.hstack([Wx[:, -2:], Wx, Wx[:, :2]])

        if difforder == 1:
            w = np.hstack([Wx[:, 1:] - Wx[:, :-1],
                           Wx[:, :1]  - Wx[:, -1:]])
            w /= dt
        elif difforder == 2:
            # calculate 2nd-order forward difference
            w = -Wxr[:, 4:] + 4 * Wxr[:, 3:-1] - 3 * Wxr[:, 2:-2]
            w /= (2 * dt)
        elif difforder == 4:
            # calculate 4th-order central difference
            w = -Wxr[:, 4:]
            w += Wxr[:, 3:-1] * 8
            w -= Wxr[:, 1:-3] * 8
            w += Wxr[:, 0:-4]
            w /= (12 * dt)
        return w

    # epsilon from Daubechies, H-T Wu, et al.
    # gamma from Brevdo, H-T Wu, et al.
    gamma = gamma or np.sqrt(EPS)
    if difforder not in (1, 2, 4):
        raise ValueError("`difforder` must be one of: 1, 2, 4 "
                         "(got %s)" % difforder)

    w = _differentiate(Wx, dt)
    w[np.abs(Wx) < gamma] = np.nan

    # calculate inst. freq for each scale
    # 2*pi norm per discretized inverse FT rather than inverse DFT
    w = np.real(-1j * w / Wx) / (2 * pi)
    w = np.abs(w)  # ??? not in original implem., but otherwise w may be < 0
    return w
