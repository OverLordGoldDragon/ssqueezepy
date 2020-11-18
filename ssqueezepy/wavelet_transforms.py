# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
from .algos import find_closest, indexed_sum, replace_at_inf_or_nan


EPS = np.finfo(np.float64).eps  # machine epsilon for float64  # TODO float32?
PI = np.pi


def synsq_squeeze(Wx, w, scales, t, transform='cwt', ssq_freqs='log', nv=None,
                  squeezing='full'):
    """Calculates the synchrosqueezed CWT or STFT of `x`. Used internally by
    `synsq_cwt_fwd` and `synsq_stft_fwd`.

    # Arguments:
        Wx or Sx: np.ndarray. CWT or STFT of `x`.
        w: np.ndarray. Phase transform at same locations in T-F plane.
        t: np.ndarray. Time vector.
        nv: int. Number of voices (CWT only).
        opts: dict. Options:
            'transform': ('CWT', 'STFT'). Underlying time-frequency transform.
            'ssq_freqs': ('log', 'linear'). Frequency bins/divisions.
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
    inverse of the normalization constant, as calculated in `synsq_adm`.

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
    def _squeeze(w, Wx, fs, transform, ssq_freqs):
        # incorporate threshold by zeroing out Inf values, so they get ignored
        Wx = replace_at_inf_or_nan(Wx, ref=w, replacement=0)
        # reassign indeterminate (ignored per above anyway) to avoid warnings
        w  = replace_at_inf_or_nan(w, ref=w, replacement=fs[-1])

        # do squeezing by finding which frequency bin each phase transform point
        # w[a, b] lands in (i.e. to which f in fs each w[a, b] is closest to)
        # equivalent to argmin(abs(w[a, b] - fs)) for every a, b
        k = (find_closest(w, fs) if ssq_freqs != 'log' else
             find_closest(np.log2(w), np.log2(fs)))

        dfs = np.diff(fs, axis=0)
        dfs = np.array([dfs[0] - (dfs[1] - dfs[0]), *dfs]).reshape(-1, 1)
        da = np.diff(scales, axis=0)
        da = np.array([da[0] - (da[1] - da[0]), *da]).reshape(-1, 1)

        # Tx[k[i, j], j] += Wx[i, j]
        sc = 1 #scales.reshape(-1, 1)
        # ??? what is dz? dz=1, Mallat pg 81 bottom
        # Tx = indexed_sum(Wx * da / dfs / nv / sc, k)
        # log(2) / nv == diff(log(scales))[0]
        Tx = indexed_sum(Wx / np.sqrt(scales) * np.log(2) / nv, k)

        # if transform == 'cwt':
            # ??? ref[1]-pg6, don't we need diff(scales)? and /= dw not *=?
            # Tx *= (1 / nv)  # ??? what's this
            # Tx *= (fs[1] - fs[0])  # ??? and this; shouldn't it be log2 if log?
        return Tx

    def _compute_associated_frequencies(t, transform, ssq_freqs):
        dT = t[-1] - t[0]
        dt = t[1]  - t[0]
        # normalized frequencies to map discrete-domain to physical:
        #     f[[cycles/samples]] -> f[[cycles/second]]
        # maximum measurable (Nyquist) frequency of data
        fM = 1 / (2 * dt)
        # minimum measurable (fundamental) frequency of data
        fm = 1 / dT

        # `na` is number of scales for CWT, number of freqs for STFT
        na, N = Wx.shape

        # frequency divisions `w_l` to search over in Synchrosqueezing
        if ssq_freqs == 'log':
            fs = fm * np.power(fM / fm, np.arange(na) / (na - 1))
        else:
            if transform == 'cwt':
                fs = np.linspace(fm, fM, na)
            else:  # 'stft'
                # ??? seems to be 0 to f_sampling/2, but why use N?
                fs = np.linspace(0, 1, N) / dt
                fs = fs[:N // 2]
        return fs

    def _process_args(w, transform, ssq_freqs, squeezing):
        if w.min() < 0:
            raise ValueError("found negatives in `w`")
        if transform not in ('cwt', 'stft'):
            raise ValueError("`transform` must be one of: cwt, stft "
                             "(got %s)" % squeezing)
        if ssq_freqs is None:
            ssq_freqs = 'log' if transform == 'cwt' else 'linear'
        if squeezing not in ('full', 'measure'):
            raise ValueError("`squeezing` must be one of: full, measure "
                             "(got %s)" % squeezing)
        return ssq_freqs

    ssq_freqs = _process_args(w, transform, ssq_freqs, squeezing)
    fs = _compute_associated_frequencies(t, transform, ssq_freqs)

    if squeezing == 'measure':  # from reference [3]
        Wx = np.ones(Wx.shape) / len(Wx)

    Tx = _squeeze(w, Wx, fs, transform, ssq_freqs)
    return Tx, fs


# TODO is this just repeating above or some difference?
def synsq_cwt_squeeze(Wx, w, t, nv):
    """Calculates the synchrosqueezed transform of `f` on a logarithmic scale.
    Used internally by `synsq_cwt_fwd`.  # TODO no it's not; wtf is this?

    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x`.
        w: np.ndarray. Estimate of frequency locations in `Wx`
                       (see `synsq_cwt_fwd`).
        t: np.ndarray. Time vector.
        nv: int. Number of voices.

    # Returns:
        Tx: synchrosqueezed output.
        fs: associated frequencies.

    Note the multiplicative correction term `f` in `_cwt_squeeze`, required
    due to the fact that the squeezing integral of Eq. (2.7), in, [1], is taken
    w.r.t. dlog(a). This correction term needs to be included as a factor of
    Eq. (2.3), which  we implement here.

    # TODO understand below
    A more detailed explanation is available in Sec. III of [2].
    Specifically, this is an implementation of Sec. IIIC, Alg. 1.
    Note the constant multiplier log(2)/nv has been moved to the
    inverse of the normalization constant, as calculated in `synsq_adm`.

    # References:
        1. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms: a
        tool for empirical mode decomposition", 2010.

        2. E. Brevdo, N.S. Fučkar, G. Thakur, and H-T. Wu, "The
        Synchrosqueezing algorithm: a robust analysis tool for signals
        with time-varying spectrum," 2011.
    """
    def _cwt_squeeze(Wx, w, scales, fs, dfs, N, lfm, lfM):
        Tx = np.zeros(Wx.shape)

        for b in range(N):
          for ai in range(len(scales)):
            if not np.isinf(w[ai, b]) and w[ai, b] > 0:
              # find w_l nearest to w[ai, b]
              k = int(np.min(np.max(
                  1 + np.floor(na / (lfM - lfm) * (np.log2(w[ai, b]) - lfm)),
                  0), na - 1))
              Tx[k, b] += Wx[ai, b] * scales[ai] ** (-0.5)
        return Tx

    dt = t[1]  - t[0]
    dT = t[-1] - t[0]

    # Maximum measurable frequency of data
    #fM = 1/(4*dt) # wavelet limit - tested
    fM = 1/(2*dt)  # standard
    # Minimum measurable frequency, due to wavelet
    fm = 1/dT;
    #fm = 1/(2*dT); # standard

    na, N = Wx.shape

    scales = np.power(2 ** (1 / nv), np.expand_dims(np.arange(1, na + 1)))
    # dscales = np.array([1, np.diff(scales, axis=0)])

    lfm = np.log2(fm)
    lfM = np.log2(fM)
    fs  = np.power(2, np.linspace(lfm, lfM, na))
    #dfs = np.array([fs[0], np.diff(fs, axis=0)])

    # Harmonics of diff. frequencies but same magniude have same |Tx|
    dfs = np.ones(fs.shape)

    if np.linalg.norm(Wx, 'fro') < EPS:
        Tx = np.zeros(Wx.shape)
    else:
        Tx = (1 / nv) * _cwt_squeeze(
            Wx, w, scales, fs, dfs, N, lfm, lfM)
    return Tx


def phase_cwt(Wx, dWx, difftype='direct', gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses direct differentiation by calculating dWx/db in frequency domain
    (the secondary output of `cwt_fwd`, see `cwt_fwd`)

    This is the analytic implementation of Eq. (7) of [1].

    # Arguments:
        Wx: np.ndarray. wavelet transform of `x` (see `cwt_fwd`).
        dWx: np.ndarray. Samples of time derivative of wavelet transform of `x`
             (see `cwt_fwd`).
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
        # w = np.abs(np.imag(w / Wx))  <- does not help, nor does simply abs(w)
    else:
        w = np.abs(np.imag(dWx / Wx))  # equivalent to abs(dWx) / abs(Wx)
    w /= (2 * PI)

    gamma = gamma or np.sqrt(EPS)
    w[np.abs(Wx) < gamma] = np.inf
    return w


def phase_cwt_num(Wx, dt, difforder=4, gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses numerical differentiation (1st, 2nd, or 4th order).

    This is a numerical differentiation implementation of Eq. (7) of [1].

    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x` (see `cwt_fwd`).
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

    # calculate inst. freq for each `ai`, normalize by (2*pi) for true freq
    w = np.real(-1j * w / Wx) / (2 * PI)
    w = np.abs(w)  # ??? not in original implem., but otherwise w may be < 0
    return w
