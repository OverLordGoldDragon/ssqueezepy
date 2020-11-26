# -*- coding: utf-8 -*-
import numpy as np
from .algos import find_closest, indexed_sum, replace_at_inf
from .utils import EPS, pi, process_scales, _infer_scaletype, _process_fs_and_t


def ssqueeze(Wx, w, ssq_freqs=None, scales=None, fs=None, t=None, transform='cwt',
             squeezing='sum'):
    """Calculates the synchrosqueezed CWT or STFT of `x`. Used internally by
    `synsq_cwt` and `synsq_stft_fwd`.

    # Arguments:
        Wx or Sx: np.ndarray
            CWT or STFT of `x`.

        w: np.ndarray
            Phase transform of `Wx` or `Sx`. Must be >=0.

        ssq_freqs: str['log', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.

        scales: str['log', 'linear'] / np.ndarray
            CWT scales. Ignored if transform='stft'.
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'linear': linearly distributed scales.
                  !!! EXPERIMENTAL; default scheme for len(x)>2048 performs
                  poorly (and there may not be a good non-piecewise scheme).

        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076

        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.

        transform: str['cwt', 'stft']
            Whether `Wx` is from CWT or STFT (`Sx`).

        squeezing: str['sum', 'lebesgue']
                - 'sum' = standard synchrosqueezing using `Wx`.
                - 'lebesgue' = as in [4], setting `Wx=ones()/len(Wx)`, which is
                not invertible but has better robustness properties in some cases.
                Not recommended unless purpose is understood.

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.

    # References:
        1. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_squeeze.m
    """
    def _ssqueeze(w, Wx, nv, ssq_freqs, transform, ssq_scaletype, cwt_scaletype):
        # incorporate threshold by zeroing out Inf values, so they get ignored
        Wx = replace_at_inf(Wx, ref=w, replacement=0)

        # do squeezing by finding which frequency bin each phase transform point
        # w[a, b] lands in (i.e. to which f in ssq_freqs each w[a, b] is closest)
        # equivalent to argmin(abs(w[a, b] - ssq_freqs)) for every a, b
        with np.errstate(divide='ignore'):
            k = (find_closest(w, ssq_freqs) if ssq_scaletype != 'log' else
                 find_closest(np.log2(w), np.log2(ssq_freqs)))

        # Tx[k[i, j], j] += Wx[i, j] * norm
        if transform == 'cwt':
            # Eq 14 [2]; Eq 2.3 [1]
            if cwt_scaletype == 'log':
                # ln(2)/nv == diff(ln(scales))[0] == ln(2**(1/nv))
                Tx = indexed_sum(Wx / scales**(1/2) * np.log(2) / nv, k)
            elif cwt_scaletype == 'linear':
                # omit /dw since it's cancelled by *dw in inversion anyway
                da = (scales[1] - scales[0])
                Tx = indexed_sum(Wx / scales**(3/2) * da, k)
        else:  # 'stft'
            # TODO validate
            Tx = indexed_sum(Wx * (ssq_freqs[1] - ssq_freqs[0]), k)
        return Tx

    def _compute_associated_frequencies(dt, na, N, transform, ssq_scaletype):
        dT = dt * N
        # normalized frequencies to map discrete-domain to physical:
        #     f[[cycles/samples]] -> f[[cycles/second]]
        # maximum measurable (Nyquist) frequency of data
        fM = 1 / (2 * dt)
        # minimum measurable (fundamental) frequency of data
        fm = 1 / dT

        # frequency divisions `w_l` to search over in Synchrosqueezing
        if ssq_scaletype == 'log':
            # [fm, ..., fM]
            ssq_freqs = fm * np.power(fM / fm, np.arange(na) / (na - 1))
        else:
            if transform == 'cwt':
                ssq_freqs = np.linspace(fm, fM, na)
            elif transform == 'stft':
                # ??? seems to be 0 to f_sampling/2, but why use N?
                # what about fm and fM?
                ssq_freqs = np.linspace(0, 1, N) / dt
                ssq_freqs = ssq_freqs[:N // 2]
        return ssq_freqs

    def _process_args(w, fs, t, N, transform, squeezing, scales):
        if w.min() < 0:
            raise ValueError("found negatives in `w`")
        if transform not in ('cwt', 'stft'):
            raise ValueError("`transform` must be one of: cwt, stft "
                             "(got %s)" % squeezing)
        if squeezing not in ('sum', 'lebesgue'):
            raise ValueError("`squeezing` must be one of: sum, lebesgue "
                             "(got %s)" % squeezing)
        if scales is None and transform == 'cwt':
            raise ValueError("`scales` can't be None if `transform == 'cwt'`")
        dt, *_ = _process_fs_and_t(fs, t, N)
        return dt

    na, N = Wx.shape
    dt = _process_args(w, fs, t, N, transform, squeezing, scales)

    scales, cwt_scaletype, _, nv = process_scales(scales, N, get_params=True)

    if not isinstance(ssq_freqs, np.ndarray):
        if isinstance(ssq_freqs, str):
            ssq_scaletype = ssq_freqs
        else:
            # default to same scheme used by `scales`
            ssq_scaletype = cwt_scaletype
        ssq_freqs = _compute_associated_frequencies(dt, na, N, transform,
                                                    ssq_scaletype)
    else:
        ssq_scaletype = _infer_scaletype(ssq_freqs)

    if squeezing == 'lebesgue':  # from reference [3]
        Wx = np.ones(Wx.shape) / len(Wx)

    Tx = _ssqueeze(w, Wx, nv, ssq_freqs, transform, ssq_scaletype, cwt_scaletype)
    return Tx, ssq_freqs


def phase_cwt(Wx, dWx, difftype='direct', gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
          w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    See above Eq 20.3 in [1], or Eq 13 in [2].

    # Arguments:
        Wx: np.ndarray
            CWT of `x` (see `cwt`).

        dWx: np.ndarray.
            Time-derivative of `Wx`, computed via frequency-domain differentiation
            (effectively, derivative of trigonometric interpolation; see [4]).

        difftype: str['direct', 'phase']
            Method by which to differentiate Wx (default='direct') to obtain
            instantaneous frequencies:
                    w(a,b) = Im( (1/2pi) * (1/Wx(a,b)) * d/db[Wx(a,b)] )

                - 'direct': using `dWx` (see `dWx`).
                - 'phase': differentiate by taking forward finite-difference of
                unwrapped angle of `Wx` (see `phase_cwt`).

        gamma: float / None
            CWT phase threshold. Sets `w=inf` for small values of `Wx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Wx) < beta] = inf
            This is used to zero `Wx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = sqrt(machine epsilon) = np.sqrt(np.finfo(np.float64).eps)

    # Returns:
        w: np.ndarray
            Phase transform for each element of `Wx`. w.shape == Wx.shape.

    # References:
        1. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        4. The Exponential Accuracy of Fourier and Chebyshev Differencing Methods.
        E. Tadmor.
        http://webhome.auburn.edu/~jzl0097/teaching/math_8970/Tadmor_86.pdf

        5. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        phase_cwt.m
    """
    # Calculate phase transform for each `ai`, normalize by 2pi
    if difftype == 'phase':
        # TODO gives bad results; shouldn't we divide by Wx?
        u = np.unwrap(np.angle(Wx)).T
        w = np.vstack([np.diff(u, axis=0), u[-1] - u[0]]).T / (2 * pi)
    else:
        with np.errstate(divide='ignore'):
            w = np.imag(dWx / Wx) / (2 * pi)

    gamma = gamma or np.sqrt(EPS)
    w[(np.abs(Wx) < gamma) | (w < 0)] = np.inf
    return w


def phase_cwt_num(Wx, dt, difforder=4, gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses numerical differentiation (1st, 2nd, or 4th order). See above Eq 20.3
    in [1], or Eq 13 in [2].

    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x` (see `cwt`).

        dt: int. Sampling period (e.g. t[1] - t[0]).

        difforder: int[1, 2, 4]
            Order of differentiation (default=4).

        gamma: float
            CWT phase threshold. Sets `w=inf` for small values of `Wx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Wx) < beta] = inf
            This is used to zero `Wx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = sqrt(machine epsilon) = np.sqrt(np.finfo(np.float64).eps)

    # Returns:
        w: np.ndarray
            Phase transform via demodulated FM-estimates. w.shape == Wx.shape.

    # References:
        1. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        phase_cwt_num.m
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
    w[np.abs(Wx) < gamma] = np.inf

    # calculate inst. freq for each scale
    # 2*pi norm per discretized inverse FT rather than inverse DFT
    w = np.real(-1j * w / Wx) / (2 * pi)
    w[w < 0] = np.inf
    return w
