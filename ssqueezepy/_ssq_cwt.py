# -*- coding: utf-8 -*-
import numpy as np
from .utils import WARN, EPS, p2up, adm_ssq, process_scales, _process_fs_and_t
from .ssqueezing import ssqueeze, phase_cwt, phase_cwt_num
from .wavelets import Wavelet
from ._cwt import cwt


def ssq_cwt(x, wavelet='morlet', scales='log', nv=None, fs=None, t=None,
            ssq_freqs=None, padtype='reflect', squeezing='sum', mapkind='maximal',
            difftype='direct', difforder=None, gamma=None):
    """Calculates the synchrosqueezed Continuous Wavelet Transform of `x`.
    Implements the algorithm described in Sec. III of [1].

    # Arguments:
        x: np.ndarray
            Input vector, 1D.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain. See help(cwt).

        scales: str['log', 'linear', 'log:maximal', ...] / np.ndarray
            CWT scales. See help(cwt).

        nv: int / None
            Number of voices (CWT only). Suggested >= 32 (default=32).

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

        ssq_freqs: str['log', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.

        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'reflect', 'symmetric', 'replicate').
            'zero' is most naive, while 'reflect' (default) partly mitigates
            boundary effects. See `help(utils.padsignal)`.

        squeezing: str['sum', 'lebesgue']
                - 'sum' = standard synchrosqueezing using `Wx`.
                - 'lebesgue' = as in [4], setting `Wx=ones()/len(Wx)`, which is
                not invertible but has better robustness properties in some cases.
                Not recommended unless you know what you're doing.

        mapkind: str['maximal', 'peak', 'energy']
            Kind of frequency mapping used, determining the range of frequencies
            spanned (fm to fM, min to max).

                - 'maximal': fm=1/dT, fM=1/(2*dt), always. Data's fundamental
                and Nyquist frequencies, determined from `fs` (or `t`).
                Other mappings can never span outside this range.
                - ('peak', 'energy'): sets fm and fM based on center frequency
                associated with `wavelet` at maximum and minimum scale,
                respectively. See help(wavelets.center_frequency)
                - 'peak': the frequency-domain trimmed bell will have its peak
                at Nyquist, meaning all other frequencies are beneath, so each
                scale is still correctly resolved but with downscaled energies.
                With sufficiently-spanned `scales`, coincides with 'maximal'.
                - 'energy': however, the bell's spectral energy is centered
                elsewhere, as right-half of bell is partly or entirely trimmed
                (left-half can be trimmed too). Use for energy-centric mapping,
                which for sufficiently-spanned `scales` will always have lesser
                fM (but ~same fM).

        difftype: str['direct', 'phase', 'numeric']
            Method by which to differentiate Wx (default='direct') to obtain
            instantaneous frequencies:
                    w(a,b) = Im( (1/2pi) * (1/Wx(a,b)) * d/db[Wx(a,b)] )

                - 'direct': use `dWx`, obtained via frequency-domain
                differentiation (see `cwt`, `phase_cwt`).
                - 'phase': differentiate by taking forward finite-difference of
                unwrapped angle of `Wx` (see `phase_cwt`).
                - 'numeric': first-, second-, or fourth-order (set by
                `difforder`) numeric differentiation (see `phase_cwt_num`).

        difforder: int[1, 2, 4]
            Order of differentiation for difftype='numeric' (default=4).

        gamma: float / None
            CWT phase threshold. Sets `w=inf` for small values of `Wx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Wx) < beta] = inf
            This is used to zero `Wx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = sqrt(machine epsilon) = np.sqrt(np.finfo(np.float64).eps)

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.
        Wx: np.ndarray [na x n]
            Continuous Wavelet Transform of `x` L1-normed (see `cwt`).
        scales: np.ndarray [na]
            Scales associated with rows of `Wx`.
        w: np.ndarray [na x n]
            Phase transform for each element of `Wx`.

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        3. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        4. Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples. G. Thakur and H.-T. Wu.
        https://arxiv.org/abs/1006.2533

        5. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_fw.m
    """
    def _process_args(N, scales, fs, t, nv, difftype, difforder, squeezing):
        if difftype not in ('direct', 'phase', 'numeric'):
            raise ValueError("`difftype` must be one of: direct, phase, numeric"
                             " (got %s)" % difftype)
        if difforder is not None:
            if difftype != 'numeric':
                WARN("`difforder` is ignored if `difftype != 'numeric'")
            elif difforder not in (1, 2, 4):
                raise ValueError("`difforder` must be one of: 1, 2, 4 "
                                 "(got %s)" % difforder)
        elif difftype == 'numeric':
            difforder = 4
        if squeezing not in ('sum', 'lebesgue'):
            raise ValueError("`squeezing` must be one of: sum, lebesgue "
                             "(got %s)" % squeezing)
        if nv is None and not isinstance(scales, np.ndarray):
            nv = 32

        dt, fs, t = _process_fs_and_t(fs, t, N)
        return dt, fs, difforder, nv

    def _phase_transform(Wx, dWx, N, dt, gamma, difftype, difforder):
        if difftype == 'direct':
            # calculate instantaneous frequency directly from the
            # frequency-domain derivative
            w = phase_cwt(Wx, dWx, difftype, gamma)
        elif difftype == 'phase':
            # !!! bad; yields negatives, and forcing abs(w) doesn't help
            # calculate inst. freq. from unwrapped phase of CWT
            w = phase_cwt(Wx, None, difftype, gamma)
        elif difftype == 'numeric':
            # !!! tested to be very inaccurate for small `a`
            # calculate derivative numericly
            _, n1, _ = p2up(N)
            Wx = Wx[:, (n1 - 4):(n1 + N + 4)]
            w = phase_cwt_num(Wx, dt, difforder, gamma)
        return Wx, w

    N = len(x)
    dt, fs, difforder, nv = _process_args(N, scales, fs, t, nv, difftype,
                                          difforder, squeezing)

    wavelet = Wavelet._init_if_not_isinstance(wavelet, N=N)
    scales, cwt_scaletype, *_ = process_scales(scales, N, wavelet, nv=nv,
                                               get_params=True)

    # l1_norm=True to spare a multiplication; for SSWT L1 & L2 are exactly same
    # anyway since we're inverting CWT over time-frequency plane
    rpadded = (difftype == 'numeric')
    Wx, scales, _, dWx = cwt(x, wavelet, scales=scales, fs=fs, nv=nv,
                             l1_norm=True, derivative=True, padtype=padtype,
                             rpadded=rpadded)

    gamma = gamma or np.sqrt(EPS)
    Wx, w = _phase_transform(Wx, dWx, N, dt, gamma, difftype, difforder)

    if ssq_freqs is None:
        # default to same scheme used by `scales`
        ssq_freqs = cwt_scaletype

    Tx, ssq_freqs = ssqueeze(Wx, w, scales=scales, fs=fs, ssq_freqs=ssq_freqs,
                             transform='cwt', squeezing=squeezing,
                             mapkind=mapkind, wavelet=wavelet)

    if difftype == 'numeric':
        Wx = Wx[:, 4:-4]
        w  = w[:,  4:-4]
        Tx = Tx[:, 4:-4]
    return Tx, ssq_freqs, Wx, scales, w


def issq_cwt(Tx, wavelet, cc=None, cw=None):
    """Inverse synchrosqueezing transform of `Tx` with associated frequencies
    in `fs` and curve bands in time-frequency plane specified by `Cs` and
    `freqband`. This implements Eq. 15 of [1].

    # Arguments:
        Tx: np.ndarray
            Synchrosqueezed CWT of `x` (see `ssq_cwt`).
            (rows=~frequencies, cols=timeshifts)

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet that was used to compute Tx, sampled in Fourier
            frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.

        cc, cw: np.ndarray / None
            Curve centerpoints, and curve (vertical) widths (bandwidths),
            together defining the portion of Tx to invert over to extract
            K "components" per Modulation Model:
                        x_k(t) = A_k(t) cos(phi_k(t)) + res;  k=0,...,K-1
            where K=len(c)==len(cw), and `res` is residual error (inversion
            over portion leftover/uncovered by cc, cw).
            None = full inversion.

    # Returns:
        x: np.ndarray [K x Tx.shape[1]]
            Components of reconstructed signal, and residual error.
            If cb & cw are None, x.shape == (Tx.shape[1],). See `cb, cw`.

    # Example:
        Tx, *_ = ssq_cwt(x, 'morlet')    # synchrosqueezed CWT
        x      = issq_cwt(Tx, 'morlet')  # reconstruction

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        3. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_iw.m
    """
    def _invert_components(Tx, cc, cw):
        # Invert Tx around curve masks in the time-frequency plane to recover
        # individual components; last one is the remaining signal
        x = np.zeros((cc.shape[1] + 1, cc.shape[0]))
        TxRemainder = Tx.copy()

        for n in range(cc.shape[1]):
            TxMask = np.zeros(Tx.shape, dtype='complex128')
            upper_cc = np.clip(cc[:, n] + cw[:, n], 0, len(Tx))
            lower_cc = np.clip(cc[:, n] - cw[:, n], 0, len(Tx))

            # cc==-1 denotes no curve at that time,
            # removing such points from inversion
            upper_cc[np.where(cc[:, n] == -1)] = 0
            lower_cc[np.where(cc[:, n] == -1)] = 1
            for m in range(Tx.shape[1]):
                idxs = slice(lower_cc[m], upper_cc[m] + 1)
                TxMask[idxs, m] = Tx[idxs, m]
                TxRemainder[idxs, m] = 0
            x[n] = TxMask.real.sum(axis=0).T

        x[n + 1] = TxRemainder.real.sum(axis=0).T
        return x

    def _process_args(cc, cw):
        if (cc is None) and (cw is None):
            return None, None, True
        if cc.ndim == 1:
            cc = cc.reshape(-1, 1)
        if cw.ndim == 1:
            cw = cw.reshape(-1, 1)
        cc = cc.astype('int32')
        cw = cw.astype('int32')
        return cc, cw, False

    cc, cw, full_inverse = _process_args(cc, cw)

    if full_inverse:
        # Integration over all frequencies recovers original signal
        x = Tx.real.sum(axis=0)
    else:
        x = _invert_components(Tx, cc, cw)

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    Css = adm_ssq(wavelet)  # admissibility coefficient
    # *2 per analytic wavelet & taking real part; Theorem 4.5 [2]
    x *= (2 / Css)
    return x
