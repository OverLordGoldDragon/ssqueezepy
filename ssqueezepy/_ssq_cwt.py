# -*- coding: utf-8 -*-
import numpy as np
from .utils import WARN, EPS, p2up, adm_ssq, process_scales
from .ssqueezing import ssqueeze, phase_cwt, phase_cwt_num
from ._cwt import cwt


def ssq_cwt(x, wavelet='morlet', scales='log', nv=None, t=None, fs=None,
            ssq_freqs=None, padtype='symmetric', squeezing='full',
            difftype='direct', difforder=None, gamma=None):
    """Calculates the synchrosqueezed Continuous Wavelet Transform of `x`.
    Implements the algorithm described in Sec. III of [1].

    # Arguments:
        x: np.ndarray
            Vector of signal samples (e.g. x = np.cos(20 * np.pi * t))
        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.
        scales: str['log', 'linear'] / np.ndarray
            CWT scales.
                - 'log': exponentially distributed scales, as pow of 2:
                         `[2^(1/nv), 2^(2/nv), ...]`
                - 'linear': linearly distributed scales.
                  !!! EXPERIMENTAL; default scheme for len(x)>2048 performs
                  poorly (and there may not be a good non-piecewise scheme).
        nv: int / None
            Number of voices (CWT only). Suggested >= 32 (default=32).
        t: np.ndarray / None
            Vector of times at which samples are taken (eg np.linspace(0, 1, n)).
            Must be uniformly-spaced.
            Defaults to `np.linspace(0, len(x)/fs, len(x), endpoint=False)`.
            Overrides `fs` if not None.
        fs: float / None
            Sampling frequency of `x`. Defaults to 1, which makes ssq
            frequencies range from 1/dT to 0.5, i.e. as fraction of reference
            sampling rate up to Nyquist limit; dT = total duration (N/fs).
            Overridden by `t`, if provided.
            Relevant on `t` and `dT`: https://dsp.stackexchange.com/a/71580/50076
        ssq_freqs: str['log', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.
        padtype: str
            Pad scheme to apply on input. One of:
                ('zero', 'symmetric', 'replicate').
            'zero' is most naive, while 'symmetric' (default) partly mitigates
            boundary effects. See `padsignal`.
        squeezing: str['full', 'measure']
                - 'full' = standard synchrosqueezing using `Wx`.
                - 'measure' = as in [4], setting `Wx=ones()`, which is not
                invertible but has better robustness properties in some cases.
                Not recommended unless you know what you're doing.
        difftype: str['direct', 'phase', 'numerical']
            Method by which to differentiate Wx (default='direct') to obtain
            instantaneous frequencies:
                    w(a,b) = Im( (1/2pi) * (1/Wx(a,b)) * d/db[Wx(a,b)] )

                - 'direct': use `dWx`, obtained via frequency-domain
                differentiation (see `cwt`, `phase_cwt`).
                - 'phase': differentiate by taking forward finite-difference of
                unwrapped angle of `Wx` (see `phase_cwt`).
                - 'numerical': first-, second-, or fourth-order (set by
                `difforder`) numeric differentiation (see `phase_cwt_num`).
        difforder: int[1, 2, 4]
            Order of differentiation for difftype='numerical' (default=4).
        gamma: float
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
        fs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.
        Wx: np.ndarray [na x n]
            Continuous Wavelet Transform of `x` L2-normed (see `cwt`);
            to L1-norm, `Wx /= np.sqrt(scales)`
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
    def _process_args(x, t, fs, nv, difftype, difforder, squeezing):
        if difftype not in ('direct', 'phase', 'numerical'):
            raise ValueError("`difftype` must be one of: direct, phase, numerical"
                             " (got %s)" % difftype)
        if difforder is not None:
            if difftype != 'numerical':
                print(WARN, "`difforder` is ignored if `difftype != 'numerical'")
            elif difforder not in (1, 2, 4):
                raise ValueError("`difforder` must be one of: 1, 2, 4 "
                                 "(got %s)" % difforder)
        elif difftype == 'numerical':
            difforder = 4
        if squeezing not in ('full', 'measure'):
            raise ValueError("`squeezing` must be one of: full, measure "
                             "(got %s)" % squeezing)
        if t is None:
            fs = fs or 1
            t = np.linspace(0., len(x) / fs, len(x), endpoint=False)
        elif not np.mean(np.abs(np.diff(t, 2, axis=0))) < 1e-7:  # float32 thresh
            raise Exception("Time vector `t` must be uniformly sampled.")
        elif len(t) != len(x):
            # not explicitly used anywhere but ensures wrong `t` wasn't supplied
            raise Exception("`t` must be of same length as `x`.")

        nv = nv or 32
        return t, difforder, nv

    def _phase_transform(Wx, dWx, gamma, n1, dt, difftype, difforder):
        if difftype == 'direct':
            # calculate instantaneous frequency directly from the
            # frequency-domain derivative
            w = phase_cwt(Wx, dWx, difftype, gamma)
        elif difftype == 'phase':
            # !!! bad; yields negatives, and forcing abs(w) doesn't help
            # calculate inst. freq. from unwrapped phase of CWT
            w = phase_cwt(Wx, None, difftype, gamma)
        elif difftype == 'numerical':
            # !!! tested to be very inaccurate for small `a`
            # calculate derivative numerically
            Wx = Wx[:, (n1 - 4):(n1 + N + 4)]
            w = phase_cwt_num(Wx, dt, difforder, gamma)
        return Wx, w

    t, difforder, nv = _process_args(x, t, fs, nv, difftype, difforder, squeezing)

    N = len(x)
    N_up, n1, n2 = p2up(N)
    rpadded = (difftype == 'numerical')

    dt = t[1] - t[0]  # sampling period, assuming uniform spacing
    scales, cwt_scaletype, *_ = process_scales(scales, N, nv=nv, get_params=True)
    # l1_norm=False to spare a multiplication; for SSWT L1 & L2 are exactly same
    # anyway since we're inverting CWT over time-frequency plane
    Wx, scales, _, dWx = cwt(x, wavelet, scales=scales, dt=dt, l1_norm=False,
                             derivative=True, padtype=padtype, rpadded=rpadded)

    gamma = gamma or np.sqrt(EPS)
    Wx, w = _phase_transform(Wx, dWx, gamma, n1, dt, difftype, difforder)

    if ssq_freqs is None:
        # default to same scheme used by `scales`
        ssq_freqs = cwt_scaletype
    # calculate the synchrosqueezed frequency decomposition
    Tx, fs = ssqueeze(Wx, w, scales, t, ssq_freqs, transform='cwt',
                      squeezing=squeezing)

    if difftype == 'numerical':
        Wx = Wx[:, 4:-4]
        w  = w[:,  4:-4]
        Tx = Tx[:, 4:-4]
    return Tx, fs, Wx, scales, w


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

    Css = adm_ssq(wavelet)  # admissibility coefficient
    # *2 per analytic wavelet & taking real part; Theorem 4.5 [2]
    x *= (2 / Css)
    return x
