import numpy as np
from .utils import WARN, est_riskshrink_thresh, p2up, adm_ssq, process_scales
from .ssqueezing import ssqueeze, phase_cwt, phase_cwt_num
from ._cwt import cwt


def ssq_cwt(x, wavelet='morlet', scales='log', t=None, freqscale=None,
            fs=None, nv=None, difftype='direct', difforder=None,
            padtype='symmetric', squeezing='full', gamma=None):
    """Calculates the synchrosqueezed continuous wavelet transform of `x`,
    with samples taken at times given in `t`. Uses `nv` voices. Implements the
    algorithm described in Sec. III of [1].

    # Arguments:
        x: np.ndarray. Vector of signal samples (e.g. x = np.cos(20 * np.pi * t))
        wavelet: wavelet
        scales: CWT scales. np.ndarray or ('log', 'linear')
                !!! beware of scales='linear'; bad current default scheme for
                capturing low frequencies for sequences longer than 2048.
                Recommended scales='log' with freqscale='linear' instead.
        t: np.ndarray / None. Vector of times samples are taken
           (e.g. np.linspace(0, 1, n)). If None, defaults to np.arange(len(x)).
           Overrides `fs` if not None.
        freqscale: synchrosqueezing plane scales. np.ndarray or ('log', 'linear')
        fs: float. Sampling frequency of `x`; overridden by `t`, if provided.
        nv: int. Number of voices. Recommended 32 or 64 by [1].
        opts: dict. Options specifying how synchrosqueezing is computed.
           'type': str. type of wavelet. See `wfiltfn` docstring.
           'gamma': float / None. Wavelet hard thresholding value. If None,
                    is estimated automatically.
           'difftype': str. 'direct', 'phase', or 'numerical' differentiation.
                    'numerical' uses MEX differentiation, which is faster and
                    uses less memory, but may be less accurate.

    # Returns:
        Tx: Synchrosqueeze-transformed `x`, columns associated w/ `t`
        fs: Frequencies associated with rows of `Tx`.
        Wx: Wavelet transform of `x` (see `cwt`)
        scales: scales associated with rows of `Wx`.
        w: Phase transform for each element of `Wx`.

    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.

        2. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms:
        an empricial mode decomposition-like tool",
        Applied and Computational Harmonic Analysis, 30(2):243-261, 2011.
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
            fs = fs or 1.  # TODO change default to 1/len(x)?
            t = np.linspace(0., len(x) / fs, len(x))
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
    scales, _freqscale, *_ = process_scales(scales, N, nv=nv, get_params=True)
    Wx, scales, dWx, _ = cwt(x, wavelet, scales=scales, dt=dt, l1_norm=False,
                             padtype=padtype, rpadded=rpadded)

    Wx, w = _phase_transform(Wx, dWx, gamma, n1, dt, difftype, difforder)

    gamma = gamma or est_riskshrink_thresh(Wx, nv)

    if freqscale is None:
        # default to same scheme used by `scales`
        # !!! scales='linear' not recommended for len(x)>2048; see docstr
        freqscale = _freqscale
    # calculate the synchrosqueezed frequency decomposition
    Tx, fs = ssqueeze(Wx, w, scales, t, freqscale, transform='cwt',
                      squeezing=squeezing)

    if difftype == 'numerical':
        Wx = Wx[:, 4:-4]
        w  = w[:,  4:-4]
        Tx = Tx[:, 4:-4]
    return Tx, fs, Wx, scales, w


def issq_cwt(Tx, wavelet, Cs=None, freqband=None):
    """Inverse synchrosqueezing transform of `Tx` with associated frequencies
    in `fs` and curve bands in time-frequency plane specified by `Cs` and
    `freqband`. This implements Eq. 15 of [1].

    # Arguments:
        Tx: np.ndarray. Synchrosqueeze-transformed `x` (see `synsq_cwt`).
        fs: np.ndarray. Frequencies associated with rows of Tx.
            (see `synsq_cwt`).
        opts: dict. Options (see `synsq_cwt`):
            'type': type of wavelet used in `synsq_cwt`

            other wavelet options ('mu', 's') should also match
            those used in `synsq_cwt`
            'Cs': (optional) curve centerpoints
            'freqs': (optional) curve bands

    # Returns:
        x: components of reconstructed signal, and residual error

    # Example:
        Tx, fs = synsq_cwt(t, x, 32)  # synchrosqueeizing
        Txf = synsq_filter_pass(Tx, fs, -np.inf, 1)  # pass band filter
        xf = synsq_cwt_inv(Txf, fs)  # filtered signal reconstruction

    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.

        2. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.
    """
    def _invert_components(Tx, Cs, freqband):
        # Invert Tx around curve masks in the time-frequency plane to recover
        # individual components; last one is the remaining signal
        x = np.zeros((Cs.shape[1] + 1, Cs.shape[0]))
        TxRemainder = Tx.copy()

        for n in range(Cs.shape[1]):
            TxMask = np.zeros(Tx.shape, dtype='complex128')
            UpperCs = np.clip(Cs[:, n] + freqband[:, n], 0, len(Tx))
            LowerCs = np.clip(Cs[:, n] - freqband[:, n], 0, len(Tx))

            # Cs==-1 denotes no curve at that time,
            # removing such points from inversion
            UpperCs[np.where(Cs[:, n] == -1)] = 0
            LowerCs[np.where(Cs[:, n] == -1)] = 1
            for m in range(Tx.shape[1]):
                idxs = slice(LowerCs[m], UpperCs[m] + 1)
                TxMask[idxs, m] = Tx[idxs, m]
                TxRemainder[idxs, m] = 0
            x[n] = TxMask.real.sum(axis=0).T

        x[n + 1] = TxRemainder.real.sum(axis=0).T
        return x

    def _process_args(Cs, freqband):
        if (Cs is None and freqband is None):
            return None, None, True
        if Cs.ndim == 1:
            Cs = Cs.reshape(-1, 1)
        if freqband.ndim == 1:
            freqband = freqband.reshape(-1, 1)
        Cs = Cs.astype('int32')
        freqband = freqband.astype('int32')
        return Cs, freqband, False

    Cs, freqband, full_inverse = _process_args(Cs, freqband)

    if full_inverse:
        # Integration over all frequencies recovers original signal
        x = Tx.real.sum(axis=0)
    else:
        x = _invert_components(Tx, Cs, freqband)

    Css = adm_ssq(wavelet)  # admissibility coefficient
    # *2 per analytic wavelet & taking real part; Theorem 4.5 [2]
    x *= (2 / Css)
    return x
