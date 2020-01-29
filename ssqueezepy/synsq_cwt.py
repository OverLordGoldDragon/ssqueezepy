# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
from utils import est_riskshrink_thresh, p2up, synsq_adm
from wavelet_transforms import phase_cwt, phase_cwt_num
from wavelet_transforms import cwt_fwd, synsq_squeeze


def synsq_cwt_fwd(x, t=None, fs=None, nv=32, opts=None):
    """Calculates the synchrosqueezing transform of vector `x`, with samples
    taken at times given in vector `t`. Uses `nv` voices. Implements the
    algorithm described in Sec. III of [1].
    
    # Arguments:
        x: np.ndarray. Vector of signal samples (e.g. x = np.cos(20 * np.pi * t))
        t: np.ndarray / None. Vector of times samples are taken 
           (e.g. np.linspace(0, 1, n)). If None, defaults to np.arange(len(x)).
           Overrides `fs` if not None.
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
        Wx: Wavelet transform of `x` (see `cwt_fwd`)
        Wx_scales: scales associated with rows of `Wx`.
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
    def _get_opts(opts):
        opts_default = {'type': 'morlet',
                        'difftype': 'direct',
                        'gamma': None}
        if opts is None:
            opts = opts_default
        else:
            opts = {}
            for opt_name in opts_default:
                opts[opt_name] = opts.get(opt_name, opts_default[opt_name])
        return opts
        
    def _wavelet_transform(x, nv, dt, opts):
        N = len(x)
        N_up, n1, n2 = p2up(N)

        if opts['difftype'] == 'direct':
            # calculate derivative directly in the wavelet domain
            # before taking wavelet transform
            opts['rpadded'] = 0

            Wx, Wx_scales, dWx, _ = cwt_fwd(x, opts['type'], nv, dt, opts)
            w = phase_cwt(Wx, dWx, opts)

        elif opts['difftype'] == 'phase':
            # take derivative of unwrapped CWT phase
            # directly in phase transform
            opts['rpadded'] = 0
            
            Wx, Wx_scales, _ = cwt_fwd(x, opts['type'], nv, dt, opts)
            w = phase_cwt(Wx, None, opts)
        else:
            # calculate derivative numerically after calculating wavelet
            # transform. This requires less memory and is more accurate
            # for lesser `a`.
            opts['rpadded'] = 1
            
            Wx, Wx_scales, _ = cwt_fwd(x, opts['type'], nv, dt, opts)
            Wx = Wx[:, (n1 - 5 + 1):(n1 + N + 3)]
            w = phase_cwt_num(Wx, dt, opts)

        return Wx, w, Wx_scales, opts

    def _validate_spacing_uniformity(t):
        if np.any([(np.diff(t, 2) / (t[-1] - t[0]) > 1e-5)]):
            raise Exception("Time vector `t` must be uniformly sampled.")

    if t is None:
        fs = fs or 1.
        t = np.linspace(0., len(x) / fs, len(x))
    else:
        _validate_spacing_uniformity(t)
    opts = _get_opts(opts)

    dt = t[1] - t[0]  # sampling period, assuming regular spacing
    Wx, w, Wx_scales, opts = _wavelet_transform(x, nv, dt, opts)

    if opts['gamma'] is None:
        opts['gamma'] = est_riskshrink_thresh(Wx, nv)
    
    # calculate the synchrosqueezed frequency decomposition
    opts['transform'] = 'CWT'  
    Tx, fs = synsq_squeeze(Wx, w, t, nv, opts)
    
    if opts['difftype'] == 'numerical':
        Wx = Wx[:, (3 + 1):(len(Wx) - 1 - 5)]
        w  = w[:   (3 + 1):(len(w)  - 1 - 5)]
        Tx = Tx[:, (3 + 1):(len(Tx) - 1 - 5)]
    
    return Tx, fs, Wx, Wx_scales, w


def synsq_cwt_inv(Tx, fs, opts={}, Cs=None, freqband=None):  #TODO Arguments
    """Inverse synchrosqueezing transform of `Tx` with associated frequencies
    in `fs` and curve bands in time-frequency plane specified by `Cs` and
    `freqband`. This implements Eq. 5 of [1].
    
    # Arguments:
        Tx: np.ndarray. Synchrosqueeze-transformed `x` (see `synsq_cwt_fwd`).
        fs: np.ndarray. Frequencies associated with rows of Tx.
            (see `synsq_cwt_fwd`).
        opts: dict. Options (see `synsq_cwt_fwd`):
            'type': type of wavelet used in `synsq_cwt_fwd`
            
            other wavelet options ('mu', 's') should also match
            those used in `synsq_cwt_fwd`
            'Cs': (optional) curve centerpoints
            'freqs': (optional) curve bands
    
    # Returns:
        x: components of reconstructed signal, and residual error
    
    # Example:
        Tx, fs = synsq_cwt_fwd(t, x, 32)  # synchrosqueeizing
        Txf = synsq_filter_pass(Tx,fs, -np.inf, 1)  # pass band filter
        xf = synsq_cwt_inv(Txf, fs)  # filtered signal reconstruction
        
    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013. 
    """
    opts = opts or {'type': 'morlet'}
    Cs = Cs or np.ones((Tx.shape[1], 1))
    freqband = Tx.shape[0]

    # Find the admissibility coefficient Cpsi
    Css = synsq_adm(opts['type'], opts)
    
    # Invert Tx around curve masks in the time-frequency plane to recover
    # individual components; last one is the remaining signal
    # Integration over all frequencies recovers original signal
    # Factor of 2 is because real parts contain half the energy
    x = np.zeros((Cs.shape[0], Cs.shape[1] + 1))
    TxRemainder = Tx
    
    for n in range(Cs.shape[1]):
        TxMask = np.zeros(Tx.shape)
        UpperCs = min(max(Cs[:, n] + freqband[:, n], 1), len(fs))
        LowerCs = min(max(Cs[:, n] - freqband[:, n], 1), len(fs))
        
        # Cs==0 corresponds to no curve at that time, so this removes
        # such points from the inversion
        UpperCs[np.where(Cs[:, n] < 1)] = 1
        LowerCs[np.where(Cs[:, n] < 1)] = 2
        for m in range(Tx.shape[1]):
            idxs = slice(LowerCs[m] - 1, UpperCs[m])
            TxMask[idxs, m] = Tx[idxs, m]
            TxRemainder[idxs, m] = 0
            
        # Due to linear discretization of integral in log(fs),
        # this becomes a simple normalized sum
        x[:, n] = (1 / Css) * np.sum(np.real(TxMask), axis=0).T

    x[:, n + 1] = (1 / Css) * np.sum(np.real(TxRemainder), axis=0).T
    x = x.T
    
    return x
