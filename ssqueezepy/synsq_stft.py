# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
from .utils import wfiltfn
from .stft_transforms import stft_fwd, phase_stft
from .wavelet_transforms import synsq_squeeze
from quadpy import quad as quadgk

PI = np.pi


def synsq_stft_fwd(t, x, opts={}):
    """Calculates the STFt synchrosqueezing transform of vector `x`, with
    samples taken at times given in vector `t`. This implements the algorithm
    described in Sec. III of [1].
    
    # Arguments:
        t: np.ndarray. Vector of times samples are taken 
           (e.g. np.linspace(0, 1, n))
        x: np.ndarray. Vector of signal samples (e.g. x = np.cos(20 * PI * t))
        opts: dict. Options:
            'type': str. Type of wavelet (see `wfiltfn`)
            's', 'mu': float. Wavelet parameters (see `wfiltfn`)
            'gamma': float. Wavelet hard thresholding value 
                     (see `cwt_freq_direct`)
    
    # Returns:
        Tx: synchrosqueezed output of `x` (columns associated with time `t`)
        fs: frequencies associated with rows of `Tx`
        Sx: STFT of `x` (see `stft_fw`)
        Sfs: frequencies associated with rows of `Sx`
        w: phase transform of `Sx`
    """
    def _validate_spacing_uniformity(t):
        if np.any([(np.diff(t, 2) / (t[-1] - t[0]) > 1e-5)]):
            raise Exception("Time vector `t` must be uniformly sampled.")

    _validate_spacing_uniformity(t)

    opts['type']    = opts.get('type', 'bump')
    opts['rpadded'] = opts.get('rpadded', False)
    
    dt = t[1] - t[0]
    
    # Calculate the modified STFT, using window of opts['winlen']
    # in frequency domain
    opts['stfttype'] = 'modified'
    Sx, Sfs, dSx = stft_fwd(x, dt, opts) 
    
    w = phase_stft(Sx, dSx, Sfs, t, opts)
    
    # Calculate the synchrosqueezed frequency decomposition
    # The parameter alpha from reference [2] is given by Sfs[1] - Sfs[0]
    opts['transform'] = 'STFT'
    Tx, fs = synsq_squeeze(Sx, w, t, None, opts)
    
    return Tx, fs, Sx, Sfs, w, dSx

    
def synsq_stft_inv(Tx, fs, opts, Cs=None, freqband=None):
    """Inverse STFT synchrosqueezing transform of `Tx` with associated
    frequencies in `fs` and curve bands in time-frequency plane
    specified by `Cs` and `freqband`. This implements Eq. 5 of [1].
    
    # Arguments:
        Tx: np.ndarray. Synchrosqueeze-transformed `x` (see `synsq_cwt_fwd`).
        fs: np.ndarray. Frequencies associated with rows of Tx.
            (see `synsq_cwt_fwd`).
        opts. dict. Options:
            'type': type of wavelet used in `synsq_cwt_fwd` (required).
            
            other wavelet options ('mu', 's') should also match those
            used in `synsq_cwt_fwd`
            'Cs': (optional) curve centerpoints
            'freqs': (optional) curve bands
    
    # Returns:
        x: components of reconstructed signal, and residual error
    
    Example:
        Tx, fs = synsq_cwt_fwd(t, x, 32)  # synchrosqueezing
        Txf = synsq_filter_pass(Tx, fs, -np.inf, 1)  # pass band filter
        xf  = synsq_cwt_inv(Txf, fs)  # filtered signal reconstruction
    """
    Cs       = Cs       or np.ones((Tx.shape[1], 1))
    freqband = freqband or Tx.shape[0]
    
    windowfunc = wfiltfn(opts['type'], opts, derivative=False)
    inf_lim = 1000  # quadpy can't handle np.inf limits
    C = quadgk(lambda x: windowfunc(x)**2, -inf_lim, inf_lim)
    if opts['type'] == 'bump':
        C *= 0.8675

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
        # NOTE: transposed + flattened to match MATLAB's 'linear indices'
        UpperCs[np.where(Cs[:, n].T.flatten() < 1)] = 1
        LowerCs[np.where(Cs[:, n].T.flatten() < 1)] = 2
        
        for m in range(Tx.shape[1]):
            idxs = slice(LowerCs[m] - 1, UpperCs[m])
            TxMask[idxs, m] = Tx[idxs, m]
            TxRemainder[idxs, m] = 0
        x[:, n] = 1 / (PI * C) * np.sum(np.real(TxMask),      axis=0).T

    x[:, n + 1] = 1 / (PI * C) * np.sum(np.real(TxRemainder), axis=0).T
    x = x.T

    return x
