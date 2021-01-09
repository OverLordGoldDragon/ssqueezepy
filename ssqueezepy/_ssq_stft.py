# -*- coding: utf-8 -*-
"""NOT FOR USE; will be ready for v0.6.0"""
import numpy as np
from .wavelets import Wavelet
from ._stft import stft, phase_stft
from .ssqueezing import ssqueeze
from scipy.integrate import quad as quadgk

pi = np.pi


def ssq_stft(x, window=None, n_fft=None, win_len=None, hop_len=1,
             dt=1, padtype='reflect', modulated=True, squeezing='sum'):
    """Calculates the STFt synchrosqueezing transform of vector `x`, with
    samples taken at times given in vector `t`. This implements the algorithm
    described in Sec. III of [1].

    # Arguments:
        t: np.ndarray. Vector of times samples are taken
           (e.g. np.linspace(0, 1, n))
        x: np.ndarray. Vector of signal samples (e.g. x = np.cos(20 * pi * t))
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
    # Calculate the modified STFT, using window of opts['winlen']
    # in frequency domain
    Sx, Sfs, dSx = stft(x, window, n_fft=n_fft, win_len=win_len,
                        hop_len=hop_len, dt=dt, padtype=padtype,
                        modulated=modulated)

    w = phase_stft(Sx, dSx, Sfs, N=len(x))

    # Calculate the synchrosqueezed frequency decomposition
    # The parameter alpha from reference [2] is given by Sfs[1] - Sfs[0]
    Tx, fs = ssqueeze(Sx, w, transform='stft', scales='linear',
                      squeezing=squeezing)

    return Tx, fs, Sx, Sfs, w, dSx


def issq_stft(Tx, fs, opts, Cs=None, freqband=None):
    """Inverse STFT synchrosqueezing transform of `Tx` with associated
    frequencies in `fs` and curve bands in time-frequency plane
    specified by `Cs` and `freqband`. This implements Eq. 5 of [1].

    # Arguments:
        Tx: np.ndarray. Synchrosqueeze-transformed `x` (see `synsq_cwt`).
        fs: np.ndarray. Frequencies associated with rows of Tx.
            (see `synsq_cwt`).
        opts. dict. Options:
            'type': type of wavelet used in `synsq_cwt` (required).

            other wavelet options ('mu', 's') should also match those
            used in `synsq_cwt`
            'Cs': (optional) curve centerpoints
            'freqs': (optional) curve bands

    # Returns:
        x: components of reconstructed signal, and residual error

    Example:
        Tx, fs = synsq_cwt(t, x, 32)  # synchrosqueezing
        Txf = synsq_filter_pass(Tx, fs, -np.inf, 1)  # pass band filter
        xf  = synsq_cwt_inv(Txf, fs)  # filtered signal reconstruction
    """
    Cs       = Cs       or np.ones((Tx.shape[1], 1))
    freqband = freqband or Tx.shape[0]

    windowfunc = Wavelet((opts['type'], opts))
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
        x[:, n] = 1 / (pi * C) * np.sum(np.real(TxMask),      axis=0).T

    x[:, n + 1] = 1 / (pi * C) * np.sum(np.real(TxRemainder), axis=0).T
    x = x.T

    return x
