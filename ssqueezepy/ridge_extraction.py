# -*- coding: utf-8 -*-
"""Authors: David Bondesson, John Muradeli

Ridge extraction from time-frequency representations (STFT, CWT, synchrosqueezed).
"""
import numpy as np
from numba import jit, prange
from .utils import EPS32, EPS64


def extract_ridges(Tf, scales, penalty=2., n_ridges=1, bw=15, transform='cwt',
                   get_params=False, parallel=True):
    """Tracks time-frequency ridges by performing forward-backward ridge tracking
    algorithm, based on ref [1] (a version of Eq. III.4).

    Also see: https://www.mathworks.com/help/signal/ref/tfridge.html

    # Arguments:
        Tf: np.ndarray
            Complex time-frequency representation.

        scales:
            Frequency scales to calculate distance penalty term.

        penalty: float
            Value to penalize frequency jumps; multiplies the square of change
            in frequency. Trialworthy values: 0.5, 2, 5, 20, 40. Higher reduces
            odds of a ridge derailing to noise, but makes harder to track fast
            frequency changes.

        n_ridges: int
            Number of ridges to be calculated.

        bw: int
            Decides how many bins will be subtracted around max energy frequency
            bins when extracting multiple ridges (2 is standard for ssq'd).
            See "bw selection".

        transform: str['cwt', 'stft']
            Treats `scales` logarithmically if 'cwt', else linearly.
            `ssq_cwt` & `ssq_stft` are still 'cwt' & 'stft'.

        get_params: bool (default False)
            Whether to also compute and return `ridge_f` & `ridge_f`.

        parallel: bool (default True)
            Whether to use parallelized JIT code; runs faster on some input sizes.

    # Returns
        ridge_idxs: np.ndarray [n_timeshifts x n_ridges]
            Indices for maximum frequency ridge(s).
        ridge_f: np.ndarray [n_timeshifts x n_ridges]
            Quantities corresponding to extracted ridges:
                - STFT: frequencies
                - CWT: scales
        ridge_e: np.ndarray [n_timeshifts x n_ridges]
            Energies corresponding to extracted ridges.

    **bw selection**

    When a component is extracted, a region around it (a number of bins above
    and below the ridge) is zeroed and no longer affects next ridge's extraction.
        - higher: more bins subtracted, lesser chance of selecting the same
        component as the ridge.
        - lower:  less bins subtracted, lesser chance of dropping an unrelated
        component before the component is considered.
        - In general, set higher if more `scales` (or greater `nv`), or lower
        frequency resolution:
            - cwt:  `wavelets.freq_resolution(wavelet, N, nondim=False)`
            - stft: `utils.window_resolution(window)`
            - `N = utils.p2up(len(x))[0]`

    # References
        1. On the extraction of instantaneous frequencies from ridges in
        time-frequency representations of signals.
        D. Iatsenko, P. V. E. McClintock, A. Stefanovska.
        https://arxiv.org/pdf/1310.7276.pdf
    """
    def generate_penalty_matrix(scales, penalty):
        """Penalty matrix describes all potential penalties of  jumping from
        current frequency (first axis) to one or several new frequencies (second
        axis)

        `scales`: frequency scale vector from time-freq transform
        `penalty`: user-set penalty for freqency jumps (standard = 1.0)
        """
        # subtract.outer(A, B) = [[A[0] - B[0], A[0] - B[1], ...],
        #                         [A[1] - B[0], A[1] - B[1], ...],]
        dist_matrix = penalty * np.subtract.outer(scales, scales)**2
        return dist_matrix.squeeze()

    def fw_bw_ridge_tracking(energy_to_track, penalty_matrix, eps):
        """Calculates acummulated penalty in forward (t=end...0) followed by
        backward (t=end...0) direction

        `energy`: squared abs time-frequency transform
        `penalty_matrix`: pre calculated penalty for all potential jumps between
                          two frequencies

        Returns: `ridge_idxs_fw_bw`: estimated forward backward frequency
                                     ridge indices
        """
        (penalized_energy_fw, ridge_idxs_fw
         ) = _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix,
                                            parallel)
        # backward calculation of frequency ridge (min log negative energy)
        ridge_idxs_fw_bw = _accumulated_penalty_energy_bw(
            energy_to_track, penalty_matrix, penalized_energy_fw,
            ridge_idxs_fw, eps, parallel)

        return ridge_idxs_fw_bw

    eps   = EPS64      if Tf.dtype == np.complex128 else EPS32
    dtype = np.float64 if Tf.dtype == np.complex128 else np.float32
    scales, eps, penalty = [np.asarray(x, dtype=dtype)
                            for x in (scales, eps, penalty)]

    scales_orig = scales.copy()
    scales = (np.log(scales) if transform == 'cwt' else
              scales).squeeze()
    energy = np.abs(Tf)**2
    n_timeshifts = Tf.shape[1]

    ridge_idxs = np.zeros((n_timeshifts, n_ridges), dtype=int)
    if get_params:
        ridge_f = np.zeros((n_timeshifts, n_ridges), dtype=dtype)
        ridge_e = np.zeros((n_timeshifts, n_ridges), dtype=dtype)

    penalty_matrix = generate_penalty_matrix(scales, penalty)

    for i in range(n_ridges):
        energy_max = energy.max(axis=0)
        energy_neg_log_norm = -np.log(energy / energy_max + eps)

        ridge_idxs[:, i] = fw_bw_ridge_tracking(energy_neg_log_norm,
                                                penalty_matrix, eps)
        if get_params:
            ridge_f[:, i] = scales_orig[ridge_idxs[:, i]]
            ridge_e[:, i] = energy[     ridge_idxs[:, i], range(n_timeshifts)]

        for time_idx in range(n_timeshifts):
            ridx = ridge_idxs[time_idx, i]
            energy[int(ridx - bw):int(ridx + bw), time_idx] = 0

    return ((ridge_idxs, ridge_f, ridge_e) if get_params else
            ridge_idxs)


def _accumulated_penalty_energy_fw(energy_to_track, penalty_matrix, parallel):
    """Calculates acummulated penalty in forward direction (t=0...end).

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre-calculated penalty for all potential jumps between
                      two frequencies

    # Returns:
        `penalized_energy`: new energy with added forward penalty
        `ridge_idxs`: calculated initial ridge with only forward penalty
    """
    penalized_energy = energy_to_track.copy()
    fn = (__accumulated_penalty_energy_fwp if parallel else
          __accumulated_penalty_energy_fw)
    fn(penalized_energy, penalty_matrix)
    ridge_idxs = np.unravel_index(np.argmin(penalized_energy, axis=0),
                                  penalized_energy.shape)[1]
    return penalized_energy, ridge_idxs


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_fw(penalized_energy, penalty_matrix):
    for idx_time in range(1, penalized_energy.shape[1]):
        for idx_freq in range(0, penalized_energy.shape[0]):
            penalized_energy[idx_freq, idx_time
                             ] += np.amin(penalized_energy[:, idx_time - 1] +
                                          penalty_matrix[idx_freq, :])

@jit(nopython=True, cache=True, parallel=True)
def __accumulated_penalty_energy_fwp(penalized_energy, penalty_matrix):
    for idx_time in range(1, penalized_energy.shape[1]):
        for idx_freq in prange(0, penalized_energy.shape[0]):
            penalized_energy[idx_freq, idx_time
                             ] += np.amin(penalized_energy[:, idx_time - 1] +
                                          penalty_matrix[idx_freq, :])


def _accumulated_penalty_energy_bw(energy_to_track, penalty_matrix,
                                   penalized_energy_fw, ridge_idxs_fw,
                                   eps, parallel):
    """Calculates acummulated penalty in backward direction (t=end...0)

    `energy_to_track`: squared abs time-frequency transform
    `penalty_matrix`: pre calculated penalty for all potential jumps between
                      two frequencies
    `ridge_idxs_fw`: calculated forward ridge

    Returns: `ridge_idxs_fw`: new ridge with added backward penalty, int array
    """
    pen_e = penalized_energy_fw
    e = energy_to_track
    fn = (__accumulated_penalty_energy_bwp if parallel else
          __accumulated_penalty_energy_bw)
    fn(e, penalty_matrix, pen_e, ridge_idxs_fw, eps)
    return np.asarray(ridge_idxs_fw).astype(int)


@jit(nopython=True, cache=True)
def __accumulated_penalty_energy_bw(e, penalty_matrix, pen_e, ridge_idxs_fw, eps):
    for idx_time in range(e.shape[1] - 2, -1, -1):
        val = (pen_e[ridge_idxs_fw[idx_time + 1], idx_time + 1] -
               e[    ridge_idxs_fw[idx_time + 1], idx_time + 1])
        for idx_freq in range(e.shape[0]):
            new_penalty = penalty_matrix[ridge_idxs_fw[idx_time + 1], idx_freq]

            if abs(val - (pen_e[idx_freq, idx_time] + new_penalty)) < eps:
                ridge_idxs_fw[idx_time] = idx_freq

@jit(nopython=True, cache=True, parallel=True)
def __accumulated_penalty_energy_bwp(e, penalty_matrix, pen_e, ridge_idxs_fw,
                                     eps):
    # adding `prange` to `tidx` makes whole computation much faster (x3-4),
    # but breaks it on *some* inputs (unpredictably)
    for tidx in range(e.shape[1] - 1):
        # `prange` only supports a step size of 1, so we use a trick
        # actually can't `prange` `tidx`, not thread-safe
        idx_time = (e.shape[1] - 2) - tidx
        val = (pen_e[ridge_idxs_fw[idx_time + 1], idx_time + 1] -
               e[    ridge_idxs_fw[idx_time + 1], idx_time + 1])
        for idx_freq in prange(e.shape[0]):
            new_penalty = penalty_matrix[ridge_idxs_fw[idx_time + 1], idx_freq]

            if abs(val - (pen_e[idx_freq, idx_time] + new_penalty)) < eps:
                ridge_idxs_fw[idx_time] = idx_freq
