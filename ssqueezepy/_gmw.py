# -*- coding: utf-8 -*-
"""Generalized Morse Wavelets.

For complete functionality, utility functions have been ported from jLab, and
largely validated to match jLab's behavior. jLab tests not ported.
"""
import numpy as np
from numpy.fft import ifft
from numba import jit
from scipy.special import (gamma   as gamma_fn,
                           gammaln as gammaln_fn)
from .algos import nCk
from .wavelets import _xifn, _process_params_dtype
from .configs import gdefaults, USE_GPU, IS_PARALLEL
from .utils.backend import torch
from .utils import backend as S

pi = np.pi


#### Base wavelets (`K=1`) ###################################################
def gmw(gamma=None, beta=None, norm=None, order=None, centered_scale=None,
        dtype=None):
    """Generalized Morse Wavelets. Returns function which computes GMW in the
    frequency domain.

    Assumes `beta != 0`; for full functionality use `_gmw.morsewave`.
    Unlike `morsewave`, works with scales rather than frequencies.

    Note that function for `norm='energy'` does *not* rescale freq-domain wavelet
    per `sqrt(scale)`, for consistency with `ssqueezepy.wavelets`.
    See `_gmw.compute_gmw` for code computing freq- and time-domain wavelets
    as arrays with proper scaling in.

    An overview: https://overlordgolddragon.github.io/generalized-morse-wavelets/
    Interactive: https://www.desmos.com/calculator/4gcaeqidxd (bandpass)
                 https://www.desmos.com/calculator/zfxnblqh8p (energy)

    # Arguments
        gamma, beta: float > 0, float > 0
            GMW parameters. See `help(_gmw.morsewave)`.

        norm: str['energy', 'bandpass']
            Normalization to use:
                'energy': L2 norm, keeps time-domain wavelet's energy at unity
                for all `freqs`, i.e. `sum(abs(psi)**2) == 1`.
                'bandpass': L1 norm, keeps freq-domain wavelet's peak value at 2
                for all `freqs`, i.e. `max(psih) == 2`, `w[argmax(psih)] == wc`.

            Additionally see `help(_gmw.morsewave)`.

        order: int (default 1)
            Order of the wavelet. `k+1`-th order wavelet is orthogonal to `k`-th.
            `k=0` will call a different but equivalent function for simpler code
            and compute efficiency.

        centered_scale: bool (default False)
            Unlike other `ssqueezepy.wavelets`, by default `scale=1` in
            `morsewave` (i.e. `freqs=1`) computes the wavelet at (peak) center
            frequency. This ensures exact equality between `scale` and
            `1 / center_frequency`, by multiplying input radians `w` by peak
            center freq.

            False by default for consistency with other `ssqueezepy` wavelets.

        dtype: str / type (np.dtype) / None
            See `help(wavelets.Wavelet)`.

    # Returns
        psihfn: function
            Function that computes GMWs, taking `w` (radian frequency)
            as argument.

    # Usage
        wavelet = gmw(3, 60)
        wavelet = Wavelet('gmw')
        wavelet = Wavelet(('gmw', {'beta': 60}))
        Wx, *_  = cwt(x, 'gmw')

    # Correspondence with Morlet
        Following pairs yield ~same frequency resolution, which is ~same
        time-frequency resolution for `mu > 5`, assuming `gamma=3` for all:
            `mu`, `beta`
           (1.70, 1.00),
           (3.00, 3.00),
           (4.00, 5.15),
           (6.00, 11.5),
           (8.00, 21.5),
           (10.0, 33.5),
           (12.0, 48.5),
           (13.4, 60.0),
        The default `beta=12` is hence to closely match Morlet's default `mu=6.`.

    # vs Morlet
        Differences grow significant when seeking excellent time localization
        (low `mu`, <4), where Morlet's approximate analyticity breaks down and
        negative frequencies are leaked, whereas GMW remains exactly analytic,
        with vanishing moments toward dc bin. Else, the two don't behave
        noticeably different for `gamma=3`.

    # References
        [1] Generalized Morse Wavelets. S. C. Olhede, A. T. Walden. 2002.
        https://spiral.imperial.ac.uk/bitstream/10044/1/1150/1/
        OlhedeWaldenGenMorse.pdf

        [2] Generalized Morse Wavelets as a Superfamily of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2012.
        https://sci-hub.st/10.1109/TSP.2012.2210890

        [3] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [4] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
    """
    _check_args(gamma=gamma, beta=beta, norm=norm, order=order)
    kw = gdefaults('_gmw.gmw', gamma=gamma, beta=beta, norm=norm, order=order,
                   centered_scale=centered_scale, dtype=dtype, as_dict=True)
    norm, k = kw.pop('norm'), kw.pop('order')
    if norm == 'energy'and dtype in ('float32', np.float32):
        raise ValueError("`norm='energy'` w/ `dtype='float32'` is unsupported; "
                         "use 'float64' instead.")

    l1_fn, l2_fn = ((gmw_l1, gmw_l2) if k == 0 else
                    (gmw_l1_k, gmw_l2_k))
    if k > 0:
        kw['k'] = k
    return (l1_fn(**kw) if norm == 'bandpass' else
            l2_fn(**kw))


def compute_gmw(N, scale, gamma=3, beta=60, time=False, norm='bandpass',
                order=0, centered_scale=False, norm_scale=True, dtype=None):
    """Evaluates GMWs, returning as arrays. See `help(_gmw.gmw)` for full docs.

    # Arguments
        N: int > 0
            Number of samples to compute.

        scale: float > 0
            Scale at which to sample the freq-domain wavelet: `psih(s * w)`.

        gamma, beta, norm, order:
            See `help(_gmw.gmw)`.

        time: bool (default False)
            Whether to compute the time-domain wavelet, `psi`.

        centered_scale: bool (default False)
            See `help(_gmw.gmw)`.

        norm_scale: bool (default True)
            Whether to rescale as `sqrt(s) * psih(s * w)` for the `norm='energy'`
            case (no effect with `norm='bandpass'`).

    # Returns
        psih: np.ndarray [N]
            Frequency-domain wavelet.
        psi: np.ndarray [N]
            Time-domain wavelet, returned if `time=True`.
    """
    _check_args(gamma=gamma, beta=beta, norm=norm, scale=scale)
    gmw_fn = gmw(gamma, beta, norm, order, centered_scale, dtype)

    w = _xifn(scale, N)
    X = np.zeros(N)
    X[:N//2 + 1] = gmw_fn(w[:N//2 + 1])

    if norm == 'energy' and norm_scale:
        wc = morsefreq(gamma, beta)
        X *= (np.sqrt(wc * scale) if centered_scale else
              np.sqrt(scale))
    X[np.isinf(X) | np.isnan(X)] = 0.

    if time:
        Xr = X.copy()
        if N % 2 == 0:
            # https://github.com/jonathanlilly/jLab/issues/13
            Xr[N//2] /= 2
        x = ifft(Xr * (-1)**np.arange(N))

    return (X, x) if time else X


def gmw_l1(gamma=3., beta=60., centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, first order, L1(bandpass)-normalized.
    See `help(_gmw.gmw)`.
    """
    _check_args(gamma=gamma, beta=beta, allow_zerobeta=False)
    wc = morsefreq(gamma, beta)

    wcl = np.log(wc)
    gamma, beta, wc, wcl = _process_params_dtype(gamma, beta, wc, wcl, dtype=dtype)

    fn = _gmw_l1_gpu if USE_GPU() else (_gmw_l1_par if IS_PARALLEL() else _gmw_l1)
    if centered_scale:
        return lambda w: fn(S.atleast_1d(w * wc, dtype), gamma, beta, wc, wcl)
    else:
        return lambda w: fn(S.atleast_1d(w, dtype), gamma, beta, wc, wcl)

@jit(nopython=True, cache=True)
def _gmw_l1(w, gamma, beta, wc, wcl):
    # NOTE: numba.jit, unlike numpy & torch, will promote to float64 with
    # array float32 and scalar float64
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans
    return 2 * np.exp(- beta * wcl + wc**gamma
                      + beta * np.log(w) - w**gamma) * w_nonneg

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l1_par(w, gamma, beta, wc, wcl):
    # NOTE: numba.jit, unlike numpy & torch, will promote to float64 with
    # array float32 and scalar float64
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans
    return 2 * np.exp(- beta * wcl + wc**gamma
                      + beta * np.log(w) - w**gamma) * w_nonneg

def _gmw_l1_gpu(w, gamma, beta, wc, wcl):
    w_nonneg = (w >= 0)
    w *= w_nonneg
    return 2 * torch.exp(- beta * wcl + wc**gamma
                         + beta * torch.log(w) - w**gamma) * w_nonneg


def gmw_l2(gamma=3., beta=60., centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, first order, L2(energy)-normalized.
    See `help(_gmw.gmw)`.
    """
    _check_args(gamma=gamma, beta=beta, allow_zerobeta=False)
    wc = morsefreq(gamma, beta)
    r = (2*beta + 1) / gamma
    rgamma = gamma_fn(r)
    (gamma, beta, wc, r, rgamma
     ) = _process_params_dtype(gamma, beta, wc, r, rgamma, dtype=dtype)

    fn = _gmw_l2_gpu if USE_GPU() else (_gmw_l2_par if IS_PARALLEL() else _gmw_l2)
    if centered_scale:
        return lambda w: fn(S.atleast_1d(w * wc, dtype), gamma, beta, wc,
                            r, rgamma)
    else:
        return lambda w: fn(S.atleast_1d(w, dtype), gamma, beta, wc, r, rgamma)

@jit(nopython=True, cache=True)
def _gmw_l2(w, gamma, beta, wc, r, rgamma):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans
    return np.sqrt(2.*pi * gamma * 2.**r / rgamma
                   ) * w**beta * np.exp(-w**gamma) * w_nonneg

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l2_par(w, gamma, beta, wc, r, rgamma):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans
    return np.sqrt(2.*pi * gamma * 2.**r / rgamma
                   ) * w**beta * np.exp(-w**gamma) * w_nonneg

def _gmw_l2_gpu(w, gamma, beta, wc, r, rgamma):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans
    return torch.sqrt(2.*pi * gamma * 2.**r / rgamma
                      ) * w**beta * torch.exp(-w**gamma) * w_nonneg


def gmw_l1_k(gamma=3., beta=60., k=1, centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, `k`-th order, L1(bandpass)-normalized.
    See `help(_gmw.gmw)`.
    """
    _check_args(gamma=gamma, beta=beta, allow_zerobeta=False)

    wc = morsefreq(gamma, beta)
    k_consts = _gmw_k_constants(gamma, beta, k, norm='bandpass', dtype=dtype)
    gamma, beta, wc = _process_params_dtype(gamma, beta, wc, dtype=dtype)

    fn = (_gmw_l1_k_gpu if USE_GPU() else
          (_gmw_l1_k_par if IS_PARALLEL() else _gmw_l1_k))
    if centered_scale:
        return lambda w: fn(S.atleast_1d(w * wc, dtype), gamma, beta, wc,
                            k_consts)
    else:
        return lambda w: fn(S.atleast_1d(w, dtype), gamma, beta, wc, k_consts)

@jit(nopython=True, cache=True)
def _gmw_l1_k(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = np.zeros(w.shape, dtype=w.dtype)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * np.exp(- beta * np.log(wc) + wc**gamma
                      + beta * np.log(w)  - w**gamma) * w_nonneg

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l1_k_par(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = np.zeros(w.shape, dtype=w.dtype)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * np.exp(- beta * np.log(wc) + wc**gamma
                      + beta * np.log(w)  - w**gamma) * w_nonneg

def _gmw_l1_k_gpu(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = w.new_zeros(w.shape)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * torch.exp(- beta * torch.log(wc) + wc**gamma
                         + beta * torch.log(w)  - w**gamma) * w_nonneg


def gmw_l2_k(gamma=3., beta=60., k=1, centered_scale=False, dtype='float64'):
    """Generalized Morse Wavelets, `k`-th order, L2(energy)-normalized.
    See `help(_gmw.gmw)`.
    """
    _check_args(gamma=gamma, beta=beta, allow_zerobeta=False)

    wc = morsefreq(gamma, beta)
    k_consts = _gmw_k_constants(gamma, beta, k, norm='energy', dtype=dtype)
    gamma, beta, wc = _process_params_dtype(gamma, beta, wc, dtype=dtype)

    fn = (_gmw_l2_k_gpu if USE_GPU() else
          (_gmw_l2_k_par if IS_PARALLEL() else _gmw_l2_k))
    if centered_scale:
        return lambda w: fn(S.atleast_1d(w * wc, dtype), gamma, beta, wc,
                            k_consts)
    else:
        return lambda w: fn(S.atleast_1d(w, dtype), gamma, beta, wc, k_consts)

@jit(nopython=True, cache=True)
def _gmw_l2_k(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = np.zeros(w.shape, dtype=w.dtype)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * np.exp(beta * np.log(w) - w**gamma) * w_nonneg

@jit(nopython=True, cache=True, parallel=True)
def _gmw_l2_k_par(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = np.zeros(w.shape, dtype=w.dtype)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * np.exp(beta * np.log(w) - w**gamma) * w_nonneg

def _gmw_l2_k_gpu(w, gamma, beta, wc, k_consts):
    w_nonneg = (w >= 0)
    w *= w_nonneg  # zero negative `w` to avoid nans

    C = w.new_zeros(w.shape)
    for m in range(len(k_consts)):
        C += k_consts[m] * (2*w**gamma)**m
    return C * torch.exp(beta * torch.log(w) - w**gamma) * w_nonneg


def _gmw_k_constants(gamma, beta, k, norm='bandpass', dtype='float64'):
    """Laguerre polynomial constants & `coeff` term.

    Higher-order GMWs are coded such that constants are pre-computed and reused
    for any `w` input, since they remain fixed for said order.
    """
    r = (2 * beta + 1) / gamma
    c = r - 1

    # compute `coeff`
    if norm == 'bandpass':
        coeff = np.sqrt(np.exp(gammaln_fn(r) + gammaln_fn(k + 1) -
                               gammaln_fn(k + r)))
    elif norm == 'energy':
        coeff = np.sqrt(2*pi * gamma * (2**r) *
                        np.exp(gammaln_fn(k + 1) - gammaln_fn(k + r)))

    # compute Laguerre polynomial constants
    L_consts = np.zeros(k + 1, dtype=dtype)
    for m in range(k + 1):
        fact = np.exp(gammaln_fn(k + c + 1) - gammaln_fn(c + m + 1) -
                      gammaln_fn(k - m + 1))
        L_consts[m] = (-1)**m * fact / gamma_fn(m + 1)

    k_consts = L_consts * coeff
    if norm == 'bandpass':
        k_consts *= 2
    k_consts = k_consts.astype(dtype)
    return k_consts


#### General order wavelets (any `K`) ########################################
def morsewave(N, freqs, gamma=3, beta=60, K=1, norm='bandpass'):
    """Generalized Morse wavelets of Olhede and Walden (2002).

    # Arguments:
        N: int > 0
            Number of samples / wavelet length

        freqs: float / list / np.ndarray
            (peak) center frequencies at which to generate wavelets,
            in *radians* (i.e. `w` in `w = 2*pi*f`).

        gamma, beta: float, float
            GMW parameters; `(gamma, beta) = (3, 60)` yields optimal
            time-frequency localization, and a good default for natural signals.
              - smaller `beta`: greater time resolution, lower freq resolution.
              - `gamma`: structurally alters the wavelet; 2 and 1 provide
                superior time localization but poor joint localization.
            See refs [2], [3].

        K: int > 0
            Will compute first `K` orthogonal GMWs, characterized by
            orders 0 through `K - 1`.
            Note this `K` is 1 greater than in original paper and than `order`
            throughout `ssqueezepy`, but is consistent with jLAB.

        norm: str['energy', 'bandpass']
            Normalization to use. See `help(_gmw.gmw)`, and below.

    # Returns:
        psih: np.ndarray [N x len(freqs) x K]
            Frequency-domain GMW, generated by sampling continuous-time function.
            Will collapse dims of length 1 (e.g. if `K=0` or `freqs` is integer).
        psi: np.ndarray [N x len(freqs) x K]
            Time-domain GMW, centered, generated via inverse DFT of `psih`.

    # References
        See `help(_gmw.gmw)`.
    __________________________________________________________________________

    **`beta==0` case**

    For BETA equal to zero, the generalized Morse wavelets describe
    a non-zero-mean function which is not in fact a wavelet. Only 'bandpass'
    normalization is supported for this case.

    In this case the frequency speficies the half-power point of the
    analytic lowpass filter.

    The frequency-domain definition of MORSEWAVE is not necessarily
    a good way to compute the zero-beta functions, however.  You will
    probably need to take a very small DT.

    **Multiple orthogonal wavelets**

    MORSEWAVE can compute multiple orthogonal versions of the generalized
    Morse wavelets, characterized by the order K.

    PSI=MORSEWAVE(N,K,GAMMA,BETA,FS) with a fifth numerical argument K
    returns an N x LENGTH(FS) x K array PSI which contains time-domain
    versions of the first K orthogonal generalized Morse wavelets.

    These K different orthogonal wavelets have been employed in
    multiwavelet polarization analysis, see Olhede and Walden (2003a,b).

    Again either bandpass or energy normalization can be applied. With
    bandpass normalization, all wavelets are divided by a constant, setting
    the peak value of the first frequency-domain wavelet equal to 2.
    """
    _check_args(gamma=gamma, beta=beta, norm=norm)
    if not isinstance(freqs, (list, tuple, np.ndarray)):
        freqs = [freqs]
    psi  = np.zeros((N, len(freqs), K), dtype='complex128')
    psif = np.zeros((N, len(freqs), K))

    for n, f in enumerate(freqs):
        psif[:, n:n+1, :], psi[:, n:n+1, :] = _morsewave1(N, abs(f), gamma, beta,
                                                          K, norm)
        if f < 0:
            psi[:,   n:n+1, :] = psi[:, n, :].conj()
            psif[1:, n:n+1, :] = np.flip(psif[1:, n, :], axis=0)

    psi = psi.squeeze()
    psif = psif.squeeze()
    return psif, psi


def _morsewave1(N, f, gamma, beta, K, norm):
    """See `help(_gmw.morsewave)`."""
    fo = morsefreq(gamma, beta)
    fact = f / fo
    w = 2*pi * np.linspace(0, 1, N, endpoint=False) / fact
    w = w.reshape(-1, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        if norm == 'energy':
            if beta == 0:
                psizero = np.exp(-w**gamma)
            else:
                # w**beta * exp(-w**gamma)
                psizero = np.exp(beta * np.log(w) - w**gamma)
        else:
            if beta == 0:
                psizero = 2 * np.exp(-w**gamma)
            else:
                # Alternate calculation to cancel things that blow up
                psizero = 2 * np.exp(- beta * np.log(fo) + fo**gamma
                                     + beta * np.log(w)  - w**gamma)

    if beta == 0:
        # Ensure nice lowpass filters for beta=0;
        # Otherwise, doesn't matter since wavelets vanishes at zero frequency
        psizero[0] /= 2  # Due to unit-step function
    psizero[np.isnan(psizero) | np.isinf(psizero)] = 0.

    X = _morsewave_first_family(fact, N, K, gamma, beta, w, psizero, norm)
    X[np.isinf(X)] = 0.

    Xr = X.copy()
    # center time-domain wavelet
    Xr *= (-1)**np.arange(len(Xr)).reshape(-1, 1, 1)
    if len(Xr) % 2 == 0:
        Xr[len(Xr) // 2] /= 2
    x = ifft(Xr, axis=0)
    return X, x


def _morsewave_first_family(fact, N, K, gamma, beta, w, psizero, norm):
    """See `help(_gmw.morsewave)`.

    See Olhede and Walden, "Noise reduction in directional signals using
    multiple Morse wavelets", IEEE Trans. Bio. Eng., v50, 51--57.
    The equation at the top right of page 56 is equivalent to the
    used expressions. Morse wavelets are defined in the frequency
    domain, and so not interpolated in the time domain in the same way
    as other continuous wavelets.
    """
    r = (2 * beta + 1) / gamma
    c = r - 1
    L = np.zeros(w.shape)
    psif = np.zeros((len(psizero), 1, K))

    for k in range(K):
        # Log of gamma function much better ... trick from Maltab's ``beta'`
        if norm == 'energy':
            A = morseafun(gamma, beta, k + 1, norm='energy')
            coeff = np.sqrt(1. / fact) * A
        elif norm == 'bandpass':
            if beta == 0:
                coeff = 1.
            else:
                coeff = np.sqrt(np.exp(gammaln_fn(r) + gammaln_fn(k + 1) -
                                       gammaln_fn(k + r)))
        L[:N//2 + 1] = laguerre(2 * w[:N//2 + 1]**gamma, k, c).reshape(-1, 1)
        psif[:, :, k] = coeff * psizero * L
    return psif


def morseafun(gamma, beta, k=1, norm='bandpass'):
    """GMW amplitude or a-function (evaluated). Used internally by other funcs.

    # Arguments
        k: int >= 1
            Order of the wavelet; see `help(_gmw.morsewave)`.

        gamma, beta: float, float
            Wavelet parameters. See `help(_gmw.morsewave)`.

        norm: str['energy', 'bandpass']
            Wavelet normalization. See `help(_gmw.morsewave)`.

    # Returns
        A: float
            GMW amplitude (freq-domain peak value).
    ______________________________________________________________________
    Lilly, J. M. (2021), jLab: A data analysis package for Matlab, v1.6.9,
    http://www.jmlilly.net/jmlsoft.html
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morseafun.m
    """
    if norm == 'energy':
        r = (2*beta + 1) / gamma
        A = np.sqrt(2*pi * gamma * (2**r) *
                    np.exp(gammaln_fn(k) - gammaln_fn(k + r - 1)))
    elif norm == 'bandpass':
        if beta == 0:
            A = 2.
        else:
            wc = morsefreq(gamma, beta)
            A = 2. / np.exp(beta * np.log(wc) - wc**gamma)
    else:
        raise ValueError("unsupported `norm`: %s;" % norm
                         + "must be one of: 'bandpass', 'energy'.")
    return A


def laguerre(x, k, c):
    """Generalized Laguerre polynomials. See `help(_gmw.morsewave)`.

    LAGUERRE is used in the computation of the generalized Morse
    wavelets and uses the expression given by Olhede and Walden (2002),
    "Generalized Morse Wavelets", Section III D.
    """
    x = np.atleast_1d(np.asarray(x).squeeze())
    assert x.ndim == 1

    y = np.zeros(x.shape)
    for m in range(k + 1):
        # Log of gamma function much better ... trick from Maltab's ``beta''
        fact = np.exp(gammaln_fn(k + c + 1) - gammaln_fn(c + m + 1) -
                      gammaln_fn(k - m + 1))
        y += (-1)**m * fact * x**m / gamma_fn(m + 1)
    return y


def morsefreq(gamma, beta, n_out=1):
    """Frequency measures for GMWs (with F. Rekibi).

    `n_out` controls how many parameters are computed and returned, in the
    following order: `wm, we, wi, cwi`, where:

        wm: modal / peak frequency
        we: energy frequency
        wi: instantaneous frequency at time-domain wavelet's center
        cwi: curvature of instantaneous frequency at time-domain wavelet's center

    All frequency quantities are *radian*, opposed to linear cyclic (i.e. `w`
    in `w = 2*pi*f`).

    For BETA=0, the "wavelet" becomes an analytic lowpass filter, and `wm`
    is not defined in the usual way. Instead, `wm` is defined as the point
    at which the filter has decayed to one-half of its peak power.

    # References
        [1] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [2] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsefreq.m
    """
    wm = (beta / gamma)**(1 / gamma)

    if n_out > 1:
        we = (1 / 2**(1 / gamma)) * (gamma_fn((2*beta + 2) / gamma) /
                                     gamma_fn((2*beta + 1) / gamma))
    if n_out > 2:
        wi = (gamma_fn((beta + 2) / gamma) /
              gamma_fn((beta + 1) / gamma))
    if n_out > 3:
        k2 = _morsemom(2, gamma, beta, n_out=3)[-1]
        k3 = _morsemom(3, gamma, beta, n_out=3)[-1]
        cwi = -(k3 / k2**1.5)

    if n_out == 1:
        return wm
    elif n_out == 2:
        return wm, we
    elif n_out == 3:
        return wm, we, wi
    return wm, we, wi, cwi


def _morsemom(p, gamma, beta, n_out=4):
    """Frequency-domain `p`-th order moments of the first order GMW.
    Used internally by other funcs.

    `n_out` controls how many parameters are coMputed and returned, in the
    following order: `Mp, Np, Kp, Lp`, where:

        Mp: p-th order moment
        Np: p-th order energy moment
        Kp: p-th order cumulant
        Lp: p-th order energy cumulant

    The p-th order moment and energy moment are defined as
        Mp = 1/(2 pi) int omegamma^p  psi(omegamma)     d omegamma
        Np = 1/(2 pi) int omegamma^p |psi(omegamma)|.^2 d omegamma
    respectively, where omegamma is the radian frequency. These are evaluated
    using the 'bandpass' normalization, which has `max(abs(psih(omegamma)))=2`.

    # References
        [1] Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede. 2009.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        [2] (c) Lilly, J. M. (2021), jLab: A data analysis package for Matlab,
        v1.6.9, http://www.jmlilly.net/jmlsoft.html
        https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsemom.m
    """
    def morsemom1(p, gamma, beta):
        return morseafun(gamma, beta, k=1) * morsef(gamma, beta + p)

    def morsef(gamma, beta):
        # normalized first frequency-domain moment "f_{beta, gamma}" of the
        # first-order GMW
        return (1 / (2*pi * gamma)) * gamma_fn((beta + 1) / gamma)

    Mp = morsemom1(p, gamma, beta)

    if n_out > 1:
        Np = (2 / 2**((1 + p) / gamma)) * morsemom1(p, gamma, 2*beta)

    if n_out > 2:
        prange = np.arange(p + 1)
        moments = morsemom1(prange, gamma, beta)
        cumulants = _moments_to_cumulants(moments)
        Kp = cumulants[p]

    if n_out > 3:
        moments = (2 / 2**((1 + prange) / gamma)
                   ) * morsemom1(prange, gamma, 2 * beta)
        cumulants = _moments_to_cumulants(moments)
        Lp = cumulants[p]

    if n_out == 1:
        return Mp
    elif n_out == 2:
        return Mp, Np
    elif n_out == 3:
        return Mp, Np, Kp
    return Mp, Np, Kp, Lp


def _moments_to_cumulants(moments):
    """Convert moments to cumulants. Used internally by other funcs.

    Converts the first N moments   `moments  =[M0,M1,...M{N-1}]`
        into the first N cumulants `cumulants=[K0,K1,...K{N-1}]`.

    Note for a probability density function, M0=1 and K0=0.
    ______________________________________________________________________
    Lilly, J. M. (2021), jLab: A data analysis package for Matlab, v1.6.9,
    http://www.jmlilly.net/jmlsoft.html
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/moms
    """
    moments = np.atleast_1d(np.asarray(moments).squeeze())
    assert moments.ndim == 1

    cumulants = np.zeros(len(moments))
    cumulants[0] = np.log(moments[0])

    for n in range(1, len(moments)):
        coeff = 0
        for k in range(1, n):
            coeff += nCk(n - 1, k - 1
                         ) * cumulants[k] * (moments[n - k] / moments[0])
        cumulants[n] = (moments[n] / moments[0]) - coeff
    return cumulants


def _check_args(gamma=None, beta=None, norm=None, order=None, scale=None,
                allow_zerobeta=True):
    """Only checks those that are passed in."""
    if gamma is not None and gamma <= 0:
        raise ValueError(f"`gamma` must be positive (got {gamma})")

    if beta is not None:
        if beta < 0:
            kind = "non-negative" if allow_zerobeta else "positive"
            raise ValueError(f"`beta` must be {kind} (got {beta})")
        elif beta == 0 and not allow_zerobeta:
            raise ValueError(f"`beta` cannot be zero (got {beta}); "
                             "use `_gmw.morsewave`, which supports it")

    if norm is not None and norm not in ('bandpass', 'energy'):
        raise ValueError(f"`norm` must be 'energy' or 'bandpass' (got '{norm}')")

    if order is not None:
        if (not isinstance(order, (int, float)) or
                (isinstance(order, float) and not order.is_integer())):
            raise TypeError("`order` must be integer (got %s)" % str(order))
        elif order < 0:
            raise ValueError("`order` must be >=0 (got %s)" % order)

    if scale is not None and scale <= 0:
        raise ValueError(f"`scale` must be positive (got {scale})")
