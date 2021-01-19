# -*- coding: utf-8 -*-
"""Generalized Morse Wavelets"""
import numpy as np
from numpy.fft import ifft
from numba import jit
from scipy.special import (gamma   as gamma_fn,
                           gammaln as gammaln_fn)
from ssqueezepy.algos import nCk
from ssqueezepy.wavelets import _xifn

pi = np.pi


#### Base wavelets (`k=0`) ###################################################
def gmw(N, f, gamma=3, beta=60, time=False, norm='bandpass'):
    """Generalized Morse Wavelets.

    Assumes `k=0` and `beta != 0`; for full functionality use `morsewave`.

    See: https://overlordgolddragon.github.io/generalized-morse-wavelets/

    # References:
        1. Generalized Morse Wavelets. S. C. Olhede and A. T. Walden. 2002.
        https://spiral.imperial.ac.uk/bitstream/10044/1/1150/1/
        OlhedeWaldenGenMorse.pdf

        2. Higher-Order Properties of Analytic Wavelets. J. M. Lilly and
        S. C. Olhede. 2009. https://sci-hub.st/10.1109/TSP.2008.2007607
    """
    if f <= 0:
        raise ValueError(f"`f` must be positive (got {f})")
    if beta < 0:
        raise ValueError(f"`beta` must be positive (got {beta})")
    elif beta == 0:
        raise ValueError(f"`beta` cannot be zero (got {beta}); "
                         "use `morsewave`, which supports it")
    if norm not in ('bandpass', 'energy'):
        raise ValueError(f"`norm` must be 'energy' or 'bandpass' (got {norm})")

    f0 = np.exp((np.log(beta) - np.log(gamma)) / gamma)
    w = _xifn(1, N) * (f0 / f)

    gmw_fn = (gmw_l1(gamma, beta) if norm == 'bandpass' else
              gmw_l2(gamma, beta, f))
    X = np.zeros(N)
    X[:N//2 + 1] = gmw_fn(w[:N//2 + 1])
    X[np.isinf(X) | np.isnan(X)] = 0.

    if time:
        Xr = X.copy()
        if N % 2 == 0:
            Xr[N//2] /= 2
        x = ifft(Xr * (-1)**np.arange(N))

    return (X, x) if time else X

# TODO divide `w` by `fo`? note compute_gmw already does it, so undo
def gmw_l1(gamma=3., beta=60.):
    wc = np.exp((np.log(beta) - np.log(gamma)) / gamma)
    return lambda w: _gmw_l1(w, gamma, beta, wc)

@jit(nopython=True, cache=True)
def _gmw_l1(w, gamma, beta, wc):
    return 2 * np.exp(- beta * np.log(wc) + wc**gamma
                      + beta * np.log(w)  - w**gamma)


def gmw_l2(gamma=3., beta=60., f=1):
    wc = np.exp((np.log(beta) - np.log(gamma)) / gamma)
    r = (2*beta + 1) / gamma
    rgamma = gamma_fn(r)
    return lambda w: _gmw_l2(w, gamma, beta, f, wc, r, rgamma)

@jit(nopython=True, cache=True)
def _gmw_l2(w, gamma, beta, f, wc, r, rgamma):
    return np.sqrt(2.*pi * (wc / f) * gamma * 2.**r / rgamma
                   ) * w**beta * np.exp(-w**gamma)


#### General order wavelets (any `k`) ########################################
def morsewave(N, freqs, gamma=3, beta=60, K=1, norm='bandpass'):
    if not isinstance(freqs, (list, tuple, np.ndarray)):
        freqs = [freqs]
    psi  = np.zeros((N, len(freqs), K), dtype='complex128')
    psif = np.zeros((N, len(freqs), K))

    for n, f in enumerate(freqs):
        psif[:, n:n+1, :], psi[:, n:n+1, :] = morsewave1(N, abs(f), gamma, beta,
                                                         K, norm)
        if f < 0:
            psi[:,   n:n+1, :] = psi[:, n, :].conj()
            psif[1:, n:n+1, :] = np.flip(psif[1:, n, :], axis=0)

    if sum(psi.shape[1:]) == 2:
        psi = psi.squeeze()
    if sum(psif.shape[1:]) == 2:
        psif = psif.squeeze()
    return psif, psi


def morsewave1(N, f, gamma, beta, K, norm):
    """
    ______________________________________________________________________
    JLAB (C) 2004--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
    """
    fo = morsefreq(gamma, beta, n_out=1)
    fact = f / fo
    w = 2*pi * np.linspace(0, 1, N, endpoint=False) / fact
    w = w.reshape(-1, 1)

    with np.errstate(divide='ignore'):
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

    psizero[0] /= 2  # Due to unit-step function
    # Ensure nice lowpass filters for beta=0;
    # Otherwise, doesn't matter since wavelets vanishes at zero frequency

    psizero[np.isnan(psizero)] = 0.

    X = _morsewave_first_family(fact, N, K, gamma, beta, w, psizero, norm)
    X[np.isinf(X)] = 0.

    # ensures wavelets are centered; exp for complex-valued rotation
    # ommat = np.broadcast_to(om[:, None], (N, 1, K))
    # Xr = X.astype('complex128') * np.exp(1j * ommat * (N + 1) / 2 * fact)
    Xr = X.copy()
    Xr *= (-1)**np.arange(len(Xr)).reshape(-1, 1, 1)
    if len(Xr) % 2 == 0:
        Xr[len(Xr) // 2] /= 2
    x = ifft(Xr, axis=0)
    return X, x


def _morsewave_first_family(fact, N, K, gamma, beta, w, psizero, norm):
    """See Olhede and Walden, "Noise reduction in directional signals using
    multiple Morse wavelets", IEEE Trans. Bio. Eng., v50, 51--57.
    The equation at the top right of page 56 is equivalent to the
    used expressions. Morse wavelets are defined in the frequency
    domain, and so not interpolated in the time domain in the same way
    as other continuous wavelets.
    ______________________________________________________________________
    JLAB (C) 2004--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
    """
    r = (2 * beta + 1) / gamma
    c = r - 1
    L = np.zeros(w.shape)
    psif = np.zeros((len(psizero), 1, K))
    index = slice(0, N // 2 + 1)

    for k in range(K):
        #Log of gamma function much better ... trick from Maltab's ``beta'`
        if norm == 'energy':
            A = morseafun(k + 1, gamma, beta, norm);
            coeff = np.sqrt(1. / fact) * A
        elif norm == 'bandpass':
            if beta != 0:
                coeff = np.sqrt(np.exp(gammaln_fn(r) + gammaln_fn(k + 1) -
                                       gammaln_fn(k + r)))
            else:
                coeff = 1.
        L[index] = laguerre(2 * w[index]**gamma, k, c).reshape(-1, 1)
        psif[:, :, k] = coeff * psizero * L
    return psif


def morseafun(k, gamma, beta, norm='bandpass'):
    """Returns the generalized Morse wavelet amplitude or a-function.

    MORSEAFUN is a low-level function called by many a number of the Morse
    wavelet functions.

    A=MORSEAFUN(GAMMA,BETA) returns the generalized Morse wavelet
    amplitude, called "A_{BETA,GAMMA}" by Lilly and Olhede (2009).

    By default, A is chosen such that the maximum of the frequency-
    domain wavelet is equal to 2, the ``bandpass normalization.''

    A=MORSEAFUN(GAMMA,BETA,'energy') instead returns the coefficient
    giving the wavelet unit energy.

    A=MORSEAFUN(K,GAMMA,BETA,'energy') returns the unit energy coefficient
    appropriate for the Kth-order wavelet.  The default choice is K=1.
    ______________________________________________________________________
    JLAB (C) 2006--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morseafun.m
    """
    if norm == 'energy':
        r = (2*beta + 1) / gamma
        a = np.sqrt(2*pi * gamma * (2**r) *
                    np.exp(gammaln_fn(k) - gammaln_fn(k + r - 1)))
    elif norm == 'bandpass':
        w = morsefreq(gamma, beta, n_out=1)
        a = 2. / np.exp(beta * np.log(w) - w**gamma)
    else:
        raise ValueError("unsupported norm: %s;" % norm
                         + "must be one of: 'bandpass', 'energy'.")
    return a


def laguerre(x, k, c):
    """Generalized Laguerre polynomials

    Y=LAGUERRE(X,K,C) where X is a column vector returns the generalized Laguerre
    polynomials specified by parameters K and C.

    LAGUERRE is used in the computation of the generalized Morse
    wavelets and uses the expression given by Olhede and Walden (2002),
    "Generalized Morse Wavelets", Section III D.
    ______________________________________________________________________
    JLAB (C) 2004--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsewave.m
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


# TODO this is RADIAN frequencies ffs
def morsefreq(gamma, beta, n_out=4):
    """Frequency measures for generalized Morse wavelets. [with F. Rekibi]

    [FM,FE,FI]=MORSEFREQ(GAMMA,BETA) calculates three different measures of
    the frequency of the lowest-order generalized Morse wavelet specified
    by parameters GAMMA and BETA.

    FM is the modal or peak, FE is the "energy" frequency, and FI is the
    instantaneous frequency at the wavelet center.

    [FM,FE,FI,CF]=MORSEFREQ(GAMMA,BETA) also computes the curvature CF of
    the instantaneous frequency at the wavelet center.

    Note that all frequency quantities here are *radian* as in cos(omegamma t)
    and not cyclic as in cos(2 pi f t).

    The input parameters must either be matrices of the same size,
    or some may be matrices and the others scalars.

    For BETA=0, the "wavelet" becomes an analytic lowpass filter, and FM
    is not defined in the usual way. Instead, FM is defined as the point
    at which the filter has decayed to one-half of its peak power.

    For details see
        Lilly and Olhede (2009).  Higher-order properties of analytic
            wavelets.  IEEE Trans. Sig. Proc., 57 (1), 146--160.

    See also MORSEBOX, MORSEPROPS, MORSEWAVE.

    Usage: fm = morsefreq(gamma,beta);
           [fm,fe,fi] = morsefreq(gamma,beta);
           [fm,fe,fi,cf] = morsefreq(gamma,beta);
    ______________________________________________________________________
    JLAB (C) 2004--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsefreq.m
    """
    fm = np.exp((np.log(beta) - np.log(gamma)) / gamma)

    if n_out > 1:
        fe = (1 / 2**(1 / gamma)) * (gamma_fn((2*beta + 2) / gamma) /
                                     gamma_fn((2*beta + 1) / gamma))
    if n_out > 2:
        fi = (gamma_fn((beta + 2) / gamma) /
              gamma_fn((beta + 1) / gamma))
    if n_out > 3:
        m2, n2, k2 = morsemom(2, gamma, beta, n_out=3)
        m3, n3, k3 = morsemom(3, gamma, beta, n_out=3)
        cf = -(k3 / k2**1.5)

    if n_out == 1:
        return fm
    elif n_out == 2:
        return fm, fe
    elif n_out == 3:
        return fm, fe, fi
    return fm, fe, fi, cf


def morsemom(p, gamma, beta, n_out=4):
    """Frequency-domain moments of generalized Morse wavelets.

    MORSEMOM is a low-level function called by several other Morse wavelet
    functions.

    [MP,NP]=MORSEMOM(P,GAMMA,BETA) computes the Pth order frequency-
    domain moment M and energy moment N of the lower-order generalized
    Morse wavelet specified by parameters GAMMA and BETA.

    The Pth moment and energy moment are defined as

            mp = 1/(2 pi) int omegamma^p  psi(omegamma)     d omegamma
            np = 1/(2 pi) int omegamma^p |psi(omegamma)|.^2 d omegamma

    respectively, where omegamma is the radian frequency.  These are evaluated
    using the 'bandpass' normalization, which has max(abs(psi(omegamma)))=2.

    The input parameters must either be matrices of the same size, or
    some may be matrices and the others scalars.

    [MP,NP,KP,LP]=MORSEMOM(...) also returns the Pth order cumulant KP and
    the Pth order energy cumulant LP.

    For details see
        Lilly and Olhede (2009).  Higher-order properties of analytic
            wavelets.  IEEE Trans. Sig. Proc., 57 (1), 146--160.

    Usage:  mp=morsemom(p,gamma,beta);
            [mp,np]=morsemom(p,gamma,beta);
            [mp,np,kp,lp]=morsemom(p,gamma,beta);
    _____________________________________________________________________
    JLAB (C) 2007--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jWavelet/morsemom.m
    """
    def _morsemom1(p, gamma, beta):
        return morseafun(1, gamma, beta) * _morsef(gamma, beta + p)

    def _morsef(gamma, beta):
        """Returns the generalized Morse wavelet first moment "f".

        F=MORSEF(GAMMA,BETA) returns the normalized first frequency-
        domain moment "F_{BETA,GAMMA}" of the lower-order generalized
        Morse wavelet specified by parameters GAMMA and BETA.
        """
        return (1 / (2*pi * gamma)) * gamma_fn((beta + 1) / gamma)

    m = _morsemom1(p, gamma, beta)

    if n_out > 1:
        n = (2 / 2**((1 + p) / gamma)) * _morsemom1(p, gamma, 2*beta)

    if n_out > 2:
        prange = np.arange(p + 1)
        moments = _morsemom1(prange, gamma, beta)
        cumulants = _moments_to_cumulants(moments)
        k = cumulants[p]

    if n_out > 3:
        moments = (2 / 2**((1 + prange) / gamma)
                   ) * _morsemom1(prange, gamma, 2 * beta)

        cumulants = _moments_to_cumulants(moments)
        l = cumulants[p]

    if n_out == 1:
        return m
    elif n_out == 2:
        return m, n
    elif n_out == 3:
        return m, n, k
    return m, n, k, l


def _moments_to_cumulants(moms):
    """Convert moments to cumulants.

    [K0,K1,...KN]=MOM2CUM(M0,M1,...MN) converts the first N moments
    M0,M1,...MN into the first N cumulants K0,K1,...KN.

    The MN and KN are all scalars or arrays of the same size.
    Note for a probability density function, M0=1 and K0=0.
    ___________________________________________________________________
    JLAB (C) 2008--2016 J.M. Lilly
    https://github.com/jonathanlilly/jLab/blob/master/jCommon/mom2cum.m
    """
    moms = np.atleast_1d(np.asarray(moms).squeeze())

    assert moms.ndim == 1

    cums = np.zeros(len(moms))
    cums[0] = np.log(moms[0])

    for n in range(1, len(moms)):
        coeff = 0
        for k in range(1, n - 1):
            coeff += nCk(n - 1, k - 1) * cums[k] * (moms[n - k] / moms[0])
        cums[n] = (moms[n] / moms[0]) - coeff
    return cums
