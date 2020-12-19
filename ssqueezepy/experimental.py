# -*- coding: utf-8 -*-
import numpy as np
from numpy.fft import fft, ifft, ifftshift
from .utils import pi, adm_cwt
from .wavelets import Wavelet
from scipy import integrate


def err_fix(x, wavelet, a0):  # primitive code, doesn't work
    """Implements corrective term in Eq. 4.66 of [1].

    1. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.
    """
    # note x is *original* (padded), so this step must be done in forward CWT
    # to be passed to icwt
    N = len(x)
    xi = (2*pi/N) * np.arange(1, N//2 + 1)

    psihfn = Wavelet(wavelet)
    # integrate from 0 to w, w spanning same spectrum as psih
    # this can be sped up by nature of brick-wall behavior, stopping computing
    # after first zero, also computing fewer in total and linearly interpolating
    Cpsi_w = [integrate.quad(
        lambda x: np.conj(psihfn(x)) * psihfn(x) / x, 0., w)[0]
        for w in a0 * xi]

    Cpsi_w.insert(0, 0)  # integral 0 to 0 = 0
    Cpsi_w.extend([0] * (N // 2 - 1))  # analytic, right-half = 0

    # integrate from 0 to inf
    Cpsi = adm_cwt(wavelet)
    # subtract from integration 0 to inf to obtain w to inf
    phi_w = Cpsi - np.array(Cpsi_w)

    # do convolution theorem with x, take care of padding etc
    corr = ifftshift(ifft(fft(x) * phi_w ** 2))
    corr /= (a0 * Cpsi)  # normalize
    return corr
