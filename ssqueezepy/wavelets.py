import numpy as np
from numba import jit
from .utils import pi


@jit(nopython=True)
def _xi(scale, N):
    # N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
    # N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
    xi = np.zeros(N)
    h = scale * (2 * pi) / N
    for i in range(N // 2 + 1):
        xi[i] = i * h
    for i in range(N // 2 + 1, N):
        xi[i] = (i - N) * h
    return xi


def morlet(mu=5.):
    cs = (1 + np.exp(-mu**2) - 2 * np.exp(-3/4 * mu**2)) ** (-.5)
    ks = np.exp(-.5 * mu**2)
    return lambda w: _morlet(w, mu, cs, ks)


@jit(nopython=True)
def _morlet(w, mu, cs, ks):
    return cs * pi**(-1/4) * (np.exp(-.5 * (mu - w)**2)
                              - ks * np.exp(-.5 * w**2))

class Wavelet():
    def __init__(self, name='morlet', N=None):
        self.name = name

        if N is not None:
            self.xi = _xi(1, N)
        else:
            self.xi = None

    def __call__(self, scale, N=None):
        if N is None:
            if self.N is None:
                raise ValueError("`N` can't be None if `self.N` is also None.")
            xi_scale = scale * self.xi
        else:
            xi_scale = _xi(scale, N)
        self._fn(xi_scale)
