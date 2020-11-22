import numpy as np
from numba import njit
from types import FunctionType

pi = np.pi


class Wavelet():
    """Wavelet transform function of the wavelet filter in question,
    Fourier domain.

    _______________________________________________________________________
    Filter types      Use for synsq?    Parameters (default)

    cmhat             yes               s (1), mu (1)
    morlet            yes               mu (2*pi)
    hhhat             yes               mu (5)
    bump              yes               s (1), mu (5)
    _______________________________________________________________________

    # TODO force analyticity @ neg frequencies if Morse also fails to?
    # Example:
        psihfn = Wavelet(('bump', {'s': .5, 'mu': 1}), N=1024)
        plt.plot(psihfn(scale=8))
    """
    SUPPORTED = ('morlet', 'bump', 'cmhat', 'hhhat')
    def __init__(self, wavelet='morlet', N=128):
        self._validate_and_set_wavelet(wavelet)

        self.xi = _xi(scale=1, N=N)

    def __call__(self, ipt, N=None):
        """psihfn(ipt) if ipt is np.ndarray, else ipt = scale, and computes
           psihfn(scale * xi), where `xi` is recomputed if `N` is not None.
        """
        if isinstance(ipt, np.ndarray) and ipt.size > 1:
            return self.fn(ipt)
        elif N is None:
            return self.fn(ipt * self.xi)
        return self.fn(_xi(ipt, N))

    def _validate_and_set_wavelet(self, wavelet):
        if isinstance(wavelet, FunctionType):
            self.fn = wavelet
            self.name = wavelet.__qualname__
            return

        errmsg = ("`wavelet` must be one of: (1) string name of supported "
                  "wavelet; (2) tuple of (1) and dict of wavelet parameters "
                  "(e.g. {'mu': 5}); (3) custom function taking `scale * xi` "
                  "as input. (got: %s)" % str(wavelet))
        if not isinstance(wavelet, (tuple, str)):
            raise TypeError(errmsg)
        elif isinstance(wavelet, tuple):
            if not (len(wavelet) == 2 and isinstance(wavelet[1], dict)):
                raise TypeError(errmsg)
            wavelet, wavopts = wavelet
        elif isinstance(wavelet, str):
            wavopts = {}

        if wavelet not in Wavelet.SUPPORTED:
            raise ValueError(f"wavelet '{wavelet}' is not supported; pass "
                             "in fn=custom_fn, or use one of:", ', '.join(
                                 Wavelet.SUPPORTED))
        if wavelet == 'morlet':
            self.fn = morlet(**wavopts)
        elif wavelet == 'bump':
            self.fn = bump(**wavopts)
        elif wavelet == 'cmhat':
            self.fn = cmhat(**wavopts)
        elif wavelet == 'hhhat':
            self.fn = hhhat(**wavopts)
        self.name = wavelet


@njit
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

#### Wavelet functions ######################################################
def morlet(mu=5.):
    cs = (1 + np.exp(-mu**2) - 2 * np.exp(-3/4 * mu**2)) ** (-.5)
    ks = np.exp(-.5 * mu**2)
    return lambda w: _morlet(w, mu, cs, ks)


@njit
def _morlet(w, mu, cs, ks):
    return cs * pi**(-1/4) * (np.exp(-.5 * (mu - w)**2)
                              - ks * np.exp(-.5 * w**2))


def bump(mu=5., s=1., om=0.):
    return lambda w: _bump(w, (w - mu) / s, om, s)


@njit
def _bump(w, _w, om, s):
    return np.exp(2 * pi * 1j * om * w) / s * (
        np.abs(_w) < .999) * np.exp(-1. / (1 - (_w * (np.abs(_w) < .999))**2)
                                   ) / .443993816053287


def cmhat(mu=1., s=1.):
    return lambda w: _cmhat(w - mu, s)


@njit
def _cmhat(_w, s):
    return 2 * np.sqrt(2/3) * pi**(-1/4) * (
        s**(5/2) * _w**2 * np.exp(-s**2 * _w**2 / 2) * (_w >= 0))


def hhhat(mu=5.):
    return lambda w: _hhhat(w - mu)


@njit
def _hhhat(_w):
    return 2 / np.sqrt(5) * pi**(-1/4) * (_w * (1 + _w) * np.exp(-1/2 * _w**2)
                                          ) * (1 + np.sign(_w))
