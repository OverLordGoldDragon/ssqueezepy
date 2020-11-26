# -*- coding: utf-8 -*-
"""NOT FOR USE; will be ready for v0.6.0"""
import numpy as np
from numpy.fft import fft, ifft
from scipy import integrate
from .utils import padsignal, buffer


pi = np.pi
EPS = np.finfo(np.float64).eps  # machine epsilon for float64


def stft_fwd(x, dt, opts={}):
    """Compute the short-time Fourier transform and modified short-time
    Fourier transform from [1]. The former is very closely based on Steven
    Schimmel's stft.m and istft.m from his SPHSC 503: Speech Signal Processing
    course at Univ. Washington.

    # Arguments:
        x: np.ndarray. Input signal vector, length `n` (need not be dyadic).
        dt: int, sampling period (defaults to 1).
        opts: dict. Options:
            'type': str. Wavelet type. See `wfiltfn`
            'winlen': int. length of window in samples; Nyquist frequency
                      is winlen/2
            'padtype': str ('symmetric', 'repliace', 'circular'). Type
                       of padding (default = 'symmetric')
            'rpadded': bool. Whether to return padded `Sx` and `dSx`
                       (default = True)
            's', 'mu', ... : window options (see `wfiltfn`)

    # Returns:
        Sx: (na x n) size matrix (rows = scales, cols = times) containing
            samples of the CWT of `x`.
        Sfs: vector containign the associated frequencies.
        dSx: (na x n) size matrix containing samples of the time-derivatives
             of the STFT of `x`.

    # References:
        1. G. Thakur and H.-T. Wu,
        "Synchrosqueezing-based Recovery of Instantaneous Frequency
        from Nonuniform Samples",
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
    """
    def _process_opts(opts, x):
        # opts['window'] is window length; opts['type'] overrides the
        # default hamming window
        opts['stft_type'] = opts.get('stft_type', 'normal')
        opts['winlen']    = opts.get('winlen',    int(np.round(len(x) / 8)))
        # 'padtype' is one of: 'symmetric', 'replicate', 'circular'
        opts['padtype']   = opts.get('padtype',  'symmetric')
        opts['rpadded']   = opts.get('rpadded',  False)

        windowfunc, diff_windowfunc = None, None
        if 'type' in opts:
            windowfunc      = wfiltfn(opts['type'], opts, derivative=False)
            diff_windowfunc = wfiltfn(opts['type'], opts, derivative=True)

        return opts, windowfunc, diff_windowfunc

    opts, windowfunc, diff_windowfunc = _process_opts(opts, x)

    # Pre-pad signal; this only works well for 'normal' STFT
    n = len(x)
    if opts['stft_type'] == 'normal':
        x, N, n1, n2 = padsignal(x, opts['padtype'], opts['winlen'])
        n1 = n1 // 2
    else:
        N = n
        n1 = 0

    if opts['stft_type'] == 'normal':
        # set up window
        if 'type' in opts:
            window = windowfunc(np.linspace(-1, 1, opts['winlen']))
            diff_window = diff_windowfunc(np.linspace(-1, 1, opts['winlen']))
        else:
            window = np.hamming(opts['winlen'])
            diff_window = np.hstack([np.diff(window), 0])
        diff_window[np.where(np.isnan(diff_window))] = 0

        # frequency range
        Sfs = np.linspace(0, 1, opts['winlen'] + 1)
        Sfs = Sfs[:np.floor(opts['winlen'] / 2).astype('int64') + 1] / dt

        # compute STFT and keep only the positive frequencies
        xbuf = buffer(x, opts['winlen'], opts['winlen'] - 1, 'nodelay')
        xbuf = np.diag(window) @ xbuf
        Sx = fft(xbuf, None, axis=0)
        Sx = Sx[:opts['winlen'] // 2 + 1] / np.sqrt(N)

        # same steps for STFT derivative
        dxbuf = buffer(x, opts['winlen'], opts['winlen'] - 1, 'nodelay')
        dxbuf = np.diag(diff_window) @ dxbuf
        dSx = fft(dxbuf, None, axis=0)
        dSx = dSx[:opts['winlen'] // 2 + 1] / np.sqrt(N)
        dSx /= dt

    elif opts['stft_type'] == 'modified':
        # modified STFt is more accurately done in the frequency domain,
        # like a filter bank over different frequency bands
        # uses a lot of memory, so best used on small blocks
        # (<5000 samples) at a time
        Sfs = np.linspace(0, 1, N) / dt
        Sx  = np.zeros((N, N))
        dSx = np.zeros((N, N))

        halfN = np.round(N / 2)
        halfwin = np.floor((opts['winlen'] - 1) / 2)
        window = windowfunc(np.linspace(-1, 1, opts['winlen'])) # TODO chk dim
        diff_window = diff_windowfunc(np.linspace(-1, 1, opts['winlen'])) * (
            2 / opts['winlen'] / dt)

        for k in range(N):
            freqs = np.arange(-min(halfN - 1, halfwin, k - 1),
                              min(halfN - 1, halfwin, N- k) + 1)
            indices = np.mod(freqs, N)
            Sx[indices, k]  = x[k + freqs] * window(halfwin + freqs + 1)
            dSx[indices, k] = x[k + freqs] * diff_window(halfwin + freqs + 1)

        Sx  = fft(Sx)  / np.sqrt(N)
        dSx = fft(dSx) / np.sqrt(N)

        # only keep the positive frequencies
        Sx  = Sx[:halfN]
        dSx = dSx[:halfN]
        Sfs = Sfs[:halfN]

    # Shorten Sx to proper size (remove padding)
    if not opts['rpadded']:
        Sx  = Sx[:,  n1:n1 + n]
        dSx = dSx[:, n1:n1 + n]

    return Sx, Sfs, dSx


def stft_inv(Sx, opts={}):
    """Inverse short-time Fourier transform.

    Very closely based on Steven Schimel's stft.m and istft.m from his
    SPHSC 503: Speech Signal Processing course at Univ. Washington.
    Adapted for use with Synchrosqueeing Toolbox.

    # Arguments:
        Sx: np.ndarray. Wavelet transform of a signal (see `stft_fwd`).
        opts: dict. Options:
            'type': str. Wavelet type. See `stft_fwd`, and `wfiltfn`.
            Others; see `stft_fwd` and source code.

    # Returns:
        x: the signal, as reconstructed from `Sx`.
    """
    def _unbuffer(x, w, o):
        # Undo the effect of 'buffering' by overlap-add;
        # returns the signal A that is the unbuffered version of B
        y = []
        skip = w - o
        N = np.ceil(w / skip)
        L = (x.shape[1] - 1) * skip + x.shape[0]

        # zero-pad columns to make length nearest integer multiple of `skip`
        if x.shape[0] < skip * N:
            x[skip * N - 1, -1] = 0  # TODO columns?

        # selectively reshape columns of input into 1d signals
        for i in range(N):
            t = x[:, range(i, len(x) - 1, N)].reshape(1, -1)
            l = len(t)
            y[i, l + (i - 1)*skip - 1] = 0
            y[i, np.arange(l) + (i - 1)*skip] = t

        # overlap-add
        y = np.sum(y, axis=0)
        y = y[:L]

        return y

    def _process_opts(opts, Sx):
        # opts['window'] is window length; opts['type'] overrides
        # default hamming window
        opts['winlen']  = opts.get('winlen',  int(np.round(Sx.shape[1] / 16)))
        opts['overlap'] = opts.get('overlap', opts['winlen'] - 1)
        opts['rpadded'] = opts.get('rpadded', False)

        if 'type' in opts:
            A = wfiltfn(opts['type'], opts)
            window = A(np.linspace(-1, 1, opts['winlen']))
        else:
            window = np.hamming(opts['winlen'])

        return opts, window

    opts, window = _process_opts(opts, Sx)

    # window = window / norm(window, 2) --> Unit norm
    n_win = len(window)

    # find length of padding, similar to outputs of `padsignal`
    n = Sx.shape[1]
    if not opts['rpadded']:
        xLen = n
    else:
        xLen == n - n_win

    # n_up = xLen + 2 * n_win
    n1 = n_win - 1
    # n2 = n_win
    new_n1 = np.floor((n1 - 1) / 2)

    # add STFT apdding if it doesn't exist
    if not opts['rpadded']:
        Sxp = np.zeros(Sx.shape)
        Sxp[:, range(new_n1, new_n1 + n + 1)] = Sx
        Sx = Sxp
    else:
        n = xLen

    # regenerate the full spectrum 0...2pi (minus zero Hz value)
    Sx = np.hstack([Sx, np.conj(Sx[np.arange(
        np.floor((n_win + 1) / 2), 3, -1)])])

    # take the inverse fft over the columns
    xbuf = np.real(ifft(Sx, None, axis=0))

    # apply the window to the columns
    xbuf *= np.matlib.repmat(window.flatten(), 1, xbuf.shape[1])

    # overlap-add the columns
    x = _unbuffer(xbuf, n_win, opts['overlap'])

    # keep the unpadded part only
    x = x[n1:n1 + n + 1]

    # compute L2-norm of window to normalize STFT with
    windowfunc = wfiltfn(opts['type'], opts, derivative=False)
    C = lambda x: integrate.quad(windowfunc(x) ** 2, -np.inf, np.inf)

    # `quadgk` is a bit inaccurate with the 'bump' function,
    # this scales it correctly
    if opts['type'] == 'bump':
        C *= 0.8675

    x *= 2 / (pi * C)

    return x


def phase_stft(Sx, dSx, Sfs, t, opts={}):
    """Calculate the phase transform of modified STFT at each (freq, time) pair:
        w[a, b] = Im( eta - d/dt(Sx[t, eta]) / Sx[t, eta] / (2*pi*j))
    Uses direct differentiation by calculating dSx/dt in frequency domain
    (the secondary output of `stft_fwd`, see `stft_fwd`).

    # Arguments:
        Sx: np.ndarray. Wavelet transform of `x` (see `stft_fwd`).
        dSx: np.ndarray. Samples of time-derivative of STFT of `x`
             (see `stft_fwd`).
        opts: dict. Options:
            'gamma': float. Wavelet threshold (default: sqrt(machine epsilon))

    # Returns:
        w: phase transform, w.shape == Sx.shape

    # References:
        1. G. Thakur and H.-T. Wu,
        "Synchrosqueezing-based Recovery of Instantaneous Frequency from
        Nonuniform Samples",
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.

        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.
    """
    opts['gamma'] = opts.get('gamma', np.sqrt(EPS))

    # calculate phase transform; modified STFT amounts to extra frequency term
    w = np.matlib.repmat(Sfs, len(t), 1).T - np.imag(dSx / Sx / (2 * pi))

    # threshold out small points
    w[np.abs(Sx) < opts['gamma']] = np.inf

    return w


# !!! deprecated, moved to wavelets.py; kept for stft_transforms.py
def wfiltfn(wavelet, derivative=False):
    """Wavelet transform function of the wavelet filter in question,
    Fourier domain.

    # Arguments:
        wavelet_type: str. See below.
        opts: dict. Options, e.g. {'s': 1, 'mu': 5}

    # Returns:
        lambda xi: psihfn(xi)

    _______________________________________________________________________
    Filter types      Use for synsq?    Parameters (default)

    mhat              no                s (1)
    cmhat             yes               s (1), mu (1)
    morlet            yes               mu (2*pi)
    shannon           no                --   (NOT recommended for analysis)
    hshannon          yes               --   (NOT recommended for analysis)
    hhat              no
    hhhat             yes               mu (5)
    bump              yes               s (1), mu (5)
    _______________________________________________________________________

    # Example:
        psihfn = wfiltfn('bump', {'s': .5, 'mu': 1})
        plt.plot(psihfn(np.arange(-5, 5.01, step=.01)))
    """
    if isinstance(wavelet, tuple):
        wavelet, wavopts = wavelet
    else:
        wavopts = {}
    supported = ('bump', 'mhat', 'cmhat', 'morlet', 'shannon',
                 'hshannon', 'hhat', 'hhhat')
    if wavelet not in supported:
        raise ValueError(("Unsupported wavelet '{}'; must be one of: {}"
                          ).format(wavelet, ", ".join(supported)))

    if wavelet == 'bump':
        mu = wavopts.get('mu', 5)
        s  = wavopts.get('s',  1)
        om = wavopts.get('om', 0)

        psihfnorig = lambda w: (np.abs(w) < .999) * np.exp(
            -1. / (1 - (w * (np.abs(w) < .999)) ** 2)) / .443993816053287

        psihfn = lambda w: np.exp(2 * pi * 1j * om * w) * psihfnorig(
            (w - mu) / s) / s
        if derivative:
            _psihfn = psihfn; del psihfn
            psihfn = lambda w: _psihfn(w) * (
                2 * pi * 1j * om - 2 * ((w - mu) / s**2) / (
                    1 - ((w - mu) / s)**2)**2)

    elif wavelet == 'mhat':  # mexican hat
        s = wavopts.get('s', 1)
        psihfn = lambda w: -np.sqrt(8) * s**(5/2) * pi**(1/4) / np.sqrt(
            3) * w**2 * np.exp(-s**2 * w**2 / 2)

    elif wavelet == 'cmhat':
        # complex mexican hat; hilbert analytic function of sombrero
        # can be used with synsq
        s  = wavopts.get('s',  1)
        mu = wavopts.get('mu', 1)
        psihfnshift = lambda w: 2 * np.sqrt(2/3) * pi**(-1/4) * (
            s**(5/2) * w**2 * np.exp(-s**2 * w**2 / 2) * (w >= 0))
        psihfn = lambda w: psihfnshift(w - mu)

    elif wavelet == 'morlet':
        # can be used with synsq for large enough `s` (e.g. >5)
        mu = wavopts.get('mu', 2 * pi)
        cs = (1 + np.exp(-mu**2) - 2 * np.exp(-3/4 * mu**2)) ** (-.5)
        ks = np.exp(-.5 * mu**2)
        psihfn = lambda w: cs * pi**(-1/4) * (np.exp(-.5 * (mu - w)**2)
                                              - ks * np.exp(-.5 * w**2))

    elif wavelet == 'shannon':
        psihfn = lambda w: np.exp(-1j * w / 2) * (np.abs(w) >= pi
                                                  and np.abs(w) <= 2 * pi)
    elif wavelet == 'hshannon':
        # hilbert analytic function of shannon transform
        # time decay is too slow to be of any use in synsq transform
        mu = wavopts.get('mu', 0)
        psihfnshift = lambda w: np.exp(-1j * w / 2) * (
            w >= pi and w <= 2 * pi) * (1 + np.sign(w))
        psihfn = lambda w: psihfnshift(w - mu)

    elif wavelet == 'hhat':  # hermitian hat
        psihfnshift = lambda w: 2 / np.sqrt(5) * pi**(-1 / 4) * (
            w * (1 + w) * np.exp(-.5 * w**2))
        psihfn = lambda w: psihfnshift(w - mu)

    elif wavelet == 'hhhat':
        # hilbert analytic function of hermitian hat; can be used with synsq
        mu = wavopts.get('mu', 5)
        psihfnshift = lambda w: 2 / np.sqrt(5) * pi**(-1/4) * (
            w * (1 + w) * np.exp(-1/2 * w**2)) * (1 + np.sign(w))
        psihfn = lambda w: psihfnshift(w - mu)

    return psihfn
