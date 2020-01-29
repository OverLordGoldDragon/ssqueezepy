# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
import numpy.matlib
from quadpy import quad as quadgk

PI = np.pi


def mad(data, axis=None):
    return np.mean(np.abs(data - np.mean(data, axis)), axis)


def est_riskshrink_thresh(Wx, nv):
    """Estimate the RiskShrink hard thresholding level.
    
    # Arguments:
        Wx:  np.ndarray. Wavelet transform of a signal.
        opt: dict. Options structure used for forward wavelet transform.
    
    # Returns:
        gamma: float. The RiskShrink hard thresholding estimate.
    """
    na, n = Wx.shape

    Wx_fine = np.abs(Wx[:nv])
    gamma = 1.4826 * np.sqrt(2 * np.log(n)) * np.mad(Wx_fine)

    return gamma


def p2up(n):
    """Calculates next power of 2, and left/right padding to center
    the original `n` locations.
    
    # Arguments:
        n: int. Non-dyadic integer.
    
    # Returns:
        up: next power of 2
        n1: length on left
        n2: length on right
    """
    eps = np.finfo(np.float64).eps  # machine epsilon for float64
    up = 2 ** (1 + np.round(np.log2(n + eps)))
    
    n1 = np.floor((up - n) / 2)
    n2 = n1
    
    if (2 * n1 + n) % 2 == 1:
        n2 = n1 + 1
    return up, n1, n2


def padsignal(x, padtype='symmetric', padlength=None):
    """Pads signal and returns indices of original signal.
    
    # Arguments:
        x: np.ndarray. Original signal.
        padtype: str ('symmetric', 'replicate').
        padlength: int. Number of samples to pad on each side. Default is
                   nearest power of 2.
    
    # Returns:
        x: padded signal.
        n_up: next power of 2.
        n1: length on left.
        n2: length on right.
    """
    padtypes = ('symmetric', 'replicate')
    if padtype not in padtypes:
        raise ValueError(("Unsupported `padtype` {}; must be one of: {}"
                          ).format(padtype, ", ".join(padtypes)))
    n = len(x)

    if padlength is None:
        # pad up to the nearest power of 2
        n_up, n1, n2 = p2up(n)
    else:
        n_up = n + 2 * padlength
        n1 = padlength + 1
        n2 = padlength    
    n_up, n1, n2 = int(n_up), int(n1), int(n2)

    if padtype == 'symmetric':
        xl = np.matlib.repmat(np.hstack([x, np.flipud(x)]),
                              m=int(np.ceil(n1 / (2 * n))), n=1).squeeze()
        xr = np.matlib.repmat(np.hstack([np.flipud(x), x]),
                              m=int(np.ceil(n2 / (2 * n))), n=1).squeeze()
    elif padtype == 'replicate':
        xl = x[0]  * np.ones(n1)
        xr = x[-1] * np.ones(n2)

    xpad = np.hstack([xl[-n1:], x, xr[:n2]])
    
    return xpad, n_up, n1, n2


def wfiltfn(wavelet_type, opts, derivative=False):
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
    supported_types = ('bump', 'mhat', 'cmhat', 'morlet', 'shannon',
                       'hshannon', 'hhat', 'hhhat')
    if wavelet_type not in supported_types:
        raise ValueError(("Unsupported `wavelet_type` '{}'; must be one of: {}"
                          ).format(wavelet_type, ", ".join(supported_types)))
    if wavelet_type == 'bump':
        mu = opts.get('mu', 5)
        s  = opts.get('s',  1)
        om = opts.get('om', 0)
        
        psihfnorig = lambda w: (np.abs(w) < .999) * np.exp(
            -1. / (1 - (w * (np.abs(w) < .999)) ** 2)) / .443993816053287
        
        psihfn = lambda w: np.exp(2 * PI * 1j * om * w) * psihfnorig(
            (w - mu) / s) / s
        if derivative:
            _psihfn = psihfn; del psihfn
            psihfn = lambda w: _psihfn(w) * (
                2 * PI * 1j * om - 2 * ((w - mu) / s**2) / (
                    1 - ((w - mu) / s)**2)**2)

    elif wavelet_type == 'mhat':  # mexican hat
        s = opts.get('s', 1)
        psihfn = lambda w: -np.sqrt(8) * s**(5/2) * PI**(1/4) / np.sqrt(
            3) * w**2 * np.exp(-s**2 * w**2 / 2)

    elif wavelet_type == 'cmhat':
        # complex mexican hat; hilbert analytic function of sombrero
        # can be used with synsq
        s  = opts.get('s',  1)
        mu = opts.get('mu', 1)
        psihfnshift = lambda w: 2 * np.sqrt(2/3) * PI**(-1/4) * (
            s**(5/2) * w**2 * np.exp(-s**2 * w**2 / 2) * (w >= 0))
        psihfn = lambda w: psihfnshift(w - mu)
    
    elif wavelet_type == 'morlet':
        # can be used with synsq for large enough `s` (e.g. >5)
        mu = opts.get('mu', 2 * PI)
        cs = (1 + np.exp(-mu**2) - 2 * np.exp(-3/4 * mu**2)) ** (-.5)
        ks = np.exp(-.5 * mu**2)
        psihfn = lambda w: cs * PI**(-1/4) * (np.exp(-.5 * (mu - w)**2)
                                              - ks * np.exp(-.5 * w**2))
    
    elif wavelet_type == 'shannon':
        psihfn = lambda w: np.exp(-1j * w / 2) * (np.abs(w) >= PI
                                                  and np.abs(w) <= 2 * PI)
    elif wavelet_type == 'hshannon':
        # hilbert analytic function of shannon transform
        # time decay is too slow to be of any use in synsq transform
        mu = opts.get('mu', 0)
        psihfnshift = lambda w: np.exp(-1j * w / 2) * (
            w >= PI and w <= 2 * PI) * (1 + np.sign(w))
        psihfn = lambda w: psihfnshift(w - mu)

    elif wavelet_type == 'hhat':  # hermitian hat
        psihfnshift = lambda w: 2 / np.sqrt(5) * PI**(-1 / 4) * (
            w * (1 + w) * np.exp(-.5 * w**2))
        psihfn = lambda w: psihfnshift(w - mu)

    elif wavelet_type == 'hhhat':
        # hilbert analytic function of hermitian hat; can be used with synsq
        mu = opts.get('mu', 5)
        psihfnshift = lambda w: 2 / np.sqrt(5) * PI**(-1/4) * (
            w * (1 + w) * np.exp(-1/2 * w**2)) * (1 + np.sign(w))
        psihfn = lambda w: psihfnshift(w - mu)

    return psihfn


def synsq_adm(wavelet_type, opts={}):
    """Calculate the synchrosqueezing admissibility constant, the term
    R_\psi in Eq. 3 of [1]. Note, here we multiply R_\psi by the inverse of
    log(2)/nv (found in Alg. 1 of [1]).
    
    Uses numerical intergration.
    
    # Arguments:
        wavelet_type: str. See `wfiltfn`.
        opts: dict. Options. See `wfiltfn`.
    
    # Returns:
        Css: proportional to 2 * integral(conj(f(w)) / w, w=0..inf)
    
    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.
    
    """
    psihfn = wfiltfn(wavelet_type, opts)
    Css = lambda x: quadgk(np.conj(psihfn(x)) / x, 0, np.inf)
    
    # Normalization constant, due to logarithmic scaling
    # in wavelet transform
    _Css = Css; del Css
    Css = lambda x: _Css(x) / np.sqrt(2 * PI) * 2 * np.log(2)
    
    return Css


def buffer(x, n, p=0, opt=None):
    """Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    # Arguments:
        x: np.ndarray. Signal array.
        n: int. Number of data segments.
        p: int. Number of values to overlap
        opt: str.  Initial condition options. Default sets the first `p` 
        values to zero, while 'nodelay' begins filling the buffer immediately.

    # Returns:
        result : (n,n) ndarray
            Buffer array created from x.
            
    # References:
        ryanjdillon: https://stackoverflow.com/questions/38453249/
        is-there-a-matlabs-buffer-equivalent-in-numpy#answer-40105995
    """
    import numpy as np

    if opt not in ('nodelay', None):
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n-p]])
                i = n-p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[:,-1][-p:], col])
        i += n-p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n-len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result
