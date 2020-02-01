# Ported from the Synchrosqueezing Toolbox, authored by
# Eugine Brevdo, Gaurav Thakur
#    (http://www.math.princeton.edu/~ebrevdo/)
#    (https://github.com/ebrevdo/synchrosqueezing/)

import numpy as np
from .utils import padsignal, wfiltfn

EPS = np.finfo(np.float64).eps  # machine epsilon for float64
PI = np.pi


def synsq_squeeze(Wx, w, t, nv=None, opts={}):
    """Calculates the synchrosqueezed CWT or STFT of `x`. Used internally by
    `synsq_cwt_fw` and `synsq_stft_fw`.
    
    # Arguments:
        Wx or Sx: np.ndarray. CWT or STFT of `x`.
        w: np.ndarray. Phase transform at same locations in T-F plane.
        t: np.ndarray. Time vector.
        nv: int. Number of voices (CWT only).
        opts: dict. Options:
            'transform': ('CWT', 'STFT'). Underlying time-frequency transform.
            'freqscale': ('log', 'linear'). Frequency bins/divisions.
            'findbins':  ('min', 'direct'). Method to find bins. 
                         'direct' is faster.
            'squeezing': ('full', 'measure'). Latter corresponds to approach
                         in [3], which is not invertible but has better
                         robustness properties in some cases.
    
    # Returns:
        Tx: synchrosqueezed output.
        fs: associated frequencies.
        
    Note the multiplicative correction term x in `synsq_cwt_squeeze_mex`, 
    required due to the fact that the squeezing integral of Eq. (2.7), in, 
    [1], is taken w.r.t. dlog(a). This correction term needs to be included 
    as a factor of Eq. (2.3), which we implement here.

    A more detailed explanation is available in Sec. III of [2].
    Note the constant multiplier log(2)/nv has been moved to the
    inverse of the normalization constant, as calculated in `synsq_adm`.
    
    # References:
        1. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms:
        an empricial mode decomposition-like tool",
        Applied and Computational Harmonic Analysis, 30(2):243-261, 2011.
	    
        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications",
        Signal Processing, 93:1079-1094, 2013.
	    
        3. G. Thakur and H.-T. Wu,  "Synchrosqueezing-based Recovery of 
        Instantaneous Frequency from Nonuniform Samples", 
        SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
    """
    def _squeeze(w, Wx, Wx_scales, fs, dfs, scaleterm, na, N, lfm, lfM, opts):
        def _vectorized(scaleterm):  # possible to optimize further
            if len(scaleterm.shape) == 1:
                scaleterm = np.expand_dims(scaleterm, 1)
            if v1:
                _k = np.round(w * dfs)
                _k[np.where(np.isnan(_k))] = len(fs)
                _k_ones = np.ones(_k.size)
                k = np.min((np.max((_k.flatten(), 1*_k_ones), axis=0),
                            len(fs)*_k_ones), axis=0).reshape(*_k.shape)
            elif v2:  # TESTED
                _k = 1 + np.round(na / (lfM - lfm) * (np.log2(w) - lfm))
                _k[np.where(np.isnan(_k))] = len(fs)
                _k_ones = np.ones(_k.size)
                k = np.min((np.max((_k.flatten(), 1*_k_ones), axis=0),
                            na*_k_ones), axis=0).reshape(*_k.shape)
            elif v3:
                w_rep = np.matlib.repmat(w[None].flatten(), len(fs), 1).reshape(
                    len(fs), *w.shape)
                _k = w_rep - fs.reshape(len(fs), 1, 1)
                _k[np.where(np.isnan(_k))] = len(fs)
                k = np.min(np.abs(_k), axis=0)   
                
            k = np.floor(k).astype('int32') - 1  # MAT to py idx
            Ws_prod = Wx * scaleterm

            for b in range(N):
                for ai in range(len(Wx_scales)):
                    Tx[k[ai, b], b] += Ws_prod[ai, b]
            return Tx
        
        def _for_loop():  # much slower; tested 11x slowdown
            if v1:
                for b in range(N):
                  for ai in range(len(Wx_scales)):
                    _k = np.round(w[ai, b] * dfs)
                    k = min(max(_k, 1), len(fs)) if not np.isnan(_k) else len(fs)
    
                    k = int(k - 1)  # MAT to py idx
                    Tx[k, b] += Wx[ai, b] * scaleterm[ai]
            elif v2:
                for b in range(N):
                  for ai in range(len(Wx_scales)):
                    _k = 1 + np.round(na / (lfM - lfm) * (
                        np.log2(w[ai, b]) - lfm))
                    k = min(max(_k, 1), na) if not np.isnan(_k) else len(fs)
    
                    k = int(k - 1)  # MAT to py idx
                    Tx[k, b] += Wx[ai, b] * scaleterm[ai]
            elif v3:
                for b in range(N):
                  for ai in range(len(Wx_scales)):
                    k = np.min(np.abs(w[ai, b] - fs))
                
                    k = int(k - 1)  # MAT to py idx
                    Tx[k, b] += Wx[ai, b] * scaleterm[ai]   
            return Tx

        # must cast to complex else value assignment discards imaginary component
        Tx = np.zeros((len(fs), Wx.shape[1])).astype('complex128')

        # do squeezing by finding which frequency bin each phase transform
        # point w(ai, b) lands in
        # look only at points where w(ai, b) is positive and finite
        v1 = (opts['findbins'] == 'direct') and (opts['freqscale'] == 'linear')
        v2 = (opts['findbins'] == 'direct') and (opts['freqscale'] == 'log')
        v3 = (opts['findbins'] == 'min')

        Tx = _vectorized(scaleterm)

        if opts['transform'] == 'CWT':
            Tx *= (1 / nv)
            Tx *= (fs[1] - fs[0])
            
        return Tx, fs

    def _compute_associated_frequencies(na, N, fm, fM, dt, opts):
        # frequency divisions `w_l` to search over in Synchrosqueezing
        if opts['freqscale'] == 'log':
            lfm = np.log2(fm)
            lfM = np.log2(fM)
            _fs = fm * np.power(fM / fm, np.arange(na - 1) / (np.floor(na) - 1))
            fs  = np.hstack([_fs, fM])
            dfs = None
        elif opts['freqscale'] == 'linear':
            if opts['transform'] == 'CWT':
                fs = np.linspace(fm, fM, na)
            elif opts['transform'] == 'STFT':
                fs = np.linspace(0, 1, N) / dt
                fs = fs[:N // 2]
            dfs = 1 / (fs[1] - fs[0])
            lfm, lfM = None, None
        return fs, dfs, lfm, lfM

    def _process_opts(opts):
        if 'freqscale' not in opts:
            if opts['transform'] == 'CWT':
                opts['freqscale'] = 'log'
            elif opts['transform'] == 'STFT':
                opts['freqscale'] = 'linear'
        opts['findbins']  = opts.get('findbins',  'direct')
        opts['squeezing'] = opts.get('squeezing', 'full')
        
        return opts
    
    opts = _process_opts(opts)

    dt = t[1]  - t[0]
    dT = t[-1] - t[0]
    
    # maximum measurable (Nyquist) frequency of data
    fM = 1 / (2 * dt)
    # minimum measurable (fundamental) frequency of data
    fm = 1 / dT
    
    # `na` is number of scales for CWT, number of freqs for STFT
    na, N = Wx.shape 
    fs, dfs, lfm, lfM = _compute_associated_frequencies(na, N, fm, fM, dt, opts)
    
    if opts['transform'] == 'CWT':
        Wx_scales = np.power(2 ** (1 / nv), 
                             np.expand_dims(np.arange(1, na + 1), 1))
        scaleterm = np.power(Wx_scales, -.5)
    elif opts['transform'] == 'STFT':
        Wx_scales = np.linspace(fm, fM, na)
        scaleterm = np.ones(Wx_scales.shape)
    
    # measure version from reference [3]
    if opts['squeezing'] == 'measure':
        Wx = np.ones(Wx.shape) / Wx.shape[0]
    
    # incorporate threshold by zeroing out Inf values, so they get ignored below
    Wx[np.isinf(w)] = 0

    Tx, fs = _squeeze(w, Wx, Wx_scales, fs, dfs, scaleterm, 
                      na, N, lfm, lfM, opts)
	# MEX version, deprecated (above code has been reworked to attain 
    # similar speed with JIT compiler)
	# Tx = 1/nv * synsq_cwt_squeeze_mex(Wx, w, as, fs, ones[fs.shape], lfm, lfM)
    return Tx, fs


def synsq_cwt_squeeze(Wx, w, t, nv):
    """Calculates the synchrosqueezed transform of `f` on a logarithmic scale.
    Used internally by `synsq_cwt_fwd`.
    
    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x`.
        w: np.ndarray. Estimate of frequency locations in `Wx` 
                       (see `synsq_cwt_fwd`).
        t: np.ndarray. Time vector.
        nv: int. Number of voices.
        
    # Returns:
        Tx: synchrosqueezed output.
        fs: associated frequencies.
        
    Note the multiplicative correction term `f` in `_cwt_squeeze`, required 
    due to the fact that the squeezing integral of Eq. (2.7), in, [1], is taken
    w.r.t. dlog(a). This correction term needs to be included as a factor of 
    Eq. (2.3), which  we implement here.
    
    A more detailed explanation is available in Sec. III of [2].
    Specifically, this is an implementation of Sec. IIIC, Alg. 1.
    Note the constant multiplier log(2)/nv has been moved to the
    inverse of the normalization constant, as calculated in `synsq_adm`.
    
    # References:
        1. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms: a
        tool for empirical mode decomposition", 2010.
        
        2. E. Brevdo, N.S. Fučkar, G. Thakur, and H-T. Wu, "The
        Synchrosqueezing algorithm: a robust analysis tool for signals
        with time-varying spectrum," 2011.
    """
    def _cwt_squeeze(Wx, w, Wx_scales, fs, dfs, N, lfm, lfM):
        Tx = np.zeros(Wx.shape)

        for b in range(N):
          for ai in range(len(Wx_scales)):
            if not np.isinf(w[ai, b]) and w[ai, b] > 0:
                # find w_l nearest to w[ai, b]
                k = int(np.min(np.max(
                    1 + np.floor(na / (lfM - lfm) * (np.log2(w[ai, b]) - lfm)),
                    0), na - 1))
                Tx[k, b] += Wx[ai, b] * Wx_scales[ai] ** (-0.5)

        return Tx

    dt = t[1]  - t[0]
    dT = t[-1] - t[0]
    
    # Maximum measurable frequency of data
    #fM = 1/(4*dt) # wavelet limit - tested
    fM = 1/(2*dt)  # standard
    # Minimum measurable frequency, due to wavelet
    fm = 1/dT;
    #fm = 1/(2*dT); # standard

    na, N = Wx.shape

    Wx_scales = np.power(2 ** (1 / nv), np.expand_dims(np.arange(1, na + 1)))
    # dWx_scales = np.array([1, np.diff(Wx_scales)])

    lfm = np.log2(fm)
    lfM = np.log2(fM)
    fs  = np.power(2, np.linspace(lfm, lfM, na))
    #dfs = np.array([fs[0], np.diff(fs)])
    
    # Harmonics of diff. frequencies but same magniude have same |Tx|
    dfs = np.ones(fs.shape)

    if np.linalg.norm(Wx, 'fro') < EPS:
        Tx = np.zeros(Wx.shape)
    else:
        Tx = (1 / nv) * _cwt_squeeze(
            Wx, w, Wx_scales, fs, dfs, N, lfm, lfM)

    return Tx


def phase_cwt(Wx, dWx, opts={}):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses direct differentiation by calculating dWx/db in frequency domain
    (the secondary output of `cwt_fwd`, see `cwt_fwd`)
    
    This is the analytic implementation of Eq. (7) of [1].
    
    # Arguments:
        Wx: np.ndarray. wavelet transform of `x` (see `cwt_fwd`).
        dWx: np.ndarray. Samples of time derivative of wavelet transform of `x`
             (see `cwt_fwd`).
        opts. dict. Options:
            'gamma': wavelet threshold (default: sqrt(machine epsilon))
    
    # Returns:
        w: phase transform, w.shape == Wx.shape.
        
    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications," 
        Signal Processing, 93:1079-1094, 2013.
        
        2. I. Daubechies, J. Lu, H.T. Wu, "Synchrosqueezed Wavelet Transforms: 
        an empricial mode decomposition-like tool", 
        Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
    """
    if opts.get('gamma', None) is None:
        opts['gamma'] = np.sqrt(EPS)
    
    # Calculate phase transform for each `ai`, normalize by (2 * pi)
    if opts.get('dtype', None) == 'phase':
        u = np.unwrap(np.angle(Wx)).T
        w = np.array([np.diff(u), u[-1] - u[0]]).T / (2 * PI)
    else:
        w = np.abs(np.imag(dWx / Wx / (2 * PI)))

    w[np.abs(Wx) < opts['gamma']] = np.inf
    return w


def phase_cwt_num(Wx, dt, opts={}):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses numerical differentiation (1st, 2nd, or 4th order).
    
    This is a numerical differentiation implementation of Eq. (7) of [1].
    
    # Arguments:
        Wx: np.ndarray. Wavelet transform of `x` (see `cwt_fwd`).
        dt: int. Sampling period (e.g. t[1] - t[0]).
        opts. dict. Options:
            'dorder': int (1, 2, 4). Differences order. (default = 4)
            'gamma': float. Wavelet threshold. (default = sqrt(machine epsilon))
    
    # Returns:
        w: demodulated FM-estimates, w.shape == Wx.shape.
    
    # References:
        1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications," 
        Signal Processing, 93:1079-1094, 2013.
    """
    def _differentiate(Wx, dt, opts):
        if opts['dorder'] == 1:
            w = np.array([Wx[:, 1:] - Wx[:, :-1],
                          Wx[:, 0]  - Wx[:, -1]])
            w /= dt
        elif opts['dorder'] == 2:
            # append for differentiating
            Wxr = np.array([Wx[:, -2:], Wx, Wx[:, :2]])
            # calculate 2nd-order forward difference
            w = -Wxr[:, 4:] + 4 * Wxr[:, 3:-1] - 3 * Wxr[:, 2:-2]
            w /= (2 * dt)
        elif opts['dorder'] == 4:
            # calculate 4th-order central difference
            w = -Wxr[:, 4:]
            w += Wxr[:, 3:-1] * 8
            w -= Wxr[:, 1:-3] * 8
            w += Wxr[:, 0:-4]
            w /= (12 * dt)
        
        return w

    def _process_opts(opts):
        # order of differentiation (1, 2, or 4)
        opts['dorder'] = opts.get('dorder', 4)

        # epsilon from Daubechies, H-T Wu, et al.
        # gamma from Brevdo, H-T Wu, et al.
        opts['gamma'] = opts.get('gamma', np.sqrt(EPS))
        
        if opts['dorder'] not in (1, 2, 4):
            raise ValueError("Differentiation order %d not supported"
                             % opts['dorder'])
        return opts

    opts = _process_opts(opts)
    
    w = _differentiate(Wx, dt, opts)
    w[np.abs(Wx) < opts['gamma']] = np.nan
    
    # calculate inst. freq for each `ai`, normalize by (2*pi) for true freq
    w = np.real(-1j * w / Wx) / (2 * PI)
    
    return w


def cwt_fwd(x, wavelet_type, nv=32, dt=1, opts={}):
    """Forward continuous wavelet transform, discretized, as described in
    Sec. 4.3.3 of [1] and Sec. IIIA for [2]. This algorithm uses the FFT and
    samples the wavelet atoms in the Fourier domain. Options such as padding
    of the original signal are allowed. Returns the vector of scales and, if
    requested, the analytic time-derivative of the wavelet transform (as 
    described in Sec. IIIB of [2]).
    
    # Arguments:
        x: np.ndarray. Input signal vector, length `n` (need not be dyadic).
        wavelet_type: str. See `wfiltfn`.
        nv: int. Number of voices. Suggested >= 32.
        dt: int. sampling period.
        opts: dict. Options:
            'padtype': ('symmetric', 'replicate', 'circular'). Type of padding.
                       (default = 'symmetric')
            'rpadded': bool. Whether to return padded Wx and dWx.
                       (default = False)
            'type', 's', 'mu', ...: str. Wavelet options (see `wfiltfn`).
    
    # Returns:
        Wx: (na x n) size matrix (rows = scales, cols = times), containing
            samples of the CWT of `x`.
        Wx_scales: `na` length vector containing the associated scales.
        dWx: (na x n) size matrix containing samples of the time-derivatives 
              of the CWT of `x`.
        xMean: mean of padded `x`.
        
    # References:
        1. Mallat, S., Wavelet Tour of Signal Processing 3rd ed.
        
        2. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu,
        "The Synchrosqueezing algorithm for time-varying spectral analysis: 
        robustness properties and new paleoclimate applications,"
        Signal Processing, 93:1079-1094, 2013.
    """
    opts['padtype'] = opts.get('padtype', 'symmetric')
    opts['rpadded'] = opts.get('rpadded', 0)
    
    n = len(x)
    
    # pad x first
    x, N, n1, n2 = padsignal(x, opts['padtype'])
    
    xMean = np.mean(x)
    x -= xMean

    # choosing more than this means the wavelet window becomes too short
    noct = np.log2(N) - 1
    assert(noct > 0 and noct % 1 == 0)
    assert(nv > 0   and nv   % 1 == 0)
    assert(dt > 0)
    assert(not np.any(np.isnan(x)))

    na = int(noct * nv)
    Wx_scales = np.power(2 ** (1 / nv), np.arange(1, na + 1))

    # must cast to complex else value assignment discards imaginary component
    Wx = np.zeros((na, N)).astype('complex128')
    dWx = Wx.copy()
    opts['dt'] = dt
    
    # x = x.T  # already shaped as a row vector
    xh = np.fft.fft(x)
    
    # for each octave
    # reworked this part to not use `wfilth`, which slows things down a lot
    # due to branching and temp objects; see that function for more comments
    k = np.arange(N)
    xi = np.zeros((1, N))
    xi[:, :N // 2]     = 2 * PI / N * np.arange(N // 2)
    xi[:, N // 2 + 1:] = 2 * PI / N * np.arange(-N // 2 + 1, 0)
    psihfn = wfiltfn(wavelet_type, opts)

    for ai in range(na):
        a = Wx_scales[ai]
        psih = psihfn(a * xi) * np.sqrt(a) / np.sqrt(2 *PI) * (-1)**k
        dpsih = (1j * xi / opts['dt']) * psih

        xcpsi = np.fft.ifftshift(np.fft.ifft(psih * xh))
        Wx[ai] = xcpsi
        
        dxcpsi = np.fft.ifftshift(np.fft.ifft(dpsih * xh))
        dWx[ai] = dxcpsi
    
    # shorten W to proper size
    if not opts['rpadded']:
        Wx  = Wx[ :, n1:n1 + n]
        dWx = dWx[:, n1:n1 + n]
        
    # output for graphing purposes; scale by `dt`
    Wx_scales *= dt
    
    return Wx, Wx_scales, dWx, xMean
