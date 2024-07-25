# -*- coding: utf-8 -*-
import numpy as np
from .utils import EPS32, EPS64, pi, p2up, adm_ssq, process_scales
from .utils import trigdiff, _process_fs_and_t
from .utils import backend as S
from .algos import replace_under_abs, phase_cwt_cpu, phase_cwt_gpu
from .ssqueezing import ssqueeze, _check_ssqueezing_args
from .wavelets import Wavelet
from ._cwt import cwt


def ssq_cwt(x, wavelet='gmw', scales='log-piecewise', nv=None, fs=None, t=None,
            ssq_freqs=None, padtype='reflect', squeezing='sum', maprange='peak',
            difftype='trig', difforder=None, gamma=None, vectorized=True,
            preserve_transform=None, astensor=True, order=0, nan_checks=None,
            patience=0, flipud=True, cache_wavelet=None,
            get_w=False, get_dWx=False):
    """Synchrosqueezed Continuous Wavelet Transform.
    Implements the algorithm described in Sec. III of [1].

    Uses `wavelet.dtype` precision.

    # Arguments:
        x: np.ndarray / torch.Tensor
            Input vector(s), 1D or 2D. See `help(cwt)`.

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain. See `help(cwt)`.

        scales: str['log', 'linear', 'log:maximal', ...] / np.ndarray
            CWT scales. See `help(cwt)`.

        nv: int / None
            Number of voices (wavelets per octave). Suggested >= 16.

        fs, t: float, np.ndarray
            See `help(_cwt.cwt)`.

        ssq_freqs: str['log', 'linear'] / np.ndarray / None
            Frequencies to synchrosqueeze CWT scales onto. Scale-frequency
            mapping is only approximate and wavelet-dependent.
            If None, will infer from and set to same distribution as `scales`.

        padtype: str / None
            Pad scheme to apply on input. See `help(utils.padsignal)`.
            `None` -> no padding.

        squeezing: str['sum', 'lebesgue'] / function
            See `help(ssqueezing.ssqueeze)`.

        maprange: str['maximal', 'peak', 'energy'] / tuple(float, float)
            Kind of frequency mapping used, determining the range of frequencies
            spanned (fm to fM, min to max).

                - 'maximal': fm=1/dT, fM=1/(2*dt), always. Data's fundamental
                and Nyquist frequencies, determined from `fs` (or `t`).
                Other mappings can never span outside this range.

                - ('peak', 'energy'): sets fm and fM based on center frequency
                associated with `wavelet` at maximum and minimum scale,
                respectively. See `help(wavelets.center_frequency)`.

                - 'peak': the frequency-domain trimmed bell will have its peak
                at Nyquist, meaning all other frequencies are beneath, so each
                scale is still correctly resolved but with downscaled energies.
                With sufficiently-spanned `scales`, coincides with 'maximal'.

                - 'energy': however, the bell's spectral energy is centered
                elsewhere, as right-half of bell is partly or entirely trimmed
                (left-half can be trimmed too). Use for energy-centric mapping,
                which for sufficiently-spanned `scales` will always have lesser
                fM (but ~same fM).

                - tuple: sets `ssq_freqrange` directly.

        difftype: str['trig', 'phase', 'numeric']
            Method by which to differentiate Wx (default='trig') to obtain
            instantaneous frequencies:
                    w(a,b) = Im( (1/2pi) * (1/Wx(a,b)) * d/db[Wx(a,b)] )

            - 'trig': use `dWx`, obtained via trigonometric (frequency-domain
            interpolant) differentiation (see `cwt`, `phase_cwt`).
            - 'phase': differentiate by taking forward finite-difference of
            unwrapped angle of `Wx` (see `phase_cwt`).
            - 'numeric': first-, second-, or fourth-order (set by `difforder`)
            numeric differentiation (see `phase_cwt_num`).

        difforder: int[1, 2, 4]
            Order of differentiation for difftype='numeric' (default=4).

        gamma: float / None
            CWT phase threshold. Sets `w=inf` for small values of `Wx` where
            phase computation is unstable and inaccurate (like in DFT):
                w[abs(Wx) < beta] = inf
            This is used to zero `Wx` where `w=0` in computing `Tx` to ignore
            contributions from points with indeterminate phase.
            Default = 10 * (machine epsilon) = 10 * np.finfo(np.float64).eps
            (or float32)

            It is recommended to standardize the input, or at least not
            pass a small-valued input, to avoid false filtering by `gamma`,
            especially if input obeys a power scaling law
            (e.g. `~1/f` with EEG/MEG, and similar with audio).
            # TODO warn user if `x.max()` is small?

        vectorized: bool (default True)
            Whether to vectorize CWT, i.e. compute quantities for all scales at
            once, which is faster but uses more memory.

        preserve_transform: bool (default None) / None
            Whether to return `Wx` as directly output from `cwt` (it might be
            altered by `ssqueeze` or `phase_transform`). Uses more memory
            per storing extra copy of `Wx`.
                - Defaults to True if `'SSQ_GPU' == '0'`, else False.

        astensor: bool (default True)
            If `'SSQ_GPU' == '1'`, whether to return arrays as on-GPU tensors
            or move them back to CPU & convert to Numpy arrays.

        order: int (default 0) / tuple[int]
            `order > 0` computes ssq of `cwt` taken with higher-order GMWs.
            If tuple, computes ssq of average of `cwt`s taken at each specified
            order. See `help(_cwt.cwt_higher_order)`.

        nan_checks: bool / None
            Checks whether input has `nan` or `inf` values, and zeros them.
            `False` saves compute. Doesn't support torch inputs.

            Defaults to `True` for NumPy inputs, else `False`.

        patience: int / tuple[int, int]
            pyFFTW parameter for faster FFT on CPU; see `help(ssqueezepy.FFT)`.

        flipud: bool (default True)
            See `help(ssqueeze)`.

        cache_wavelet: bool (default None) / None
            See `help(cwt)`.

        get_w, get_dWx: bool (default False)
            `get_w`:
                True: will compute phase transform separately, assign it to
                array `w` and return it.
                False: will compute synchrosqueezing directly from `Wx` and
                `dWx` without assigning to intermediate array, which is faster
                (by 20-30%) and takes less memory.
            `get_dWx`:
                True: will return dWx
                False: discards dWx after computing `w` or synchrosqueezing.
                `get_dWx=True` with `get_w=True` uses most memory.
            These options do not affect `Tx`.

    # Returns:
        Tx: np.ndarray [nf x n]
            Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
            (nf = len(ssq_freqs); n = len(x))
            `nf = na` by default, where `na = len(scales)`.
        Wx: np.ndarray [na x n]
            Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
        ssq_freqs: np.ndarray [nf]
            Frequencies associated with rows of `Tx`.
        scales: np.ndarray [na]
            Scales associated with rows of `Wx`.
        w: np.ndarray [na x n]  (if `get_w=True`)
            Phase transform for each element of `Wx`.
        dWx: [na x n] np.ndarray (if `get_dWx=True`)
            See `help(_cwt.cwt)`.

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        3. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        4. Synchrosqueezed Wavelet Transform Explanation. John Muradeli.
        https://dsp.stackexchange.com/a/71399/50076

        5. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_fw.m
    """
    def _process_args(x, scales, fs, t, nv, difftype, difforder, squeezing,
                      maprange, wavelet, get_w):
        if x.ndim == 2 and get_w:
            raise NotImplementedError("`get_w=True` unsupported with batched "
                                      "input.")
        difforder = _check_ssqueezing_args(squeezing, maprange, wavelet,
                                           difftype, difforder, get_w,
                                           transform='cwt')
        if nv is None and not isinstance(scales, np.ndarray):
            nv = 32

        N = x.shape[-1]
        dt, fs, t = _process_fs_and_t(fs, t, N)
        return N, dt, fs, difforder, nv

    def _phase_transform(Wx, dWx, N, dt, gamma, difftype, difforder):
        if difftype == 'trig':
            # calculate instantaneous frequency directly from the
            # frequency-domain derivative
            w = phase_cwt(Wx, dWx, difftype, gamma)
        elif difftype == 'phase':
            # !!! bad; yields negatives, and forcing abs(w) doesn't help
            # calculate inst. freq. from unwrapped phase of CWT
            w = phase_cwt(Wx, None, difftype, gamma)
        elif difftype == 'numeric':
            # !!! tested to be very inaccurate for small scales
            # calculate derivative numerically
            _, n1, _ = p2up(N)
            Wx = Wx[:, (n1 - 4):(n1 + N + 4)]
            w = phase_cwt_num(Wx, dt, difforder, gamma)
        return Wx, w

    N, dt, fs, difforder, nv = _process_args(x, scales, fs, t, nv, difftype,
                                             difforder, squeezing, maprange,
                                             wavelet, get_w)
    wavelet = Wavelet._init_if_not_isinstance(wavelet, N=N)

    # CWT with higher-order GMWs
    if isinstance(order, (tuple, list, range)) or order > 0:
        # keep padding for `trigdiff`
        kw = dict(wavelet=wavelet, scales=scales, fs=fs, nv=nv,
                  l1_norm=True, derivative=False, padtype=padtype, rpadded=True,
                  vectorized=vectorized, astensor=True,
                  cache_wavelet=cache_wavelet, nan_checks=nan_checks)
        _, n1, _ = p2up(N)
        average = isinstance(order, (tuple, list, range))

        Wx, scales = cwt(x, order=order, average=average, **kw)
        dWx = trigdiff(Wx, fs, rpadded=True, N=N, n1=n1)
        Wx = Wx[:, n1:n1 + N]
        if S.is_tensor(Wx):
            Wx = Wx.contiguous()

    scales, cwt_scaletype, *_ = process_scales(scales, N, wavelet, nv=nv,
                                               get_params=True)
    # regular CWT
    if order == 0:
        # l1_norm=True to spare a multiplication; for SSQ_CWT L1 & L2 are exactly
        # same anyway since we're inverting CWT over time-frequency plane
        rpadded = (difftype == 'numeric')
        Wx, scales, dWx = cwt(x, wavelet, scales=scales, fs=fs, nv=nv,
                              l1_norm=True, derivative=True, padtype=padtype,
                              rpadded=rpadded, vectorized=vectorized,
                              astensor=True, patience=patience,
                              cache_wavelet=cache_wavelet, nan_checks=nan_checks)

    # make copy of `Wx` if specified
    if preserve_transform is None:
        preserve_transform = not S.is_tensor(Wx)
    if preserve_transform:
        _Wx = (Wx.copy() if not S.is_tensor(Wx) else
               Wx.detach().clone())
    else:
        _Wx = Wx

    # gamma
    if gamma is None:
        gamma = 10 * (EPS64 if S.is_dtype(Wx, 'complex128') else EPS32)

    # compute `w` if `get_w` and free `dWx` from memory if `not get_dWx`
    if get_w:
        _Wx, w = _phase_transform(_Wx, dWx, N, dt, gamma, difftype, difforder)
        _dWx = None  # don't use in `ssqueeze`
        if not get_dWx:
            dWx = None
    else:
        w = None
        _dWx = dWx

    # default to same scheme used by `scales`
    if ssq_freqs is None:
        ssq_freqs = cwt_scaletype
    # affects `maprange` computation if non-tuple
    was_padded = bool(padtype is not None)

    # synchrosqueeze
    Tx, ssq_freqs = ssqueeze(_Wx, w, ssq_freqs, scales, fs=fs,
                             squeezing=squeezing, maprange=maprange,
                             wavelet=wavelet, gamma=gamma, was_padded=was_padded,
                             flipud=flipud, dWx=_dWx, transform='cwt')

    # postprocessing & return
    if difftype == 'numeric':
        Wx = Wx[:, 4:-4]
        Tx = Tx[:, 4:-4]
        w  = w[:,  4:-4] if w is not None else None

    if not astensor and S.is_tensor(Tx):
        Tx, Wx, w, dWx, scales, ssq_freqs = [
            g.cpu().numpy() if S.is_tensor(g) else g
            for g in (Tx, Wx, w, dWx, scales, ssq_freqs)]
    scales = scales.squeeze()

    if get_w and get_dWx:
        return Tx, Wx, ssq_freqs, scales, w, dWx
    elif get_w:
        return Tx, Wx, ssq_freqs, scales, w
    elif get_dWx:
        return Tx, Wx, ssq_freqs, scales, dWx
    else:
        return Tx, Wx, ssq_freqs, scales


def issq_cwt(Tx, wavelet='gmw', cc=None, cw=None):
    """Inverse synchrosqueezing transform of `Tx` with associated frequencies
    in `fs` and curve bands in time-frequency plane specified by `Cs` and
    `freqband`. This implements Eq. 15 of [1].

    # Arguments:
        Tx: np.ndarray
            Synchrosqueezed CWT of `x` (see `ssq_cwt`).
            (rows=~frequencies, cols=timeshifts)

        wavelet: str / tuple[str, dict] / `wavelets.Wavelet`
            Wavelet that was used to compute Tx, sampled in Fourier
            frequency domain.
                - str: name of builtin wavelet. `ssqueezepy.wavs()`
                - tuple[str, dict]: name of builtin wavelet and its configs.
                  E.g. `('morlet', {'mu': 5})`.
                - `wavelets.Wavelet` instance. Can use for custom wavelet.

        cc, cw: np.ndarray / None
            Curve centerpoints, and curve (vertical) widths (bandwidths),
            together defining the portion of Tx to invert over to extract
            K "components" per Modulation Model:
                x_k(t) = A_k(t) cos(phi_k(t)) + res;  k=0,...,K-1
            where K=len(c)==len(cw), and `res` is residual error (inversion
            over portion leftover/uncovered by cc, cw).
            None = full inversion.

    # Returns:
        x: np.ndarray [K x Tx.shape[1]]
            Components of reconstructed signal, and residual error.
            If cb & cw are None, x.shape == (Tx.shape[1],). See `cb, cw`.

    # Example:
        Tx, *_ = ssq_cwt(x, 'gmw')    # synchrosqueezed CWT
        x      = issq_cwt(Tx, 'gmw')  # reconstruction

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        3. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf

        4. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        synsq_cwt_iw.m
    """
    cc, cw, full_inverse = _process_component_inversion_args(cc, cw)

    if full_inverse:
        # Integration over all frequencies recovers original signal
        x = Tx.real.sum(axis=0)
    else:
        x = _invert_components(Tx, cc, cw)

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    Css = adm_ssq(wavelet)  # admissibility coefficient
    # *2 per analytic wavelet & taking real part; Theorem 4.5 [2]
    x *= (2 / Css)
    return x


def _invert_components(Tx, cc, cw):
    # Invert Tx around curve masks in the time-frequency plane to recover
    # individual components; last one is the remaining signal
    x = np.zeros((cc.shape[1] + 1, cc.shape[0]))
    TxRemainder = Tx.copy()

    for n in range(cc.shape[1]):
        TxMask = np.zeros(Tx.shape, dtype='complex128')
        upper_cc = np.clip(cc[:, n] + cw[:, n], 0, len(Tx))
        lower_cc = np.clip(cc[:, n] - cw[:, n], 0, len(Tx))

        # cc==-1 denotes no curve at that time,
        # removing such points from inversion
        upper_cc[np.where(cc[:, n] == -1)] = 0
        lower_cc[np.where(cc[:, n] == -1)] = 1
        for m in range(Tx.shape[1]):
            idxs = slice(lower_cc[m], upper_cc[m] + 1)
            TxMask[idxs, m] = Tx[idxs, m]
            TxRemainder[idxs, m] = 0
        x[n] = TxMask.real.sum(axis=0).T

    x[n + 1] = TxRemainder.real.sum(axis=0).T
    return x


def _process_component_inversion_args(cc, cw):
    if (cc is None) and (cw is None):
        full_inverse = True
    else:
        full_inverse = False
        if cc.ndim == 1:
            cc = cc.reshape(-1, 1)
        if cw.ndim == 1:
            cw = cw.reshape(-1, 1)
        cc = cc.astype('int32')
        cw = cw.astype('int32')
    return cc, cw, full_inverse


def phase_cwt(Wx, dWx, difftype='trig', gamma=None, parallel=None):
    """Calculate the phase transform at each (scale, time) pair:
          w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    See above Eq 20.3 in [1], or Eq 13 in [2].

    # Arguments:
        Wx: np.ndarray
            CWT of `x` (see `help(cwt)`).

        dWx: np.ndarray.
            Time-derivative of `Wx`, computed via frequency-domain differentiation
            (effectively, derivative of trigonometric interpolation; see [4]).

        difftype: str['trig', 'phase']
            Method by which to differentiate Wx (default='trig') to obtain
            instantaneous frequencies:
                    w(a,b) = Im( (1/2pi) * (1/Wx(a,b)) * d/db[Wx(a,b)] )

                - 'trig': using `dWx, the time-derivative of the CWT of `x`,
                computed via frequency-domain differentiation (effectively,
                derivative of trigonometric interpolation; see [4]). Implements
                as described in Sec IIIB of [2].
                - 'phase': differentiate by taking forward finite-difference of
                unwrapped angle of `Wx`. Does not support GPU or multi-threaded
                CPU execution.

        gamma: float / None
            See `help(ssqueezepy.ssq_cwt)`.

        parallel: bool (default `ssqueezepy.IS_PARALLEL()`)
            Whether to use multiple CPU threads (ignored if input is tensor).

    # Returns:
        w: np.ndarray
            Phase transform for each element of `Wx`. w.shape == Wx.shape.

    # References:
        1. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf

        4. The Exponential Accuracy of Fourier and Chebyshev Differencing Methods.
        E. Tadmor.
        http://webhome.auburn.edu/~jzl0097/teaching/math_8970/Tadmor_86.pdf

        5. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        phase_cwt.m
    """
    def _process_input(Wx, parallel, gamma):
        S.warn_if_tensor_and_par(Wx, parallel)
        gpu = S.is_tensor(Wx)
        if difftype != 'trig':
            if gpu:
                raise ValueError("`difftype != 'trig'` unsupported with tensor "
                                 "inputs.")
            elif parallel:
                raise ValueError("`difftype != 'trig'` unsupported with "
                                 "`parallel`.")
        if gamma is None:
            gamma = np.sqrt(EPS64 if S.is_dtype(Wx, 'complex128') else EPS32)
        return gamma, gpu

    gamma, gpu = _process_input(Wx, parallel, gamma)

    if difftype == 'trig':
        if gpu:
            w = phase_cwt_gpu(Wx, dWx, gamma)
        else:
            w = phase_cwt_cpu(Wx, dWx, gamma, parallel)

    elif difftype == 'phase':
        # TODO gives bad results; shouldn't we divide by Wx?
        u = np.unwrap(np.angle(Wx)).T
        w = np.vstack([np.diff(u, axis=0), u[-1] - u[0]]).T / (2*pi)
        np.abs(w, out=w)
        replace_under_abs(w, ref=Wx, value=gamma, replacement=np.inf)
    else:
        raise ValueError(f"unsupported `difftype` '{difftype}'; must be one of "
                         "'trig', 'phase'.")
    return w


def phase_cwt_num(Wx, dt, difforder=4, gamma=None):
    """Calculate the phase transform at each (scale, time) pair:
        w[a, b] = Im((1/2pi) * d/db (Wx[a,b]) / Wx[a,b])
    Uses numeric differentiation (1st, 2nd, or 4th order). See above Eq 20.3
    in [1], or Eq 13 in [2].

    # Arguments:
        Wx: np.ndarray
            CWT of `x` (see `cwt`).

        dt: float
            Sampling period (e.g. t[1] - t[0]).

        difforder: int[1, 2, 4]
            Order of differentiation (default=4).

        gamma: float
            See `help(ssqueezepy.ssq_cwt)`.

    # Returns:
        w: np.ndarray
            Phase transform via demodulated FM-estimates. w.shape == Wx.shape.

    # References:
        1. A Nonlinear squeezing of the CWT Based on Auditory Nerve Models.
        I. Daubechies, S. Maes.
        https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf

        2. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications.
        G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        3. Synchrosqueezing Toolbox, (C) 2014--present. E. Brevdo, G. Thakur.
        https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/
        phase_cwt_num.m
    """
    # unreliable; bad results on high freq pure tones
    def _differentiate(Wx, dt):
        if difforder in (2, 4):
            # append for differentiating
            Wxr = np.hstack([Wx[:, -2:], Wx, Wx[:, :2]])

        if difforder == 1:
            w = np.hstack([Wx[:, 1:] - Wx[:, :-1],
                           Wx[:, :1]  - Wx[:, -1:]])
            w /= dt
        elif difforder == 2:
            # calculate 2nd-order forward difference
            w = -Wxr[:, 4:] + 4 * Wxr[:, 3:-1] - 3 * Wxr[:, 2:-2]
            w /= (2 * dt)
        elif difforder == 4:
            # calculate 4th-order central difference
            w = -Wxr[:, 4:]
            w += Wxr[:, 3:-1] * 8
            w -= Wxr[:, 1:-3] * 8
            w += Wxr[:, 0:-4]
            w /= (12 * dt)
        return w

    if difforder not in (1, 2, 4):
        raise ValueError("`difforder` must be one of: 1, 2, 4 "
                         "(got %s)" % difforder)

    w = _differentiate(Wx, dt)

    # calculate inst. freq for each scale
    # 2*pi norm per discretized inverse FT rather than inverse DFT
    w = np.real(-1j * w / Wx) / (2*pi)

    # epsilon from Daubechies, H-T Wu, et al.
    # gamma from Brevdo, H-T Wu, et al.
    gamma = gamma or 10 * (EPS64 if Wx.dtype == np.complex128 else EPS32)
    w[np.abs(Wx) < gamma] = np.inf

    # see `phase_cwt`, though negatives may no longer be in minority
    w = np.abs(w)
    return w
