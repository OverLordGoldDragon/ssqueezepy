# -*- coding: utf-8 -*-
import numpy as np
from .ssqueezing import ssqueeze
from ._ssq_cwt import phase_cwt, phase_cwt_num
from ._ssq_stft import phase_stft, _make_Sfs
from .utils.common import EPS32, EPS64, p2up, trigdiff
from .utils import backend as S


def phase_ssqueeze(Wx, dWx=None, ssq_freqs=None, scales=None, Sfs=None, fs=1.,
                   t=None, squeezing='sum', maprange=None, wavelet=None,
                   gamma=None, was_padded=True, flipud=False,
                   rpadded=False, padtype=None, N=None, n1=None,
                   difftype=None, difforder=None,
                   get_w=False, get_dWx=False, transform='cwt'):
    """Take `phase_transform` then `ssqueeze`. Can be used on an arbitrary
    CWT/STFT-like time-frequency transform `Wx`.

    Experimental; prefer `ssq_cwt` & `ssq_stft`.

    # Arguments:
        Wx, dWx (see w), ssq_freqs, scales, Sfs, fs, t, squeezing, maprange,
        wavelet, gamma, was_padded, flipud:
            See `help(ssqueezing.ssqueeze)`.

        rpadded: bool (default None) / None
            Whether `Wx` (and `dWx`) is passed in padded. `True` will unpad
            `Wx` and `dWx`  before SSQ. Also, if `dWx` is None:
                - `rpadded==False`: will pad `Wx` in computing `dWx` if
                `padtype!=None`, then unpad both before SSQ
                - `rpadded==True`: won't pad `Wx` regardless of `padtype`

        padtype: str / None
            Used if `rpadded==False`. See `help(utils.padsignal)`. Note that
            padding `Wx` isn't same as passing padded `Wx` from `cwt`, but it
            can get close.

        N, n1: int / None
            Needed if `rpadded==True` to unpad `Wx` & `dWx` as `Wx[:, n1:n1 + N]`.

        difftype, difforder: str
            Used if `dWx = None` and `transform == 'cwt'`; see `help(ssq_cwt)`.

        get_w, get_dWx: bool
            See `help(ssq_cwt)`.

    # Returns:
        Tx, Wx, ssq_freqs, scales, Sfs, w, dWx
    """
    w, Wx, dWx, Sfs, gamma = phase_transform(
        Wx, dWx, difftype, difforder=difforder, gamma=gamma, rpadded=rpadded,
        padtype=padtype, N=N, n1=n1, get_w=get_w, fs=fs, transform=transform)

    if w is not None and not get_dWx:
        dWx = None

    if maprange is None:
        maprange = 'peak' if transform == 'cwt' else 'maximal'
    Tx, ssq_freqs = ssqueeze(Wx, w, ssq_freqs, scales, Sfs, fs=fs, t=t,
                             squeezing=squeezing, maprange=maprange,
                             wavelet=wavelet, gamma=gamma, was_padded=was_padded,
                             flipud=flipud, dWx=dWx, transform=transform)
    return Tx, Wx, ssq_freqs, scales, Sfs, w, dWx


def phase_transform(Wx, dWx=None, difftype='trig', difforder=4, gamma=None,
                    fs=1., Sfs=None, rpadded=False, padtype='reflect', N=None,
                    n1=None, get_w=False, transform='cwt'):
    """Unified method for CWT & STFT SSQ phase transforms.

    See `help(_ssq_cwt.phase_cwt)` and `help(_ssq_stft.phase_stft)`.
    """
    def _cwt(Wx, dWx, fs, gamma, N, n1, difftype, difforder, rpadded, padtype,
             get_w):
        # infer `N` and/or `n1`
        if N is None and not rpadded:
            N = Wx.shape[-1]
        if n1 is None:
            _, n1, _ = p2up(N)
        # compute `dWx` if not supplied
        if dWx is None:
            dWx = trigdiff(Wx, fs, padtype, rpadded, N=N, n1=n1, transform='cwt')

        if get_w:
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
                # calculate derivative numericly
                Wx = Wx[:, (n1 - 4):(n1 + N + 4)]
                dt = 1 / fs
                w = phase_cwt_num(Wx, dt, difforder, gamma)
        else:
            w = None
        return w, Wx, dWx

    def _stft(Wx, dWx, fs, gamma, Sfs, get_w):
        if Sfs is None:
            Sfs = _make_Sfs(Wx, fs)
        if get_w:
            w = phase_stft(Wx, dWx, Sfs, gamma)
        else:
            w = None
        return w, Wx, dWx, Sfs

    # validate args
    if transform == 'stft' and dWx is None:
        raise NotImplementedError("`phase_transform` without `dWx` for "
                                  "STFT is not currently supported.")
    if rpadded and N is None:
        raise ValueError("`rpadded=True` requires `N`")
    if Wx.ndim > 2 and get_w:
        raise NotImplementedError("`get_w=True` unsupported with batched input.")

    # gamma
    if gamma is None:
        gamma = np.sqrt(EPS64 if S.is_dtype(Wx, 'complex128') else EPS32)

    # take phase transform if `get_w` else only compute `dWx` (if None)
    if transform == 'cwt':
        w, Wx, dWx = _cwt(Wx, dWx, fs, gamma, N, n1, difftype, difforder,
                          rpadded, padtype, get_w)
        Sfs = None
    elif transform == 'stft':
        w, Wx, dWx, Sfs = _stft(Wx, dWx, fs, gamma, Sfs, get_w)

    return w, Wx, dWx, Sfs, gamma
