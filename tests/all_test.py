# -*- coding: utf-8 -*-
"""Lazy tests just to ensure nothing breaks
"""
#### Disable Numba JIT during testing, as pytest can't measure its coverage ##
# TODO find shorter way to do this
print("numba.njit is now monke")
print("numba.jit  is now monke")
def njit(fn):
    def decor(*args, **kw):
        return fn(*args, **kw)
    return decor

def jit(*args, **kw):
    def wrap(fn):
        def decor(*_args, **_kw):
            return fn(*_args, **_kw)
        return decor
    return wrap

import numba
njit_orig = numba.njit
jit_orig  = numba.jit
numba.njit = njit
numba.jit  = jit
##############################################################################
import pytest
import numpy as np
from ssqueezepy.wavelets import Wavelet, center_frequency, freq_resolution
from ssqueezepy.wavelets import time_resolution, _xifn
from ssqueezepy.wavelets import _aifftshift_even, _afftshift_even
from ssqueezepy.utils import cwt_scalebounds, buffer, est_riskshrink_thresh
from ssqueezepy.utils import window_resolution, window_area, padsignal
from ssqueezepy.utils import find_max_scale, unbuffer, _assert_positive_integer
from ssqueezepy.utils import _infer_scaletype, _process_fs_and_t, make_scales
from ssqueezepy._cwt import _icwt_norm
from ssqueezepy._stft import get_window
from ssqueezepy._ssq_stft import issq_stft
from ssqueezepy.visuals import hist, plot, plots, scat, plotscat, imshow
from ssqueezepy.visuals import wavelet_tf
from ssqueezepy.toolkit import lin_band, cos_f, sin_f, mad_rms, amax
from ssqueezepy.test_signals import TestSignals
from ssqueezepy.configs import gdefaults, float_if_number
from ssqueezepy import ssq_cwt, issq_cwt, cwt, icwt, ssqueeze
from ssqueezepy import _gmw

#### Ensure cached imports reloaded ##########################################
from types import ModuleType
from imp import reload
import ssqueezepy

reload(numba)
numba.njit = njit
numba.jit  = jit
reload(ssqueezepy)
for name in dir(ssqueezepy):
    obj = getattr(ssqueezepy, name)
    if isinstance(obj, ModuleType):
        reload(obj)
##############################################################################

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0


def test_ssq_cwt():
    x = np.random.randn(64)
    for wavelet in ('morlet', ('morlet', {'mu': 20}), 'bump'):
        Tx, *_ = ssq_cwt(x, wavelet)
        issq_cwt(Tx, wavelet)

    kw = dict(x=x, wavelet='morlet')
    params = dict(
        squeezing=('lebesgue',),
        scales=('linear', 'log:minimal', 'linear:naive',
                np.power(2**(1/16), np.arange(1, 32))),
        difftype=('phase', 'numeric'),
        padtype=('zero', 'replicate'),
        maprange=('maximal', 'energy', 'peak', (1, 32)),
    )

    for name in params:
        for value in params[name]:
            try:
                _ = ssq_cwt(**kw, **{name: value})
            except Exception as e:
                raise Exception(f"{name}={value} failed with:\n{e}")

    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=2)
    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=1)


def test_cwt():
    x = np.random.randn(64)
    Wx, *_ = cwt(x, 'morlet', vectorized=True)
    _ = icwt(Wx, 'morlet', one_int=True)
    _ = icwt(Wx, 'morlet', one_int=False)

    Wx2, *_ = cwt(x, 'morlet', vectorized=False)
    mae = np.mean(np.abs(Wx - Wx2))
    assert mae <= 1e-16, f"MAE = {mae} > 1e-16 for for-loop vs vectorized `cwt`"

    _ = est_riskshrink_thresh(Wx, nv=32)
    _ = _icwt_norm(scaletype='linear', l1_norm=False)

    x[0] = np.nan
    x[1] = np.inf
    x[2] = -np.inf
    _ = cwt(x, 'morlet', vectorized=False, derivative=True, l1_norm=False)


def test_ssq_stft():
    Tsx = np.random.randn(128, 128)
    pass_on_error(issq_stft, Tsx, modulated=False)
    pass_on_error(issq_stft, Tsx, hop_len=2)


def test_wavelets():
    for wavelet in ('morlet', ('morlet', {'mu': 4}), 'bump'):
        wavelet = Wavelet(wavelet)

    wavelet = Wavelet(('morlet', {'mu': 5}))
    wavelet.viz(name='overview')
    wavelet.info(nondim=1)
    wavelet.info(nondim=0)

    #### Visuals #############################################################
    for name in wavelet.VISUALS:
        if 'anim:' in name:  # heavy-duty computations, skip animating
            kw = {'testing': True}
        else:
            kw = {}
        try:
            wavelet.viz(name, N=256, **kw)
        except TypeError as e:
            if "positional argument" not in str(e):
                raise TypeError(e)
            try:
                wavelet.viz(name, scale=10, N=256, **kw)
            except TypeError as e:
                if "positional argument" not in str(e):
                    raise TypeError(e)
                wavelet.viz(name, scales='log', N=256, **kw)

    _ = cwt_scalebounds(wavelet, N=512, viz=3)

    #### misc ################################################################
    wavelet = Wavelet(lambda x: x)
    _ = _xifn(scale=10, N=128)


def test_toolkit():
    Tx = np.random.randn(20, 20)
    Cs, freqband = lin_band(Tx, slope=1, offset=.1, bw=.025)

    _ = cos_f([1], N=64)
    _ = sin_f([1], N=64)
    _ = amax(Tx)
    _ = mad_rms(np.random.randn(10), np.random.randn(10))


def test_visuals():
    x = np.random.randn(10)
    hist(x, show=1, stats=1)

    y = x * (1 + 1j)
    plot(y, complex=1, c_annot=1, vlines=1, ax_equal=1,
         xticks=np.arange(len(y)), yticks=y)
    plot(y, abs=1, vert=1, dx1=1, ticks=0)

    scat(x, vlines=1, hlines=1)
    scat(y, complex=1, ticks=0)
    plotscat(y, show=1, xlims=(-1, 1), dx1=1, ylabel="5")

    plots([y, y], tight=1, show=1)
    plots([y, y], nrows=2)
    plots([y, y], ncols=2)

    g = np.random.randn(4, 4)
    imshow(g * (1 + 2j), complex=1)
    imshow(g, ridge=1, ticks=0)

    pass_on_error(plot, None, None)
    pass_on_error(wavelet_tf, 'morlet', notext=True)


def test_utils():
    _ = buffer(np.random.randn(20), 4, 1)

    wavelet = Wavelet(('morlet', {'mu': 6}))
    _ = center_frequency(wavelet, viz=1)
    _ = freq_resolution( wavelet, viz=1, scale=3, force_int=0)
    _ = time_resolution( wavelet, viz=1)

    xh = np.random.randn(128)
    xhs = np.zeros(xh.size)
    _aifftshift_even(xh, xhs)
    _afftshift_even(xh, xhs)

    _ = padsignal(xh, padlength=len(xh)*2, padtype='symmetric')
    _ = padsignal(xh, padlength=len(xh)*2, padtype='wrap')
    x2d = np.random.randn(4, 64)
    _ = padsignal(x2d, padlength=96, padtype='symmetric')

    g = np.ones((128, 200))
    unbuffer(g, xh, 1, n_fft=len(xh), N=None, win_exp=0)
    unbuffer(g, xh, 1, n_fft=len(xh), N=g.shape[1], win_exp=2)

    #### errors / warnings ###################################################
    pass_on_error(find_max_scale, 1, 1, -1, -1)

    pass_on_error(cwt_scalebounds, 1, 1, preset='etc', min_cutoff=0)
    pass_on_error(cwt_scalebounds, 1, 1, min_cutoff=-1)
    pass_on_error(cwt_scalebounds, 1, 1, min_cutoff=.2, max_cutoff=.1)
    pass_on_error(cwt_scalebounds, 1, 1, cutoff=0)

    pass_on_error(_assert_positive_integer, -1, 'w')

    pass_on_error(_infer_scaletype, 1)
    pass_on_error(_infer_scaletype, np.array([1]))
    pass_on_error(_infer_scaletype, np.array([1., 2., 5.]))

    pass_on_error(_process_fs_and_t, 1, np.array([1]), 2)
    pass_on_error(_process_fs_and_t, 1, np.array([1., 2, 4]), 3)
    pass_on_error(_process_fs_and_t, -1, None, 1)

    pass_on_error(make_scales, 128, scaletype='banana')
    pass_on_error(padsignal, np.random.randn(3, 4, 5))


def test_anim():
    # bare minimally (still takes long, but covers many lines of code)
    wavelet = Wavelet(('morlet', {'mu': 6}))
    wavelet.viz('anim:time-frequency', N=8, scales=np.linspace(10, 20, 3))


def test_ssqueezing():
    Wx = np.random.randn(4, 4)
    w = np.abs(Wx)

    pass_on_error(ssqueeze, Wx, w, transform='greenland')
    pass_on_error(ssqueeze, Wx, w, transform='cwt', scales=None)
    pass_on_error(ssqueeze, Wx, w, transform='cwt', wavelet=None,
                   maprange='maximal')
    pass_on_error(ssqueeze, Wx, w, transform='stft', maprange='minimal')
    pass_on_error(ssqueeze, Wx, w, transform='stft', ssq_freqs='linear')
    pass_on_error(ssqueeze, Wx, w, transform='abs')
    pass_on_error(ssqueeze, Wx, w, squeezing='big_bird')
    pass_on_error(ssqueeze, Wx, w, squeezing=lambda x: x**2)
    pass_on_error(ssqueeze, Wx, w, squeezing='abs')


def test_get_window():
    _ = get_window('hann', win_len=128, n_fft=None)
    pass_on_error(get_window, 1, 2)


def test_windows():
    window = get_window(None, win_len=100, n_fft=128)

    window_area(window, time=True,  frequency=True)
    window_area(window, time=True,  frequency=False)
    window_area(window, time=False, frequency=True)
    window_resolution(window)


def test_morse_utils():
    """Test miscellaneous utility funcs."""
    _gmw.morseafun(3, 60, 1, 'bandpass')
    _gmw.morseafun(3, 60, 1, 'energy')

    for n_out in range(1, 5):
        _gmw.morsefreq(3, 60, n_out=n_out)
        _gmw._morsemom(1, 3, 60, n_out=n_out)
    _gmw._moments_to_cumulants(np.random.uniform(0, 1, 5))

    pass_on_error(_gmw._check_args, gamma=-1)
    pass_on_error(_gmw._check_args, beta=-1)
    pass_on_error(_gmw._check_args, norm='cactus')
    pass_on_error(_gmw._check_args, scale=-1)


def testtest_signals():
    tsigs = TestSignals()
    pass_on_error(tsigs, dft='doot')

    fn = lambda *args, **kw: (np.random.randn(100, 100), {})
    tsigs.test_transforms(fn)

    pass_on_error(tsigs._process_input, 'etc:t')
    pass_on_error(tsigs._process_input, ['a', 1])
    pass_on_error(tsigs._process_input, ['a', (1, 2)])

    backup = tsigs.default_args.copy()
    tsigs.default_args['am-cosine'] = dict(amin=.1)
    pass_on_error(tsigs._process_input, 'am-cosine')
    tsigs.default_args['am-cosine'] = 2
    pass_on_error(tsigs._process_input, 'am-cosine')
    tsigs.default_args.update(backup)


def test_configs():
    pass_on_error(gdefaults, None)
    pass_on_error(float_if_number, 'number')


def pass_on_error(fn, *args, **kw):
    try: fn(*args, **kw)
    except: pass


if __name__ == '__main__':
    if VIZ:
        test_ssq_cwt()
        test_cwt()
        test_ssq_stft()
        test_wavelets()
        test_toolkit()
        test_visuals()
        test_utils()
        test_anim()
        test_ssqueezing()
        test_get_window()
        test_windows()
        test_morse_utils()
        test_configs()
    else:
        pytest.main([__file__, "-s"])

        # restore original in case it matters for future testing
        reload(numba)
        numba.njit = njit_orig
        numba.jit  = jit_orig
        reload(ssqueezepy)
        for name in dir(ssqueezepy):
            obj = getattr(ssqueezepy, name)
            if isinstance(obj, ModuleType):
                reload(obj)
        print("numba.njit is no longer monkey")
        print("numba.jit  is no longer monkey")
