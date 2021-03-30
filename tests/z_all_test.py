# -*- coding: utf-8 -*-
"""Lazy tests just to ensure nothing breaks.

`z_` to ensure test runs last per messing with module namespaces.
"""
#### Disable Numba JIT during testing, as pytest can't measure its coverage ##
# TODO find shorter way to do this
def njit(fn):
    def decor(*args, **kw):
        return fn(*args, **kw)
    return decor

def jit(*args, **kw):
    def wrap(fn):
        return fn
    return wrap

import numba
njit_orig = numba.njit
jit_orig  = numba.jit
##############################################################################
import os
import pytest
import numpy as np
from ssqueezepy._cwt import _icwt_norm
from ssqueezepy.configs import gdefaults
from ssqueezepy import Wavelet, TestSignals, ssq_cwt, issq_cwt, cwt, icwt
from ssqueezepy import ssq_stft, issq_stft, ssqueeze, get_window, extract_ridges
from ssqueezepy import _gmw, utils, visuals, wavelets, toolkit
from ssqueezepy.utils.common import find_closest_parallel_is_faster
from ssqueezepy.ssqueezing import _check_ssqueezing_args

#### Ensure cached imports reloaded ##########################################
from types import ModuleType
from imp import reload
import ssqueezepy


def reload_all():
    reload(ssqueezepy)
    for name in dir(ssqueezepy):
        obj = getattr(ssqueezepy, name)
        if isinstance(obj, ModuleType) and name in ssqueezepy._modules_toplevel:
            reload(obj)

##############################################################################

# no visuals here but 1 runs as regular script instead of pytest, for debugging
VIZ = 0

def test_numba_monke():
    """Run this *at test time* rather than collection so changes
    don't apply to other test files. This is for coverage of @jit'd funcs.
    """
    numba.njit = njit
    numba.jit  = jit
    print("numba.njit is now monke")
    print("numba.jit  is now monke")

    reload(numba)
    numba.njit = njit
    numba.jit  = jit
    reload_all()


def test_ssq_cwt():
    os.environ['SSQ_GPU'] = '0'  # in case concurrent tests set it to '1'
    np.random.seed(5)
    x = np.random.randn(64)
    for wavelet in ('morlet', ('morlet', {'mu': 20}), 'bump'):
        Tx, *_ = ssq_cwt(x, wavelet)
        issq_cwt(Tx, wavelet)

    kw = dict(x=x, wavelet='morlet')
    params = dict(
        squeezing=('lebesgue',),
        scales=('linear', 'log:minimal', 'linear:naive',
                np.power(2**(1/8), np.arange(1, 32))),
        difftype=('phase', 'numeric'),
        padtype=('zero', 'replicate'),
        maprange=('maximal', 'energy', 'peak', (1, 32)),
    )

    for name in params:
        for value in params[name]:
            errored = True
            try:
                if name == 'maprange' and value in ('maximal', (1, 32)):
                    _ = ssq_cwt(**kw, **{name: value}, scales='log', get_w=1)
                else:
                    _ = ssq_cwt(**kw, **{name: value}, get_w=1)
                errored = False
            finally:
                if errored:
                    print(f"\n{name}={value} failed\n")

    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=2, get_w=1)
    _ = ssq_cwt(x, wavelet, fs=2, difftype='numeric', difforder=1, get_w=1)


def test_cwt():
    os.environ['SSQ_GPU'] = '0'
    x = np.random.randn(64)
    Wx, *_ = cwt(x, 'morlet', vectorized=True)
    _ = icwt(Wx, 'morlet', one_int=True)
    _ = icwt(Wx, 'morlet', one_int=False)

    Wx2, *_ = cwt(x, 'morlet', vectorized=False)
    mae = np.mean(np.abs(Wx - Wx2))
    assert mae <= 1e-16, f"MAE = {mae} > 1e-16 for for-loop vs vectorized `cwt`"

    _ = utils.est_riskshrink_thresh(Wx, nv=32)
    _ = _icwt_norm(scaletype='linear', l1_norm=False)

    x[0] = np.nan
    x[1] = np.inf
    x[2] = -np.inf
    _ = cwt(x, 'morlet', vectorized=False, derivative=True, l1_norm=False)


def test_ssq_stft():
    os.environ['SSQ_GPU'] = '0'
    Tsx = np.random.randn(128, 128)
    pass_on_error(issq_stft, Tsx, modulated=False)
    pass_on_error(issq_stft, Tsx, hop_len=2)


def test_wavelets():
    os.environ['SSQ_GPU'] = '0'
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

    _ = utils.cwt_scalebounds(wavelet, N=512, viz=3)

    #### misc ################################################################
    wavelet = Wavelet(lambda x: x)
    _ = wavelets._xifn(scale=10, N=128)


def test_toolkit():
    Tx = np.random.randn(20, 20)
    Cs, freqband = toolkit.lin_band(Tx, slope=1, offset=.1, bw=.025)

    _ = toolkit.cos_f([1], N=64)
    _ = toolkit.sin_f([1], N=64)
    _ = toolkit.where_amax(Tx)
    _ = toolkit.mad_rms(np.random.randn(10), np.random.randn(10))


def test_visuals():
    os.environ['SSQ_GPU'] = '0'
    x = np.random.randn(10)
    visuals.hist(x, show=1, stats=1)

    y = x * (1 + 1j)
    visuals.plot(y, complex=1, c_annot=1, vlines=1, ax_equal=1,
         xticks=np.arange(len(y)), yticks=y)
    visuals.plot(y, abs=1, vert=1, dx1=1, ticks=0)

    visuals.scat(x, vlines=1, hlines=1)
    visuals.scat(y, complex=1, ticks=0)
    visuals.plotscat(y, show=1, xlims=(-1, 1), dx1=1, ylabel="5")

    visuals.plots([y, y], tight=1, show=1)
    visuals.plots([y, y], nrows=2)
    visuals.plots([y, y], ncols=2)

    g = np.random.randn(4, 4)
    visuals.imshow(g * (1 + 2j), complex=1)
    visuals.imshow(g, ridge=1, ticks=0)

    pass_on_error(visuals.plot, None, None)
    pass_on_error(visuals.wavelet_tf, 'morlet', notext=True)


def test_utils():
    os.environ['SSQ_GPU'] = '0'
    _ = utils.buffer(np.random.randn(20), 4, 1)

    wavelet = Wavelet(('morlet', {'mu': 6}))
    _ = wavelets.center_frequency(wavelet, viz=1)
    _ = wavelets.freq_resolution( wavelet, viz=1, scale=3, force_int=0)
    _ = wavelets.time_resolution( wavelet, viz=1)

    xh = np.random.randn(128)
    xhs = np.zeros(xh.size)
    wavelets._aifftshift_even(xh, xhs)
    wavelets._afftshift_even(xh, xhs)

    _ = utils.padsignal(xh, padlength=len(xh)*2, padtype='symmetric')
    _ = utils.padsignal(xh, padlength=len(xh)*2, padtype='wrap')
    x2d = np.random.randn(4, 64)
    _ = utils.padsignal(x2d, padlength=96, padtype='symmetric')

    g = np.ones((128, 200))
    utils.unbuffer(g, xh, 1, n_fft=len(xh), N=None, win_exp=0)
    utils.unbuffer(g, xh, 1, n_fft=len(xh), N=g.shape[1], win_exp=2)

    scales = utils.process_scales('log', 1024, Wavelet())
    _ = utils.find_downsampling_scale(Wavelet(), scales, method='any',
                                      viz_last=1)
    _ = utils.find_downsampling_scale(Wavelet(), scales, method='all')

    #### errors / warnings ###################################################
    pass_on_error(utils.find_max_scale, 1, 1, -1, -1)

    pass_on_error(utils.cwt_scalebounds, 1, 1, preset='etc', min_cutoff=0)
    pass_on_error(utils.cwt_scalebounds, 1, 1, min_cutoff=-1)
    pass_on_error(utils.cwt_scalebounds, 1, 1, min_cutoff=.2, max_cutoff=.1)
    pass_on_error(utils.cwt_scalebounds, 1, 1, cutoff=0)

    pass_on_error(utils.cwt_utils._assert_positive_integer, -1, 'w')

    pass_on_error(utils.infer_scaletype, 1)
    pass_on_error(utils.infer_scaletype, np.array([1]))
    pass_on_error(utils.infer_scaletype, np.array([1., 2., 5.]))

    pass_on_error(utils._process_fs_and_t, 1, np.array([1]), 2)
    pass_on_error(utils._process_fs_and_t, 1, np.array([1., 2, 4]), 3)
    pass_on_error(utils._process_fs_and_t, -1, None, 1)

    pass_on_error(utils.make_scales, 128, scaletype='banana')
    pass_on_error(utils.padsignal, np.random.randn(3, 4, 5))


def test_anim():
    # bare minimally (still takes long, but covers many lines of code)
    wavelet = Wavelet(('morlet', {'mu': 6}))
    wavelet.viz('anim:time-frequency', N=8, scales=np.linspace(10, 20, 3))


def test_ssqueezing():
    os.environ['SSQ_GPU'] = '0'
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

    utils.window_area(window, time=True,  frequency=True)
    utils.window_area(window, time=True,  frequency=False)
    utils.window_area(window, time=False, frequency=True)
    utils.window_resolution(window)


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


def test_test_signals():
    os.environ['SSQ_GPU'] = '0'
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


def test_cwt_higher_order():
    os.environ['SSQ_GPU'] = '0'
    N = 256

    tsigs = TestSignals()
    x, t = tsigs.par_lchirp(N=N)
    x += x[::-1]

    for noise in (False, True):
        if noise:
            x += np.random.randn(len(x))
        Wx_k, scales = cwt(x, 'gmw', order=range(3), average=False)

        visuals.viz_cwt_higher_order(Wx_k, scales, 'gmw')
        print("=" * 80)

    _ = cwt(x, ('gmw', {'norm': 'energy'}), order=(0, 1), average=True,
            l1_norm=False)
    _ = cwt(x, 'gmw', order=1, average=False, derivative=True)


def test_viz_gmw_orders():
    os.environ['SSQ_GPU'] = '0'
    N = 256
    gamma, beta, norm = 3, 60, 'bandpass'
    n_orders = 3
    scale = 5
    visuals.viz_gmw_orders(N, n_orders, scale, gamma, beta, norm)


def test_trigdiff():
    """Ensure `trigdiff` matches `cwt(derivative=True)`."""
    os.environ['SSQ_GPU'] = '0'
    N = 256
    x = np.random.randn(N)
    Wx, _, dWx = cwt(x, derivative=True, rpadded=True)

    _, n1, _ = utils.p2up(N)
    dWx2 = utils.trigdiff(Wx, rpadded=True, N=N, n1=n1)
    dWx = dWx[:, n1:n1+N]

    mae = np.mean(np.abs(dWx - dWx2))
    th = 1e-15 if dWx.dtype == np.cfloat else 1e-7
    assert mae < th, mae


def test_logscale_transition_idx():
    """Ensure the function splits `idx` such that `scales` are split as
    `[scales[:idx], scales[idx:]]`
    """
    scales = np.exp(np.linspace(0, 5, 512))
    idx = 399
    for downsample in (2, 3, 4):
        scales1 = scales[:idx]
        scales2 = scales[idx + downsample - 1::downsample]
        scales = np.hstack([scales1, scales2])

        tidx = utils.logscale_transition_idx(scales)
        assert idx == tidx, "{} != {}".format(idx, tidx)


def test_dtype():
    """Ensure `cwt` and `ssq_cwt` compute at appropriate precision depending
    on `Wavelet.dtype`, returning float32 & complex64 arrays for single precision.
    """
    os.environ['SSQ_GPU'] = '0'
    wav32, wav64 = Wavelet(dtype='float32'), Wavelet(dtype='float64')
    x = np.random.randn(256)
    outs32    = ssq_cwt(x, wav32)
    outs64    = ssq_cwt(x, wav64)
    outs32_o2 = ssq_cwt(x, wav32, order=2)

    names = ('Tx', 'Wx', 'ssq_freqs', 'scales', 'w', 'dWx')
    outs32    = {k: v for k, v in zip(names, outs32)}
    outs32_o2 = {k: v for k, v in zip(names, outs32_o2)}
    outs64    = {k: v for k, v in zip(names, outs64)}

    for k, v in outs32.items():
        if k == 'ssq_freqs':
            assert v.dtype == np.float64, ("float32", k, v.dtype)
            continue
        assert v.dtype in (np.float32, np.complex64),  ("float32", k, v.dtype)
    for k, v in outs32_o2.items():
        if k == 'ssq_freqs':
            assert v.dtype == np.float64, ("float32", k, v.dtype)
            continue
        assert v.dtype in (np.float32, np.complex64),  ("float32", k, v.dtype)
    for k, v in outs64.items():
        if k == 'ssq_freqs':
            assert v.dtype == np.float64, ("float32", k, v.dtype)
            continue
        assert v.dtype in (np.float64, np.complex128), ("float64", k, v.dtype)


def test_find_closest_parallel_is_faster():
    find_closest_parallel_is_faster((50, 200))


def test_wavelet_info():
    for parallel in ('0', '1'):
        os.environ['SSQ_PARALLEL'] = parallel
        Wavelet(('gmw', {'norm': 'bandpass'})).info()
        Wavelet(('gmw', {'norm': 'energy'})).info()
        Wavelet(('gmw', {'norm': 'bandpass', 'order': 1})).info()
        Wavelet(('gmw', {'norm': 'energy', 'order': 1})).info()

        for name in ('morlet', 'bump', 'cmhat', 'hhhat'):
            Wavelet(name).info()


def test_ridge_extraction():
    """For @jit coverage."""
    Wx, scales = cwt(np.random.randn(128))
    _ = extract_ridges(Wx, scales, transform='cwt', parallel=False)
    _ = extract_ridges(Wx, scales, transform='cwt', parallel=True)


def test_check_ssqueezing_args():
    pass_on_error(_check_ssqueezing_args, 1)
    pass_on_error(_check_ssqueezing_args, 'sum', maprange=('a', 'b'))
    pass_on_error(_check_ssqueezing_args, 'sum', maprange=dict(a=1))
    pass_on_error(_check_ssqueezing_args, 'sum', maprange='peak')
    pass_on_error(_check_ssqueezing_args, 'sum', difftype='o')
    pass_on_error(_check_ssqueezing_args, 'sum', difftype='phase', get_w=0)
    pass_on_error(_check_ssqueezing_args, 'sum', difftype='phase', difforder=4,
                  get_w=1)
    pass_on_error(_check_ssqueezing_args, 'sum', difftype='numeric', difforder=3,
                  get_w=1)

    _check_ssqueezing_args('sum', difftype='phase', difforder=4, get_w=1)
    _check_ssqueezing_args('sum', maprange='peak', transform='stft')


def test_misc():
    _ = cwt(np.random.randn(128), 'gmw', cache_wavelet=True)
    _ = cwt(np.random.randn(128), Wavelet(), cache_wavelet=True, vectorized=False)

    _ = ssq_stft(np.random.randn(100), get_w=1, get_dWx=1)

    pass_on_error(cwt, np.random.randn(2, 2, 2))
    pass_on_error(cwt, 5)
    pass_on_error(ssq_stft, np.random.randn(2, 2, 2), get_w=1)


def test_configs():
    pass_on_error(gdefaults, None)


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
        test_test_signals()
        test_cwt_higher_order()
        test_viz_gmw_orders()
        test_trigdiff()
        test_logscale_transition_idx()
        test_dtype()
        test_find_closest_parallel_is_faster()
        test_wavelet_info()
        test_ridge_extraction()
        test_check_ssqueezing_args()
        test_misc()
        test_configs()
    else:
        pytest.main([__file__, "-s"])

    # restore original in case it matters for future testing
    reload(numba)
    numba.njit = njit_orig
    numba.jit  = jit_orig
    reload_all()
    print("numba.njit is no longer monke")
    print("numba.jit  is no longer monke")
