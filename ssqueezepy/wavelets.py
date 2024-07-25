# -*- coding: utf-8 -*-
import numpy as np
import gc
from numba import jit
from types import FunctionType
from scipy import integrate
from .algos import find_maximum
from .configs import gdefaults, USE_GPU, IS_PARALLEL
from .utils import backend as S
from .utils.fft_utils import ifft, fftshift, ifftshift
from .utils.backend import torch, Q, atleast_1d


class Wavelet():
    """Central wavelet class. `__call__` computes Fourier frequency-domain
    wavelet, `psih`, `.psifn` computes time-domain wavelet, `psi`.

    `Wavelet.SUPPORTED` for names of built-in wavelets passable to `__init__()`;
    `Wavelet.VISUALS`   for names of visualizations    passable to `viz()`.
    `viz()` to run visuals, `info()` to print relevant wavelet info.

    # Arguments:
        wavelet: str / tuple[str, dict] /FunctionType
            Name of supported wavelet (must be one of `Wavelet.SUPPORTED`)
            or custom function. Or tuple, name of wavelet and its configs,
            e.g. `('morlet', {'mu': 5})`.

        N: int
            Default length of wavelet.

        dtype: str / type (np.dtype) / None
            dtype at which wavelets are generated; can't change after __init__.
            Must be one of `Wavelet.DTYPES`. If None, uses value from
            `configs.ini`, global (if set) or wavelet-specific.

            'float32' is unsupported for GMW's `norm='energy'` and will be
            overridden by 'float64' (with a warning if it was passed to __init__).

    # Example:
        wavelet = Wavelet(('morlet', {'mu': 7}), N=1024)
        plt.plot(wavelet(scale=8))
    """
    SUPPORTED = {'gmw', 'morlet', 'bump', 'cmhat', 'hhhat'}
    VISUALS = {'time-frequency', 'heatmap', 'waveforms', 'filterbank',
               'harea', 'std_t', 'std_w', 'anim:time-frequency'}
    DTYPES = {'float32', 'float64'}
    # TODO ensure everything is accounted
    # Attributes whose data is stored on GPU (if env flag 'SSQ_GPU' == '1')
    ON_GPU = {'xi', '_Psih', '_Psih_scale'}
    # Time-frequency attributes
    TF_PROPS = {'wc', 'wc_ct', 'scalec_ct', 'std_t', 'std_w',
                'std_t_d', 'std_w_d'}

    def __init__(self, wavelet='gmw', N=1024, dtype=None):
        self._dtype = self._process_dtype(dtype, as_str=True
                                          ) if dtype is not None else None
        self._validate_and_set_wavelet(wavelet)

        self.N = N  # also sets _xi

    #### Main methods / properties ###########################################
    def __call__(self, w=None, *, scale=None, N=None, nohalf=True, imag_th=1e-8):
        """wavelet(w) if called with positional argument, w = float or array, else
           wavelet(scale * xi), where `xi` is recomputed if `N` is not None.

        `nohalf=False` (default=True) halves the Nyquist bin for even-length
        psih to ensure proper time-domain wavelet decay and analyticity:
            https://github.com/jonathanlilly/jLab/issues/13

        If evaluated wavelet's imaginary component is less than `imag_th`*(sum of
        real), will drop it; set to None to disable.
        """
        if w is not None:
            psih = self.fn(S.asarray(w, self.dtype))
        else:
            psih = self.fn(self.xifn(scale, N))

        if not nohalf:
            psih = self._halve_nyquist(psih)
        if (S.is_dtype(psih, ('complex64', 'complex128')) and
                (imag_th is not None) and
                (psih.imag.sum() / psih.real.sum() < imag_th)):
            psih = psih.real
        return psih

    @staticmethod
    def _halve_nyquist(psih):
        """https://github.com/jonathanlilly/jLab/issues/13"""
        N = len(psih) if psih.ndim == 1 else psih.shape[1]
        if N % 2 == 0:
            if psih.ndim == 1:
                psih[N//2] /= 2
            else:
                psih[:, N//2] /= 2
        return psih

    def psifn(self, w=None, *, scale=None, N=None):
        """Compute time-domain wavelet; simply `ifft(psih)` with appropriate
        extra steps.
        """
        psih = self(w, scale=scale, N=N, nohalf=False)
        if psih.ndim in (1, 2):
            pn = (-1)**S.arange(psih.shape[-1], dtype=self.dtype)
        else:
            raise ValueError("`psih` must yield to 1D or 2D (got %s)" % psih.ndim)

        # * pn = freq-domain spectral reversal to center time-domain wavelet
        psi = ifft(psih * pn, axis=-1)
        return psi

    def xifn(self, scale=None, N=None):
        """Computes `xi`, radian frequencies at which `wavelet` is sampled,
        as fraction of sampling frequency: 0 to pi & -pi to 0, scaled by
        `scale` - or more precisely:

            N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
            N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
        """
        if isinstance(scale, (np.ndarray, torch.Tensor)) and len(scale) > 1:
            if scale.squeeze().ndim > 1:
                raise ValueError("2D `scale` unsupported")
            elif scale.ndim == 1:
                scale = scale.reshape(-1, 1)  # add dim for proper broadcast
        elif scale is None:
            scale = 1.

        scale = S.asarray(scale, dtype=self.dtype)
        if N is None:
            xi = scale * self.xi
        else:
            xi = scale * S.asarray(_xifn(scale=1., N=N,
                                         dtype=getattr(np, self.dtype)))
        return xi

    def Psih(self, scale=None, N=None, nohalf=True):
        """Return pre-computed `psih` at scale(s) `scale` of length `N` if
        same `scale` & `N` were passed previously, else compute anew.

        `dtype` will override `self.dtype` if not None.

        If both `scale` & `N` are None, will return previously computed `Psih`.
        """
        pN = getattr(self, '_Psih_N', S.array([-1]))
        ps = getattr(self, '_Psih_scale', S.array([-1]))
        N_is_None = N is None
        N = N or self.N
        if ((scale is None and N_is_None) or
                (N == pN and (len(scale) == len(ps) and S.allclose(scale, ps)))):
            return self._Psih

        # first empty existing to free memory
        if getattr(self, '_Psih', None) is not None:
            self._Psih = None
            gc.collect()

        self._Psih = self(scale=scale, N=N, nohalf=nohalf)
        self._Psih_N = N
        self._Psih_scale = scale
        return self._Psih

    @property
    def N(self):
        """Default value used when `N` is not passed to a `Wavelet` method."""
        return self._N

    @N.setter
    def N(self, value):
        """Ensure `xi` always matches `N`."""
        self._N = value
        self._xi = S.asarray(_xifn(scale=1, N=value,
                                   dtype=getattr(np, self.dtype)))

    @property
    def xi(self):
        """`xi` computed at `scale=1` and `N=self.N`. See `help(Wavelet.xifn)`."""
        return self._xi

    @property
    def dtype(self):
        """dtype at which psih and psi are generated; can't change post-init."""
        return self._dtype

    #### Properties ##########################################################
    @property
    def name(self):
        """Name of underlying freq-domain function, processed by
        `wavelets._fn_to_name`.
        """
        return _fn_to_name(self.fn)

    @property
    def config_str(self):
        """`self.config` formatted into a nice string."""
        if self.config:
            cfg = ""
            for k, v in self.config.items():
                if k in ('norm', 'centered_scale', 'dtype'):
                    # too long, no real need
                    continue
                elif k == 'order' and v == 0:
                    # no need to include base wavelet's order
                    continue
                elif isinstance(v, float) and v.is_integer():
                    v = int(v)
                cfg += "{}={}, ".format(k, v)
            cfg = cfg.rstrip(', ')
        else:
            cfg = "Default configs"
        return cfg

    @property
    def wc(self):
        """Energy center frequency at scale=scalec_ct [(radians*cycles)/samples]

        Ideally we'd compute at `scale=1`, but that's trouble for 'energy' center
        frequency; see `help(wavelets.center_frequency)`. Away from scale
        extrema, 'energy' and 'peak' are same for bell-like |wavelet(w)|^2.

        Reported as "dimensional" in `info()` since it's tied to same `scale`
        used for computing `std_t_d` & `std_t_w`
        """
        if getattr(self, '_wc', None) is None:
            self._wc = center_frequency(self, scale=self.scalec_ct, N=self.N,
                                        kind='energy')
        return self._wc

    @property
    def wc_ct(self):
        """'True' radian peak center frequency, i.e. `w` which maximizes the
        underlying continuous-time function. Can be used to find `scale`
        that centers the wavelet anywhere from 0 to pi in discrete space.

        Reported as "nondimensional" in `info()` since it's scale-decoupled.
        """
        if getattr(self, '_wc_ct', None) is None:
            self._wc_ct = center_frequency(self, kind='peak-ct', N=self.N)
        return self._wc_ct

    @property
    def scalec_ct(self):
        """'Center scale' in sense of `wc_ct`, making wavelet peak at pi/4.
        See `help(Wavelet.wc_ct)`.
        """
        if getattr(self, '_scalec_ct', None) is None:
            self._scalec_ct = (4/pi) * self.wc_ct
        return self._scalec_ct

    @property
    def std_t(self):
        """Non-dimensional time resolution"""
        if getattr(self, '_std_t', None) is None:
            # scale=10 arbitrarily chosen to yield good compute-accurary
            self._std_t = time_resolution(self, scale=self.scalec_ct, N=self.N,
                                          nondim=True)
        return self._std_t

    @property
    def std_w(self):
        """Non-dimensional frequency resolution (radian)"""
        if getattr(self, '_std_w', None) is None:
            self._std_w = freq_resolution(self, scale=self.scalec_ct, N=self.N,
                                          nondim=True)
        return self._std_w

    @property
    def std_f(self):
        """Non-dimensional frequency resolution (cyclic)"""
        return self.std_w / (2*pi)

    @property
    def harea(self):
        """Heisenberg area: std_t * std_w >= 0.5"""
        return self.std_t * self.std_w

    @property
    def std_t_d(self):
        """Dimensional time resolution [samples/(cycles*radians)]"""
        if getattr(self, '_std_t_d', None) is None:
            self._std_t_d = time_resolution(self, scale=self.scalec_ct, N=self.N,
                                            nondim=False)
        return self._std_t_d

    @property
    def std_w_d(self):
        """Dimensional frequency resolution [(cycles*radians)/samples]"""
        if getattr(self, '_std_w_d', None) is None:
            self._std_w_d = freq_resolution(self, scale=self.scalec_ct, N=self.N,
                                            nondim=False)
        return self._std_w_d

    @property
    def std_f_d(self):
        """Dimensional frequency resolution [cycles/samples]"""
        return self.std_w_d / (2*pi)

    #### Misc ################################################################
    def info(self, nondim=True, reset=False):
        """Prints time & frequency resolution quantities. Refer to pertinent
        methods' docstrings on how each quantity is computed, and to
        tests/props_test.py on various dependences (e.g. `std_t` on `N`).
        If `reset`, will recompute all quantities (can be used with e.g. new `N`).

        See `help(Wavelet.x)`, x: `std_t, std_w, wc, wc_ct, scalec_ct`.

        Detailed overview: https://dsp.stackexchange.com/q/72042/50076
        """
        if reset:
            self.reset_properties()

        if nondim:
            cfg = self.config_str
            dim_t = dim_w = "non-dimensional"
            std_t, std_w = self.std_t, self.std_w
            wc_txt = "wc_ct, (cycles*radians)"
            wc = self.wc_ct
        else:
            cfg = self.config_str + " -- scale=%.2f" % self.scalec_ct
            dim_t = "samples/(cycles*radians)"
            dim_w = "(cycles*radians)/samples"
            std_t, std_w = self.std_t_d, self.std_w_d
            wc_txt = "wc,    (cycles*radians)/samples; %.2f" % self.scalec_ct
            wc = self.wc
        harea = std_t * std_w

        print(("{} wavelet\n"
               "\t{}\n"
               "\tCenter frequency: {:<10.6f} [{}]\n"
               "\tTime resolution:  {:<10.6f} [std_t, {}]\n"
               "\tFreq resolution:  {:<10.6f} [std_w, {}]\n"
               "\tHeisenberg area:  {:.12f}"
               ).format(self.name, cfg, wc, wc_txt,
                        std_t, dim_t, std_w, dim_w, harea))

    def reset_properties(self):
        """Reset time-frequency properties (`Wavelet.TF_PROPS`), i.e.
        recompute for current `self.N`.
        """
        for name in self.TF_PROPS:
            setattr(self, f'_{name}', None)
            getattr(self, name)  # trigger recomputation

    def viz(self, name='overview', **kw):
        """`Wavelet.VISUALS` for list of supported `name`s."""
        if name == 'overview':
            for name in ('heatmap', 'harea', 'filterbank', 'time-frequency'):
                kw['N'] = kw.get('N', self.N)
                self._viz(name, **kw)
        elif name not in Wavelet.VISUALS:
            raise ValueError(f"visual '{name}' not supported; must be one of: "
                             + ', '.join(Wavelet.VISUALS))
        else:
            self._viz(name, **kw)

    def _viz(self, name, **kw):
        kw['wavelet'] = kw.get('wavelet', self)
        kw['N'] = kw.get('N', self.N)
        {
            'heatmap':    visuals.wavelet_heatmap,
            'waveforms':  visuals.wavelet_waveforms,
            'filterbank': visuals.wavelet_filterbank,
            'harea':      visuals.sweep_harea,
            'std_t':      visuals.sweep_std_t,
            'std_w':      visuals.sweep_std_w,
            'time-frequency':      visuals.wavelet_tf,
            'anim:time-frequency': visuals.wavelet_tf_anim,
        }[name](**kw)

    def _desc(self, N=None, scale=None, show_N=True):
        """Nicely-formatted parameter summary, used in other methods"""
        if self.config_str != "Default configs":
            ptxt = self.config_str.rstrip(', ') + ', '
        else:
            ptxt = ""

        N = N or self.N
        if scale is None:
            title = "{} wavelet | {}N={}".format(self.name, ptxt, N)
        else:
            title = "{} wavelet | {}scale={:.2f}, N={}".format(
                self.name, ptxt, scale, N)

        if not show_N:
            title = title[:title.find(f"N={N}")].rstrip(', ')
        return title

    @classmethod
    def _process_dtype(self, dtype, as_str=None):
        """Ensures `dtype` is supported, and converts per `as_str` (if True,
        numpy/torch -> str, else vice versa; if None, returns as-is).
        """
        if isinstance(dtype, str):
            assert_is_one_of(dtype, 'dtype', Wavelet.DTYPES)
            if not as_str:
                return getattr(Q, dtype)
        elif not isinstance(dtype, (type, np.dtype, torch.dtype)):
            raise TypeError("`dtype` must be string or type (np./torch.dtype) "
                            "(got %s)" % dtype)
        return dtype if not as_str else str(dtype).split('.')[-1]

    #### Init ################################################################
    @classmethod
    def _init_if_not_isinstance(self, wavelet, **kw):
        """Circumvents type change from IPython's super-/auto-reload,
        but first checks with usual isinstance."""
        if isinstance_by_name(wavelet, Wavelet):
            return wavelet
        return Wavelet(wavelet, **kw)

    def _validate_and_set_wavelet(self, wavelet):
        def process_dtype(wavopts, user_passed_float32):
            """Handles GMW's `norm='energy'` w/ dtype='float32'."""
            if wavopts.get('norm', 'bandpass') == 'energy':
                if user_passed_float32:
                    WARN("`norm='energy'` w/ `dtype='float32'` is unsupported; "
                         "will use 'float64' instead.")
                wavopts['dtype'] = 'float64'
                self._dtype = 'float64'
            elif self.dtype is not None:
                wavopts['dtype'] = self.dtype

        def set_dtype_from_out():
            # 32 will promote to 64 if other params are 64
            out_dtype = self.fn(S.asarray([1.], dtype='float32')).dtype
            if any(tp in str(out_dtype) for tp in ('complex64', 'complex128')):
                # 'bump' wavelet case
                out_dtype = ('float32' if 'complex64' in str(out_dtype) else
                             'float64')
            self._dtype = self._process_dtype(out_dtype, as_str=True)

        if isinstance(wavelet, FunctionType):
            self.fn = wavelet
            set_dtype_from_out()
            self.config = {}
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

        user_passed_float32 = any('float32' in str(t)
                                  for t in (self.dtype, wavopts.get('dtype', 0)))
        if isinstance(wavelet, str):
            wavelet = wavelet.lower()
            module = 'wavelets' if wavelet != 'gmw' else '_gmw'
            wavopts = gdefaults(f"{module}.{wavelet}", get_all=True,
                                as_dict=True, default_order=True, **wavopts)

        process_dtype(wavopts, user_passed_float32)
        assert_is_one_of(wavelet, 'wavelet', Wavelet.SUPPORTED)
        self.fn = {
            'gmw':    gmw,
            'morlet': morlet,
            'bump':   bump,
            'cmhat':  cmhat,
            'hhhat':  hhhat,
        }[wavelet](**wavopts)

        if self.dtype is None:
            set_dtype_from_out()
        self.config = wavopts


@jit(nopython=True, cache=True)
def _xifn(scale, N, dtype=np.float64):
    """N=128: [0, 1, 2, ..., 64, -63, -62, ..., -1] * (2*pi / N) * scale
       N=129: [0, 1, 2, ..., 64, -64, -63, ..., -1] * (2*pi / N) * scale
    """
    xi = np.zeros(N, dtype=dtype)
    h = scale * (2 * pi) / N
    for i in range(N // 2 + 1):
        xi[i] = i * h
    for i in range(N // 2 + 1, N):
        xi[i] = (i - N) * h
    return xi

def _process_params_dtype(*params, dtype, auto_gpu=True):
    if dtype is None:
        dtype = S.asarray(params[0]).dtype
    if auto_gpu:
        dtype = Wavelet._process_dtype(dtype, as_str=True)
        params = [S.astype(S.asarray(p), dtype) for p in params]
    else:
        dtype = Wavelet._process_dtype(dtype, as_str=True)
        params = [np.asarray(p).astype(dtype) for p in params]
    return params if len(params) > 1 else params[0]

#### Wavelet functions ######################################################
def morlet(mu=None, dtype=None):
    """Higher `mu` -> greater frequency, lesser time resolution.
    Recommended range: 4 to 16. For `mu > 6` the wavelet is almsot exactly
    Gaussian for most scales, providing maximum joint resolution.

    `mu=13.4` matches Generalized Morse Wavelets' `(beta, gamma) = (3, 60)`.
    For full correspondence see `help(_gmw.gmw)`.

    https://en.wikipedia.org/wiki/Morlet_wavelet#Definition
    https://www.desmos.com/calculator/0nslu0qivv
    """
    mu, dtype = gdefaults('wavelets.morlet', mu=mu, dtype=dtype)
    cs = (1 + np.exp(-mu**2) - 2 * np.exp(-3/4 * mu**2)) ** (-.5)
    ks = np.exp(-.5 * mu**2)
    mu, cs, ks = _process_params_dtype(mu, cs, ks, dtype=dtype)

    # all other consts go to `C`; needed for numba.jit to not type promote to
    # float64 due to Python floats (e.g. `2.`)
    C = S.asarray([-.5, np.sqrt(2) * cs * pi**.25], dtype=dtype)

    fn = _morlet_gpu if USE_GPU() else (_morlet_par if IS_PARALLEL() else _morlet)
    return lambda w: fn(atleast_1d(w, dtype), mu, ks, C)

@jit(nopython=True, cache=True)
def _morlet(w, mu, ks, C):
    return C[1]* (np.exp(C[0] * (w - mu)**2) - ks * np.exp(C[0] * w**2))

@jit(nopython=True, cache=True, parallel=True)
def _morlet_par(w, mu, ks, C):
    return C[1]* (np.exp(C[0] * (w - mu)**2) - ks * np.exp(C[0] * w**2))

def _morlet_gpu(w, mu, ks, C):
    return C[1] * (torch.exp(C[0] * (w - mu)**2) - ks * torch.exp(C[0] * w**2))


def bump(mu=None, s=None, om=None, dtype=None):
    """Bump wavelet.
    https://www.mathworks.com/help/wavelet/gs/choose-a-wavelet.html
    """
    mu, s, om, dtype = gdefaults('wavelets.bump', mu=mu, s=s, om=om, dtype=dtype)
    if 'float' in dtype:
        dtype = 'complex' + str(2 * int(dtype.strip('float')))
    mu, s, om = [S.asarray(g, dtype) for g in (mu, s, om)]
    C = S.asarray([2 * pi * 1j * om, .443993816053287], dtype=dtype)
    C0 = S.asarray(.999, dtype='float' + str(int(dtype.strip('complex'))//2))

    fn = _bump_gpu if USE_GPU() else (_bump_par if IS_PARALLEL() else _bump)
    return lambda w: fn(atleast_1d(w, dtype), (atleast_1d(w, dtype) - mu) / s,
                        s, C, C0)

@jit(nopython=True, cache=True)
def _bump(w, _w, s, C, C0):
    return np.exp(C[0] * w) / s * (
        np.abs(_w) < C0) * np.exp(
            -1 / (1 - (_w * (np.abs(_w) < C0))**2)) / C[1]

@jit(nopython=True, cache=True, parallel=True)
def _bump_par(w, _w, s, C, C0):
    return np.exp(C[0] * w) / s * (
        np.abs(_w) < C0) * np.exp(
            -1 / (1 - (_w * (np.abs(_w) < C0))**2)) / C[1]

def _bump_gpu(w, _w, s, C, C0):
    return torch.exp(C[0] * w) / s * (
        torch.abs(_w) < C0) * torch.exp(
            -1 / (1 - (_w * (torch.abs(_w) < C0))**2)) / C[1]


def cmhat(mu=None, s=None, dtype=None):
    """Complex Mexican Hat wavelet.
    https://en.wikipedia.org/wiki/Complex_mexican_hat_wavelet
    """
    mu, s, dtype = gdefaults('wavelets.cmhat', mu=mu, s=s, dtype=dtype)
    mu, s = _process_params_dtype(mu, s, dtype=dtype)
    C = S.asarray([5/2, 2 * np.sqrt(2/3) * pi**(-1/4)], dtype=dtype)

    fn = _cmhat_gpu if USE_GPU() else (_cmhat_par if IS_PARALLEL() else _cmhat)
    return lambda w: fn(atleast_1d(w, dtype) - mu, s, C)

@jit(nopython=True, cache=True)
def _cmhat(_w, s, C):
    return C[1] * (s**C[0] * _w**2 * np.exp(-s**2 * _w**2 / 2) * (_w >= 0))

@jit(nopython=True, cache=True, parallel=True)
def _cmhat_par(_w, s, C):
    return C[1] * (s**C[0] * _w**2 * np.exp(-s**2 * _w**2 / 2) * (_w >= 0))

def _cmhat_gpu(_w, s, C):
    return C[1] * (s**C[0] * _w**2 * torch.exp(-s**2 * _w**2 / 2) * (_w >= 0))


def hhhat(mu=None, dtype=None):
    """Hilbert analytic function of Hermitian Hat."""
    mu, dtype = gdefaults('wavelets.hhhat', mu=mu, dtype=dtype)
    mu = _process_params_dtype(mu, dtype=dtype)
    C = S.asarray([-1/2, 2 / np.sqrt(5) * pi**(-1/4)], dtype=dtype)

    fn = _hhhat_gpu if USE_GPU() else (_hhhat_par if IS_PARALLEL() else _hhhat)
    return lambda w: fn(atleast_1d(w, dtype) - mu, C)

@jit(nopython=True, cache=True)
def _hhhat(_w, C):
    return C[1] * (_w * (1 + _w) * np.exp(C[0] * _w**2)) * (1 + np.sign(_w))

@jit(nopython=True, cache=True, parallel=True)
def _hhhat_par(_w, C):
    return C[1] * (_w * (1 + _w) * np.exp(C[0] * _w**2)) * (1 + np.sign(_w))

def _hhhat_gpu(_w, C):
    return C[1] * (_w * (1 + _w) * torch.exp(C[0] * _w**2)) * (1 + torch.sign(_w))


#### Wavelet properties ######################################################
def center_frequency(wavelet, scale=None, N=1024, kind='energy', force_int=None,
                     viz=False):
    """Center frequency (radian) of `wavelet`, either 'energy', 'peak',
    or 'peak-ct'.

    Detailed overviews:
        (1) https://dsp.stackexchange.com/a/76371/50076
        (2) https://dsp.stackexchange.com/q/72042/50076

    **Note**: implementations of `center_frequency`, `time_resolution`, and
    `freq_resolution` are discretized approximations of underlying
    continuous-time parameters. This is a flawed approach (see (1)).
      - Caution is advised for scales near minimum and maximim (obtained via
        `cwt_scalebounds(..., preset='maximal')`), where inaccuracies may be
        significant.
      - For intermediate scales and sufficiently large N (>=1024), the methods
        are reliable. May improve in the future

    # Arguments
        wavelet: wavelets.Wavelet

        scale: float / None
            Scale at which to compute `wc`; ignored if `kind='peak-ct'`.

        N: int
            Length of wavelet.

        kind: str['energy', 'peak', 'peak-ct']
            - 'energy': weighted mean of wavelet energy, or energy expectation;
              Eq 4.52 of [1]:
                wc_1     = int w |wavelet(w)|^2 dw  0..inf
                wc_scale = int (scale*w) |wavelet(scale*w)|^2 dw 0..inf
                         = wc_1 / scale
            - 'peak': value of `w` at which `wavelet` at `scale` peaks
              (is maximum) in discrete time, i.e. constrained 0 to pi.
            - 'peak-ct': value of `w` at which `wavelet` peaks (without `scale`,
              i.e. `scale=1`), i.e. peak location of the continuous-time function.
              Can be used to find `scale` at which `wavelet` is most well-behaved,
              e.g. at eighth of sampling frequency (centered between 0 and fs/4).
            - 'energy' == 'peak' for wavelets exactly even-symmetric about mode
              (peak location)

        force_int: bool / None
            Relevant only if `kind='energy'`, then defaulting to True. Set to
            False to compute via formula - i.e. first integrate at a
            "well-behaved" scale, then rescale. For intermediate scales, this
            won't yield much difference. For extremes, it matches the
            continuous-time results closer - but this isn't recommended, as it
            overlooks limitations imposed by discretization (trimmed/undersampled
            freq-domain bell).

        viz: bool (default False)
            Whether to visualize obtained center frequency.

    **Misc**

    For very high scales, 'energy' w/ `force_int=True` will match 'peak'; for
    very low scales, 'energy' will always be less than 'peak'.

    To convert to Hz:
        wc [(cycles*radians)/samples] / (2pi [radians]) * fs [samples/second]
        = fc [cycles/second]

    See tests/props_test.py for further info.

    # References
        1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz(wc, params):
        w, psih, apsih2 = params
        _w = w[N//2-1:]; _psih = psih[N//2-1:]; _apsih2 = apsih2[N//2-1:]

        wc = wc if (kind != 'peak-ct') else pi/4
        vline = (wc, dict(color='tab:red', linestyle='--'))
        plot(_w, _psih, show=1, vlines=vline,
             title="psih(w)+ (frequency-domain wavelet, pos half)")
        plot(_w, _w * _apsih2, show=1,
             title="w^2 |psih(w)+|^2 (used to compute wc)")
        print("wc={}".format(wc))

    def _params(wavelet, scale, N):
        w = S.asarray(aifftshift(_xifn(1, N)))
        psih = asnumpy(wavelet(S.asarray(scale) * w))
        apsih2 = np.abs(psih)**2
        w = asnumpy(w)
        return w, psih, apsih2

    def _energy_wc(wavelet, scale, N, force_int):
        use_formula = not force_int
        if use_formula:
            scale_orig = scale
            wc_ct = _peak_ct_wc(wavelet, N)[0]
            scale = (4/pi) * wc_ct

        w, psih, apsih2 = _params(wavelet, scale, N)
        wc = (integrate.trapezoid(apsih2 * w) /
              integrate.trapezoid(apsih2))

        if use_formula:
            wc *= (scale / scale_orig)
        return float(wc), (w, psih, apsih2)

    def _peak_wc(wavelet, scale, N):
        w, psih, apsih2 = _params(wavelet, scale, N)
        wc = w[np.argmax(apsih2)]
        return float(wc), (w, psih, apsih2)

    def _peak_ct_wc(wavelet, N):
        wc, _ = find_maximum(wavelet.fn)
        # need `scale` such that `wavelet` peaks at `scale * xi.max()/4`
        # thus: `wc = scale * (pi/2)` --> `scale = (4/pi)*wc`
        scale = S.asarray((4/pi) * wc)
        w, psih, apsih2 = _params(wavelet, scale, N)
        return float(wc), (w, psih, apsih2)

    if force_int and 'peak' in kind:
        NOTE("`force_int` ignored with 'peak' in `kind`")
    assert_is_one_of(kind, 'kind', ('energy', 'peak', 'peak-ct'))

    if kind == 'peak-ct' and scale is not None:
        NOTE("`scale` ignored with `peak = 'peak-ct'`")

    if scale is None and kind != 'peak-ct':
        # see _peak_ct_wc
        wc, _ = find_maximum(wavelet.fn)
        scale = (4/pi) * wc

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    if kind == 'energy':
        force_int = force_int or True
        wc, params = _energy_wc(wavelet, scale, N, force_int)
    elif kind == 'peak':
        wc, params = _peak_wc(wavelet, scale, N)
    elif kind == 'peak-ct':
        wc, params = _peak_ct_wc(wavelet, N)

    if viz:
        _viz(wc, params)
    return wc


def freq_resolution(wavelet, scale=10, N=1024, nondim=True, force_int=True,
                    viz=False):
    """Compute wavelet frequency width (std_w) for a given scale and N; larger N
    -> less discretization error, but same N as in application works best
    (larger will be "too accurate" and misrepresent true discretized values).

    `nondim` will divide by peak center frequency and return unitless quantity.

    Eq 22 in [1], Sec 4.3.2 in [2].
    Detailed overview: https://dsp.stackexchange.com/q/72042/50076
    See tests/props_test.py for further info.

    # References
        1. Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        2. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz():
        _w = w[N//2-1:]; _psih = psih[N//2-1:]; _apsih2 = apsih2[N//2-1:]

        plot(_w, _psih, show=1,
             title="psih(w)+ (frequency-domain wavelet, pos half)")
        plot(_w, (_w-wce)**2 * _apsih2, show=1,
             title="(w-wc)^2 |psih(w)+|^2 (used to compute var_w)")
        print("std_w={}".format(std_w))
        if use_formula:
            NOTE(f"integrated at scale={scale} then used formula; "
                 "see help(freq_resolution) and try force_int=True")

    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    # formula criterion not optimal; thresholds will vary by wavelet config
    use_formula = ((scale < 4 or scale > N / 5) and not force_int)
    if use_formula:
        scale_orig = scale
        scale = (4/pi) * wavelet.wc_ct

    w = aifftshift(_xifn(1, N))
    psih = asnumpy(wavelet(scale * w))
    wce = center_frequency(wavelet, scale, force_int=force_int, kind='energy')

    apsih2 = np.abs(psih)**2
    var_w = (integrate.trapezoid((w - wce)**2 * apsih2, w) /
             integrate.trapezoid(apsih2, w))

    std_w = np.sqrt(var_w)
    if use_formula:
        std_w *= (scale / scale_orig)
        scale = scale_orig
    if nondim:
        wcp = center_frequency(wavelet, scale, kind='peak')
        std_w /= wcp
    if viz:
        _viz()
    return std_w


def time_resolution(wavelet, scale=10, N=1024, min_decay=1e3, max_mult=2,
                    min_mult=2, force_int=True, nondim=True, viz=False):
    """Compute wavelet time resolution for a given scale and N; larger N
    -> less discretization error, but same N as in application should suffice.

    Eq 21 in [1], Sec 4.3.2 in [2].
    Detailed overview: https://dsp.stackexchange.com/q/72042/50076

    `nondim` will multiply by peak center frequency and return unitless quantity.
    ______________________________________________________________________________

    **Interpretation**

    Measures time-span of 68% of wavelet's energy (1 stdev for Gauss-shaped
    |psi(t)|^2). Inversely-proportional with `N`, i.e. same `scale` spans half
    the fraction of sequence that's twice long. Is actually *half* the span
    per unilateral (radius) std.

        std_t ~ scale (T / N)
    ______________________________________________________________________________

    **Implementation details**

    `t` may be defined from `min_mult` up to `max_mult` times the original span
    for computing stdev since wavelet may not decay to zero within target frame.
    For any mult > 1, this is biased if we are convolving by sliding windows of
    length `N` in CWT, but we're not (see `cwt`); our scheme captures full wavelet
    characteristics, i.e. as if conv/full decayed length (but only up to mult=2).

    `min_decay` controls decay criterion of time-wavelet domain in integrating,
    i.e. ratio of max to endpoints of |psi(t)|^2 must exceed this. Will search
    up to `max_mult * N`-long `t`.

    For small `scale` (<~3) results are harder to interpret and defy expected
    behavior per discretization complications (call with `viz=True`). Workaround
    via computing at stable scale and calculating via formula shouldn't work as
    both-domain behaviors deviate from continuous, complete counterparts.
    ______________________________________________________________________________

    See tests/props_test.py for further info.

    # References
        1. Higher-Order Properties of Analytic Wavelets.
        J. M. Lilly, S. C. Olhede.
        https://sci-hub.st/10.1109/TSP.2008.2007607

        2. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
        https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    def _viz():
        _w    = aifftshift(xi)[Nt//2-1:]
        _psih = aifftshift(psih)[Nt//2-1:]

        plot(_w, _psih, show=1,
             title="psih(w)+ (frequency-domain wavelet, pos half)")
        plot(t, t**2 * apsi2, title="t^2 |psi(t)|^2 (used to compute var_t)",
             show=1)
        _viz_cwt_scalebounds(wavelet, N, max_scale=scale, std_t=std_t, Nt=Nt)

        print("std_t={}\nlen(t), len(t)/N, t_min, t_max = {}, {}, {}, {}".format(
            std_t, len(t), len(t)/N, t.min(), t.max()))
        if use_formula:
            NOTE(f"integrated at scale={scale} then used formula; "
                 "see help(time_resolution) and try force_int=True")

    def _make_integration_t(wavelet, scale, N, min_decay, max_mult, min_mult):
        """Ensure `psi` decays sufficiently at integration bounds"""
        for mult in np.arange(min_mult, max_mult + 1):
            Nt = int(mult * N)
            apsi2 = np.abs(asnumpy(wavelet.psifn(scale=scale, N=Nt)))**2
            # ensure sufficient decay at endpoints (assumes ~symmetric decay)
            if apsi2.max() / apsi2[:max(10, Nt//100)].mean() > min_decay:
                break
        else:
            raise Exception(("Couldn't find decay timespan satisfying "
                             "`(min_decay, max_mult) = ({}, {})` for `scale={}`; "
                             "decrease former or increase latter or check "
                             "`wavelet`".format(min_decay, max_mult, scale)))

        # len(t) == mult*N (independent of T)
        # `t` doesn't have zero-mean but that's correct for psi's peak & symmetry
        T = N
        t = np.arange(-mult * T/2, mult * T/2, step=T/N)
        return t

    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    # formula criterion not optimal; thresholds will vary by wavelet config
    use_formula = ((scale < 4 or scale > N / 5) and not force_int)
    if use_formula:
        scale_orig = scale
        scale = (4/pi) * wavelet.wc_ct

    t = _make_integration_t(wavelet, scale, N, min_decay, max_mult, min_mult)
    Nt = len(t)

    xi = _xifn(1, Nt)
    psih = asnumpy(wavelet(scale * xi, nohalf=False))
    psi = asnumpy(ifft(psih * (-1)**np.arange(Nt)))

    apsi2 = np.abs(psi)**2
    var_t = (integrate.trapezoid(t**2 * apsi2, t) /
             integrate.trapezoid(apsi2, t))

    std_t = np.sqrt(var_t)
    if use_formula:
        std_t *= (scale_orig / scale)
        scale = scale_orig
    if nondim:
        # 'energy' yields values closer to continuous-time counterparts,
        # but we seek accuracy relative to discretized values
        wc = center_frequency(wavelet, scale, N=N, kind='peak')
        std_t *= wc
    if viz:
        _viz()
    return std_t


#### Misc ####################################################################
def afftshift(xh):
    """Needed since analytic wavelets keep Nyquist bin at N//2 positive bin
    whereas FFT convention is to file it under negative (see `_xi`).
    Moves right N//2 + 1 bins to left.
    """
    if len(xh) % 2 == 0:
        return _afftshift_even(xh, np.zeros(len(xh), dtype=xh.dtype))
    return fftshift(xh)

@jit(nopython=True, cache=True)
def _afftshift_even(xh, xhs):
    N = len(xh)
    for i in range(N // 2 + 1):
        xhs[i] = xh[i + N // 2 - 1]
    for i in range(N // 2 + 1, N):
        xhs[i] = xh[i - N // 2 - 1]
    return xhs


def aifftshift(xh):
    """Inversion also different; moves left N//2+1 bins to right."""
    if len(xh) % 2 == 0:
        return _aifftshift_even(xh, np.zeros(len(xh), dtype=xh.dtype))
    return ifftshift(xh)

@jit(nopython=True, cache=True)
def _aifftshift_even(xh, xhs):
    N = len(xh)
    for i in range(N // 2 + 1):
        xhs[i + N//2 - 1] = xh[i]
    for i in range(N // 2 + 1, N):
        xhs[i - N//2 - 1] = xh[i]
    return xhs


def _fn_to_name(fn):
    """`_` to ` `, removes `<lambda>` & `.`, handles `SPECIALS`."""
    SPECIALS = {'Gmw ': 'GMW '}
    name = fn.__qualname__.replace('_', ' ').replace('<locals>', '').replace(
        '<lambda>', '').replace('.', '').title()

    for k, v in SPECIALS.items():
        name = name.replace(k, v)
    return name


def isinstance_by_name(obj, ref):
    """IPython reload can make isinstance(Obj(), Obj) fail; won't work if
    Obj has __str__ overridden."""
    def _class_name(obj):
        name = getattr(obj, '__qualname__', getattr(obj, '__name__', ''))
        return (getattr(obj, '__module__', '') + '.' + name).lstrip('.')
    return _class_name(type(obj)) == _class_name(ref)


##############################################################################
from ._gmw import gmw
from . import visuals
from .visuals import plot, _viz_cwt_scalebounds
from .utils.common import WARN, NOTE, pi, assert_is_one_of
from .utils.backend import asnumpy
