# -*- coding: utf-8 -*-
"""
Signals for testing effectiveness of time-frequency transforms against
variety of localization characteristics.

1. **sine**: pure sine or cosine at one frequency, `cos(2pi f t)`
    a. sine
    b. cosine
    c. phase-shifted
    d. trimmed (others complete exactly one cycle)  # TODO

2. **<name>:am**: <name> with amplitude modulation, i.e. `A(t) * fn(t)`
    a. |sine|
    b. |cosine|
    c. exp
    d. gauss

3. **#<name>**: superimpose reflected <name> onto itself, i.e. `x += x[::-1]`

4. **lchirp**: linear chirp, `cos(2pi t**2/2)`, spanning `fmin` to `fmax`

5. **echirp**: exponential chirp, `cos(2pi exp(t))`, spanning `fmin` to `fmax`

6. **hchirp**: hyperbolic chirp, `cos(2pi a/(b - t))`, spanning `fmin` to `fmax`

7, 8, 9: **par_lchirp, par_echirp, par_hchirp**: linear, exponential, hyperbolic
         chirps, superimposed, with frequency modulation in parallel,
         spanning `fmin1` to `fmax1` and `fmin2` to `fmax2`.

10. **jumps**: large instant frequency transitions, `cos(2pi f*t), f=2 -> f=100`

11. **packed**: closely-spaced bands of sinusoids with majority overlap, e.g.
                `cos(w*t[No:]) + cos((w+1)*t[-No:]) + cos((w+3)*t[No:]) + ...`,
                `No = .8*len(t)`.
"""
import inspect
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from textwrap import wrap

from ._ssq_cwt import ssq_cwt
from ._ssq_stft import ssq_stft
from .utils import WARN
from .wavelets import Wavelet
from .visuals import plot, plots, imshow


pi = np.pi
DEFAULT_N = 256
DEFAULT_ARGS = {
    'cosine': dict(f=8, phi0=0),
    'sine':   dict(f=8, phi0=0),
    'lchirp': dict(tmin=0, tmax=1, fmin=0,  fmax=None),
    'echirp': dict(tmin=0, tmax=1, fmin=1, fmax=None),
    'hchirp': dict(tmin=0, tmax=1, fmin=.5,  fmax=None),
    'jumps':  dict(),
    'low':    dict(),
    'am-cosine': dict(amin=.1),
    'am-sine':   dict(amin=.1),
    'am-exp':    dict(amin=.1),
    'am-gauss':  dict(amin=.01),
    'sine:am-cosine': (dict(f=16), dict(amin=.5)),
}
DEFAULT_TKW = dict(tmin=0, tmax=1, endpoint=True)


#### Test signals ############################################################
class TestSignals():
    """Signals of varying time-frequency characteristics. Convenience methods
    to plot multiple signals and their transforms under varying wavelet / window
    parameters.

    `.demo(signals)` to visualize `signals`, `test_transforms(fn)` to apply `fn`
    to and visualize output.

    See `examples/` on Github, and
    https://overlordgolddragon.github.io/test-signals/

    **Sweep functions**
        For `lchirp`, `echirp`, & `hchirp`, `N` will be determined automatically
        if `tmin`, `tmax`, `fmin`, and `fmax` are provided, minimally such that
        no aliasing occurs.

    **Demo signals**
        `TestSignals.DEMO` holds list of `signals` names invoked when passing
        `signals='all'`, which can be changed.

    # Arguments
        N: int
            Will use this as default `N` anytime `N` is left unspecified.

        default_args: dict
            `{<signal_name>: {'param_name': value}}` pairs, where `signal_name`
            is one of `SUPPORTED`. See `_test_signals.DEFAULT_ARGS`.

        default_tkw: dict
            Example with all key-value pairs: `dict(tmin=0, tmax=1)`.

        warn_alias: bool (default True)
            Whether to print warning if generated signal aliases (f > fs/2);
            to disable, pass `warn_alias=False` to `__init__()`, or set directly
            on instance (`TestSignals().warn_alias=False`).

    # TODO complete docstrings
    """
    SUPPORTED = ['cosine', 'sine', 'lchirp', 'echirp', 'echirp_pc', 'hchirp',
                 'par-lchirp', 'par-echirp', 'par-hchirp', 'jumps', 'packed',
                 'am-sine', 'am-cosine', 'am-exp', 'am-gauss']
    # what to show with `signal='all'`, and in what order
    DEMO = ['cosine', 'sine',
            'lchirp', 'echirp', 'hchirp',
            '#lchirp', '#echirp', '#hchirp',
            'par-lchirp', 'par-echirp', 'par-hchirp', '#par-lchirp',
            'jumps', 'packed',
            'am-sine', 'am-cosine', 'am-exp', 'am-gauss']

    def __init__(self, N=None, default_args=None, default_tkw=None,
                 warn_alias=True):
        self.default_N    = N    or DEFAULT_N
        self.default_args = default_args or DEFAULT_ARGS
        self.default_tkw  = default_tkw  or DEFAULT_TKW
        self.warn_alias   = warn_alias

        # set defaults on unspecified
        for k, v in DEFAULT_ARGS.items():
            self.default_args[k] = self.default_args.get(k, v)
        for k, v in DEFAULT_TKW.items():
            self.default_tkw[k] = self.default_tkw.get(k, v)

    #### test signals ########################################################
    def _maybe_warn_alias(self, phi, tol=.02):
        # allow non-trivial overshoot as it may occur but not worth warning
        if self.warn_alias:
            fmax = np.diff(phi).max()
            if (fmax - np.pi) > tol:
                WARN("`%s` has aliased w/ max(diff(phi))=%.6f>%.6f" % (
                    inspect.stack()[2][3], fmax, pi))

    def sine(self, N=None, f=1, phi0=0, **tkw):
        """sin(2pi*f*t + phi)"""
        tkw['endpoint'] = tkw.get('endpoint', False)
        t, *_ = self._process_params(N, tkw)

        phi = 2*pi * f * t + phi0
        self._maybe_warn_alias(phi)
        return np.sin(phi), t

    def cosine(self, N=None, f=1, phi0=0, **tkw):
        """cos(2pi*f*t + phi)"""
        tkw['endpoint'] = tkw.get('endpoint', False)
        t, *_ = self._process_params(N, tkw)

        phi = 2*pi * f * t + phi0
        self._maybe_warn_alias(phi)
        return np.cos(phi), t

    def _generate(self, fn, N, fmin, fmax, **tkw):
        """Used by chirps."""
        t, tmin, tmax, fmax = self._process_params(N, tkw, fn, fmin, fmax)
        phi = fn(t, tmin, tmax, fmin, fmax)
        self._maybe_warn_alias(phi)
        return np.cos(phi), t

    def lchirp(self, N=None, fmin=0, fmax=None, **tkw):
        """
        >>>   f(t) = a*t + b
        >>> phi(t) = (a/2)*(t^2 - tmin^2) + b*(t - tmin)
        >>> a = (fmin - fmax) / (tmin - tmax)
            b = (fmin*tmax - fmax*tmin) / (tmax - tmin)
        """
        return self._generate(self._lchirp_fn, N, fmin, fmax, **tkw)

    def _lchirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        a = (fmin - fmax) / (tmin - tmax)
        b = (fmin*tmax - fmax*tmin) / (tmax - tmin)

        phi = (a/2)*(t**2 - tmin**2) + b*(t - tmin)
        phi *= (2*pi)
        if get_w:
            w = a*t + b
            w *= (2*pi)
        return (phi, w) if get_w else phi

    def echirp(self, N=None, fmin=.1, fmax=None, **tkw):
        """
        >>> f(t)   = a*b^t
        >>> phi(t) = (a/ln(b)) * (b^t - b^tmin)
        >>> a = (fmin^tmax / fmax^tmin) ^ 1/(tmax - tmin)
            b = fmax^(1/tmax) * (1/a)^(1/tmax)
        """
        return self._generate(self._echirp_fn, N, fmin, fmax, **tkw)

    def _echirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        a = (fmin**tmax / fmax**tmin) ** (1/(tmax - tmin))
        b = fmax**(1/tmax) * (1/a)**(1/tmax)

        phi = (a/np.log(b)) * (b**t - b**tmin)
        phi *= (2*pi)
        if get_w:
            w = a*b**t
            w *= (2*pi)
        return (phi, w) if get_w else phi

    def echirp_pc(self, N=None, fmin=0, fmax=None, **tkw):
        """Alternate design that keeps f'(t) fixed at `e`, but is no longer
        geometric in the sense `f(t2) / f(t1) = const.`. "echirp plus constant".

        >>> f(t)   = a*exp(t) + b
        >>> phi(t) = a*(exp(t) - exp(tmin)) + b*(t - tmin)
        >>> a = (fmax - fmin)/(exp(tmax) - exp(tmin))
            b = (fmin*exp(tmax) - fmax*exp(tmin)) / (exp(tmax) - exp(tmin))
        """
        return self._generate(self._echirp_pc_fn, N, fmin, fmax, **tkw)

    def _echirp_pc_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        a, b, c, d = fmin, fmax, tmin, tmax
        A = (b - a) / (np.exp(d) - np.exp(c))
        B = (a*np.exp(d) - b*np.exp(c)) / (np.exp(d) - np.exp(c))

        phi = A*(np.exp(t) - np.exp(tmin)) + B*(t - tmin)
        phi *= (2*pi)
        if get_w:
            w = A*np.exp(t) + B
            w *= (2*pi)
        return (phi, w) if get_w else phi

    def hchirp(self, N=None, fmin=.1, fmax=None, **tkw):
        """
        >>> f(t)   = A / (B - t)^2
        >>> phi(t) = A * (1/(B - t) + 1/(tmin - B))
        >>> a, b, c, d = fmin, fmax, tmin, tmax
            A = AN / AD, B = BN / BD,
            AN = 2*sqrt(a^3*b^3*(c - d)^4) + a^2*b*(c - d)^2 + a*b^2*(c - d)^2
            AD = (a - b)^2
            BN = sqrt(a^3*b^3*(c-d)^4) + a^2*b*c*(c-d) + a*b^2*d*(d - c)
            BD = a*b*(a - b)*(c - d)
        """
        return self._generate(self._hchirp_fn, N, fmin, fmax, **tkw)

    def _hchirp_fn(self, t, tmin, tmax, fmin, fmax, get_w=False):
        a, b, c, d = fmin, fmax, tmin, tmax

        AN = (2*np.sqrt(a**3*b**3*(c - d)**4) + a**2*b*(c - d)**2
              + a*b**2*(c - d)**2)
        AD = (a - b)**2
        BN = np.sqrt(a**3*b**3*(c - d)**4) + a**2*b*c*(c - d) + a*b**2*d*(d - c)
        BD = a*b*(a - b)*(c - d)
        A = AN / AD
        B = BN / BD

        phi = A * (1/(B - t) + 1/(tmin - B))
        phi *= (2*pi)
        if get_w:
            w = A / (B - t)**2
            w *= (2*pi)
        return (phi, w) if get_w else phi

    def par_lchirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Linear frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        N = N or self.default_N
        fdiff_default = N/10

        if fmin1 is None:
            fmin1 = self.default_args['lchirp'].get('fmin', 0)
        if fmin2 is None:
            fmin2 = fmin1 + fdiff_default
        if fmax2 is None or fmax1 is None:
            if fmax1 is None:
                fmax2 = N/2
                fmax1 = fmax2 - fdiff_default
            else:
                fmax2 = min(N/2, fmax1 + fdiff_default)

        x1, t = self.lchirp(N, fmin1, fmax1, **tkw)
        x2, _ = self.lchirp(N, fmin2, fmax2, **tkw)
        x = x1 + x2
        return x, t

    def par_echirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Exponential frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        N = N or self.default_N
        fratio_default = 1.5

        if fmin1 is None:
            fmin1 = self.default_args['echirp'].get('fmin', 1)
        if fmin2 is None:
            fmin2 = fmin1 * fratio_default
        if fmax2 is None or fmax1 is None:
            if fmax1 is None:
                fmax2 = N/2
                fmax1 = fmax2 / fratio_default
            else:
                fmax2 = min(N/2, fmax1 * fratio_default)

        x1, t = self.echirp(N, fmin1, fmax1, **tkw)
        x2, _ = self.echirp(N, fmin2, fmax2, **tkw)
        x = x1 + x2
        return x, t

    def par_hchirp(self, N=None, fmin1=None, fmax1=None, fmin2=None, fmax2=None,
                   **tkw):
        """Hyperbolic frequency modulation in parallel. Should have
        `fmax2 > fmax1`, `fmin2 > fmin1`, and shared `tmin`, `tmax`.
        """
        N = N or self.default_N
        fratio_default = 3

        if fmin1 is None:
            fmin1 = self.default_args['hchirp'].get('fmin', 1)
        if fmin2 is None:
            fmin2 = fmin1 * fratio_default
        if fmax2 is None or fmax1 is None:
            if fmax1 is None:
                fmax2 = N/2
                fmax1 = fmax2 / fratio_default
            else:
                fmax2 = min(N/2, fmax1 * fratio_default)

        x1, t = self.hchirp(N, fmin1, fmax1, **tkw)
        x2, _ = self.hchirp(N, fmin2, fmax2, **tkw)
        x = x1 + x2
        return x, t

    def am_sine(self, N=None, f=1, amin=0, amax=1, phi=0, **tkw):
        N = N or self.default_N
        _A, t = self.sine(N, f, phi, **tkw)
        _A = (_A + 1) / 2
        return amin + (amax - amin) * _A, t

    def am_cosine(self, N=None, f=1, amin=0, amax=1, phi=0, **tkw):
        N = N or self.default_N
        _A, t = self.cosine(N, f, phi, **tkw)
        _A = (_A + 1) / 2
        return amin + (amax - amin) * _A, t

    def am_exp(self, N=None, amin=.1, amax=1, **tkw):
        """Use `echirp`'s expression for `f(t)`"""
        N = N or self.default_N
        t, tmin, tmax = self._process_params(N, tkw)
        _A = self._echirp_fn(t, tmin, tmax, amin, amax, get_w=True)[1]
        _A /= (2*pi)
        return _A, t

    def am_gauss(self, N=None, amin=.1, amax=1, **tkw):
        N = N or self.default_N
        t = _t(-1, 1, N)
        _A = np.exp( -((t - t.mean())**2 * 5) )
        return amin + (amax - amin)*_A, t

    def jumps(self, N=None, freqs=None, **tkw):
        N = N or self.default_N
        t, tmin, tmax = self._process_params(N, tkw)
        if freqs is None:
            freqs = [1, N/4, N/2, N/16]
        tdiff = tmax - tmin

        x_freqs = []
        endpoint = tkw.get('endpoint', self.default_tkw.get('endpoint', False))
        t_all = _t(tmin, tdiff * len(freqs), N * len(freqs), endpoint)

        for i, f in enumerate(freqs):
            t = t_all[i*N : (i+1)*N]
            x_freqs.append(np.cos(2*pi * f * t))
        x, t = np.hstack(x_freqs), t_all

        return x, t

    def packed(self, N=None, freqs=None, overlap=.8, **tkw):
        N = N or self.default_N
        t, tmin, tmax = self._process_params(N, tkw)
        if freqs is None:
            freqs = [.5, 1, 2, N/10, N/10 + N/50, N/10 + N/25,
                     N/5, N/4, N/3, N/3 + N/10]
        N_overlap = int(overlap*len(t))

        x = np.zeros(len(t))
        for i, f in enumerate(freqs):
            idxs = (slice(0, N_overlap) if (i % 2 == 0) else
                    slice(-N_overlap, None))
            x[idxs] += np.cos(2*pi * f * t[idxs])
        return x, t

    #### Test functions ######################################################
    def demo(self, signals='all', sweep=False, N=None, dft=None):
        data = self.make_signals(signals, N)
        if dft not in (None, 'rows', 'cols'):
            raise ValueError(f"`dft` must be 'rows', 'cols', or None (got {dft})")
        elif dft == 'cols':
            dft_kw = dict(ncols=2, h=.55, w=1)
        elif dft == 'rows':
            dft_kw = dict(nrows=2)

        for name, (x, t, (fparams, aparams)) in data.items():
            title = self._title(name, len(x), fparams, aparams)
            if dft:
                axrf = np.abs(rfft(x))
                pkw = [{'title': title}, {'title': f"rDFT({name})"}]
                plots([t, None], [x, axrf], pkw=pkw, show=1, **dft_kw)
            else:
                plot(t, x, show=1, title=title)

    def test_transforms(self, fn, signals='all', N=None):
        """Make `fn` return `None` to skip visuals (e.g. if already done by `fn`).

        Input signature is `fn(x, t, params)`, where
        `params = (name, fparams, aparams)`. Output, if not None, must be
        `(Tf, pkw)`, where `Tf` is a 2D np.ndarray time-frequency transform,
        and `pkw` is keyword arguments to `ssqueezepy.visuals.imshow`
        (can be empty dict).
        """
        data = self.make_signals(signals, N)
        default_pkw = dict(abs=1, cmap='jet', show=1)

        for name, (x, t, (fparams, aparams)) in data.items():
            out = fn(x, t, (name, fparams, aparams))

            if out is not None:
                out, pkw = out
                default_pkw['title'] = self._title(name, N, fparams, aparams)
                for k, v in default_pkw.items():
                    pkw[k] = pkw.get(k, v)

                if isinstance(out, (tuple, list)):
                    for o in out:
                        imshow(o, **pkw)
                else:
                    imshow(out, **pkw)

    #### utils ###############################################################
    def make_signals(self, signals='all', N=None):
        def _process_args(name, fparams, aparams):
            fname, aname = (name.split(':') if ':' in name else
                            (name, ''))
            fname, aname = fname.replace('-', '_'), aname.replace('-', '_')
            fname = fname.lstrip('#')  # in case present

            fn  = (getattr(self, fname) if fname else
                   lambda *args, **kw: (np.ones(args[0]), None))
            afn = (getattr(self, aname) if aname else
                   lambda *args, **kw: (np.ones(args[0]), None))

            tkw = {}
            for dc in (fparams, aparams):  # `aparams` take precedence
                for k, v in dc.items():
                    if k in ('tmin', 'tmax', 'endpoint'):
                        tkw[k] = v
            return fn, afn, fname, aname, tkw

        names, params_all = self._process_input(signals)

        data = {}
        for name, (fparams, aparams) in zip(names, params_all):
            fn, afn, fname, aname, tkw = _process_args(name, fparams, aparams)
            x, t = fn(N, **fparams)
            x *= afn(len(x), **aparams, **tkw)[0]
            if name[0] == '#':
                x += x[::-1]

            data[name] = (x, t, (fparams, aparams))
        return data

    @classmethod
    def _title(self, signal, N, fparams, aparams, wrap_len=50):
        fparams = self._process_varname_alias(signal, N, fparams)
        fparams = dict(N=N, **fparams)

        ptxt = ', '.join(f"{k}={v}" for k, v in fparams.items())
        title = "{} | {}".format(signal, ptxt)
        if aparams:
            atxt = ', '.join(f"{k}={v}" for k, v in aparams.items())
            title += ', %s' % atxt
        title = _wrap(title, wrap_len)
        return title

    @staticmethod
    def _process_varname_alias(signal, N, fparams):
        fparams = fparams.copy()
        for k, v in fparams.items():
            if (k == 'fmax' and v is None and
                    signal in ['lchirp', 'echirp', 'hchirp']):
                fparams['fmax'] = N / 2
        return fparams

    def _process_params(self, N, tkw, fn=None, fmin=None, fmax=None):
        tkw = tkw.copy()
        for k in self.default_tkw:
            tkw[k] = tkw.get(k, self.default_tkw[k])

        if N is None:
            tmin, tmax = tkw['tmin'], tkw['tmax']
            if any(var is None for var in (tmin, tmax, fmin, fmax)):
                N = self.default_N
            else:
                f_fn = lambda *args, **kw: fn(*args, **kw, get_w=True)[1]
                N = self._est_N_nonalias(f_fn, tmin, tmax, fmin, fmax)

        if fmax is None:
            fmax = N // 2

        t = _t(**tkw, N=N)
        tmin, tmax = tkw['tmin'], tkw['tmax']
        return ((t, tmin, tmax, fmax) if fn else
                (t, tmin, tmax))

    def _est_N_nonalias(self, f_fn, tmin, tmax, fmin, fmax):
        """Find smallest `N` (number of samples) such that signal generated
        from `tmin` to `tmax` will not alias.

        https://dsp.stackexchange.com/a/72942/50076

        max_phi_increment = fmax_fn * (t[1] - t[0])
        t[1] - t[0] = (tmax - tmin) / (N - 1)  [[endpoint=True]]
        max_phi_increment = pi
        fmax_fn * (tmax - tmin) / (N - 1) = pi
        1 + fmax_fn * (tmax - tmin) / pi = N
        """
        # sample sufficiently finely
        t = np.linspace(tmin, tmax, 50000, endpoint=True)
        fmax_fn = np.max(f_fn(t, tmin, tmax, fmin, fmax))

        min_nonalias_N = int(np.ceil(1 + fmax_fn*(tmax - tmin)/pi))
        return min_nonalias_N

    def _process_input(self, signals):
        """
        `signals`:
            - Ensure is string, or list/tuple of strings or of lists/tuples,
            each list/tuple being a (str, dict) or (str, (dict, dict)) pair.
            - Ensure each string is in `SUPPORTED`, and has an accompanying
            `params` pair (if not, set from `defalt_args`).
            - Loads parameters into two separate dictionaries, one for
            'carrier' / base function, other for (amplitude) 'modulator'.
            Defaults loaded according to precedence: `name:am-name` overrides
            `name` and `am-name`, but latter two are used if former isn't set.
        """
        def raise_type_error(signal):
            raise TypeError("all tuple or list elements of `signals` "
                            "must be (str, dict) or (str, (dict, dict)) pairs "
                            "(got (%s))" % ', '.join(
                                map(lambda s: type(s).__name__, signal)))

        if isinstance(signals, (list, tuple)):
            for signal in signals:
                if isinstance(signal, str):
                    if ':' in signal:
                        fname, aname = signal.split(':')
                    else:
                        fname, aname = signal, ''

                    for name in (fname, aname):
                        if name != '' and name not in self.SUPPORTED:
                            raise ValueError(f"'{name}' is not supported; "
                                             "must be one of: "
                                             + ", ".join(self.SUPPORTED))
                elif isinstance(signal, (list, tuple)):
                    if not (isinstance(signal[0], str) and
                            isinstance(signal[1], (dict, list, tuple))):
                        raise_type_error(signal)
                    elif (isinstance(signal[1], (list, tuple)) and
                          not (isinstance(signal[1][0], dict) and
                               isinstance(signal[1][1], dict))):
                        raise_type_error(signal)
                else:
                    raise TypeError("all elements of `signals` must be string, "
                                    "or tuple or list of (string, dict) or "
                                    "(string, (dict, dict)) pairs "
                                    "(found %s)" % type(signal))
        elif not isinstance(signals, str):
            raise TypeError("`signals` must be string, list, or tuple "
                            "(got %s)" % type(signals))

        if signals == 'all':
            signals = self.DEMO.copy()
        elif not isinstance(signals, (list, tuple)):
            signals = [signals]

        names, params_all = [], []
        for signal in signals:
            if isinstance(signal, (tuple, list)):
                name, params = signal
                if isinstance(params, (list, tuple)):
                    fparams, aparams = params
                else:
                    fparams, aparams = params, {}
            else:
                name, fparams, aparams = signal, {}, {}

            if name[0] == '#':
                add_reversed = True
                name = name[1:]
            else:
                add_reversed = False

            if 'am-' in name:
                if name.startswith('am-'):
                    if name.endswith(':'):
                        name = name.rstrip(':')
                    fname, aname = 'cosine', name
                    defaults = (self.default_args.get(fname, {}),
                                self.default_args.get(aname, {}))
                    name = fname + ':' + aname
                else:
                    defaults = self.default_args.get(name, {})
                    fname, aname = name.split(':')

                if isinstance(defaults, (list, tuple)):
                    fdefaults, adefaults = defaults
                elif isinstance(defaults, dict) and defaults != {}:
                    fdefaults, adefaults = defaults, {}
                else:
                    fdefaults, adefaults = self.default_args.get(fname, {}), {}

                if adefaults == {}:
                    adefaults = self.default_args.get(aname, {})

                for k, v in fdefaults.items():
                    fparams[k] = fparams.get(k, v)
                for k, v in adefaults.items():
                    aparams[k] = aparams.get(k, v)

                if name.startswith('am-'):
                    fdefaults, adefaults = adefaults, fdefaults
            else:
                for k, v in self.default_args.get(name, {}).items():
                    fparams[k] = fparams.get(k, v)

            if add_reversed:
                name = '#' + name
            names.append(name)
            params_all.append([fparams, aparams])

        # store latest result for debug purposes
        self._names = names
        self._params_all = params_all
        return names, params_all

    #### prebuilt test methods ##############################################
    def wavcomp(self, wavelets, signals='all', N=None, w=1.2, h=None,
                tight_kw=None):
        if not isinstance(wavelets, (list, tuple)):
            wavelets = [wavelets]
        wavs = []
        for wavelet in wavelets:
            wavs.append(Wavelet._init_if_not_isinstance(wavelet))

        fn = lambda x, t, params: self._wavcomp_fn(
            x, t, params, wavelets, w=w, h=h, tight_kw=tight_kw)
        self.test_transforms(fn, signals=signals, N=N)

    def _wavcomp_fn(self, x, t, params, wavelets, w=1.2, h=None, tight_kw=None):
        h = h or .45 * len(wavelets)
        fig, axes = plt.subplots(len(wavelets), 2, figsize=(w * 12, h * 12))
        for i, wavelet in enumerate(wavelets):
            Tx, Wx, *_ = ssq_cwt(x, wavelet, t=t)
            Tx = np.flipud(Tx)

            name, fparams, aparams = params
            title1, title2 = self._title_cwt(wavelet, name, x, fparams, aparams)

            pkw = dict(abs=1, cmap='jet', ticks=0, fig=fig)
            imshow(Wx, **pkw, ax=axes[i, 0], show=0, title=title1)
            imshow(Tx, **pkw, ax=axes[i, 1], show=0, title=title2)

        tight_kw = tight_kw or {}
        defaults = dict(left=0, right=1, bottom=0, top=1, wspace=.01,
                        hspace=.13)
        for k, v in defaults.items():
            tight_kw[k] = v
        plt.subplots_adjust(**tight_kw)
        plt.show()

    def cwt_vs_stft(self, wavelet, window, signals='all', N=None,
                    win_len=None, n_fft=None, window_name=None, config_str='',
                    w=1.2, h=.9, tight_kw=None):
        fn = lambda x, t, params: self._cwt_vs_stft_fn(
            x, t, params, wavelet, window, win_len, n_fft, window_name,
            config_str, w, h, tight_kw)
        self.test_transforms(fn, signals=signals, N=N)

    def _cwt_vs_stft_fn(self, x, t, params, wavelet, window, win_len=None,
                        n_fft=None, window_name=None, config_str='', w=1.2, h=.9,
                        tight_kw=None):
        fs = 1 / (t[1] - t[0])
        Tsx, Sx, *_ = ssq_stft(x, window, n_fft=n_fft, win_len=win_len, fs=fs)
        Twx, Wx, *_ = ssq_cwt(x, wavelet, t=t)
        Twx, Tsx, Sx = np.flipud(Twx), np.flipud(Tsx), np.flipud(Sx)

        fig, axes = plt.subplots(2, 2, figsize=(w * 12, h * 12))

        name, fparams, aparams = params
        ctitle1, ctitle2 = self._title_cwt( wavelet, name, x, fparams, aparams)
        stitle1, stitle2 = self._title_stft(window,  name, x, fparams, aparams,
                                            win_len, n_fft, window_name,
                                            config_str)

        pkw = dict(abs=1, cmap='jet', ticks=0, fig=fig)
        imshow(Wx,  **pkw, ax=axes[0, 0], show=0, title=ctitle1)
        imshow(Twx, **pkw, ax=axes[0, 1], show=0, title=ctitle2)
        imshow(Sx,  **pkw, ax=axes[1, 0], show=0, title=stitle1)
        imshow(Tsx, **pkw, ax=axes[1, 1], show=0, title=stitle2)

        tight_kw = tight_kw or {}
        defaults = dict(left=0, right=1, bottom=0, top=1, wspace=.01,
                        hspace=.2)
        for k, v in defaults.items():
            tight_kw[k] = v
        plt.subplots_adjust(**tight_kw)
        plt.show()

    @staticmethod
    def _title_cwt(wavelet, name, x, fparams, aparams, wrap_len=50):
        title = TestSignals._title(name, len(x), fparams, aparams)

        # special case: GMW
        wname = wavelet.name.replace(' L1', '').replace(' L2', '')

        twav = '%s wavelet | %s' % (wname, wavelet.config_str)
        ctitle1 = title + '\nabs(CWT) | ' + twav
        ctitle2 = 'abs(SSQ_CWT)'

        ctitle1 = _wrap(ctitle1, wrap_len)
        return ctitle1, ctitle2

    @staticmethod
    def _title_stft(window, name, x, fparams, aparams, win_len=None, n_fft=None,
                    window_name='', config_str='', wrap_len=50):
        title = TestSignals._title(name, len(x), fparams, aparams)

        twin = "{} window | win_len={}, n_fft={}, {}".format(
            window_name, win_len, n_fft, config_str)
        stitle1 = title + '\nabs(STFT) | ' + twin
        stitle2 = 'abs(SSQ_STFT)'

        stitle1 = _wrap(stitle1, wrap_len)
        return stitle1, stitle2


def _wrap(txt, wrap_len=50):
    """Preserves line breaks and includes `'\n'.join()` step."""
    return '\n'.join(['\n'.join(
        wrap(line, 90, break_long_words=False, replace_whitespace=False))
        for line in txt.splitlines() if line.strip() != ''])


def _t(tmin, tmax, N, endpoint=False):
    return np.linspace(tmin, tmax, N, endpoint=endpoint)
