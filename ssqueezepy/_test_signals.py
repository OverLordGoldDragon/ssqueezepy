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

3. **lchirp**: linear chirp, `cos(2pi t**2/2)`, spanning `fmin` to `fmax`

4. **echirp**: exponential chirp, `cos(2pi exp(t))`, spanning `fmin` to `fmax`

5. **hchirp**: hyperbolic chirp, `cos(2pi a/(b - t))`, spanning `fmin` to `fmax`

6. **jumps**: large instant frequency transitions, `cos(2pi f*t), f=2 -> f=100`

Note that for signals involving `fmax`, aliasing may occur even if `fmax < N/2`:
https://dsp.stackexchange.com/q/72329/50076
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from textwrap import wrap

from ._ssq_cwt import ssq_cwt
from ._ssq_stft import ssq_stft
from .wavelets import Wavelet
from .visuals import plot, plots, imshow


pi = np.pi
DEFAULT_N = 256
DEFAULT_ARGS = {
    'sine':   dict(f=8, phi=0),
    'cosine': dict(f=8, phi=0),
    'lchirp': dict(tmin=0, tmax=1, fmin=0, fmax=None),
    'echirp': dict(tmin=0, tmax=1, fmin=0, fmax=None),
    'hchirp': dict(tmin=0, tmax=1, fmin=1, fmax=None),
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
def _t(tmin, tmax, N, endpoint=False):
    return np.linspace(tmin, tmax, N, endpoint=endpoint)


class TestSignals():
    """Signals of varying time-frequency characteristics. Convenience methods
    to plot multiple signals and their transforms under varying wavelet / window
    parameters.

    `.demo(signals)` to visualize `signals`, `test_transforms(fn)` to apply `fn`
    to and visualize output.

    See `examples/` on Github, and
    https://overlordgolddragon.github.io/test-signals/

    # TODO complete docstrings
    """
    SUPPORTED = ['sine', 'cosine', 'lchirp', 'echirp', 'hchirp', 'jumps',
                 'am-sine', 'am-cosine', 'am-exp', 'am-gauss']
    # extra for when signals='all' is passed in
    EXTRAS_ALL = ['#lchirp', '#echirp', '#hchirp']

    def __init__(self, N=None, default_args=None, default_tkw=None):
        self.default_N    = N    or DEFAULT_N
        self.default_args = default_args or DEFAULT_ARGS
        self.default_tkw  = default_tkw  or DEFAULT_TKW

        # set defaults on unspecified
        for k, v in DEFAULT_ARGS.items():
            self.default_args[k] = self.default_args.get(k, v)
        for k, v in DEFAULT_TKW.items():
            self.default_tkw[k] = self.default_tkw.get(k, v)

    #### test signals ########################################################
    def sine(self, N, f=1, phi=0, **tkw):
        """sin(2pi*f*t + phi)"""
        tkw['endpoint'] = tkw.get('endpoint', False)
        t, *_ = self._process_tkw(N, tkw)
        return np.sin(2*pi * f * t + phi), t

    def cosine(self, N, f=1, phi=0, **tkw):
        """cos(2pi*f*t + phi)"""
        return self.sine(N, f, phi=pi/2 + phi, **tkw)

    def lchirp(self, N, fmin=0, fmax=None, **tkw):
        """
        >>>   f(t) = a*t + b
        >>> phi(t) = (a/2)*(t^2 - tmin^2) + b*(t - tmin)
        >>> a = (fmin - fmax) / (tmin - tmax)
            b = (fmin*tmax - fmax*tmin) / (tmax - tmin)
        """
        t, tmin, tmax = self._process_tkw(N, tkw)
        fmax = fmax or N / 2
        a = (fmin - fmax) / (tmin - tmax)
        b = (fmin*tmax - fmax*tmin) / (tmax - tmin)

        phi = (a/2)*(t**2 - tmin**2) + b*(t - tmin)
        return np.cos(2*pi * phi), t

    def echirp(self, N, fmin=0, fmax=None, **tkw):
        """
        >>> f(t)   = a*exp(t) + b
        >>> phi(t) = a*(exp(t) - exp(tmin)) + b*(t - tmin)
        >>> a = (fmax - fmin)/(exp(tmax) - exp(tmin))
            b = (fmin*exp(tmax) - fmax*exp(tmin)) / (exp(tmax) - exp(tmin))
        """
        t, tmin, tmax = self._process_tkw(N, tkw)
        fmax = fmax or N / 2
        a, b, c, d = fmin, fmax, tmin, tmax
        A = (b - a) / (np.exp(d) - np.exp(c))
        B = (a*np.exp(d) - b*np.exp(c)) / (np.exp(d) - np.exp(c))

        phi = A*(np.exp(t) - np.exp(tmin)) + B*(t - tmin)
        return np.cos(2*pi * phi), t

    def am_sine(self, N, f=1, amin=0, amax=1, phi=0, **tkw):
        _A, t = self.sine(N, f, phi, **tkw)
        _A = (_A + 1) / 2
        return amin + (amax - amin) * _A, t

    def am_cosine(self, N, f=1, amin=0, amax=1, phi=0, **tkw):
        _A, t = self.cosine(N, f, phi, **tkw)
        _A = (_A + 1) / 2
        return amin + (amax - amin) * _A, t

    def am_exp(self, N, amin=.1, amax=1, **tkw):
        """Use `echirp`'s expression for `f(t)`"""
        t, tmin, tmax = self._process_tkw(N, tkw)
        a, b, c, d = amin, amax, tmin, tmax
        A = (b - a) / (np.exp(d) - np.exp(c))
        B = (a*np.exp(d) - b*np.exp(c)) / (np.exp(d) - np.exp(c))
        return A*np.exp(t) + B, t

    def am_gauss(self, N, amin=.1, amax=1, **tkw):
        t = _t(-1, 1, N)
        _A = np.exp( -((t - t.mean())**2 * 5) )
        return amin + (amax - amin)*_A, t

    def hchirp(self, N, fmin=1, fmax=None, **tkw):
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
        t, tmin, tmax = self._process_tkw(N, tkw)
        fmax = fmax or N / 2
        a, b, c, d = fmin, fmax, tmin, tmax

        AN = (2*np.sqrt(a**3*b**3*(c - d)**4) + a**2*b*(c - d)**2
              + a*b**2*(c - d)**2)
        AD = (a - b)**2
        BN = np.sqrt(a**3*b**3*(c - d)**4) + a**2*b*c*(c - d) + a*b**2*d*(d - c)
        BD = a*b*(a - b)*(c - d)
        A = AN / AD
        B = BN / BD

        phi = A * (1/(B - t) + 1/(tmin - B))
        return np.cos(2*np.pi * phi), t

    def jumps(self, N, freqs=None, **tkw):
        t, tmin, tmax = self._process_tkw(N, tkw)
        if freqs is None:
            freqs = [1, N/4, N/2, N/16]
        tdiff = tmax - tmin

        x_freqs = []
        endpoint = tkw.get('endpoint', DEFAULT_TKW.get('endpoint', False))
        t_all = _t(tmin, tdiff * len(freqs), N * len(freqs), endpoint)

        for i, f in enumerate(freqs):
            t = t_all[i*N : (i+1)*N]
            x_freqs.append(np.cos(2*pi * f * t))
        x, t = np.hstack(x_freqs), t_all

        return x, t

    #### Test functions ######################################################
    def demo(self, signals='all', sweep=False, N=None, dft=None):
        N = N or self.default_N
        data = self.make_signals(signals, N)
        if dft not in (None, 'rows', 'cols'):
            raise ValueError(f"`dft` must be 'rows', 'cols', or None (got {dft})")
        elif dft == 'cols':
            dft_kw = dict(ncols=2, h=.55, w=1)
        elif dft == 'rows':
            dft_kw = dict(nrows=2)

        for name, (x, t, (fparams, aparams)) in data.items():
            title = self._title(name, N, fparams, aparams)
            if dft:
                axrf = np.abs(rfft(x))
                pkw = [{'title': title}, {'title': f"rDFT({name})"}]
                plots([t, None], [x, axrf], pkw=pkw, show=1, **dft_kw)
            else:
                plot(t, x, show=1, title=title)

    def test_transforms(self, fn, signals='all', N=None):
        """Make `fn` return `None` to skip visuals (e.g. if already done by `fn`).
        """
        N = N or self.default_N
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
        N = N or self.default_N

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
        fparams = self._process_alias(signal, N, fparams)
        fparams = dict(N=N, **fparams)

        ptxt = ', '.join(f"{k}={v}" for k, v in fparams.items())
        title = "{} | {}".format(signal, ptxt)
        if aparams:
            atxt = ', '.join(f"{k}={v}" for k, v in aparams.items())
            title += ', %s' % atxt
        title = _wrap(title, wrap_len)
        return title

    @staticmethod
    def _process_alias(signal, N, fparams):
        fparams = fparams.copy()
        for k, v in fparams.items():
            if (k == 'fmax' and v is None and
                    signal in ['lchirp', 'echirp', 'hchirp']):
                fparams['fmax'] = N / 2
        return fparams

    def _process_tkw(self, N, tkw):
        tkw = tkw.copy()
        for k in DEFAULT_TKW:
            tkw[k] = tkw.get(k, DEFAULT_TKW[k])
        t = _t(**tkw, N=N)
        return (t, *[tkw[k] for k in ('tmin', 'tmax')])

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
            signals = self.SUPPORTED.copy()
            for extra in self.EXTRAS_ALL:
                signals.insert(signals.index(extra[1:]) + 1, extra)
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

        fn = lambda x, t, params: self._wavcomp_fn(x, t, params, wavelets)
        self.test_transforms(fn, signals=signals, N=N)

    def _wavcomp_fn(self, x, t, params, wavelets, w=1.2, h=None, tight_kw=None):
        h = h or .45 * len(wavelets)
        fig, axes = plt.subplots(len(wavelets), 2, figsize=(w * 12, h * 12))
        for i, wavelet in enumerate(wavelets):
            Tx, _, Wx, *_ = ssq_cwt(x, wavelet, t=t)
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
        Tsx, _, Sx, *_ = ssq_stft(x, window, n_fft=n_fft, win_len=win_len, fs=fs)
        Twx, _, Wx, *_ = ssq_cwt(x, wavelet, t=t)
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
