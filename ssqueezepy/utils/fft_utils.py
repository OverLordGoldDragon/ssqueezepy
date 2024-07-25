# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
from scipy.fft import fftshift as sfftshift, ifftshift as sifftshift
from scipy.fft import fft as sfft, rfft as srfft, ifft as sifft, irfft as sirfft
from pathlib import Path
from . import backend as S
from ..configs import IS_PARALLEL

try:
    from torch.fft import (fft as tfft, rfft as trfft,
                           ifft as tifft, irfft as tirfft,
                           fftshift as tfftshift, ifftshift as tifftshift)
except ImportError:
    pass

try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(600)
except ImportError:
    pyfftw = None

UTILS_DIR = Path(__file__).parent

__all__ = [
    'fft',
    'rfft',
    'ifft',
    'irfft',
    'fftshift',
    'ifftshift',
    'FFT',
    'FFT_GLOBAL',
]

#############################################################################


class FFT():
    """Global class for ssqueezepy FFT methods.

    Will use GPU via PyTorch if environment flag `'SSQ_GPU'` is set to `'1'`.
    Will use `scipy.fft` or `pyfftw` depending on `patience` argument (and
    whether `pyfftw` is installed).
    Both will use `threads` CPUs to accelerate computing.

    In a nutshell, if you plan on re-running FFT on input of same shape and dtype,
    prefer `patience=1`, which introduces a lengthy first-time overhead but may
    compute significantly faster afterwards.

    # Arguments (`fft`, `rfft`, `ifft`, `irfft`):
        x: np.ndarray
            1D or 2D.

        axis: int
            FFT axis. One of `0, 1, -1`.

        patience: int / tuple[int, int]
            If int:
                0: will use `scipy.fft`
                1: `pyfftw` with flag `'FFTW_PATIENT'`
                2: `pyfftw` with flag `'FFTW_EXHAUSTIVE'`
            Else, if tuple, second element specifies `planning_timelimit`
            passed to `pyfftw.FFTW` (so tuple requires `patience[0] != 0`).

            Set `planning_timelimit = None` to allow planning to finish,
            but beware; `patience = 1` can take hours for large inputs, and `2`
            even longer.

        astensor: bool (default False)
            If computing on GPU, whether to return as `torch.Tensor` (if False,
            will move to CPU and convert to `numpy.ndarray`).

        n: int / None
            Only for `irfft`; length of original input. If None, will default to
            `2*(x.shape[axis] - 1)`.

    __________________________________________________________________________
    # Arguments (`__init__`):
        planning_timelimit: int
            Default.

        wisdom_dir: str
            Where to save wisdom to or load from. Empty string means
            `ssqueezepy/utils/`.

        threads: int
            Number of CPU threads to use. -1 = maximum.

        patience: int
            Default `patience`.

        cache_fft_objects: bool (default False)
            If True, `pyfftw` objects generated throughout session are stored in
            `FFT._input_history`, and retrieved if all of below match:
                `(x.shape, x.dtype, real, patience, n)`
            where `patience` includes `planning_timelimit` as a tuple.
            Default False since loading from wisdom is very fast anyway.

        verbose: bool (default True)
            Controls whether a message is printed upon `patience >= 1`.
    __________________________________________________________________________
    **Wisdom**

    `pyfftw` uses "wisdom", basically storing and reusing generated FFT plans
    if input attributes match:
        (`x.shape`, `x.dtype`, `axis`, `flags`, `planning_timelimit`)
    `flags` and `planning_timelimit` are set via `patience`.

    With each `pyfftw` use, `save_wisdom()` is called, writing to `wisdom32` and
    `wisdom64` bytes files in `ssqueezepy/utils`. Each time ssqueezepy runs in a
    new session, `load_wisdom()` is called to load these values, so wisdom is
    only expansive.
    """
    def __init__(self, planning_timelimit=120, wisdom_dir=UTILS_DIR, threads=None,
                 patience=0, cache_fft_objects=False, verbose=1):
        self.planning_timelimit = planning_timelimit
        self.wisdom_dir = wisdom_dir
        self._user_threads = threads
        self._patience = patience  # default patience
        self._process_patience(patience)  # error if !=0 and pyfftw not installed
        self.cache_fft_objects = cache_fft_objects
        self.verbose = verbose

        if pyfftw is not None:
            pyfftw.config.NUM_THREADS = self.threads

            self._wisdom32_path = str(Path(self.wisdom_dir, 'wisdom32'))
            self._wisdom64_path = str(Path(self.wisdom_dir, 'wisdom64'))
            self._wisdom32, self._wisdom64 = b'', b''
            self._input_history = {}
            self.load_wisdom()

    @property
    def threads(self):
        """Set dynamically if `threads` wasn't passed in __init__."""
        if self._user_threads is None:
            return (multiprocessing.cpu_count() if IS_PARALLEL() else 1)
        return self._user_threads

    @property
    def patience(self):
        """Setter will also set `planning_timelimit` if setting to tuple."""
        return self._patience

    @patience.setter
    def patience(self, value):
        self._validate_patience(value)
        if isinstance(value, tuple):
            self._patience, self.planning_timelimit = value
        else:
            self._patience = value

    #### Main methods #########################################################
    def fft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('fft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sfft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=False)
        return fft_object()

    def rfft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('rfft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return srfft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=True)
        return fft_object()

    def ifft(self, x, axis=-1, patience=None, astensor=False):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('ifft', x, dim=axis, astensor=astensor)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sifft(x, axis=axis, workers=self.threads)

        fft_object = self._get_save_fill(x, axis, patience, real=False,
                                         inverse=True)
        return fft_object()

    def irfft(self, x, axis=-1, patience=None, astensor=False, n=None):
        """See `help(ssqueezepy.utils.FFT)`."""
        out = self._maybe_gpu('irfft', x, dim=axis, astensor=astensor, n=n)
        if out is not None:
            return out

        patience = self._process_patience(patience)
        if patience == 0:
            return sirfft(x, axis=axis, workers=self.threads, n=n)

        fft_object = self._get_save_fill(x, axis, patience, real=True,
                                         inverse=True, n=n)
        return fft_object()

    def fftshift(self, x, axes=-1, astensor=False):
        out = self._maybe_gpu('fftshift', x, dim=axes, astensor=astensor)
        if out is not None:
            return out
        return sfftshift(x, axes=axes)

    def ifftshift(self, x, axes=-1, astensor=False):
        out = self._maybe_gpu('ifftshift', x, dim=axes, astensor=astensor)
        if out is not None:
            return out
        return sifftshift(x, axes=axes)

    def _maybe_gpu(self, name, x, astensor=False, **kw):
        if S.is_tensor(x):
            fn = {'fft': tfft, 'ifft': tifft,
                  'rfft': trfft, 'irfft': tirfft,
                  'fftshift': tfftshift, 'ifftshift': tifftshift}[name]
            out = fn(S.asarray(x), **kw)
            return out if astensor else out.cpu().numpy()
        return None

    #### FFT makers ###########################################################
    def _get_save_fill(self, x, axis, patience, real, inverse=False, n=None):
        fft_object = self.get_fft_object(x, axis, patience, real, inverse, n)
        self.save_wisdom()
        fft_object.input_array[:] = x
        return fft_object

    def get_fft_object(self, x, axis, patience=1, real=False, inverse=False,
                       n=None):
        combo = (x.shape, x.dtype, axis, real, n)
        if self.cache_fft_objects and combo in self._input_history:
            fft_object = self._input_history[combo]
        else:
            fft_object = self._get_fft_object(x, axis, patience, real, inverse, n)
            if self.cache_fft_objects:
                self._input_history[combo] = fft_object
        return fft_object

    def _get_fft_object(self, x, axis, patience, real, inverse, n):
        (shapes, dtypes, flags, planning_timelimit, direction
         ) = self._process_input(x, axis, patience, real, inverse, n)
        shape_in, shape_out = shapes
        dtype_in, dtype_out = dtypes

        a = pyfftw.empty_aligned(shape_in,  dtype=dtype_in)
        b = pyfftw.empty_aligned(shape_out, dtype=dtype_out)
        fft_object = pyfftw.FFTW(a, b, axes=(axis,), flags=flags,
                                 planning_timelimit=planning_timelimit,
                                 direction=direction, threads=self.threads)
        return fft_object

    def _process_input(self, x, axis, patience, real, inverse, n):
        self._validate_input(x, axis, real, patience, inverse)

        # patience, planning time, forward / inverse
        if isinstance(patience, tuple):
            patience, planning_timelimit = patience
        else:
            planning_timelimit = self.planning_timelimit
        flags = ['FFTW_PATIENT'] if patience == 1 else ['FFTW_EXHAUSTIVE']
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'

        # shapes
        shape_in = x.shape
        shape_out = self._get_output_shape(x, axis, real, inverse, n)

        # dtypes
        double = x.dtype in (np.float64, np.complex128)
        cdtype = 'complex128' if double else 'complex64'
        rdtype = 'float64'    if double else 'float32'
        dtype_in  = rdtype if (real and not inverse) else cdtype
        dtype_out = rdtype if (real and inverse)     else cdtype

        # notify user of procedure
        if self.verbose:
            if planning_timelimit is None:
                adjective = "very long" if patience == 2 else "long"
                print("Planning optimal FFT algorithm; this may "
                      "take %s..." % adjective)
            else:
                print("Planning optimal FFT algorithm; this will take up to "
                      "%s secs" % planning_timelimit)

        return ((shape_in, shape_out), (dtype_in, dtype_out), flags,
                planning_timelimit, direction)

    def _get_output_shape(self, x, axis, real=False, inverse=False, n=None):
        if not inverse:
            n_fft = x.shape[axis]
            fft_out_len = (n_fft//2 + 1) if real else n_fft
        else:
            if real:
                n_fft = n if (n is not None) else 2*(x.shape[axis] - 1)
            else:
                n_fft = x.shape[axis]
            fft_out_len = n_fft

        if x.ndim != 1:
            shape = list(x.shape)
            shape[axis] = fft_out_len
            shape = tuple(shape)
        else:
            shape = (fft_out_len,)
        return shape

    #### Misc #################################################################
    def load_wisdom(self):
        for name in ('wisdom32', 'wisdom64'):
            path = getattr(self, f"_{name}_path")
            if Path(path).is_file():
                with open(path, 'rb') as f:
                    setattr(self, f"_{name}", f.read())
        pyfftw.import_wisdom((self._wisdom64, self._wisdom32, b''))

    def save_wisdom(self):
        """Will overwrite."""
        self._wisdom64, self._wisdom32, _ = pyfftw.export_wisdom()
        for name in ('wisdom32', 'wisdom64'):
            path = getattr(self, f"_{name}_path")
            with open(path, 'wb') as f:
                f.write(getattr(self, f"_{name}"))

    def _validate_input(self, x, axis, real, patience, inverse):
        """Assert is single/double precision and is 1D/2D."""
        supported = ('float32', 'float64', 'complex64', 'complex128')
        dtype = str(x.dtype)
        if dtype not in supported:
            raise TypeError("unsupported `x.dtype`: %s " % dtype
                            + "(must be one of: %s)" % ', '.join(supported))
        if (real and not inverse) and dtype.startswith('complex'):
            raise TypeError("`x` cannot be complex for `rfft`")

        if axis not in (0, 1, -1):
            raise ValueError("unsupported `axis`: %s " % axis
                             + "; must be 0, 1, or -1")

        self._validate_patience(patience)

    def _validate_patience(self, patience):
        if not isinstance(patience, (int, tuple)):
            raise TypeError("`patience` must be int or tuple "
                            "(got %s)" % type(patience))
        elif isinstance(patience, int):
            from .common import assert_is_one_of
            assert_is_one_of(patience, 'patience', (0, 1, 2))

    def _process_patience(self, patience):
        patience = patience if (patience is not None) else self.patience
        if pyfftw is None and patience != 0:
            raise ValueError("`patience != 0` requires `pyfftw` installed.")
        return patience


FFT_GLOBAL = FFT()

fft   = FFT_GLOBAL.fft
rfft  = FFT_GLOBAL.rfft
ifft  = FFT_GLOBAL.ifft
irfft = FFT_GLOBAL.irfft
fftshift  = FFT_GLOBAL.fftshift
ifftshift = FFT_GLOBAL.ifftshift
