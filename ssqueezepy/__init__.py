# -*- coding: utf-8 -*-
"""
MIT License
===========

Some ssqueezepy source files under other terms (see NOTICE.txt).

Copyright (c) 2020 John Muradeli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


__version__ = '0.6.6-dev'
__title__ = 'ssqueezepy'
__author__ = 'John Muradeli'
__license__ = __doc__
__project_url__ = 'https://github.com/OverLordGoldDragon/ssqueezepy'


try:
    import matplotlib.pyplot as plt
except ImportError:
    class PltDummy():
        def __getattr__(self, name):
            raise ValueError("`ssqueezepy.visuals` requires "
                             "`matplotlib` installed.")
    plt = PltDummy()


from . import utils
from . import ssqueezing
from . import _cwt
from . import _stft
from . import _ssq_cwt
from . import _ssq_stft
from . import _gmw
from . import _test_signals
from . import wavelets
from . import ridge_extraction
from . import toolkit
from . import visuals
from . import algos
from . import configs
from . import experimental


from .ssqueezing import *
from ._cwt import *
from ._stft import *
from ._ssq_cwt import *
from ._ssq_stft import *
from ._gmw import *
from ._test_signals import *
from .wavelets import *
from .ridge_extraction import *
from .utils.fft_utils import *

from .configs import IS_PARALLEL, USE_GPU


def wavs():
    return wavelets.Wavelet.SUPPORTED


_modules_toplevel = [
    '_cwt', '_gmw', '_ssq_cwt', '_ssq_stft', '_stft', '_test_signals', 'algos',
    'configs', 'experimental', 'ridge_extraction', 'ssqueezing', 'toolkit',
    'visuals', 'wavelets', 'utils'
]
