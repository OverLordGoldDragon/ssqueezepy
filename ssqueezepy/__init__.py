# -*- coding: utf-8 -*-
"""
MIT License
===========

Some ssqueezepy source files under other terms (see NOTICE.txt).

Copyright (c) 2020 OverLordGoldDragon

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


__version__ = '0.5.1'
__title__ = 'ssqueezepy'
__author__ = 'OverLordGoldDragon'
__license__ = __doc__
__project_url__ = 'https://github.com/OverLordGoldDragon/ssqueezepy'


from . import ssqueezing
from . import _cwt
from . import _stft
from . import _ssq_cwt
from . import _ssq_stft
from . import wavelets
from . import utils
from . import toolkit
from . import visuals
from . import algos
from . import experimental

from .ssqueezing import *
from ._ssq_cwt import *
from ._ssq_stft import *
from ._cwt import *
from ._stft import *
from .wavelets import *


def wavs():
    return wavelets.Wavelet.SUPPORTED
