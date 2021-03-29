# -*- coding: utf-8 -*-
"""Experimental feature example."""
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import numpy as np
from ssqueezepy import TestSignals, ssq_cwt, Wavelet
from ssqueezepy.visuals import imshow
from ssqueezepy.experimental import phase_ssqueeze

#%%
x = TestSignals(N=2048).par_lchirp()[0]
x += x[::-1]
wavelet = Wavelet()

Tx0, Wx, _, scales, *_ = ssq_cwt(x, wavelet, get_dWx=1)
Tx1, *_ = phase_ssqueeze(Wx, wavelet=wavelet, scales=scales, flipud=1)

adiff = np.abs(Tx0 - Tx1)
print(adiff.mean(), adiff.max(), adiff.sum())
#%%
# main difference near boundaries; see `help(trigdiff)` w/ `rpadded=False`
imshow(Tx1, abs=1)
