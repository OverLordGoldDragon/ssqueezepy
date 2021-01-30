# -*- coding: utf-8 -*-
from ssqueezepy import Wavelet, TestSignals

#%%# Viz signals #############################################################
# set `dft` to 'rows' or 'cols' to also plot signals' DFT, along rows or columns
tsigs = TestSignals(N=256)
dft = (None, 'rows', 'cols')[0]
tsigs.demo(dft=dft)

#%%# How to specify `signals` ################################################
signals = [
    'am-cosine',
    ('hchirp', dict(fmin=.2)),
    ('sine:am-cosine', (dict(f=32, phi=1), dict(amin=.3))),
]
tsigs.demo(signals, N=512)

#%%# Viz CWT & SSQ_CWT with different wavelets ###############################
wavelets = [Wavelet(('GMW', {'beta': 5})),
            Wavelet(('GMW', {'beta': 22})),
            ]
tsigs.wavcomp(wavelets)

#%%#
tsigs.wavcomp(wavelets, signals=[('#echirp', dict(fmin=.1))], N=512)
