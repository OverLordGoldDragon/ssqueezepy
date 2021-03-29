# -*- coding: utf-8 -*-
"""Show CWT with higher-order Generalized Morse Wavelets on parallel reflect-added
linear chirps, with and without noise, and show GMW waveforms.
"""
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import numpy as np
from ssqueezepy import cwt, TestSignals
from ssqueezepy.visuals import viz_cwt_higher_order, viz_gmw_orders

#%%# CWT with higher-order GMWs #############################################
N = 1024
order = 2

tsigs = TestSignals()
x, t = tsigs.par_lchirp(N=N)
x += x[::-1]

for noise in (False, True):
    if noise:
        x += np.random.randn(len(x))
    Wx_k, scales = cwt(x, 'gmw', order=range(order + 1), average=False)

    viz_cwt_higher_order(Wx_k, scales, 'gmw')
    print("=" * 80)

#%%# Higher-order GMWs #######################################################
gamma, beta, norm = 3, 60, 'bandpass'
n_orders = 3
scale = 5

viz_gmw_orders(N, n_orders, scale, gamma, beta, norm)
