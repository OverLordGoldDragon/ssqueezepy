# -*- coding: utf-8 -*-
import numpy as np
import scipy.signal as sig
from ssqueezepy import Wavelet, TestSignals
from ssqueezepy.utils import window_resolution

tsigs = TestSignals(N=256)
#%%# Viz signals #############################################################
# set `dft` to 'rows' or 'cols' to also plot signals' DFT, along rows or columns
dft = (None, 'rows', 'cols')[0]
tsigs.demo(dft=dft)

#%%# How to specify `signals` ################################################
signals = [
    'am-cosine',
    ('hchirp', dict(fmin=.2)),
    ('sine:am-cosine', (dict(f=32, phi=1), dict(amin=.3))),
]
tsigs.demo(signals, N=512)

#%%# With `dft` ##################
tsigs.demo(signals, dft='rows')
tsigs.demo(signals, dft='cols')

#%%# Viz CWT & SSQ_CWT with different wavelets ###############################
tsigs = TestSignals(N=256)
wavelets = [Wavelet(('gmw', {'beta': 5})),
            Wavelet(('gmw', {'beta': 22}))]
tsigs.wavcomp(wavelets)

#%%#
tsigs.wavcomp(wavelets, signals=[('#echirp', dict(fmin=.1))], N=512)

#%%# Viz CWT vs STFT (& SSQ'd) ###############################################
N  = 256
n_fft = N
win_len = n_fft
wavelet = Wavelet(('GMW', {'beta': 21.5}))

NW = win_len//2 - 1
window = np.abs(sig.windows.dpss(win_len, NW))
window_name = 'DPSS'
config_str = '\nNW=%s' % NW

# ensure `wavelet` and `window` have ~same time & frequency resolutions
print("std_w, std_t, harea\nwavelet: {:.4f}, {:.4f}, {:.8f}"
      "\nwindow:  {:.4f}, {:.4f}, {:.8f}".format(
          wavelet.std_w, wavelet.std_t, wavelet.harea,
          *window_resolution(window)))
#%%
tsigs.cwt_vs_stft(wavelet, window, signals='all', N=N, win_len=win_len,
                  n_fft=n_fft, window_name=window_name, config_str=config_str)
