# -*- coding: utf-8 -*-
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import numpy as np
import scipy.signal as sig
from ssqueezepy import Wavelet, TestSignals
from ssqueezepy.utils import window_resolution

tsigs = TestSignals(N=2048)
#%%# Viz signals #############################################################
# set `dft` to 'rows' or 'cols' to also plot signals' DFT, along rows or columns
dft = (None, 'rows', 'cols')[0]
tsigs.demo(dft=dft)

#%%# How to specify `signals` ################################################
signals = [
    'am-cosine',
    ('hchirp', dict(fmin=.2)),
    ('sine:am-cosine', (dict(f=32, phi0=1), dict(amin=.3))),
]
tsigs.demo(signals, N=2048)

#%%# With `dft` ##################
tsigs.demo(signals, dft='rows')
tsigs.demo(signals, dft='cols')

#%%# Viz CWT & SSQ_CWT with different wavelets ###############################
tsigs = TestSignals(N=2048)
wavelets = [
    Wavelet(('gmw', {'beta': 60})),
    Wavelet(('gmw', {'beta': 5})),
]
tsigs.wavcomp(wavelets, signals='all')

#%%#
tsigs.wavcomp(wavelets, signals=[('#echirp', dict(fmin=.1))], N=2048)

#%%# Viz CWT vs STFT (& SSQ'd) ###############################################
# (N, beta, NW): (512, 42.5, 255); (256, 21.5, 255)
N = 2048
signals = 'all'

n_fft = N
win_len = 720
tsigs = TestSignals(N=N)
wavelet = Wavelet(('GMW', {'beta': 60}))

NW = win_len//2 - 1
window = np.abs(sig.windows.dpss(win_len, NW))
window = np.pad(window, (N - len(window))//2)
assert len(window) == N
window_name = 'DPSS'
config_str = 'NW=%s, win_len=%s, win_pad_len=%s' % (
    NW, win_len, len(window) - win_len)

# ensure `wavelet` and `window` have ~same time & frequency resolutions
print("std_w, std_t, harea\nwavelet: {:.4f}, {:.4f}, {:.8f}"
      "\nwindow:  {:.4f}, {:.4f}, {:.8f}".format(
          wavelet.std_w, wavelet.std_t, wavelet.harea,
          *window_resolution(window)))
#%%
kw = dict(wavelet=wavelet, window=window, win_len=None, n_fft=n_fft,
          window_name=window_name, config_str=config_str)
tsigs.cwt_vs_stft(N=N, signals=signals, **kw)

#%%# Noisy example ###########################################################
N = 2048
snr = -2  # in dB
signals = 'packed-poly'

tsigs = TestSignals(N=N, snr=snr)
tsigs.cwt_vs_stft(N=N, signals=signals, **kw)

#%%# Ridge extraction ########################################################
N = 512
signals = 'poly-cubic'
snr = None
n_ridges = 3
penalty = 25

tsigs = TestSignals(N=N, snr=snr)
kw = dict(N=N, signals=signals, n_ridges=n_ridges, penalty=penalty)
tsigs.ridgecomp(transform='cwt',  **kw)
tsigs.ridgecomp(transform='stft', **kw)
