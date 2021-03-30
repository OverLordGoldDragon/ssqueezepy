# -*- coding: utf-8 -*-
if __name__ != '__main__':
    raise Exception("ran example file as non-main")

import os
import numpy as np
import gc
import pandas as pd
import scipy.signal as sig
import librosa
from pywt import cwt as pcwt
from timeit import timeit as _timeit

from ssqueezepy import cwt, stft, ssq_cwt, ssq_stft, Wavelet
from ssqueezepy.utils import process_scales, padsignal
from ssqueezepy.ssqueezing import _compute_associated_frequencies

def timeit(fn, number=10):
    return _timeit(fn, number=number) / number

#%%# Bench funcs #############################################################
def print_report(header, times):
    print(("{}\n"
           "CWT:      {:.3f} sec\n"
           "STFT:     {:.3f} sec\n"
           "SSQ_CWT:  {:.3f} sec\n"
           "SSQ_STFT: {:.3f} sec\n"
           ).format(header, *list(times.values())[-4:]))

def time_ssq_cwt(x, dtype, scales, cache_wavelet, ssq_freqs):
    wavelet = Wavelet(dtype=dtype)
    kw = dict(wavelet=wavelet, scales=scales, ssq_freqs=ssq_freqs)
    if cache_wavelet:
        for _ in range(3):  # warmup run
            _ = ssq_cwt(x, cache_wavelet=True, **kw)
            del _; gc.collect()
    return timeit(lambda: ssq_cwt(x, cache_wavelet=cache_wavelet, **kw))

def time_ssq_stft(x, dtype, n_fft):
    for _ in range(3):
        _ = ssq_stft(x, dtype=dtype, n_fft=n_fft)
        del _; gc.collect()
    return timeit(lambda: ssq_stft(x, dtype=dtype, n_fft=n_fft))

def time_cwt(x, dtype, scales, cache_wavelet):
    wavelet = Wavelet(dtype=dtype)
    if cache_wavelet:
        for _ in range(3):  # warmup run
            _ = cwt(x, wavelet, scales=scales, cache_wavelet=True)
            del _; gc.collect()
    return timeit(lambda: cwt(x, wavelet, scales=scales,
                              cache_wavelet=cache_wavelet))

def time_stft(x, dtype, n_fft):
    for _ in range(3):
        _ = stft(x, dtype=dtype, n_fft=n_fft)
        del _; gc.collect()
    return timeit(lambda: stft(x, dtype=dtype, n_fft=n_fft))

def time_all(x, dtype, scales, cache_wavelet, ssq_freqs, n_fft):
    num = str(len(x))[:-3] + 'k'
    return {num: '',
            f'{num}-cwt':      time_cwt(x, dtype, scales, cache_wavelet),
            f'{num}-stft':     time_stft(x, dtype, n_fft),
            f'{num}-ssq_cwt':  time_ssq_cwt(x, dtype, scales, cache_wavelet,
                                            ssq_freqs),
            f'{num}-ssq_stft': time_ssq_stft(x, dtype, n_fft)
            }

#%%# Setup ###################################################################
# warmup
x = np.random.randn(1000)
for dtype in ('float32', 'float64'):
    wavelet = Wavelet(dtype=dtype)
    _ = ssq_cwt(x, wavelet, cache_wavelet=False)
    _ = ssq_stft(x, dtype=dtype)
del _, wavelet

#%%# Prepare reusable parameters such that STFT & CWT output shapes match ####
N0, N1 = 10000, 160000  # selected such that CWT pad length ratios are same
n_rows = 300
n_fft = n_rows * 2 - 2

wavelet = Wavelet()
scales = process_scales('log-piecewise', N1, wavelet=wavelet)[:n_rows]
ssq_freqs = _compute_associated_frequencies(
    scales, N1, wavelet, 'log-piecewise', maprange='peak',
    was_padded=True, dt=1, transform='cwt')

kw = dict(scales=scales, ssq_freqs=ssq_freqs, n_fft=n_fft)
t_all = {}

#%%# Baseline ################################################################
print("// BASELINE (dtype=float32, cache_wavelet=True)")

os.environ['SSQ_PARALLEL'] = '0'
os.environ['SSQ_GPU'] = '0'
t_all['base'] = {}
dtype = 'float32'

for N in (N0, N1):
    x = np.random.randn(N)
    t_all['base'].update(time_all(x, dtype=dtype, cache_wavelet=True, **kw))
    print_report(f"/ N={N}", t_all['base'])

#%%# Parallel + wavelet cache ################################################
print("// PARALLEL + CACHE (dtype=float32, cache_wavelet=True)")

os.environ['SSQ_PARALLEL'] = '1'
os.environ['SSQ_GPU'] = '0'
t_all['parallel'] = {}
for N in (N0, N1):
    x = np.random.randn(N)
    t_all['parallel'].update(time_all(x, dtype='float32', cache_wavelet=True,
                                      **kw))
    print_report(f"/ N={N}", t_all['parallel'])

#%%# GPU + wavelet cache #####################################################
print("// GPU + CACHE (dtype=float32, cache_wavelet=True)")

os.environ['SSQ_GPU'] = '1'
t_all['gpu'] = {}
for N in (N0, N1):
    x = np.random.randn(N)
    t_all['gpu'].update(time_all(x, dtype='float32', cache_wavelet=True, **kw))
    print_report(f"/ N={N}", t_all['gpu'])

#%%
df = pd.DataFrame(t_all)
print(df)

#%% PyWavelets ###############################################################
for N in (N0, N1):
    x = np.random.randn(N)
    xp = padsignal(x)
    t = timeit(lambda: pcwt(xp, wavelet='cmor1.5-1.0', scales=scales,
                            method='fft'))
    print("pywt_cwt-%s:" % N, t)

#%% Scipy
for N in (N0, N1):
    x = np.random.randn(N)
    xp = padsignal(x)
    t = timeit(lambda: sig.cwt(xp, wavelet=sig.morlet,
                               widths=np.arange(4, 4 + len(scales))))
    print("scipy_cwt-%s:" % N, t)

#%%
for N in (N0, N1):
    x = np.random.randn(N)
    t = timeit(lambda: sig.stft(x, nperseg=n_fft, nfft=n_fft, noverlap=n_fft-1))
    print("scipy_stft-%s:" % N, t)

#%% Librosa
# NOTE: we bench here with float64 since float32 is slower for librosa as of 0.8.0
for N in (N0, N1):
    x = np.random.randn(N)
    t = timeit(lambda: librosa.stft(x, n_fft=n_fft, hop_length=1, dtype='float64'))
    print("librosa_stft-%s:" % N, t)

#%%#
"""
i7-7700HQ, GTX 1070
                    base  parallel       gpu
10k
10k-cwt         0.126293  0.046184  0.003928
10k-stft          0.1081  0.038459  0.005337
10k-ssq_cwt     0.372002  0.147907  0.009412
10k-ssq_stft    0.282463  0.146660  0.027790
160k
160k-cwt        2.985540  1.252456  0.036721
160k-stft       1.657803  0.418435  0.064341
160k-ssq_cwt    8.384496  3.157575  0.085638
160k-ssq_stft   4.649919  2.483205  0.159171

pywt_cwt-10000:  3.5802361100000097
pywt_cwt-160000: 12.683934910000016

scipy_cwt-10000:  0.5228888900000129
scipy_cwt-160000: 10.741505060000009

scipy_stft-10000: 0.11830254000001332
scipy_stft-160000: 1.92775223000001

librosa_stft-10000: 0.09094287000000259
librosa_stft-160000: 1.383814400000001
"""