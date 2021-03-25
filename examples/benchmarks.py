# -*- coding: utf-8 -*-
import os
import numpy as np
import gc
import pandas as pd

from ssqueezepy import cwt, stft, ssq_cwt, ssq_stft, Wavelet
from ssqueezepy.utils import process_scales
from ssqueezepy.ssqueezing import _compute_associated_frequencies
from timeit import timeit as _timeit

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
            f'{num}-ssq_stft': time_ssq_stft(x, dtype, n_fft)}

#%%# Setup ###################################################################
# warmup
x = np.random.randn(1000)
for dtype in ('float32', 'float64'):
    wavelet = Wavelet(dtype='float64')
    _ = ssq_cwt(x, wavelet, cache_wavelet=False)
    _ = ssq_stft(x, dtype='float64')
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
print("// BASELINE (dtype=float64, cache_wavelet=False)")

os.environ['SSQ_PARALLEL'] = '0'
os.environ['SSQ_GPU'] = '0'
t_all['base'] = {}
for N in (N0, N1):
    x = np.random.randn(N)
    t_all['base'].update(time_all(x, dtype='float64',
                                  cache_wavelet=False, **kw))
    print_report(f"/ N={N}", t_all['base'])

#%%# Parallel + wavelet cache #################################################
print("// PARALLEL + CACHE (dtype=float32, cache_wavelet=True)")

os.environ['SSQ_PARALLEL'] = '1'
os.environ['SSQ_GPU'] = '0'
t_all['parallel'] = {}
for N in (N0, N1):
    x = np.random.randn(N)
    t_all['parallel'].update(time_all(x, dtype='float32',
                                      cache_wavelet=True, **kw))
    print_report(f"/ N={N}", t_all['parallel'])

#%%# GPU + wavelet cache #################################################
print("// GPU + CACHE (dtype=float32, cache_wavelet=True)")

os.environ['SSQ_GPU'] = '1'
t_all['gpu'] = {}
for N in (N0, N1):
    x = np.random.randn(N)
    t_all['gpu'].update(time_all(x, dtype='float32',
                                 cache_wavelet=True, **kw))
    print_report(f"/ N={N}", t_all['gpu'])

#%%
df = pd.DataFrame(t_all)
print(df)

#%%#
"""
                    base  parallel       gpu
10k
10k-cwt         0.623983  0.046184  0.003928
10k-stft          0.1509  0.066056  0.005701
10k-ssq_cwt     0.905996  0.147907  0.015036
10k-ssq_stft    0.398091  0.199419  0.056197
160k
160k-cwt       11.591294  1.252456   0.03672
160k-stft       2.320408  1.100392  0.064341
160k-ssq_cwt   17.764265  3.157575  0.092638
160k-ssq_stft   6.566033   3.53078  0.191282
"""