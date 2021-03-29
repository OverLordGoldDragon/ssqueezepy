# Usage guide

See full README and `examples/` at https://github.com/OverLordGoldDragon/ssqueezepy/ . Also 
see method/object docstrings via e.g. `help(cwt)`, `help(wavelet)`. Other READMEs/references:

 - Ridge extraction: https://github.com/OverLordGoldDragon/ssqueezepy/tree/master/examples/ridge_extraction
 - Generalized Morse Wavelets: https://overlordgolddragon.github.io/generalized-morse-wavelets/
 - Testing suite: https://overlordgolddragon.github.io/test-signals/


## Performance guide

Supported modes of execution:

 1. **CPU, single thread** (`os.environ['SSQ_PARALLEL'] = '0'`)
 
 2. **CPU, multi-threaded** (`os.environ['SSQ_PARALLEL'] = '1'`, default): will use all CPU threads via `@numba.jit(parallel=True)`, and 
 `scipy.fft.fft(workers=multiprocessing.cpu_count())`
 
 3. **GPU** (`os.environ['SSQ_GPU'] = '1'`), overrides `'SSQ_PARALLEL'`. Requires:
     - [CuPy >= 8.0.0](https://docs.cupy.dev/en/stable/install.html) and [PyTorch >= 1.8.0](https://pytorch.org/get-started/locally/)
     - Compatible device (mainly NVIDIA, AMD support is experimental).

All modes (including 4 & 5 below) are [tested](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/tests/all_test.py) to yield the same output 
(within float precision). Additional considerations:

 4. **Batched execution**: supported by all forward transforms (`cwt, stft, ssq_cwt, ssq_stft`), i.e. feeding multiple independent samples at once 
 (as in machine learning), like `x.shape == (n_signals, signal_len)`. This further speeds up compute, and is more memory-efficient than transforming 
 separately in a for-loop.
 
 5. **float32 & float64**: both supported, former (default) halving memory use and nearly doubling speed with negligible loss of accuracy for most uses.
     - `cwt, ssq_cwt`: pass in `wavelet` that's a `Wavelet` with `dtype` (or e.g. `('gmw', {'dtype': 'float64'})`)
     - `stft, ssq_stft`: pass in kwarg `dtype='float32'` directly
	 
 6. **[pyFFTW](https://github.com/pyFFTW/pyFFTW)**: optional dependency that can maximize FFT speed on a CPU; see `help(ssqueezepy.FFT)`.
     - "wisdom" is saved & loaded automatically; recommended to back up `utils/wisdom32` (& `64`) before updating `ssqueezepy`
	 
 7. **Changing on the fly**: execution mode can be changed without restarting the Python kernel. The only exception is `Wavelet` objects, which stay in 
 whichever mode they were instantiated, so if e.g. setting `'SSQ_GPU' = '1'`, make a new `wavelet = Wavelet()`.
 
 8. **Changing via configs.ini**: `ssqueezepy` doesn't automatically set any env flags, the PARALLEL default comes from editable `ssqueezepy/configs.ini`.
 Env flags, however, override `configs.ini`.
 
 9. **Caching**: is used to speed up repeated computes.
     - `torch.cuda.empty_cache()` is **not** recommended; instead, to free memory, do like: `import gc; Tx = []; gc.collect()`.
     - `cwt` & `ssq_cwt` support `cache_wavelet=True` (default if `Wavelet` is passed & `vectorized=True`), which will store and (when possible) reuse 
	 computed wavelets (`help(Wavelet.Psih)`).
     - `@numba.jit(cache=True)` is utilized extensively; virtually any method will run much faster the second time.

 10. **Benchmarking tip**: when timing parts of `torch` code on `device='cuda'`, call `torch.cuda.synchronize()` before `time()` (start and end).

## Reading `ssqueezing.py` code

Recommended to read `v0.6.0` code as it is lot simpler while accomplishing same (except much slower).


## Selecting `scales` (CWT)

We can set below values to configure `scales`; see `examples/scales_selection.py`.

```python
# signal length
N = 2048
# your signal here
t = np.linspace(0, 1, N, endpoint=False)
x = np.cos(2*np.pi * 16 * t) + np.sin(2*np.pi * 64 * t)

# choose wavelet
wavelet = 'gmw'
# choose padding scheme for CWT (doesn't affect scales selection)
padtype = 'reflect'

# one of: 'log', 'log-piecewise', 'linear'
scaletype = 'log-piecewise'
# one of: 'minimal', 'maximal', 'naive' (not recommended)
preset = 'maximal'
# number of voices (wavelets per octave); more = more scales
nv = 32
# downsampling factor for higher scales (used only if `scaletype='log-piecewise'`)
downsample = 4
# show this many of lowest-frequency wavelets
show_last = 20
```

Sample output:

<img src="https://user-images.githubusercontent.com/16495490/108127210-59f40f80-70c4-11eb-838e-735c35346144.png">
