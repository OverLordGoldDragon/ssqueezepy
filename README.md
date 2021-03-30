<p align="center"><img src="https://user-images.githubusercontent.com/16495490/99882586-faa86f80-2c3a-11eb-899c-b3984e98b1c7.png" width="300"></p>


# Synchrosqueezing in Python

[![ssqueezepy CI](https://github.com/OverLordGoldDragon/ssqueezepy/actions/workflows/conda.yml/badge.svg)](https://github.com/OverLordGoldDragon/ssqueezepy/actions/workflows/conda.yml)
[![codecov](https://codecov.io/gh/OverLordGoldDragon/ssqueezepy/branch/master/graph/badge.svg?token=8L7YPN5N19)](https://codecov.io/gh/OverLordGoldDragon/ssqueezepy)
[![PyPI version](https://badge.fury.io/py/ssqueezepy.svg)](https://badge.fury.io/py/ssqueezepy)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7cee422639034bcebe0f10ca4b95a506)](https://www.codacy.com/gh/OverLordGoldDragon/ssqueezepy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/ssqueezepy&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!-- [![Build Status](https://travis-ci.com/OverLordGoldDragon/ssqueezepy.svg?branch=master)](https://travis-ci.com/OverLordGoldDragon/ssqueezepy)
[![Coverage Status](https://coveralls.io/repos/github/OverLordGoldDragon/ssqueezepy/badge.svg?branch=master&service=github)](https://coveralls.io/github/OverLordGoldDragon/ssqueezepy) -->

Synchrosqueezing is a powerful _reassignment method_ that focuses time-frequency representations, and allows extraction of instantaneous amplitudes and frequencies. [Friendly overview.](https://dsp.stackexchange.com/a/71399/50076)


## Features
  - Continuous Wavelet Transform (CWT), forward & inverse, and its Synchrosqueezing
  - Short-Time Fourier Transform (STFT), forward & inverse, and its Synchrosqueezing
  - Wavelet visualizations and testing suite
  - Generalized Morse Wavelets
  - Ridge extraction
  - Fastest wavelet transforms in Python<sup>1</sup>, beating MATLAB

<sub>1: feel free to open Issue showing otherwise</sub>


## Installation
`pip install ssqueezepy`. Or, for latest version (most likely stable):

`pip install git+https://github.com/OverLordGoldDragon/ssqueezepy`

## GPU & CPU acceleration

Multi-threaded execution is enabled by default (disable via `os.environ['SSQ_PARALLEL'] = '0'`). GPU requires [CuPy >= 8.0.0](https://docs.cupy.dev/en/stable/install.html)
and [PyTorch >= 1.8.0](https://pytorch.org/get-started/locally/) installed (enable via `os.environ['SSQ_GPU'] = '1'`). `pyfftw` optionally supported for maximum CPU FFT speed.
See [Performance guide](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/ssqueezepy/README.md#performance-guide).

## Benchmarks

[Code](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/examples/benchmarks.py). Transforms use padding, `float32` precision (`float64` supported), and output shape
`(300, len(x))`, averaged over 10 runs. `pyfftw` not used, which'd speed 1-thread & parallel further. Benched on author's i7-7700HQ, GTX 1070.

`len(x)`-transform | 1-thread CPU | parallel | gpu | pywavelets | scipy | librosa
:----------------:|:----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:
10k-cwt       | 0.126 | 0.0462 | 0.00393 | 3.58 | 0.523 | -
10k-stft      | 0.108 | 0.0385 | 0.00534 | -    | 0.118 | 0.0909
10k-ssq_cwt   | 0.372 | 0.148  | 0.00941 | -    | -     | -
10k-ssq_stft  | 0.282 | 0.147  | 0.0278  | -    | -     | -
160k-cwt      | 2.99  | 1.25   | 0.0367  | 12.7 | 10.7  | -
160k-stft     | 1.66  | 0.418  | 0.0643  | -    | 1.93  | 1.38
160k-ssq_cwt  | 8.38  | 3.16   | 0.0856  | -    | -     | -
160k-ssq_stft | 4.65  | 2.48   | 0.159   | -    | -     | -


## Examples

### 1. Signal recovery under severe noise

![image](https://user-images.githubusercontent.com/16495490/99879090-b9f12c00-2c23-11eb-8a40-2011ce84df61.png)

### 2. Medical: EEG

<img src="https://user-images.githubusercontent.com/16495490/99880110-c88f1180-2c2a-11eb-8932-90bf3406a20d.png">

<img src="https://user-images.githubusercontent.com/16495490/104537035-9f8b6b80-5632-11eb-9fa4-444efec6c9be.png">

### 3. Testing suite: CWT vs STFT, reflect-added parallel linear chirp

<img src="https://user-images.githubusercontent.com/16495490/107452011-e7ce7880-6b61-11eb-972f-8aa5ea093dc8.png">

### 4. Ridge extraction: cubic polynom. F.M. + pure tone; noiseless & 1.69dB SNR

<img src="https://user-images.githubusercontent.com/16495490/107919540-f4e5d000-6f84-11eb-9f86-dbfd34733084.png">

[More](https://github.com/OverLordGoldDragon/ssqueezepy/tree/master/examples/ridge_extraction)

### 5. Testing suite: GMW vs Morlet, reflect-added hyperbolic chirp (extreme time-loc.)

<img src="https://user-images.githubusercontent.com/16495490/107903903-d9b69880-6f63-11eb-9478-8ead016cf6f8.png">

### 6. Higher-order GMW CWT, reflect-added parallel linear chirp, 3.06dB SNR

<img src="https://user-images.githubusercontent.com/16495490/107921072-66bf1900-6f87-11eb-9bf5-afd0a6bbbc4d.png">

[More examples](https://overlordgolddragon.github.io/test-signals/)


## Introspection

`ssqueezepy` is equipped with a visualization toolkit, useful for exploring wavelet behavior across scales and configurations. (Also see [explanations and code](https://dsp.stackexchange.com/a/72044/50076))

<p align="center">
  <img src="https://raw.githubusercontent.com/OverLordGoldDragon/ssqueezepy/master/examples/imgs/anim_tf_morlet20.gif" width="500">
</p>

<img src="https://raw.githubusercontent.com/OverLordGoldDragon/ssqueezepy/master/examples/imgs/morlet_5vs20_tf.png">
<img src="https://user-images.githubusercontent.com/16495490/107297978-e6338080-6a8d-11eb-8a11-60bfd6e4137d.png">

<hr>


## Minimal example

```python
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft

def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='jet')
    plt.show()

#%%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

plt.plot(xo); plt.show()
plt.plot(x);  plt.show()

#%%# CWT + SSQ CWT ####################################
Twxo, Wxo, *_ = ssq_cwt(xo)
viz(xo, Twxo, Wxo)

Twx, Wx, *_ = ssq_cwt(x)
viz(x, Twx, Wx)

#%%# STFT + SSQ STFT ##################################
Tsxo, Sxo, *_ = ssq_stft(xo)
viz(xo, np.flipud(Tsxo), np.flipud(Sxo))

Tsx, Sx, *_ = ssq_stft(x)
viz(x, np.flipud(Tsx), np.flipud(Sx))
```

Also see ridge extraction [README](https://github.com/OverLordGoldDragon/ssqueezepy/tree/master/examples/ridge_extraction).


## Learning resources

 1. [Continuous Wavelet Transform, & vs STFT](https://ccrma.stanford.edu/~unjung/mylec/WTpart1.html)
 2. [Synchrosqueezing's phase transform, intuitively](https://dsp.stackexchange.com/a/72238/50076)
 3. [Wavelet time & frequency resolution visuals](https://dsp.stackexchange.com/a/72044/50076)
 4. [Why oscillations in SSQ of mixed sines? Separability visuals](https://dsp.stackexchange.com/a/72239/50076)
 5. [Zero-padding's effect on spectrum](https://dsp.stackexchange.com/a/70498/50076)

**DSP fundamentals**: I recommend starting with 3b1b's [Fourier Transform](https://youtu.be/spUNpyF58BY), then proceeding with [DSP Guide](https://www.dspguide.com/CH7.PDF) chapters 7-11.
The Discrete Fourier Transform lays the foundation of signal processing with real data. Deeper on DFT coefficients [here](https://dsp.stackexchange.com/a/70395/50076), also [3b1b](https://youtu.be/g8RkArhtCc4).


## Contributors (noteworthy)

 - [David Bondesson](https://github.com/DavidBondesson): ridge extraction (`ridge_extraction.py`; `examples/`: `extracting_ridges.py`, `ridge_extraction/README.md`)


## References

`ssqueezepy` was originally ported from MATLAB's [Synchrosqueezing Toolbox](https://github.com/ebrevdo/synchrosqueezing), authored by E. Brevdo and G. Thakur [1]. Synchrosqueezed Wavelet Transform was introduced by I. Daubechies and S. Maes [2], which was followed-up in [3], and adapted to STFT in [4]. Many implementation details draw from [5]. Ridge extraction based on [6].

  1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu. ["The Synchrosqueezing algorithm for time-varying spectral analysis: robustness properties and new paleoclimate applications"](https://arxiv.org/abs/1105.0010), Signal Processing 93:1079-1094, 2013.
  2. I. Daubechies, S. Maes. ["A Nonlinear squeezing of the Continuous Wavelet Transform Based on Auditory Nerve Models"](https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf).
  3. I. Daubechies, J. Lu, H.T. Wu. ["Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode Decomposition"](https://arxiv.org/pdf/0912.2437.pdf), Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
  4. G. Thakur, H.T. Wu. ["Synchrosqueezing-based Recovery of Instantaneous Frequency from Nonuniform Samples"](https://arxiv.org/abs/1006.2533), SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
  5. Mallat, S. ["Wavelet Tour of Signal Processing 3rd ed"](https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf).
  6. D. Iatsenko, P. V. E. McClintock, A. Stefanovska. ["On the extraction of instantaneous frequencies from ridges in time-frequency representations of signals"](https://arxiv.org/pdf/1310.7276.pdf).


## License

ssqueezepy is MIT licensed, as found in the [LICENSE](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/LICENSE) file. Some source functions may be under other authorship/licenses; see [NOTICE.txt](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/NOTICE.txt).
