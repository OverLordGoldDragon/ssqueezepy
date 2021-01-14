<p align="center"><img src="https://user-images.githubusercontent.com/16495490/99882586-faa86f80-2c3a-11eb-899c-b3984e98b1c7.png" width="300"></p>


# Synchrosqueezing in Python

[![Build Status](https://travis-ci.com/OverLordGoldDragon/ssqueezepy.svg?branch=master)](https://travis-ci.com/OverLordGoldDragon/ssqueezepy)
[![Coverage Status](https://coveralls.io/repos/github/OverLordGoldDragon/ssqueezepy/badge.svg?branch=master&service=github)](https://coveralls.io/github/OverLordGoldDragon/ssqueezepy)
[![PyPI version](https://badge.fury.io/py/ssqueezepy.svg)](https://badge.fury.io/py/ssqueezepy)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7cee422639034bcebe0f10ca4b95a506)](https://www.codacy.com/gh/OverLordGoldDragon/ssqueezepy/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/ssqueezepy&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Synchrosqueezing is a powerful _reassignment method_ that focuses time-frequency representations, and allows extraction of instantaneous amplitudes and frequencies. [Friendly overview.](https://dsp.stackexchange.com/a/71399/50076)


## Features
  - Continuous Wavelet Transform (CWT), forward & inverse, and its Synchrosqueezing
  - Short-Time Fourier Transform (STFT), forward & inverse, and its Synchrosqueezing
  - Clean code with explanations and learning references
  - Wavelet visualizations

### Coming soon
  - Generalized Morse Wavelets
  
## Installation
`pip install ssqueezepy`. Or, for latest version (most likely stable): 

`pip install git+https://github.com/OverLordGoldDragon/ssqueezepy`

## Examples

### 1. Signal recovery under severe noise

![image](https://user-images.githubusercontent.com/16495490/99879090-b9f12c00-2c23-11eb-8a40-2011ce84df61.png)

### 2. Medical: EEG

<img src="https://user-images.githubusercontent.com/16495490/99880110-c88f1180-2c2a-11eb-8932-90bf3406a20d.png">

<img src="https://user-images.githubusercontent.com/16495490/104537035-9f8b6b80-5632-11eb-9fa4-444efec6c9be.png">

## Introspection

`ssqueezepy` is equipped with a visualization toolkit, useful for exploring wavelet behavior across scales and configurations. (Also see [explanations and code](https://dsp.stackexchange.com/a/72044/50076))

<p align="center">
  <img src="https://raw.githubusercontent.com/OverLordGoldDragon/ssqueezepy/master/examples/imgs/anim_tf_morlet20.gif" width="500">
</p>

<img src="https://raw.githubusercontent.com/OverLordGoldDragon/ssqueezepy/master/examples/imgs/morlet_5vs20_tf.png">
<img src="https://raw.githubusercontent.com/OverLordGoldDragon/ssqueezepy/master/examples/imgs/morlet_5vs20_hm.png">

<br>
<hr>

## Minimal example

```python
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, ssq_stft

def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
    plt.show()
    plt.imshow(np.flipud(np.abs(Tx)), aspect='auto', vmin=0, vmax=.2, cmap='jet')
    plt.show()   

#%%# Define signal ####################################    
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]
x = xo + np.sqrt(2) * np.random.randn(N)

plt.plot(xo); plt.show()
plt.plot(x); plt.show()

#%%# CWT + SSQ CWT ####################################
Twxo, _, Wxo, *_ = ssq_cwt(xo, 'morlet')
viz(xo, Twxo, Wxo)

Twx, _, Wx, *_ = ssq_cwt(x, 'morlet')
viz(x, Twx, Wx)

#%%# STFT + SSQ STFT ##################################
Tsxo, _, Sxo, *_ = ssq_stft(xo)
viz(xo, Tsxo, np.flipud(Sxo))

Tsx, _, Sx, *_ = ssq_stft(x)
viz(x, Tsx, np.flipud(Sx))
```

## Learning resources

 1. [Continuous Wavelet Transform, & vs STFT](https://ccrma.stanford.edu/~unjung/mylec/WTpart1.html)
 2. [Synchrosqueezing's phase transform, intuitively](https://dsp.stackexchange.com/a/72238/50076)
 3. [Wavelet time & frequency resolution visuals](https://dsp.stackexchange.com/a/72044/50076)
 4. [Why oscillations in SSQ of mixed sines? Separability visuals](https://dsp.stackexchange.com/a/72239/50076)
 5. [Zero-padding's effect on spectrum](https://dsp.stackexchange.com/a/70498/50076)

**DSP fundamentals**: I recommend starting with 3b1b's [Fourier Transform](https://youtu.be/spUNpyF58BY), then proceeding with [DSP Guide](https://www.dspguide.com/CH7.PDF) chapters 7-11.
The Discrete Fourier Transform lays the foundation of signal processing with real data. Deeper on DFT coefficients [here](https://dsp.stackexchange.com/a/70395/50076), also [3b1b](https://youtu.be/g8RkArhtCc4).

## References

`ssqueezepy` was originally ported from MATLAB's [Synchrosqueezing Toolbox](https://github.com/ebrevdo/synchrosqueezing), authored by E. Brevdo and G. Thakur [1]. Synchrosqueezed Wavelet Transform was introduced by I. Daubechies and S. Maes [2], which was followed-up in [3], and adapted to STFT in [4]. Many implementation details draw from [5].

  1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu. ["The Synchrosqueezing algorithm for time-varying spectral analysis: robustness properties and new paleoclimate applications"](https://arxiv.org/abs/1105.0010), Signal Processing 93:1079-1094, 2013. 
  2. I. Daubechies, S. Maes. ["A Nonlinear squeezing of the Continuous Wavelet Transform Based on Auditory Nerve Models"](https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf). 
  3. I. Daubechies, J. Lu, H.T. Wu. ["Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode Decomposition"](https://arxiv.org/pdf/0912.2437.pdf), Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
  4. G. Thakur, H.T. Wu. ["Synchrosqueezing-based Recovery of Instantaneous Frequency from Nonuniform Samples"](https://arxiv.org/abs/1006.2533), SIAM Journal on Mathematical Analysis, 43(5):2078-2095, 2011.
  5. Mallat, S. ["Wavelet Tour of Signal Processing 3rd ed"](https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf).

## License

ssqueezepy is MIT licensed, as found in the [LICENSE](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/LICENSE) file. Some source functions may be under other authorship/licenses; see [NOTICE.txt](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/NOTICE.txt).
