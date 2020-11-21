<p align="center"><img src="https://user-images.githubusercontent.com/16495490/99882586-faa86f80-2c3a-11eb-899c-b3984e98b1c7.png" width="300"></p>


# Synchrosqueezing in Python

Synchrosqueezed Wavelet Transform implemented in Python.

Synchrosqueezing is a powerful _reassignment method_ that focuses time-frequency representations, and allows extraction of instantaneous amplitudes and frequencies. [Friendly overview.](https://dsp.stackexchange.com/a/71399/50076)

Significant changes to some code structure are expected until v0.6.0, but whatever's not explicitly marked as problematic will work as intended.

## Features
  - Forward & inverse CWT-based Synchrosqueezing
  - Forward & inverse Continuous Wavelet Transform (CWT)
  - Clean code with explanations and learning references

### Coming soon
  - Forward & inverse Short-Time Fourier Transform (STFT)
  - STFT-based Synchrosqueezing
  - Generalized Morse Wavelets
  
## Installation
`pip install git+https://github.com/OverLordGoldDragon/ssqueezepy` or clone repository; PyPi-available after 0.5.0.

## Examples

### 1. Signal recovery under severe noise

![image](https://user-images.githubusercontent.com/16495490/99879090-b9f12c00-2c23-11eb-8a40-2011ce84df61.png)

### 2. Medical: EEG

<img src="https://user-images.githubusercontent.com/16495490/99880110-c88f1180-2c2a-11eb-8932-90bf3406a20d.png">

<img src="https://user-images.githubusercontent.com/16495490/99880131-f1170b80-2c2a-11eb-9ace-807df257ad23.png">


## Minimal example

```python
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt

def viz(x, Tx, Wx):
    plt.plot(x);  plt.show()    
    plt.imshow(np.abs(Wx), aspect='auto', cmap='jet')
    plt.show()
    plt.imshow(np.flipud(np.abs(Tx)), aspect='auto', vmin=0, vmax=.1, cmap='jet')
    plt.show()   
    
#%%# Define signal ####################################    
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * np.exp(t / 3))
x = xo + np.sqrt(4) * np.random.randn(N)

#%%# SSQ CWT + CWT ####################################
Txo, _, Wxo, scales_xo, _ = ssq_cwt(xo, 'morlet')
Wxo /= np.sqrt(scales_xo)  # L1 norm
viz(xo, Txo, Wxo)

Tx, _, Wx, scales_x, _ = ssq_cwt(x, 'morlet')
Wx /= np.sqrt(scales_x)  # L1 norm 
viz(x, Tx, Wx)
```

## References

`ssqueezepy` was originally ported from MATLAB's [Synchrosqueezing Toolbox](https://github.com/ebrevdo/synchrosqueezing), authored by E. Brevdo and G. Thakur [1]. Synchrosqueezed Wavelet Transform was introduced by I. Daubechies and S. Maes [2], which was followed-up in [3]. Many implementation details draw from [4].

  1. G. Thakur, E. Brevdo, N.-S. Fučkar, and H.-T. Wu. ["The Synchrosqueezing algorithm for time-varying spectral analysis: robustness properties and new paleoclimate applications"](https://arxiv.org/abs/1105.0010), Signal Processing 93:1079-1094, 2013. 
  2. I. Daubechies, S. Maes. ["A Nonlinear squeezing of the CWT Based on Auditory Nerve Models"](https://services.math.duke.edu/%7Eingrid/publications/DM96.pdf). 
  3. I. Daubechies, J. Lu, H.T. Wu. ["Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode Decomposition"](https://arxiv.org/pdf/0912.2437.pdf), Applied and Computational Harmonic Analysis 30(2):243-261, 2011.
  4. Mallat, S. ["Wavelet Tour of Signal Processing 3rd ed"](https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf).
