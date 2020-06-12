# CWT: ssqueezepy vs. PyWavelets

## PyWavelets

```python
# basics + customs
import numpy as np
import matplotlib.pyplot as plt
from ssutil import pywt_preproc, pre_pywt_cwt

# used by pywt.ct
import scipy.fft
fftmodule = scipy.fftpack
next_fast_len = fftmodule.next_fast_len
```
```python
# define signal
t = np.linspace(0, 12, 1000)
s = np.cos(t**2 + t + np.cos(t))

# preprocess signal and obtain `scales` as in ssqueezepy.synsq_cwt
OPTS = {'type': 'bump', 'mu':1}
data, scales = pre_pywt_cwt(s, OPTS)

# pywt.cwt preprocessing
wavelet = 'morl'; method = 'fft'; axis = -1
out, x, int_psi, size_scale0, precision = pywt_preproc(
    data, scales, wavelet, method, axis)
```
```python
def cwt_step(i, x, data, scales, out, int_psi, size_scale0, precision):
    """CWT coefficient computation for a single scale, returning
    intermediate parameters for inspection
    """
    #### compute scale-specific parameters ##########################
    scale = scales[i]
    step = x[1] - x[0]

    j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
    j = j.astype(int)  # floor
    if j[-1] >= int_psi.size:
        j = np.extract(j < int_psi.size, j)
    int_psi_scale = int_psi[j][::-1]

    size_scale = next_fast_len(len(data) + int_psi_scale.size - 1)
    if size_scale != size_scale0:
        # Must recompute fft_data when the padding size changes
        fft_data = fftmodule.fft(data, size_scale, axis=-1)
    size_scale0 = size_scale

    #### compute CWT coefficient and keep intermediates #############
    fft_wav = fftmodule.fft(int_psi_scale, size_scale, axis=-1)
    prod = fft_wav * fft_data
    conv = fftmodule.ifft(prod, axis=-1)
    conv = conv[..., :len(data) + int_psi_scale.size - 1]
    
    coef = - np.sqrt(scale) * np.diff(conv, axis=-1)
    return fft_data, fft_wav, prod, conv, coef

kw_common = dict(x=x, data=data, scales=scales, out=out, int_psi=int_psi,
                 size_scale0=size_scale0, precision=precision)
```
```python
"""Visualize"""
plt.plot(np.real(fft_data)); #plt.show()
plt.plot(np.imag(fft_data));
plt.ylim(-200, 200)
plt.show()
#%%############################################################################
plt.plot(np.real(prod)); #plt.show()
plt.plot(np.imag(prod));
plt.ylim(-30, 30)
plt.show()
#%%############################################################################
plt.plot(np.real(fft_wav)); #plt.show()
plt.plot(np.imag(fft_wav)); plt.show()
# the components are 180deg out of phase

# plt.hist(np.angle(np.real(fft_wav) / np.imag(fft_wav)))
## or
# f_re = np.real(fft_wav)
# f_im = np.imag(fft_wav)
# c = np.inner(f_re, np.conj(f_im)) / np.sqrt(
#     np.inner(f_re, np.conj(f_re)) * np.inner(f_im, np.conj(f_im)))
# print(np.angle(c))  # == 3.141592653589793
#%%############################################################################
plt.plot(np.real(conv));
plt.plot(np.imag(conv)); plt.show()
plt.plot(np.imag(conv)); plt.show()
plt.hist(np.imag(conv), bins=500); plt.show()
# imaginary component is ~0, and appears to be normally-distributed noise
# real component is chirp-like
#%%############################################################################
plt.plot(np.real(coef)); plt.show()
plt.plot(np.imag(coef)); plt.show()
# imaginary component has properties same as conv's
```
```python
"""Show how int_psi_scale varies from smallest to largest scales"""
#### Compute scales ######################################################
int_psi_scales = []
for scale in scales:
    step = x[1] - x[0]
    j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
    j = j.astype(int)  # floor
    if j[-1] >= int_psi.size:
        j = np.extract(j < int_psi.size, j)
    int_psi_scales.append(int_psi[j][::-1])

#### Plot scales #########################################################
_, axes = plt.subplots(10, 1, sharex=False, sharey=True, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ips = int_psi_scales[16 * (i + 1) - 1]
    ax.plot(ips)
    ax.set_xlim(0, len(ips) - 1)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.01)
plt.show()
```
```python
"""Show `scales`"""
plt.plot(scales)
#%%##################################################
"""Show `len(int_psi_scale)` for varying `scales`"""
int_psi_scale_lens = list(map(len, int_psi_scales))
plt.plot(int_psi_scale_lens)
# `int_psi_scales` is a linear function of `scales`; `scales` is exponential.
# this can be verified by plotting below
plt.plot(scales, int_psi_scale_lens)
```

<img src="https://user-images.githubusercontent.com/16495490/84498476-c57a1780-acc1-11ea-8bf5-e2c38a0f010e.png">

<hr>

<img src="https://user-images.githubusercontent.com/16495490/84475084-feec5c00-ac9c-11ea-868f-3ca1ffe78cd5.png">
