# CWT: ssqueezepy vs. PyWavelets

## PyWavelets

```python
# basics + customs
import numpy as np
import matplotlib.pyplot as plt
from ssutil import pywt_preproc, pre_pywt_cwt

# used by pywt.ct
import scipy.fft
fftmodule = scipy.fft
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
wavelet = 'morl'; sampling_period = 1.; method = 'fft'; axis = -1
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
# Get visuals data
i = 0  # increment to 'step through' CWT generation for various scales
fft_data, fft_wav, prod, conv, coef = cwt_step(i, **kw_common)
```
```python
# Visualize
plt.plot(np.real(fft_data)); #plt.show()
plt.plot(np.imag(fft_data));
plt.ylim(-200, 200)
plt.show()
#%%######################################
plt.plot(np.real(prod)); #plt.show()
plt.plot(np.imag(prod));
plt.ylim(-30, 30)
plt.show()
#%%######################################
plt.plot(np.real(fft_wav)); #plt.show()
plt.plot(np.imag(fft_wav)); plt.show()
#%%######################################
plt.plot(np.real(conv)); plt.show()
plt.plot(np.imag(conv)); plt.show()
#%%######################################
int_psi_scales = []
for i, scale in enumerate(scales):
    step = x[1] - x[0]
    j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
    j = j.astype(int)  # floor
    if j[-1] >= int_psi.size:
        j = np.extract(j < int_psi.size, j)
    int_psi_scales.append(int_psi[j][::-1])

_, axes = plt.subplots(10, 1, sharex=False, sharey=True, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ax.plot(int_psi_scales[16 * (i + 1) - 1])
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.01)
plt.show()
#%%######################################
int_psi_scale_lens = list(map(len, int_psi_scales))
plt.plot(int_psi_scale_lens)
```

<img src="https://user-images.githubusercontent.com/16495490/84474522-0b23e980-ac9c-11ea-9fc9-e381971d86af.png">

<hr>

<img src="https://user-images.githubusercontent.com/16495490/84475084-feec5c00-ac9c-11ea-868f-3ca1ffe78cd5.png">
