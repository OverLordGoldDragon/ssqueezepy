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
    # BREAKDOWN: data.ndim == 1.
    #     data.shape[-1] -> len(data); coef.shape[-1] -> len(coef)
    # __________________________________________________________________________
    # `int_psi`: window function; integrated wavelet `wavelet` from -Inf to `x`
    # len(int_psi) == len(x)
    #
    # `x`: points of integration, determined by `precision`:
    #     - higher: more points, smaller `step`
    #     - lower:  less points, bigger  `step`
    #
    # `j`: indexing for `int_psi`; range is nearly same throughout scales -
    #      main change is in *resolution*, or number of points between min/max
    #
    # `int_psi_scale`: scaled window function; `int_psi` at scale `scale`:
    #     - higher scale: more points, granularity
    #     - also flipped about origin (since same # of pts on both sides)
    #
    # `size_scale`: nearest power of 2 to combined lengths of `data`
    #               and `int_psi_scale` for efficient FFT computation.
    #               If changes w.r.t. previous `scale`, `fft_data` is recomputed
    #               to account for change in padding size.
    # `fft_data`: fft (discrete fourier transform) of `data`, with `n=size_scale`.
    #     n: fft length. If n < len(x), x is truncated.
    #                    If n > len(x), x is zero-padded.
    #
    # `conv`: inverse fft of (fft_wav * fft_data).
    #         Right-trimmed to combined len of `data` and `int_psi_scale`,
    #         minus 1.
    #
    # `coef`: difference between successive elements of `conv` (np.diff(conv)).
    #         `coef` is trimmed afterwards, but its contents are unchanged.
    # __________________________________________________________________________
    # Larger `scale` -> `int_psi_scale` has more points ->
    #                -> `size_scale` is larger -> `fft_data` has more points
    # __________________________________________________________________________
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
    return fft_data, fft_wav, prod, conv, coef, int_psi_scale, j, scale

kw_common = dict(x=x, data=data, scales=scales, out=out, int_psi=int_psi,
                 size_scale0=size_scale0, precision=precision)
```
```python
i = 0  # increment to 'step through' CWT generation for various scales
(fft_data, fft_wav, prod, conv, coef, int_psi_scale, j, scale
) = cwt_step(i, **kw_common)
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
def _style_axis(ax, scale_idx, ips, scales):
    ax.set_xlim(0, len(ips) - 1)

    ax.annotate("scale = %.2f" % scales[scale_idx], weight='bold', fontsize=14,
                xy=(.8, .8), xycoords='axes fraction')

    xmin, xmax = ax.get_xlim()
    ax.annotate(xmin, fontsize=12, xy=(.02, .1), xycoords='axes fraction')
    ax.annotate(xmax, fontsize=12, xy=(.95, .1), xycoords='axes fraction')
    
_, axes = plt.subplots(10, 1, sharex=False, sharey=True, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    scale_idx = 16 * (i + 1) - 1
    ips = int_psi_scales[scale_idx]
    ax.plot(ips)
    _style_axis(ax, scale_idx, ips, scales)
    ## style axes ##################################

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.01)
plt.show()
```
```python
"""Show `scales`"""
plt.plot(scales); plt.show()
#%%##################################################
"""Show `len(int_psi_scale)` for varying `scales`"""
int_psi_scale_lens = list(map(len, int_psi_scales))
plt.plot(int_psi_scale_lens); plt.show()
# `int_psi_scales` is a linear function of `scales`; `scales` is exponential.
# this can be verified by plotting below
plt.plot(scales, int_psi_scale_lens)
```

<img src="https://user-images.githubusercontent.com/16495490/86709602-27ad0a80-c02b-11ea-8224-bf60e0495a6b.png">

<hr>

```python
import numpy as np
import matplotlib.pyplot as plt
from ssutil import ssq_preproc, pre_pywt_cwt

import scipy.fftpack
import scipy.signal as sig
fftmodule = scipy.fftpack
next_fast_len = fftmodule.next_fast_len
PI = np.pi
```
```python
t = np.linspace(0, 12, 1000)
s = np.cos(t**2 + t + np.cos(t))

OPTS = {'type': 'bump', 'mu':1}
data, scales = pre_pywt_cwt(s, OPTS)
wavelet_type = 'morlet'; nv=32; dt=1

Wx, Wx_scales, psihfn, xi, k, xh = ssq_preproc(s, wavelet_type, nv, dt, OPTS)
```
```python
"""Visuals helper methods"""
def _style_axis(ax, psih, a):
    ax.set_xlim(0, max(psih.shape))
    ax.annotate("scale = %.2f" % a, weight='bold', fontsize=13,
                xy=(.45, .8), xycoords='axes fraction')
    if ax.colNum == 1:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

def viz_scales(fn, sharey=True):
    fig, axes = plt.subplots(10, 1, sharex=True, sharey=sharey, figsize=(14, 11))
    for j, ax in enumerate(axes.flat):
        a = Wx_scales[j * 8]
        psih = fn(a)
        ax.plot(psih.squeeze())
        _style_axis(ax, psih, a)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.01, wspace=.01)
    plt.show()

def viz_scales_v2(fn, ylim=None, sharey=True):
    fig, axes = plt.subplots(10, 2, sharex=True, sharey=sharey, figsize=(14, 11))

    for i in range(0, 2):
        for j in range(0, 10):
            scale_idx = min(j * 17 + (len(Wx_scales) + 17) // 2 * i,
                            len(Wx_scales) - 1)
            a = Wx_scales[scale_idx]
            psih = fn(a)

            ax = axes[j, i]
            ax.plot(psih.squeeze())
            _style_axis(ax, psih, a)

    if ylim:
        ax.set_ylim(*ylim)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.01, wspace=.01)
    plt.show()
```


<img src="https://user-images.githubusercontent.com/16495490/84475084-feec5c00-ac9c-11ea-868f-3ca1ffe78cd5.png">

<hr>

<img src="https://user-images.githubusercontent.com/16495490/84676532-ad62fc00-af3e-11ea-86dd-6d9131b27d56.png">
