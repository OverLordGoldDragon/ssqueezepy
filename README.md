# Synchrosqueezing in Python
Synchrosqueezing Toolbox ported to Python, authored by Eugene Brevdo, Gaurav Thakur. Original: https://github.com/ebrevdo/synchrosqueezing

**Reviewers needed**; the repository is in a development stage - details below.

## Features
  - Forward & inverse CWT- and STFT-based Synchrosqueezing
  - Forward & inverse discretized Continuous Wavelet Transform (CWT)
  - Forward & inverse discretized Short-Time Fourier Transform (STFT)
  - Phase CWT & STFT

## Reviewers needed
An eventual goal is a merged Pull Request to PyWavelets ([relevant Issue](https://github.com/PyWavelets/pywt/issues/258)). Points of review include:
  1. **Correctness**; plots generated from CWT transforms resemble those in the publications, but not entirely
  2. **Completeness**; parts of code in the original Github are missing or incorrect, and I'm unable to replace them all
  3. **Unit tests**; I'm not familiar with Synchrosqueezing itself, so I don't know how to validate its various functionalities
  4. **Licensing**; unsure how to proceed here; [original's](https://github.com/ebrevdo/synchrosqueezing/blob/master/LICENSE) says to "Redistributions in binary form must reproduce the above copyright notice" - but I'm not "redistributing" it, I'm distributing my rewriting of it
  5. <s>**Code style**</s>; I'm aware PyWavelets conforms with PEP8 (but I don't), so I'll edit PR code accordingly
  
## Review To-do:

**Correctness**:
  - [ ] 1. Example 1
  - [ ] 2. Example 2

**Completeness**:
  - [ ] 1. `freqband` in `synsq_cwt_inv` and `synsq_stft_inf` is defaulted to an integer, but indexed into as a 2D array; the two have nearly identical docstrings, and reference the same equation, but the equation appears completely irrelevant to both.
  - [ ] 2. `quadgk` has been ported as quadpy's [`quad`](https://github.com/nschloe/quadpy/blob/master/quadpy/line_segment/_tools.py#L16) (linked its wrapped function), which does not support infinite integration bounds, and has [trouble](https://github.com/nschloe/quadpy/issues/236) with computing `synsq_stft_inv`'s integral. Needs a workaround.
  - [ ] 3. As seen in examples, the y-axis shows "scales", not frequencies - and the relation between the two is neither clear nor linear; it also isn't linear w.r.t. `len(t)`, `nv`, or `fs`. Publications show frequencies instead.
 
**Unit tests**: Whatever suffices for PyWavelets will suffice for me

## Implementation To-do:
  One checkmark = code written; two = reviewed

| Status | Toolbox name | Repository name | Repository file |
| --- | --- | --- | --- |
| [ ] [**x**] | [`synsq_cwt_fw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_cwt_fw.m) | `synsq_cwt_fwd` | [synsq_cwt.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/synsq_cwt.py) |
| [ ] [**x**] | [`synsq_cwt_iw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_cwt_iw.m) | `synsq_cwt_inv` | [synsq_cwt.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/synsq_cwt.py) |
| [ ] [**x**] | [`synsq_stft_fw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_stft_fw.m) | `synsq_stft_fwd` | [synsq_stft.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/synsq_stft.py) |
| [ ] [**x**] | [`synsq_stft_iw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_stft_iw.m) | `synsq_stft_inv` | [synsq_stft.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/synsq_stft.py) |
| [ ] [**x**] | [`synsq_squeeze`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_squeeze.m) | `synsq_squeeze` | [wavelet_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/wavelet_transforms.py) |
| [ ] [**x**] | [`synsq_cwt_squeeze`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_cwt_squeeze.m) | `synsq_cwt_squeeze` | [wavelet_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/wavelet_transforms.py) |
| [ ] [**x**] | [`phase_cwt`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/phase_cwt.m) | `phase_cwt` | [wavelet_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/wavelet_transforms.py) |
| [ ] [**x**] | [`phase_cwt_num`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/phase_cwt_num.m) | `phase_cwt_num` | [wavelet_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/wavelet_transforms.py) |
| [ ] [**x**] | [`cwt_fw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/cwt_fw.m) | `cwt_fwd` | [wavelet_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/wavelet_transforms.py) |
| [ ] [ ]     | [`cwt_iw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/cwt_iw.m) |
| [ ] [**x**] | [`stft_fw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/stft_fw.m) | `stft_fwd` | [stft_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/stft_transforms.py) |
| [ ] [**x**] | [`stft_iw`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/stft_iw.m) | `stft_inv` | [stft_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/stft_transforms.py) |
| [ ] [**x**] | [`phase_stft`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/phase_stft.m) | `phase_stft` | [stft_transforms.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/stft_transforms.py) |
| [ ] [**x**] | [`padsignal`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/padsignal.m) | `padsignal` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |
| [ ] [**x**] | [`wfiltfn`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/wfiltfn.m) | `wfiltfn` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |
| [ ] [ ] | [`wfilth`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/wfilth.m) |
| [ ] [**x**] | [`synsq_adm`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/synsq_adm.m) | `synsq_adm` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |
| [ ] [**x**] | [`buffer`](https://www.mathworks.com/help/signal/ref/buffer.html) | `buffer` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |
| [**x**] [**x**] | [`est_riskshrink_thresh`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/est_riskshrink_thresh.m) | `est_riskshrink_thresh` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |
| [**x**] [**x**] | [`p2up`](https://github.com/ebrevdo/synchrosqueezing/blob/master/synchrosqueezing/p2up.m) | `p2up` | [utils.py](https://github.com/OverLordGoldDragon/synchrosqueezing_python/blob/master/synchrosqueezing/utils.py) |

There are more unlisted (see original repo), but not all will be implemented, in particular GUI implementations.

## Differences w.r.t. original

 - **Renamed variables/functions**; more Pythonic & readable
 - **Removed unused arguments / variables**
 - **Improved nan/inf handling**
 - **Added examples**; original repo lacks any examples in README
 - **Indexing / var references**; MATLAB is 1-indexed, and handles object reference / assignment, and 'range' ops, differently
 - **Edited docstrings**; filled missing info, & few corrections
 - **Moved functions**; each no longer has its own file, but is grouped with other relevant functions
 - **Code style**; grouped parts of code as sub-functions for improved readability; indentation for vertical alignment; other
 - **Performance**; this repo may work faster or slower, as Numpy arrays are faster than C arrays, but some of original funcs use MEX-optimized code with no Numpy equivalent. Also using dense instead of sparse matrices (see below).
 - **Performance**; this repo _will_ work **10x+ faster** for some of the methods which were vectorized out of for-loops
 
 **Other**:
  - Dense instead of sparse matrices for `stft_fwd` in [stft_transforms.py](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/synchrosqueezing/stft_transforms.py), as Numpy doesn't handle latter in ops involved



## Examples

See [examples.py](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/examples.py). Links to: [paper [1]](https://sci-hub.se/https://doi.org/10.1016/j.sigpro.2012.11.029), [paper[2]](https://arxiv.org/pdf/0912.2437.pdf). `_inv` methods (reconstruction, inversion) have been omitted as they involve `freqband`.

**EX 1:** Paper [1], pg. 1086

Only real components shown; imaginary are nearly identical, sometimes sign-opposite.

<img src="https://user-images.githubusercontent.com/16495490/73328411-176e5380-4273-11ea-8bc1-502dfd107444.png" width="600">
<img src="https://user-images.githubusercontent.com/16495490/73328308-b777ad00-4272-11ea-9b7a-89cca0042d1e.png" width="600">
<img src="https://user-images.githubusercontent.com/16495490/73328370-e9890f00-4272-11ea-9418-9ff741de66f8.png" width="600">
<img src="https://user-images.githubusercontent.com/16495490/73328515-8350bc00-4273-11ea-883b-1b253d4cd603.png" width="550">

synsq-CWT (`synsq_cwt_fwd`) appears to produce strongest agreement with paper (FIG 4), while none of STFT yield any resemblance of anything in the papers. It's also unclear whether `synsq_squeeze` was used for "synsq" in FIG 4 instead.

**EX 2:** Paper [2], pg. 18

<img src="https://user-images.githubusercontent.com/16495490/73328745-518c2500-4274-11ea-8140-489ee2403118.png" width="600">
<img src="https://user-images.githubusercontent.com/16495490/73328804-8a2bfe80-4274-11ea-8c04-6a3b29791e52.png" width="600">
<img src="https://user-images.githubusercontent.com/16495490/73328894-d8410200-4274-11ea-9731-9537ea9588bf.png" width="550">

Similar situation as EX 1; again CWT has close resemblance, and STFT is in a separate reality. The two apparent discrepancies w/ CWT are: (1) slope of the forking incline, steeper in FIG. 3; (2) position of horizontal line, lower in FIG. 3. As for the black lines in FIG 3, they seem to be the (manual) "markings" mentioned under the figure in the paper.
