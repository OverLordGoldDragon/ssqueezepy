# Synchrosqueezing in Python
[Synchrosqueezing Toolbox](https://github.com/ebrevdo/synchrosqueezing) ported to Python.

`ssqueezepy`'s come out of retirement; see [Releases](https://github.com/OverLordGoldDragon/ssqueezepy/releases). I've validated all main methods; the repo's now production-ready. The rest is a matter of extending.

Significant changes to some code structure are expected until v0.6.0, but whatever's not explicitly marked as problematic will work as intended. README will also change soon.

## Features
  - Forward & inverse CWT- and STFT-based Synchrosqueezing
  - Forward & inverse discretized Continuous Wavelet Transform (CWT)
  - Forward & inverse discretized Short-Time Fourier Transform (STFT)
  - Phase CWT & STFT
  - More


## Differences w.r.t. original

 - **Renamed variables/functions**; more Pythonic & readable
 - **Removed unused arguments / variables**
 - **Improved nan/inf handling**
 - **Added examples**; original repo lacks any examples in README
 - **Indexing / var references**; MATLAB is 1-indexed, and handles object reference / assignment, and 'range' ops, differently
 - **Edited docstrings**; filled missing info, & few corrections
 - **Moved functions**; each no longer has its own file, but is grouped with other relevant functions
 - **Code style**; grouped parts of code as sub-functions for improved readability; indentation for vertical alignment; other
 - **Performance**; this repo will work faster and for less memory for some methods (to be documented)
 
 **Other**:
  - Dense instead of sparse matrices for `stft_fwd` in [stft_transforms.py](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/synchrosqueezing/stft_transforms.py), as Numpy doesn't handle latter in ops involved




