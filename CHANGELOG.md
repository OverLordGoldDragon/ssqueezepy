### 0.5.5 (1-14-2021): STFT & Synchrosqueezed STFT

#### FEATURES
 - `stft`, `istft`, `ssq_stft`, and `issq_stft` implemented and validated
 - Added to `utils.py`: `buffer`, `unbuffer`, `window_norm`, `window_resolution`, and `window_area`
 - Replaced `numba.njit` with `numba.jit(nopython=True, cache=True)`, accelerating recomputing

#### BREAKING
 - `cwt()` no longer returns `x_mean`
 - `padsignal` now only returns padded input by default; `get_params=True` for old behavior
 - Moved methods: `phase_cwt` & `phase_cwt_num` from `ssqueezing` to `_ssq_cwt`
 - _In future release_: return order of `cwt` and `stft` will be changed to have `Wx, dWx` and `Sx, dSx`, and `ssq_cwt` and `ssq_stft` to have `Tx, Wx` and `Tx, Sx`

#### MISC
 - `wavelet` positional argument in `cwt` is now a keyword argument that defaults to `'morlet'`
 - Support for `padsignal(padtype='wrap')`
 - Docstring, comment cleanups
