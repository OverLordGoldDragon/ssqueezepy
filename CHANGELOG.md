### 0.6.0 (2-18-2021): Generalized Morse Wavelets, Ridge Extraction, Testing Suite

#### FEATURES (major)
 - Added [Generalized Morse Wavelets](https://overlordgolddragon.github.io/generalized-morse-wavelets/) (`gmw`, `morsewave` in `_gmw.py`)
 - Added automatic time-frequency ridge extraction, `ridge_extraction.py`
 - Added signal testing suite, `_test_signals.py`, and [examples](https://overlordgolddragon.github.io/test-signals/)
 - Added higher-order CWT (via GMWs); `_cwt.cwt_higher_order`
 - Added `configs.ini`, used to control function defaults globally
 - Improved default `scales` to not over-represent low frequencies

#### FEATURES (other)
 - `visuals`: added `wavelet_filterbank`, `viz_cwt_higher_order`, `viz_gmw_orders` (first callable as `wavelet.viz('filterbank')`)
 - `visuals.wavelet_tf`: autopicks `scale` for `scale=None` to give a nice visual for any `wavelet`
 - `ssq_cwt` & `ssq_stft`: added arg `preserve_transform` to (see docstrings)
 - `padsignal`: 2D input support, of form `(n_signals, signal_length)` (i.e. will pad every row vector).
 - `cwt`: support for `padtype=None`
 - `maprange`: tuple of floats now supported (`help(_ssq_cwt.ssq_cwt)`)
 - `Wavelet.info()` and `@property`s of `Wavelet` revamped for generality; added `@property`s: `wc_ct`, `scalec_ct`.
 - `wavelets.center_frequency`: added `kind='peak-ct'`
 - `utils.find_max_scale` now simpler and more effective, guaranteeing complete spectral coverage for low frequencies

#### BREAKING
 - `utils.py` -> `utils/*`: `common.py`, `cwt_utils.py`, `stft_utils.py`
 - The default wavelet has been changed from `'morlet'` to `'gmw'`
 - Changed Morlet's default parameters to closely match GMW's defaults per time & frequency resolution
 - `ssq_cwt(mapkind=)` default change: `'maximal'` to `'peak'`
 - `scales` default change: implicit `preset` from `'maximal'` to `'minimal'` for low scales, `'maximal'` for high
 - `ssq_cwt` return order change: `Tx, ssq_freqs, Wx, scales, w` to `Tx, Wx, ssq_freqs, scales, w, dWx` (additionally returning `dWx`)
 - `ssq_stft` return order change: `Tx, ssq_freqs, Sx, Sfs, dSx, w` to `Tx, Sx, ssq_freqs, Sfs, w, dSx`
 - `ssqueezing` & `ssq_cwt`: renamed `mapkind` to `maprange`
 - `difftype`: `'direct'` -> `'trig'`
 - `_infer_scaletype` -> `infer_scaletype`
 - `_integrate_analytic` -> `integrate_analytic`
 - `find_max_scale` -> `find_max_scale_alt`, but `find_max_scale` is still (but a different) function

#### MISC
 - `phase_cwt`: takes `abs(w)` instead of zeroing negatives
 - `wavelet` in `icwt` and `issq_cwt` now defaults to the default wavelet
 - `cwt`: added args `order`, `average`
 - `stft` & `ssq_stft`: added `t` argument
 - `stft` default `window` increased frequency resolution
 - `visuals.imshow()`: `cmap` now defaults to `'jet'` instead of `'bone'` for `abs=True`
 - Added Examples to README, adjusted Minimal Example
 - `NOTICE.txt`: added jLab
 - `setup.py`: added short & long description, copyright, keywords

#### FIXES
 - `visuals.wavelet_heatmap`: string `scales` now functional
 - `visuals`: `w` overreached into negative frequencies for odd `N` in `wavelet_tf`, `wavelet_tf_anim`, & `wavelet_heatmap`
 - `icwt`: `padtype` now functional

#### FILE CHANGES
 - `ssqueezepy/` added: `_gmw.py`, `_test_signals.py`, `ridge_extraction.py`, `configs.ini`, `README.md`
 - `ssqueezepy/` added `utils/`, split `utils.py` into `common.py`, `cwt_utils.py`, `stft_utils.py`, `__init__.py`, & moved to `utils/`.
 - `tests/` added: `gmw_test.py`, `test_signals_test.py`, `ridge_extraction_test.py`
 - `examples/` added: `extracting_ridges.py`, `scales_selection.py`, `ridge_extract_readme/`: `README.md`, `imgs/*`
 - Created `MANIFEST.in`


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
