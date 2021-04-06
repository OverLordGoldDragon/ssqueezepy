### 0.6.1 (3-24-2021): GPU & CPU acceleration; caching

#### FEATURES
 - GPU acceleration & multi-thread CPU support for all forward transforms (`cwt, stft, ssq_cwt, ssq_stft`); see [Performance guide](https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/ssqueezepy/README.md#performance-guide)
 - `ssqueezepy.FFT`, supporting single- & multi-threaded CPU execution, and GPU execution, optionally via `pyfftw`
 - `dtype='float32'` and `'float64'` support for `cwt, stft, ssq_cwt, ssq_stft, Wavelet`
 - `Wavelet.Psih(scale=, N=)` will store the computed wavelet(s) and, if subsequent calls have identical `scale` and `N`, will return it directly without recomputing (significant speedup).
 
#### BREAKING
 - Dependency added: `ftz` 
 - Default `downsample`: 3 -> 4 in `utils.cwt_utils.make_scales`
 - `EPS` deprecated in favor of `EPS32` & `EPS64` for respective precisions
 - `ssq_cwt(flipud=True)` default now returns `Tx = np.flipud(Tx)` relative to previous versions
 - `ssq_cwt` & `ssq_stft` number of variables returned now depend on `get_w, get_dWx` parameters; see docstrings
 - `dtype` defaults to `'float32'` (can change via `configs.ini`); neither `cwt` nor `stft`, for most applications, require extreme precision like filters do, so defaults should prioritize compute
 - `TestSignals.make_signals()` now returns list of signals by default instead of dict with meta info (now accessible via `get_params=True`)

#### FIXES
 - `ssq_stft` would still default `n_fft = len(x)`; defaulter line removed, delegated to `stft`.
 - `ssqueezing`: improperly handled return of `infer_scaletype`
 - `ssqueezing`: `_get_center_frequency` computed at `N` instead of `p2up(N)` with `padtype != None`

#### MISC
 - `configs.ini`: added new configurable defaults
 - `ssqueezing.ssqueeze`: added `padtype` arg (see FIXES)
 - `ssqueezing.ssqueeze` & `ssq_cwt`: added `find_closest_parallel` arg (see its docstring)
 - `utils.cwt_utils`: `find_downsampling_scale` added argument `N`
 - `visuals.imshow`: removed default `'interpolation' = 'none'`
 - Added `# Arguments:` docstring to `Wavelet`
 - `cwt` significantly sped up: 1) per `Wavelet` reuse; 2) rid of `ifftshift` and `*pn` (they undo each other); 3) eliminated redundant allocation in `vectorized`


#### NOTES
 - Undocumented changes; skimming docstrings / source code should suffice for most purposes


### 0.6.0 (2-19-2021): Generalized Morse Wavelets, Ridge Extraction, Testing Suite

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
