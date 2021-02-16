# Usage guide

See full README and `examples/` at https://github.com/OverLordGoldDragon/ssqueezepy/ . Also 
see method/object docstrings via e.g. `help(cwt)`, `help(wavelet)`.

## Selecting `scales` (CWT)

We can set below values to configure `scales`; see `examples/scales_selection.py`.

```python
# signal length
N = 2048
# your signal here
t = np.linspace(0, 1, N, endpoint=False)
x = np.cos(2*np.pi * 16 * t) + np.sin(2*np.pi * 64 * t)

# choose wavelet
wavelet = 'gmw'
# choose padding scheme for CWT (doesn't affect scales selection)
padtype = 'reflect'

# one of: 'log', 'log-piecewise', 'linear'
scaletype = 'log-piecewise'
# one of: 'minimal', 'maximal', 'naive' (not recommended)
preset = 'maximal'
# number of voices (wavelets per octave); more = more scales
nv = 32
# downsampling factor for higher scales (used only if `scaletype='log-piecewise'`)
downsample = 4
# show this many of lowest-frequency wavelets
show_last = 20
```

Sample output:

<img src="https://user-images.githubusercontent.com/16495490/108126103-ddacfc80-70c2-11eb-8c1b-c3bc37256e1f.png">
