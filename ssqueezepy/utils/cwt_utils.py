# -*- coding: utf-8 -*-
import numpy as np
from scipy import integrate
from .common import WARN, assert_is_one_of, p2up
from .backend import torch, asnumpy
from ..configs import gdefaults

pi = np.pi

__all__ = [
    'adm_ssq',
    'adm_cwt',
    'cwt_scalebounds',
    'process_scales',
    'infer_scaletype',
    'make_scales',
    'logscale_transition_idx',
    'nv_from_scales',
    'find_min_scale',
    'find_max_scale',
    'find_downsampling_scale',
    'integrate_analytic',
    'find_max_scale_alt',
    '_process_fs_and_t',
]


def adm_ssq(wavelet):
    """Calculates the synchrosqueezing admissibility constant, the term
    R_psi in Eq 15 of [1] (also see Eq 2.5 of [2]). Uses numeric intergration.

        integral(conj(wavelet(w)) / w, w=0..inf)

    # References:
        1. The Synchrosqueezing algorithm for time-varying spectral analysis:
        robustness properties and new paleoclimate applications. G. Thakur,
        E. Brevdo, N.-S. Fučkar, and H.-T. Wu.
        https://arxiv.org/abs/1105.0010

        2. Synchrosqueezed Wavelet Transforms: a Tool for Empirical Mode
        Decomposition. I. Daubechies, J. Lu, H.T. Wu.
        https://arxiv.org/pdf/0912.2437.pdf
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet).fn
    Css = integrate_analytic(lambda w: np.conj(asnumpy(wavelet(w))) / w)
    Css = Css.real if abs(Css.imag) < 1e-15 else Css
    return Css


def adm_cwt(wavelet):
    """Calculates the cwt admissibility constant as per Eq. (4.67) of [1].
    Uses numeric integration.

        integral(wavelet(w) * conj(wavelet(w)) / w, w=0..inf)

    1. Wavelet Tour of Signal Processing, 3rd ed. S. Mallat.
    https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet).fn
    Cpsi = integrate_analytic(lambda w: np.conj(asnumpy(wavelet(w))
                                                ) * asnumpy(wavelet(w)) / w)
    Cpsi = Cpsi.real if abs(Cpsi.imag) < 1e-15 else Cpsi
    return Cpsi


def cwt_scalebounds(wavelet, N, preset=None, min_cutoff=None, max_cutoff=None,
                    cutoff=None, bin_loc=None, bin_amp=None, use_padded_N=True,
                    viz=False):
    """Finds range of scales for which `wavelet` is "well-behaved", as
    determined by `preset`. Assumes `wavelet` is uni-modal (one peak in freq
    domain); may be inaccurate otherwise.

    `min_scale`: found such that freq-domain wavelet takes on `cutoff` of its max
    value on the greatest bin.
      - Lesser `cutoff` -> lesser `min_scale`, always

    `max_scale`: search determined by `preset`:
        - 'maximal': found such that freq-domain takes `bin_amp` of its max value
          on the `bin_loc`-th (non-dc) bin
          - Greater `bin_loc` or lesser `bin_amp` -> lesser `max_scale`, always

        - 'minimal': found more intricately independent of precise bin location,
          but is likely to omit first several bins entirely; see
          `help(utils.find_max_scale_alt)`.
          - Greater `min_cutoff` -> lesser `max_scale`, generally

    `viz==2` for more visuals, `viz==3` for even more.

    # Arguments:
        wavelet: `wavelets.Wavelet`
            Wavelet sampled in Fourier frequency domain. See `help(cwt)`.

        N: int
            Length of wavelet to use.

        min_cutoff, max_cutoff: float > 0 / None
            Used to find max scale with `preset='minimal'`.
            See `help(utils.find_max_scale_alt)`

        cutoff: float / None
            Used to find min scale. See `help(utils.find_min_scale)`

        preset: str['maximal', 'minimal', 'naive'] / None
            - 'maximal': yields a larger max and smaller min.
            - 'minimal': strives to keep wavelet in "well-behaved" range of std_t
            and std_w, but very high or very low frequencies' energies will be
            under-represented. Is closer to MATLAB's default `cwtfreqbounds`.
            - 'naive': returns (1, N), which is per original MATLAB Toolbox,
            but a poor choice for most wavelet options.
            - None: will use `min_cutoff, max_cutoff, cutoff` values, else
            override `min_cutoff, max_cutoff` with those of `preset='minimal'`,
            and of `cutoff` with that of `preset='maximal'`:
                (min_cutoff, max_cutoff, cutoff) = (0.6, 0.8, -.5)

        use_padded_N: bool (default True)
            Whether to use `N=p2up(N)` in computations. Typically `N == len(x)`,
            but CWT pads to next power of 2, which is the actual wavelet length
            used, which typically behaves significantly differently at scale
            extrema, thus recommended default True. Differs from passing
            `N=p2up(N)[0]` and False only for first visual if `viz`, see code.

    # Returns:
        min_scale, max_scale: float, float
            Minimum & maximum scales.
    """
    def _process_args(preset, min_cutoff, max_cutoff, cutoff, bin_loc, bin_amp):
        defaults = dict(min_cutoff=.6, max_cutoff=.8, cutoff=-.5)

        if preset is not None:
            if any((min_cutoff, max_cutoff, cutoff)):
                WARN("`preset` will override `min_cutoff, max_cutoff, cutoff`")
            elif preset == 'minimal' and any((bin_amp, bin_loc)):
                WARN("`preset='minimal'` ignores `bin_amp` & `bin_loc`")

            assert_is_one_of(preset, 'preset',
                             ('maximal', 'minimal', 'naive'))
            if preset in ('naive', 'maximal'):
                min_cutoff, max_cutoff = None, None
                if preset == 'maximal':
                    cutoff = -.5
            else:
                min_cutoff, max_cutoff, cutoff = defaults.values()
        else:
            if min_cutoff is None:
                min_cutoff = defaults['min_cutoff']
            elif min_cutoff <= 0:
                raise ValueError("`min_cutoff` must be >0 (got %s)" % min_cutoff)

            if max_cutoff is None:
                max_cutoff = defaults['max_cutoff']
            elif max_cutoff < min_cutoff:
                raise ValueError("must have `max_cutoff > min_cutoff` "
                                 "(got %s, %s)" % (max_cutoff, min_cutoff))

        bin_loc = bin_loc or (2 if preset == 'maximal' else None)
        bin_amp = bin_amp or (1 if preset == 'maximal' else None)
        cutoff  = cutoff if (cutoff is not None) else defaults['cutoff']

        return min_cutoff, max_cutoff, cutoff, bin_loc, bin_amp

    def _viz():
        _viz_cwt_scalebounds(wavelet, N=M, Nt=M, min_scale=min_scale,
                             max_scale=max_scale, cutoff=cutoff)
        if viz >= 2:
            wavelet_waveforms(wavelet, M, min_scale)
            wavelet_waveforms(wavelet, M, max_scale)
        if viz == 3:
            scales = make_scales(M, min_scale, max_scale)
            sweep_harea(wavelet, M, scales)

    min_cutoff, max_cutoff, cutoff, bin_loc, bin_amp = _process_args(
        preset, min_cutoff, max_cutoff, cutoff, bin_loc, bin_amp)

    if preset == 'naive':  # still _process_args for the NOTE
        return 1, N

    M = p2up(N)[0] if use_padded_N else N
    min_scale = find_min_scale(wavelet, cutoff=cutoff)

    if preset in ('minimal', None):
        max_scale = find_max_scale_alt(wavelet, M, min_cutoff=min_cutoff,
                                       max_cutoff=max_cutoff)
    elif preset == 'maximal':
        max_scale = find_max_scale(wavelet, M, bin_loc=bin_loc, bin_amp=bin_amp)

    if viz:
        _viz()
    return min_scale, max_scale


def _assert_positive_integer(g, name=''):
    if not (g > 0 and float(g).is_integer()):
        raise ValueError(f"'{name}' must be a positive integer (got {g})")


def process_scales(scales, N, wavelet=None, nv=None, get_params=False,
                   use_padded_N=True):
    """Makes scales if `scales` is a string, else validates the array,
    and returns relevant parameters if requested.

        - Ensures, if array,  `scales` is 1D, or 2D with last dim == 1
        - Ensures, if string, `scales` is one of ('log', 'linear')
        - If `get_params`, also returns (`scaletype`, `nv`, `na`)
           - `scaletype`: inferred from `scales` ('linear' or 'log') if array
           - `nv`, `na`: computed newly only if not already passed
    """
    def _process_args(scales, nv, wavelet):
        preset = None
        if isinstance(scales, str):
            if ':' in scales:
                scales, preset = scales.split(':')
            elif scales == 'log-piecewise':
                preset = 'maximal'

            assert_is_one_of(scales, 'scales',
                             ('log', 'log-piecewise', 'linear'))
            if nv is None:
                nv = 32
            if wavelet is None:
                raise ValueError("must set `wavelet` if `scales` isn't array")
            scaletype = scales

        elif isinstance(scales, (np.ndarray, torch.Tensor)):
            scales = asnumpy(scales)
            if scales.squeeze().ndim != 1:
                raise ValueError("`scales`, if array, must be 1D "
                                 "(got shape %s)" % str(scales.shape))
            scaletype, _nv = infer_scaletype(scales)
            if scaletype == 'log':
                if nv is not None and _nv != nv:
                    raise Exception("`nv` used in `scales` differs from "
                                    "`nv` passed (%s != %s)" % (_nv, nv))
                nv = _nv
            elif scaletype == 'log-piecewise':
                nv = _nv  # will be array
            scales = scales.reshape(-1, 1)  # ensure 2D for broadcast ops later

        else:
            raise TypeError("`scales` must be a string or Numpy array "
                            "(got %s)" % type(scales))

        if nv is not None and not isinstance(nv, np.ndarray):
            _assert_positive_integer(nv, 'nv')
            nv = int(nv)
        return scaletype, nv, preset

    scaletype, nv, preset = _process_args(scales, nv, wavelet)
    if isinstance(scales, (np.ndarray, torch.Tensor)):
        scales = scales.reshape(-1, 1)
        return (scales if not get_params else
                (scales, scaletype, len(scales), nv))

    #### Compute scales & params #############################################
    min_scale, max_scale = cwt_scalebounds(wavelet, N=N, preset=preset,
                                           use_padded_N=use_padded_N)
    scales = make_scales(N, min_scale, max_scale, nv=nv, scaletype=scaletype,
                         wavelet=wavelet)
    na = len(scales)

    return (scales if not get_params else
            (scales, scaletype, na, nv))


def infer_scaletype(scales):
    """Infer whether `scales` is linearly or exponentially distributed (if latter,
    also infers `nv`). Used internally on `scales` and `ssq_freqs`.

    Returns one of: 'linear', 'log', 'log-piecewise'
    """
    scales = asnumpy(scales).reshape(-1, 1)
    if not isinstance(scales, np.ndarray):
        raise TypeError("`scales` must be a numpy array (got %s)" % type(scales))
    elif scales.dtype not in (np.float32, np.float64):
        raise TypeError("`scales.dtype` must be np.float32 or np.float64 "
                        "(got %s)" % scales.dtype)

    th_log = 4e-15 if scales.dtype == np.float64 else 8e-7
    th_lin = th_log * 1e3  # less accurate for some reason

    if np.mean(np.abs(np.diff(np.log(scales), 2, axis=0))) < th_log:
        scaletype = 'log'
        # ceil to avoid faulty float-int roundoffs
        nv = int(np.round(1 / np.diff(np.log2(scales), axis=0)[0].squeeze()))

    elif np.mean(np.abs(np.diff(scales, 2, axis=0))) < th_lin:
        scaletype = 'linear'
        nv = None

    elif logscale_transition_idx(scales) is None:
        raise ValueError("could not infer `scaletype` from `scales`; "
                         "`scales` array must be linear or exponential. "
                         "(got diff(scales)=%s..." % np.diff(scales, axis=0)[:4])

    else:
        scaletype = 'log-piecewise'
        nv = nv_from_scales(scales)

    return scaletype, nv


def make_scales(N, min_scale=None, max_scale=None, nv=32, scaletype='log',
                wavelet=None, downsample=None):
    """Recommended to first work out `min_scale` & `max_scale` with
    `cwt_scalebounds`.

    # Arguments:
        N: int
            `len(x)` or `len(x_padded)`.

        min_scale, max_scale: float, float
            Set scale range. Obtained e.g. from `utils.cwt_scalebounds`.

        nv: int
            Number of voices (wavelets) per octave.

        scaletype: str['log', 'log-piecewise', 'linear']
            Scaling kind to make.
            `'log-piecewise'` uses `utils.find_downsampling_scale`.

        wavelet: wavelets.Wavelet
            Used only for `scaletype='log-piecewise'`.

        downsample: int
            Downsampling factor. Used only for `scaletype='log-piecewise'`.

    # Returns:
        scales: np.ndarray
    """
    if scaletype == 'log-piecewise' and wavelet is None:
        raise ValueError("must pass `wavelet` for `scaletype == 'log-piecewise'`")
    if min_scale is None and max_scale is None and wavelet is not None:
        min_scale, max_scale = cwt_scalebounds(wavelet, N, use_padded_N=True)
    else:
        min_scale = min_scale or 1
        max_scale = max_scale or N
    downsample = int(gdefaults('utils.cwt_utils.make_scales',
                               downsample=downsample))

    # number of 2^-distributed scales spanning min to max
    na = int(np.ceil(nv * np.log2(max_scale / min_scale)))
    # floor to keep freq-domain peak at or to right of Nyquist
    # min must be more precise, if need integer rounding do on max
    mn_pow = int(np.floor(nv * np.log2(min_scale)))
    mx_pow = mn_pow + na

    if scaletype == 'log':
        # TODO discretize per `logspace` instead
        scales = 2 ** (np.arange(mn_pow, mx_pow) / nv)

    elif scaletype == 'log-piecewise':
        scales = 2 ** (np.arange(mn_pow, mx_pow) / nv)
        idx = find_downsampling_scale(wavelet, scales)
        if idx is not None:
            # `+downsample - 1` starts `scales2` as continuing from `scales1`
            # at `scales2`'s sampling rate; rest of ops are based on this design,
            # such as `/nv` in ssq, which divides `scales2[0]` by `nv`, but if
            # `scales2[0]` is one sample away from `scales1[-1]`, seems incorrect
            scales1 = scales[:idx]
            scales2 = scales[idx + downsample - 1::downsample]
            scales = np.hstack([scales1, scales2])

    elif scaletype == 'linear':
        # TODO poor scheme (but there may not be any good one)
        min_scale, max_scale = 2**(mn_pow/nv), 2**(mx_pow/nv)
        na = int(np.ceil(max_scale / min_scale))
        scales = np.linspace(min_scale, max_scale, na)

    else:
        raise ValueError("`scaletype` must be 'log' or 'linear'; "
                         "got: %s" % scaletype)
    scales = scales.reshape(-1, 1)  # ensure 2D for broadcast ops later
    return scales


def logscale_transition_idx(scales):
    """Returns `idx` that splits `scales` as `[scales[:idx], scales[idx:]]`.
    """
    scales = asnumpy(scales)
    scales_diff2 = np.abs(np.diff(np.log(scales), 2, axis=0))
    idx = np.argmax(scales_diff2) + 2
    diff2_max = scales_diff2.max()
    # every other value must be zero, assert it is so
    scales_diff2[idx - 2] = 0

    th = 1e-14 if scales.dtype == np.float64 else 1e-6

    if not np.any(diff2_max > 100*np.abs(scales_diff2).mean()):
        # everything's zero, i.e. no transition detected
        return None
    elif not np.all(np.abs(scales_diff2) < th):
        # other nonzero diffs found, more than one transition point
        return None
    else:
        return idx


def nv_from_scales(scales):
    """Infers `nv` from `scales` assuming `2**` scales; returns array
    of length `len(scales)` if `scaletype = 'log-piecewise'`.
    """
    scales = asnumpy(scales)
    logdiffs = 1 / np.diff(np.log2(scales), axis=0)
    nv = np.vstack([logdiffs[:1], logdiffs])

    idx = logscale_transition_idx(scales)
    if idx is not None:
        nv_transition_idx = np.argmax(np.abs(np.diff(nv, axis=0))) + 1
        assert nv_transition_idx == idx, "%s != %s" % (nv_transition_idx, idx)
    return nv


def find_min_scale(wavelet, cutoff=1):
    """Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that its last (largest) positive value evaluates
    `wavelet` to `cutoff * max(psih)`. If cutoff > 0, it lands to right of peak,
    else to left (i.e. peak excluded).
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    w_peak, peak = find_maximum(wavelet.fn)
    if cutoff > 0:
        # search to right of peak
        step_start, step_limit = w_peak, 10*w_peak
    else:
        # search to left of peak
        step_start, step_limit = 0, w_peak

    w_cutoff, _ = find_first_occurrence(wavelet.fn, value=abs(cutoff) * peak,
                                        step_start=step_start,
                                        step_limit=step_limit)
    min_scale = w_cutoff / pi
    return min_scale


def find_max_scale(wavelet, N, bin_loc=1, bin_amp=1):
    """Finds `scale` such that freq-domain wavelet's amplitude is `bin_amp`
    of maximum at `bin_loc` bin. Set `bin_loc=1` to ensure no lower frequencies
    are lost, but likewise mind redundancy (see `make_scales`).
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    # get scale at which full freq-domain wavelet is likely to fit
    wc_ct = center_frequency(wavelet, kind='peak-ct', N=N)
    scalec_ct = (4/pi) * wc_ct

    # get freq_domain wavelet, positive half (asm. analytic)
    psih = asnumpy(wavelet(scale=scalec_ct, N=N)[:N//2 + 1])
    # get (radian) frequencies at which it was sampled
    xi = asnumpy(wavelet.xifn(scalec_ct, N))
    # get index of psih's peak
    midx = np.argmax(psih)
    # get index where `psih` attains `bin1_amp` of its max value, to left of peak
    w_bin = xi[np.where(psih[:midx] < psih.max()*bin_amp)[0][-1]]

    # find scale such that wavelet amplitude is `bin_amp` of max at `bin_loc` bin
    max_scale = scalec_ct * (w_bin / xi[bin_loc])
    return max_scale


def find_downsampling_scale(wavelet, scales, span=5, tol=3, method='sum',
                            nonzero_th=.02, nonzero_tol=4., N=None, viz=False,
                            viz_last=False):
    """Find `scale` past which freq-domain wavelets are "excessively redundant",
    redundancy determined by `span, tol, method, nonzero_th, nonzero_tol`.

    # Arguments
        wavelet: np.ndarray / wavelets.Wavelet
            CWT wavelet.

        scales: np.ndarray
            CWT scales.

        span: int
            Number of wavelets to cross-correlate at each comparison.

        tol: int
            Tolerance value, works with `method`.

        method: str['any', 'all', 'sum']
            Condition relating `span` and `tol` to determine whether wavelets
            are packed "too densely" at a given cross-correlation, relative
            to "joint peak".

                'any': at least one of wavelet peaks lie `tol` or more bins away
                'all': all wavelet peaks lie `tol` or more bins away
                'sum': sum(distances between wavelet peaks and joint peak) > `tol`

        nonzero_th: float
            Wavelet points as a fraction of respective maxima to consider
            nonzero (i.e. `np.where(psih > psih.max()*nonzero_th)`).

        nonzero_tol: float
            Average number of nonzero points in a `span` group of wavelets above
            which testing is exempted. (e.g. if 5 wavelets have 25 nonzero points,
            average is 5, so if `nonzero_tol=4`, the `scale` is skipped/passed).

        N: int / None
            Length of wavelet to use. Defaults to 2048, which generalizes well
            along other defaults, since those params (`span`, `tol`, etc) would
            need to be scaled alongside `N`.

        viz: bool (default False)
            Visualize every test for debug purposes.

        viz_last: bool (default True)
            Visualize the failing scale (recommended if trying by hand);
            ignored if `viz=True`.
    """
    def check_group(psihs_peaks, joint_peak, method, tol):
        too_dense = False
        distances = np.abs(psihs_peaks[1] - joint_peak)

        if method == 'any':
            dist_max = distances.max()
            if dist_max < tol:
                too_dense = True
        elif method == 'all':
            dist_satisfied = (distances > tol)
            if not np.all(dist_satisfied):
                too_dense = True
        elif method == 'sum':
            dist_sum = distances.sum()
            if dist_sum < tol:
                too_dense = True
        return too_dense

    def _viz(psihs, psihs_peaks, joint_peak, psihs_nonzeros, i):
        max_nonzero_idx = np.where(psihs_nonzeros)[1].max()

        plot(psihs.T[:max_nonzero_idx + 3], color='tab:blue',
             vlines=(joint_peak, {'color': 'tab:red'}))
        scat(psihs_peaks[1], psihs[psihs_peaks].T, color='tab:red', show=1)

        distances = np.abs(psihs_peaks[1] - joint_peak)
        print("(idx, peak distances from joint peak, joint peak) = "
              "({}, {}, {})".format(i, distances, joint_peak), flush=True)

    assert_is_one_of(method, 'method', ('any', 'all', 'sum'))
    if not isinstance(wavelet, np.ndarray):
        wavelet = Wavelet._init_if_not_isinstance(wavelet)

    N = N or 2048
    Psih = (wavelet if isinstance(wavelet, (np.ndarray, torch.Tensor)) else
            wavelet(scale=scales, N=N))
    Psih = asnumpy(Psih)

    if len(Psih) != len(scales):
        raise ValueError("len(Psih) != len(scales) "
                         "(%s != %s)" % (len(Psih), len(scales)))

    # analytic, drop right half (all zeros)
    Psih = Psih[:, :Psih.shape[1]//2]
    n_scales = len(Psih)
    n_groups = n_scales - span - 1
    psihs_peaks = None

    for i in range(n_groups):
        psihs = Psih[i:i + span]

        psihs_nonzeros = (psihs > nonzero_th*psihs.max(axis=1)[:, None])
        avg_nonzeros = psihs_nonzeros.sum() / span
        if avg_nonzeros > nonzero_tol:
            continue

        psihs_peaks = np.where(psihs == psihs.max(axis=1)[:, None])
        joint_peak = np.argmax(np.prod(psihs, 0))  # mutually cross-correlate

        too_dense = check_group(psihs_peaks, joint_peak, method, tol)
        if too_dense:
            break

        if viz:
            _viz(psihs, psihs_peaks, joint_peak, psihs_nonzeros, i)

    if (viz or viz_last) and psihs_peaks is not None:
        print(("Failing scale: (idx, scale) = ({}, {:.2f})\n"
               "out of max:    (idx, scale) = ({}, {:.2f})"
               ).format(i, float(scales[i]), len(scales) - 1, float(scales[-1])))
        _viz(psihs, psihs_peaks, joint_peak, psihs_nonzeros, i)

    return i if (i < n_groups - 1) else None


def integrate_analytic(int_fn, nowarn=False):
    """Assumes function that's zero for negative inputs (e.g. analytic wavelet),
    decays toward right, and is unimodal: int_fn(t<0)=0, int_fn(t->inf)->0.
    Integrates using trapezoidal rule, from 0 to inf (equivalently).

    Integrates near zero separately in log space (useful for e.g. 1/x).
    """
    def _est_arr(mxlim, N):
        t = np.linspace(mxlim, .1, N, endpoint=False)[::-1].copy()
        arr = int_fn(t)

        max_idx = np.argmax(arr)
        min_neglect_idx = _min_neglect_idx(np.abs(arr[max_idx:]),
                                           th=1e-15) + max_idx
        return arr, t, min_neglect_idx

    def _find_convergent_array():
        mxlims = [1, 20, 80, 160]
        for m, mxlim in zip([1, 1, 4, 8], mxlims):
            arr, t, min_neglect_idx = _est_arr(mxlim, N=10000*m)
            # ensure sufficient decay between peak and right endpoint, and
            # that `arr` isn't a flatline (contains wavelet peak)
            if ((len(t) - min_neglect_idx > 1000 * m) and
                np.sum(np.abs(arr)) > 1e-5):
                break
        else:
            if int_nz < 1e-5:
                raise Exception("Could not find converging or non-negligibly"
                                "-valued bounds of integration for `int_fn`")
            elif not nowarn:
                WARN("Integrated only from 1e-15 to 0.1 in logspace")
        return arr[:min_neglect_idx], t[:min_neglect_idx]

    def _integrate_near_zero():
        # sample `intfn` more finely as it might be extremely narrow near zero.
        # this still doesn't work well as float64 zeros the numerator before /w,
        # but the true integral will be negligibly small most of the time anyway
        # (.001 to .1 may not be negligible, however; better captured by logspace)
        t = np.logspace(-15, -1, 1000)
        arr = int_fn(t)
        return integrate.trapezoid(arr, t)

    int_nz = _integrate_near_zero()
    arr, t = _find_convergent_array()
    return integrate.trapezoid(arr, t) + int_nz


def find_max_scale_alt(wavelet, N, min_cutoff=.1, max_cutoff=.8):
    """
    Design the wavelet in frequency domain. `scale` is found to yield
    `scale * xi(scale=1)` such that two of its consecutive values land
    symmetrically about the peak of `psih` (i.e. none *at* peak), while
    still yielding `wavelet(w)` to fall between `min_cutoff`* and `max_cutoff`*
    `max(psih)`. `scale` is selected such that the symmetry is attained
    using smallest possible bins (closest to dc). Steps:

        1. Find `w` (input value to `wavelet`) for which `wavelet` is maximized
        (i.e. peak of `psih`).
        2. Find two `w` such that `wavelet` attains `min_cutoff` and `max_cutoff`
        times its maximum value, using `w` in previous step as upper bound.
        3. Find `div_size` such that `xi` lands at both points of symmetry;
        `div_size` == increment between successive values of
        `xi = scale * xi(scale=1)`.
            - `xi` begins at zero; along the cutoff bounds, and us selecting
            the smallest number of divisions/increments to reach points of
            symmetry, we guarantee a unique `scale`.

    This yields a max `scale` that'll generally lie in 'nicely-behaved' region
    of std_t; value can be used to fine-tune further.
    See `visuals.sweep_std_t`.
    """
    if max_cutoff <= 0 or min_cutoff <= 0:
        raise ValueError("`max_cutoff` and `min_cutoff` must be positive "
                         "(got %s, %s)" % (max_cutoff, min_cutoff))
    elif max_cutoff <= min_cutoff:
        raise ValueError("must have `max_cutoff > min_cutoff` "
                         "(got %s, %s)" % (max_cutoff, min_cutoff))

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    w_peak, peak = find_maximum(wavelet.fn)

    # we solve the inverse problem; instead of looking for spacing of xi
    # that'd land symmetrically about psih's peak, we pick such points
    # above a set ratio of peak's value and ensure they divide the line
    # from left symmetry point to zero an integer number of times

    # define all points of wavelet from cutoff to peak, left half
    w_cutoff, _ = find_first_occurrence(wavelet.fn, value=min_cutoff * peak,
                                        step_start=0, step_limit=w_peak)

    w_ltp = np.arange(w_cutoff, w_peak, step=1/N)  # left-to-peak

    # consider every point on wavelet(w_ltp) (except peak) as candidate cutoff
    # point, and pick earliest one that yields integer number of increments
    # from left point of symmetry to origin
    div_size = (w_peak - w_ltp[:-1]) * 2  # doubled so peak is skipped
    n_divs = w_ltp[:-1] / div_size
    # diff of modulus; first drop in n_divs is like [.98, .99, 0, .01], so at 0
    # we've hit an integer, and n_divs grows ~linearly so behavior guaranteed
    # -.8 arbitrary to be ~1 but <1
    try:
        idx = np.where(np.diff(n_divs % 1) < -.8)[0][0]
    except:
        raise Exception("Failed to find suffciently-integer xi divisions; try "
                        "widening (min_cutoff, max_cutoff)")
    # the div to base the scale on (angular bin spacing of scale*xi)
    div_scale = div_size[idx + 1]

    # div size of scale=1 (spacing between angular bins at scale=1)
    w_1div = pi / (N / 2)

    max_scale = div_scale / w_1div
    return max_scale


def _process_fs_and_t(fs, t, N):
    """Ensures `t` is uniformly-spaced and of same length as `x` (==N)
    and returns `fs` and `dt` based on it, or from defaults if `t` is None.
    """
    if fs is not None and t is not None:
        WARN("`t` will override `fs` (both were passed)")
    if t is not None:
        if len(t) != N:
            # not explicitly used anywhere but ensures wrong `t` wasn't supplied
            raise Exception("`t` must be of same length as `x` "
                            "(%s != %s)" % (len(t), N))
        elif not np.mean(np.abs(np.diff(t, 2, axis=0))) < 1e-7:  # float32 thr.
            raise Exception("Time vector `t` must be uniformly sampled.")
        fs = 1 / (t[1] - t[0])
    else:
        if fs is None:
            fs = 1
        elif fs <= 0:
            raise ValueError("`fs` must be > 0")
    dt = 1 / fs
    return dt, fs, t


#############################################################################
from ..algos import _min_neglect_idx, find_maximum, find_first_occurrence
from ..wavelets import Wavelet, center_frequency
from ..visuals import plot, scat, _viz_cwt_scalebounds, wavelet_waveforms
from ..visuals import sweep_harea
