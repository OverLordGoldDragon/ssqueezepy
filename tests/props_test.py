# -*- coding: utf-8 -*-
"""Test computation of various wavelet properties (time-frequency resolution,
center frequency, etc). Note that these don't grid-search but sweep one
parameter at a time while keeping others fixed.

Certain thresholds were set greater as they fail on Travis (but not on author's
machine).
"""
import pytest
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy.wavelets import Wavelet, time_resolution, freq_resolution
from ssqueezepy.wavelets import center_frequency
from ssqueezepy.visuals import scat

VIZ = 0  # set to 1 to enable various visuals and run without pytest


def test_energy_center_frequency():
    """If kind='energy' passes, long shot for 'peak' to fail, so just test
    former; would still be interesting to investigate 'peak' vs 'energy',
    esp. at scale extrema.
    """
    def _test_mu_dependence(wc0, mu0, scale0, N0, th):
        """wc ~ mu"""
        mus = np.arange(mu0 + 1, 21)
        errs = np.zeros(len(mus))

        for i, mu in enumerate(mus):
            wavelet = Wavelet(('morlet', {'mu': mu, 'dtype': 'float64'}))
            wc = center_frequency(wavelet, scale=scale0, N=N0, kind='energy')
            errs[i] = abs((wc / wc0) - (mu / mu0))

        _assert_and_viz(th, errs, mus, 'mu',
                        "Center frequency (energy), morlet")

    def _test_scale_dependence(wc0, mu0, scale0, N0, th):
        """wc ~ 1/scale

        For small `scale`, the bell is trimmed and the (energy) center frequency
        is no longer at mode/peak (both of which are also trimmed), but is
        less; we don't test these to keep code clean.
        """
        scales = 2**(np.arange(16, 53) / 8)  # [4, ..., 90.5]
        errs = np.zeros(len(scales))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, scale in enumerate(scales):
            wc = center_frequency(wavelet, scale=scale, N=N0, kind='energy')
            errs[i] = abs((wc / wc0) - (scale0 / scale))

        _assert_and_viz(th, errs, np.log2(scales), 'log2(scale)',
                        "Center frequency (energy), morlet")

    def _test_scale_dependence_high(wc0, mu0, scale0, N0, th):
        """wc ~ 1/scale

        High `scale` subject to more significant discretization error in
        frequency domain.
        """
        scales = 2**(np.arange(53, 81) / 8)  # [90.5, ..., 1024]
        errs = np.zeros(len(scales))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, scale in enumerate(scales):
            wc = center_frequency(wavelet, scale=scale, N=N0, kind='energy')
            errs[i] = abs((wc / wc0) - (scale0 / scale))

        _assert_and_viz(th, errs, np.log2(scales), 'log2(scale)',
                        "Center frequency (energy), morlet | High scales")

    def _test_N_dependence(wc0, mu0, scale0, N0, th):
        """Independent"""
        Ns = (np.array([.25, .5, 2, 4, 9]) * N0).astype('int64')
        errs = np.zeros(len(Ns))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, N in enumerate(Ns):
            wc = center_frequency(wavelet, scale=scale0, N=N, kind='energy')
            errs[i] = abs(wc - wc0)

        _assert_and_viz(th, errs, Ns, 'N',
                        "Center frequency (energy), morlet")

    mu0 = 5
    scale0 = 10
    N0 = 1024

    wavelet0 = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))
    wc0 = center_frequency(wavelet0, scale=scale0, N=N0, kind='energy')

    args = (wc0, mu0, scale0, N0)
    _test_mu_dependence(        *args, th=1e-7)
    _test_scale_dependence(     *args, th=1e-14)
    _test_scale_dependence_high(*args, th=1e-1)
    _test_N_dependence(         *args, th=1e-14)


def test_time_resolution():
    """Some thresholds are quite large per center_frequency(, kind='peak') as
    opposed to 'energy', with force_int=True, especially for large scales.
    These are per great deviations from continuous-time counterparts,
    but this is simply a reflection of discretization limitations (trimmed or
    unsmooth bell) and finite precision error.
    """
    def _test_mu_dependence(std_t_nd0, std_t_d0, mu0, scale0, N0, th):
        """Nondimensional: std_t ~ 1/mu -- Dimensional: independent"""
        mus = np.arange(mu0 + 1, 21)
        errs = np.zeros((2, len(mus)))

        for i, mu in enumerate(mus):
            wavelet = Wavelet(('morlet', {'mu': mu, 'dtype': 'float64'}))
            std_t_nd = time_resolution(wavelet, scale0, N0, nondim=True)
            std_t_d  = time_resolution(wavelet, scale0, N0, nondim=False)

            errs[0, i] = abs((std_t_nd / std_t_nd0) - (mu / mu0))
            errs[1, i] = abs(std_t_d - std_t_d0)

        _assert_and_viz(th, errs, 2*[mus], 'mu',
                        "Time resolution, morlet")

    def _test_scale_dependence(std_t_nd0, std_t_d0, mu0, scale0, N0, th):
        """Nondimensional: independent* -- Dimensional: std_t ~ scale.

        *Nondimensional breaks down for low scales (<~3), where freq-domain
        wavelet is trimmed (even beyond mode), deviating center frequency from
        continuous-time counterpart.

        Particularly large `th` per default `(min_decay, max_mult) = (1e6, 2)`
        in `time_resolution`, which reasonably limits the extended wavelet
        duration in time-domain in (paddded) CWT (but this limitation might be
        undue; I'm unsure. https://dsp.stackexchange.com/q/70810/50076).

        _nd here is just an extra division by 'peak' center_frequency.
        """
        scales = 2**(np.arange(16, 81) / 8)  # [4, ..., 1024]
        errs = np.zeros((2, len(scales)))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))
        kw = dict(wavelet=wavelet, N=N0, force_int=False)

        for i, scale in enumerate(scales):
            std_t_nd = time_resolution(**kw, scale=scale, nondim=True)
            std_t_d  = time_resolution(**kw, scale=scale, nondim=False)

            errs[0, i] = abs(std_t_nd  - std_t_nd0)
            errs[1, i] = abs((std_t_d / std_t_d0) - (scale / scale0))

        _assert_and_viz(th, errs, 2*[np.log2(scales)], 'log2(scale)',
                        "Time resolution, morlet")

    def _test_N_dependence(std_t_nd0, std_t_d0, mu0, scale0, N0, th):
        """Independent

        `th` can be 1e-14 if dropping odd-sampled case where time-domain wavelet
        has suboptimal decay: https://github.com/jonathanlilly/jLab/issues/13
        (also dropping low-sampled case)
        """
        Ns = (np.array([.1, 1/3, .5, 2, 4, 9]) * N0).astype('int64')
        errs = np.zeros((2, len(Ns)))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, N in enumerate(Ns):
            std_t_nd = time_resolution(wavelet, scale0, N, nondim=True)
            std_t_d  = time_resolution(wavelet, scale0, N, nondim=False)

            errs[0, i] = abs(std_t_nd - std_t_nd0)
            errs[1, i] = abs(std_t_d  - std_t_d0)

        _assert_and_viz(th, errs, [Ns, Ns], 'N',
                        "Time resolution, morlet")

    mu0 = 5
    scale0 = 10
    N0 = 1024

    wavelet0 = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))
    std_t_nd0 = time_resolution(wavelet0, scale0, N0, nondim=True)
    std_t_d0  = time_resolution(wavelet0, scale0, N0, nondim=False)

    args = (std_t_nd0, std_t_d0, mu0, scale0, N0)
    _test_mu_dependence(   *args, th=[1e-1, 1e-6])
    _test_scale_dependence(*args, th=[2e-0, 1e-1])
    _test_N_dependence(    *args, th=[1e-1, 1e-8])


def test_freq_resolution():
    def _test_mu_dependence(std_w_nd0, std_w_d0, mu0, scale0, N0, th):
        """Nondimensional: std_w ~ mu -- Dimensional: independent"""
        mus = np.arange(mu0 + 1, 21)
        errs = np.zeros((2, len(mus)))

        for i, mu in enumerate(mus):
            wavelet = Wavelet(('morlet', {'mu': mu, 'dtype': 'float64'}))
            std_w_nd = freq_resolution(wavelet, scale0, N0, nondim=True)
            std_w_d  = freq_resolution(wavelet, scale0, N0, nondim=False)

            errs[0, i] = abs((std_w_nd / std_w_nd0) - (mu0 / mu))
            errs[1, i] = abs(std_w_d - std_w_d0)

        _assert_and_viz(th, errs, 2*[mus], 'mu',
                        "Frequency resolution, morlet")

    def _test_scale_dependence(std_w_nd0, std_w_d0, mu0, scale0, N0, th):
        """Nondimensional: independent* -- Dimensional: std_w ~ 1/scale

        Particularly large `th` per nontrivial discretization finite-precision
        error for large scales, with small number of samples representing
        the non-zero region. We don't "fix" this to match continuous-time
        behavior again since it's more accurate per our CWT.
        """
        scales = 2**(np.arange(16, 81) / 8)  # [4, ..., 1024]
        errs = np.zeros((2, len(scales)))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, scale in enumerate(scales):
            std_w_nd = freq_resolution(wavelet, scale, N0, nondim=True)
            std_w_d  = freq_resolution(wavelet, scale, N0, nondim=False)

            errs[0, i] = abs(std_w_nd  - std_w_nd0)
            errs[1, i]  = abs((std_w_d / std_w_d0) - (scale0 / scale))

        _assert_and_viz(th, errs, 2*[np.log2(scales)], 'log2(scale)',
                        "Frequency resolution, morlet")

    def _test_N_dependence(std_w_nd0, std_w_d0, mu0, scale0, N0, th):
        """Independent"""
        Ns = (np.array([.25, .5, 2, 4, 9]) * N0).astype('int64')
        errs = np.zeros((2, len(Ns)))
        wavelet = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))

        for i, N in enumerate(Ns):
            std_w_nd = freq_resolution(wavelet, scale0, N, nondim=True)
            std_w_d  = freq_resolution(wavelet, scale0, N, nondim=False)

            errs[0, i] = abs(std_w_nd - std_w_nd0)
            errs[1, i] = abs(std_w_d  - std_w_d0)

        _assert_and_viz(th, errs, 2*[Ns], 'N',
                        "Frequency resolution, morlet")

    mu0 = 5
    scale0 = 10
    N0 = 1024

    wavelet0 = Wavelet(('morlet', {'mu': mu0, 'dtype': 'float64'}))
    std_w_nd0 = freq_resolution(wavelet0, scale0, N0, nondim=True)
    std_w_d0  = freq_resolution(wavelet0, scale0, N0, nondim=False)

    args = (std_w_nd0, std_w_d0, mu0, scale0, N0)
    _test_mu_dependence(   *args, th=(1e-2, 1e-6))
    _test_scale_dependence(*args, th=(6e-1, 1e-1))
    _test_N_dependence(    *args, th=(1e-2, 1e-13))


def _assert_and_viz(th, errs, params, pname, test_name, logscale=True):
    """Tuple errs for coloring"""
    def _list_and_copy(arrs):
        # copy() to ensure external arrays are unaffected
        if not isinstance(arrs, list):
            arrs = [arrs.copy()]
        else:
            ls = []
            for arr in arrs:
                ls.append(arr.copy())
            arrs = ls
        return arrs

    errs_all   = np.atleast_2d(errs)
    params_all = np.atleast_2d(params)
    th_all     = np.atleast_1d(th)

    had_error = False
    for th, errs, params in zip(th_all, errs_all, params_all):
        for err, param in zip(errs, params):
            if err > th:
                had_error = True
                break
        if had_error:
            ee, eparam, eth = err, param, th
            break

    if VIZ:
        title=f"{test_name}: abs(err) vs {pname}"
        if logscale:
            title = title.replace("abs(err)", "log10(abs(err))")
            for errs in errs_all:
                errs[errs < 1e-15] = 1e-15
                errs[:] = np.log10(errs)
            th_all = np.log10(th_all)

        colors = 'blue orange green red'.split()
        for i, (errs, params) in enumerate(zip(errs_all, params_all)):
            scat(params, errs, title=title)
            plt.axhline(th_all[i], color="tab:%s" % colors[i])
        plt.show()

        if logscale:
            th_all = 10**th_all  # undo for AssertionError

    if had_error:
        raise AssertionError("%.2e > %.1e, %s=%.2f" % (ee, eth, pname, eparam))


if __name__ == '__main__':
    if VIZ:
        test_energy_center_frequency()
        test_freq_resolution()
        test_time_resolution()
    else:
        pytest.main([__file__, "-s"])
