# -*- coding: utf-8 -*-
"""Convenience visual methods"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft
from pathlib import Path
from .algos import find_closest, find_maximum


#### Visualizations ##########################################################
def wavelet_tf(wavelet, N=2048, scale=100, notext=False, width=1.1, height=1):
    """Visualize `wavelet` joint time-frequency resolution. Plots frequency-domain
    wavelet (psih) along y-axis, and time-domain wavelet (psi) along x-axis.

    Orthogonal units (e.g. y-axis for psi) are meaningless; function values
    aren't to scale, but *widths* are, so time-frequency uncertainties are
    accurately captured.

    `wavelet` is instance of `wavelets.Wavelet` or its valid `wavelet` argument.
    See also: https://www.desmos.com/calculator/nqowgloycy
    """
    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    #### Compute psi & psihf #################################################
    psi  = ifft(wavelet(scale * _xifn(1, N)) * (-1)**np.arange(N))
    apsi = np.abs(psi)
    t = np.arange(-N/2, N/2, step=1)

    w = aifftshift(_xifn(1, N))[N//2-1:]
    psih = wavelet(scale * w)

    #### Compute stdevs & respective indices #################################
    wc    = center_frequency(wavelet, scale, N)
    std_w = freq_resolution(wavelet, scale, N, nondim=0)
    std_t = time_resolution(wavelet, scale, N, nondim=0, min_decay=1)
    _wc = np.pi - wc

    wlix = np.argmin(np.abs(w - (_wc - std_w)))
    wrix = np.argmin(np.abs(w - (_wc + std_w)))
    wl, wr = w[wlix], w[wrix]

    tlix = np.argmin(np.abs(t - (0 - std_t)))
    trix = np.argmin(np.abs(t - (0 + std_t)))
    tl, tr = t[tlix], t[trix]

    ## Rescale psi so that its y-coords span 1/5 of psih's x-coords, & vice-versa
    frac = 5
    psig  = psi  * (w.max() / apsi.max()) / frac
    apsig = apsi * (w.max() / apsi.max()) / frac
    psihg = psih * (t.max() / psih.max()) / frac
    # additionally shift psih to psi's left
    psihg += t.min()

    ## Find intersections
    w_xminu, w_xmax = psihg[::-1][wlix], tr
    w_xmind = psihg[::-1][wrix]  # psih not necessarily symmetric
    w_ymin, w_ymax = wl, wr
    t_xmin, t_xmax = tl, tr
    t_yminl, t_ymax = apsig[tlix], wr
    t_yminr = apsig[trix]  # same for psi

    #### Plot ################################################################
    plot(t, psig, complex=1, h=1.5)
    plot(t, apsig, linestyle='--', color='k')
    plot(psihg[::-1], w, color='purple')

    # bounds lines
    lkw = dict(color='k', linewidth=1)
    plot([t_xmin,  t_xmin], [t_yminl, t_ymax], **lkw)
    plot([t_xmax,  t_xmax], [t_yminr, t_ymax], **lkw)
    plot([w_xminu, w_xmax], [w_ymin,  w_ymin], **lkw)
    plot([w_xmind, w_xmax], [w_ymax,  w_ymax], **lkw)
    plt.xlim(t.min()*1.02, t.max()*1.02)

    # radians 0 to pi from top to bottom(=psi's mean)
    ylabels = np.round(np.linspace(np.pi, 0, 7), 1)
    plt.yticks(np.linspace(0, np.pi, len(ylabels)), ylabels)

    if notext:
        plt.gcf().set_size_inches(12*width, 12*height)
        plt.show()
        return
    #### Title, annotations, labels, styling #################################
    ## Annotation: info summary
    txt = ("    wc = {:<6.5f} rad-c/s\n"
           " std_t = {:<6.4f} s/c-rad\n"
           " std_w = {:<6.5f} rad-c/s\n"
           "area/4 = {:.12f}\n"
           "       = std_t * std_w\n\n"
           "(rad-c/s=\n radians*cycles/samples)"
           ).format(wc, std_t, std_w, std_t * std_w)
    _kw = dict(xycoords='axes fraction', xy=(.7, .76), weight='bold',
               fontsize=16)
    try:
        # 'Consolas' for vertical align
        plt.annotate(txt, family='Consolas', **_kw)
    except:
        plt.annotate(txt, **_kw)  # in case platform lacks 'Consolas'

    ## Title: wavelet name & parameters
    title = wavelet._desc(N=N, scale=scale)
    plt.title(title, loc='left', weight='bold', fontsize=16)

    ## Styling
    plt.xlabel("samples", weight='bold', fontsize=15)
    plt.ylabel("radians", weight='bold', fontsize=15)

    plt.gcf().set_size_inches(12*width, 12*height)
    plt.show()


def wavelet_tf_anim(wavelet, N=2048, scales=None, width=1.1, height=1,
                    savepath='wavanim.gif', testing=False):
    """This method computes same as `wavelet_tf` but for all scales at once,
    and animates 'intelligently'. See help(wavelet_tf).

    `scales=None` will default to 'log:minimal' with (.9*min_scale,
    0.25*max_scale). These are selected to show the wavelet a little outside of
    "well-behaved" range (without slashing max_scale, it's a lot outside such
    range). May not work for every wavelet or all of their configs.
    """
    def _make_anim_scales(scales, wavelet, N):
        if scales is None:
            scales = 'log:minimal'
            mn, mx = cwt_scalebounds(wavelet, N=N, preset='maximal',
                                     double_N=False)
            scales = make_scales(N, 0.90*mn, 0.25*mx, scaletype='log')
        else:
            scales = process_scales(scales, N, wavelet, double_N=False)

        # compute early and late scales more densely as they capture more
        # interesting behavior, so animation will slow down smoothly near ends
        scales = scales.squeeze()
        na = len(scales)

        s0 = (25/253)*na  # empircally-determined good value

        srepl = max(int(s0), 1)  # scales to keep from each end
        srepr = max(int(s0), 1)
        smull = 4        # extension factor
        smulr = 3

        sright = np.linspace(scales[-srepr], scales[-1],    srepr * smulr)
        sleft  = np.linspace(scales[0],      scales[srepl], srepl * smull)
        sright = np.hstack([sright, sright[-1].repeat(smulr*2)])  # smooth loop
        sleft  = np.hstack([sleft[0].repeat(smull*2), sleft])

        scales = np.hstack([sleft, scales[srepl:-srepr], sright])
        scales  = scales.reshape(-1, 1)
        return scales

    from .utils import NOTE, process_scales, cwt_scalebounds, make_scales
    from matplotlib.animation import FuncAnimation
    import matplotlib
    matplotlib.use("Agg")
    NOTE("Switched matplotlib to 'Agg' backend for animating")

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = _make_anim_scales(scales, wavelet, N)

    #### Compute Psi & Psih ##################################################
    Psi = wavelet.psifn(scale=scales, N=N)
    aPsi = np.abs(Psi)
    t = np.arange(-N/2, N/2, step=1)

    w = aifftshift(_xifn(1, N))[N//2-1:]
    Psih = wavelet(scales * w)

    #### Compute stdevs & respective indices #################################
    Wc    = np.zeros(len(scales))
    std_W = Wc.copy()
    std_T = Wc.copy()

    for i, scale in enumerate(scales):
        Wc[i]    = center_frequency(wavelet, float(scale), N, kind='energy')
        std_W[i] = freq_resolution( wavelet, float(scale), N, nondim=0)
        std_T[i] = time_resolution( wavelet, float(scale), N, nondim=0,
                                    min_decay=1)
    _Wc = np.pi - Wc

    Wlix = find_closest((_Wc - std_W).reshape(-1, 1), w).squeeze()
    Wrix = find_closest((_Wc + std_W).reshape(-1, 1), w).squeeze()
    Wl, Wr = w[Wlix], w[Wrix]

    Tlix = find_closest(0 - std_T.reshape(-1, 1), t).squeeze()
    Trix = find_closest(0 + std_T.reshape(-1, 1), t).squeeze()
    Tl, Tr = t[Tlix], t[Trix]

    ## Rescale Psi so that its y-coords span 1/5 of Psih's x-coords, & vice-versa
    frac = 5
    Psig  = Psi  * (w.max() / aPsi.max(axis=-1)).reshape(-1, 1) / frac
    aPsig = aPsi * (w.max() / aPsi.max(axis=-1)).reshape(-1, 1) / frac
    Psihg = Psih * (t.max() / Psih.max(axis=-1)).reshape(-1, 1) / frac
    # additionally shift Psih to Psi's left
    Psihg += t.min()

    ## Find intersections ####################################################
    sidx = np.arange(len(scales))

    W_xminu, W_xmax = Psihg[:, ::-1][sidx, Wlix], Tr
    W_xmind = Psihg[:, ::-1][sidx, Wrix]  # Psih not necessarily symmetric
    W_ymin, W_ymax = Wl, Wr

    T_xmin, T_xmax = Tl, Tr
    T_yminl, T_ymax = aPsig[sidx, Tlix], Wr
    T_yminr = aPsig[sidx, Trix]  # same for Psi

    ## Set up plot objects ###################################################
    fig, ax = plt.subplots()
    ax.set_xlim([t.min()*1.02, t.max()*1.02])
    ax.set_ylim([-aPsig.max()*1.05, np.pi*1.02])

    ylabels = np.round(np.linspace(np.pi, 0, 7), 1)
    plt.yticks(np.linspace(0, np.pi, len(ylabels)), ylabels)

    fig.set_size_inches(12*width, 12*height)

    ## Title: wavelet name & parameters
    title = wavelet._desc(N=N)
    ax.set_title(title, loc='left', weight='bold', fontsize=16)

    line1, = ax.plot([], [], color='tab:blue')
    line2, = ax.plot([], [], color='tab:orange')
    line3, = ax.plot([], [], color='k', linestyle='--')
    line4, = ax.plot([], [], color='purple')

    lkw = dict(color='k', linewidth=1)
    line5, = ax.plot([], [], **lkw)
    line6, = ax.plot([], [], **lkw)
    line7, = ax.plot([], [], **lkw)
    line8, = ax.plot([], [], **lkw)

    tkw = dict(horizontalalignment='center', verticalalignment='center',
               transform=ax.transAxes, fontsize=15, weight='bold')
    txt = ax.text(.9, .95, "scale=%.2f" % scales[0], **tkw)
    fig.tight_layout()

    #### Animate #############################################################
    def unique_savepath(savepath):
        """Ensure doesn't overwrite existing"""
        sp = Path(savepath)
        savename = sp.stem

        if sp.is_file():
            paths = [str(p.stem) for p in Path(savepath).parent.iterdir()
                     if savename in p.stem]
            maxnum = 0
            for p in paths:
                num = p.replace(savename, '')
                if num != '' and int(num) > maxnum:
                    maxnum = int(num)
            sp = Path(sp.parent, savename + str(maxnum + 1) + sp.suffix)
        sp = str(sp)
        return sp

    def animate(i):
        line1.set_data(t, Psig[i].real)
        line2.set_data(t, Psig[i].imag)
        line3.set_data(t, aPsig[i])
        line4.set_data(Psihg[i][::-1], w)

        line5.set_data([T_xmin[i],  T_xmin[i]], [T_yminl[i], T_ymax[i]])
        line6.set_data([T_xmax[i],  T_xmax[i]], [T_yminr[i], T_ymax[i]])
        line7.set_data([W_xminu[i], W_xmax[i]], [W_ymin[i],  W_ymin[i]])
        line8.set_data([W_xmind[i], W_xmax[i]], [W_ymax[i],  W_ymax[i]])

        txt.set_text("scale=%.2f" % scales[i])
        return line1, line2, line3, line4, line5, line6, line7, line8

    sp = unique_savepath(savepath)
    print(("Successfully computed parameters, scales ranging {:.2f} to {:.2f}; "
           "animating...\nWill save to: {}").format(
               scales.min(), scales.max(), sp), flush=True)

    frames = np.hstack([range(len(scales)), range(len(scales) - 1)[::-1]])
    if testing:  # animation takes long; skip when unit-testing
        print("Passed `testing=True`, won't animate")
        return
    anim = FuncAnimation(fig, animate, frames=frames, interval=60,
                         blit=True, repeat=False)

    anim.save(sp, writer='imagemagick')
    print("Animated and saved to", sp, flush=True)


def wavelet_heatmap(wavelet, scales='log', N=2048):
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    if not isinstance(scales, np.ndarray):
        from .utils import process_scales
        scales = process_scales('log', N, wavelet, double_N=False)

    #### Compute time- & freq-domain wavelets for all scales #################
    Psi = wavelet.psifn(scale=scales, N=N)

    w = aifftshift(_xifn(1, N))[N//2-1:]
    Psih = wavelet(scales * w)

    #### Plot ################################################################
    mx = np.abs(Psi).max() * .01
    title0 = wavelet._desc(N=N)

    kw = dict(ylabel="scales", xlabel="samples")
    imshow(Psi.real,   norm=(-mx, mx), yticks=scales,
           title=title0 + " | Time-domain; real part", **kw)
    imshow(Psi, abs=1, norm=(0, mx), yticks=scales,
           title=title0 + " | Time-domain; abs-val", **kw)
    kw['xlabel'] = "radians"
    imshow(Psih, abs=1, cmap='jet', yticks=scales,
           xticks=np.linspace(0, np.pi, N//2),
           title=title0 + "| Freq-domain; abs-val", **kw)


def sweep_std_t(wavelet, N, scales='log', get=False, **kw):
    from .utils import process_scales

    def _process_kw(kw):
        kw = kw.copy()  # don't change external dict
        defaults = dict(min_decay=1, max_mult=2, min_mult=2,
                        nondim=False, force_int=True)
        for k, v in kw.items():
            if k not in defaults:
                raise ValueError(f"unsupported kwarg '{k}'; must be one of: "
                                 + ', '.join(defaults))

        for k, v in defaults.items():
            kw[k] = kw.get(k, v)
        return kw

    kw = _process_kw(kw)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet)

    std_ts = np.zeros(scales.size)
    for i, scale in enumerate(scales):
        std_ts[i] = time_resolution(wavelet, scale=scale, N=N, **kw)

    title = "std_t [{}] vs log2(scales) | {} wavelet, {}".format(
        "nondim" if kw['nondim'] else "s/c-rad", wavelet.name, wavelet.config_str)
    hlines = ([N/2, N/4], dict(color='k', linestyle='--'))
    plot(np.log2(scales), std_ts, title=title, hlines=hlines, show=1)

    if get:
        return std_ts


def sweep_std_w(wavelet, N, scales='log', get=False, **kw):
    from .utils import process_scales

    def _process_kw(kw):
        kw = kw.copy()  # don't change external dict
        defaults = dict(nondim=False, force_int=True)
        for k, v in kw.items():
            if k not in defaults:
                raise ValueError(f"unsupported kwarg '{k}'; must be one of: "
                                 + ', '.join(defaults))

        for k, v in defaults.items():
            kw[k] = kw.get(k, v)
        return kw

    kw = _process_kw(kw)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    scales = process_scales(scales, N, wavelet)

    std_ws = np.zeros(scales.size)
    for i, scale in enumerate(scales):
        std_ws[i] = freq_resolution(wavelet, scale=scale, N=N, **kw)

    title = "std_w [{}] vs log2(scales) | {} wavelet, {}".format(
        "nondim" if kw['nondim'] else "s/c-rad", wavelet.name, wavelet.config_str)
    plot(np.log2(scales), std_ws, title=title, show=1)

    if get:
        return std_ws


def sweep_harea(wavelet, N, scales='log', get=False, kw_w=None, kw_t=None):
    """Sub-.5 and near-0 areas will occur for very high scales as a result of
    discretization limitations. Zero-areas have one non-zero frequency-domain,
    and std_t==N/2, with latter more accurately set to infinity (which we don't).

    Sub-.5 are per freq-domain assymetries degrading time-domain decay,
    and limited bin discretization integrating unreliably (yet largely
    meaningfully; the unreliable-ness appears emergent from discretization).
    """
    from .utils import process_scales

    kw_w, kw_t = (kw_w or {}), (kw_t or {})
    scales = process_scales(scales, N, wavelet)
    wavelet = Wavelet._init_if_not_isinstance(wavelet)

    std_ws = sweep_std_w(wavelet, N, scales, get=True, **kw_w)
    plt.show()
    std_ts = sweep_std_t(wavelet, N, scales, get=True, **kw_t)
    plt.show()
    hareas = std_ws * std_ts

    hline = (.5, dict(color='tab:red', linestyle='--'))
    title = "(std_w * std_t) vs log2(scales) | {} wavelet, {}".format(
        wavelet.name, wavelet.config_str)
    plot(np.log2(scales), hareas, color='k', hlines=hline, title=title)
    plt.show()

    if get:
        return hareas, std_ws, std_ts


def wavelet_waveforms(wavelet, N, scale, zoom=True):
    from .utils import find_maximum

    wavelet = Wavelet._init_if_not_isinstance(wavelet)
    ## Freq-domain sampled #######################
    w_peak, _ = find_maximum(wavelet.fn)

    w_ct = np.linspace(0, w_peak*2, max(4096, 2*N))  # 'continuous-time'
    w_dt = np.linspace(0, np.pi, N//2) * scale  # sampling pts at `scale`
    psih_ct = wavelet(w_ct)
    psih_dt = wavelet(w_dt)

    title = ("wavelet(w) sampled by xi at scale={:.2f}, N={} | {} wavelet, {}"
             ).format(scale, N, wavelet.name, wavelet.config_str)
    plot(w_ct, psih_ct, title=title, xlabel="radians")
    scat(w_dt, psih_dt, color='tab:red')

    plt.legend(["psih at scale=1", "sampled at scale=%.2f" % scale], fontsize=13)
    plt.axvline(w_peak, color='tab:red', linestyle='--')
    plt.show()

    ## Freq-domain #######################
    # if peak not near left, don't zoom; same as `if .. (w_peak >= w_dt.max())`
    if not zoom or (np.argmax(psih_dt) > .05 * N/2):
        end = None
    else:
        peak_idx = np.argmax(psih_dt)
        end = np.where(psih_dt[peak_idx:] < 1e-4*psih_dt.max())[0][0]
        end += peak_idx + 3  # +3: give few more indices for visual

    w_dtn = w_dt * (np.pi / w_dt.max())  # norm to span true w
    plot(w_dtn[:end], psih_dt[:end], xlabel="radians",
         title="Freq-domain waveform (psih)" + ", zoomed" * (end is not None))
    scat(w_dtn[:end], psih_dt[:end], color='tab:red', show=1)

    ## Time-domain #######################
    psi = wavelet.psifn(scale=scale, N=N)
    apsi = np.abs(psi)
    t = np.arange(-N/2, N/2, step=1)

    # don't zoom unless there's fast decay
    peak_idx = np.argmax(apsi)
    if not zoom or (apsi.max() / apsi[peak_idx:].min() <= 1e3):
        start, end = 0, None
    else:
        dt = np.where(apsi[peak_idx:] < 1e-3*apsi.max())[0][0]
        start, end = (N//2 - dt, N//2 + dt + 1)

    plot(t[start:end], psi[start:end], complex=1, xlabel="samples",
         title="Time-domain waveform (psi)" + ", zoomed" * (end is not None))
    plot(t[start:end], apsi[start:end], color='k', linestyle='--', show=1)


def _viz_cwt_scalebounds(wavelet, N, min_scale=None, max_scale=None,
                         std_t=None, cutoff=1, stdevs=2, Nt=None):
    """Can be used to visualize time & freq domains separately, where
    `min_scale` refers to scale at which to show the freq-domain wavelet, and
    `max_scale` the time-domain one.
    """
    def _viz_max(wavelet, N, max_scale, std_t, stdevs, Nt):
        from .wavelets import time_resolution

        Nt = Nt or 2*N  # assume single extension
        if Nt is None:
            Nt = 2*N
        if std_t is None:
            # permissive max_mult to not crash visual
            std_t = time_resolution(wavelet, max_scale, N, nondim=False,
                                    min_mult=2, max_mult=2, min_decay=1)

        t = np.arange(-Nt/2, Nt/2, step=1)
        t -= t.mean()
        psi = wavelet.psifn(scale=max_scale, N=len(t))

        plot(t, np.abs(psi)**2, ylims=(0, None),
             title="|Time-domain wavelet|^2, extended (outside dashed)")

        plt.axvline(std_t,          color='tab:red')
        plt.axvline(std_t * stdevs, color='tab:green')
        # mark target (non-extended) frame
        _ = [plt.axvline(v, color='k', linestyle='--') for v in (-N/2, N/2-1)]

        _kw = dict(fontsize=16, xycoords='axes fraction', weight='bold')
        plt.annotate("1 stdev",
                     xy=(.88, .95), color='tab:red',   **_kw)
        plt.annotate("%s stdevs" % stdevs,
                     xy=(.88, .90), color='tab:green', **_kw)
        plt.show()

    def _viz_min(wavelet, N, min_scale, cutoff):
        w = _xifn(1, N)[:N//2 + 1]  # drop negative freqs
        psih = wavelet(min_scale * w, nohalf=True)
        _, mx = find_maximum(wavelet)

        plot(w, psih, title=("Frequency-domain wavelet, positive half "
                             "(cutoff=%s, peak=%.3f)" % (cutoff, mx)))
        plt.axhline(mx * abs(cutoff), color='tab:red')
        plt.show()

    if min_scale is not None:
        _viz_min(wavelet, N, min_scale, cutoff)
    if max_scale is not None:
        _viz_max(wavelet, N, max_scale, std_t, stdevs, Nt)
    if not (min_scale or max_scale):
        raise ValueError("Must set at least one of `min_scale`, `max_scale`")


#### Visual tools ## messy code ##############################################
def imshow(data, title=None, show=1, cmap=None, norm=None, complex=None, abs=0,
           w=None, h=None, ridge=0, ticks=1, aspect='auto',
           yticks=None, xticks=None, xlabel=None, ylabel=None, **kw):
    kw['interpolation'] = kw.get('interpolation', 'none')
    if norm is None:
        mx = np.max(np.abs(data))
        vmin, vmax = ((-mx, mx) if not abs else
                      (0, mx))
    else:
        vmin, vmax = norm
    if cmap is None:
        cmap = 'bone' if abs else 'bwr'
    _kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, aspect=aspect, **kw)

    if abs:
        plt.imshow(np.abs(data), **_kw)
    elif complex:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(data.real, **_kw)
        axes[1].imshow(data.imag, **_kw)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)
    else:
        plt.imshow(data.real, **_kw)

    if w or h:
        plt.gcf().set_size_inches(12 * (w or 1), 12 * (h or 1))

    if ridge:
        data_mx = np.where(np.abs(data) == np.abs(data).max(axis=0))
        plt.scatter(data_mx[1], data_mx[0], color='r', s=4)

    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks)
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)

    _maybe_title(title)
    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, ax_equal=False, complex=0, abs=0,
         c_annot=False, w=None, h=None, dx1=False, xlims=None, ylims=None,
         vert=False, vlines=None, hlines=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, ax=None, fig=None, ticks=True, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    if vert:
        x, y = y, x
    if complex:
        ax.plot(x, y.real, color='tab:blue', **kw)
        ax.plot(x, y.imag, color='tab:orange', **kw)
        if c_annot:
            _kw = dict(fontsize=15, xycoords='axes fraction', weight='bold')
            ax.annotate("real", xy=(.93, .95), color='tab:blue', **_kw)
            ax.annotate("imag", xy=(.93, .90), color='tab:orange', **_kw)
    else:
        if abs:
            y = np.abs(y)
        ax.plot(x, y, **kw)
    if dx1:
        ax.set_xticks(np.arange(len(x)))

    if vlines:
        vhlines(vlines, kind='v')
    if hlines:
        vhlines(hlines, kind='h')
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if xticks is not None or yticks is not None:
        _ticks(xticks, yticks)

    _maybe_title(title)
    _scale_plot(fig, ax, show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0),
                xlabel=xlabel, ylabel=ylabel)


def plots(X, Y=None, nrows=None, ncols=None, tight=True, sharex=False,
          sharey=False, skw=None, pkw=None, _scat=0, show=1, **kw):
    """Example:
    X = [[None, np.arange(xc, xc + wl)],
         [None, np.arange(xc + hop, xc + hop + wl)],
         None,
         None]
    Y = [[x, window],
         [x, window],
         xbuf[:, xbc],
         xbuf[:, xbc + 1]]
    pkw = [[{}]*2, [{}]*2, *[{'color': 'tab:green'}]*2]
    plots(X, Y, nrows=2, ncols=2, sharey='row', tight=tight, pkw=pkw)
    """
    def _process_args(X, Y, nrows, ncols, tight, skw, pkw, kw):
        X = X if isinstance(X, list) else [X]
        Y = Y if isinstance(Y, list) else [Y]
        skw = skw or {}
        pkw = pkw or [{}] * len(X)

        if nrows is None and ncols is None:
            nrows, ncols = len(X), 1
        elif nrows is None:
            nrows = max(len(X) // ncols, 1)
        elif ncols is None:
            ncols = max(len(X) // nrows, 1)

        default = dict(left=0, right=1, bottom=0, top=1, hspace=.1, wspace=.05)
        if tight:
            if not isinstance(tight, dict):
                tight = default.copy()
            else:
                for name in default:
                    if name not in tight:
                        tight[name] = default[name]

        kw['w'] = kw.get('w', .8)
        kw['h'] = kw.get('h', .8)  # default 'tight' enlarges plot
        return X, Y, nrows, ncols, tight, skw, pkw, kw

    X, Y, nrows, ncols, tight, skw, pkw, kw = _process_args(
        X, Y, nrows, ncols, tight, skw, pkw, kw)

    _, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, **skw)
    for ax, x, y, _pkw in zip(axes.flat, X, Y, pkw):
        if isinstance(x, list):
            for _x, _y, __pkw in zip(x, y, _pkw):
                plot(_x, _y, ax=ax, **__pkw, **kw)
                if _scat:
                    scat(_x, _y, ax=ax, **__pkw, **kw)
        else:
            plot(x, y, ax=ax, **_pkw, **kw)
            if _scat:
                scat(x, y, ax=ax, **_pkw, **kw)

    if tight:
        plt.subplots_adjust(**tight)
    if show:
        plt.show()


def scat(x, y=None, title=None, show=0, ax_equal=False, s=18, w=None, h=None,
         xlims=None, ylims=None, dx1=False, vlines=None, hlines=None, ticks=1,
         complex=False, xlabel=None, ylabel=None, ax=None, fig=None, **kw):
    ax  = ax  or plt.gca()
    fig = fig or plt.gcf()

    if x is None and y is None:
        raise Exception("`x` and `y` cannot both be None")
    elif x is None:
        x = np.arange(len(y))
    elif y is None:
        y = x
        x = np.arange(len(x))

    if complex:
        ax.scatter(x, y.real, s=s, **kw)
        ax.scatter(x, y.imag, s=s, **kw)
    else:
        ax.scatter(x, y, s=s, **kw)
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    _maybe_title(title)
    if vlines:
        vhlines(vlines, kind='v')
    if hlines:
        vhlines(hlines, kind='h')
    _scale_plot(fig, ax, show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0),
                xlabel=xlabel, ylabel=ylabel)


def plotscat(*args, **kw):
    show = kw.pop('show', False)
    plot(*args, **kw)
    scat(*args, **kw)
    if show:
        plt.show()


def hist(x, bins=500, title=None, show=0, stats=0):
    x = np.asarray(x)
    _ = plt.hist(x.ravel(), bins=bins)
    _maybe_title(title)
    if show:
        plt.show()
    if stats:
        mu, std, mn, mx = (x.mean(), x.std(), x.min(), x.max())
        print("(mean, std, min, max) = ({}, {}, {}, {})".format(
            *_fmt(mu, std, mn, mx)))
        return mu, std, mn, mx


def vhlines(lines, kind='v'):
    lfn = plt.axvline if kind=='v' else plt.axhline

    if not isinstance(lines, (list, tuple)):
        lines, lkw = [lines], {}
    elif isinstance(lines, list):
        lkw = {}
    elif isinstance(lines, tuple):
        lines, lkw = lines
        lines = lines if isinstance(lines, list) else [lines]
    else:
        raise ValueError("`lines` must be list or (list, dict) "
                         "(got %s)" % lines)
    for line in lines:
        lfn(line, **lkw)


def _fmt(*nums):
    return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
             ("%.3f" % n)) for n in nums]

def _ticks(xticks, yticks):
    def fmt(ticks):
        return ("%.d" if all(float(h).is_integer() for h in ticks) else
                "%.2f")

    if yticks is not None:
        idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
        yt = [fmt(yticks) % h for h in np.asarray(yticks)[idxs]]
        plt.yticks(idxs, yt)
    if xticks is not None:
        idxs = np.linspace(0, len(xticks) - 1, 8).astype('int32')
        xt = [fmt(xticks) % h for h in np.asarray(xticks)[idxs]]
        plt.xticks(idxs, xt)

def _maybe_title(title):
    if title is not None:
        title, kw = (title if isinstance(title, tuple) else
                     (title, {}))
        defaults = dict(loc='left', weight='bold', fontsize=16)
        for name in defaults:
            kw[name] = kw.get(name, defaults[name])
        plt.title(str(title), **kw)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, dx1=False, xlabel=None, ylabel=None):
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin

    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)
    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8*(w or 1), 8*(h or 1))
    if xlims:
        ax.set_xlim(*xlims)
    if ylims:
        ax.set_ylim(*ylims)
    if dx1:
        plt.xticks(np.arange(dx1))
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if xlabel is not None:
        plt.xlabel(xlabel, weight='bold', fontsize=15)
    if ylabel is not None:
        plt.ylabel(ylabel, weight='bold', fontsize=15)
    if show:
        plt.show()


#############################################################################
from .wavelets import Wavelet, _xifn, aifftshift
from .wavelets import center_frequency, freq_resolution, time_resolution
