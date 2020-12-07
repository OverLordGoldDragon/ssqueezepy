# -*- coding: utf-8 -*-
"""Convenience visual methods"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, ifftshift
from .algos import find_closest


#### Visual tools ## messy code ##############################################
def imshow(data, title=None, show=1, cmap=None, norm=None, complex=None, abs=0,
           w=None, h=None, ridge=0, ticks=1, yticks=None, aspect='auto', **kw):
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
        axes[0].imshow(data.imag, **_kw)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                            wspace=0, hspace=0)
    else:
        plt.imshow(data.real, **_kw)

    if w or h:
        plt.gcf().set_size_inches(14 * (w or 1), 8 * (h or 1))

    if ridge:
        data_mx = np.where(np.abs(data) == np.abs(data).max(axis=0))
        plt.scatter(data_mx[1], data_mx[0], color='r', s=4)
    if not ticks:
        plt.xticks([])
        plt.yticks([])
    if yticks is not None:
        idxs = np.linspace(0, len(yticks) - 1, 8).astype('int32')
        yt = ["%.2f" % h for h in yticks[idxs]]
        plt.yticks(idxs, yt)
    _maybe_title(title)
    if show:
        plt.show()


def plot(x, y=None, title=None, show=0, ax_equal=False, complex=0,
         c_annot=False, w=None, h=None, dx1=False, xlims=None, ylims=None,
         vert=False, **kw):
    if y is None:
        y = x
        x = np.arange(len(x))
    if vert:
        x, y = y, x
    if complex:
        plt.plot(x, y.real, color='tab:blue', **kw)
        plt.plot(x, y.imag, color='tab:orange', **kw)
        if c_annot:
            _kw = dict(fontsize=15, xycoords='axes fraction', weight='bold')
            plt.annotate("real", xy=(.93, .95), color='tab:blue', **_kw)
            plt.annotate("imag", xy=(.93, .90), color='tab:orange', **_kw)
    else:
        plt.plot(x, y, **kw)
    if dx1:
        plt.xticks(np.arange(len(x)))
    _maybe_title(title)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0))


def scat(x, y=None, title=None, show=0, ax_equal=False, s=18, w=None, h=None,
         xlims=None, ylims=None, dx1=False, **kw):
    if y is None:
        y = x
        x = np.arange(len(x))
    plt.scatter(x, y, s=s, **kw)
    _maybe_title(title)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h,
                xlims=xlims, ylims=ylims, dx1=(len(x) if dx1 else 0))


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


def _fmt(*nums):
    return [(("%.3e" % n) if (abs(n) > 1e3 or abs(n) < 1e-3) else
             ("%.3f" % n)) for n in nums]


def _maybe_title(title):
    if title is not None:
        plt.title(str(title), loc='left', weight='bold', fontsize=16)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None,
                xlims=None, ylims=None, dx1=False):
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
    if show:
        plt.show()


def plotenergy(x, axis=1, **kw):
    plot(np.sum(np.abs(x) ** 2, axis=axis), **kw)


#### Visualizations ##########################################################
def _viz_cwt_scalebounds(psihfn, N, min_scale=None, max_scale=None,
                         cutoff=1, stdevs=2):
    """Can be used to visualize time & freq domains separately, where
    `min_scale` refers to scale at which to show the freq-domain wavelet, and
    `max_scale` the time-domain one.
    """
    from .wavelets import _xi
    def _viz_max():
        from .wavelets import time_resolution

        t = np.arange(3, step=1/N)  # 3 per 3*T in time_resolution
        t -= t.mean()
        psi = ifftshift(ifft(psihfn(_xi(max_scale, len(t)))))
        std_t = time_resolution(psihfn, max_scale, N)

        plot(t, np.abs(psi)**2, ylims=(0, None),
             title="|Time-domain wavelet|^2, extended (outside dashed)")

        plt.axvline(std_t,          color='tab:red')
        plt.axvline(std_t * stdevs, color='tab:green')
        # mark target (non-extended) frame
        [plt.axvline(v, color='k', linestyle='--') for v in (-.5, .5)]

        _kw = dict(fontsize=16, xycoords='axes fraction', weight='bold')
        plt.annotate("1 stdev",
                     xy=(.88, .95), color='tab:red',   **_kw)
        plt.annotate("%s stdevs" % stdevs,
                     xy=(.88, .90), color='tab:green', **_kw)
        plt.show()

    def _viz_min():
        psih = psihfn(_xi(min_scale, N))[:N//2 + 1]

        plot(psih, title=("Frequency-domain wavelet, positive half "
                          "(cutoff=%s)" % cutoff))
        plt.axhline(psih.max() * cutoff, color='tab:red')
        plt.show()

    if min_scale is not None:
        _viz_min()
    if max_scale is not None:
        _viz_max()
    if not (min_scale or max_scale):
        raise ValueError("Must set at least one of `min_scale`, `max_scale`")


def wavelet_tf(wavelet, N=2048, scale=100, notext=False, width=1.1, height=1):
    """Visualize `wavelet` joint time-frequency resolution. Plots frequency-domain
    wavelet (psih) along y-axis, and time-domain wavelet (psi) along x-axis.

    Orthogonal units (e.g. y-axis for psi) are meaningless; function values
    aren't to scale, but *widths* are, so time-frequency uncertainties are
    accurately captured.

    `wavelet` is instance of `wavelets.Wavelet` or its valid `wavelet` argument.
    See also: https://www.desmos.com/calculator/nqowgloycy
    """
    psihfn = (wavelet if isinstance(wavelet, Wavelet) else
              Wavelet(wavelet))

    #### Compute psi & psihf #################################################
    psi  = ifft(psihfn(scale * _xi(1, N)) * (-1)**np.arange(N))
    apsi = np.abs(psi)
    t = np.arange(-N/2, N/2, step=1)

    w = aifftshift(_xi(1, N))[N//2-1:]
    psih = psihfn(scale * w)

    #### Compute stdevs & respective indices #################################
    wc    = center_frequency(psihfn, scale, N)
    std_w = freq_resolution(psihfn, scale, N, nondim=0)
    std_t = time_resolution(psihfn, scale, N, nondim=0)
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
    plot([w_xminu, w_xmax], [w_ymin , w_ymin], **lkw)
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
    _kw = dict(s=txt, xycoords='axes fraction', xy=(.7, .76), weight='bold',
               fontsize=16)
    try:
        plt.annotate(family='Consolas', **_kw)  # 'Consolas' for vertical align
    except:
        plt.annotate(**_kw)  # in case platform lacks 'Consolas'

    ## Title: wavelet name & parameters
    title = psihfn._desc(N=N, scale=scale)
    plt.title(title, loc='left', weight='bold', fontsize=16)

    ## Styling
    plt.xlabel("samples", weight='bold', fontsize=15)
    plt.ylabel("radians", weight='bold', fontsize=15)

    plt.gcf().set_size_inches(12*width, 12*height)
    plt.show()


def wavelet_tf_anim(wavelet, N=2048, scales=None, width=1.1, height=1,
                    savepath='wavanim.gif'):
    """This method computes same as `wavelet_tf` but for all scales at once,
    and animates 'intelligently'. Recommended to leave `scales` to None
    """
    def _make_anim_scales(psihfn, N):
        scales = process_scales('log', N, psihfn, nv=32, minbounds=True)

        # compute early and late scales more densely as they capture more
        # interesting behavior, so animation will slow down smoothly near ends
        scales = scales.squeeze()
        na = len(scales)

        s0 = (25/253)*na  # empircally-determined good value

        srepl = int(s0)  # scales to keep from each end
        srepr = int(s0)
        smull = 3        # extension factor
        smulr = 3

        sright = np.linspace(scales[-srepr], scales[-1],   srepr * smulr)
        sright = np.hstack([sright, sright[-1].repeat(smulr*2)])  # smooth loop
        sleft  = np.linspace(scales[0],      scales[srepl], srepl * smull)
        sleft = np.hstack([sleft[0].repeat(smulr*2), sleft])

        scales = np.hstack([sleft, scales[srepl:-srepr], sright])
        scales  = scales.reshape(-1, 1)
        return scales

    from .utils import NOTE, process_scales
    from matplotlib.animation import FuncAnimation
    import matplotlib
    matplotlib.use("Agg")
    NOTE("Switched matplotlib to 'Agg' backend for animating")

    psihfn = (wavelet if isinstance(wavelet, Wavelet) else
              Wavelet(wavelet))
    scales = _make_anim_scales(psihfn, N)

    #### Compute Psi & Psih ##################################################
    Psi = ifft(psihfn(scales * _xi(1, N), nohalf=False
                      ) * (-1)**np.arange(N).reshape(1, -1),
               axis=-1)
    aPsi = np.abs(Psi)
    t = np.arange(-N/2, N/2, step=1)

    w = aifftshift(_xi(1, N))[N//2-1:]
    Psih = psihfn(scales * w)

    #### Compute stdevs & respective indices #################################
    Wc    = np.zeros(len(scales))
    std_W = Wc.copy()
    std_T = Wc.copy()

    for i, scale in enumerate(scales):
        Wc[i]    = center_frequency(psihfn, float(scale), N, kind='energy')
        std_W[i] = freq_resolution( psihfn, float(scale), N, nondim=0)
        std_T[i] = time_resolution( psihfn, float(scale), N, nondim=0)
    _Wc = np.pi - Wc

    Wcix = find_closest(_Wc.reshape(-1, 1), w).squeeze()
    Wlix = find_closest((_Wc - std_W).reshape(-1, 1), w).squeeze()
    Wrix = Wcix + (Wcix - Wlix)
    Wl, Wr = w[Wlix], w[Wrix]

    Tcix = np.argmin(np.abs(t - 0)).repeat(len(scales))
    Tlix = find_closest(-std_T.reshape(-1, 1), t).squeeze()
    Trix = Tcix + (Tcix - Tlix)
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
    ax.set_ylim([-aPsi.max()*1.8, np.pi*1.02])

    ylabels = np.round(np.linspace(np.pi, 0, 7), 1)
    plt.yticks(np.linspace(0, np.pi, len(ylabels)), ylabels)

    fig.set_size_inches(12*width, 12*height)

    ## Title: wavelet name & parameters
    title = psihfn._desc(N=N)
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

    #### Animate #############################################################
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

    print("Successfully computed parameters; animating...", flush=True)
    frames = np.hstack([range(len(scales)), range(len(scales) - 1)[::-1]])
    anim = FuncAnimation(fig, animate, frames=frames, interval=60,
                         blit=True, repeat=False)

    anim.save(savepath, writer='imagemagick')
    print("Animated and saved to", savepath, flush=True)


def wavelet_heatmap(wavelet, scales='log', N=2048, minbounds=False):
    psihfn = (wavelet if isinstance(wavelet, Wavelet) else
              Wavelet(wavelet))
    if not isinstance(scales, np.ndarray):
        from .utils import process_scales
        scales = process_scales('log', N, wavelet, nv=32, minbounds=minbounds)

    #### Compute time- & freq-domain wavelets for all scales #################
    _psih = psihfn(scales * _xi(1, N), nohalf=False
                   ) * (-1)**np.arange(N).reshape(1,-1)
    Psi = ifft(_psih, axis=-1)

    w = aifftshift(_xi(1, N))[N//2-1:]
    Psih = psihfn(scales * w)

    #### Plot ################################################################
    mx = np.abs(Psi).max() * .01
    title0 = psihfn._desc(N=N)

    imshow(Psi.real,   norm=(-mx, mx), yticks=scales,
           title=title0 + " | Time-domain; real part")
    imshow(Psi, abs=1, norm=(0, mx),   yticks=scales,
           title=title0 + " | Time-domain; abs-val")
    imshow(Psih, abs=1, cmap='jet', yticks=scales,
           title=title0 + "| Freq-domain; abs-val")

#############################################################################
from .wavelets import Wavelet, _xi, aifftshift
from .wavelets import center_frequency, freq_resolution, time_resolution