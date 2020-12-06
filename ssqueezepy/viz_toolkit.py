# -*- coding: utf-8 -*-
"""Convenience visual methods"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import ifft, ifftshift


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
    else:
        if (complex is None and np.sum(np.abs(np.imag(data))) < 1e-8) or (
                complex is False):
            plt.imshow(data.real, **_kw)
        else:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(data.real, **_kw)
            axes[0].imshow(data.imag, **_kw)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                wspace=0, hspace=0)
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
        plt.plot(x, y.real, **kw)
        plt.plot(x, y.imag, **kw)
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
        plt.title(str(title), loc='left', weight='bold', fontsize=17)


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


def viz_wavelet_tf(wavelet, N=2048, scale=100, pnv=None, notext=False,
                   width=1.1, height=1):
    """Visualize `wavelet` joint time-frequency resolution. Plots frequency-domain
    wavelet (psih) along y-axis, and time-domain wavelet (psi) along x-axis.

    Orthogonal units (e.g. y-axis for psi) are meaningless; function values
    aren't to scale, but *widths* are, so time-frequency uncertainties are
    accurately captured.

    `wavelet` is instance of `wavelets.Wavelet` or its valid `wavelet` argument.
    See also: https://www.desmos.com/calculator/nqowgloycy
    """
    from .wavelets import Wavelet, _xi, aifftshift
    from .wavelets import center_frequency, freq_resolution, time_resolution

    psihfn = (wavelet if isinstance(wavelet, Wavelet) else
              Wavelet(wavelet))

    #### Compute psi & psihfn ################################################
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

    wcix = np.argmin(np.abs(w - _wc))
    wlix = np.argmin(np.abs(w - (_wc - std_w)))
    wrix = np.argmin(np.abs(w - (_wc + std_w)))
    # wrix = wcix + (wcix - wlix)
    wl, wr = w[wlix], w[wrix]

    tcix = np.argmin(np.abs(t - 0))
    tlix = np.argmin(np.abs(t - (0 - std_t)))
    trix = tcix + (tcix - tlix)
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
    ptxt = ""
    if pnv is not None:
        ptxt = ""
        for name, value in pnv.items():
            ptxt += "{}={:.2f}, ".format(name, value)
    elif psihfn.config_str != "Default configs":
        ptxt = psihfn.config_str
    ptxt = ptxt.rstrip(', ')

    title = "{} wavelet | {}, scale={}, N={}".format(psihfn.name, ptxt, scale, N)
    plt.title(title, loc='left', weight='bold', fontsize=16)

    ## Styling
    plt.xlabel("samples", weight='bold', fontsize=15)
    plt.ylabel("radians", weight='bold', fontsize=15)

    plt.gcf().set_size_inches(12*width, 12*height)
