"""Convenience visual methods"""
import numpy as np
import matplotlib.pyplot as plt


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
         w=None, h=None, **kw):
    if y is None:
        y = x
        x = np.arange(len(x))
    if complex:
        plt.plot(x, y.real, **kw)
        plt.plot(x, y.imag, **kw)
    else:
        plt.plot(x, y, **kw)
    _maybe_title(title)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h)


def scat(x, y=None, title=None, show=0, ax_equal=False, s=18, w=None, h=None,
         **kw):
    if y is None:
        y = x
        x = np.arange(len(x))
    plt.scatter(x, y, s=s, **kw)
    _maybe_title(title)
    _scale_plot(plt.gcf(), plt.gca(), show=show, ax_equal=ax_equal, w=w, h=h)


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
        plt.title(str(title), loc='left', weight='bold', fontsize=15)


def _scale_plot(fig, ax, show=False, ax_equal=False, w=None, h=None):
    xmin, xmax = ax.get_xlim()
    rng = xmax - xmin
    ax.set_xlim(xmin + .018 * rng, xmax - .018 * rng)
    if w or h:
        fig.set_size_inches(14*(w or 1), 8*(h or 1))
    if ax_equal:
        yabsmax = max(np.abs([*ax.get_ylim()]))
        mx = max(yabsmax, max(np.abs([xmin, xmax])))
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-mx, mx)
        fig.set_size_inches(8*(w or 1), 8*(h or 1))
    if show:
        plt.show()


def plotenergy(x, axis=1, **kw):
    plot(np.sum(np.abs(x) ** 2, axis=axis), **kw)
