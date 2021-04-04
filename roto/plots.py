"""
Artists and functions for generating plots and plot elements.
"""

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np

from .images import array_to_rgba
from .stats import ci


def shaded_ci(x, data, ci_type='empirical', level=0.95, ax=None, **poly):
    """Plot shaded confidence intervals for row-oriented sample array."""
    lower, upper = ci(data, which=ci_type, alpha=1-level, axis=0)
    return shaded_region(x, lower, upper, ax=ax, **poly)

def shaded_error(x, mu, err, **poly):
    """Plot a shaded error interval (mu-err, mu+err) around mu."""
    return shaded_region(x, mu - err, mu + err, **poly)

def shaded_region(x, lower, upper, ax=None, adjustlims=True, **poly):
    """Plot a shaded region [lower, upper] over the range x."""
    if ax is None:
        ax = plt.gca()
    x, lo, hi = list(map(np.array, [x, lower, upper]))
    style = dict(lw=0, fill=True, zorder=-1, clip_box=ax.bbox)
    style.update(poly)
    P = Polygon(np.c_[np.r_[x,x[::-1]], np.r_[lo,hi[::-1]]], **style)
    ax.add_artist(P)
    if adjustlims:
        ax.set_ylim(bottom=min(ax.get_ylim()[0], lower.min()),
                    top=max(ax.get_ylim()[1], upper.max()))
    plt.draw()
    return P

def heatmap(x, y, ax=None, bins=32, range=None, **rgba):
    """Plot a masked intensity map of (x,y) scatter data.

    Keywords are passed to array_to_rgba(...).
    """
    ax = ax is None and plt.gca() or ax
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=range)
    xlim, ylim = [(e[0], e[-1]) for e in (xe, ye)]
    rgba = array_to_rgba(H.T, mask=(H.T==0), **rgba)
    im = ax.imshow(rgba, origin='lower', aspect='auto',
        interpolation='nearest', extent=xlim + ylim, zorder=-1)
    ax.set(xlim=xlim, ylim=ylim)
    plt.draw_if_interactive()
    return im

def condmap(x, y, ax=None, bins=32, range=None, **rgba):
    """Plot a conditional probability map of p(y|x) for the given data.

    Keywords are passed to array_to_rgba(...).
    """
    ax = ax is None and plt.gca() or ax
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=range)
    xlim, ylim = [(e[0], e[-1]) for e in (xe, ye)]
    M = H.T / H.T.sum(axis=0).reshape((1,-1))
    M[~np.isfinite(M)] = 0.0  # zero-out zero-count columns
    rgba = array_to_rgba(M, **rgba)
    im = ax.imshow(rgba, origin='lower', aspect='auto',
        interpolation='nearest', extent=xlim + ylim, zorder=-1)
    ax.set(xlim=xlim, ylim=ylim)
    plt.draw_if_interactive()
    return im

def grouped_bar_plot(data, groups, values, errors=None, baselines=None, ax=None,
    width=0.8, label_str='%s', legend=True, legend_loc=1, **kwds):
    """Grouped bar plot of M groups with N values each

    Handles single group bar plots as well (M=1), just pass in 1D arrays and
    a single group name.

    Arguments:
    data -- MxN matrix of data values to plot
    groups -- list of group names, to be used as x-tick labels
    values -- list of (name, color) tuple pairs specifying the values within
        each group with a corresponding mpl color specification
    errors -- optional MxN matrix of errors corresponding to the data
    baselines -- optional MxN matrix of baseline levels to be plotted as
        dashed lines across the corresponding bars
    ax -- axis object to contain the bar plot (default to new figure)
    width -- bar width as a proportion of maximum available horizontal space
    label_str -- formatting string for value names used for the legend
    legend -- whether to display the values legend
    legend_loc -- legend location passed to MPL legend command

    Remaining keywords are passed to the MPL bar command. Rectangle handles
    for the bars are returned.
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    data = np.asarray(data)
    if errors is not None:
        errors = np.asarray(errors)
        assert errors.shape==data.shape, 'errors mismatch with data shape'
    if data.ndim == 1:
        groups = [groups]
        data = np.asarray([data])
        if errors is not None:
            errors = np.asarray([errors])
        if baselines is not None:
            baselines = np.asarray([baselines])
    if type(values[0]) not in (list, tuple):
        colors = 'bgrcmykw'
        values = [(name, colors[np.fmod(i, len(colors))]) for i,name in
            enumerate(values)]

    value_list = [ name for name, color in values ]
    color_dict = { name: color for name, color in values }
    group_size = len(value_list)
    bar_width = width / group_size
    centers = np.arange(0, len(groups))
    lefts = []
    heights = []
    yerr = []
    colors = []
    for i, group in enumerate(groups):
        for j, value in enumerate(value_list):
            lefts.append(i+(j-group_size/2.0)*bar_width)
            heights.append(data[i, j])
            colors.append(color_dict[value])
            if errors is not None:
                yerr.append(errors[i, j])
            if baselines is not None:
                ax.plot([lefts[-1], lefts[-1]+bar_width],
                    [baselines[i, j]]*2, 'k--', lw=2, zorder=10)

    bar_kwds = dict(width=bar_width, color=colors, linewidth=1)
    bar_kwds.update(**kwds)
    if errors is not None:
        bar_kwds.update(yerr=yerr, capsize=0, ecolor='k')

    h = ax.bar(lefts, heights, **bar_kwds)
    if legend:
        ax.legend(h[:group_size], [label_str%value for value in value_list],
            loc=legend_loc)

    ax.set_xticks(centers)
    ax.set_xticklabels(groups, size='small')
    ax.set_xlim(-0.5, len(groups)-0.5)
    return h
