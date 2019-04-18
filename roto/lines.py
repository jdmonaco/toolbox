"""
Artists and functions for generating plots and plot elements.
"""

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

from .arrays import find_groups


def cmapline(x, y, c, ax=None, cmap=None, **fmt):
    """Plot a continuous trace where each segment is colormapped to data.

    Note: Generally, the size of the color array `c` should be one less than
    the size of the x/y arrays, in order to match the number of segments.

    Arguments:
    x, y -- x/y arrays for points along the trace
    c -- intensity array to be used for colormapping
    ax -- optional, axes object where the trace should be plotted
    cmap -- optional, colormap for mapping intensities to colors

    Remaining arguments are passed to `LineCollection.set(...)`.

    Returns:
    LineCollection object
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cdata = np.squeeze(c)
    assert len(cdata) <= len(segments), 'more colors ({}) provided than ' \
        'segments ({})'.format(len(cdata), len(segments))

    cmap = cmap or 'viridis'
    lc = LineCollection(segments, cmap=cmap, **fmt)
    lc.set_array(cdata)

    ax = ax or plt.gca()
    ax.add_collection(lc)
    ax.axis('tight')  # questionable, but plot may be out of window otherwise
    plt.draw_if_interactive()
    return lc


class LineCollectionPlot(object):

    def __init__(self, **fmt):
        self._lc = LineCollection([], **fmt)
        self._fmt = fmt

    def get_lc(self):
        """Get the line collection object."""
        return self._lc

    def set(self, **kwds):
        """Set properties on the trace line collection."""
        self._lc.set(**kwds)
        self._fmt.update(kwds)
        plt.draw_if_interactive()

    def reset(self):
        """Reset line collection properties."""
        self.set(**self._fmt)

    def remove(self):
        """Remove the trace from its current axes."""
        self._lc.remove()
        plt.draw_if_interactive()

    def _plot(self, ax, fmt):
        ax = ax is None and plt.gca() or ax
        if self._lc not in ax.get_children():
            ax.add_collection(self._lc)
        if fmt:
            self.set(**fmt)
        else:
            plt.draw_if_interactive()
        return self._lc


class HighlightsPlot(LineCollectionPlot):

    """
    Efficiently plot a large number of highlight line segments.
    """

    def __init__(self, x, y, **linefmt):
        """Initialize highlight plotting with the full data series.

        Arguments:
        x, y -- full data series that will be hightlighted

        Remaining arguments are passed to `LineCollection.set(...)`.
        """
        self._pts = np.c_[x, y]

        fmt = dict(color='r', linestyle='solid', lw=2, alpha=0.8)
        fmt.update(linefmt)

        LineCollectionPlot.__init__(self, **fmt)

    def plot(self, ix, min_segment_len=2, ax=None, **kwds):
        """Plot the data highlights as line segments.

        Arguments:
        ix -- boolean index array indicating highlighted points

        Keyword arguments:
        min_segment_len -- minimum group size to highlight
        ax -- axes object where the trace should be plotted

        Remaining arguments are passed to `LineCollection.set(...)`.

        Returns the line collection object.
        """
        grps = find_groups(ix, min_size=min_segment_len)
        if len(grps):
            segs = tuple(self._pts[i:j] for i, j in grps)
        else:
            segs = np.array([])

        c = self.get_lc()
        c.set_segments(segs)

        return self._plot(ax, kwds)


class TimeTracePlot(LineCollectionPlot):

    """
    Plot an arbitrary temporal trace from a time-series signal.
    """

    def __init__(self, t, x, y, dt=5.0, tau=1.0, cmap=None, **linefmt):
        """Set up the time trace plotting and store the time-series data.

        Arguments:
        t -- time array
        x, y -- data arrays for the full time series
        dt -- trace duration in seconds
        tau -- time constant for the exponential trace coloring
        cmap -- colormap for coloring the trace

        Remaining arguments are passed to `LineCollection.set(...)`.
        """
        self._t = t
        self._x = x
        self._y = y
        self._dt = dt
        self._tau = tau
        self._segments = self._get_all_segments()

        _cm = cmap is None and 'gray_r' or cmap
        fmt = dict(cmap=_cm, norm=Normalize(vmin=0, vmax=1, clip=True))
        fmt.update(linefmt)

        LineCollectionPlot.__init__(self, **fmt)

    def _get_all_segments(self):
        points = np.array([self._x, self._y]).T.reshape(-1, 1, 2)
        return np.concatenate([points[:-1], points[1:]], axis=1)

    def set_dt(self, newdt):
        """Set the total duration of the time trace."""
        self._dt = max(0.0, newdt)

    def set_tau(self, newtau):
        """Set the time constant for the exponential coloring of the trace."""
        self._tau = max(0.001, newtau)

    def plot(self, t0, ax=None, **kwds):
        """Plot a trailing time trace for a specified time point.

        Arguments:
        t0 -- time point from which the trace should trail

        Keyword arguments:
        ax -- axes object where the trace should be plotted

        Remaining keywords are passed to `LineCollection.set(...)`.

        Returns the line collection object.
        """
        ix = np.logical_and(self._t[1:] >= t0 - self._dt, self._t[1:] <= t0)
        if ix.any():
            segs = self._segments[ix]
            dt = self._t[np.r_[False,ix]] - t0
            h = np.exp(dt / self._tau)
        else:
            segs = np.array([])
            h = np.array([])

        c = self.get_lc()
        c.set_segments(segs)
        c.set_array(h)

        return self._plot(ax, kwds)
