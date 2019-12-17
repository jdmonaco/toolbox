"""
Functions for plotting and changing elements of a figure axis.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .images import tiling_dims


def padding(ax, pad=0.1, xpad=None, ypad=None):
    """Add proportional padding around the current plot axis."""
    xpadding(ax, pad=xpad or pad)
    ypadding(ax, pad=ypad or pad)

def xpadding(ax, pad=0.1):
    """Add horizontal padding to the left and right of a plot."""
    v = ax.axis()
    ptp = v[1] - v[0]
    padding = pad * ptp
    ax.axis((v[0] - padding, v[1] + padding) + v[2:])

def ypadding(ax, pad=0.1):
    """Add vertical padding to the bottom and top of a plot."""
    v = ax.axis()
    ptp = v[3] - v[2]
    padding = pad * ptp
    ax.axis(v[:2] + (v[2] - padding, v[3] + padding))

def xlabel(ax, label, **kwargs):
    """Create a fake ylabel using plt.Axes.text(...)."""
    fmt = dict(transform=ax.transAxes, va='top', ha='center',
            size=mpl.rcParams['axes.labelsize'])
    fmt.update(kwargs)
    ax.text(0.5, -0.05, label, **fmt)
    plt.draw_if_interactive()

def ylabel(ax, label, **kwargs):
    """Create a fake xlabel using plt.Axes.text(...)."""
    fmt = dict(transform=ax.transAxes, va='center', ha='right', rotation=90,
            size=mpl.rcParams['axes.labelsize'])
    fmt.update(kwargs)
    ax.text(-0.05, 0.5, label, **fmt)
    plt.draw_if_interactive()

def despine(ax, top=False, right=False, left=True, bottom=True):
    ax.tick_params(top=top, right=right, left=left, bottom=bottom)
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)

def quicktitle(ax, text, **kwds):
    """Put short title on top an axis plot, optimized for low-margin plots

    Keywords are passed to ax.text(...)
    """
    text_fmt = dict(ha='center', va='bottom', size='small', zorder=100)
    text_fmt.update(**kwds)
    text_fn = hasattr(ax, 'text2D') and ax.text2D or ax.text
    h = text_fn(0.5, 1.0, str(text), color='k', transform=ax.transAxes,
            **text_fmt)
    plt.draw_if_interactive()
    return h

def textlabel(ax, text, side='right', **kwds):
    """Put short text label in a box on the top right corner of an axis plot

    Keywords are passed to ax.text(...)
    """
    text_fmt = dict(ha=side, va='top', size='medium', zorder=100)
    text_fmt.update(**kwds)
    text_fn = hasattr(ax, 'text2D') and ax.text2D or ax.text
    h = text_fn(dict(left=0, right=1)[side], 1, str(text), color='k',
        transform=ax.transAxes, bbox=dict(fc='w'), **text_fmt)
    plt.draw_if_interactive()
    return h

def make_panels(position, projection={}):
    """Create axes with customized positions mapped to keys

    The position description should be a mapping (dict) from keys to position
    (l, b, w, h) rects, subplot (r,c,N) tuples, or subplot integer codes (e.g.,
    221). The specified axes are created in the current figure and a mapping
    from keys to subplot axes is returned.

    Optionally, non-standard axes projections ('3d', 'polar') can be specified
    in the projection dict.
    """
    f = plt.gcf()
    f.clf()
    axdict = {}
    if '3d' in list(projection.values()):
        from mpl_toolkits.mplot3d import Axes3D
    for k in position:
        pos = position[k]
        proj = projection.get(k, 'rectilinear')
        if np.iterable(pos) and len(pos) == 3:
            axdict[k] = f.add_subplot(*pos, projection=proj)
        elif np.iterable(pos) and len(pos) == 4:
            axdict[k] = f.add_axes(pos, projection=proj)
        elif type(pos) is int:
            axdict[k] = f.add_subplot(pos, projection=proj)
        else:
            raise ValueError('bad subplot/axes: %s'%pos)
    return axdict


class AxesList(list):

    """
    Pipeline-able list of Axes objects.
    """

    def add_figure(self, f=None):
        if f is None:
            f = plt.gcf()
        elif type(f) in (int,str):
            f = plt.figure(f)
        self.extend([a for a in f.get_children() if hasattr(a, "draw_artist")])
        return self

    def make_grid(self, shape):
        """Create a grid of subplots in the current figure, based on either the
        number of required plots or a (nrows, ncols) tuple
        """
        f = plt.gcf()
        f.clf()
        if type(shape) is int:
            N = shape
            rows, cols = tiling_dims(N)
        elif type(shape) is tuple and len(shape) == 2:
            rows, cols = shape
            N = rows*cols
        else:
            raise ValueError('invalid grid shape parameter: %s'%str(shape))
        for r in range(rows):
            for c in range(cols):
                panel = cols*r + c + 1
                if panel > N:
                    break
                plt.subplot(rows, cols, panel)
        return self.add_figure(f)

    def _bounding_box(self):
        lims = np.array([ax.axis() for ax in self]).T
        left, right = lims[0].min(), lims[1].max()
        bottom, top = lims[2].min(), lims[3].max()
        return left, right, bottom, top

    def _add_padding(self, bbox, factor=0.1):
        dx, dy = bbox[1]-bbox[0], bbox[3]-bbox[2]
        left, right = bbox[0]-dx*factor, bbox[1]+dx*factor
        bottom, top = bbox[2]-dy*factor, bbox[3]+dy*factor
        return left, right, bottom, top

    def xnorm(self, padding=None):
        return self.normalize(padding=padding, yaxis=False)
    def ynorm(self, padding=None):
        return self.normalize(padding=padding, xaxis=False)

    def normalize(self, padding=None, xaxis=True, yaxis=True):
        bbox = self._bounding_box()
        if padding:
            bbox = self._add_padding(bbox, factor=float(padding))
        for ax in self:
            if xaxis:
                ax.set(xlim=bbox[:2])
            if yaxis:
                ax.set(ylim=bbox[2:])
        return self

    def map(self, func, args):
        assert len(args) == len(self), 'argument size mismatch'
        plt.ioff()
        [getattr(self[i], func)(args[i]) for i in range(len(self))]
        self.draw()
        return self

    def apply(self, func="set", *args, **kwds):
        plt.ioff()
        [getattr(ax, func)(*args, **kwds) for ax in self]
        self.draw()
        return self

    def set(self, **kwds):
        return self.apply(**kwds)

    def axis(self, mode):
        [ax.axis(mode) for ax in self]
        return self
    def equal(self):
        return self.axis('equal')
    def scaled(self):
        return self.axis('scaled')
    def tight(self):
        return self.axis('tight')
    def image(self):
        return self.axis('image')
    def off(self):
        return self.axis('off')
    def on(self):
        [ax.set_axis_on() for ax in self]
        return self

    def gallery(self):
        return self.equal().normalize().off()

    def draw(self):
        plt.ion()
        plt.draw()
