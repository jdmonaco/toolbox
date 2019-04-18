"""
Custom colormap definitions

Written by Joe Monaco
Copyright (c) 2007-2008 Columbia Unversity. All Rights Reserved.
"""

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib as mpl

from .decorators import memoize


def default_cmap(circular=False, reverse=False):
    """Default colormap instance."""
    if circular:
        return huslmap()
    if reverse:
        return mpl.cm.viridis_r
    return mpl.cm.viridis

@memoize
def huslmap(h=0.92, s=1.0, l=0.67):
    """Created a husl-based circular colormap that is colorblind friendly.

    The husl map and default parameters avoid strong reds and greens.
    """
    # Use seaborn for its palettes, but leave rcParams unchanged
    rc = mpl.rcParams.copy()
    import seaborn as sns

    # Make the MPL colormap from the Seaborn/HUSL color palette
    N = 256
    husl_colors = sns.husl_palette(n_colors=N, h=h, s=s, l=l)
    husl_cmap = ListedColormap(husl_colors, name='husl', N=N)

    # Restore the current MPL rc settings
    mpl.rcParams.update(rc)

    return husl_cmap

def get_diffmap_for(M, mid_value=0.0, **kwargs):
    """For a given intensity matrix and mid-point value *mid_value*, return
    the difference map (see diffmap) with the proper mid-point color.
    """
    Mmin, Mmax = M.min(), M.max()
    return diffmap(mid=(mid_value-Mmin) / (Mmax - Mmin), **kwargs)


def diffmap(mid=0.5, use_black=False):
    """Conventional differencing map with graded red and blue for values less
    than and greater than, respectively, the mean of the data. Values approaching
    the mean are increasingly whitened, and the mean value is white.

    Keyword arguments:
    mid -- specify the midpoint value, colored white or black
    use_black -- if True, the midpoint value is black instead of white by default
    """
    m = int(not use_black)
    segmentdata = { 'red':   [(0, 1, 1), (mid, m, m), (1, 0, 0)],
                    'green': [(0, 0, 0), (mid, m, m), (1, 0, 0)],
                    'blue':  [(0, 0, 0), (mid, m, m), (1, 1, 1)] }
    return LSC('RdWhBu', segmentdata)
