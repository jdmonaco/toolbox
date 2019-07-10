"""
Functions for saving images of data.
"""

from PIL import Image
from matplotlib.colors import colorConverter
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from roto.colormaps import default_cmap


def map_to_rgba(M, **kwargs):
    """Convert map matrix into a masked RGBA color matrix (MxNx4).

    Map matrix `M` must be either an (m,n)- or (2,m,n)-shaped array.

    Mask is created from nan/inf values and passed to `array_to_rgba` along
    with any keyword arguments.
    """
    M = np.atleast_2d(M)
    if M.ndim == 2:
        Mt = M.T
        mask = ~np.isfinite(Mt)
    elif M.ndim == 3 and M.shape[0] == 2:
        Mt = np.empty((2, M.shape[2], M.shape[1]))
        Mt[0] = M[0].T
        Mt[1] = M[1].T
        mask = ~(np.isfinite(Mt[0]) & np.isfinite(Mt[1]))
    else:
        raise ValueError("map matrix has invalid size %s" % str(M.shape))

    if not mask.any():
        mask = None
    kwargs.update(mask=mask)

    return array_to_rgba(Mt, **kwargs)

def stack_to_image(stack, filename, **kwargs):
    """Save a stitched and flattened image of a vertical stack of 2D arrays."""
    N = stack.shape[0]
    sx, sy = stack.shape[1:]
    r = c = max(tiling_dims(N))
    res = (sx * r, sy * c)
    Pimg = np.zeros(res)

    for i in reversed(range(N)):
        x0 = sx * (i % c)
        x1 = x0 + sx
        y0 = sy * (r - 1 - int(i / r))
        y1 = y0 + sy
        Pimg[x0:x1,y0:y1] = stack[i]

    #NOTE: Not sure about this transpose or the row-flip in the loop above
    imgarr = Pimg.T
    kwargs.update(mask=~np.isfinite(imgarr))
    array_to_image(imgarr, filename, **kwargs)

def array_to_image(M, filename, **kwargs):
    """Save 2D matrix to image file.

    Remaining keywords are passed to `array_to_rgba`.

    Arguments:
    M -- matrix of pixel intensities
    filename -- filename for the saved image
    """
    rgba_to_image(array_to_rgba(np.flipud(M), **kwargs), filename)

def _fill_rgba(shape, color):
    rgba = np.empty(shape + (4,), 'uint8')
    rgba[:] = uint8color(color)
    return rgba

def array_to_rgba(M, mask=None, mask_color='w', cmap=None,
    cmin=None, cmax=None, satnorm=False):
    """Convert intensity matrix to RGBA color matrix (MxNx4).

    Keyword arguments:
    mask -- boolean masking matrix, optional
    mask_color -- mpl color for masked pixels if mask provided
    cmap -- mpl colormap instance or name (default 'viridis' or 'husl')
    cmin -- minimum clipping bound of the color range (default data min)
    cmax -- maximum color bound (default data max)
    satnorm -- normalize saturatation matrix range to [0,1]
    """
    M, S = M.copy(), None
    if M.ndim == 3:
        M, S = M

    # Return fully masked image if array (or satmat) is masked
    if mask is not None:
        Mnan = M[~mask]
        if Mnan.size:
            M[mask] = Mnan.min()  # preserve norm'n
        else:
            return _fill_rgba(M.shape, mask_color)
        if S is not None:
            Snan = S[~mask]
            if Snan.size:
                S[mask] = Snan.min()
            else:
                return _fill_rgba(S.shape, mask_color)

    if S is not None:
        np.clip(S, 0.0, 1.0, out=S)
        if satnorm:
            S[:] = S / S.max()

    if cmin is None:
        cmin = M.min()
    if cmax is None:
        cmax = M.max()
    np.clip(M, cmin, cmax, out=M)

    if cmap is None:
        cmap = default_cmap(circular=S is not None)

    cmap = plt.get_cmap(cmap)
    if cmin == cmax:
        rgba = _fill_rgba(M.shape, cmap(0.0))
    else:
        rgba = cmap((M - cmin) / (cmax - cmin), bytes=True)

    if S is not None:
        satmask(rgba, S, out=rgba)

    if mask is not None:
        rgba[mask] = uint8color(mask_color)

    return rgba

def satmask(rgba, sat, out=None):
    """Apply MxN saturation array to MxNx4 RGBA color matrix."""
    mat = out is None and rgba.copy() or out
    mat[...,:3] = (sat.reshape(sat.shape + (1,)) * rgba[...,:3]).astype('uint8')
    return mat

def uint8color(color):
    """Convert mpl color spec to uint8 4-tuple."""
    return tuple(int(255*v) for v in colorConverter.to_rgba(color))

def rgba_to_image(rgba, filename):
    """Save RGBA color matrix to image file."""
    img = Image.fromarray(rgba, 'RGBA')
    img.save(filename)

def tiling_dims(N):
    """Square-ish (rows, columns) for tiling N things."""
    d = np.ceil(np.sqrt(N))
    return int(np.ceil(N / d)), int(d)
