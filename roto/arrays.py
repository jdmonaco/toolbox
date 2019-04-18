"""
Array handling functions and operations.
"""

import numpy as np
import scipy.signal


def _extrema(x, which, wrapped):
    if wrapped:
        x = np.r_[x[-1], x, x[0]]
    ex = np.r_[0, np.diff((np.diff(x) >= 0).astype('i')), 0]

    if wrapped:
        ex = ex[1:-1]
    return np.nonzero(which(ex))[0]

def minima(x, wrapped=False):
    """Index array of the local minima of a continuous signal."""
    return _extrema(x, lambda x: x == +1, wrapped)

def maxima(x, wrapped=False):
    """Index array of the local maxima of a continuous signal."""
    return _extrema(x, lambda x: x == -1, wrapped)

def peaks(x, wrapped=False):
    """Index array of local extrema of a continuous signal."""
    return _extrema(x, lambda x: x != 0, wrapped)

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K

def halfwave(x, copy=False):
    """Half-wave rectifier for arrays or scalars

    NOTE: Specify copy=True if array data should be copied before performing
    halwave rectification.
    """
    if type(x) is np.ndarray and x.ndim:
        if copy:
            x = x.copy()
        x[x<0.0] = 0.0
    else:
        x = float(x)
        if x < 0.0:
            x = 0.0
    return x

def blur(x, pixels=1, padding='same', odd=True):
    """Gaussian blur with optional padding.

    Arguments:
    x -- 1D array of continuous values to blue

    Keyword arguments:
    pixels -- width (std. dev.) of blur in pixels/bins
    padding -- 'same'|'wrap'|'none'|scalar
        type of padding to use for the signal

    Returns:
    1D array similar to the input
    """
    size = int(np.ceil(8 * pixels))
    if odd and size % 2 == 0:
        size += 1
    if size > x.size:
        size = x.size
    if padding == 'none' or padding is None:
        padding = 0.0
    if padding == 'same':
        x = np.r_[np.tile(x[0], size), x, np.tile(x[-1], size)]
    elif padding == 'wrap':
        x = np.r_[x[-size:], x, x[:size]]
    else:
        x = np.r_[np.tile(padding, size), x, np.tile(padding, size)]

    G = scipy.signal.gaussian(size, pixels)
    G /= np.trapz(G)
    X = scipy.signal.convolve(x, G, mode='same')
    return X[size:-size]

def circblur(x, radians=None, degrees=None):
    """Gaussian blur (1D) of a circular array."""
    bins = x.size
    if degrees is not None:
        pixels = degrees * (bins / 360.0)
    elif radians is not None:
        pixels = radians * (bins / (2 * np.pi))
    else:
        raise ValueError('must specify blur width in radians or degrees')
    return blur(x, pixels=pixels, padding='wrap')

def boxcar(x, M=4, centered=True, out=None):
    """Perform a boxcar filter on the input signal.

    Keyword arguments:
    M -- number of averaged samples (default 4)
    centered -- recenter the filtered signal to reduce lag
    """
    length = x.shape[0]
    if length <= 2*M:
        raise ValueError('signal too short for specified filter window')

    z = np.zeros(length+M-1)
    for i in range(M):
        z += np.r_[np.zeros(i)+x[0], x, np.zeros(M-i-1)+x[-1]]

    start_ix = 0
    end_ix = length
    if centered:
        start_ix += int(M/2)
        end_ix += int(M/2)

    sf = z[start_ix:end_ix] / M
    if out is not None:
        out[:] = sf
        return out
    return sf

def discretize(x, levels=5, circular=False, vmin=None, vmax=None,
    inplace=False):
    """Discretize continuous array values into equal-spaced levels."""
    q = x = np.atleast_1d(x)
    fnt = np.isfinite(x)
    if not inplace:
        q = x.copy()
    if vmin is None:
        vmin = x[fnt].min()
    if vmax is None:
        vmax = x[fnt].max()
    ptp = vmax - vmin
    lv = np.linspace(vmin, vmax, levels + int(circular))
    ix = ((x[fnt] - vmin) / ptp * int(levels)).astype(int)
    q[fnt] = lv[np.clip(ix, 0, levels - 1)]
    return q

def align_center(M, ix, align='rows'):
    """Create a center-aligned copy of a 2D matrix, where the aligned rows or
    columns are wrapped around

    For matrix with even number of columns, the lower-index column around the
    center is used as the alignment index.

    Arguments:
    M -- 2D matrix to be aligned
    ix -- index array for bins to align in each row or column
    align -- whether to align 'rows' or 'columns' (default 'rows')
    """
    if align == 'cols':
        M = M.T

    R, C = M.shape
    W = np.empty_like(M)
    delta = ix - int((C - 0.5) / 2)
    for i in range(R):
        d = delta[i]
        for j in range(C):
            W[i, j] = M[i, (j+d)%C]

    if align == 'cols':
        return W.T
    return W

def datahash(*arrays):
    """Hash the data buffers of some arrays."""
    hashstr = ''
    for arr in arrays:
        w = arr.flags.writeable
        arr.flags.writeable = False
        hashstr += '[{}{}]'.format(hash(bytes(arr)), arr.shape)
        arr.flags.writeable = w
    return hash(hashstr)

def find_groups(ix, min_size=1):
    """Find contiguous groups of samples in a binary group membership array.

    Arguments:
    ix -- boolean index array indicating group membership
    min_size -- ignore groups below the minimum size

    Returns list of (start, end) slice tuples.
    """
    ix = np.asarray(ix, '?')
    last_ix = len(ix) - 1
    groups = []
    cursor = 0
    new_start = -1
    while cursor <= last_ix:
        if ix[cursor]:
            if new_start == -1:
                new_start = cursor
            while cursor <= last_ix and ix[cursor]:
                cursor += 1
        elif new_start >= 0:
            if cursor - new_start >= min_size:
                groups.append((new_start, cursor))
            new_start = -1
            while cursor <= last_ix and not ix[cursor]:
                cursor += 1
        else:
            cursor += 1
    if new_start >= 0 and cursor - new_start >= min_size:
        groups.append((new_start, cursor))
    return np.asarray(groups, 'i')

def merge_adjacent_groups(groups, tol=0):
    """Combine adjacent groups wth some tolerance for gaps."""
    if not len(groups):
        return []
    merged = []
    groups = list(map(list, sorted(groups)))
    merged.append(groups[0])
    for g in groups[1:]:
        if g[0] <= merged[-1][1] + tol:
            merged[-1][1] = g[1]
        else:
            merged.append(g)
    return merged
