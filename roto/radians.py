"""
Functions for handling radian values.
"""

from matplotlib.ticker import ScalarFormatter
from numpy import pi
import numpy as np


two_pi = 2.0 * pi


# Circular differences

def cdiffsc(a, b, degrees=False):
    """Smallest circular difference between two scalar angles."""
    CIRC_MAX = degrees and 360 or two_pi
    delta = (a % CIRC_MAX) - (b % CIRC_MAX)
    mag = abs(delta)
    if mag > CIRC_MAX / 2:
        if delta > 0:
            return delta - CIRC_MAX
        else:
            return CIRC_MAX - mag
    else:
        return delta

def cdiff(u, v):
    """Smallest circular difference between two angle arrays."""
    if not (np.iterable(u) or np.iterable(v)):
        return cdiffsc(u, v)
    delta = np.fmod(u, two_pi) - np.fmod(v, two_pi)
    mag = np.absolute(delta)
    res = np.empty_like(delta)
    # cond 1: mag > pi, delta > 0
    ix = np.logical_and(mag > pi, delta > 0)
    res[ix] = delta[ix] - two_pi
    # cond 2: mag > pi, delta <= 0
    ix = np.logical_and(mag > pi, delta <= 0)
    res[ix] = two_pi - mag[ix]
    # cond 3: mag <= pi
    ix = mag <= pi
    res[ix] = delta[ix]
    return res


# Radian output formatting

def prettify(r, inline=False, modtwopi=False):
    """Format radian number with pi symbols."""
    sgn = ''
    if r < 0:
        sgn = '-'
    r = abs(r)
    if inline:
        frac, div = r'', r'/'
    else:
        frac, div = r'\frac', r''

    if np.isclose(r, two_pi):
        return r'$%s2\pi$' % sgn
    if modtwopi:
        r = r % two_pi
    if np.isclose(r, 0.0):
        return r'0'
    if np.isclose(r % pi, 0.0):
        m = r / pi
        if m == 1:
            return r'$%s\pi$' % sgn
        return r'$%s%d\pi$' % (sgn, int(np.round(m)))
    if np.isclose(r % (pi / 2), 0.0):
        m = r / (pi / 2)
        if m == 1:
            return r'$%s%s{\pi}%s{2}$' % (sgn, frac, div)
        return r'$%s%s{%d\pi}%s{2}$' % (sgn, frac, int(np.round(m)), div)
    if np.isclose(r % (pi / 3), 0.0):
        m = r / (pi / 3)
        if m == 1:
            return r'$%s%s{\pi}%s{3}$' % (sgn, frac, div)
        return r'$%s%s{%d\pi}%s{3}$' % (sgn, frac, int(np.round(m)), div)
    if np.isclose(r % (pi / 4), 0.0):
        m = r / (pi / 4)
        if m == 1:
            return r'$%s%s{\pi}%s{4}$' % (sgn, frac, div)
        return r'$%s%s{%d\pi}%s{4}$' % (sgn, frac, int(np.round(m)), div)
    if np.isclose(r % (pi / 8), 0.0):
        m = r / (pi / 8)
        if m == 1:
            return r'$%s%s{\pi}%s{8}$' % (sgn, frac, div)
        return r'$%s%s{%d\pi}%s{8}$' % (sgn, frac, int(np.round(m)), div)
    return '%s%.3gr' % (sgn, r)


class RadianFormatter(ScalarFormatter):

    """
    A scalar tick formatter that prettifies radian values.
    """

    def __init__(self, inline=False, **kwargs):
        self._inline = inline
        kwargs.update(useOffset=False)
        ScalarFormatter.__init__(self, **kwargs)

    def pprint_val(self, x):
        return prettify(x, inline=self._inline)


# Angular rotation

def rotxy(x, y, angles):
    """Rotate 2D point(s) around the origin by angle(s) with broadcasting.

    The arguments may be scalar or arrays, as long as all arrays are aligned
    for broadcasting. An exception will occur if the arguments cannot be
    broadcast together.

    Arguments:
    x,y -- scalar point or (x, y) arrays of points
    angles -- scalar or array of angles in radians for rotating (x, y)

    Returns:
    Scalar tuple (xrot, yrot) of the rotated point, or
    Array tuple (xrot, yrot) of rotated points
    """
    x, y, angles = np.broadcast_arrays(x, y, angles)
    sh = x.shape
    X = np.c_[x.ravel(), y.ravel()]
    ar = angles.ravel()
    cos_a = np.cos(ar)
    sin_a = np.sin(ar)
    R = np.moveaxis(np.array([[cos_a, -sin_a], [sin_a, cos_a]]), -1, 0)
    rotated = np.array([np.dot(R[i], X[i]) for i in range(len(X))])
    xr, yr = rotated.T
    if xr.size == 1:
        return float(xr), float(yr)
    return xr.reshape(sh), yr.reshape(sh)

def rotoffset(x, y, x0, y0, angles):
    """Rotate 2D point(s) around an offset point by angle(s).

    Arguments like `rotxy` with addition of offset point (x0,y0).
    """
    dx, dy = x - x0, y - y0
    xr, yr = rotxy(dx, dy, angles)
    return xr + x0, yr + y0
