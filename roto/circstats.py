"""
Functions for circular statistics.
"""

import numpy as np
import scipy.stats as st


def mean_resultant_vector(theta, weights=None, binsize=None, axis=-1):
    """Mean resultant vector for a set of angles (or binned angular data).

    Arguments:
    theta -- array of radian angle values
    weights -- optional weights (i.e., for binned angle counts)
    binsize -- optional bin size for bias correction of binned data
    axis -- axis for averaging the input angles

    Returns:
    (2,...)-shaped array of resultant vector angles and lengths
    """
    if weights is None:
        y_bar = np.sin(theta).sum(axis=axis) / theta.shape[axis]
        x_bar = np.cos(theta).sum(axis=axis) / theta.shape[axis]
    else:
        totw = weights.sum(axis=axis)
        y_bar = np.sum(weights * np.sin(theta), axis=axis) / totw
        x_bar = np.sum(weights * np.cos(theta), axis=axis) / totw
        if binsize is not None:
            correction = binsize / (2 * np.sin(binsize / 2))
            y_bar *= correction
            x_bar *= correction

    angle = np.arctan2(y_bar, x_bar)
    length = np.hypot(x_bar, y_bar)

    return np.c_[angle, length].T.squeeze()

def median_angle(theta, axis=-1):
    """Median of a set of angles (undefined for binned data).

    Arguments:
    theta -- array of radian angle values
    axis -- axis indexing distinct sets of input angles/weights

    Returns:
    median angles value(s)
    """
    raise NotImplementedError

def mean_angle(theta, weights=None, axis=-1):
    """Mean resultant vector angle for a set of angles (or weighted data).

    Arguments same as `mean_resultant_vector`.
    """
    return mean_resultant_vector(theta, weights, axis)[0]

mean = mean_angle

def mean_length(theta, weights=None, binsize=None, axis=-1):
    """Mean resultant vector length for a set of angles (or weighted data).

    Arguments same as `mean_resultant_vector`.
    """
    return mean_resultant_vector(theta, weights, binsize, axis)[1]

def var(theta, weights=None, binsize=None, axis=-1):
    """Sample circular variance (bounded to [0,1]).

    Arguments same as `mean_resultant_vector`.
    """
    return 1 - mean_length(theta, weights, binsize, axis)

def angular_deviation(theta, weights=None, binsize=None, axis=-1):
    """Angular deviation (bounded to [0,sqrt(2)]; Zar, 1999).

    Arguments same as `mean_resultant_vector`.
    """
    return np.sqrt(2 * var(theta, weights, binsize, axis))

std = angular_deviation

def unbounded_angular_deviation(theta, weights=None, binsize=None, axis=-1):
    """Unbounded [0,\infty] circular standard deviation (Mardia, 1972).

    Arguments same as `mean_resultant_vector`.
    """
    return np.sqrt(-2 * np.log(mean_length(theta, weights, binsize, axis)))

std_mardia = unbounded_angular_deviation


# For historical reasons...

def runningvar(theta, Nbins=360):
    """Running implementation of sample circular variance.

    Arguments:
    theta -- array of radian angle values
    numbins -- number of intervals across [0, 2pi] to minimize

    Returns:
    circular variance

    See also:
    Weber RO (1997). J. Appl. Meteorol. 36(10), 1403-1415.
    """
    N = len(theta)
    delta_t = 2 * np.pi / Nbins
    lims = (0, 2 * np.pi)
    x = np.arange(delta_t, 2*np.pi + delta_t, delta_t)
    n, xmin, w, extra = st.histogram(theta, numbins=Nbins, defaultlimits=lims)

    tbar = np.empty((Nbins,), 'd')
    S = np.empty((Nbins,), 'd')
    s2 = np.empty((Nbins,), 'd')

    tbar[0] = (x*n).sum() / N                                               # A1
    S[0] = ((x**2)*n).sum() / (N - 1)                                       # A2
    s2[0] = S[0] - N * (tbar[0]**2) / (N - 1)                               # A3

    for k in range(1, Nbins):
        tbar[k] = tbar[k-1] + (2*np.pi) * n[k-1] / N                        # A4
        S[k] = S[k-1] + (2*np.pi) * (2*np.pi + 2*x[k-1]) * n[k-1] / (N - 1) # A5
        s2[k] = S[k] - N * (tbar[k]**2) / (N - 1)                           # A6

    return s2.min()
