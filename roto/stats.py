"""
Statistics functions and classes.
"""

import os
import sys
import collections

from scipy.signal import gaussian, convolve
import numpy as np
import scipy.stats as st

from pouty import warn


Interval = collections.namedtuple("Interval", "lower upper")
TtestResults = collections.namedtuple("Ttest", "T p")


def oneway(M):
    k = M.shape[1]
    N = M.size
    F, p = st.f_oneway(*M.T)
    return k-1, N-k, F, p

def oneway_str(M):
    return 'F(%d,%d) = %.3f, p < %f' % oneway(M)

def friedman(M):
    k = M.shape[1]
    N = M.size
    F, p = st.friedmanchisquare(*M.T)
    return k-1, N-k, F, p

def friedman_str(M):
    return 'X2(%d,%d) = %.3f, p < %e' % friedman(M)


class SampleDistribution(object):

    """
    Random sampling distribution based on a counts histogram
    """

    def __init__(self, H, values):
        self.p = np.cumsum(KT_estimate(H))
        self.values = np.asarray(values)

    @classmethod
    def from_integers(cls, x):
        counts, H = integer_hist(x, relative=False)
        return cls(H, counts)

    def __call__(self, *args):
        """Call with shape parameters (d1[, d2[, ...]]) to get random samples
        or without any parameters for a scalar sample
        """
        r = np.random.rand(*args)
        if type(r) is float:
            samples = self.values[(r < self.p).nonzero()[0][0]]
        elif type(r) is np.ndarray:
            samples = np.array(
                [self.values[np.nonzero(x < self.p)[0][0]]
                    for x in r.flat]).reshape(r.shape)
        return samples

    rand = __call__


def FDR_control(p_values, alpha=0.05):
    """Control the false discovery rate with adaptive alpha procedure

    Returns index array for controlled p-values signficant at alpha.
    """
    p = np.array(p_values)
    six = np.argsort(p)
    k = p.size
    factor = np.arange(1, k+1) / k
    # return six[p[six] <= alpha * factor]
    return p <= (alpha * factor)[six]

def KT_estimate(H, zero_trim=True):
    """From a 1D histogram of counts, create a probability distribution using
    the Krichesky-Trofimov estimate to avoid zeros.

    By default, leading and trailing zero counts are not adjusted. To include
    these zeros in the estimate, set *zero_trim* to False.
    """
    H = np.asarray(H)
    N = H.sum()
    if zero_trim and N:
        nz = H.nonzero()[0]
        i, j = nz[0], nz[-1] + 1
    else:
        i, j = 0, H.size
    return np.r_[H[:i], (H[i:j] + 0.5) / (N + 0.5*(j-i)), H[j:]]

def KT_estimate2(H, zero_trim=True):
    """For a 2D histogram, rows are treated as distinct probability
    densities: if H[i,j] is the count (x_i, y_i) for variables x and y,
    then each row of the return value is the K-T estimate of P(y|x_i).
    """
    assert H.ndim == 2, "histogram must be 2D"
    P = np.zeros_like(H, 'd')
    for i in range(H.shape[0]):
        if H[i].sum():
            P[i] = KT_estimate(H[i], zero_trim=zero_trim)
    return P

def rank_partition(x, N):
    """Sort values and partition into N quasi-equal-sized sub-arrays
    """
    xs = np.sort(np.asarray(x))
    p = [round(xs.size*(i/float(N))) for i in range(0,N+1)]
    return [xs[slice(*ix)] for ix in zip(p[:-1], p[1:])]
quintiles = lambda x: rank_partition(x, 5)
quartiles = lambda x: rank_partition(x, 4)
deciles = lambda x: rank_partition(x, 10)
quintiles.__doc__ = quartiles.__doc__ = deciles.__doc__ = rank_partition.__doc__

def pvalue_2samp(x, y):
    nx, ny = x.size, y.size
    pop = np.r_[np.zeros(nx), np.ones(ny)]
    xy = np.r_[np.array(x), np.array(y)]
    popsort = pop[np.argsort(xy)]
    return np.sum(np.abs(np.diff(popsort))) / (nx + ny - 2)

def pvalue(obs, null):
    return ((obs <= np.asarray(null)).sum()+1) / float(len(null))

def sem(values):
    return np.std(values, ddof=1) / np.sqrt(len(values))

def ci(x, which='empirical', alpha=0.05, axis=0):
    """Compute confidence intervals for an arrays of row-oriented samples.
    
    Arguments:
    which -- 'empirical' | 'parametric'
    alpha -- two-tailed critical value, equivalent to 1-percentile
    axis -- array axis over which to compute the confidence intervals
    """
    if which == 'empirical':
        return ci_empirical(x, alpha=alpha, axis=axis)
    elif which == 'parametric':
        return ci_parametric(x, alpha=alpha, axis=axis)
    raise ValueError(f'unknown type "{which}" for confidence intervals')

def ci_empirical(x, alpha=0.05, axis=0):
    """Empirical confidence intervals for array of samples.
    """
    x = np.asarray(x)
    xs = x.shape[axis]
    s = np.argsort(x, axis=axis)
    c = max(0, int((alpha / 2) * xs))
    if c <= 1:
        warn('shape {} is low for alpha {}', xs, alpha)
    return np.take_along_axis(x, np.take(s, [c,xs-c-1], axis), axis)

def ci_parametric(x, alpha=0.05, axis=0):
    """Normal (symmetric) confidence intervals for array of samples.
    """
    x = np.asarray(x)
    n = x.shape[axis]
    m = x.mean(axis=axis)
    s = x.std(axis=axis, ddof=1)
    t_crit = st.distributions.t.isf(alpha / 2, n - 1)
    ll = m - t_crit * s / np.sqrt(n)
    ul = m + t_crit * s / np.sqrt(n)
    return Interval(ll, ul)

def IQ_interval(x, factor=0.0):
    """Get the innerquartile interval with an additional factor extension
    """
    xs = np.sort(np.asarray(x))
    il, iu = int(0.25 * xs.size), int(0.75 * xs.size)
    l = (xs[il-1] + xs[il]) / 2.0
    u = (xs[iu-1] + xs[iu]) / 2.0
    d = (u - l) * factor
    return Interval(l - d, u + d)

def IQR(x):
    """The interquartile range as the difference between Q1 and Q3.
    """
    IQ = IQ_interval(x)
    return IQ.upper - IQ.lower

def freedman_diaconis_bins(a, vmin=None, vmax=None):
    """Histogram bins specified by the Freedman-Diaconis rule."""
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    a = np.atleast_1d(a)
    h = 2 * IQR(a) / a.size**(1/3)
    bins = np.arange(vmin, vmax+h, h)
    return bins

def t_one_tailed(x, mu=1.0):
    """[Deprecated] one-sample one-tailed Student's t-test
    """
    return t_one_sample(x, mu, tails=1)

def t_one_sample(x, mu, tails=2):
    """One- or two-sided t-tests for one sample compared to null mean
    """
    assert tails in (1,2), "tails must be 1 or 2, found %s"%str(tails)
    x = np.asarray(x)
    N = x.size
    df = N - 1
    t_obs = (x.mean() - mu) / (x.std(ddof=1) / np.sqrt(N))
    p_value = tails * st.t.sf(abs(t_obs), df)
    return TtestResults(t_obs, p_value)

def t_paired(x, y, mu=0, tails=2):
    """Dependent t-tests for paired sample differences, pairs should be
    ordered across both samples
    """
    assert tails in (1,2), "tails must be 1 or 2, found %s"%str(tails)
    x, y = np.asarray(x), np.asarray(y)
    x_d = x - y
    N = x_d.size
    df = 2 * N - 2
    t_obs = (x_d.mean() - mu) / (x_d.std(ddof=1) / np.sqrt(N))
    p_value = tails * st.t.sf(abs(t_obs), df)
    return TtestResults(t_obs, p_value)

def t_two_sample(x, y, tails=2):
    """One- or two-sided t-tests for two unequal-size samples assuming equal variance
    """
    assert tails in (1,2), "invalid: tails must be 1 or 2, found %s"%str(tails)
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.size, y.size
    df = nx + ny - 2
    s_xy = np.sqrt(((nx - 1)*x.var() + (ny - 1)*y.var()) / df)
    t_obs = (x.mean() - y.mean()) / (s_xy * np.sqrt(1./nx + 1./ny))
    p_value = tails * st.t.sf(abs(t_obs), df)
    return TtestResults(t_obs, p_value)

def t_welch(x, y, tails=2, return_tpdf=False):
    """Welch's t-test for two unequal-size samples, not assuming equal variances
    """
    assert tails in (1,2), "invalid: tails must be 1 or 2, found %s"%str(tails)
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.size, y.size
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    df = ((vx/nx + vy/ny)**2 / # Welch-Satterthwaite equation
        ((vx/nx)**2 / (nx - 1) + (vy/ny)**2 / (ny - 1)))
    t_obs = (x.mean() - y.mean()) / np.sqrt(vx/nx + vy/ny)
    p_value = tails * st.t.sf(abs(t_obs), df)
    if return_tpdf:
        return dict(t=t_obs, p=p_value, df=df)
    return TtestResults(t_obs, p_value)

def zscore(a, population=None):
    """Return Z-scored array same size as input array

    Keyword arguments:
    population -- optionally specify the reference population as a larger
        sample array or a (mean, std) tuple of the population; if not
        specified, as by default, then the input array is used
    """
    x = np.asarray(a)
    if population is None:
        mu, sigma = x.mean(), x.std(ddof=1)
    else:
        if type(population) in (tuple, list) and len(population) == 2:
            mu, sigma = population
        elif type(population) is np.ndarray:
            mu, sigma = population.mean(), population.std(ddof=1)
        else:
            raise ValueError('population must be (mu, sigma) tuple or array')
    Z = (x - mu) / sigma
    if Z.size < 2:
        Z = float(np.squeeze(Z))
    return Z

def bootstrap_array(f_values, values, bootstraps=1000):
    """Build a bootstrap distribution by random resampling of an array of
    input values and a function that converts an input array into a
    scalar outcome measure
    """
    x = np.array(values)
    x_bs = x[np.random.randint(x.size, size=(bootstraps, x.size))]
    return np.array(list(map(f_values, x_bs)))

def bootstrap_hit_fraction(frac, N, bootstraps=1000):
    """Given a hit fraction and the number of samples, back out the sample
    vector and bootstrap a sample distribution around the hit fraction
    """
    N = float(N)
    hits = int(frac * N)
    s = np.zeros(N, '?')
    s[:hits] = 1 # recreate sample hit-vector
    bs = np.empty((bootstraps,), 'd')
    for i in range(bootstraps):
        bs[i] = np.sum(s[np.random.randint(N, size=N)]) / N
    return bs

def bootstrap(X, N, H0, *args):
    """Get a one-tail p-value for an algorithmic sampling process

    H0(*args) must return a scalar null sample value.

    The sign of the returned p-value indicates whether X is less than (-) or
    greater than (+) the median of the sample distribution.

    Arguments:
    X -- the value for which to return a p-value
    N -- sampling size of the empirical distribution (beware O(n))
    H0 -- function that implements sampling process for the null result
    args -- additional arguments will be passed to H0
    """
    assert isinstance(H0, collections.Callable), 'H0 must be a callable that returns a scalar sample'
    tail = 0
    for i in range(N):
        tail += int(H0(*args) >= X)
    if tail > float(N)/2:
        tail = tail - N # negative p-value for X less than median
    if not tail:
        sys.stderr.write('warning: bootstrap needs N > %d; returning upper bound\n'%N)
        tail = 1
    return tail / float(N)

def smooth_pdf(a, sd=None):
    """Get a smoothed pdf of an array of data for visualization

    Keyword arguments:
    sd -- S.D. of the gaussian kernel used to perform the smoothing (default is
        1/20 of the data range)

    Return 2-row (x, pdf(x)) smoothed probability density estimate.
    """
    a = np.array(a).flatten()
    if sd is None:
        sd = 0.05 * a.ptp()
    data = a.copy().flatten() # get 1D copy of array data
    nbins = len(data) > 1000 and len(data) or 1000 # num bins >~ O(len(data))
    f, l = np.histogram(data, bins=nbins, normed=True) # fine pdf
    sd_bins = sd * (float(nbins) / a.ptp()) # convert sd to bin units
    kern_size = int(10*sd_bins) # sufficient convolution kernel size
    g = gaussian(kern_size, sd_bins) # generate smoothing kernel
    c = np.cumsum(f, dtype='d') # raw fine-grained cdf
    cext = np.r_[np.array((0,)*(2*kern_size), 'd'), c,
        np.array((c[-1],)*(2*kern_size), 'd')] # wrap data to kill boundary effect
    cs = convolve(cext, g, mode='same') # smooth the extended cdf
    ps = np.diff(cs) # differentiate smooth cdf to get smooth pdf
    dl = l[1]-l[0] # get bin delta
    l = np.r_[np.arange(l[0]-kern_size*dl, l[0], dl), l,
        np.arange(l[-1]+dl, l[-1]+kern_size*dl, dl)] # pad index to match bounds
    ps = ps[kern_size:kern_size+len(l)] # crop pdf to same length as index
    ps /= np.trapz(ps, x=l) # normalize pdf integral to unity
    return np.c_[l, ps].T # return 2-row concatenation of x and pdf(x)

def integer_hist(a, int_range=None, open_range=False, relative=False):
    """Compute a histogram of array data across integer bins

    Parameters:
    a -- the data to be histogrammed (ndim > 1 is flattened)
    int_range -- inclusive (min, max) range for integer values
    open_range -- whether high bin is unbounded or not
    relative -- whether count should be relative frequency or raw counts

    Returns (values, count):
    values -- integer bin values for the histogram
    count -- bin frequencies, whether relative frequency or raw count
    """
    data = np.round(a).flatten()
    if int_range:
        values = np.arange(int(int_range[0]), int(int_range[1])+1)
    else:
        values = np.arange(int(data.min()), int(data.max())+1)
    N = values.size
    if relative:
        count = np.empty(N, 'd')
    else:
        count = np.empty(N, 'l')
    for bin, c in enumerate(values):
        if open_range and bin == N - 1:
            count[bin] = (data >= c).sum()
        else:
            count[bin] = (data == c).sum()
    if relative:
        count /= count.sum()
    return values, count
