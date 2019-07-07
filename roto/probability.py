"""
Functions operating on (log) probabilities.
"""

import numpy as np
from numpy import exp, log, expm1, log1p


def logsumexp(log_probs, axis=None):
    """
    Redefine scipy.special.logsumexp
    see: http://bayesjumping.net/log-sum-exp-trick/
    """
    _max = np.max(log_probs)
    ds = log_probs - _max
    exp_sum = exp(ds).sum(axis=axis)
    return _max + log(exp_sum)
