"""
Functions for random numbers and sampling.
"""

import numpy as np


def randc(N):
    """Uniform random samples (2, N) from the unit disc."""
    t = 2 * np.pi * np.random.rand(N)
    r = np.random.rand(N) + np.random.rand(N)
    r[r > 1] = 2 - r[r > 1]
    return r * np.vstack((np.cos(t), np.sin(t)))
