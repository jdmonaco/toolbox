"""
Set operations over numpy arrays.

NB: These are now built into numpy (at least 1.11+):
    https://github.com/numpy/numpy/blob/v1.11.0/numpy/lib/arraysetops.py
"""

import numpy as np


# Set operation functions

def intersection(u, v):
    """Get array intersection of input arrays u and v"""
    return _do_set_op(u, v, 'intersection')

def union(u, v):
    """Get array union of input arrays u and v"""
    return _do_set_op(u, v, 'union')

def difference(u, v):
    """Get array difference of input arrays u and v"""
    return _do_set_op(u, v, 'difference')

def symmetric_difference(u, v):
    """Get array symmetric_difference of input arrays u and v"""
    return _do_set_op(u, v, 'symmetric_difference')


# Generic set-to-array translation function

def _do_set_op(u, v, set_op):
    assert type(u) is np.ndarray and type(v) is np.ndarray, 'need arrays'
    u_func = getattr(set(u), set_op)
    return np.array(list(u_func(set(v))))
