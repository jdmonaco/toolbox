"""
Some useful generators.
"""

import numpy as np


def unique(a):
    """Generator for in-order unique items in an iterable."""
    u = set()
    for value in a:
        if value not in u:
            u.add(value)
            yield value

def chain(source, *transforms):
    """Chain a series of generators in intuitive order
    http://tartley.com/?p=1471
    """
    args = source
    for transform in transforms:
        args = transform(args)
    return args

def outer_pairs(A, B):
    """Generator for iterating through all possible pairs of items in the
    given pair of sequences.
    """
    for first in A:
        for second in B:
            yield (first, second)

def unique_pairs(seq, cross=None):
    """Generator for iterating through all the unique pairs of items in the
    given sequence.
    """
    N = len(seq)
    for i, first in enumerate(seq[:-1]):
        for j in range(i+1, N):
            second = seq[j]
            yield (first, second)

def ravelnz(a):
    """Generator yields non-zero elements of an array."""
    for b in np.asarray(a).ravel():
        if np.isfinite(b) and b:
            yield b
