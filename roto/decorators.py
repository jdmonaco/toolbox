"""
Collection of useful decorators.
"""

from decorator import decorator
from functools import wraps, update_wrapper
from collections import deque

import numpy as np

from pouty import debug

from .arrays import datahash


class lazyprop(object):

    """
    Method decorator for one-time conversion of result to class attribute

    From: http://stackoverflow.com/a/6849299
    """

    def __init__(self, func):
        self.func = func
        update_wrapper(self, func)

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.func(obj)
        setattr(obj, self.func.__name__, value)
        debug('lazyprop: setting {} on {}', self.func.__name__, obj)
        return value


@decorator
def memoize(f, *args, **kwds):
    if not hasattr(f, 'cache'):
        debug('memoize: creating cache on {}', f.__name__)
        f.cache = {}
    if kwds:
        key = args, frozenset(iter(kwds.items()))
    else:
        key = args
    cache = f.cache
    if key in cache:
        debug('memoize: retrieving {} from {}', key, f.__name__)
        return cache[key]
    cache[key] = result = f(*args, **kwds)
    debug('memoize: caching {} in {}', key, f.__name__)
    return result

@decorator
def datamemoize(f, *args, **kwds):
    """Hash any array arguments."""
    if not hasattr(f, 'cache'):
        debug('datamemoize: creating cache on {}', f.__name__)
        f.cache = {}
    arrays = []
    notarrays = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            arrays.append(arg)
        else:
            notarrays.append(arg)
    debug('datamemoize: hashing {} arrays', len(arrays))
    arrayhash = datahash(*arrays)
    if kwds:
        key = arrayhash, tuple(notarrays), frozenset(iter(kwds.items()))
    else:
        key = arrayhash, tuple(notarrays)
    cache = f.cache
    if key in cache:
        debug('datamemoize: retrieving {} from {}', key, f.__name__)
        return cache[key]
    cache[key] = result = f(*args, **kwds)
    debug('datamemoize: caching {} in {}', key, f.__name__)
    return result


#
# Old-style functools.wraps decorators
#

def wraps_memoize(f):
    cache = {}
    @wraps(f)
    def wrapper(*args, **kwds):
        key = (tuple(args), tuple(sorted(kwds.items())))
        if key not in cache:
            cache[key] = f(*args, **kwds)
        return cache[key]
    return wrapper

def wraps_memoize_limited(max_cache_size):
    def real_memoize_limited(f):
        cache = {}
        key_queue = deque()
        @wraps(f)
        def wrapper(*args, **kwds):
            key = (tuple(args), tuple(sorted(kwds.items())))
            if key not in cache:
                cache[key] = f(*args, **kwds)
                key_queue.append(key)
                if len(key_queue) > max_cache_size:
                    was_cached =cache.pop(key_queue.popleft(), None)
                return cache[key]
            return wrapper
    return real_memoize_limited
