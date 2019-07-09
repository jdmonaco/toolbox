"""
Automatic caching storage for expensive computed sigals.
"""

from pouty import log
from toolbox.numpy import random
from roto.strings import naturalize
from roto.datapath import attr_hash


class AutomaticCache(object):

    """
    Auto-cache computed results with argument-based keys.

    Subclasses override the _compute() method to perform the computation.

    Set `key_attrs` class attribute to a tuple of arguments to be used as
    caching keys. Set `data_root` class attribute to the root group name under
    which cached data should be saved. Set `save_attrs` to a tuple of attribute
    names contain the results to be saved & loaded from the datafile cache.

    The `seed` constructor keyword is provided as a convenience for setting the
    seed of a numpy.RandomState object stored in the `rnd` instance attribute.
    """

    data_root = 'cache'
    key_attrs = ('seed',)
    save_attrs = ()

    def __init__(self, seed=None, **kwargs):
        """
        All keyword-arguments become instance attributes used as cache keys.
        """
        for arg, value in kwargs.items():
            setattr(self, arg, value)

        for name in self.save_attrs:
            setattr(self, name, None)

        for key in self.key_attrs:
            if not hasattr(self, key):
                log(key, prefix='MissingCacheKey', error=True)

        klass = self.__class__.__name__
        seed = klass if seed is None else seed
        self.seed = sum(map(ord, seed))
        if 'seed' not in self.key_attrs:
            self.key_attrs = tuple(self.key_attrs) + ('seed',)

        self.attrs = {k:getattr(self, k) for k in self.key_attrs}
        self.hash = attr_hash(self, *self.key_attrs)
        self.rnd = random.RandomState(seed=self.seed)
        self.cachename = '{}_{}'.format(naturalize(klass), self.hash)

    def compute(self, context=None):
        """
        Perform the computation, then cache & return the results.
        """
        if context is not None and self.check_cache(context):
            return self._load(context)

        self._compute()

        if context is not None:
            self._cache(context)

        return tuple([getattr(self, name) for name in self.save_attrs])

    def _compute(self):
        """
        Subclasses override this method to perform the computation.

        The constructor arguments are available as instance attributes.

        The results of the computation should be set to the instance attributes
        listed in the `save_attrs` class attribute. Those results will be
        returned in a tuple as ordered by `save_attrs`.

        Use the `rnd` instance attribute for random number generation in order
        to use the AutomaticCache random seed handling.
        """
        raise NotImplementedError

    def _cache(self, context):
        """
        Cache the computed data to the given context's datafile.
        """
        for name in self.save_attrs:
            if getattr(self, name) is None:
                raise RuntimeError('run compute() first')

        for name in self.save_attrs:
            context.save_array(getattr(self, name), self.cachename, name,
                    root=self.data_root, attrs=self.attrs)

    def _check_cache(self, context):
        """
        Check whether the key specification has been cached.
        """
        return context.has_node(self.cachename, root=self.data_root)

    def _load(self, context):
        """
        Load cached data keyed by the current specification.
        """
        context.out(context.datapath(self.cachename, root='layout'),
                prefix='AutoCacheLoad')

        results = []
        for name in self.save_attrs:
            setattr(self, name, context.read_array(self.cachename, name,
                root=self.data_root)
            results.append(getattr(self, name))
        return tuple(results)
