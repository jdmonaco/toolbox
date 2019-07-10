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

    Set `key_params` class attribute to a tuple of argument names to be used as
    caching keys. Set `data_root` class attribute to the root group name under
    which cached data should be saved. Set `cache_attrs` to a tuple of attribute
    names that contain the data results to be persisted in the datafile cache.
    Set `save_attrs` to other attribute names for, e.g., derived scalar
    quantities that should persist through the cache's key-value store.

    The `seed` constructor keyword is provided as a convenience for setting the
    seed of a numpy.RandomState object stored in the `rnd` instance attribute.
    """

    data_root = 'cache'
    key_params = ('seed',)
    cache_attrs = ()
    save_attrs = ()

    def __init__(self, seed=None, **kwargs):
        """
        All keyword-arguments become instance attributes used as cache keys.
        """
        for arg, value in kwargs.items():
            setattr(self, arg, value)

        for name in self.cache_attrs + self.save_attrs:
            setattr(self, name, None)

        for key in self.key_params:
            if not hasattr(self, key):
                log(key, prefix='MissingCacheKey', error=True)

        klass = self.__class__.__name__
        seed = klass if seed is None else seed
        self.seed = sum(map(ord, seed))
        if 'seed' not in self.key_params:
            self.key_params = tuple(self.key_params) + ('seed',)

        self.attrs = {k:getattr(self, k) for k in self.key_params}
        self.hash = attr_hash(self, *self.key_params)
        self.rnd = random.RandomState(seed=self.seed)
        self.cachename = '{}_{}'.format(naturalize(klass), self.hash)

    def compute(self, context=None):
        """
        Perform the computation and cache the results in context's datafile.
        """
        if context is not None and self._check_cache(context):
            return self._load(context)

        self._compute()

        if context is not None:
            self._cache(context)

    def _compute(self):
        """
        Subclasses override this method to perform the computation.

        The constructor arguments are available as instance attributes.

        The results of the computation should be set to the instance attributes
        listed in the `cache_attrs` class attribute. Derived scalar quantities
        should be set to the instance attributes listed in the `save_attrs`
        class attribute.

        Use the `rnd` instance attribute for random number generation in order
        to use the AutomaticCache random seed handling.
        """
        raise NotImplementedError

    def _cache(self, context):
        """
        Cache the computed data to the given context's datafile.
        """
        for name in self.cache_attrs:
            if getattr(self, name) is None:
                raise RuntimeError('run compute() first')

        # Add the scalar attributes to be cached
        self.attrs.update({k:getattr(self, k) for k in self.save_attrs})

        # Create the cache data group with all attributes
        grp = context.create_group(self.cachename, attrs=self.attrs,
                root=self.data_root)

        for name in self.cache_attrs:
            data = getattr(self, name)
            context.save_array(data, name, attrs=self.attrs, root=grp)

    def _check_cache(self, context):
        """
        Check whether the key specification has been cached.
        """
        return context.has_node(self.cachename, root=self.data_root)

    def _load(self, context):
        """
        Load cached data keyed by the current specification.
        """
        grp = context.get_node(self.cachename, root=self.data_root)

        for name in self.cache_attrs:
            setattr(self, name, context.read_array(name, root=grp))

        for name in self.save_attrs:
            setattr(self, name, grp._v_attrs[name])

        context.out(grp._v_pathname, prefix='AutoCacheLoad')

    def clear_cache(self, context):
        """
        Remove the cached data for the current specification.
        """
        if not self._check_cache(context):
            return

        context.get_datafile(readonly=False)
        cache = context.get_node(self.cachename, root=self.data_root)
        cache._f_remove(recursive=True)
        context.close_datafile()

    def delete_all_cache_data(self, context):
        """
        Completely delete all cached data that has been stored for this object
        type, regardless of key parameters or specification.
        """
        import tables as tb
        context.get_datafile(readonly=False)
        try:
            root = context.get_node(root=self.data_root)
        except tb.NoSuchNodeError:
            context.out('Could not find data root (/{})', self.data_root,
                    prefix='CacheDelete', error=True)
        else:
            root._f_remove(recursive=True)
        finally:
            context.close_datafile()