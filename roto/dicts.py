"""
Functions and ontologies on dictionaries.
"""


def merge_two_dicts(a, b):
    """Merge two dicts into one dict, with the second overriding."""
    d = a.copy()
    d.update(b)
    return d

def merge_dicts(*dicts):
    """Ordered merge of dicts, with right overriding left."""
    d = {}
    for newdict in dicts:
        d.update(newdict)
    return d


class AttrDict(object):

    """
    A dictionary with mirrored attribute access.
    """

    def __init__(self, adict=None):
        if adict is None:
            adict = dict()
        self.__adict__ = adict

    def __getattr__(self, attr):
        if attr not in self.__adict__:
            raise AttributeError("'%s' attribute is not set" % attr)
        return self.__adict__[attr]

    def __setattr__(self, attr, value):
        if attr == '__adict__':
            object.__setattr__(self, attr, value)
            return
        self.__adict__[attr] = value

    def __setitem__(self, key, item):
        if isinstance(key, str):
            if not key.isidentifier():
                raise ValueError('key is not a valid name: {}'.format(key))
        else:
            raise ValueError('key is not a string: {}'.format(key))
        self.__adict__[key] = item

    def update(self, *d, **kwargs):
        """
        Convenience method for dict-like update.
        """
        assert len(d) < 2, 'up to one positional argument allowed'
        self.__adict__.update(*d, **kwargs)

    def copy(self):
        """
        Convenience method for dict-like copy.
        """
        return AttrDict(adict=self.__adict__.copy())

    def get(self, key, default=None):
        """
        Convenience method for dict-like get.
        """
        return self.__adict__.get(key, default)

    def keys(self):
        """
        Convenience method for dict-like keys.
        """
        return self.__adict__.keys()

    def values(self):
        """
        Convenience method for dict-like values.
        """
        return self.__adict__.values()

    def items(self):
        """
        Convenience method for dict-like items.
        """
        return self.__adict__.items()

    def __getitem__(self, key):
        return self.__adict__[key]

    def __delitem__(self, key):
        del self.__adict__[key]

    def __contains__(self, key):
        return key in self.__adict__

    def __len__(self):
        return len(self.__adict__)

    def __iter__(self):
        return iter(self.__adict__)

    def __str__(self):
        return str(self.__adict__)

    def __repr__(self):
        return f'AttrDict({self})'


class Tree(dict):

    """
    Simple auto-vivifying dict using __missing__:
    https://en.wikipedia.org/wiki/Autovivification#Python
    """

    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


class Reify(object):

    """
    Convert nested dictionary into hierarchical tree object
    """

    def __init__(self, *args, **onto):
        self._n = 0
        for arg in args:
            if isinstance(arg, dict):
                onto.update(arg)
        self.__name__ = onto.pop('__name__', 'root')
        for key in list(onto.keys()):
            if not key.isidentifier():
                log.warning('cannot reify non-identifier key: %s', key)
                continue
            if isinstance(onto[key], dict):
                onto[key]['__name__'] = '%s.%s' % (self.__name__, key)
                setattr(self, key, Reify(onto[key]))
            else:
                setattr(self, key, onto[key])
            self._n += 1

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return f'{self}: {self._n} {(self._n == 1) and "child" or "children"}'
