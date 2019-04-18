"""
Functions for handling python dicts.
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
