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

    def __init__(self, adict=None, **kwargs):
        if adict is None: adict = dict()
        AttrDict.__setattr__(self, '_adict', adict)
        self.update(kwargs)

    def __getattr__(self, name):
        if name == '_adict':
            return object.__getattribute__(self, name)
        if name not in self._adict:
            raise AttributeError(f'attribute {name!r} is not set')
        return self._adict[name]

    def __setattr__(self, name, value):
        if name == '_adict':
            object.__setattr__(self, name, value)
            return
        if isinstance(name, str):
            if not name.isidentifier():
                raise ValueError(f'not a valid name ({name!r})')
        else:
            raise ValueError(f'not a string {name!r}')
        self._adict[name] = value

    def __setitem__(self, name, item):
        AttrDict.__setattr__(self, name, item)

    def update(self, *d, **kwargs):
        """
        Convenience method for dict-like update.
        """
        assert len(d) < 2, 'up to one positional argument allowed'
        self._adict.update(*d, **kwargs)

    def copy(self):
        """
        Convenience method for dict-like copy.
        """
        return AttrDict(adict=self._adict.copy())

    def get(self, name, default=None):
        """
        Convenience method for dict-like get.
        """
        return self._adict.get(name, default)

    def names(self):
        """
        Convenience method for dict-like names.
        """
        return self._adict.names()

    def values(self):
        """
        Convenience method for dict-like values.
        """
        return self._adict.values()

    def items(self):
        """
        Convenience method for dict-like items.
        """
        return self._adict.items()

    def __getitem__(self, name):
        return self._adict[name]

    def __delitem__(self, name):
        del self._adict[name]

    def __contains__(self, name):
        return name in self._adict

    def __bool__(self):
        return bool(self._adict)

    def __len__(self):
        return len(self._adict)

    def __iter__(self):
        return iter(self._adict)

    def __str__(self):
        klass = self.__class__.__name__
        indent = ' '*4
        r = f'{klass}('
        if len(self):
            r += '\n'
        for k, v in self.items():
            lines = f'{k} = {repr(v)},'.split('\n')
            for line in lines:
                r += indent + line + '\n'
        return r + ')'

    def __repr__(self):
        return f'{self.__class__.__name__}({self._adict})'


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
