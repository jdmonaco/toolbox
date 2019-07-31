"""
A 'null' object that, like most neutrinos, does not interact.
"""


class _Null(object):

    """
    The null-object design pattern.

    From the Python Cookbook, Second Edition: Recipe 6.17.
    """

    def __new__(cls, *p, **kw):
        """
        This is a singleton, so ensure there will only be one instance.
        """
        if '_inst' not in vars(cls):
            cls._inst = object.__new__(cls, *p, **kw)
        return cls._inst

    def __init__(self, *p, **kw): pass
    def __call__(self, *p, **kw): return self
    def __str__(self): return "Null()"
    def __repr__(self): return "Null()"
    def __bool__(self): return False
    def __getattr__(self, name): return self
    def __delattr__(self, name): return self
    def __setattr__(self, name): return self
    def __getitem__(self, i): return self
    def __setitem__(self, *p): pass


# Create the singleton

Null = _Null()
