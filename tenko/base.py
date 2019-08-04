"""
Base class for context components.
"""

from pouty.console import ConsolePrinter


class TenkoObject(object):

    """
    Provide auto-naming and basic output functionality.
    """

    __counts = {}

    def __init__(self, name=None, color=None, textcolor=None):
        self.klass = self.__class__.__name__
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            self.name = f'{self.klass}_{c:03d}'
        else:
            self.name = name

        self.out = ConsolePrinter(prefix=self.name, prefix_color=color,
                message_color=textcolor)
        self.debug = self.out.debug

    def __str__(self):
        return f'{self.name}<{self.klass}>'

    def __repr__(self):
        return f'{self.klass}(name={self.name!r})'

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)
