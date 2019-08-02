"""
Base class for context components.
"""

from pouty.console import ConsolePrinter


class TenkoObject(object):

    """
    Provide auto-naming and basic output functionality.
    """

    __counts = {}

    def __init__(self, name=None):
        self.klass = self.__class__.__name__
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            self.name = f'{self.klass}-{c:03d}'
        else:
            self.name = name

        self.out = ConsolePrinter(prefix=self.name)
        self.debug = self.out.debug
