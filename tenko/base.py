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
        super().__init__()

        # Set the class name to an instance attribute
        if hasattr(self.__class__, 'name'):
            self.klass = self.__class__.name
        else:
            self.klass = self.__class__.__name__

        # Set the instance name to a class-based count if unspecified
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            self.name = f'{self.klass}_{c:03d}'
        else:
            self.name = name

        # Add a ConsolePrinter instance attribute for printing
        if not hasattr(self, 'out'):
            self.out = ConsolePrinter(prefix=self.name, prefix_color=color,
                    message_color=textcolor)

        # Add a debug function
        if not hasattr(self, 'debug'):
            self.debug = self.out.debug

    def __repr__(self):
        return f'{self.klass}(name={self.name!r})'

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)
