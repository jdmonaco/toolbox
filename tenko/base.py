"""
Base class for context components.
"""

from pouty.console import ConsolePrinter


class TenkoObject(object):

    """
    Provide auto-naming and basic output functionality.
    """

    __counts = {}

    def __init__(self, name=None, color=None, textcolor=None, **kwargs):
        self._initialized = False

        # Set color/textcolor defaults for TenkoObject instances
        color = 'green' if color is None else color
        textcolor = 'default' if textcolor is None else textcolor

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
            if c == 0:
                self.name = self.klass
            else:
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

        # Warn about unconsumed kwargs
        for key, value in kwargs.items():
            self.out(f'{key} = {value!r}', prefix='UnconsumedKwargs',
                     warning=True)

        self._initialized = True

    def __repr__(self):
        if hasattr(self, '__qualname__'):
            return self.__qualname__
        if hasattr(self, '__module__') and hasattr(self, '__name__'):
            return f'{self.__module__}{self.__class__.__name__}'
        return object.__repr__(self)

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)
