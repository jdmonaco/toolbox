"""
Simple output for analysis and simulations

PYthon OUTput → P OUT Y → POUTY.

* Flexible console printing
* macOS systems notifications (terminal-notifier or AppleScript)
* Anybar widget color control
"""

from .console import log, debug, printf, box, hline, ConsolePrinter
from .notifier import Notifier
from .anybar import AnyBar


__version__ = '0.1.2'


def quiet_mode(newmode=None):
    """
    Set or toggle quiet mode.
    """
    newmode = not console.QUIET_MODE if newmode is None else bool(newmode)
    console.QUIET_MODE = newmode

def debug_mode(newmode=None):
    """
    Set or toggle debug mode.
    """
    newmode = not console.DEBUG_MODE if newmode is None else bool(newmode)
    console.DEBUG_MODE = newmode