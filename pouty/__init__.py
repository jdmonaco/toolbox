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


def debug_mode(active=None):
    """
    Activate/deactivate or simply query debug-mode logging.
    """
    if active is None:
        return console.DEBUG_MODE
    console.DEBUG_MODE = bool(active)
