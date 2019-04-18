"""
Simple output for analysis and simulations

PYthon OUTput → P OUT Y → POUTY.

* Flexible console printing
* macOS systems notifications (terminal-notifier or AppleScript)
* Anybar widget color control
"""

from .console import log, debug, printf, ConsolePrinter
from .notifier import Notifier
from .anybar import AnyBar


__version__ = '0.1.2'


def debug_mode(active):
    """Activate or deactivate debug-mode logging."""
    console.DEBUG_MODE = bool(active)
