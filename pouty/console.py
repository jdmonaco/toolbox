"""
Colorful bash output.
"""

import os
from sys import stdout, stderr, platform
from time import strftime

from .notifier import Notifier


DEBUG_MODE = False
WINDOWS = (platform == 'win32')


# Terminal color functions

def _color_format(code, text):
    return f'\033[{code}m{text}\033[0m'

def snow(s):      return _color_format("1;37", s)  # white
def white(s):     return _color_format("0;37", s)  # lightgray
def lightgray(s): return _color_format("1;36", s)  # lightcyan
def smoke(s):     return _color_format("1;36", s)  # lightcyan
def dimgray(s):   return _color_format("1;33", s)  # yellow
def gray(s):      return _color_format("1;30", s)  # gray
def cadetblue(s): return _color_format("1;34", s)  # lightblue
def seafoam(s):   return _color_format("1;32", s)  # lightgreen
def cyan(s):      return _color_format("0;36", s)  # cyan
def blue(s):      return _color_format("0;34", s)  # blue
def purple(s):    return _color_format("1;35", s)  # pink
def pink(s):      return _color_format("0;35", s)  # purple
def red(s):       return _color_format("0;31", s)  # red
def orange(s):    return _color_format("1;31", s)  # lightred
def yellow(s):    return _color_format("0;33", s)  # brown
def ochre(s):     return _color_format("0;33", s)  # brown
def green(s):     return _color_format("0;32", s)  # green


# Synonyms for backwards compatibility

_lightred   = orange
_lightgreen = seafoam
_brown      = yellow
_lightblue  = cadetblue
_lightcyan  = lightgray


# Color name-function mapping

COL_FUNC = {
    'snow'       : snow,
    'white'      : white,
    'lightgray'  : lightgray,
    'smoke'      : smoke,
    'dimgray'    : dimgray,
    'gray'       : gray,
    'cadetblue'  : cadetblue,
    'seafoam'    : seafoam,
    'cyan'       : cyan,
    'blue'       : blue,
    'purple'     : purple,
    'pink'       : pink,
    'red'        : red,
    'orange'     : orange,
    'yellow'     : yellow,
    'ochre'      : ochre,
    'green'      : green,
    'lightred'   : _lightred,
    'lightgreen' : _lightgreen,
    'brown'      : _brown,
    'lightblue'  : _lightblue,
    'lightcyan'  : _lightcyan
}

COLORS = list(COL_FUNC.keys())

def show_colors():
    for c in COLORS:
        print(COL_FUNC[c](' ' + c.ljust(11) + "\u25a0"*68))


class ConsolePrinter(object):

    """
    A callable console printer with color, log files, and notifications.
    """

    def __init__(self, prefix='[%Y-%m-%d %H:%M:%S] ', prefix_color='cyan',
        message_color='lightgray', quiet=False, outputfile=None,
        timestamp=True):
        """Create a colorful callable console printing object.

        Keyword arguments:
        prefix -- set the default prefix string for all messages
        prefix_color, message_color -- color names for prefix and message text
        quiet -- suppress messages about opening and closing log files
        outputfile -- optional file name where messages should be echo'd
        timestamp -- print timestamps in `outputfile` if specified
        """
        self._prefix = prefix
        self._quiet = quiet

        self.set_prefix_color(prefix_color)
        self.set_message_color(message_color)

        self._outputfile = self._fd = None
        self.set_outputfile(outputfile)
        self.set_timestamps(timestamp)

        self._notifier = None
        self._hanging = False

    def _isopen(self):
        return self._fd is not None and not self._fd.closed

    def openfile(self, newfile=False):
        """Open the currently set output file."""
        if self._outputfile is None:
            self('No output file has been set', error=True)
            return
        if self._isopen():
            return
        mode = 'w' if newfile else 'a'
        try:
            self._fd = open(self._outputfile, mode, 1)
        except IOError:
            self._outputfile = self._fd = None
            self('IOError: Could not open: %s' % self._outputfile, error=True)
        else:
            if not self._quiet:
                self(self._outputfile, prefix='OpenedFile')

    def set_prefix_color(self, color):
        """Set a new color for the prefix text."""
        if WINDOWS:
            self._pref = str
        else:
            self._pref = COL_FUNC[color]

    def set_message_color(self, color):
        """Set a new color for the message text."""
        if WINDOWS:
            self._msgf = str
        else:
            self._msgf = COL_FUNC[color]

    def set_outputfile(self, fpath, newfile=False):
        """Set the path to a new output file for logging messages."""
        if fpath == self._outputfile and self._isopen():
            return
        self.closefile()
        self._outputfile = fpath
        self._fd = None
        if self._outputfile is not None:
            self.openfile(newfile=newfile)

    def set_timestamps(self, active):
        """Set timestamping on/off for log files."""
        self._timestamp = bool(active)

    def closefile(self):
        """Close the current output file."""
        if not self._isopen():
            self._fd = None
            return
        self._fd.close()
        self._fd = None
        if not self._quiet:
            self(self._outputfile, prefix='ClosedFile')

    def removefile(self):
        """Delete the current output file."""
        self.closefile()
        os.unlink(self._outputfile)
        self('Deleted: %s' % self._outputfile)

    def __call__(self, *msg, **fmt):
        """Display a message with color prefix and multi-line indenting.

        Arguments:
        *msg -- a required string message with optional substitutions

        Keyword arguments:
        prefix -- override default prefix string
        hideprefix -- make prefix invisible but preserve indent
        debug -- specify where this is a debug message
        error -- specify whether this is an error message
        popup -- display the message with a system popup notification
        **fmt -- remaining kwargs provide formating substitutions
        """
        # Parse the keyword arguments
        prefix = fmt.pop('prefix', self._prefix)
        hideprefix = fmt.pop('hideprefix', False)
        debug = fmt.pop('debug', False)
        error = fmt.pop('error', False)
        warning = fmt.pop('warning', False)
        popup = fmt.pop('popup', False)

        # Construct the display prefix
        if debug:
            prefix = 'debug'
        elif '%' in prefix:
            prefix = strftime(prefix)
        pre = f'{prefix}: '
        if hideprefix:
            pre = ' '*len(prefix)

        # Construct the display message
        msg, args = msg[0], msg[1:]
        msg = msg.format(*args, **fmt)
        if error:
            msg = f'Error: {msg}'
        elif warning:
            msg = f'Warning: {msg}'

        # Console color print with prefix and indentation
        if error:
            pref = red
            msgf = red
            console = stderr
        elif warning:
            pref = orange
            msgf = orange
            console = stderr
        elif debug:
            pref = dimgray
            msgf = dimgray
            console = stderr
        else:
            pref = self._pref
            msgf = self._msgf
            console = stdout

        # Print the colored output to the console
        pre_len = len(pre) + 1
        lines = msg.split('\n')
        if self._hanging: self.newline()
        print(pref(pre) + msgf(lines[0].rstrip()), file=console)
        for line in lines[1:]:
            print(' ' * pre_len + msgf(line.rstrip()), file=console)

        # Timestamped file print if available
        if self._isopen():
            if self._timestamp:
                fmt = '%H:%M:%S %m-%d-%y'
                self._fd.write('[ %s ]  ' % strftime(fmt))
            if prefix != self._prefix and '%' not in prefix and not hideprefix:
                self._fd.write(pre)
            if error:
                self._fd.write('! -> Error: ')
            elif warning:
                self._fd.write('! -> Warning: ')
            self._fd.write(msg + '\n')
            self._fd.flush()

        # Request a system popup notification
        if popup:
            if self._notifier is None:
                self._notifier = Notifier(prog=self._prefix, echo=False)
            if error:
                self._notifier.notify(msg[7:], pre[:-2], 'Error')
            elif warning:
                self._notifier.notify(msg[9:], pre[:-2], 'Warning')
            else:
                self._notifier.notify(msg, pre[:-2])

    def box(self, filled=True, color=None):
        """Draw a Unicode box glyph to the console."""
        self.printf('\u25a1\u25a0'[filled], color=color)

    def hline(self, ch='—', length=80, color=None):
        """Print a horizontal rule line."""
        if self._hanging: self.newline()
        self.printf(ch * length, color=color)
        self.newline()

    def newline(self):
        """Insert a newline."""
        self.printf('\n')

    def printf(self, s, color=None):
        """Raw flushed color output to the console."""
        colf = self._pref if color is None else COL_FUNC[color]

        if WINDOWS: print(s, end='', flush=True)
        else:       print(colf(s), end='', flush=True)

        if self._isopen():
            self._fd.write(s)
            self._fd.flush()

        self._hanging = not s.endswith('\n')


# Convenience functions

Logger = ConsolePrinter(prefix='log')

def log(*args, **kwargs):
    Logger(*args, **kwargs)

def debug(*args, **kwargs):
    if DEBUG_MODE:
        kwargs.update(debug=True)
        Logger(*args, **kwargs)

def printf(s, c='green'):
    Logger.printf(s, color=c)
