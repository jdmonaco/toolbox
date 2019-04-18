"""
Set the color of your AnyBar widget.
"""

import os
import sys
import socket
import subprocess

from .shell import Shell


INITIAL_PORT = 1738
COLORS = ( 'white', 'red', 'orange', 'yellow', 'green', 'cyan',
           'blue', 'purple', 'black', 'question', 'exclamation' )


_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


class AnyBar(object):

    """
    An instance of the AnyBar menubar display application.
    """

    _instances = []

    def __init__(self, color=None, port=None, pid=None):
        self.socket = _socket
        self.color = 'white' if color is None else color
        self.port = port
        self.pid = pid
        self.start()

    def __str__(self):
        return f'AnyBar(color=\'{self.color}\', port={self.port}, ' \
               f'pid={self.pid})'

    def __repr__(self):
        return str(self)

    def start(self):
        """
        Start a new AnyBar instance.
        """
        already = False
        running = Shell.pgrep('AnyBar')
        if self.pid is not None:
            if self.pid in running:
                if self.port is None:
                    self.quit()
                else:
                    already = True
                    print(f'AnyBar: already running ({self.pid})',
                            file=sys.stderr)
            else:
                self.pid = self.port = None

        # Set the port based on number of currently running AnyBars
        if not already and self.port is None:
            self.port = INITIAL_PORT + len(running)

        # Start a new instance of the AnyBar application
        if self.pid is None:
            if Shell.setenv('ANYBAR_PORT', self.port) == 0:
                self.pid = Shell.open_('AnyBar', newinstance=True)
                if self.pid is not None:
                    self.__class__._instances.append(self)
            else:
                print(f'AnyBar: unable to set port ({self.port})',
                        file=sys.stderr)

    def quit(self):
        """
        Quit the AnyBar instance.
        """
        res = subprocess.run(['kill', '-HUP', str(self.pid)])
        if res.returncode == 0:
            self.pid = None
            self.port = None
        else:
            print(f'AnyBar: could not quit instance ({self.pid})',
                    file=sys.stderr)
        self.__class__._instances.remove(self)

    @classmethod
    def quit_all(cls):
        """
        Quit all running instances of AnyBar.
        """
        insts = cls._instances.copy()
        [abar.quit() for abar in reversed(insts)]
        if len(cls._instances):
            print('AnyBar: could not quit all ({",".join(cls._instances)})',
                    file=sys.stderr)

    def set_color(self, color=None):
        """
        Set the color of the AnyBar widget.

        Valid colors: {}.
        """
        color = self.color if color is None else color
        if self.port is None or self.pid is None:
            print('AnyBar: not running', file=sys.stderr)
            return
        if color not in COLORS:
            raise ValueError("Invalid color: {}".format(color))
        self.socket.sendto(color.encode('utf-8'), ('localhost', self.port))
        self.color = color

    set_color.__doc__ = set_color.__doc__.format(', '.join(["\'%s\'" % c
        for c in COLORS]))
