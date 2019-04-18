"""
System notifications.
"""

import os
import sys
import time
import subprocess as sp

from .shell import Shell


class Notifier(object):

    def __init__(self, prog=None, echo=True):
        self.prog = prog is None and sys.argv[0] or prog
        self._echo = echo
        self.on()

    def on(self):
        self._notifier = Shell.which('terminal-notifier')
        self._osascript = Shell.which('osascript')

    def off(self):
        self._notifier = self._osascript = None

    def notify(self, msg, title=None, subtitle=None):
        if title is None:
            title = self.prog
        s = msg[0].upper() + msg[1:]
        if self._notifier:
            self._terminal_notify(s, title, subtitle)
        elif self._osascript:
            self._applescript_notify(s, title, subtitle)

    def remove(self):
        if self._notifier:
            self._terminal_notify_remove()

    def _terminal_notify(self, msg, title, subtitle):
        cmd = [self._notifier]
        cmd.extend(['-message', msg])
        cmd.extend(['-title', title])
        if subtitle:
            cmd.extend(['-subtitle', subtitle])
        cmd.extend(['-group', self.prog])
        with open(os.devnull, 'w') as null:
            if sp.call(cmd, stdout=null, stderr=null) != 0:
                print('Warning: Notifier (terminal-notifier) failed:', msg)

    def _terminal_notify_remove(self, tries=5, delay=1.0):
        cmd = [self._notifier]
        cmd.extend(['-remove', self.prog])
        with open(os.devnull, 'w') as null:
            while sp.check_output(cmd, stdout=null, stderr=null) and tries:
                time.sleep(delay)
                tries -= 1

    def _applescript_notify(self, msg, title, subtitle):
        cmd = 'display notification "%s" with title "%s"' % (msg, title)
        if subtitle:
            cmd += 'subtitle "%s"' % subtitle
        if sp.call([self._osascript, '-e', cmd]) != 0:
            print('Warning: Notifier (AppleScript) failed:', msg)
