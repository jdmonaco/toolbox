"""
Wrappers for some common shell functions.
"""

import os
import re
import sys
import time
import subprocess as subp


class Shell(object):

    @staticmethod
    def open_(appname, *, newinstance=False):
        """
        Open the MacOS application with the specified base name.
        """
        if sys.platform != 'darwin':
            return None
        apath = f'/Applications/{appname}.app'
        if not os.path.exists(apath):
            print('Shell.open_: application does not exist ({appname})',
                    file=sys.stderr)
            return None
        cmd = ['open', '-a', apath]
        if newinstance:
            cmd.insert(1, '-n')
        res = subp.run(cmd)
        if res.returncode == 0:
            return Shell.pgrep(appname, newest=True)
        return None

    @staticmethod
    def clamshell():
        """
        Return whether the Macbook lid is closed.
        """
        if Shell.which('ioreg') is None:
            return False
        apple_state = Shell.run('ioreg -r -k AppleClamshellState -d 4')
        clam_re = re.compile('\"AppleClamshellState\" = (\w*)$')
        for line in apple_state.split('\n'):
            match = clam_re.search(line)
            if match:
                yesno = str(match.groups()[0])
                if yesno.lower() == 'yes':
                    return True
                return False

    @staticmethod
    def hostname(*, short=True):
        """
        Get the hostname of the current machine.
        """
        opt = '-s' if short else '-f'
        return Shell.run('hostname', opt)

    @staticmethod
    def whoami():
        """
        Get the username of the current user.
        """
        return Shell.run('whoami')

    @staticmethod
    def run(prg, *args):
        """
        Run a shell command with args and return stdout as a string.
        """
        if ' ' in prg:
            prg = prg.split()
        elif type(prg) is str:
            prg = [prg]
        cmd = prg + list(args)
        res = subp.run(cmd, stdout=subp.PIPE)
        if res.returncode == 0:
            output = res.stdout.decode().strip()
            return output
        return ''

    @staticmethod
    def killall(progname):
        """
        Use the `killall` shell command to kill every process with the name.
        """
        killall = Shell.which('killall')
        if killall is None:
            print('Shell.killall: not on path', file=sys.stderr)
            return
        cmd = [killall, progname]
        res = subp.run(cmd)
        return res.returncode

    @staticmethod
    def pgrep(pattern, newest=False, oldest=False):
        """
        Run pgrep on the given pattern and return the process ids.
        """
        pgrep = Shell.which('pgrep')
        cmd = [pgrep]
        if newest:
            cmd.append('-n')
        elif oldest:
            cmd.append('-o')
        cmd.append(pattern)
        res = subp.run(cmd, stdout=subp.PIPE)
        if res.returncode == 0:
            if newest or oldest:
                return int(res.stdout.decode().strip())
            return list(map(int, res.stdout.decode().split()))
        return []

    @staticmethod
    def setenv(varname, val):
        """
        Set a shell environment variable to the specified value.
        """
        try:
            os.environ[varname] = str(val)
        except:
            print(f'Shell.setenv: could not set {varname}', file=sys.stderr)
            return 2
        return int(varname not in os.environ)

    @staticmethod
    def which(cmd, mode=os.F_OK | os.X_OK):
        """
        Find the named command on the executable search path.
        """
        def _access_check(fn, mode):
            return (os.path.exists(fn) and os.access(fn, mode)
                    and not os.path.isdir(fn))

        if os.path.dirname(cmd):
            if _access_check(cmd, mode):
                return cmd
            return None

        path = os.environ.get("PATH", os.defpath)
        if not path:
            return None
        path = path.split(os.pathsep)

        seen = set()
        for adir in path:
            normdir = os.path.normcase(adir)
            if not normdir in seen:
                seen.add(normdir)
                name = os.path.join(adir, cmd)
                if _access_check(name, mode):
                    return name
        return None
