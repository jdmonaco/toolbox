"""
Replication of some shell functions.
"""

import os
import sys
import time
import subprocess as subp


class Shell(object):

    @staticmethod
    def open_(appname, newinstance=False):
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
