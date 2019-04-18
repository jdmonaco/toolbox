"""
Manage an HDF file as a data store.
"""

__all__ = ['DataStore', 'close_all']

import os
import sys
import time
import warnings
import subprocess

import tables as tb

import pouty

from .handles import TABLES_FILES


HOME = os.getenv('HOME')
if sys.platform == 'win32':
    HOME = os.getenv("USERPROFILE")


def close_all():
    """Close all HDF files and clear the cache."""
    for k in TABLES_FILES:
        TABLES_FILES[k].close()
    TABLES_FILES.clear()
    tb.file._open_files.close_all()


class DataStore(object):

    """
    Manage access to a data storage file (HDF).
    """

    def __init__(self, name='data', where=None, logfunc=None, quiet=False):
        self._name = name
        self._parent = where is None and os.getcwd() or os.path.abspath(where)
        self._backup_path = os.path.join(self._parent, 'backups')
        self._quiet = quiet

        self.out = logfunc and logfunc or pouty.log

        self._file = None
        self._path = os.path.join(self._parent, '%s.h5' % self._name)

    def _check_cache(self):
        try:
            self._file = TABLES_FILES[self._path]
        except KeyError:
            pass
        else:
            if not self._file.isopen:
                try:
                    del TABLES_FILES[self._path]
                except KeyError:
                    pass
                self._file = None

    @staticmethod
    def _truncate(p):
        if p.startswith(HOME):
            return f'~{p[len(HOME):]}'
        return p

    def __str__(self):
        self._check_cache()
        status = ''
        if self._file and self._file.isopen:
            mode = 'ro'
            if self._file.mode in 'aw':
                mode = 'w'
            status = ' (%s)' % mode
        return '%s%s' % (self._truncate(self._path), status)

    def __repr__(self):
        self._check_cache()
        status = 'Unopened'
        if self._file:
            if self._file.isopen:
                mode = 'Readonly'
                if self._file.mode in 'aw':
                    mode = 'Writeable'
                status = 'Open: %s' % mode
            else:
                status = 'Closed'
        return "<DataStoreFile(%s) at '%s'>" % (status, self._truncate(
            self._path))

    def path(self):
        return self._path

    def __enter__(self):
        """Open file within a context statement."""
        return self.get(readonly=False)

    def __exit__(self, etype, evalue, tb):
        """Close file at the end of a context."""
        self.close()

    def get(self, readonly=None):
        """Get handle to the data file.

        Keyword arguments:
        readonly -- set with boolean to force access mode
        """
        self._check_cache()
        if self._file and self._file.isopen:
            if self._check_mode(readonly):
                return self._file
            self.close()

        if readonly is None:
            readonly = os.path.isfile(self._path)

        self._open_file(readonly)
        return self._file

    def _check_mode(self, readonly):
        if readonly is None:
            return True
        return ((self._file.mode == 'r' and readonly) or
                (self._file.mode == 'a' and not readonly))

    def _open_file(self, readonly):
        """Open a new file handle to the data file."""
        if not os.path.isdir(self._parent):
            os.makedirs(self._parent)

        mode = readonly and 'r' or 'a'
        try:
            self._file = tb.open_file(self._path, mode=mode)
        except IOError:
            self.out('Error opening data store file', error=True)
        else:
            TABLES_FILES[self._path] = self._file
            if not self._quiet:
                self.out('Opened: {}', self)

    def flush(self):
        """Flush the data file."""
        self._check_cache()
        if self._file and self._file.isopen:
            self._file.flush()

    def close(self):
        """Close the data file if it's open."""
        self._check_cache()
        if not (self._file and self._file.isopen):
            return

        try:
            self._file.close()
        except IOError:
            self.out('Error closing data store file, try again', error=True)
        else:
            try:
                del TABLES_FILES[self._path]
            except KeyError:
                pass
            if not self._quiet:
                self.out('Closed: {}', self)

    def backup(self, tag=None):
        """Move the data file to a backup folder and create a clean copy
        in its place.
        """
        if not os.path.exists(self._path):
            self.out('Backup Error: File does not exist: {}', self._truncate(
                self._path), error=True)
            return

        label = tag is None and time.strftime('%Y-%m-%d-%H-%M') or tag
        filename = '%s-%s.h5' % (self._name, label)
        dest = os.path.join(self._backup_path, filename)
        if not os.path.isdir(self._backup_path):
            os.makedirs(self._backup_path)

        self.close()

        if subprocess.call(['mv', self._path, dest]) != 0:
            self.out('Backup Error: Failed to move {}', self._truncate(
                self._path), error=True)
            return

        warnings.filterwarnings('ignore', category=tb.NaturalNameWarning)

        with tb.open_file(dest, mode='r') as df:
            df.copy_file(self._path)

        if not self._quiet:
            self.out('Backup: {}', self._truncate(dest))
