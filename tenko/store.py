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

from roto.paths import tilde

from .handles import TABLES_FILES
from .base import TenkoObject


def close_all():
    """Close all HDF files and clear the cache."""
    for k in TABLES_FILES:
        TABLES_FILES[k].close()
    TABLES_FILES.clear()
    tb.file._open_files.close_all()


class DataStore(TenkoObject):

    """
    Manage access to a data storage file (HDF).
    """

    def __init__(self, name='data', where=None):
        super().__init__(name=name, color='ochre')
        self.parent = where is None and os.getcwd() or os.path.abspath(where)
        self.backup_path = os.path.join(self.parent, 'backups')

        self.h5file = None
        self.h5path = os.path.join(self.parent, f'{self.name}.h5')

    def _check_cache(self):
        try:
            self.h5file = TABLES_FILES[self.h5path]
        except KeyError:
            pass
        else:
            if not self.h5file.isopen:
                try:
                    del TABLES_FILES[self.h5path]
                except KeyError:
                    pass
                self.h5file = None

    def __str__(self):
        self._check_cache()
        status = ''
        if self.h5file and self.h5file.isopen:
            mode = 'ro'
            if self.h5file.mode in 'aw':
                mode = 'w'
            status = ' (%s)' % mode
        return '%s%s' % (tilde(self.h5path), status)

    def __repr__(self):
        self._check_cache()
        status = 'Unopened'
        if self.h5file:
            if self.h5file.isopen:
                mode = 'Readonly'
                if self.h5file.mode in 'aw':
                    mode = 'Writeable'
                status = 'Open: %s' % mode
            else:
                status = 'Closed'
        return "<DataStoreFile(%s) at '%s'>" % (status, tilde(self.h5path))

    def path(self):
        return self.h5path

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
        if self.h5file and self.h5file.isopen:
            if self._check_mode(readonly):
                return self.h5file
            self.close()

        if readonly is None:
            readonly = os.path.isfile(self.h5path)

        self._open_file(readonly)
        return self.h5file

    def _check_mode(self, readonly):
        if readonly is None:
            return True
        return ((self.h5file.mode == 'r' and readonly) or
                (self.h5file.mode == 'a' and not readonly))

    def _open_file(self, readonly):
        """Open a new file handle to the data file."""
        if not os.path.isdir(self.parent):
            os.makedirs(self.parent)

        mode = readonly and 'r' or 'a'
        try:
            self.h5file = tb.open_file(self.h5path, mode=mode)
        except IOError:
            self.out(self.h5path, prefix='DataFileIOError', error=True)
        else:
            TABLES_FILES[self.h5path] = self.h5file

    def flush(self):
        """Flush the data file."""
        self._check_cache()
        if self.h5file and self.h5file.isopen:
            self.h5file.flush()

    def close(self):
        """Close the data file if it's open."""
        self._check_cache()
        if not (self.h5file and self.h5file.isopen):
            return

        try:
            self.h5file.close()
        except IOError:
            self.out(self.h5path, prefix='DataFileIOError', error=True)
        else:
            try:
                del TABLES_FILES[self.h5path]
            except KeyError:
                pass

    def backup(self, tag=None):
        """Move the data file to a backup folder and create a clean copy
        in its place.
        """
        if not os.path.exists(self.h5path):
            self.out(self.h5path, prefix='MissingDataFile', error=True)
            return

        label = tag is None and time.strftime('%Y-%m-%d-%H-%M') or tag
        filename = '%s-%s.h5' % (self.name, label)
        dest = os.path.join(self.backup_path, filename)
        if not os.path.isdir(self.backup_path):
            os.makedirs(self.backup_path)

        self.close()

        if subprocess.call(['mv', self.h5path, dest]) != 0:
            self.out(self.h5path, prefix='BackupMoveFailed', error=True)
            return

        warnings.filterwarnings('ignore', category=tb.NaturalNameWarning)

        with tb.open_file(dest, mode='r') as df:
            df.copy_file(self.h5path)

        self.out(dest, prefix='DatafileBackedUp')
