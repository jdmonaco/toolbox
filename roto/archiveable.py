"""
Archive/restore persistence for objects containing numerical arrays.
"""

from os import path, unlink, getcwd, chdir, rmdir, makedirs
from sys import stderr
import pickle
import tarfile

import numpy as np


ARRAYS_FN = 'arrays.npz'
PICKLE_FN = 'object.pickle'


def load(filename):
    """Return the archived ArrayContainer object from the specified archive"""
    return ArchiveableObject.restore(filename)


class ArchiveableObject(object):

    """
    Base class provides methods for subclasses containing large numerical
    arrays as attributes.

    Available methods:
    tofile -- instance methods that saves an array container object and
        its numerical arrays to an efficient archive data file
    fromfile -- class method that loads a previously saved array
        container object from an array container archive
    """

    def archive(self, filename):
        """Store array container object in a compressed archive file specified
        by the filename argument (relative to current directory).
        """
        # Validate archive filename
        if not filename.endswith('.tar.gz'):
            if filename[-1] != '.':
                filename += '.'
            filename += 'tar.gz'
        if path.exists(filename):
            stderr.write('Error: Save file %s already exists'%filename)
            return

        # Create a temp directory for component files
        file_path = path.abspath(filename)
        data_path = path.split(file_path)[0]
        tmpdir = path.join(data_path, '_tmp_%06d'%np.random.randint(1000000))
        while path.exists(tmpdir):
            tmpdir = path.join(data_path, '_tmp_%06d'%np.random.randint(1000000))

        # Change to new temp directory
        cwd = getcwd()
        makedirs(tmpdir)
        chdir(tmpdir)

        # Get attributes list for this object
        attr_list = [x for x in dir(self) if not x.startswith('_')]

        # Create dict of numpy arrays in this ratemap object
        map_arrays = {}
        for attr in attr_list:
            value = getattr(self, attr)
            if type(value) is np.ndarray:
                map_arrays[attr] = value
                setattr(self, attr, np.array([])) # empty array for pickling

        # Save all numpy arrays to NPZ file
        try:
            np.savez(ARRAYS_FN, **map_arrays)
        except IOError as e:
            raise IOError('failed to write arrays file')

        # Pickle the rest of this object
        try:
            fd = file(PICKLE_FN, 'w')
            pickle.dump(self, fd)
        except IOError:
            raise IOError('failed to write pickle file')
        finally:
            fd.close()

        # Archive the array and pickle files
        tar = tarfile.open(name=file_path, mode='w:gz')
        tar.add(ARRAYS_FN)
        tar.add(PICKLE_FN)
        tar.close()
        unlink(ARRAYS_FN)
        unlink(PICKLE_FN)

        # Repopulate the object with its array data
        for arr in map_arrays:
            setattr(self, arr, map_arrays[arr])

        # Remove the temp directory
        chdir(cwd)
        try:
            rmdir(tmpdir)
        except IOError:
            raise IOError('could not remove tempdir:\n%s'%tmpdir)

    @classmethod
    def restore(cls, filename):
        """Retrieve a stored array container object from the data stored in the
        file specified by filename.
        """
        # Validate filename
        if not path.isfile(filename):
            raise TypeError('bad file name for saved data')

        # Get full paths and temp dir path
        file_path = path.abspath(filename)
        data_path = path.split(file_path)[0]
        tmpdir = path.join(data_path, '_tmp_%06d'%np.random.randint(1000000))
        while path.exists(tmpdir):
            tmpdir = path.join(data_path, '_tmp_%06d'%np.random.randint(1000000))

        # Create and change to temp dir
        cwd = getcwd()
        makedirs(tmpdir)
        chdir(tmpdir)

        # Extract the tar/gzip archive to the temp dir
        try:
            tar = tarfile.open(name=file_path, mode='r:*')
        except:
            raise ValueError('failed to open archive file for reading')
        else:
            tar.extractall(path=tmpdir)
            tar.close()

        # Verify that the right files exist
        if not (path.exists(ARRAYS_FN) and path.exists(PICKLE_FN)):
            raise ValueError('invalid or corrupted archive file')

        # Load the array and object data
        array_data = np.load(ARRAYS_FN)
        map_object = np.load(PICKLE_FN)
        unlink(ARRAYS_FN)
        unlink(PICKLE_FN)

        # Repopulate the object with its array data
        for arr in array_data.files:
            setattr(map_object, arr, array_data[arr])

        # Clean up and return loaded object
        chdir(cwd)
        try:
            rmdir(tmpdir)
        except IOError:
            raise IOError('could not remove tempdir:\n%s'%tmpdir)
        return map_object
