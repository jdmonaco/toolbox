"""
Managed context for data analysis runs.
"""

import os
import pdb
import sys
import time
try:
    import simplejson as json
except ImportError:
    import json
import inspect
import datetime
import subprocess
from importlib import import_module
from decorator import decorator
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import pandas as pd

from toolbox import HOME, PROJDIR
from pouty import AnyBar, ConsolePrinter
from roto import data, datapath as tpath
from roto.figures import get_svg_figinfo
from roto.paths import uniquify, tilde
from roto.strings import snake2title, slugify, naturalize
from roto.dicts import AttrDict, merge_two_dicts

from . import parallel
from .repo import git_revision
from .store import DataStore


CALLFILE = 'call.log'
INITFILE = 'init.json'
ENVFILE = 'env.json'


@decorator
def step(_f_, *args, **kwargs):
    """Declare a method as a compute step in this context."""
    self = args[0]
    status = { 'OK': False }
    self._step_enter(_f_, args, kwargs)
    res = self._step_execute(_f_, args, kwargs, status)
    self._step_exit(_f_, args, kwargs, status)
    return res


class AbstractBaseContext(object):

    """
    Smart context for open and reproducible data analysis.

    Class methods:
    load -- create a new context object for a previous run

    Instance variables:
    c -- attribute access to the context namespace

    Instance methods:
    mkdir -- make a subdirectory
    reveal -- open the run directory in Finder (OS X)
    out -- send messages to console, logs, notifications, and anybar
    open_logfile -- open a new log file
    use_timestamps -- toggle timestamping for log file messages
    close_logfile -- close the most recent log file
    set_parallel_profile -- set ipython profile for the parallel client
    set_datafile -- change the datafile path
    get_datafile -- open the datafile, creating it if necessary
    flush_datafile -- flush the datafile
    close_datafile -- close the datafile
    create_group -- create a group in the datafile
    create_table -- create a table in the datafile
    save_array -- save a numpy array or matrix to the datafile
    save_simulation -- save Brian simulation output to the datafile
    read_simulation -- load simulation output into DataFrames
    set_static_figures -- set/toggle whether figure windows are reused
    figure -- open a new figure (wrapper for `plt.figure(...)`)
    savefig -- save a figure (default: most recent)
    closefig -- close a figure (default: most recent)
    save_figures -- save all open figure windows
    close_figures -- close all open figure windows
    set_figfmt -- change the image format for saving figures
    """

    @classmethod
    def factory(ABC, classname, projname, version, rootdir, datadir, repodir,
        resdir, logcolor='pink'):
        class _Class(ABC):
            pass
        _Class.__name__ = classname
        _Class.__doc__ = ABC.__doc__
        _Class.projname = projname
        _Class.version = version
        _Class.repodir = repodir
        _Class.rootdir = rootdir
        _Class.datadir = datadir
        _Class.resdir = resdir
        _Class.logcolor = logcolor
        return _Class

    def _arg(self, name, value, dflt=None, norm=False, path=False,
        optional=False):
        """
        Return non-null keyword values or the class attribute-based default.
        """
        if value is not None:
            if path:
                return os.path.abspath(value)
            if norm:
                return slugify(value)
            return value
        if dflt is not None:
            if path:
                return os.path.abspath(dflt)
            if norm:
                return slugify(dflt)
            return dflt
        if name is not None and hasattr(self.__class__, name):
            cls_dflt = getattr(self.__class__, name)
            if cls_dflt is not None:
                if path:
                    return os.path.abspath(cls_dflt)
                if norm:
                    return slugify(cls_dflt)
                return cls_dflt
        if not optional:
            print(f'Warning: missing value for \'{name}\'', file=sys.stderr)
        return None

    def __init__(self, desc=None, tag=None, projname=None, version=None,
        repodir=None, rootdir=None, datadir=None, resdir=None, regdir=None,
        moduledir=None, h5file=None, ctxdir=None, admindir=None, tmpdir=None,
        rundir=None, profile=None, logcolor=None, figfmt=None, staticfigs=None,
        quiet=None):
        """Set up the analysis context.

        Keyword arguments:
        desc -- short phrase describing the run
        rundir -- path to folder that will contain run output
        rootdir -- parent folder for all analysis output
        resdir -- parent folder for results
        h5file -- specify an alternate path for the datafile
        figfmt -- default figure format for saving images (mpl is equivalent
                  to using the rcParams['savefig.format'] setting)
        staticfigs -- reuse figure windows for labeled figures
        logcolor -- prefix color for shell log messages (default: purple)
        profile -- ipython profile to use for parallel client
        """
        self._name = self._arg('__name__', None, norm=True)
        self._desc = self._arg('desc', desc, norm=True, optional=True)
        self._tag = self._arg('tag', tag, norm=True, optional=True)
        self._projname = self._arg('projname', projname, norm=True)
        self._version = self._arg('version', version, norm=True)
        self._profile = self._arg('profile', profile, dflt=self._projname)
        self._logcolor = self._arg('logcolor', logcolor, norm=True)
        self._figfmt = self._arg('figfmt', figfmt, dflt='mpl')
        self._staticfigs = self._arg('staticfigs', staticfigs, dflt=True)
        self._quiet = self._arg('quiet', quiet, dflt=True)

        self._repodir = self._arg('repodir', repodir, path=True)
        self._rootdir = self._arg('rootdir', rootdir, dflt=os.path.join(
            PROJDIR, self._projname or 'tenko'), path=True)
        self._datadir = self._arg('datadir', datadir, dflt=os.path.join(
            self._rootdir, 'data'))
        self._resdir = self._arg('resdir', resdir, dflt=os.path.join(
            self._rootdir, 'results'))
        self._regdir = self._arg('regdir', regdir, path=True, optional=True)
        self._moduledir = self._arg('moduledir', moduledir, dflt=os.path.join(
            self._rootdir, self.__class__.__module__.split('.')[-1]))
        self._h5file = self._arg('h5file', h5file, dflt=os.path.join(
            self._moduledir, f'{self._name}.h5'))

        ctxdflt = os.path.join(self._moduledir, self._version)
        if self._desc is not None: ctxdflt += f'-{self._desc}'
        if self._tag is not None: ctxdflt += f'+{self._tag}'
        self._ctxdir = self._arg('ctxdir', ctxdir, dflt=ctxdflt)

        self._admindir = self._arg('admindir', admindir, dflt=os.path.join(
            self._ctxdir, 'admin'))
        self._tmpdir = self._arg('tmpdir', tmpdir, dflt=os.path.join(
            self._ctxdir, 'temp'))
        self._rundir = self._arg('rundir', rundir, dflt=self._tmpdir)

        # Check for existing context data and load it if available
        loaded = False
        if os.path.isdir(self._admindir):
            if os.path.isfile(os.path.join(self._admindir, INITFILE)):
                self.load(self, load_instance=True)
                loaded = True

        # Create the console output object
        self._out = ConsolePrinter(prefix=self.__class__.__name__,
                quiet=self._quiet, prefix_color=self._logcolor)
        if loaded: self.hline()

        # Create the data store object for the HDF data file
        self._datafile = None
        self._set_datafile(self._h5file)

        # Set the ipyparallel profile
        self.set_parallel_profile(self._profile)

        # Load the persistent namespace with attribute access
        self._load_env()
        self.c = AttrDict(self._env)

        # Context state variables
        self._figures = {}
        self._savefig = {}
        self._lastfig = None
        self._savefig_path = None
        self._holdfigs = False
        self._lastcall = None
        self._running = False
        self._anybar = None

        # Finished initializing!
        self._save()
        self.printf(f'{self}', color=self._logcolor)

    def __str__(self):
        col_w = 13
        s = ['Class:'.ljust(col_w) + self.__class__.__name__]
        s += ['Module:'.ljust(col_w) + self.__class__.__module__]
        if self._desc:
            s += ['Description:'.ljust(col_w) + f"'{self._desc}'"]
        if self._tag:
            s += ['Tag:'.ljust(col_w) + f"'{self._tag}'"]
        s += ['ProjectDir:'.ljust(col_w) + tilde(self._rootdir)]
        s += ['DataDir:'.ljust(col_w) + tilde(self._datadir)]
        s += ['ResultsDir:'.ljust(col_w) + tilde(self._resdir)]
        if self._regdir:
            s += ['RegDir:'.ljust(col_w) + tilde(self._regdir)]
        s += ['ModuleDir:'.ljust(col_w) + tilde(self._moduledir)]
        s += ['Datafile:'.ljust(col_w) + tilde(self._datafile.path())]
        s += ['ContextDir:'.ljust(col_w) + tilde(self._ctxdir)]
        env_keys = self.c.__odict__.keys()
        if env_keys:
            s += ['EnvKeys:'.ljust(col_w) + ', '.join(env_keys)]
        return '\n'.join(s) + '\n'

    # Mapping methods

    def __setitem__(self, key, item):
        if isinstance(key, str):
            if not key.isidentifier():
                raise ValueError('key is not a valid name: {}'.format(key))
        else:
            raise ValueError('key is not a string: {}'.format(key))
        self._env[key] = item

    def __getitem__(self, key):
        return self._env[key]

    def __delitem__(self, key):
        del self._env[key]

    def __contains__(self, key):
        return key in self._env

    def __len__(self):
        return len(self._env)

    def __iter__(self):
        return iter(self._env)

    def _save_env(self):
        envpath = os.path.join(self._admindir, ENVFILE)
        if not os.path.exists(self._admindir):
            os.makedirs(self._admindir)
        with open(envpath, 'w') as f:
            json.dump(self._env, f, skipkeys=True, indent=2, sort_keys=False)

    def _load_env(self):
        sfn = os.path.join(self._admindir, ENVFILE)
        if not os.path.isfile(sfn): return
        with open(sfn, 'r') as f:
            self._env = json.load(f)

    # Load/save methods

    @classmethod
    def load(cls, instance_or_context_dir, load_instance=False):
        """Return a new context for a previous run directory."""
        out = ConsolePrinter(prefix="{}Loader".format(cls.__name__),
                prefix_color='brown')

        if hasattr(instance_or_context_dir, '_admindir'):
            inst = instance_or_context_dir
            admindir = instance_or_context_dir._admindir
        elif type(instance_or_context_dir) is str:
            inst = None
            admindir = os.path.join(instance_or_context_dir, 'admin')
        else:
            out('Requires a context object or admin path',
                    prefix='BadArgument', error=True)
            return

        initpath = os.path.join(admindir, INITFILE)
        if not os.path.isfile(initpath):
            out(initpath, prefix='MissingFile', error=True)
            return

        try:
            with open(initpath, 'r') as fd:
                initargs = json.load(fd)
        except ValueError:
            out(initpath, prefix='InvalidJSON', error=True)
            return
        else:
            out(os.path.split(admindir)[0], prefix='LoadedContext')

        if load_instance and inst is not None:
            for k, v in initargs.items():
                setattr(inst, f'_{k}', v)
            return inst
        return cls(**initargs)

    def _save(self):
        """Save the constructor parameters for this object."""
        initpath = os.path.join(self._admindir, INITFILE)
        if not os.path.exists(self._admindir):
            os.makedirs(self._admindir)

        with open(initpath, 'w') as fd:
            json.dump({
                'desc'       : self._desc,
                'tag'        : self._tag,
                'projname'   : self._projname,
                'version'    : self._version,
                'repodir'    : self._repodir,
                'rootdir'    : self._rootdir,
                'datadir'    : self._datadir,
                'resdir'     : self._resdir,
                'regdir'     : self._regdir,
                'moduledir'  : self._moduledir,
                'h5file'     : self._datafile.path(),
                'ctxdir'     : self._ctxdir,
                'admindir'   : self._admindir,
                'tmpdir'     : self._tmpdir,
                'rundir'     : self._rundir,
                'profile'    : self._profile,
                'logcolor'   : self._logcolor,
                'figfmt'     : self._figfmt,
                'staticfigs' : self._staticfigs,
                'quiet'      : self._quiet
            }, fd, indent=2, sort_keys=False)

    def register(self):
        """Link this context into the results directory."""
        self.close_logfile()
        self.close_datafile()

        # Register as: <version>[-<desc>]+<class-name>[+<tag>]
        resdest = os.path.join(self._resdir, self._version)
        if self._desc:
            resdest += f'-{self._desc}'
        resdest += f'+{self._name}'
        if self._tag:
            resdest += f'+{self._tag}'

        if os.path.isdir(resdest):
            self.out('Link already exists: {}', resdest,
                    error=True)
            return

        if not os.path.isdir(self._resdir):
            os.makedirs(self._resdir)

        try:
            os.symlink(self._ctxdir, resdest, target_is_directory=True)
        except IOError:
            self.out(resdest, prefix='FailedLink', error=True)
            return

        # Hard link the data file into the context directory
        if self._datafile:
            parent, name = os.path.split(self._datafile.path())
            if os.path.abspath(parent) == self._moduledir:
                new_h5file = os.path.join(self._ctxdir, name)
                os.link(self._h5file, new_h5file)
                self.set_datafile(new_h5file)

        # Set/save the registration path
        self._regdir = resdest
        self._save()

        self.out(self._regdir, prefix='Registration')

    # Run directory path methods

    def path(self, *rpath):
        """Get an absolute path in the run directory."""
        if not os.path.exists(self._rundir):
            os.makedirs(self._rundir)
        return os.path.join(self._rundir, *rpath)

    def mkdir(self, *rpath):
        """Create a subdirectory within the run directory."""
        dpath = self.path(*rpath)
        if os.path.isdir(dpath):
            return dpath
        os.makedirs(dpath)
        return dpath

    def subfolder(self, *rpath, prefix=False):
        """Make a unique subfolder under the analysis directory."""
        stem = self.path(*rpath)
        if prefix:
            path = uniquify(stem, fmt='%02d-%s', reverse_fmt=True)
        else:
            path = uniquify(stem, fmt='%s-%02d')
        subf = rpath[:-1] + (os.path.split(path)[1],)
        return self.mkdir(*subf)


    # Console output methods

    def out(self, *args, **kwargs):
        """Display and log an output message.

        Arguments (except `anybar`) are passed to `ConsolePrinter`.

        Keyword arguments:
        anybar -- optional, color name to set the AnyBar widget color
        """
        color = kwargs.pop('anybar', None)
        if color is not None:
            self.set_anybar_color(color)
        self._out(*args, **kwargs)

    def printf(self, *args, **kwargs):
        """Send characters to stdout."""
        self._out.printf(*args, **kwargs)

    def box(self, filled=True, color=None):
        self._out.box(filled=filled, color=color)

    def newline(self):
        self._out.newline()

    def hline(self, color='white'):
        self._out.hline(color=color)

    def start_anybar(self, color='white'):
        """Create an AnyBar instance for controlling an AnyBar widget."""
        if self._anybar is not None: return
        ab = AnyBar()
        if ab.pid:
            self._anybar = ab
            self.set_anybar_color(color)

    def set_anybar_color(self, color):
        """If there is an active AnyBar widget, set its color."""
        if self._anybar is None: return
        self._anybar.set_color(color)

    # Logging methods

    def open_logfile(self, stem=None, newfile=False, timestamps=True):
        """Start a new log file with optional timestamping."""
        if stem is None:
            if self._running:
                stem = self._lastcall['step']
            else:
                stem = self._name
        fn = self.path('%s.log' % stem)
        self.use_timestamps(timestamps)
        self._out.set_outputfile(fn, newfile=newfile)
        return fn

    def use_timestamps(self, active):
        """Timestamp messages recorded in log files."""
        self._out.set_timestamps(active)

    def close_logfile(self):
        """Close the current log file."""
        self._out.closefile()

    # Step wrapping methods

    def _step_enter(self, method, args, kwargs):
        spec = inspect.getargspec(method)

        argnames = spec.args[1:]
        argvalues = args[1:]
        params = list(zip(argnames, argvalues))

        tag = None
        for name, value in params:
            if name == 'tag':
                tag = value
                break

        self._lastcall = info = {
            'time': time.localtime(),
            'revision': git_revision(self._repodir),
            'subclass': self.__class__.__name__,
            'step': method.__name__,
            'tag': tag,
            'params': params,
            'defaults': spec.defaults,
            'kwname': spec.keywords,
            'kwvalues': kwargs
        }

        # Run directory is the temp directory during execution
        self._rundir = self._tmpdir
        if not os.path.exists(self._tmpdir):
            os.makedirs(self._tmpdir)

        # Clear out the temp directory for the run
        tmplist = os.listdir(self._tmpdir)
        for fn in tmplist:
            path = os.path.join(self._tmpdir, fn)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                p = subprocess.run(['rm', '-rf', path])
                if p.returncode != 0:
                    self.out(path, prefix='ProblemRemoving', error=True)

        # Start the AnyBar widget if available
        self.start_anybar()

        # Save any pre-run changes to the key-value store
        self._save_env()

        self.open_logfile(info['step'].replace('_', '-'), timestamps=False)
        self.hline()
        self.out('Running step: {}', info['step'], popup=True)
        if params:
            self.out('Call parameters:\n{}',
                     self._format_params(params, spec.defaults))
        if kwargs:
            self.out('Keywords (\'{}\'):\n{}',
                     info['kwname'], self._format_keywords(kwargs))
        self.hline()

    def _format_params(self, params, defaults, bullet='-'):
        plist = []
        for i, (name, value) in enumerate(reversed(params)):
            if type(value) is pd.DataFrame:
                repr_value = 'DataFrame{}'.format(value.shape)
            else:
                repr_value = repr(value)
            plist.append('{} {} = {}'.format(bullet, name, repr_value))
            if defaults is not None and i < len(defaults):
                dflt = defaults[-1-i]
                if value != dflt:
                    plist[-1] += ' [default: {}]'.format(repr(dflt))
        return '\n'.join(reversed(plist))

    def _format_keywords(self, kwds, bullet='-'):
        return '\n'.join(['{} {} = {}'.format(bullet, k, repr(v))
                          for k,v in kwds.items()])

    def _step_execute(self, method, args, kwargs, status):
        status['OK'] = True
        self._running = True
        self.set_anybar_color('orange')  # to indicate running

        # Save figure and interactive state to go non-interactive during call
        prevfigset = frozenset(self._figures.keys())
        was_interactive = plt.isinteractive()
        if was_interactive:
            plt.ioff()

        try:
            result = method(*args, **kwargs)
        except Exception as e:
            status['OK'] = False
            result = None
            self.out('Exception in {}:\n{}: {}',
                     method.__name__, e.__class__.__name__, str(e),
                     error=True, popup=True, anybar='exclamation')
            pdb.post_mortem(sys.exc_info()[2])
        else:
            self.set_anybar_color('green')
        finally:
            self.close_datafile()

            # Show any new figures and restore interactive state
            curfigset = frozenset(self._figures.keys())
            if was_interactive:
                plt.ion()
                if prevfigset.symmetric_difference(curfigset):
                    plt.show()
                if len(plt.get_figlabels()):
                    plt.draw()

            # Restore matplotlib configuration if it was changed locally
            mpl.rc_file_defaults()

        return result

    def  _step_exit(self, method, args, kwargs, status):
        step = method.__name__
        tag = self._lastcall['tag']

        if status['OK']:
            self._save_env()
            self._save_call_log()

            # Copy the python module file to the run directory
            pyfile = import_module(self.__class__.__module__).__file__
            p = subprocess.run(['cp', pyfile, self._tmpdir])
            pyfile_copied = p.returncode == 0
            if not pyfile_copied:
                self.out(pyfile, prefix='CopyFailed', warning=True)
            _, basepy = os.path.split(pyfile)
            prevpyfile = None

            # Final output (run) directory is based on method name & tag
            self._rundir = os.path.join(self._ctxdir, step)
            if tag: self._rundir += '+{}'.format(slugify(tag))
            if not os.path.exists(self._rundir):
                os.makedirs(self._rundir)

            # Move any previous files to a unique 'history' subfolder
            runlist = list(filter(lambda x: x != 'history',
                os.listdir(self._rundir)))
            if len(runlist):
                histdir = uniquify(self.mkdir('history'), fmt=os.path.join(
                    '%s', '%02d'))
                os.makedirs(histdir)
                for fn in runlist:
                    runpath = os.path.join(self._rundir, fn)
                    histpath = os.path.join(histdir, fn)
                    os.rename(runpath, histpath)
                    if fn == basepy:
                        prevpyfile = histpath

                self.out(f'Moved {len(runlist)} items to ' +
                         histdir, prefix='FileBackup')

            # Generate python module diff file
            if pyfile_copied and prevpyfile is not None:
                diffpath = os.path.join(self._tmpdir, '{}.diff'.format(basepy))
                os.system(' '.join(['diff', '-w', prevpyfile, pyfile,
                        '>"{}"'.format(diffpath)]))

            # Move all the current (temp) output files to the run directory
            tmplist = os.listdir(self._tmpdir)
            for fn in tmplist:
                os.rename(os.path.join(self._tmpdir, fn),
                          os.path.join(self._rundir, fn))

            self.out(self._rundir, prefix='OutputDir')
            self._save()

            self.hline()
            dt = time.time() - time.mktime(self._lastcall['time'])
            self.out(f'Finished \'{step}\' in {dt:.1f} seconds.', popup=dt>60)

        self.close_logfile()
        self._running = False
        self._anybar.quit()
        self._anybar = None

        if not status['OK']:
            raise RuntimeError('Stopping due to exception in {}'.format(step))

    def _save_call_log(self):
        logfn = self.path(CALLFILE)
        history = []
        if os.path.isfile(logfn):
            with open(logfn, 'r') as logfd:
                history = logfd.readlines()
        call = self._lastcall
        with open(logfn, 'w') as logfd:
            logfd.write('Time: {}\n'.format(time.strftime('%c', call['time'])))
            signature = ', '.join(['%s=%s' % (k,repr(v))
                for k,v in call['params']])
            if call['kwvalues']:
                if call['params']:
                    signature += ', '
                signature += ', '.join(['%s=%s' % (k,repr(v))
                    for k,v in call['kwvalues'].items()])
            logfd.write('Call: {}.{}({})\n'.format(call['subclass'],
                call['step'], signature))
            logfd.write('Revision: {}\n\n'.format(call['revision']))
            logfd.writelines(history)

    # Parallel methods

    def set_parallel_profile(self, profile):
        """Set the ipython profile to use for the parallel client."""
        if profile:
            parallel.set_default_profile(profile)

    def get_parallel_client(self, profile=None):
        return parallel.client(profile)

    def close_parallel_client(self):
        parallel.close()

    # Datafile methods

    def _set_datafile(self, newpath):
        """Set a new datafile path."""
        if self._datafile:
            self._datafile.close()

        dpath = os.path.abspath(newpath)
        parent, fn = os.path.split(os.path.splitext(dpath)[0])
        self._datafile = DataStore(name=fn, where=parent, logfunc=self._out,
                quiet=self._quiet)
        self._h5file = self._datafile.path()

    def get_datafile(self, readonly=None):
        """Get a handle to the HDF data file for this analysis."""
        return self._datafile.get(readonly=readonly)

    def flush_datafile(self):
        """Flush the data store file to disk."""
        self._datafile.flush()

    def close_datafile(self):
        """Close the data store file."""
        self._datafile.close()

    def backup_datafile(self, tag=None):
        """Backup the data file and create a clean active copy."""
        self._datafile.backup()

    def datapath(self, *path, version=None, desc=None, classtag=None,
        step=None, tag=None, root=None):
        """An HDF data path anchored to a versioned & run-tagged root group."""
        if root is not None:
            root = tpath.join('/', root)
            return tpath.join(root, *path)

        # Get step & tag from last run if available
        _last_step = _last_tag = None
        if self._lastcall:
            _last_step = self._lastcall['step']
            _last_tag = self._lastcall['tag']

        # Select given or current values of path components
        Vers = version or self._version
        Desc = desc or self._desc
        Step = step or _last_step
        Ctag = classtag or self._tag
        Rtag = tag or _last_tag

        assert Step is not None, "missing step name for data"

        # Construct the path components
        base = 'v{}'.format(naturalize(Vers))
        if Desc: base += '__{}'.format(naturalize(Desc))
        run = naturalize(Step)
        if Ctag: run += '__cls_{}'.format(naturalize(Ctag))
        if Rtag: run += '__run_{}'.format(naturalize(Rtag))

        return tpath.join('/', base, run, *path)

    def has_node(self, *path, **root):
        """Whether a data node exists."""
        p = self.datapath(*path, **root)
        try:
            self.get_datafile().get_node(p)
        except tb.NoSuchNodeError as e:
            return False
        return True

    def get_node(self, *path, **root):
        """Get a handle to a data node if it exists."""
        if len(path) == 1 and isinstance(path[0], tb.Node):
            p = path[0]._v_pathname
        else:
            p = self.datapath(*path, **root)
        node = None
        try:
            node = self.get_datafile().get_node(p)
        except tb.NoSuchNodeError as e:
            self.out(f'Missing node: {p}', prefix='NodeError', error=True)
            raise(e)
        return node

    def read_node(self, *path, **root):
        """Read the given node."""
        node = self.get_node(*path, **root)
        if not hasattr(node, 'read'):
            self.out('Node is not readable: {}', node._v_pathname, error=True)
            raise IOError('Data node does not have a read method')
        return node.read()

    def read_array(self, *path, **root):
        """Read array data from the given node."""
        node = self.get_node(*path, **root)
        if not isinstance(node, tb.Array):
            self.out('Not an array: {}', node._v_pathname, error=True)
            raise TypeError('Can only read array data from an array node')
        return node.read()

    def read_dataframe(self, *path, **root):
        """Read pandas dataframe from the given node."""
        node = self.get_node(*path, **root)
        key = node._v_pathname
        self.close_datafile()
        pdf = pd.HDFStore(self._datafile.path())
        try:
            df = pdf[key]
        except KeyError:
            raise IOError('No dataframe stored at %s' % key)
        finally:
            pdf.close()
        return df

    def read_simulation(self, *path, **root):
        """Read Brian simulation output from the data path."""
        grp = self.get_node(*path, **root)
        if grp._v_attrs['tenko_type'] != 'brian':
            self.out('Not a Brian simulation: %s' % grp._v_pathname,
                    error=True)
            raise TypeError('Can only read stored Brian simulation data')

        # While HDF5 file is open, filter parent group for monitor nodes
        network_name = grp._v_attrs['name']
        parent = grp._v_pathname
        dfnames = []
        for mon_node in grp._f_iter_nodes(classname='Group'):
            if 'monitor_type' not in mon_node._v_attrs:
                self.out('skipping {}', mon_node._v_pathname, error=True)
                continue
            name = mon_node._v_name
            dfnames.append(name)

        # Create namedtuple with dataframes loaded from HDF5 file
        dfs = {}
        for name in dfnames:
            dfs[name] = self.read_dataframe(parent, name)

        simdata = namedtuple('%sData' % network_name, dfs.keys())
        return simdata(**dfs)

    def create_group(self, *path, **root):
        """Create a new group in the datafile."""
        where, name = tpath.split(self.datapath(*path, **root))
        dfile = self.get_datafile(False)
        try:
            grp = dfile.get_node(where, name=name)
        except tb.NoSuchNodeError:
            grp = data.new_group(dfile, where, name)
        else:
            self.out('group already exists: {}', grp._v_pathname,
                    prefix='Warning', error=True)
        return grp

    def create_table(self, descr, *path, attrs={}, **root):
        """Create a new table in the datafile."""
        return self._new_node('table', data.new_table, path, descr, attrs,
                root, force=True)

    def save_array(self, arr, *path, attrs={}, **root):
        """Save a data array to the datafile."""
        return self._new_node('array', data.new_array, path, np.asarray(arr),
                attrs, root)

    def save_dataframe(self, df, *path, attrs={}, **root):
        """Save a pandas Series/DataFrame/Panel to the datafile."""
        return self._new_node('dataframe', data.new_dataframe, path, df, attrs,
                root, pandas=True)

    def save_simulation(self, network, *path, attrs={}, **root):
        """Save Brian simulation output in group/DataFrame structure."""
        import brian2 as br
        monitors = [obj for obj in network.objects if type(obj) in
                        (br.StateMonitor, br.SpikeMonitor,
                            br.PopulationRateMonitor)]

        for mon in monitors:
            mon_attrs = {'name': mon.name,
                'title': f'Network: {network.name}, Monitor: {mon.name}'}
            if hasattr(mon, 'record_variables'):
                record_variables = [v for v in mon.record_variables
                                        if v not in ('t', 'i')]
            columns = {}

            if type(mon) is br.StateMonitor:
                mon_attrs['monitor_type'] = 'state'
                ts = np.tile(mon.t / mon.t.unit, len(mon.record))  # timestamps
                neuron_ix = mon.record.repeat(mon.t.shape[0])  # int unit index

            elif type(mon) is br.SpikeMonitor:
                mon_attrs['monitor_type'] = 'spike'
                ts = mon.t / mon.t.unit  # spike times in seconds
                neuron_ix = (mon.i / mon.i.unit).astype('i')  # int unit index

            elif type(mon) is br.PopulationRateMonitor:
                mon_attrs['monitor_type'] = 'poprate'
                record_variables = ['rate']
                ts = mon.t / mon.t.unit  # timestamp in seconds
                neuron_ix = np.zeros(ts.size) - 1  # flag as population (-1)

            columns.update(t=ts, neuron=neuron_ix)

            for varname in record_variables:
                trace = getattr(mon, varname)

                if type(trace) is np.ndarray:
                    unit = 'scalar'
                elif type(trace) is br.core.variables.VariableView:
                    unit = repr(trace.unit)
                    trace = trace / trace.unit
                elif type(trace) is br.units.fundamentalunits.Quantity:
                    if trace.is_dimensionless:
                        unit = 'scalar'
                    else:
                        unit = repr(trace.dimensions)
                    trace = np.asarray(trace)
                else:
                    self.out('skipping {} \'{}\': unknown type ({})'.format(
                        mon.name, varname, type(trace)), error=True)
                    continue

                if type(mon) is br.StateMonitor:
                    trace = trace.reshape(-1)
                mon_attrs['%s_unit' % varname] = unit
                columns[varname] = trace

            self.save_dataframe(pd.DataFrame(data=columns, columns=['t',
                    'neuron'] + record_variables),
                    *(path + (mon.name,)), attrs=mon_attrs, **root)

        # Write context attributes to parent group of simulation data
        self.get_datafile(False)
        grp = self.get_node(*path, **root)
        grp_attrs = dict(tenko_type='brian', name=network.name)
        grp_attrs = merge_two_dicts(grp_attrs, attrs)
        self._write_v_attrs(grp, grp_attrs)
        return grp

    def _new_node(self, ntype, new_node, path, X, attrs, root, pandas=False,
        **kwds):
        p = self.datapath(*path, **root)
        where, name = tpath.split(p)
        title = attrs.pop('title', snake2title(name))
        kwds.update(createparents=True, title=title)
        if pandas:
            self.close_datafile()
            dfile = pd.HDFStore(self._datafile.path())
        else:
            dfile = self.get_datafile(False)
        node = new_node(dfile, where, name, X, **kwds)
        self._write_v_attrs(node, attrs)
        pathname = node._v_pathname
        if ntype == 'array':
            pathname += ' |{}|'.format('x'.join(list(map(str, X.shape))))
        self.out(f'{pathname} ("{title}")', prefix=f'Saved{ntype.title()}')
        if pandas:
            dfile.close()
        return node

    def _write_v_attrs(self, node, attrs):
        call = self._lastcall
        na = node._v_attrs
        if attrs is not None:
            for k in attrs.keys():
                na[k] = attrs[k]
        if call is None:
            na['time'] = time.localtime()
            return node
        na['time'] = time.mktime(call['time'])
        for key in ('revision', 'subclass', 'step'):
            na[key] = call[key]
        for name, value in call['params']:
            na[name] = value
        for kwd, value in call['kwvalues'].items():
            na[kwd] = value

    # Figure methods

    def figure_from_template(self, svgpath, label=None, figsize=None, **figkw):
        """Load figure template to create figure and axes.

        Arguments:
        svgpath -- path to SVG file (tiny profile) where rectangles define axes
        label -- optional figure label, defaults to template basename
        figsize -- optionally override the figure size specified in the svg

        Remaining keywords are passed to the `figure` method.

        Returns:
        fig, axd -- tuple of figure handle and axes dictionary
        """
        if not svgpath.endswith('.svg'):
            svgpath += '.svg'
        if label is None:
            label = os.path.basename(os.path.splitext(svgpath)[0])
        self.out('Figure template: {}', svgpath)
        svg = get_svg_figinfo(svgpath)
        figsize = svg['figsize'] if figsize is None else figsize
        figkw.update(figsize=figsize, label=label, clear=True)
        f = self.figure(**figkw)
        axdict = {key: f.add_axes(rect) for key, rect in svg['axes'].items()}
        return f, axdict

    def set_static_figures(self, b=None):
        """Set or toggle whether figure windows are static across calls."""
        old = self._staticfigs

        if b is None:
            self._staticfigs = not self._staticfigs
        else:
            self._staticfigs = bool(b)

        if old != self._staticfigs:
            self._save()

    def figure(self, label=None, clear=False, title=None, handle=None, **kwds):
        """Create or clear a labeled figure window.

        Remaining keywords are passed to `plt.figure`.
        """
        if label is None:
            if self._lastfig:
                label = self._lastfig
            else:
                raise ValueError('no current figure')

        if handle is not None:
            self._figures[label] = handle

        try:
            fig = self._figures[label]
        except KeyError:
            if self._staticfigs:
                kwds.update(num=label)
            fig = self._figures[label] = plt.figure(**kwds)
        else:
            if 'figsize' in kwds:
                fig.set_size_inches(kwds.pop('figsize'), forward=True)
            fig.set(**kwds)
        finally:
            if clear:
                fig.clear()
                fig.subplotpars.left = None
                fig.subplotpars.right = None
                fig.subplotpars.bottom = None
                fig.subplotpars.top = None
                fig.subplotpars.wspace = None
                fig.subplotpars.hspace = None
            self._lastfig = label
            if title is not None:
                fig.suptitle(title)

        return fig

    def savefig(self, label=None, basepath=None, tag=None, tight_padding=None,
        closeafter=False, **savefig):
        """Save an open figure as an image file.

        Arguments:
        label -- optional, label of figure to be saved (default, last figure)
        basepath -- optional, save path for image file up to base name
        tag -- optional, extra text post-pended to file base name
        tight_padding -- optional, padding in inches for tight bounds

        Remaining keywords are passed to `Figure.savefig()`.
        """
        label = label or self._lastfig
        if label not in self._figures:
            self.out('Figure does not exist: {}', label, error=True)
            return

        # Set label-based path if optional base path not specified
        stem = self.path(label) if basepath is None else basepath
        if tag:
            stem += f'-{tag}'

        # Generate unique path with the given format extension
        ext = savefig.pop('format', self._figfmt)
        if ext == 'mpl':
            ext = mpl.rcParams['savefig.format']
        path = uniquify(stem, ext=ext, fmt='%s-%02d')
        parent, img = os.path.split(path)
        if not os.path.isdir(parent):
            os.makedirs(parent)

        if tight_padding is not None:
            savefig.update(bbox_inches='tight', pad_inches=tight_padding)

        # Persist the current figure save settings
        savefig.update(format=ext)
        self._savefig.update(savefig)

        self._figures[label].savefig(path, **self._savefig)
        self.out('Saved: {}', path)
        self._savefig_path = path

        if closeafter:
            self.closefig(label=label)

    def openfig(self):
        """Open the last image file saved by using `savefig`."""
        if self._savefig_path is None:
            self.out('There is no previously saved image to open.', error=True)
            return
        if sys.platform != 'darwin':
            self.out('Image viewing only available on OS X.', error=True)
            return
        p = subprocess.run(['open', self._savefig_path])
        if p.returncode != 0:
            self.out('Error opening: {}', self._savefig_path, error=True)
        else:
            self.out('Opened: {}', self._savefig_path)

    def closefig(self, label=None):
        """Close an open figure."""
        if label is None:
            label = self._lastfig

        if label not in self._figures:
            self.out('Figure does not exist: {}', label, error=True)
            return

        plt.close(self._figures[label])
        if not self._holdfigs:
            del self._figures[label]

        if label == self._lastfig:
            self._lastfig = None

    def save_figures(self, **savefig):
        """Save the currently open figures as image files.

        Keyword arguments are passed to `plt.Figure.savefig`:
        """
        for key in list(self._figures.keys()):
            self.savefig(key, **savefig)

    def close_figures(self):
        """Close all open figure windows."""
        self._holdfigs = True
        for key in self._figures:
            self.closefig(key)
        self._holdfigs = False
        self._figures.clear()

    def set_figfmt(self, fmt, save=False):
        """Set the image format for saving figures."""
        assert fmt in ('png','pdf','svg','ps','eps','mpl'), \
                'bad figure format: {}'.format(fmt)
        self._figfmt = fmt
        if save:
            self._save()
