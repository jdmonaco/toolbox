"""
Context mixin for handling parameter defaults, values, and files.
"""

try:
    import simplejson as json
except ImportError:
    import json

import os
from importlib import import_module
from collections import OrderedDict

import numpy as np
from numpy.random import seed

from maps.geometry import EnvironmentGeometry
from roto.dicts import AttrDict


def dumpjson(fpath, data):
    """
    Save key-value data to JSON file at the specified path.
    """
    kwjson = dict(indent=2, separators=(', ', ': '))
    with open(fpath, 'w') as fd:
        json.dump({k:v for k,v in data.items() if v is not None}, fd, **kwjson)


class ParametersMixin(object):

    def _init_attr(self, name, initval):
        """
        Initialize an instance attribute without overwriting.
        """
        if hasattr(self, name):
            return
        setattr(self, name, initval)

    def set_parameter_defaults(self, **dflts):
        """
        Set default values for parameter keys using keyword arguments.
        """
        self._init_attr('_defaults', OrderedDict())
        self._defaults.update(**dflts)

    def get_parameters_from_file(self, pfile):
        """
        Retrieve parameter values from a JSON parameter file.

        Returns (parampath, params) tuple.
        """
        if not pfile.endswith('.json'):
            pfile += '.json'
        if os.path.isfile(pfile):
            pfilepath = pfile
        else:
            ctxpfile = os.path.join(self._ctxdir, pfile)
            modpfile = os.path.join(self._moduledir, pfile)
            if os.path.isfile(ctxpfile):
                pfilepath = ctxpfile
            elif os.path.isfile(modpfile):
                pfilepath = modpfile
            else:
                self.out(pfile, prefix='MissingParamFile', error=True)
                raise ValueError('Unable to find parameter file')
        with open(pfilepath, 'r') as fd:
            params_from_file = json.load(fd)
        return pfilepath, params_from_file

    def _get_module_scope(self):
        return import_module(self.__class__.__module__).__dict__

    def set_parameters(self, pfile=None, **params):
        """
        Set model parameters according to file, keywords, or defaults.
        """
        # Import values from parameters file if specified
        if pfile is not None:
            fpath, fparams = self.get_parameters_from_file(pfile)
            fparams.update(params)
            params = fparams
            self.out(fpath, prefix='ParameterFile')

        # Write JSON file with parameter defaults
        if hasattr(self, '_defaults'):
            dfltpath = os.path.join(self._ctxdir, 'defaults.json')
            dumpjson(dfltpath, self._defaults)
            defaults = self._defaults
        else:
            defaults = OrderedDict()

        # Write JSON file with actual updated parameters in effect
        p = []
        if self._tag:
            p.append(self._tag)
        if self._lastcall and 'tag' in self._lastcall:
            p.append(self._lastcall['tag'])
        psavefn = '-'.join(list(map(
            lambda s: s.strip().lower().replace(' ','-'), p))) + '.json'
        parampath = self.path(psavefn)
        dparams = defaults.copy()
        dparams.update(params)
        dumpjson(parampath, dparams)

        # Set parameters as global variables in the object's module scope and as
        # an attribute dict `p` on the object itself
        self.out('Independent parameters:')
        modscope = self._get_module_scope()
        modscope.update(dparams)
        self.p = AttrDict(dparams)
        for name, value in dparams.items():
            logstr = f' - {name} = {value}'
            dflt = defaults.get(name)
            if dflt is not None and value != dflt:
                logstr = f' * {name} = {value} [default: {dflt}]'
            self.out(logstr, hideprefix=True)

    def load_environment_parameters(self, env):
        """
        Import environment geometry into the module and object scope.
        """
        modscope = self._get_module_scope()
        modscope['E'] = self.E = E = EnvironmentGeometry(env)
        Ivars = list(sorted(E.info.keys()))
        Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            np.ndarray), E.__dict__.keys())))
        modscope.update(E.info)
        modscope.update({k:getattr(E, k) for k in Avars})
        self.out(repr(E), prefix='Geometry')
        for v in Ivars:
            setattr(self, v, E.info[v])
            self.out(f' - {v} = {getattr(self, v)}', prefix='Geometry',
                hideprefix=True)
        for k in Avars:
            setattr(self, k, getattr(E, k))
            self.out(' - {} ({})', k, 'x'.join(list(map(str, getattr(self,
                k).shape))), prefix='Geometry', hideprefix=True)

    def set_dependent_parameters(self, **params):
        """
        Set other parameters whose values depend on the independent parameters.
        """
        self.out('Dependent parameters:')
        self._init_attr('p', AttrDict())
        modscope = self._get_module_scope()
        modscope.update(params)
        for name, value in params.items():
            self.p[name] = value
            self.out(f' - {name} = {value}', hideprefix=True)

    def set_random_seed(self, key):
        """
        Set the numpy random seed according to a key string or hash.
        """
        newseed = sum(list(map(ord, key)))
        seed(newseed)
        self.out(f'{newseed} [key: \'{key}\']', prefix='RandomSeed')


#
# Original code below
#

    # def set_parameters(self, pfile=None, **params):
        # """
        # Set model parameters according to file, keywords, or defaults.
        # """
        # # Model parameters - Import from parameters file
        # if pfile is not None:
            # if not pfile.endswith('.json'):
                # pfile += '.json'
            # if os.path.isfile(pfile):
                # pfilepath = pfile
            # else:
                # ctxpfile = os.path.join(self._ctxdir, pfile)
                # modpfile = os.path.join(self._moduledir, pfile)
                # if os.path.isfile(ctxpfile):
                    # pfilepath = ctxpfile
                # elif os.path.isfile(modpfile):
                    # pfilepath = modpfile
                # else:
                    # self.out(pfile, prefix='MissingParamFile', error=True)
                    # raise ValueError('Unable to find parameter file')
            # with open(pfilepath, 'r') as fd:
                # params_from_file = json.load(fd)
            # params_from_file.update(params)
            # params = params_from_file
            # self.out(pfilepath, prefix='ParameterFile')

        # # Model parameters - Default values
        # # Model parameter - Write JSON file with parameter defaults
        # kwjson = dict(indent=2, separators=(', ', ': '))
        # dfltpath = os.path.join(self._ctxdir, 'defaults.json')
        # with open(dfltpath, 'w') as fd:
            # json.dump({k:v for k,v in defaults.items() if v is not None},
                    # fd, **kwjson)

        # # Model parameter - Write JSON file with actual updated parameters
        # p = []
        # if self._tag:
            # p.append(self._tag)
        # if self._lastcall:
            # p.append(self._lastcall['step'])
            # if self._lastcall['tag']:
                # p.append(self._lastcall['tag'])
        # psavefn = '-'.join(list(map(
            # lambda s: s.strip().lower().replace(' ','-'), p))) + '.json'
        # parampath = self.path(psavefn)
        # dparams = defaults.copy()
        # dparams.update(params)
        # with open(parampath, 'w') as fd:
            # json.dump({k:v for k,v in dparams.items() if v is not None},
                    # fd, **kwjson)

        # # Force tag to test name if a test is being performed
        # if 'test' in dparams and dparams['test'] is not None:
            # self._lastcall['tag'] = params['test']
            # self.out('\'{}\'', params['test'], prefix='RunningTest')

        # # Model parameters - Set as global variables and log the values
        # self.out('Independent parameters:')
        # for name, dflt in defaults.items():
            # exec(f'global {name}; {name} = dparams[\'{name}\']')
            # val = globals()[name]
            # if val == dflt:
                # logstr = f' - {name} = {dflt}'
            # else:
                # logstr = f' * {name} = {val} [default: {dflt}]'
            # self.out(logstr, hideprefix=True)

        # # Environment - Import environmental geometry into the global scope,
        # # into the persistent key-value store, and as instance attributes of
        # # the simulation object
        # global E
        # self.hline()
        # E = self.E = EnvironmentGeometry(env)
        # Ivars = list(sorted(E.info.keys()))
        # Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            # ndarray), E.__dict__.keys())))
        # self.out(repr(E), prefix='Geometry')
        # for v in Ivars:
            # exec(f'global {v}; self[\'{v}\'] = self.{v} = {v} = E.info[\'{v}\']')
            # self.out(f' - {v} = {self[v]}', prefix='Geometry', hideprefix=True)
        # for k in Avars:
            # exec(f'global {k}; self.{k} = {k} = E.{k}')
            # self.out(' - {} ({})', k, 'x'.join(list(map(str, getattr(self,
                # k).shape))), prefix='Geometry', hideprefix=True)
        # self['env'] = env
        # self._save_env()

        # # Other parameters - Dependent values
        # self.hline()
        # self.out('Dependent parameters:')
        # depvalues = OrderedDict(
            # N            = 1 if single_agent else N_S,
            # m_mean       = m_single if single_agent else m_multi,
            # D_max_scaled = D_max*G_scale,
            # sigma_scaled = sigma*G_scale,
            # kappa_scaled = kappa*G_scale,
            # A_scaled     = A_sw*G_scale,
            # B_scaled     = B_sw*G_scale**2,
            # J_scaled     = J_sw*G_scale,
            # K_scaled     = K_sw*G_scale
        # )
        # for name, val in depvalues.items():
            # exec(f'global {name}; {name} = {val}')
            # self.out(f' - {name} = {val}', hideprefix=True)
        # self.hline()
