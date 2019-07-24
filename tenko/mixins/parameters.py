"""
Context mixin for handling parameter defaults, values, and files.
"""

import os

try:
    import simplejson as json
except ImportError:
    import json

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import numpy as np
from numpy.random import seed

from maps.geometry import EnvironmentGeometry
from roto.dicts import AttrDict


class ParametersMixin(object):

    def set_parameter_defaults(self, **dflts):
        """
        Set default values for parameter keys using keyword arguments.
        """
        self._init_attr('_defaults', AttrDict())
        self._defaults.update(**dflts)

    def set_parameters(self, pfile=None, **params):
        """
        Set model parameters according to file, keywords, or defaults.
        """
        # Import values from parameters file if specified
        if pfile is not None:
            fpath, fparams = self.get_json(pfile)
            fparams.update(params)
            params = fparams
            self.out(fpath, prefix='ParameterFile')

        # Write JSON file with parameter defaults
        if hasattr(self, '_defaults'):
            self.write_json(self._defaults, 'defaults.json', base='context')
            defaults = self._defaults
        else:
            defaults = dict()

        # Write JSON file with actual updated parameters in effect
        p = []
        if self._tag:
            p.append(self._tag)
        if self._lastcall and 'tag' in self._lastcall:
            if self._lastcall['tag'] is not None:
                p.append(self._lastcall['tag'])
        psavefn = '-'.join(list(map(
            lambda s: s.strip().lower().replace(' ','-'), p))) + '.json'
        dparams = defaults.copy()
        dparams.update(params)
        self.write_json(dparams, psavefn)

        # Set parameters as global variables in the object's module scope and as
        # an attribute dict `p` on the object itself
        self.out('Independent parameters:')
        modscope = self._get_global_scope()
        modscope.update(dparams)
        self.p = AttrDict(dparams)
        for name, value in dparams.items():
            logstr = f'- {name} = {value}'
            dflt = defaults.get(name)
            if dflt is not None and value != dflt:
                logstr = f'* {name} = {value} [default: {dflt}]'
            self.out(logstr, hideprefix=True)

    def update_parameters(self, **params):
        """
        Modify global & instance parameter values after initialization.
        """
        for name, value in params.items():
            if name not in self.p:
                self.out(name, prefix='UnknownParameter', warning=True)
                continue
            self.p[name] = value
            self._get_global_scope()[name] = value

    def get_panel_widgets(self, step=0.01, **sliders):
        """
        Construct a slider widget Panel based on (start, end) tuples.
        """
        names = list(sliders.keys())
        widgets = [pn.widgets.FloatSlider(name=k, value=self.p[k], start=s,
                    end=e, step=step) for k, (s, e) in sliders.items()]
        values = [w.param.value for w in widgets]

        # Store a dict of the parameter-widget mappings as an attribute
        self._init_attr('widgets', {})
        self.widgets.update({w.name:w for w in widgets})

        @pn.depends(*values)
        def callback(*values):
            params = {k:w.value for k, w in zip(names, widgets)}
            self.update_parameters(**params)
            return pn.Column(*widgets)
        return callback

    def get_panel_controls(self):
        """
        Get a column of buttons and text input for saving/restoring parameters.
        """
        paramfile_input = pn.widgets.TextInput(name='Filename',
                placeholder='params')
        filename_txt = pn.pane.Markdown('*(unsaved)*')
        saveall_box = pn.widgets.Checkbox(name='Save all', value=False)
        uniquify_box = pn.widgets.Checkbox(name='Force unique', value=False)
        save_btn = pn.widgets.Button(name='Save', button_type='primary')
        restore_btn = pn.widgets.Button(name='Restore', button_type='success')
        dflt_btn = pn.widgets.Button(name='Defaults', button_type='warning')
        zero_btn = pn.widgets.Button(name='Zero', button_type='danger')

        # Construct nested dicts of gain and model parameters
        State = self._get_state()
        gain = {g.name:g.gain_sliders for g in State.network.neuron_groups}
        neuron = {g.name:g.neuron_sliders for g in State.network.neuron_groups}

        # TODO: This code is pretty slow, due to the group searches I think.
        # It also does not really seem to work well. It should be fairly
        # heavily debugged, which is difficult given that these are UI
        # callbacks on a single thread. A Network dict of groups would help.

        def save(value):
            psavefn = paramfile_input.value
            saveall = saveall_box.value
            unique = uniquify_box.value
            params = {}
            if saveall and hasattr(self, 'p'):
                self.p.backup_to(params)
            params.update({name:w.value for name, w in self.widgets.items()})
            for grp in State.network.neuron_groups:
                params.update({f'g_{grp.name}_{k}':slider.value
                               for k, slider in gain[grp.name].items()})
                params.update({f'{k}_{grp.name}':slider.value
                               for k, slider in neuron[grp.name].items()})
            p = self.write_json(params, psavefn, base='context', unique=unique)
            filename_txt.object = p

        def restore(value):
            psavefn = paramfile_input.value
            fullpath, params = self.get_json(psavefn)
            filename_txt.object = fullpath
            self.update_parameters(**params)
            for name, value in params.items():
                if name in self.widgets:
                    self.widgets[name].value = value
                elif name.startswith('g_'):
                    _, post, pre = name.split('_')
                    for grp in State.network.neuron_groups:
                        if grp.name == post:
                            break
                    else:
                        continue
                    if pre in grp.gain:
                        grp.gain[pre] = value
                else:
                    tokens = name.split('_')
                    post = tokens[-1]
                    pname = '_'.join(tokens[:-1])
                    for grp in State.network.neuron_groups:
                        if grp.name == post:
                            break
                    else:
                        continue
                    if pname in grp.spec:
                        setattr(grp, pname, value)

        def defaults(value):
            if not hasattr(self, '_defaults'):
                return
            self.update_parameters(**self._defaults)
            for name, value in self._defaults.items():
                if name in self.widgets:
                    self.widgets[name].value = value

            for grp in State.network.neuron_groups:
                grp.reset()
                for pre, slider in gain[grp.name].items():
                    slider.value = grp.gain[pre]
                for pname, slider in neuron[grp.name].items():
                    slider.value = getattr(grp, pname)

        def zeros(value):
            for slider in self.widgets.values():
                slider.value = slider.start
                self.update_parameters(**{slider.name:slider.start})

            for grp in State.network.neuron_groups:
                for pre, slider in grp.gain_sliders.items():
                    slider.value = 0.0
                    grp.gain[pre] = 0.0
                for pname, slider in grp.neuron_sliders.items():
                    slider.value = slider.start
                    setattr(grp, pname, slider.start)

        save_btn.param.watch(save, 'clicks')
        restore_btn.param.watch(restore, 'clicks')
        dflt_btn.param.watch(defaults, 'clicks')
        zero_btn.param.watch(zeros, 'clicks')

        return pn.Row(
                    pn.Column(
                        paramfile_input,
                        filename_txt,
                        saveall_box,
                        uniquify_box,
                    ),
                    pn.Column(
                        save_btn,
                        restore_btn,
                        dflt_btn,
                        zero_btn,
                    ),
                )

    def load_environment_parameters(self, env):
        """
        Import environment geometry into the module and object scope.
        """
        modscope = self._get_global_scope()
        modscope['Env'] = self.e = E = EnvironmentGeometry(env)
        Ivars = list(sorted(E.info.keys()))
        Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            np.ndarray), E.__dict__.keys())))
        modscope.update(E.info)
        modscope.update({k:getattr(E, k) for k in Avars})
        self.out(repr(E), prefix='Geometry')
        for v in Ivars:
            self.out(f'- {v} = {getattr(E, v)}', prefix='Geometry',
                hideprefix=True)
        for k in Avars:
            self.out('- {} ({})', k, 'x'.join(list(map(str, getattr(E,
                k).shape))), prefix='Geometry', hideprefix=True)

    def set_dependent_parameters(self, **params):
        """
        Set other parameters whose values depend on the independent parameters.
        """
        self.out('Dependent parameters:')
        self._init_attr('p', AttrDict())
        modscope = self._get_global_scope()
        modscope.update(params)
        for name, value in params.items():
            self.p[name] = value
            self.out(f'- {name} = {value}', hideprefix=True)
