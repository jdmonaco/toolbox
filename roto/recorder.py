"""
Context mixin for recording model variables across a simulation.
"""

import numpy as np


DEFAULT_DURATION = 1.0
DEFAULT_DT = 0.001


class RecorderMixin(object):

    def recorder_init(self, duration=DEFAULT_DURATION, dt=DEFAULT_DT,
        pbar_width=100):
        """
        Set up the timesteps for the model simulation.
        """
        self._recorder_traces = dict()
        self._recorder_variables = dict()
        self._recorder_timesteps = np.arange(0, duration+dt, dt)
        self._recorder_Nt = len(self._recorder_timesteps)
        self._recorder_pbarpct = self._recorder_Nt >= pbar_width
        self._recorder_pbarmod = int(self._recorder_Nt/pbar_width)
        self._recorder_n = -1  # flag for unstarted simulation
        self._recorder_t = self._recorder_timesteps[self._recorder_n]

    def set_recorder_variables(self, **variables):
        """
        Add recording monitors for keyword-specified model variables.

        Note: data are stored by reference, and thus require that update
        variables use the same array buffers throughut the simulation.
        """
        if not hasattr(self, '_recorder_traces'):
            raise RuntimeError('you must run recorder_init() first')
        for name, data in variables.items():
            self.recorder_add_monitor(name, data)

    def recorder_add_monitor(self, name, data):
        """
        Add a new monitor for a recording variable.

        Note: In the simulation loop, recorder_update() should be called first
        so that initial values at t=0 are stored, followed by model updates of
        the variables.
        """
        assert self._recorder_n == -1, 'simulation has already started'
        assert name not in self._recorder_variables, f'monitor exists ({name})'
        assert type(name) is str, 'variable name must be a string'
        assert type(data) is np.ndarray, 'variable data must be an array'

        trace = np.zeros((self._recorder_Nt,) + data.shape, data.dtype)
        self._recorder_traces[name] = trace
        self._recorder_variables[name] = data

    def recorder_t(self):
        return self._recorder_t

    def recorder_t_text(self, fmt='t = {:0.3f} s'):
        return fmt.format(self._recorder_t)

    def recorder_progressbar(self, color='purple'):
        if self._recorder_pbarpct:
            if self._recorder_n % self._recorder_pbarmod == 0:
                self.box(filled=False, color=color)
        else:
            self.box(filled=False, color=color)

    def recorder_update(self):
        """
        Update the time-series and monitored data traces.
        """
        self._recorder_n += 1
        self._recorder_t = self._recorder_timesteps[self._recorder_n]

        # Set data trace to current value of variable data
        for name, data in self._recorder_variables.items():
            self._recorder_traces[name][self._recorder_n] = data

    def save_recorder_traces(self, **dpath):
        """
        Save all monitored data traces to context datafile.

        Keyword arguments are passed to datapath().
        """
        self.save_array(self._recorder_timesteps, 't', **dpath)
        for name, data in self._recorder_traces.items():
            self.save_array(data, name, **dpath)
