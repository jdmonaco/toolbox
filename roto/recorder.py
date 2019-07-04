"""
Record model array variables and state values across a simulation.
"""

import numpy as np


DEFAULT_DURATION = 1.0
DEFAULT_DT = 0.001


class ModelRecorder(object):

    def __init__(self, context, duration=DEFAULT_DURATION, dt=DEFAULT_DT,
        pbar_width=100, **initial_values):
        """
        Add recording monitors for keyword-specified variables and states.

        Note: variable arrays are stored by reference and require that update
        variables use the same data buffers throughut the simulation.
        """
        self.context = context
        self.duration = duration
        self.dt = dt

        self.timesteps = np.arange(0, duration+dt, dt)
        self.Nt = len(self.timesteps)
        self.pbarpct = self.Nt >= pbar_width
        self.pbarmod = int(self.Nt/pbar_width)
        self.n = -1  # flag for unstarted simulation
        self.t = self.timesteps[self.n]

        self.traces = dict()
        self.variables = dict()
        self.state = dict()
        self.state_traces = dict()

        for name, data in initial_values.items():
            if isinstance(data, np.ndarray):
                self.add_variable_monitor(name, data)
                continue
            self.add_state_monitor(name, data)

    def add_variable_monitor(self, name, data):
        """
        Add a new monitor for a data array variable.

        Note: In the simulation loop, update() should be called first
        so that initial values at t=0 are stored, followed by model updates of
        the variables.
        """
        assert self.n == -1, 'simulation has already started'
        assert name not in self.variables, f'monitor exists ({name})'
        assert type(name) is str, 'variable name must be a string'
        assert type(data) is np.ndarray, 'variable data must be an array'

        self.traces[name] = np.zeros((self.Nt,) + data.shape, data.dtype)
        self.variables[name] = data

    def add_state_monitor(self, name, value):
        """
        Add a new monitor for a state value (scalars of any type).
        """
        assert self.n == -1, 'simulation has already started'
        assert name not in self.state, f'state monitor exists ({name})'
        assert type(name) is str, 'variable name must be a string'
        assert np.size(value) == 1, 'state must be a scalar value'

        self.state_traces[name] = np.zeros(self.Nt, np.array(value).dtype)
        self.state[name] = value

    def t_text(self, fmt='t = {:0.3f} s'):
        return fmt.format(self.t)

    def progressbar(self, color='purple'):
        """
        Once-per-update console output for a simulation progress bar.
        """
        if self.pbarpct:
            if self.n % self.pbarmod == 0:
                self.context.box(filled=False, color=color)
        else:
            self.context.box(filled=False, color=color)

    def update(self, **newstate):
        """
        Update the time series, variable monitors, and state monitors.

        Note: Updated state values must be passed in as keyword arguments, but
        variable data references are stored and do not need to be passed in.
        """
        self.n += 1
        self.t = self.timesteps[self.n]

        # Set data trace to current value of variable data
        for name, data in self.variables.items():
            self.traces[name][self.n] = data

        # Update state values, holding non-updated values fixed
        for name, value in self.state.items():
            if name in newstate:
                self.state_traces[name][self.n] = newstate[name]
                self.state[name] = newstate[name]
                continue
            self.state_traces[name][self.n] = self.state[name]

    def save_recordings(self, **dpath):
        """
        Save all monitored recordings of variables and state to the datafile.

        Keyword arguments are passed to datapath(...).
        """
        self.context.save_array(self.timesteps, 't', **dpath)
        for name, data in self.traces.items():
            self.context.save_array(data, name, **dpath)
        for name, data in self.state_traces.items():
            self.context.save_array(data, name, **dpath)
