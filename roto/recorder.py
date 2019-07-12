"""
Record state, model variables, and spike/event data across a simulation.
"""

import numpy as np
import pandas as pd


DEFAULT_DURATION = 1.0
DEFAULT_DT = 0.001


class ModelRecorder(object):

    """
    Automatic recording of state, model variables, and spike/event timing.

    Notes
    -----
    (1) In the simulation loop, update() should be called first so that
    initial values at t=0 are stored, followed by variable updates.

    (2) Variable arrays are stored by reference and require that updated
    variables use the same data buffers throughout the simulation.

    (3) Initial values to the constructor may be non-boolean arrays
    (variables), boolean arrays (spike/event), or scalars (simulation state).
    Further, the value may be a tuple with two elements: (1) a data array as
    described in the last sentence, and (2) a `record` value to be passed to
    one of the `add_*_monitor` methods as described in note #4.

    (4) In the simulation loop, `update` should be called first so that initial
    values at t=0 are stored, followed by model updates of the variables. Set
    `record` to a list of indexes (for axis 0 of data) or a scalar integer
    index for selective recordings.

    (5) Updated state values must be passed to `update` as keyword arguments,
    but variable and spike/event references are stored and do not need to be
    passed in.
    """

    def __init__(self, context, duration=DEFAULT_DURATION, dt=DEFAULT_DT,
        dt_rec=DEFAULT_DT, progress_width=80, interact=False, **initial_values):
        """
        Add recording monitors for keyword-specified variables and states.

        Boolean arrays (dtype, '?') are automatically considered to be spike/
        event output. Spike/event timing will be recording for every simulation
        timestep (regardless of `dt_rec`) and appended to a pandas DataFrame
        object with 'unit' and 't' columns. Spike/event dataframes are also
        saved to the context datafile in the `save_recordings` method.
        """
        assert dt_rec >= dt, 'recording interval must be >= simulation dt'
        self.klass = self.__class__.__name__
        self.context = context
        self.interact = interact
        self.duration = duration

        # Simulation time & progress tracking
        self.dt = dt
        self.ts = np.arange(0, duration+dt, dt)
        self.Nt = len(self.ts)
        self.Nprogress = 0
        self.progress_width = progress_width
        self.n = -1  # simulation frame index, flag for unstarted simulation
        self.t = -dt

        # Recording time tracking
        self.dt_rec = dt_rec
        self.ts_rec = np.arange(0, duration+dt_rec, dt_rec)
        self.Nt_rec = len(self.ts_rec)
        self.n_rec = -1  # recording frame index
        self.t_rec = -dt_rec
        self._rec_mod = np.inf  # recording trigger; inf triggers at t=0

        # Data storage keyed by variable names
        self.unit_slices = dict()
        self.traces = dict()
        self.variables = dict()
        self.state = dict()
        self.state_traces = dict()

        # Spike/event timing storage
        self.units = dict()
        self.unit_indexes = dict()
        self.timing = dict()
        self.events = dict()

        # Use the keywords and intial values to automatically set up monitors
        # for model variables, states, and spike/event signals
        for name, data in initial_values.items():
            record = True
            if type(data) is tuple:
                if len(data) != 2:
                    self.context('Tuple values must be length 2',
                            prefix=self.klass, warning=True)
                    continnue
                data, record = data

            if isinstance(data, np.ndarray):
                if data.dtype == bool:
                    self.add_spike_monitor(name, data, record=record)
                else:
                    self.add_variable_monitor(name, data, record=record)
            elif np.isscalar(data):
                self.add_state_monitor(name, data)
            else:
                self.context('Not an array or scalar state ({})', data,
                        prefix=self.klass, warning=True)

    def add_variable_monitor(self, name, data, record=True):
        """
        Add a new monitor for a data array variable.
        """
        assert self.n == -1, 'simulation has already started'
        assert name not in self.variables, f'monitor exists ({name})'
        assert name not in self.events, f'event monitor exists ({name})'
        assert type(name) is str, 'variable name must be a string'
        assert type(data) is np.ndarray, 'variable data must be an array'

        if type(record) is bool and record == True:
            self.unit_slices[name] = sl = slice(None)
        elif np.iterable(record):
            self.unit_slices[name] = sl = list(record)
        elif np.isscalar(record) and isinstance(record, int):
            if 0 <= record < data.shape[0]:
                self.unit_slices[name] = sl = [record]
        else:
            raise ValueError(f'invalid record value ({record})')
        rdata = data[sl]

        self.traces[name] = np.zeros((self.Nt_rec,) + rdata.shape, data.dtype)
        self.variables[name] = data

    def add_spike_monitor(self, name, data, record=True):
        """
        Add a new monitor for a boolean spike or event vector.

        Note: Set `record` to a list of indexes (for axis 0 of data)
        or a scalar integer index for selective recordings.
        """
        assert self.n == -1, 'simulation has already started'
        assert name not in self.events, f'spike/event monitor exists ({name})'
        assert name not in self.variables, f'monitor with same name ({name})'
        assert type(name) is str, 'spike/event name must be a string'
        assert type(data) is np.ndarray, 'spike/event data must be an array'
        assert data.dtype == bool, 'spike/event data must be boolean'

        if type(record) is bool and record == True:
            self.unit_slices[name] = sl = slice(None)
        elif np.iterable(record):
            self.unit_slices[name] = sl = list(record)
        elif np.isscalar(record) and isinstance(record, int):
            if 0 <= record < data.shape[0]:
                self.unit_slices[name] = sl = [record]
        else:
            raise ValueError(f'invalid record value ({record})')
        rdata = data[sl]

        self.units[name] = np.array([], 'i')
        self.timing[name] = np.array([], 'f')
        self.unit_indexes[name] = np.arange(data.shape[0])[sl]
        self.events[name] = data

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

    def update(self, **newstate):
        """
        Update the time series, variable monitors, and state monitors.
        """
        self.n += 1
        if self.interact:
            self.t += self.dt
            return
        self.t = self.ts[self.n]

        # Record spike/event timing for every simulation timestep
        for name, data in self.events.items():
            recdata = data[self.unit_slices[name]]
            units = self.unit_indexes[name][recdata]
            if units.size == 0:
                continue
            timing = np.zeros(len(unit), 'f') + self.t
            self.units[name] = np.concatenate(self.units[name], units)
            self.timing[name] = np.concatenate(self.units[name], timing)

        # Trigger state and variable recording by decrease in t % t_rec
        _rec_mod = self.t % self.dt_rec
        between_samples = _rec_mod >= self._rec_mod
        self._rec_mod = _rec_mod
        if between_samples:
            return

        # Update recording index and time
        self.n_rec += 1
        self.t_rec = self.ts_rec[self.n_rec]
        if self.n_rec >= self.Nt_rec:
            self.context('Recording duration exceeded (t={:.3})',
                    self.ts_rec[-1], prefix=self.klass, warning=True)
            return

        # Set data trace to current value of variable data
        for name, data in self.variables.items():
            self.traces[name][self.n_rec] = data[self.unit_slices[name]]

        # Update state values, holding non-updated values fixed
        for name, value in self.state.items():
            if name in newstate:
                self.state_traces[name][self.n_rec] = newstate[name]
                self.state[name] = newstate[name]
                continue
            self.state_traces[name][self.n_rec] = self.state[name]

    def progressbar(self, filled=False, color='purple'):
        """
        Once-per-update console output for a simulation progress bar.
        """
        if self.interact: return
        pct = (self.n + 1) / self.Nt
        barpct = self.Nprogress / self.progress_width
        while barpct < pct:
            self.context.box(filled=filled, color=color)
            self.Nprogress += 1
            barpct = self.Nprogress / self.progress_width

    def save(self, *path, **root):
        """
        Save monitored state, variable, and spike/event recordings.
        """
        for name in self.events.keys():
            df = pd.DataFrame(data=np.c_[self.units[name], self.timing[name]],
                    columns=('unit', 't'))
            self.context.save_dataframe(df, *path, name, **root)

        if self.interact:
            self.context.out(
                    'No variable/state recordings in interactive mode',
                    prefix=self.klass, warning=True)
            return

        self.context.save_array(self.ts_rec, *path, 't', **root)

        for name, data in self.traces.items():
            self.context.save_array(data, *path, name, **root)

        for name, data in self.state_traces.items():
            self.context.save_array(data, *path, name, **root)
