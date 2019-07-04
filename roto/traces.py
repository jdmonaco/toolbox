"""
Temporal trace plots.
"""

from collections import deque

from .dicts import merge_two_dicts


class RealtimeTracesPlot(object):

    """
    Manage a collection of real-time windowed trace plots and axes limits.
    """

    def __init__(self, window=10.0, dt=1.0, t0=0.0, time_axis='x', fmt=None,
        datapad=0.05, legend=True, plot=False, **traces):
        """
        Initialize with keywords of trace names and tuple values with an axis
        object and optional dictionary of plot arguments. Common plot arguments
        should be passed as the `fmt` dict. The axis controlling time can be
        specified as 'x', 'y', or None (no axis limits control).
        """
        self.ax = {}
        self.fmt = {}
        self.names = []
        for name, values in traces.items():
            self.names.append(name)
            trace_fmt = {}
            if type(values) is not tuple:
                ax = values
            elif len(values) == 1:
                ax, = values
            elif len(values) == 2:
                ax, trace_fmt = values
            trace_fmt['label'] = name
            self.ax[name] = ax
            self.fmt[name] = merge_two_dicts(fmt, trace_fmt)

        self.qmax = qmax = int(window / dt)
        self.q_t = deque([], qmax)
        self.q = {name:deque([], qmax) for name in self.names}

        self.t = None
        self.t0 = t0
        self.dt = dt
        self.window = window
        self.time_axis = time_axis
        self.datapad = datapad
        self.legend = legend
        self.axobjs = set(self.ax.values())

        if self.time_axis == 'x':
            self.setlim = {name:ax.set_ylim for name, ax in self.ax.items()}
        elif self.time_axis == 'y':
            self.setlim = {name:ax.set_xlim for name, ax in self.ax.items()}

        if plot: self.plot()

    def plot(self):
        """
        Initialize empty line plots for each data trace.
        """
        self.lines = {}
        self.artists = []
        for name in self.names:
            self.lines[name] = self.ax[name].plot([], [], **self.fmt[name])[0]
            self.artists.append(self.lines[name])
        if self.legend:
            [ax.legend(loc='upper left', frameon=False) for ax in self.axobjs]
        return self.artists

    def update(self, t=None, **traces):
        """
        Update traces and axes limits with new values.

        Note: a keyword argument should be provided for each named trace.
        """
        if t is None:
            if self.t is None:
                self.t = self.t0
            else:
                self.t += self.dt
        else:
            self.t = t

        # Add new values to traces and update the plot data
        self.q_t.append(self.t)
        for name, value in traces.items():
            self.q[name].append(float(value))
            self.lines[name].set_data(self.q_t, self.q[name])

        if self.time_axis is None:
            return

        # Set trace window limits on time axis and get current data limits
        tlim = (self.q_t[0], max(self.q_t[-1], self.q_t[0] + self.window))
        if self.time_axis == 'x':
            [ax.set_xlim(tlim) for ax in self.axobjs]
            dlim = {name:ax.get_ylim() for name, ax in self.ax.items()}
        elif self.time_axis == 'y':
            [ax.set_ylim(tlim) for ax in self.axobjs]
            dlim = {name:ax.get_xlim() for name, ax in self.ax.items()}

        # Set data limits on axis (e.g., y-axis if time axis is x-axis)
        for name in self.names:
            dmax = max(self.q[name])
            dmin = min(self.q[name])
            pad = self.datapad * (dmax - dmin)
            dlim = self.ax[name].get_ylim() if self.time_axis == 'x' else \
                    self.ax[name].get_xlim()
            self.setlim[name](min(dmin-pad, dlim[0]), max(dmax+pad, dlim[1]))
