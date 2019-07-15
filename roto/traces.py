"""
Temporal trace plots.
"""

from collections import deque

from numpy import cumsum, inf

from .dicts import merge_two_dicts


class RealtimeTracesPlot(object):

    """
    Manage a collection of real-time windowed trace plots and axes limits.
    """

    def __init__(self, window=10.0, dt=1.0, t0=0.0, time_axis='x', fmt=None,
        datapad=0.05, legend=True, datalim='auto', plot=False, **traces):
        """
        Initialize with keywords of trace names and tuple values with an axis
        object and optional dictionary of plot arguments. Common plot arguments
        should be passed as the `fmt` dict. If the key-value `rolled=True` is
        passed as a plot argument, then a rolling windowed difference trace is
        automatically calculated for the data stream. The axis controlling time
        can be specified as 'x', 'y', or None (no axis limits control). Two
        data-limits behaviors are available: 'auto' for full adaptation to the
        current plotted trace data, or 'expand' to only allow extension of the
        data limits, increasing the range, without contracting again.
        """
        self.ax = {}
        self.axtraces = []
        self.fmt = {}
        self.names = []
        self.rolls = []
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

            self.fmt[name] = merge_two_dicts(fmt, trace_fmt)
            self.ax[name] = ax

            # Invert the trace-axes mapping for adaptive data limits
            found = False
            for axtrace in self.axtraces:
                if ax is axtrace[0]:
                    axtrace[1].append(name)
                    found = True
                    break
            if not found:
                self.axtraces.append([ax, [name]])

            rolled = self.fmt[name].pop('rolled', False)
            if rolled:
                self.rolls.append(name)

        self.q_max = q_max = int(window / dt)
        self.q_t = deque([], q_max)
        self.q = {name:deque([], q_max) for name in self.names}
        self.q_rolls = {name:deque([], q_max) for name in self.rolls}

        self.t = None
        self.t0 = t0
        self.dt = dt
        self.window = window
        self.time_axis = time_axis
        self.datapad = datapad
        self.legend = legend
        self.datalim = datalim
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
        if hasattr(self, 'artists'):
            return self.artists
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
            if name in self.rolls:
                self.q_rolls[name].append(float(value))
                q_sum = cumsum(self.q_rolls[name])
                self.q[name].append(
                        (q_sum[-1] - q_sum[0])/len(self.q_rolls[name]))
            else:
                self.q[name].append(float(value))
            self.lines[name].set_data(self.q_t, self.q[name])

        if self.time_axis is None:
            return

        # Set trace window limits on time axis
        tlim = (self.q_t[0], max(self.q_t[-1], self.q_t[0] + self.window))
        if self.time_axis == 'x':
            [ax.set_xlim(tlim) for ax in self.axobjs]
        elif self.time_axis == 'y':
            [ax.set_ylim(tlim) for ax in self.axobjs]

        # Set data limits on axis (e.g., y-axis if time axis is x-axis)
        if self.datalim == 'auto':
            for ax, names in self.axtraces:
                axdmin, axdmax = inf, -inf
                for name in names:
                    dmin = min(self.q[name])
                    dmax = max(self.q[name])
                    pad = self.datapad * (dmax - dmin)
                    axdmin = min(dmin - pad, axdmin)
                    axdmax = max(dmax + pad, axdmax, axdmin + 0.1)
                if self.time_axis == 'x':
                    ax.set_ylim(axdmin, axdmax)
                elif self.time_axis == 'y':
                    ax.set_xlim(axdmin, axdmax)

        elif self.datalim == 'expand':
            for name in self.names:
                dmin = min(self.q[name])
                dmax = max(self.q[name])
                pad = self.datapad * (dmax - dmin)
                dlim = self.ax[name].get_ylim() if self.time_axis == 'x' else \
                        self.ax[name].get_xlim()
                self.setlim[name](min(dmin-pad, dlim[0]),
                                  max(dmax+pad, dlim[1]))
