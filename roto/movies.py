"""
Generate movies visualizing data in figure windows.
"""

import os
import subprocess
import multiprocessing

from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from scipy.misc import imfilter, imresize, toimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pouty import ConsolePrinter

from .paths import uniquify


out = ConsolePrinter(prefix='RotoMovies')


class FigureMovie(object):

    """
    Make a movie based on animating data within a figure window.
    """

    def __init__(self, moviepath, aa=2.0, res=480, width=8.0, height=6.0,
        duration=2.0, fps=24.0, loc="bottom right", labelpad=0.015, **figargs):
        """Set up the figure window for generating the movie.

        Arguments:
        moviepath -- string path to the movie file to be created

        Keyword arguments:
        aa -- antialiasing factor (computed by smooth downsizing)
        res -- video resolution in number of pixel rows
        width, height -- figure dimension defaults in inches
        duration -- total length of time for the video
        fps -- video frame rate
        loc -- a string description of the frame label location that matches
               "(bottom|center|top) (left|center|right)"
        labelpad -- fraction of figure size for padding the frame label

        Remaining arguments are passed to `plt.figure(...)`.
        """
        ext = os.path.splitext(moviepath)[1]
        if not ext:
            moviepath += '.mp4'

        self.ncpus = multiprocessing.cpu_count()
        self.path = moviepath
        self.duration = duration
        self.fps = fps

        self._frame = None
        self._update_t = lambda t: None

        self._labelargs = {}
        self._label_t = lambda t: 't = %0.1f s' % t
        self._label_text = None
        self.set_label_fmt(weight='medium', size='small', family='sans-serif')

        self._padding = labelpad
        self.set_label_location(loc)

        self._aa = max(1.0, aa)
        self._res = int(res)
        sz = figargs.pop('figsize', (width, height))
        figargs.update(figsize=sz, facecolor='w', frameon=False)
        self._figargs = figargs

        self._backend = mpl.get_backend()
        self.get_frame()  # initialize the figure frame

        out('Movie clip initialized:\n{}', self.path)

    ### Private methods ###

    def _draw_label(self):
        if self._label_text is not None:
            self._label_text.remove()
            self._label_text = None
        if self._frame is None:
            return
        self._label_text = self._frame.text(self._labelx, self._labely, '',
                **self._labelargs)

    def _frame_t(self, t):
        self._update_t(t)
        self._label_text.set_text(self._label_t(t))
        if self._aa <= 1.0:
            return mplfig_to_npimage(self._frame)
        return imresize(
                imfilter(
                  mplfig_to_npimage(self._frame), 'smooth'), 1 / self._aa)

    def _enter(self):
        self._interactive = plt.isinteractive()
        plt.ioff()
        self.get_frame()

    def _exit(self):
        plt.switch_backend(self._backend)
        if self._interactive:
            plt.ion()

    ### Accessor methods ###

    def set_antialiasing(self, aa):
        """Set the antialiasing factor for rendering frames."""
        self._aa = max(1.0, aa)
        self.update_dpi()

    def set_resolution(self, res):
        """Set the row resolution of the video."""
        self._res = int(res)
        self.update_dpi()

    def set_label_location(self, loc, pad=None):
        if pad is None:
            pad = self._padding
        try:
            vloc, hloc = loc.strip().split()
            self._labely = {'bottom':0.0+pad, 'center':0.5, 'top':1.0-pad}[vloc]
            self._labelx = {'left':0.0+pad, 'center':0.5, 'right':1.0-pad}[hloc]
        except (AttributeError, TypeError, KeyError):
            raise ValueError('bad frame label location: "%s"' % loc)
        self.set_label_fmt(ha=hloc, va=vloc)
        self._padding = pad

    def set_label_fmt(self, **fmt):
        """Update frame label formatting with `plt.text` keywords."""
        self._labelargs.update(fmt)
        self._draw_label()

    def set_label_function(self, label_t):
        """Function for generating the descriptive label for each frame."""
        self._label_t = label_t

    def set_update_function(self, update_t):
        """Function to update data figure for each frame."""
        self._update_t = update_t

    def get_frame(self):
        """Retrieve the figure instance to be used as the video frame."""
        if not self._frame:
            if mpl.get_backend() != 'agg':
                plt.switch_backend('agg')
            self.set_frame(plt.figure(**self._figargs))
        return self._frame

    def set_frame(self, fig):
        """Set the recording frame to an existing figure window."""
        try:
            fig.canvas.tostring_rgb
        except AttributeError:
            out('Figure does not have an agg canvas', error=True)
            return
        else:
            self.close()

        self._frame = fig
        self._draw_label()
        self.update_dpi()

    def restore_backend(self):
        """Restore mpl backend and interactivity after movie rendering."""
        self._exit()

    def update_dpi(self):
        """Recalculate the figure frame dpi."""
        if self._frame is None:
            return
        w, h = self._frame.get_size_inches()
        dpi = int(self._aa * self._res / h)
        self._figargs.update(figsize=(w, h), dpi=dpi)
        self._frame.set_dpi(dpi)

    ### Interface methods ###

    def render(self, newpath=None, keep_open=False):
        """Render the entire movie and compress it into video."""
        savepath = self.path if newpath is None else newpath
        if os.path.isfile(savepath):
            out('Found existing movie file!')
            if not input('Replace? (y/N) ').strip().lower().startswith('y'):
                return
            os.unlink(savepath)

        out('Rendering video:')
        self._enter()
        clip = VideoClip(make_frame=self._frame_t, duration=self.duration)
        clip.write_videofile(savepath, fps=self.fps, threads=self.ncpus,
                verbose=False)
        self._exit()
        if not keep_open:
            self.close()
        out('Finished writing video file:\n{}', savepath)

    def snapshot(self, t, desc=None):
        """Save a single image or sequence of snapshots for the given times."""
        if np.iterable(t):
            [self.snapshot(frame_t) for frame_t in t]
            return

        self._enter()
        im = toimage(self._frame_t(t))
        base = os.path.splitext(self.path)[0]
        ext = '.png'
        if desc is None:
            fn = uniquify(base, ext=ext, fmt='%s_%03d')
        else:
            fn = '%s_%s%s' % (base, desc, ext)
        try:
            im.save(fn)
        except:
            out('Error saving the snapshot: {}', fn, error=True)
        else:
            out('Saved snapshot: {}', fn)
        finally:
            self._exit()

    def view(self):
        """View the video file in the default video viewer."""
        if not os.path.isfile(self.path):
            return
        if sys.platform == 'win32':
            out('Sorry, Windows.')
            return
        if subprocess.call(['open', self.path]) != 0:
            out('Problem opening movie file: {}', self.path, error=True)

    def close(self):
        """Close the figure frame."""
        if not self._frame:
            return
        plt.close(self._frame.number)
        self._frame = None
