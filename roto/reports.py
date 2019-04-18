"""
Automatic PDF report generation functionality.
"""

import os

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pylab as plt
import numpy as np

from pouty import ConsolePrinter


class PageReport(object):

    """
    Generate a page-centric multi-page PDF report of analysis data.
    """

    def __init__(self, pdfpath, numloc='bottom center', numpad=0.015, width=8.5,
        height=11.0, **figargs):
        """Create a new report object and open the pdf file.

        Arguments:
        pdfpath -- string path to the pdf report file to be created

        Keyword arguments:
        numloc -- None to disable page numbering or string description of the
            page number location that matches
            "(bottom|center|top) (left|center|right)"
        numpad -- fraction of page size to use as padding around page number
        width, height -- page dimension defaults

        Remaining arguments are passed to `plt.figure(...)`.
        """
        if not pdfpath.endswith('.pdf'):
            pdfpath += '.pdf'

        self.out = ConsolePrinter(prefix=self.__class__.__name__,
                prefix_color='green')

        if os.path.isfile(pdfpath):
            self.out('Found existing report file: {}', pdfpath)
            if input('Replace? (y/N) ').strip().lower().startswith('y'):
                os.unlink(pdfpath)
            else:
                raise IOError('existing report')

        self.path = pdfpath
        size = figargs.pop('figsize', (width, height))
        figargs.update(figsize=size)
        self.figargs = figargs
        self.page = None
        self.pagenum = 0

        self._labelargs = {}
        self._label_func = lambda p: 'p. %d' % p
        self._label_text = None
        self.set_label_fmt(weight='medium', size='small', family='sans-serif')

        self._padding = numpad
        self.set_label_location(numloc)

        self.open()

    # Private methods

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _draw_label(self):
        if self._label_text is not None:
            if self._label_text in self.page.get_children():
                self._label_text.remove()
            self._label_text = None
        if self.page is None:
            return
        self._label_text = self.page.text(self._numx, self._numy,
                self._label_func(self.pagenum),
                **self._labelargs)

    # Setter methods for page number label

    def set_label_location(self, loc, pad=None):
        if pad is None:
            pad = self._padding
        try:
            vloc, hloc = loc.strip().split()
            self._numy = {'bottom':0.0+pad, 'center':0.5, 'top':1.0-pad}[vloc]
            self._numx = {'left':0.0+pad, 'center':0.5, 'right':1.0-pad}[hloc]
        except (AttributeError, TypeError, KeyError):
            raise ValueError('bad page number location: "%s"' % loc)
        self.set_label_fmt(ha=hloc, va=vloc)
        self._padding = pad

    def set_label_fmt(self, **fmt):
        """Update frame label formatting with `plt.text` keywords."""
        self._labelargs.update(fmt)

    def set_label_function(self, label_p):
        """Function for generating the descriptive label for each frame."""
        self._label_func = label_p

    # Public methods

    def open(self):
        """Open the report file to begin creating pages."""
        if self.page is not None:
            self.out('Report is already open', error=True)
            return
        self.pdf = PdfPages(self.path)
        self.was_interactive = plt.isinteractive()
        plt.ioff()
        self.page = plt.figure(**self.figargs)
        self.pagenum = 1
        self.out('Opened report: {}', self.path)

    def save_page(self):
        """Save the current page and advance to the next."""
        self.out('Saving page {}...', self.pagenum)
        self._draw_label()
        self.pdf.savefig(figure=self.page)
        self.pagenum += 1
        self.page.clear()
        plt.figure(self.page.number)  # ensure figure is active

    def close(self):
        """Close the report file."""
        self.pdf.close()
        plt.close(self.page)
        if self.was_interactive:
            plt.ion()
        self.out('Closed report: {}', self.path)


class AxesReport(PageReport):

    """
    Generate an axes-centric PDF report of analysis data.
    """

    def __init__(self, pdfpath, naxes=None, rows=7, cols=5, xnorm=True, ynorm=True,
        polar=False, **kwargs):
        """Create a new report object and open the pdf file.

        Arguments:
        pdfpath -- string path to the pdf report file to be created

        Keyword arguments:
        naxes -- optional, specify the total number of axes to be plotted
        rows, cols -- number of rows/columns of plot axes per page
        xnorm, ynorm -- normalize x/y-axis limits across all axes on a page
        polar -- default to polar axes instead of Cartesian

        Remaining keyword arguments as passed to `PageReport.__init__`.
        """
        super().__init__(pdfpath, **kwargs)

        self.naxes = naxes
        self.nrows = rows
        self.ncols = cols
        self.xnorm = xnorm
        self.ynorm = ynorm
        self.polar = polar

        self.firstrow = False
        self.lastrow = False
        self.firstcol = False
        self.lastcol = False
        self.lastpanel = False
        self.firstonpage = False
        self.lastonpage = False
        self.lastax = None
        self.dirty = False

        self._xnorm_list = []
        self._ynorm_list = []
        self._row = 0
        self._col = -1  # allows advancement to 0 on first axes
        self._axtotal = 0

    # Plotting, figure, and document creation: support functions

    def next_axes(self, polar=None):
        """Advance to the next panel and retrieve the plot axis."""
        self._advance_panel()
        self._check_page_and_clear()

        self.lastax = ax = plt.subplot(self.nrows, self.ncols,
                self.ncols * self._row + self._col + 1,
                polar=self.polar if polar is None else polar)
        self._axtotal += 1
        self.dirty = True

        if self.xnorm: self._xnorm_list.append(ax)
        if self.ynorm: self._ynorm_list.append(ax)

        self.firstrow = (self._row == 0)
        self.firstcol = (self._col == 0)
        self.lastcol = (self._col == self.ncols-1)
        self.lastrow = (self._row == self.nrows-1)
        self.firstonpage = self.firstcol and self.firstrow

        if self.naxes is not None:
            self.lastrow |= (self._axtotal > self.naxes - self.ncols)
            self.lastpanel = (self._axtotal == self.naxes)
        self.lastonpage = (self.lastcol and self.lastrow) or self.lastpanel

        return ax

    def denormx(self, ax=None):
        ax = ax is None and self.lastax or ax
        if ax in self._xnorm_list:
            self._xnorm_list.remove(ax)

    def denormy(self, ax=None):
        ax = ax is None and self.lastax or ax
        if ax in self._ynorm_list:
            self._ynorm_list.remove(ax)

    def breakline(self):
        self._col = -1  # allows advancement to 0 on next axes
        if self.dirty:
            self._row += 1
        self._check_page_and_clear()

    def newline(self):
        self.breakline()
        self.breakline()

    def close(self):
        if self.dirty:
            self._normalize_axes()
            self.save_page()
        super().close()

    # Private support methods

    def _advance_panel(self):
        self._col += 1
        if self._col > self.ncols - 1:
            self._col = 0
            self._row += 1

    def _check_page_and_clear(self):
        if self._row > self.nrows - 1:
            self._normalize_axes()
            self.save_page()
            self.dirty = False
            self._row = 0
            self._xnorm_list = []
            self._ynorm_list = []

    def _normalize_axes(self):
        if self._xnorm_list:
            xaxis = np.array([ax.axis() for ax in self._xnorm_list
                                if not hasattr(ax, 'set_rlim')])
            xmin = xaxis[:,0].min()
            xmax = xaxis[:,1].max()

            for ax in self._xnorm_list:
                ax.set_xlim(xmin, xmax)

        if self._ynorm_list:
            yaxis = np.array([ax.axis() for ax in self._ynorm_list])
            ymin = yaxis[:,2].min()
            ymax = yaxis[:,3].max()

            for ax in self._ynorm_list:
                ax.set_ylim(ymin, ymax)
