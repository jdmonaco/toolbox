"""
Classes for writing data to disk.
"""


class CSVWriter(object):

    """
    Pass in a filename and list(tuple(colname, 's|d|f')) to define columns,
    call `get_row` for a row dictionary, fill it up, and then call `write_row`
    and close when you're done.
    """

    def __init__(self, fn, cols, sep=','):
        self._init = dict(s='', d=0, f=0.0)
        self._cols = cols
        self._rowstr = sep.join(['%%(%s)%s' % col for col in _cols]) + '\n'
        self.filename = fn
        self._fd = open(fn, 'w')
        self._fd.write(','.join([col for col, dtype in _cols]) + '\n')
        sys.stdout.write(f'Opened spreadsheet {fn}.\n')

    def get_row(self):
        return { col: self._init[dtype] for col, dtype in self._cols }

    def write_row(self, record):
        self._fd.write(self._rowstr % record)

    def close(self):
        self._fd.close()
        sys.stdout.write(f'Closed spreadsheet {self.filename}.\n')
