"""
Values that may be needed by toolbox subpackages.
"""


__version__ = "0.1.2"


import os as _os
import sys as _sys


HOME = _os.getenv('HOME')
if _sys.platform == 'win32':
    HOME = _os.getenv("USERPROFILE")
PROJDIR = _os.path.join(HOME, 'projects')
DATADIR = _os.path.join(HOME, 'data')
