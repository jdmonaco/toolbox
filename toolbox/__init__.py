"""
Simple container for the tools packages.
"""


VERSION = "0.1.2"


import os as _os
import sys as _sys

import pouty
import roto
import maps
import analyze
import tenko


HOME = _os.getenv('HOME')
if _sys.platform == 'win32':
    HOME = _os.getenv("USERPROFILE")
PROJDIR = _os.path.join(HOME, 'projects')
DATADIR = _os.path.join(HOME, 'data')
