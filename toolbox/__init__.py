"""
Shared values across toolbox consumers.
"""


__version__ = "0.1.2"


import os as _os
import sys as _sys


# Basic directory structure
HOME = _os.getenv('HOME')
if _sys.platform == 'win32':
    HOME = _os.getenv("USERPROFILE")
PROJDIR = _os.path.join(HOME, 'projects')
DATADIR = _os.path.join(HOME, 'data')


# Machine information
IMACPRO_DPI = 217.57
MACBOOKPRO_DPI = 220.53
LG_ULTRAWIDE_DPI = 109.68
