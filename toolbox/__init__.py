"""
Shared values across toolbox consumers.
"""


__version__ = "0.1.3"


import os as _os
import sys as _sys


# Essential directory structure

HOME = _os.getenv('HOME')
if _sys.platform == 'win32':
    HOME = _os.getenv("USERPROFILE")
PROJDIR = _os.path.join(HOME, 'projects')
DATADIR = _os.path.join(HOME, 'data')


# Screen resolution information

IMACPRO_DPI = int(217.57)
MACBOOKPRO_DPI = int(220.53)
LG_ULTRAWIDE_DPI = int(109.68)
STANDARD_DPI = 96


# Host resolution mapping

HOST_DPI = dict(
        hilbert_retina = MACBOOKPRO_DPI,
        hilbert = LG_ULTRAWIDE_DPI,
        hebb = IMACPRO_DPI,
        unknown = STANDARD_DPI,
)
