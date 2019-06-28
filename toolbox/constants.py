"""
Globstar constant values for interactive work or simulation code.
"""

from .numpy import pi, nextafter


TWOPI    = twopi    = 2*pi
ONEMEPS  = onemeps  = nextafter(1, 0)
ZEROPEPS = zeropeps = nextafter(0, 1)
