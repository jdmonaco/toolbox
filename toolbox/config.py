"""
Global configuration for the toolbox project.
"""

__all__ = ('Config',)


from roto.dicts import AttrDict
from tenko.state import Tenko


class ToolboxConfig(AttrDict):
    pass


Config = ToolboxConfig()


# Configuration values

Config.name = 'toolbox'
Config.dpi = Tenko.screendpi
