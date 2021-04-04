"""
Functions for handling paths.
"""

import os

from toolbox import HOME


def tilde(p):
    return p.replace(HOME, '~')

def uniquify(stem, ext=None, fmt='{stem!s}-{u:02d}{ext}'):
    """
    Insert a unique identifier into a file or directory path

    Arguments:
    stem -- the full path up until a unique identifier is needed

    Keyword arguments:
    ext -- this extension will be added to the path (default none)
    fmt -- format specification with one string (%s) and one integer (%d) for
        the stem and unique identifier; an extension may be included, but then
        **ext** should not be specified (default '%s%02d')
    reverse_fmt -- specify whether the stem string and the unique id integer
        are reversed in the *fmt* specification (i.e., %d before %s)

    Returns a modified path based on **stem**, a unique identifier and **ext**,
    as formatted by **fmt**.
    """
    ext = '' if ext is None else ext
    if ext:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        if stem.endswith(ext):
            stem, _ = os.path.splitext(stem)
    u = -1
    pth = f'{stem}{ext}'
    while os.path.exists(pth):
        u += 1
        pth = fmt.format(stem=stem, u=u, ext=ext)
    return pth
