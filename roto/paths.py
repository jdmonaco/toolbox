"""
Functions for handling paths.
"""

import os

from toolbox import HOME


def tilde(p):
    if p.startswith(HOME):
        return f'~{p[len(HOME):]}'
    return p

def uniquify(stem, ext='', fmt='{!s}-{:02d}', reverse_fmt=False):
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
    if ext:
        ext = ext if ext.startswith('.') else f'.{ext}'
    if stem.endswith(ext):
        stem = stem[:-len(ext)]
    fullfmt = fmt + ext
    if reverse_fmt:
        head, tail = os.path.split(stem)
        filename = lambda i: os.path.join(head, fullfmt.format(i, tail))
    else:
        filename = lambda i: fullfmt.format(stem, i)
    i = 0
    nextfn = filename(0)
    while os.path.exists(nextfn):
        i += 1
        nextfn = filename(i)
    return nextfn
