"""
Functions for handling paths.
"""

import os

from toolbox import HOME


def truncate(p):
    if p.startswith(HOME):
        return f'~{p[len(HOME):]}'
    return p

def uniquify(stem, ext="", fmt="%s%02d", reverse_fmt=False):
    """Insert a unique identifier into a file or directory path

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
        ext = ext.startswith('.') and ext or ('.' + ext)
    if reverse_fmt:
        head, tail = os.path.split(stem)
        filename = lambda i: os.path.join(head, fmt%(i, tail)) + ext
    else:
        filename = lambda i: fmt%(stem, i) + ext
    i = 0
    while os.path.exists(filename(i)):
        i += 1
    return filename(i)
