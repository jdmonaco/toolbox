"""
Functions for manipulating strings.
"""

import re


def snake2title(s):
    """Convert 'snake_case' string to 'Title Case' string."""
    return ' '.join(s.split('_')).strip().title()

def camel2snake(name, sep='_'):
    """Convert a camel-case name to snake case.

    Shamelessly based on a Stackoverflow answer:
    http://stackoverflow.com/a/1176023
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1%s\2' % sep, name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1%s\2' % sep, s1).lower()
    return s2.replace('%s%s' % (sep, sep), sep)

def parse_value(value):
    """Convert an expression to a python value."""
    value = to_str(value)
    try:
        value = eval(value)
    except (SyntaxError, NameError, TypeError):
        print('parse_value: could not evaluate \'{}\''.format(value))
    return value

def to_float(arg):
    """Attempt to convert a value to float, but fall back to string."""
    try:
        res = float(arg)
    except ValueError:
        res = arg
    return res


# For Unicode decoding/encoding

def to_str(bytes_or_str):
    """Given a string or bytes instances, return a Unicode str."""
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value

def to_bytes(bytes_or_str):
    """Given a string or bytes instances, return a bytes object."""
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value
