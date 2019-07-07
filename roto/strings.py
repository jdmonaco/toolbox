"""
Functions on strings.
"""

import re


def naturalize(s):
    """Normalize to 'natural' naming for identifiers or data storage."""
    return camel2snake(s).strip().lower().replace(' ', '_').replace('-', '_'
            ).replace('.', '_')

def sluggify(s):
    """Normalize to a url-style slug: hyphenated lower-case words."""
    return camel2snake(s, sep='-').lower().strip().replace(' ', '-')

def camel2snake(s, sep='_'):
    """Convert a camel-case name to snake case.

    Shamelessly stolen from a Stackoverflow answer:
    http://stackoverflow.com/a/1176023
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1%s\2' % sep, s)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1%s\2' % sep, s1).lower()
    return s2.replace('%s%s' % (sep, sep), sep)

def snake2title(s):
    """Convert 'snake_case' string to 'Title Case' string."""
    return ' '.join(s.split('_')).strip().title()


# Unicode decoding/encoding

def to_str(bytes_or_str):
    """Given a string or bytes instance, return a string."""
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode('utf-8')
    else:
        value = bytes_or_str
    return value

def to_bytes(bytes_or_str):
    """Given a string or bytes instance, return a bytes object."""
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode('utf-8')
    else:
        value = bytes_or_str
    return value
