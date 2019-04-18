"""
Functions to help in the creation of complex figures.
"""

import os
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt

from pouty import log


DPI = mpl.rcParams['figure.dpi']


def test_svg_conversion(svgfile):
    info = get_svg_figinfo(svgfile)

    wasint = plt.isinteractive()
    plt.ioff()

    fig = plt.figure(figsize=info['figsize'])
    for name, rect in list(info['axes'].items()):
        ax = plt.axes(rect)
        ax.text(0.5, 0.5, name, ha='center', va='center', size='medium',
                weight='bold')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    savefn = os.path.splitext(svgfile)[0] + '.pdf'
    fig.savefig(savefn)
    log('Saved: {}', savefn, prefix='TestSVGConversion')

    if wasint:
        plt.ion()

def get_svg_figinfo(svgfile):
    """Get figure and axes information based on an SVG file.

    Input file should contain only rectangles and conform to Tiny 1.1 or 1.2
    specification.

    Returns dict with 'figsize' tuple and 'axes' list, where each item is a
    dict with a 'name' string and 'rect' position list for `plt.axes()`.
    """
    nodes = _parse_svg(svgfile)
    info = {}

    # Get page dimensions from viewBox attributes
    svg = list(filter(lambda x: x['type'] == 'svg', nodes))[0]
    if 'viewBox' in svg:
        x0, y0, x1, y1 = tuple(map(float, svg['viewBox'].split('_')))
        svg_width = x1 - x0
        svg_height = y1 - y0
    else:
        svg_width = mpl.rcParams['figure.figsize'][0] * DPI
        svg_height = mpl.rcParams['figure.figsize'][1] * DPI
    info['figsize'] = svg_width / DPI, svg_height / DPI

    # Get axes positions
    rects = [x for x in nodes if x['type'] == 'rect']
    for rect in rects:
        rect['x'] /= svg_width  # xform to [0,1] figure coordinates
        rect['width'] /= svg_width
        rect['y'] /= svg_height
        rect['top'] = rect['y']
        rect['height'] /= svg_height
        rect['y'] = 1 - rect['y'] - rect['height']  # calculate bottom

    # Sort axes by top-left corner position
    rects.sort(key=itemgetter('top', 'x'))

    # Store the axes names and position rectangles
    autoname = 65  # start at 'A'
    info['axes'] = {}
    for rect in rects:
        if 'name' in rect:
            name = rect['name']
        else:
            name = chr(autoname)
            autoname += 1
        info['axes'][name] = [rect[k] for k in ('x', 'y', 'width', 'height')]

    return info

def _parse_svg(filename):
    with open(filename, 'r') as fd:
        svg = fd.read()

    descrs = []
    i = 0
    while i < len(svg):
        if svg[i] != "<":
            i += 1
            continue
        tag_end = svg.find(' ', i)
        tag = svg[i+1:tag_end]
        if tag[0] in '?!/':
            i += 1
            continue
        s = 'type="%s"' % tag
        i = tag_end
        inquote = False
        while svg[i] not in '/>' or inquote:
            if svg[i] == '"':
                inquote = not inquote
            if inquote and svg[i] == ' ':
                s += '_'
            elif not inquote and svg[i] in ':-':
                s += '_'
            else:
                s += svg[i]
            i += 1
        descrs.append(s.split())

    nodes = []
    for attrs in descrs:
        node = eval('dict(%s)' % (','.join(attrs)))
        for key in node:
            if node[key].endswith('px'):
                node[key] = node[key][:-2]
            try:
                num = float(node[key])
            except ValueError:
                pass
            else:
                node[key] = num
        nodes.append(node)
    return nodes
