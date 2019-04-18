"""
Write/clone tables/arrays/groups to/between HDF files.
"""

import os

import numpy as np
import tables as tb

from pouty import log

from .datapath import join, split

class OverwriteDisallowed(Exception):
    pass


def new_table(f, where, name, description, force=False, **kwargs):
    """Add `force` keyword to tables.File.create_table().

    Arguments:
    force -- force erasure of existing table without asking
    h5file -- alternate file in which to create the array

    Returns `tables.Table` node.
    """
    try:
        table = f.get_node(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        if force:
            log('new_table: Erasing {} table', table._v_pathname)
            do_erase = 'y'
        else:
            do_erase = input(
                'new_table: Erase %s table? (y/N) ' % table._v_pathname)
        if not do_erase.lower().strip().startswith('y'):
            raise OverwriteDisallowed('%s table already exists' %
                    table._v_pathname)
        f.remove_node(table)

    kwargs.update(description=description)
    return f.create_table(where, name, **kwargs)

def has_node(f, where, name=None):
    """Indicate whether a node exists at the specified path."""
    try:
        node = f.get_node(where, name=name)
    except tb.NoSuchNodeError:
        return False
    return True

def new_array(f, where, name, x, overwrite=True, **kwargs):
    """Add `overwrite` keyword to tables.File.create_array().

    Note: Masked arrays are handled by creating a Group at the specifed path
    and saving the data and mask buffers as Array objects within that Group.
    An attribute is set on the Group node indicating to read_array() that the
    child Arrays should be read and returned as a masked array.

    Arguments:
    overwrite -- automatically remove array if it already exists
    h5file -- alternate file in which to create the array

    Returns `tables.Array` node.
    """
    try:
        array = f.get_node(where, name=name)
    except tb.NoSuchNodeError:
        pass
    else:
        if not overwrite:
            raise OverwriteDisallowed('%s array already exists' %
                    array._v_pathname)
        f.remove_node(array, recursive=True)

    if isinstance(x, np.ma.MaskedArray):
        grp = new_group(f, where, name)
        grp._v_attrs['masked_array_group'] = True
        f.create_array(grp, 'data', obj=x.data, title='Data Buffer')
        f.create_array(grp, 'mask', obj=x.mask, title='Mask Buffer')
        return grp
    else:
        return f.create_array(where, name, obj=x, **kwargs)

def read_array(f, where, name=None):
    """Read array data from the given node."""
    try:
        node = f.get_node(where, name=None)
    except tb.NoSuchNodeError:
        log('{} does not exist', os.path.join(where, name), error=True)
        return

    if isinstance(node, tb.Group) and 'masked_array_group' in node._v_attrs:
        arr = np.ma.MaskedArray(data=node.data.read(), mask=node.mask.read())
    elif isinstance(node, tb.Array):
        arr = node.read()
    else:
        raise TypeError('Not an array node')
    return arr

def new_dataframe(pdf, where, name, x, overwrite=True, **kwargs):
    """Add `overwrite` keyword to tables.File.create_array().

    Arguments:
    pdf -- pandas file store
    overwrite -- automatically remove array if it already exists
    h5file -- alternate file in which to create the array

    Returns `tables.Group` node where the dataframe is stored.
    """
    key = join(where, name)
    try:
        df = pdf.get_node(key)
    except KeyError:
        pass
    else:
        if not overwrite:
            raise OverwriteDisallowed('%s dataframe already exists' %
                    df._v_pathname)

    pdf[key] = x
    grp = pdf.get_node(key)
    grp._v_attrs['title'] = kwargs.pop('title', 'Dataframe')
    return grp

def new_group(f, where, name, **kwargs):
    """Enforce `createparents=True` in creating a new group."""
    kwargs.update(createparents=True)

    try:
        group = f.create_group(where, name, **kwargs)
    except tb.NodeError:
        group = f.get_node(where, name=name)
        log('{} already exists', group._v_pathname, error=True)
    return group

def clone_node(node, destfile, parent=None, name=None):
    """Clone a node from one file to another file."""
    name = node._v_name if name is None else name
    parent = node._v_parent._v_pathname if parent is None else parent
    destpath = join(parent, name)

    if isinstance(node, tb.Array):
        log('Cloning array: {} to {}', node._v_pathname, destpath)
        arr = clone = new_array(destfile, parent, name, node.read(),
            createparents=True, title=node.title)
        for k in node._v_attrs._v_attrnames:
            setattr(arr._v_attrs, k, node._v_attrs[k])
    elif isinstance(node, tb.Table):
        log('Cloning table: {} to {}', node._v_pathname, destpath)
        tbl = clone = new_table(destfile, parent, name, node.description,
            createparents=True, title=node.title)
        row = tbl.row
        for record in node.iterrows():
            for col in node.colnames:
                row[col] = record[col]
            row.append()
        for k in node._v_attrs._v_attrnames:
            setattr(tbl._v_attrs, k, node._v_attrs[k])
        tbl.flush()

    return clone

def clone_subtree(srcfile, tree, destfile, destroot='/', classname=None):
    """Clone a subtree within one file into a different (possibly new) file."""
    if type(srcfile) is str:
        srcfile = tb.open_file(srcfile, 'r')

    if not srcfile.isopen:
        raise IOError('Source file is not open')

    if type(destfile) is str:
        destfile = tb.open_file(destfile, 'a')

    if not (destfile.isopen and destfile._iswritable()):
        raise IOError('Destination file is not open and writable')

    try:
        srctree = srcfile.get_node(tree)
    except tb.NoSuchNodeError as e:
        log('Source group does not exist: {}', tree, error=True)
        return

    for node in srcfile.walk_nodes(srctree, classname=classname):
        if isinstance(node, tb.Group):
            continue
        destpath = join(destroot, node._v_pathname[1:])
        parent, name = split(destpath)
        clone_node(node, destfile, parent=parent, name=name)

    log('Finished cloning subtree: {}', srctree._v_pathname)
    return destfile.get_node(destroot)
