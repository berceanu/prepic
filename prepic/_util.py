# -*- coding: utf-8 -*-
from collections import Iterable

from unyt.array import unyt_quantity

"""Module containing various utilities that don't really fit anywhere else."""


def iteritems_nested(d):
    """
    Collect dictionary keys until the deepest level.
    """

    def fetch(suffixes, v0):
        if isinstance(v0, dict):
            for k, v in v0.items():
                yield from fetch(suffixes + [k], v)
        else:
            yield (suffixes, v0)

    return fetch([], d)


def flatten_dict(d):
    """
    Concatenate dictionary keys.
    """
    return dict(('.'.join(ks), v) for ks, v in iteritems_nested(d))


def todict(obj, baseobj=unyt_quantity):
    """
    Recursively convert a Python object graph to sequences (lists)
    and mappings (dicts) of primitives (bool, int, float, string, ...)
    """
    if isinstance(obj, baseobj):
        return obj
    elif isinstance(obj, dict):
        return dict((key, todict(val)) for key, val in obj.items())
    elif isinstance(obj, Iterable):
        return [todict(val) for val in obj]
    elif hasattr(obj, '__dict__'):
        return todict(vars(obj))
    elif hasattr(obj, '__slots__'):
        return todict(dict((name, getattr(obj, name)) for name in getattr(obj, '__slots__')))
    return obj
