# -*- coding: utf-8 -*-
"""
Utility functions

"""
from collections import Iterable

from unyt.array import unyt_quantity


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
    return dict((".".join(ks), v) for ks, v in iteritems_nested(d))


def todict(obj, baseobj=unyt_quantity):
    """ Convert a Python object to primitives.

    Recursively convert a Python object graph to sequences (lists)
    and mappings (dicts) of primitives (bool, int, float, string, ...)

    Examples
    --------
    >>> import unyt as u
    >>> mytuple = (3 * u.m, 2 * u.s, 6 * u.m/u.s)
    >>> todict(mytuple)
    [unyt_quantity(3, 'm'), unyt_quantity(2, 's'), unyt_quantity(6, 'm/s')]
    >>> class MyClass:
    ...     __slots__ = ['distance', 'speed']
    ...     def __init__(self, distance, speed):
    ...         self.distance = distance
    ...         self.speed = speed
    ...
    >>> slotted = MyClass(distance=3 * u.m, speed=6 * u.m/u.s)
    >>> todict(slotted)
    {'distance': unyt_quantity(3, 'm'), 'speed': unyt_quantity(6, 'm/s')}
    """
    if isinstance(obj, baseobj):
        return obj
    elif isinstance(obj, dict):
        return dict((key, todict(val)) for key, val in obj.items())
    elif isinstance(obj, Iterable):
        return [todict(val) for val in obj]
    elif hasattr(obj, "__dict__"):
        return todict(vars(obj))
    elif hasattr(obj, "__slots__"):
        return todict(
            dict((name, getattr(obj, name)) for name in getattr(obj, "__slots__"))
        )
    return obj
