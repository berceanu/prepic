"""
Abstract base classes and interface for prepic.
"""

import warnings

from unyt import allclose_units

from prepic.util import todict, flatten_dict


class BaseClass:
    """Implements equality testing between class instances which inherit it.

    Does a floating point comparison (including unit checking) between common
    instance attributes of two instances of a class that inherits from `BaseClass`.
    The instance attributes are collected recursively, ie. if the child class contains
    sub-classes as attributes, their attributes are collected as well.

    Examples
    --------
    >>> import unyt as u
    >>> class MyClass(BaseClass):
    ...     def __init__(self, attr1, attr2):
    ...         self.attr1 = attr1
    ...         self.attr2 = attr2
    ...
    >>> inst1 = MyClass(attr1=5.2 * u.m, attr2= 3.2 * u.s)
    >>> inst2 = MyClass(attr1=2.5 * u.m, attr2= 3.2 * u.s)
    >>> inst1 == inst2
    False
    >>> inst1 == 5.2 * u.m
    False
    """

    def __eq__(self, other):
        """Overrides the default implementation"""
        if not isinstance(other, type(self)):
            return False

        self_vars = flatten_dict(todict(self))
        other_vars = flatten_dict(todict(other))
        common_vars = self_vars.keys() & other_vars.keys()

        # instances are not equal if any of the common attributes have different values
        for key in common_vars:
            self_val = self_vars[key]
            other_val = other_vars[key]

            if not allclose_units(self_val, other_val, 1e-5):
                warnings.warn(
                    f"Difference in {key}: {self_val} vs {other_val}", RuntimeWarning
                )
                return False

        return True
