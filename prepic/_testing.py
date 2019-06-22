# -*- coding: utf-8 -*-
from itertools import chain

import unyt as u
from unyt._testing import assert_allclose_units
from prepic._util import todict, flatten_dict

"""Module containing utilities for testing dimensional analysis."""


def allclose_units(actual, desired, rtol=1e-7, atol=0, **kwargs):
    try:
        assert_allclose_units(actual, desired, rtol, atol, **kwargs)
    except AssertionError:
        return False
    return True


def __round__(self):
    return type(self)(round(float(self)), self.units)


class BaseClass:
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, type(self)):
            self_vars = flatten_dict(todict(self))
            other_vars = flatten_dict(todict(other))
            common_vars = self_vars.keys() & other_vars.keys()

            for key in common_vars:
                self_val = self_vars[key]
                other_val = other_vars[key]
                if not allclose_units(self_val, other_val, 1e-5):
                    # print(f"Difference in {key}: {self_val} vs {other_val}")
                    return False
            return True
        return False


class UnitsError(ValueError):
    pass


def right_units(arg, dim):
    """Checks the argument has the right dimensionality.

    :param arg: variable to check
    :type arg: :py:class:`unyt.array.unyt_quantity`
    :param dim: SI base quantity, eg. 'time', 'length', etc.
    :type dim: str
    :return: True if check successful
    :rtype: bool
    :raises AttributeError: if `arg` doesn't have `.units` attribute
    """
    try:
        arg_dim = arg.units.dimensions
    except AttributeError:
        arg_dim = None
    return arg_dim == getattr(u.dimensions, dim)


def check_dimensions(**arg_units):
    """Decorator for checking units of function arguments.

    :param arg_units: dictionary of the form {'arg1': dimension1, etc}
    :type arg_units: dict
    :return: :func:`check_nr_args`
    """

    def check_nr_args(f):
        """Check correct number of arguments and decorate :func:`f`

        :param f: original function being decorated
        :return: :func:`new_f` decorated func
        """
        number_of_args = f.__code__.co_argcount
        names_of_args = f.__code__.co_varnames

        assert (
            len(arg_units) == number_of_args
        ), f'decorator number of arguments not equal with function number of arguments in "{f.__name__}"'

        def new_f(*args, **kwargs):
            """The new function being returned from the decorator.

            Checks units of `args` and `kwargs`, then run original function.

            :param args: positional arguments of :func:`f`
            :param kwargs: keyword arguments of :func:`f`
            :return: result of original function :func:`f`
            :raises UnitsError: if the units don't match
            """
            for arg_name, arg_value in chain(zip(names_of_args, args), kwargs.items()):
                dimension = arg_units[arg_name]
                if arg_name in arg_units and not right_units(arg_value, dimension):
                    raise UnitsError(
                        f"arg '{arg_name}'={repr(arg_value)} does not match {dimension}"
                    )
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_nr_args
