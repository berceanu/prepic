# -*- coding: utf-8 -*-
"""Module containing utilities for testing dimensional analysis."""


# todo remove
def __round__(self):
    return type(self)(round(float(self)), self.units)


# todo phase out the usage and remove function
def has_units(quant, dim):
    """Checks the argument has the right dimensionality.

    Parameters
    ----------
    quant : :py:class:`unyt.array.unyt_quantity`
        Quantity whose dimensionality we want to check.
    dim : :py:class:`sympy.core.symbol.Symbol`
        SI base unit (or combination of units), eg. length/time

    Returns
    -------
    bool
        True if check successful.

    Examples
    --------
    >>> import unyt as u
    >>> from unyt.dimensions import length, time
    >>> has_units(3 * u.m/u.s, length/time)
    True
    >>> has_units(3, length)
    False
    """
    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = None
    return arg_dim == dim
