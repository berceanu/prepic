# -*- coding: utf-8 -*-
import logging

from unyt import allclose_units
from prepic._util import todict, flatten_dict

"""Module containing utilities for testing dimensional analysis."""

logger = logging.getLogger(__name__)


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
                    logger.warning(f"Difference in {key}: {self_val} vs {other_val}")
                    return False
            return True
        return False


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
    >>> _has_units(3 * u.m/u.s, length/time)
    True
    >>> _has_units(3, length)
    False
    """
    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = None
    return arg_dim == dim


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
