from itertools import chain
from numpy import pi as π

import unyt as u

r_e = (1 / (4 * π * u.eps_0) * u.qe ** 2 / (u.me * u.clight ** 2)).to("micrometer")


class UnitsError(ValueError):
    pass


def right_units(arg, dimension):
    """Checks the argument has the right dimensionality.

    :param arg: variable to check
    :type arg: :py:class:`unyt.array.unyt_quantity`
    :param dimension: SI base quantity, eg. 'time', 'length', etc.
    :type dimension: str
    :return: True if check successful
    :rtype: bool
    :raises AttributeError: if `arg` doesn't have `.units` attribute
    """
    try:
        arg_dimension = arg.units.dimensions
    except AttributeError:
        arg_dimension = None
    return arg_dimension == getattr(u.dimensions, dimension)


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


@check_dimensions(a0="dimensionless", λL="length")
def intensity_from_a0(a0, λL=0.8 * u.micrometer):
    """Compute peak laser intensity in the focal plane.

    Args:
        a0 (float, dimensionless): normalized laser vector potential
        λL (float, length): laser wavelength

    Returns:
        I0 (float, energy/time/area): peak laser intensity in the focal plane
    """
    return π / 2 * u.clight / r_e * u.me * u.clight ** 2 / λL ** 2 * a0 ** 2


if __name__ == "__main__":
    res = intensity_from_a0(a0=7.202530529256849*u.dimensionless, λL=0.8*u.second) # raises
    print(res)
