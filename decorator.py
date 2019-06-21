import unyt as u


def check_arg(arg, dimension):
    """Checks the argument has the right dimensionality.

    :param arg: variable to check
    :type arg: :py:class:`unyt.array.unyt_quantity`
    :param dimension: SI base quantity, eg. 'time', 'length', etc.
    :type dimension: str
    """
    msg = f"arg must have units of {dimension}"
    try:
        arg_dimension = arg.units.dimensions
    except AttributeError:
        arg_dimension = None
    assert arg_dimension == getattr(u.dimensions, dimension), msg


def dimensions(exception_name, **arg_types):
    def check_nr_args(f):
        number_of_args = f.__code__.co_argcount
        names_of_args = f.__code__.co_varnames

        assert (
            len(arg_types) == number_of_args
        ), f'accept number of arguments not equal with function number of arguments in "{f.__name__}"'

        def new_f(*args, **kwargs):
            for arg_name, arg_value in zip(names_of_args, args):
                if arg_name in arg_types and not isinstance(
                    arg_value, arg_types[arg_name]
                ):
                    raise exception_name(
                        f"arg '{arg_name}'={repr(arg_value)} does not match {arg_types[arg_name]}"
                    )
            for arg_name, arg_value in kwargs.items():
                if arg_name in arg_types and not isinstance(
                    arg_value, arg_types[arg_name]
                ):
                    raise exception_name(
                        f"arg '{arg_name}'={repr(arg_value)} does not match {arg_types[arg_name]}"
                    )
            return f(*args, **kwargs)

        new_f.__name__ = f.__name__
        return new_f

    return check_nr_args


@dimensions(Exception, a=str, b=list, c=int)
def foo(a, b=None, c=15):
    msg = f"{a} + {b} + {c}"
    return msg


# @dimensions('length')
# def w0_to_fwhm(w0):
#     """Computes Gaussian laser FWHM from its beam waist.
#
#     Args:
#         w0 (float, length): beam waist @ 1/e^2 intensity
#
#     Returns:
#         fwhm (float, length): beam FWHM @ 1/2 intensity
#     """
#     check_arg(w0, 'length')
#     return 2 * w0 / np.sqrt(2 / np.log(2))


if __name__ == "__main__":
    res = foo("bla", b=[], c=17)
    print(res)


