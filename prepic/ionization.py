"""
Model ionization processes

"""
import unyt as u

dim = u.dimensions


@u.accepts(i0=dim.flux)
def helium_ionization_state(i0):
    """Compute the ionization state of Helium.

    Parameters
    ----------
    i0: float, energy/time/area
        Peak laser intensity in the focal plane.

    Examples
    --------
    >>> print(helium_ionization_state(1e14 * u.watt / u.cm ** 2))
    0+
    >>> print(helium_ionization_state(4.4e15 * u.watt / u.cm ** 2))
    1+
    >>> print(helium_ionization_state(1e17 * u.watt / u.cm ** 2))
    2+

    """
    if i0 < 1.4e15 * u.watt / u.cm ** 2:
        return "0+"
    if i0 < 8.8e15 * u.watt / u.cm ** 2:
        return "1+"
    return "2+"
