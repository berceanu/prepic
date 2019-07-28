"""
Model ionization processes

"""
from unyt import accepts
from unyt.dimensions import flux
import unyt as u


@accepts(i0=flux)
def helium_ionization_state(i0):
    """Compute the ionization state of Helium.

    Parameters
    ----------
    i0: float, energy/time/area
        Peak laser intensity in the focal plane.
    """
    if i0 < 1.4e15 * u.watt / u.cm ** 2:
        return "0+"
    elif i0 < 8.8e15 * u.watt / u.cm ** 2:
        return "1+"
    else:
        return "2+"
