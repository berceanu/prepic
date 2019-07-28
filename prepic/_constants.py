"""
Various physical constants

"""
import numpy as np
import unyt as u

# classical electron radius
r_e = (1 / (4 * np.pi * u.eps_0) * u.qe ** 2 / (u.me * u.clight ** 2)).to("micrometer")
