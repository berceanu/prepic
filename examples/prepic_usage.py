import pprint  # optional, pretty-print dictionaries
from collections import namedtuple  # optional, for grouping input parameters

import numpy as np
import unyt as u  # for physical units support

from prepic import lwfa

Flame = namedtuple(
    "Flame",
    [
        "npe",  # electron plasma density
        "w0",  # laser beam waist (Gaussian beam assumed)
        "ɛL",  # laser energy on target (focused into the FWHM@intensity spot)
        "τL",  # laser pulse duration (FWHM@intensity)
        "prop_dist",  # laser propagation distance (acceleration length)
    ],
)
param = Flame(
    npe=6.14e18 / u.cm ** 3,
    w0=6.94 * u.micrometer,
    ɛL=1.0 * u.joule,
    τL=30 * u.femtosecond,
    prop_dist=1.18 * u.mm,
)

flame_beam = lwfa.GaussianBeam(w0=param.w0)

print(flame_beam)

flame_laser = lwfa.Laser(ɛL=param.ɛL, τL=param.τL, beam=flame_beam)
print(flame_laser)

print(f"critical density for this laser is {flame_laser.ncrit:.1e}")
flame_laser.E0  # each attribute is a ``unyt_quantity``

print(flame_laser.beam.zR)

flame_plasma = lwfa.Plasma(
    n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
)
print(flame_plasma)

flame_plasma.laser.beam.zR

pprint.pprint(flame_plasma.__dict__)
print(f"\nThe dephasing length is {flame_plasma.dephasing:.1f}.")

bubble_r = 2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
print(f"The bubble radius is {bubble_r.to('micrometer'):.1f}.\n")

plasma_with_bubble = lwfa.Plasma(
    n_pe=param.npe,
    laser=flame_laser,
    bubble_radius=bubble_r,
    propagation_distance=param.prop_dist,
)
print(plasma_with_bubble.Q)
print(plasma_with_bubble.η.to_value('dimensionless'))

matched_plasma_flame = lwfa.matched_laser_plasma(a0=flame_laser.a0)
print(matched_plasma_flame)  # notice density, spot size, etc. changed!
print()
print(matched_plasma_flame.Q)  # compare to previous value
print(matched_plasma_flame.ΔE)  # also much improved :)

sim_flame = lwfa.Simulation(matched_plasma_flame)
print(sim_flame)
