from collections import namedtuple

import numpy as np
import unyt as u

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

if __name__ == "__main__":
    param = Flame(
        npe=6.14e18 / u.cm ** 3,
        w0=6.94 * u.micrometer,
        ɛL=1.0 * u.joule,
        τL=30 * u.femtosecond,
        prop_dist=1.18 * u.mm,
    )

    flame_beam = lwfa.GaussianBeam(w0=param.w0)

    flame_laser = lwfa.Laser(ɛL=param.ɛL, τL=param.τL, beam=flame_beam)

    flame_plasma = lwfa.Plasma(
        n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
    )

    bubble_r = (2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp).to("micrometer")

    new_plasma = lwfa.Plasma(
        n_pe=param.npe,
        laser=flame_laser,
        bubble_radius=bubble_r,
        propagation_distance=param.prop_dist,
    )

    matched_flame = lwfa.matched_laser_plasma(a0=flame_plasma.laser.a0)

    sim_flame = lwfa.Simulation(flame_plasma)
