from collections import namedtuple
import unyt as u
import numpy as np

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

flame_laser = lwfa.Laser(
    ɛL=param.ɛL, τL=param.τL, beam=flame_beam
)
print(flame_laser)

flame_plasma = lwfa.Plasma(
    n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
)
print(flame_plasma)

bubble_r = (
    2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
).to("micrometer")
print(f"bubble radius is {bubble_r:.1f}")

new_plasma = lwfa.Plasma(
    n_pe=param.npe,
    laser=flame_laser,
    bubble_radius=bubble_r,
    propagation_distance=param.prop_dist,
)
print(new_plasma)

matched_flame = lwfa.matched_laser_plasma(a0=flame_plasma.laser.a0)
print(matched_flame)

sim_flame = lwfa.Simulation(flame_plasma)
print(sim_flame)