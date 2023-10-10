from collections import namedtuple

import unyt as u

from prepic import GaussianBeam, Laser, Plasma


Param = namedtuple("Param", ["npe", "w0", "ɛL", "τL", "prop_dist"])

param = Param(
    npe=1.0e18 / u.cm**3,
    w0=5.0 * u.micrometer,
    ɛL=1.5 * u.joule,
    τL=28.0 * u.femtosecond,
    prop_dist=10.00 * u.mm,
)

laser = Laser(ɛL=param.ɛL, τL=param.τL, beam=GaussianBeam(w0=param.w0))

plasma = Plasma(
    n_pe=param.npe,
    laser=laser,
    bubble_radius=param.w0,
    propagation_distance=param.prop_dist,
)

print(f"a0 = {plasma.laser.a0:.2f}")
print(f"P0 = {plasma.laser.P0:.2f}")
print(f"I0 = {plasma.laser.I0:.2f}")

print(
    (
        f"For a plasma with a density of {plasma.npe:.1e} we get an "
        f"electron energy gain of {plasma.ΔE:.1f} and a total accelerated charge "
        f"of {plasma.Q:.1f} over an acceleration distance of {plasma.Lacc:.1f}."
    )
)

print("\nFurther details: ")
print(f"{plasma}")
