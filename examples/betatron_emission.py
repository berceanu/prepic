from collections import namedtuple

import unyt as u
from prepic import Laser, GaussianBeam, Plasma, Radiator

Param = namedtuple("Param", ["npe", "w0", "ɛL", "τL", "prop_dist"])

if __name__ == "__main__":
    p1 = Param(  # 15J example from http://doi.org/f4j98s
        npe=1.1e18 / u.cm ** 3,
        w0=21.0 * u.micrometer,
        ɛL=15.0 * u.joule,
        τL=48.0 * u.femtosecond,
        prop_dist=23.0 * u.mm,
    )
    p2 = Param(  # external guiding / injection example from from http://doi.org/f4j98s
        npe=5.1e17 / u.cm ** 3,
        w0=21.0 * u.micrometer,
        ɛL=3.0 * u.joule,
        τL=47.0 * u.femtosecond,
        prop_dist=52.0 * u.mm,
    )
    p3 = Param(  # CETAL case
        npe=1.5e18 / u.cm ** 3,
        w0=18.0 * u.micrometer,
        ɛL=7.7 * u.joule,
        τL=40.0 * u.femtosecond,
        prop_dist=14.0 * u.mm,
    )

    for p in (p1, p2, p3):
        laser = Laser(ɛL=p.ɛL, τL=p.τL, beam=GaussianBeam(w0=p.w0))

        plasma = Plasma(
            n_pe=p.npe,
            laser=laser,
            bubble_radius=p.w0,
            propagation_distance=p.prop_dist,
        )

        radiator = Radiator(plasma=plasma)

        print("ɛL = %s" % p.ɛL)
        print(radiator)
        print(f"γ={radiator.γ}")
        print()
