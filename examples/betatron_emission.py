from collections import namedtuple

import unyt as u
from prepic import lwfa

Param = namedtuple("Param", ["npe", "w0", "ɛL", "τL", "prop_dist"])

if __name__ == "__main__":
    p1 = Param(
        npe=1.1e18 / u.cm ** 3,
        w0=21.0 * u.micrometer,
        ɛL=15.0 * u.joule,
        τL=48.0 * u.femtosecond,
        prop_dist=23.0 * u.mm,
    )
    p2 = Param(
        npe=5.1e17 / u.cm ** 3,
        w0=21.0 * u.micrometer,
        ɛL=3.0 * u.joule,
        τL=47.0 * u.femtosecond,
        prop_dist=52.0 * u.mm,
    )
    p3 = Param(
        npe=1.5e18 / u.cm ** 3,
        w0=18.0 * u.micrometer,
        ɛL=7.7 * u.joule,
        τL=40.0 * u.femtosecond,
        prop_dist=14.0 * u.mm,
    )

    for p in (p1, p2, p3):
        laser = lwfa.Laser(ɛL=p.ɛL, τL=p.τL, beam=lwfa.GaussianBeam(w0=p.w0))

        plasma = lwfa.Plasma(
            n_pe=p.npe,
            laser=laser,
            bubble_radius=p.w0,
            propagation_distance=p.prop_dist,
        )

        radiator = lwfa.Radiator(plasma=plasma)

        print("ɛL = %s" % p.ɛL)
        print(radiator)
        print()
