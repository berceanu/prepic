from collections import namedtuple

import unyt as u

from prepic import lwfa

Flame = namedtuple("Flame", ["npe", "w0", "ɛL", "τL", "prop_dist"])

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

    # vary plasma density #
    npe_values = map(lambda v: round(v, 1), (1e18, 2e18, 3e18, 6.14e18))
    plasmas = {}
    for v in npe_values:
        plasmas[v] = lwfa.Plasma(
            n_pe=v / u.cm ** 3, laser=flame_laser, propagation_distance=param.prop_dist
        )

    print(plasmas[2e18])
