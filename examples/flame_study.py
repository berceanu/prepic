from collections import namedtuple

import unyt as u

from prepic import lwfa

Flame = namedtuple("Flame", ["npe", "w0", "ɛL", "τL", "prop_dist", "f_dist", "diam"])

if __name__ == "__main__":
    # we take the pulse energy to be the energy in focus, i.e. 40% of 3J.
    param = Flame(
        npe=1.0e18 / u.cm ** 3,
        w0=15.0 * u.micrometer,
        ɛL=1.2 * u.joule,
        τL=28.0 * u.femtosecond,
        prop_dist=5.0 * u.mm,
        f_dist=1.0 * u.meter,
        diam=100 * u.mm,
    )

    beam = lwfa.GaussianBeam.from_focal_distance(
        focal_distance=param.f_dist, beam_diameter=param.diam
    )
    print(
        (
            f"The diffraction-limited beam waist for an OAP with focal lenght of {param.f_dist:.1f} "
            f"and a laser beam diameter of {param.diam:.1f} is w0={beam.w0:.1f} at 1/e^2 intensity.\n"
        )
    )

    laser = lwfa.Laser(ɛL=param.ɛL, τL=param.τL, beam=lwfa.GaussianBeam(w0=param.w0))

    plasma = lwfa.Plasma(
        n_pe=param.npe,
        laser=laser,
        bubble_radius=param.w0,
        propagation_distance=param.prop_dist,
    )

    print(
        (
            f"For a plasma with a density of {plasma.npe:.1e} we get an "
            f"electron energy gain of {plasma.ΔE:.1f} and a total accelerated charge "
            f"of {plasma.Q:.1f} over an acceleration distance of {plasma.Lacc:.1f}."
        )
    )

    print("\nFurther details: ")
    print(f"{plasma}")
