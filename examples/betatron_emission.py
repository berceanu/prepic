from collections import namedtuple

import unyt as u
import numpy as np
from prepic import lwfa

Param = namedtuple("Param", ["npe", "w0", "ɛL", "τL", "prop_dist"])
mc2 = (u.me * u.clight ** 2).to("megaelectronvolt")

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

    α = 1510.3 * u.micrometer ** (-1 / 2)
    β = 3 * np.sqrt(2.0e19) * u.cm ** (-3 / 2)

    for p in (p1, p2, p3):
        laser = lwfa.Laser(ɛL=p.ɛL, τL=p.τL, beam=lwfa.GaussianBeam(w0=p.w0))

        plasma = lwfa.Plasma(
            n_pe=p.npe,
            laser=laser,
            bubble_radius=p.w0,
            propagation_distance=p.prop_dist,
        )

        print(f"{plasma}\n")

        rβ = (α * np.sqrt(laser.a0 / plasma.npe)).to("micrometer")

        γ = (plasma.ΔE / mc2).to_value("dimensionless")

        K = (np.sqrt(γ / 2) * plasma.kp * rβ).to_value("dimensionless")
        print(f"K={K:.1f}")

        θr = 1.0e03 * K / γ * u.miliradian
        print(f"θr={θr:.1f}")

        λu = (np.sqrt(2 * γ) * plasma.λp).to("mm")
        ħωc = 1.86e-03 * K * γ ** 2 / λu.to_value("micrometer") * u.kiloelectronvolt
        print(f"ħωc={ħωc:.1f}")

        Nγ = 3.31e-02 * K
        print(f"Nγ={Nγ:.1f} photons / (electron x betatron period)")

        Nβ = (β / np.sqrt(plasma.npe)).to_value("dimensionless")
        print(f"Nβ = {Nβ:.1f} betatron oscillations")
        Nshot = Nγ * plasma.N.to_value("dimensionless") * Nβ
        print(f"Nshot= {Nshot:.1e} photons per shot\n\n")
