from collections import namedtuple

import unyt as u

from prepic import GaussianBeam, Laser, matched_laser_plasma

Cetal = namedtuple("Cetal", ["w0", "ɛL", "τL"])

if __name__ == "__main__":
    param = Cetal(w0=18.0 * u.micrometer, ɛL=7.7 * u.joule, τL=40.0 * u.femtosecond)

    beam = GaussianBeam.from_focal_distance(
        focal_distance=3.2 * u.meter, beam_diameter=200 * u.mm
    )
    print(f"Smallest possible beam waist for this setup is w0={beam.w0:.1f}.\n")

    laser = Laser(ɛL=param.ɛL, τL=param.τL, beam=GaussianBeam(w0=param.w0))
    print(f"{laser}\n")

    matched_plasma = matched_laser_plasma(a0=laser.a0)
    print(f"{matched_plasma}\n")

    print(f"Plasma density: {matched_plasma.npe:.1e}")
    print(f"Acceleration distance: {matched_plasma.Lacc:.1f}")
    print(f"Electron energy gain: {matched_plasma.ΔE:.1f}")
    print(f"Total accelerated charge: {matched_plasma.Q:.1f}")
