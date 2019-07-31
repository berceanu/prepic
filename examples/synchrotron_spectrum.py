from collections import namedtuple

import unyt as u

from prepic import GaussianBeam, Laser, Plasma, Radiator

Param = namedtuple("Param", ["npe", "w0", "ɛL", "τL", "prop_dist"])

p = Param(  # external guiding / injection example from from http://doi.org/f4j98s
    npe=5.1e17 / u.cm ** 3,
    w0=21.0 * u.micrometer,
    ɛL=3.0 * u.joule,
    τL=47.0 * u.femtosecond,
    prop_dist=52.0 * u.mm,
)

laser = Laser(ɛL=p.ɛL, τL=p.τL, beam=GaussianBeam(w0=p.w0))

plasma = Plasma(
    n_pe=p.npe, laser=laser, bubble_radius=p.w0, propagation_distance=p.prop_dist
)

radiator = Radiator(plasma=plasma)

fig = radiator.frequency_spectrum().plot()

fig.savefig("frequency.png")
