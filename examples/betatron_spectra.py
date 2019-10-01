from collections import namedtuple

import unyt as u
from matplotlib import pyplot

from prepic import Plasma, Laser, GaussianBeam, Radiator
from prepic.radiation import DifferentialSpectrum

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

############################

ds = DifferentialSpectrum(radiator)

fig_dd = pyplot.figure(figsize=(8, 8))

ds.doubly_differential(fig_dd)

fig_dd.savefig("doubly_differential.png")


fig, axarr = pyplot.subplots(
    nrows=1, ncols=2, constrained_layout=True, figsize=(2 * 6.4, 4.8)
)
axes = {"angle": axarr.flat[0], "energy": axarr.flat[1]}

ds.angle_integrated(axes["angle"])
ds.energy_integrated(axes["energy"])

fig.savefig("integrated.png")

#############################

print("ħω_avg = {:.1f}".format(radiator.ħω_avg))
print("ħωc = {:.1f}".format(radiator.ħωc))
