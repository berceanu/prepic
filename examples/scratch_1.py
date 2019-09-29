from collections import namedtuple
from functools import partial

import numpy as np
import unyt as u
from matplotlib import pyplot
from sliceplots import Plot2D, plot1d

from prepic import Plasma, Laser, GaussianBeam, Radiator
from prepic.radiation import differential_intensity_distribution

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

dist_func = partial(
    differential_intensity_distribution, ωc=radiator.ωc, γ=radiator.γ
)  # = f(ω, θ)

# todo factor of hbar
ω = np.linspace(1e-5 * radiator.ωc, 2 * radiator.ωc, 50)
θ = np.linspace(0, 0.4, 50) * u.miliradian  # todo change back to 50

ωω, θθ = np.meshgrid(ω, θ)

spectrum = dist_func(ωω, θθ)

fig = pyplot.figure(figsize=(8, 8))

print("ω_avg = {:.1e}".format(radiator.ω_avg))
print("ωc = {:.1e}".format(radiator.ωc))

Plot2D(
    fig=fig,
    arr2d=spectrum,
    h_axis=ω,
    v_axis=θ,
    xlabel=r"$\omega$ [%s]" % ω.units,
    ylabel=r"$\theta$ [%s]" % θ.units,
    zlabel=r"$\frac{d^2I}{d \hbar \omega d \Omega}$",
    hslice_val=0.2 * u.miliradian,
    vslice_val=radiator.ωc,
    hslice_opts={"color": "#1f77b4", "lw": 1.5, "ls": "-"},
    vslice_opts={"color": "#d62728", "ls": "-"},
    cbar=False,
)

np.trapz(spectrum, axis=0)  # (50,) f(ω) // integrate over θ
np.trapz(spectrum, axis=1)  # (10,) f(θ) // integrate over ω


_, ax = pyplot.subplots()
plot1d(
    ax=ax,
    h_axis=ω,
    v_axis=np.trapz(spectrum, axis=0),
    xlabel=r"$\omega$ [%s]" % ω.units,
    ylabel=r"$\frac{dI}{d \hbar \omega}$",
    color="#d62728",
)

_, ax = pyplot.subplots()
plot1d(
    ax=ax,
    h_axis=θ,
    v_axis=np.trapz(spectrum, axis=1),
    xlabel=r"$\theta$ [%s]" % θ.units,
    ylabel=r"$\frac{dI}{d \Omega}$",
    color="#d62728",
)


a = np.arange(6).reshape(2, 3)
# array([[0, 1, 2],
#        [3, 4, 5]])
np.trapz(a, axis=0)
# array([1.5, 2.5, 3.5])
np.trapz(a, axis=1)
# array([2.,  8.])
