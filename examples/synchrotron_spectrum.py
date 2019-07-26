import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import unyt as u

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def s_function(y):
    y = y.to_value("dimensionless")
    integral = integrate.quad(lambda x: special.kv(5 / 3, x), y, np.inf)
    if integral[1] > 1e-05:
        raise FloatingPointError(
            "S integration error too large, %s at y = %s" % (integral[1], y)
        )
    else:
        return integral[0]


def total_radiated_energy(ωc, γ):
    """Computes total energy radiated per betatron oscillation.
    Jackson Section 14.6"""
    intensity = 2 / 9 * u.qe ** 2 / (u.eps_0 * u.clight) * ωc * γ
    return intensity.to("keV")


def photon_frequency_distribution(ω, ωc, γ):
    """Computes the number of photons per unit frequency interval,
    integrated over all angles. Also per betatron oscillation and per electron.
    Jackson Section 14.6"""
    a = 9 * np.sqrt(3) / (8 * np.pi)  # prefactor
    y = (ω / ωc).to("dimensionless")
    dN_over_dy = a * total_radiated_energy(ωc, γ) / (u.hbar * ωc) * y * s_function(y)
    return dN_over_dy.to("dimensionless")


def photon_angle_distribution(ω, ωc, γ, θ):
    """Computes the number of photons per unit solid angle, integrated over
    all frequencies, observed at an angle θ from the electron's plane of oscillation.
    Jackson Section 14.6"""
    a = (7 * u.qe ** 2 * ωc / (96 * np.pi * u.eps_0 * u.clight) / (u.hbar * ωc)).to(
        "dimensionless"
    )
    dN_over_dΩ = (
        a
        * γ ** 2
        / ((1 + γ ** 2 * θ ** 2) ** (5 / 2))
        * (1 + 5 / 7 * γ ** 2 * θ ** 2 / (1 + γ ** 2 * θ ** 2))
    )
    return dN_over_dΩ.to("dimensionless")


ωc = (197.0 * u.kiloelectronvolt / u.hbar).to(1 / u.fs)
ω_avg = (60.7 * u.keV / u.hbar).to(1 / u.fs)

print(ω_avg / ωc * 55)

omegas = np.linspace(1e-5 * ωc, 2 * ωc, 100)
freq_distr = np.empty(omegas.size) * u.dimensionless


for i, ω in enumerate(omegas):
    freq_distr[i] = photon_frequency_distribution(
        ω=ω, ωc=ωc, γ=4956.472175044521 * u.dimensionless
    )

norm_freq = (omegas / ωc).to_value("dimensionless")
distr = freq_distr.to_value("dimensionless")

print(distr)

fig = Figure()  # figsize=(6.4, 6.4)
fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)

ax.plot(norm_freq, distr)

ax.fill_between(norm_freq, distr, where=norm_freq < 1, facecolor="C3", alpha=0.5)

ax.axvline(x=ω_avg / ωc, linestyle="--", color="C3")
ax.text(ω_avg / ωc, 0, r"$\langle \omega \rangle$")

ax.set(ylabel=r"$\frac{dN}{dy}$", xlabel=r"$y = \omega / \omega_c$", xlim=[-0.1, 2.0])
ax.set_ylim(bottom=0)

width = 6.4  # 3.487
height = width / 1.618

fig.set_size_inches(width, height)
canvas.print_png(f"synchrotron.png")
