import numpy as np
import unyt as u

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from prepic.radiation import photon_frequency_distribution

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
