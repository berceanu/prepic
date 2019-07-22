from collections import defaultdict

import numpy as np
import unyt as u
from cycler import cycler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from unyt._testing import assert_allclose_units

from labellines import labelLines
from prepic import lwfa

line_styles = ["-", "--", ":", "-."]
line_colors = ["C0", "C1", "C3", "C4"]

cyl = cycler(linestyle=line_styles) * cycler(color=line_colors)
loop_cy_iter = cyl()
STYLE = defaultdict(lambda: next(loop_cy_iter))

if __name__ == "__main__":
    fig = Figure(figsize=(6.4, 6.4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    beam = lwfa.GaussianBeam(w0=15.0 * u.micrometer, λL=0.8 * u.micrometer)
    electron_densities = np.logspace(-1, 1, 25) * 1e18 / (u.cm ** 3)

    for a0 in np.linspace(2.0, 8.0, 7) * u.dimensionless:
        laser = lwfa.Laser.from_a0(a0=a0, τL=30.0 * u.femtosecond, beam=beam)

        x_data = []
        y_data = []
        for npe in electron_densities:
            plasma = lwfa.Plasma(n_pe=npe, laser=laser)
            assert_allclose_units(plasma.npe, npe)
            x_data.append(plasma.npe)
            y_data.append(plasma.ΔE)

        h_axis = u.unyt_array(x_data)
        v_axis = u.unyt_array(y_data)

        ax.plot(
            h_axis.value,
            v_axis.value,
            color=STYLE[str(a0.to_value("dimensionless"))]["color"],
            linestyle=STYLE[str(a0.to_value("dimensionless"))]["linestyle"],
            label=f"$a_0 =$ {a0.to_value('dimensionless')}",
        )

        ax.set(
            ylabel=f"$\\Delta E$ [${v_axis.units.latex_repr}]$",
            ylim=[1e2, 1e4],
            xlabel=f"$n_e$ [${h_axis.units.latex_repr}$]",
            xlim=[electron_densities[0].value, electron_densities[-1].value],
        )
        ax.set_yscale("log")
        ax.set_xscale("log")

    labelLines(ax.get_lines(), align=False, fontsize=8)

    fig.suptitle(f"w0={beam.w0}, λL={beam.λL}, τL={laser.τL}")
    canvas.print_png(
        f"energy_scaling_vs_density_{str(laser.τL).replace(' ' , '_')}.png"
    )
