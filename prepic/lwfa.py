# -*- coding: utf-8 -*-
"""Laser WakeField Acceleration module."""
import numpy as np
import unyt as u
from unyt import accepts
from unyt.dimensions import time

from prepic.laser import GaussianBeam, Laser
from prepic.plasma import Plasma, r_e


@accepts(ωp=1 / time, τL=time)
def interaction_regime(ωp, τL):
    """Outputs the laser-plasma interaction regime.

    Parameters
    ----------
    ωp: float, 1/time
        Plasma frequency.
    τL: float, time
        Laser pulse duration at FWHM in intensity.
    """

    def magnitude(x):
        """Get order of magnitude of ``x``.
        >>> magnitude(100)
        2
        """
        return int(np.log10(x))

    ω_mag = magnitude((1 / ωp).to_value("femtosecond"))
    τ_mag = magnitude(τL.to_value("femtosecond"))

    if ω_mag == τ_mag:
        return "LWFA"
    elif τ_mag > ω_mag:
        return "SMLWFA/DLA"
    else:
        raise NotImplementedError("Unknown interaction regime.")


def matched_laser_plasma(a0, beam=GaussianBeam()):
    """Computes matched laser params and plasma density.

    From condition that dephasing length equals pump depletion length
    and condition for self-guided propagation
    we get optimal laser pulse duration `τL` and plasma density `n_pe`.
    From matching laser beam waist to plasma (α=1)
    we get the optimal beam waist `w0`.
    We also assume the bubble (blowout) radius to be R = w0 (β=1).

    Args:
        a0 (float, dimensionless): normalized laser vector potential
        beam (:obj:`GaussianBeam`, optional): instance providing laser wavelength

    Returns:
        :obj:`Plasma` instance with matched params

    Ref: Lu, W. et al., Phys. Rev. ST Accel. Beams 10 (6): 061301
    Note: these scaling laws are valid up to a critical value `a0c`.
    """
    τL = (2 / (3 * np.pi) * beam.λL / u.clight * a0 ** 3).to("femtosecond")
    n_pe = (np.pi / (r_e * beam.λL ** 2 * a0 ** 5)).to("1/cm**3")
    w0 = np.sqrt(a0 / (np.pi * r_e * n_pe))

    gbeam = GaussianBeam(w0=w0, λL=beam.λL)
    laser = Laser.from_a0(a0=a0, τL=τL, beam=gbeam)

    # critical normalized vector potential
    a0c = (2 * np.sqrt(laser.ncrit / n_pe)).to("dimensionless")
    if a0 > a0c:
        raise ValueError(f"Scaling laws valid up to a0c={a0c:.1f}")

    return Plasma(n_pe=n_pe, laser=laser, bubble_radius=w0)
