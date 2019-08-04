"""
Classes for modelling the plasma, without any matching

"""
from prepic.base import BaseClass
from prepic.constants import r_e

import numpy as np
import unyt as u

dim = u.dimensions


@u.accepts(ωp=1 / dim.time, τL=dim.time)
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


class Plasma(BaseClass):
    """Class containing plasma parameters.

    Attributes
    ----------
    npe : float, 1/volume
        Plasma electron (number) density.
    ωp : float, 1/time
        Plasma frequency.
    lp : float, length
        Unit of length.
    tp : float, time
        Unit of time.
    λp : float, length
        Plasma skin depth.
    kp : float, 1/length
        Plasma wavenumber.
    Ewb : float, energy/charge/length
        Cold, 1D wave-breaking field.
    laser : :obj:`Laser`
        Instance containing laser params.
    γp : float, dimensionless
        Plasma γ factor.
    Pc : float, energy/time
        Critical power for self-focusing.
    dephasing : float, length
        Electron dephasing length.
    depletion : float, length
        Pump depletion length.
    Ez_avg : float, energy/charge/length
        Average accelerating field \
        in the direction of electron propagation.
    R : float, length
        Radius of the plasma bubble.
    Lacc : float, length
        Distance over which laser propagates.
    N : float, dimensionless
        Estimated number of electrons in the bunch.
    Q : float, charge
        Estimated total electron bunch charge.
    ΔE : float, energy
        Maximum energy gained by one electron \
        propagating for Lacc \
        see Lu et al., 2007 Phys. Rev. ST. Accel. Beams.
    η : float, dimensionless
        Energy transfer efficiency, defined as \
        total bunch energy `N` * `ΔE` / laser energy `ɛL` \
        under matching conditions, `η` ~ 1 / (2 * a0).

    Examples
    --------
    >>> import unyt as u
    >>> Plasma(n_pe=1e18 / u.cm**3)
    <Plasma(1e+18 cm**(-3), None, None)>
    """

    def __init__(self, n_pe, laser=None, bubble_radius=None, propagation_distance=None):
        """Creates plasma with given density.

        Parameters
        ----------
        n_pe : float, 1/volume
            Plasma electron (number) density.
        laser : :obj:`Laser`, optional
            Instance containing laser params.
        bubble_radius : float, length, optional
            Radius of the plasma bubble.
        propagation_distance : float, length, optional
            Length of plasma region (defaults to `dephasing`).
        """
        self.npe = n_pe.to("1/cm**3")
        self.λp = np.sqrt(np.pi / (r_e * self.npe)).to("micrometer")
        self.kp = (2 * np.pi / self.λp).to("1/micrometer")
        self.ωp = (u.clight * self.kp).to("1/femtosecond")

        self.Ewb = (u.me * u.clight * self.ωp / np.abs(u.qe)).to("megavolt/mm")

        self.lp = (u.clight / self.ωp).to("micrometer")
        self.tp = (1 / self.ωp).to("femtosecond")

        if laser:
            self.laser = laser

            self.γp = (self.laser.ωL / self.ωp).to("dimensionless")
            self.Pc = (17 * self.γp ** 2 * u.gigawatt).to("terawatt")

            self.dephasing = (
                4 / 3 * self.γp ** 2 * np.sqrt(self.laser.a0) / self.kp
            ).to("mm")
            self.depletion = (self.γp ** 2 * u.clight * self.laser.τL).to("mm")

            self.Ez_avg = (self.Ewb * np.sqrt(self.laser.a0) / 2).to("megavolt/mm")

            if propagation_distance:
                self.Lacc = propagation_distance.to("mm")
            else:
                self.Lacc = self.dephasing

            self.ΔE = (np.abs(u.qe) * self.Ez_avg * self.Lacc).to("megaelectronvolt")

            if bubble_radius:
                self.R = bubble_radius.to("micrometer")

                self.N = (1 / 30 * (self.kp * self.R) ** 3 / (self.kp * r_e)).to(
                    "dimensionless"
                )
                self.Q = (self.N * np.abs(u.qe)).to("picocoulomb")

                self.η = (self.N * self.ΔE / self.laser.ɛL).to("dimensionless")
            else:
                self.R = None
        else:
            self.laser = None
            self.R = None

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.npe}, {repr(self.laser)}, {self.R})>"

    def __str__(self):
        msg = f"Plasma with nₚ={self.npe:.1e}, ωₚ={self.ωp:.3f}, kₚ={self.kp:.3f}, λₚ={self.λp:.1f}, Ewb={self.Ewb:.1f}"
        if self.laser:
            n_ratio = (self.npe / self.laser.ncrit).to("dimensionless")
            msg = (
                f"Plasma with nₚ={self.npe:.1e} ({n_ratio.to_value('dimensionless'):.2e} × nc), ωₚ={self.ωp:.3f}, "
                f"kₚ={self.kp:.3f}, λₚ={self.λp:.1f}, Ewb={self.Ewb:.1f}"
            )
            _ = interaction_regime(ωp=self.ωp, τL=self.laser.τL)
            # fixme
            # assert regime == "LWFA", regime
            msg += (
                f"\nPc={self.Pc:.1f}, Ldeph={self.dephasing:.2f}, Ldepl={self.depletion:.2f}, "
                f"ΔE={self.ΔE:.1f} over Lacc={self.Lacc:.2f}"
            )
            if self.R:
                msg += (
                    f"\nN={self.N.to_value('dimensionless'):.1e} electrons, Q={self.Q:.1f}, "
                    f"η={self.η.to_value('dimensionless'):.3f}"
                )
        return msg
