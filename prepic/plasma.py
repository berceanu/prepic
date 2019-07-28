"""
Classes for modelling the plasma, without any matching

"""
from prepic._base_class import BaseClass
from prepic.lwfa import interaction_regime
from unyt import accepts
from unyt.dimensions import flux
import unyt as u
import numpy as np

# classical electron radius
r_e = (1 / (4 * np.pi * u.eps_0) * u.qe ** 2 / (u.me * u.clight ** 2)).to("micrometer")


@accepts(i0=flux)
def helium_ionization_state(i0):
    """Compute the ionization state of Helium.

    Parameters
    ----------
    i0: float, energy/time/area
        Peak laser intensity in the focal plane.
    """
    if i0 < 1.4e15 * u.watt / u.cm ** 2:
        return "0+"
    elif i0 < 8.8e15 * u.watt / u.cm ** 2:
        return "1+"
    else:
        return "2+"


class Plasma(BaseClass):
    """Class containing plasma parameters.
    Attributes:
        npe (float, 1/volume): plasma electron (number) density
        ωp (float, 1/time): plasma frequency
        lp (float, length): unit of length
        tp (float, time): unit of time
        λp (float, length): plasma skin depth
        kp (float, 1/length): plasma wavenumber
        Ewb (float, energy/charge/length): cold, 1D wave-breaking field
        laser (:obj:`Laser`): instance containing laser params
        γp (float, dimensionless): plasma γ factor
        Pc (float, energy/time): critical power for self-focusing
        dephasing (float, length): electron dephasing length
        depletion (float, length): pump depletion length
        Ez_avg (float, energy/charge/length): average accelerating field \
                                    in the direction of electron propagation
        R (float, length): radius of the plasma bubble
        Lacc (float, length): distance over which laser propagates
        N (float, dimensionless): estimated number of electrons in the bunch
        Q (float, charge): estimated total electron bunch charge
        ΔE (float, energy): maximum energy gained by one electron \
                        propagating for Lacc \
                        see Lu et al., 2007 Phys. Rev. ST. Accel. Beams
        η (float, dimensionless): energy transfer efficiency, defined as \
                        total bunch energy `N` * `ΔE` / laser energy `ɛL` \
            under matching conditions, `η` ~ 1 / (2 * a0)
    """

    def __init__(self, n_pe, laser=None, bubble_radius=None, propagation_distance=None):
        """Creates plasma with given density.
        Args:
            n_pe (float, 1/volume): plasma electron (number) density
            laser (:obj:`Laser`, optional): instance containing laser params
            bubble_radius (float, length, optional): radius of the plasma bubble
            propagation_distance (float, length, optional): length of plasma region
                                                    defaults to `dephasing`
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
            regime = interaction_regime(ωp=self.ωp, τL=self.laser.τL)
            assert regime == "LWFA", regime
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