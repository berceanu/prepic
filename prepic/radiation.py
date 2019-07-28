"""
Classes for modelling emitted radiation from laser-plasma interaction

"""
from prepic._base_class import BaseClass
from prepic.plasma import r_e
import unyt as u
import numpy as np


class Radiator(BaseClass):
    """Class for estimating the properties of emitted radiation of a given laser-plasma.

    Attributes
    ----------
    α: :obj:`unyt_quantity`
        Fine structure constant.
    τ0: :obj:`unyt_quantity`
        Radiation-reaction time-scale.
    a: :obj:`unyt_quantity`
        Scaling constant used to calculate the betatron amplitude.
    b: :obj:`unyt_quantity`
        Scaling constant used to calculate the number of betatron periods.
    γ: :obj:`unyt_quantity`
        Lorentz factor of accelerated electrons.
    rβ: :obj:`unyt_quantity`
        Betatron oscillation amplitude.
    Nβ: :obj:`unyt_quantity`
        Number of betatron periods (oscillations).
    λu: :obj:`unyt_quantity`
        Spatial period of (local) electron betatron oscillations.
    K:  :obj:`unyt_quantity`
        Betatron oscillation amplitude (strength parameter).
        K ≫ 1 means we are in the wiggler regime.
    ħωc: :obj:`unyt_quantity`
        Critical photon energy of the synchrotron-like radiation spectrum.
    ħω_avg: :obj:`unyt_quantity`
        Mean photon energy of the synchrotron-like radiation spectrum.
    Nγ: :obj:`unyt_quantity`
        Number of photons emitted at ``ħω_avg`` per betatron period and per electron.
    θ_par: :obj:`unyt_quantity`
        Opening angle of the radiation in the plane of electron oscillations.
    θ_perp: :obj:`unyt_quantity`
        Opening angle of the radiation in the plane perpendicular to electron oscillations.
    N_shot: :obj:`unyt_quantity`
        Total number of radiated photons per laser shot.
    N_RR: :obj:`unyt_quantity`
        Threshold number of oscillations for noticeable radiation-reaction effects.

    Note
    ----
    ``a`` and ``b`` were obtained by fitting the example data from http://doi.org/f4j98s.
    The opening angle ``θ_par`` is defined as the maximum deflection angle of the electron trajectory,
    such that the full width of the angular distribution of the radiated energy in the electron
    oscillation plane is 2×``θ_par``.
    """

    α = (u.qe ** 2 / (4 * np.pi * u.eps_0 * u.hbar * u.clight)).to("dimensionless")
    τ0 = (2 * r_e / (3 * u.clight)).to("yoctosecond")
    a = 1510.3 * u.micrometer ** (-1 / 2)
    b = 3 * np.sqrt(2.0e19) * u.cm ** (-3 / 2)

    def __init__(self, plasma):
        """Default constructor.

        Parameters
        ----------
        plasma : :obj:`Plasma`
            Instance containing a :obj:`Laser`
        """
        if not plasma.laser:
            raise TypeError("Given `Plasma` instance must contain `Laser` instance.")
        else:
            self.plasma = plasma

        self.γ = (self.plasma.ΔE / (u.me * u.clight ** 2)).to("dimensionless")
        self.rβ = (self.a * np.sqrt(self.plasma.laser.a0 / self.plasma.npe)).to(
            "micrometer"
        )
        self.Nβ = (self.b / np.sqrt(self.plasma.npe)).to("dimensionless")
        self.λu = (np.sqrt(2 * self.γ) * self.plasma.λp).to("mm")
        self.K = (np.sqrt(self.γ / 2) * self.plasma.kp * self.rβ).to("dimensionless")
        self.θ_perp = (1 / self.γ * u.radian).to("miliradian")
        self.N_RR = (
            self.λu / (2 * np.pi ** 2 * u.clight * self.τ0 * self.γ * self.K ** 2)
        ).to("dimensionless")

        if self.K > 5:  # wiggler regime
            self.ħωc = (3 / 2 * self.K * self.γ ** 2 * u.h * u.clight / self.λu).to(
                "kiloelectronvolt"
            )
            self.ħω_avg = (8 / (15 * np.sqrt(3)) * self.ħωc).to("kiloelectronvolt")
            self.Nγ = 5 * np.sqrt(3) * np.pi * self.α * self.K / 6
            self.θ_par = (self.K / self.γ * u.radian).to("miliradian")
            self.N_shot = (self.Nγ * self.Nβ * self.plasma.N).to("dimensionless")
        else:
            # todo implement undulator case
            raise NotImplementedError("The undulator case is not yet implemented.")

    def __str__(self):
        msg = (
            f"Betatron radiation is emitted at a mean energy of <ħω> = {self.ħω_avg:.1f}, while the critical "
            f"energy is ħωc = {self.ħωc:.1f}. The number of photons emitted at <ħω> per betatron period "
            f"and per electron is Nᵧ = {self.Nγ.to_value('dimensionless'):.1f}, while the total number of "
            f"betatron photons per laser shot is {self.N_shot.to_value('dimensionless'):.1e}. "
            f"The half width of the angular distribution of the radiated energy in the "
            f"electron oscillation plane is θᵣ = {self.θ_par:.1f}. "
        )

        if self.N_RR > 10 * self.Nβ:
            msg += f"Radiation-reaction effects are negligible."
        else:
            # todo implement radiation-reaction effects
            raise NotImplementedError(
                "Radiation-reaction effects are not yet implemented."
            )

        return msg

    # todo add synchrotron spectrum

    def __repr__(self):
        return f"<{self.__class__.__name__}({repr(self.plasma)})>"
