# -*- coding: utf-8 -*-
import numpy as np
import unyt as u
from numpy import pi as π
from unyt import check_dimensions
from unyt.dimensions import dimensionless, length, flux

from ._testing import BaseClass, has_units

"""Laser WakeField Acceleration module."""

# classical electron radius
r_e = (1 / (4 * π * u.eps_0) * u.qe ** 2 / (u.me * u.clight ** 2)).to("micrometer")


# Utility functions


@check_dimensions(w0=length)
def w0_to_fwhm(w0):
    """Computes Gaussian laser FWHM from its beam waist.

    Args:
        w0 (float, length): beam waist @ 1/e^2 intensity

    Returns:
        fwhm (float, length): beam FWHM @ 1/2 intensity
    """
    return 2 * w0 / np.sqrt(2 / np.log(2))


@check_dimensions(fwhm=length)
def fwhm_to_w0(fwhm):
    """Computes Gaussian laser beam waist from its FWHM.

    Args:
        fwhm (float, length): beam FWHM @ 1/2 intensity

    Returns:
        w0 (float, length): beam waist @ 1/e^2 intensity
    """
    return 1 / 2 * np.sqrt(2 / np.log(2)) * fwhm


@check_dimensions(a0=dimensionless, λL=length)
def intensity_from_a0(a0, λL=0.8 * u.micrometer):
    """Compute peak laser intensity in the focal plane.

    Args:
        a0 (float, dimensionless): normalized laser vector potential
        λL (float, length): laser wavelength

    Returns:
        I0 (float, energy/time/area): peak laser intensity in the focal plane
    """
    return π / 2 * u.clight / r_e * u.me * u.clight ** 2 / λL ** 2 * a0 ** 2


@check_dimensions(i0=flux, λL=length)
def a0_from_intensity(i0, λL=0.8 * u.micrometer):
    """Compute laser normalized vector potential.

    Args:
        i0 (float, energy/time/area): peak laser intensity in the focal plane
        λL (float, length): laser wavelength

    Returns:
        a0 (float, dimensionless): normalized laser vector potential
    """
    return np.sqrt(i0 / (π / 2 * u.clight / r_e * u.me * u.clight ** 2 / λL ** 2))


@check_dimensions(i0=flux)
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


# @check_dimensions(ωp="1/time", τL="time")  # todo https://github.com/yt-project/unyt/issues/91
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


class GaussianBeam(BaseClass):
    """Contains the (geometric) parameters for a Gaussian laser beam.

    Attributes:
        w0 (float, length): beam waist @ 1/e^2 intensity
        fwhm (float, length): beam FWHM @ 1/2 intensity
        λL (float, length): wavelength
        zR (float, length): Rayleigh length
    """

    def __init__(self, w0=None, fwhm=None, λL=0.8 * u.micrometer):
        """Default beam constructor.
        Can take *either* `w0` or `fwhm` as parameters.
        """
        self.λL = λL.to("micrometer")

        if w0 and not fwhm:
            self.w0 = w0.to("micrometer")
            self.fwhm = w0_to_fwhm(self.w0).to("micrometer")
        elif fwhm and not w0:
            self.fwhm = fwhm.to("micrometer")
            self.w0 = fwhm_to_w0(self.fwhm).to("micrometer")
        elif w0 and fwhm:
            raise ValueError("both w0 and fwhm given, only give one")
        else:  # both None
            self.w0 = None
            self.fwhm = None

        if self.w0:
            self.zR = (π * self.w0 ** 2 / self.λL).to("milimeter")
        else:
            self.zR = None

    @classmethod
    def from_f_number(cls, f_number, λL=0.8 * u.micrometer):
        """Construct beam by giving the OAP's f/#.

        Args:
            :param f_number: f/# of the off-axis parabolic mirror (float, dimensionless)
            :param λL: laser wavelength (float, length, optional)
        """
        assert has_units(f_number, dimensionless), "f_number should be dimensionless"

        w0 = 2 * np.sqrt(2) / π * λL * f_number
        return cls(w0=w0, λL=λL)

    @classmethod
    def from_focal_distance(cls, focal_distance, beam_diameter, λL=0.8 * u.micrometer):
        """Constuct beam from OAP's focal distance and beam diameter.

        Args:
            :param focal_distance: focal distance of the off-axis parabolic mirror (float, units of length)
            :param beam_diameter: beam diameter after compressor (float, units of length)
            :param λL: laser wavelength (float, length, optional)
        """
        assert has_units(focal_distance, length), "focal_distance should be a length"
        assert has_units(beam_diameter, length), "beam_diameter should be a length"

        return cls.from_f_number(f_number=focal_distance / beam_diameter, λL=λL)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.w0}, {self.λL})>"

    def __str__(self):
        msg = f"beam with λL={self.λL:.2f}"
        if self.w0 and self.fwhm:
            msg = f"beam with w0={self.w0:.1f} (FWHM={self.fwhm:.1f}), zᵣ={self.zR:.2f}, λL={self.λL:.2f}"
        return msg


# Laser, without matching


class Laser(BaseClass):
    """Class containing laser parameters.

    Attributes:
        beam (:obj:`GaussianBeam`): class instance containing beam params
        ɛL (float, energy): pulse energy on target (after compressor
                            and beam transport, focused into the FWHM@intensity spot)
        τL (float, time): pulse duration at FWHM in intensity
        kL (float, 1/length): wavenumber
        ωL (float, 1/time): angular frequency
        ncrit (float, 1/volume): critical plasma density for this laser
        P0 (float, energy/time): power
        I0 (float, energy/time/area): peak intensity in the focal plane
        a0 (float, dimensionless): normalized vector potential
        E0 (float, energy/charge/length): peak electric field
    """

    def __init__(self, ɛL, τL, beam=GaussianBeam()):
        """Creates laser with given parameters."""
        self.beam = beam
        self.kL = (2 * π / self.beam.λL).to("1/micrometer")
        self.ωL = (u.clight * self.kL).to("1/femtosecond")
        self.ncrit = (π / (r_e * self.beam.λL ** 2)).to("1/cm**3")

        self.ɛL = ɛL.to("joule")
        self.τL = τL.to("femtosecond")
        self.P0 = (2 * np.sqrt(np.log(2) / π) * self.ɛL / self.τL).to("terawatt")

        if self.beam.w0:
            self.I0 = (
                2
                / π
                * np.sqrt(4 * np.log(2) / π)
                * self.ɛL
                / (self.τL * self.beam.w0 ** 2)
            ).to("watt/cm**2")
            self.a0 = a0_from_intensity(i0=self.I0, λL=self.beam.λL).to("dimensionless")
            self.E0 = (u.clight * u.me * self.ωL / np.abs(u.qe) * self.a0).to(
                "megavolt/mm"
            )
        else:
            self.I0 = None
            self.a0 = None
            self.E0 = None

    @classmethod
    def from_a0(cls, a0, ɛL=None, τL=None, beam=GaussianBeam()):
        """Construct laser by giving its normalized vector potential a0.
        Must supply either (ɛL,τL), (ɛL,beam) or (τL,beam).
        """
        prefactor = (4 * np.log(2) / π) ** (3 / 2)
        i0 = intensity_from_a0(a0=a0, λL=beam.λL).to("watt/cm**2")

        if ɛL and τL and (not beam.fwhm):
            fwhm = np.sqrt((prefactor * ɛL) / (i0 * τL))
        elif ɛL and beam.fwhm and (not τL):
            fwhm = beam.fwhm
            τL = (prefactor * ɛL) / (i0 * fwhm ** 2)
        elif τL and beam.fwhm and (not ɛL):
            fwhm = beam.fwhm
            ɛL = i0 * τL * fwhm ** 2 / prefactor
        else:
            raise TypeError("Must supply either (ɛL,τL), (ɛL,beam) or (τL,beam).")

        return cls(ɛL=ɛL, τL=τL, beam=GaussianBeam(fwhm=fwhm))

    @classmethod
    def from_intensity(cls, intensity, ɛL=None, τL=None, beam=GaussianBeam()):
        """Construct laser by giving its intensity I0.
        Must supply either (ɛL,τL), (ɛL,beam) or (τL,beam).
        """
        a0 = a0_from_intensity(i0=intensity, λL=beam.λL)
        return cls.from_a0(a0=a0, ɛL=ɛL, τL=τL, beam=beam)

    @classmethod
    def from_power(cls, power, beam, ɛL=None, τL=None):
        """Construct laser by giving its power P0 and beam size.
        Must supply either ɛL or τL.
        """
        prefactor = 2 * np.sqrt(np.log(2) / π)

        if ɛL and (not τL):
            τL = prefactor * ɛL / power
        elif τL and (not ɛL):
            ɛL = power * τL / prefactor
        else:  # either both or none
            raise TypeError("Must supply either ɛL or τL.")

        return cls(ɛL=ɛL, τL=τL, beam=beam)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.ɛL}, {self.τL}, {repr(self.beam)})>"

    def __str__(self):
        msg = (
            f"laser with kL={self.kL:.3f}, ωL={self.ωL:.3f}, ɛL={self.ɛL:.1f}, "
            f"τL={self.τL:.1f}, P₀={self.P0:.1f}"
        )

        if (self.I0 is not None) and (self.E0 is not None) and (self.a0 is not None):
            msg += f"\nI₀={self.I0:.1e}, a₀={self.a0.to_value('dimensionless'):.1f}, E₀={self.E0:.1e}"
            msg += f"\nHelium ionization state: {helium_ionization_state(i0=self.I0)}"

        return msg


# Plasma, without matching


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
        self.λp = np.sqrt(π / (r_e * self.npe)).to("micrometer")
        self.kp = (2 * π / self.λp).to("1/micrometer")
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


# Matching conditions


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
    τL = (2 / (3 * π) * beam.λL / u.clight * a0 ** 3).to("femtosecond")
    n_pe = (π / (r_e * beam.λL ** 2 * a0 ** 5)).to("1/cm**3")
    w0 = np.sqrt(a0 / (π * r_e * n_pe))

    gbeam = GaussianBeam(w0=w0, λL=beam.λL)
    laser = Laser.from_a0(a0=a0, τL=τL, beam=gbeam)

    # critical normalized vector potential
    a0c = (2 * np.sqrt(laser.ncrit / n_pe)).to("dimensionless")
    if a0 > a0c:
        raise ValueError(f"Scaling laws valid up to a0c={a0c:.1f}")

    return Plasma(n_pe=n_pe, laser=laser, bubble_radius=w0)


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


class Simulation(BaseClass):
    """Class for estimating the recommended simulation parameters.
    Attributes:
        Δx (float, length): transverse spatial resolution
        Δy (float, length): transverse spatial resolution
        Δz (float, length): longitudinal spatial resolution
        nx (int, dimensionless): transverse number of cells
        ny (int, dimensionless): transverse number of cells
        nz (int, dimensionless): longitudinal number of cells
        L (float, length): length of cubic simulation box
        ppc (int, dimensionless): number of particles per cell
        npart (int, dimensionless): total number of (macro-)particles in the
            simulation box
        dt (float, time): simulation time step per iteration
        t_interact (float, time): time it takes for the moving window to slide
            across the plasma
        nstep (int, dimensionless): number of iterations to perform
    Note:
        Here longitudinal means along the laser propagation direction.
        Recommended number of particles per cell: 64 (1D), 10 (2D), 8 (3D).
    """

    def __init__(self, plasma, box_length=None, ppc=None):
        """Estimate recommended simulation params for given plasma (and laser).
        Args:
            plasma (:obj:`Plasma`): instance containing laser and plasma params
            box_length (float, length, optional): length of the cubic
                simulation box. Defaults to 4λₚ.
            ppc (int, dimensionless, optional): number of particles per cell.
                Defaults to 8 (3D).
        """
        if not plasma.laser:
            raise TypeError("Given `Plasma` instance must contain `Laser` instance.")
        else:
            self.plasma = plasma
        if not box_length:
            self.L = 4 * self.plasma.λp
        else:
            self.L = box_length.to("micrometer")
        if not ppc:
            self.ppc = u.unyt_quantity(8, "dimensionless", dtype="int")
        else:
            self.ppc = ppc

        self.Δx = self.plasma.lp / 10
        self.Δy = self.Δx
        self.Δz = self.plasma.laser.beam.λL / 20

        self.nx = u.unyt_quantity(
            (self.L / self.Δx).to_value("dimensionless"), "dimensionless", dtype="int"
        )
        self.ny = self.nx
        self.nz = u.unyt_quantity(
            (self.L / self.Δz).to_value("dimensionless"), "dimensionless", dtype="int"
        )

        self.npart = self.nx * self.ny * self.nz * self.ppc

        self.dt = (self.Δz / u.clight).to("femtosecond")
        self.t_interact = ((self.plasma.Lacc + self.L) / u.clight).to("femtosecond")

        self.nstep = u.unyt_quantity(
            (self.t_interact / self.dt).to_value("dimensionless"),
            "dimensionless",
            dtype="int",
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}({repr(self.plasma)}, {self.L}, {self.ppc})>"

    def __str__(self):
        msg = (
            f"3D simulation with box size ({self.L:.1f})³, Δx={self.Δx:.3f}, Δy={self.Δy:.3f}, "
            f"Δz={self.Δz:.3f}, nx={int(self.nx.to_value('dimensionless'))}, "
            f"ny={int(self.ny.to_value('dimensionless'))}, nz={int(self.nz.to_value('dimensionless'))}, "
            f"{int(self.npart.to_value('dimensionless')):e} macro-particles, "
            f"{int(self.nstep.to_value('dimensionless')):e} time steps"
        )
        return msg
