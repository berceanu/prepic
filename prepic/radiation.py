"""
Classes for modelling emitted radiation from laser-plasma interaction

"""
import warnings

import numpy as np
import unyt as u
from scipy.integrate import quad
from scipy.special import kv
from unyt import accepts, returns
from unyt.dimensions import dimensionless, energy, time

from prepic._base_class import BaseClass
from prepic._constants import r_e


@returns(energy)
@accepts(ωc=1 / time, γ=dimensionless)
def _total_radiated_energy(ωc, γ):
    r"""Computes total energy radiated per betatron oscillation.

    .. math:: I = \frac{2}{9} \omega_c \gamma \frac{e^2}{\epsilon_0 c}

    Parameters
    ----------
    γ : float, dimensionless
        Electron Lorentz factor.
    ωc : float, 1/time
        Critical synchrotron frequency.

    Returns
    -------
    intensity : float, energy
        Total radiated energy per electron, per betatron oscillation.

    References
    ----------
    See [1]_, Section 14.6.

    .. [1] Jackson, J. D. (1999). Classical electrodynamics.

    Examples
    --------
    >>> I = _total_radiated_energy(ωc=3e5 / u.fs, γ=5e3 * u.dimensionless)
    >>> print("{:.1f}".format(I))
    20119.6 keV
    """
    intensity = 2 / 9 * u.qe ** 2 / (u.eps_0 * u.clight) * ωc * γ
    return intensity.to("keV")


@returns(dimensionless)
@accepts(y=dimensionless)
def _s_function(y, max_abserr=1e-5):
    r"""Shape of the synchrotron spectrum.

    Integral that appears in the photon frequency distribution of a
    synchrotron spectrum.

    .. math:: S(y) = \int_{y}^{\infty} dx\, K_{5/3}(x)

    Parameters
    ----------
    y : float, dimensionless
        Ratio of frequency to critical frequency, :math:`\omega/\omega_c`.
    max_abserr : float
        Maximum threshold for the absolute integration error.

    Returns
    -------
    float, dimensionless
        The result of numerical integration.

    Raises
    ------
    FloatingPointError
        If the absolute integration error is too large.
    IntegrationWarning
        If the integral is divergent, or slowly convergent.

    References
    ----------
    See [1]_, Section 14.6.

    .. [1] Jackson, J. D. (1999). Classical electrodynamics.

    Examples
    --------
    >>> import unyt as u
    >>> res = _s_function(0.5 * u.dimensionless)
    >>> print(res)
    1.7416382937508474 dimensionless
    >>> _s_function(0.0 * u.dimensionless)
    Traceback (most recent call last):
    ...
    scipy.integrate.quadpack.IntegrationWarning: The integral is probably divergent, or slowly convergent.
    >>> _s_function(1e-5 * u.dimensionless, max_abserr=1e-6)
    Traceback (most recent call last):
    ...
    FloatingPointError: S integration error too large, 9.50743117487246e-06 at y=1e-05 dimensionless
    """
    warnings.filterwarnings("error")

    result, abserr = quad(lambda x: kv(5 / 3, x), y, np.inf)

    if abserr > max_abserr:
        raise FloatingPointError(
            "S integration error too large, %s at y=%s" % (abserr, y)
        )
    else:
        return result * u.dimensionless


@returns(dimensionless)
@accepts(ω=1 / time, ωc=1 / time, γ=dimensionless)
def photon_frequency_distribution(ω, ωc, γ):
    r"""Computes the number of photons per unit frequency interval.

    Computes the number of photons per unit frequency interval, at frequency :math:`\omega`,
    per betatron oscillation and per electron, integrated over all angles.

    .. math::

        \frac{dN}{dy} = \frac{9 \sqrt{3}}{8 \pi} \frac{I}{\hbar \omega_c} y S(y)

        I = \frac{2}{9} \omega_c \gamma \frac{e^2}{\epsilon_0 c}

        S(y) = \int_{y=\omega/\omega_c}^{\infty} dx\, K_{5/3}(x)

    Parameters
    ----------
    ω : float, 1/time
        Observation frequency.
    ωc : float, 1/time
        Critical synchrotron frequency.
    γ : float, dimensionless
        Electron Lorentz factor.

    Returns
    -------
    dN_over_dy : float, dimensionless
        Number of photons per unit frequency interval :math:`y=\omega/\omega_c`.

    References
    ----------
    See [1]_, Section 14.6.

    .. [1] Jackson, J. D. (1999). Classical electrodynamics.

    Examples
    --------
    >>> Ny = photon_frequency_distribution(ω=9e4 / u.fs, ωc=3e5 / u.fs, γ=5e3 * u.dimensionless)
    >>> print("{:.1f}".format(Ny))
    58.0 dimensionless

    """
    a = 9 * np.sqrt(3) / (8 * np.pi)  # prefactor
    y = (ω / ωc).to("dimensionless")
    dN_over_dy = a * _total_radiated_energy(ωc, γ) / (u.hbar * ωc) * y * _s_function(y)
    return dN_over_dy.to("dimensionless")


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

    Notes
    -----
    ``a`` and ``b`` were obtained by fitting the example data from http://doi.org/f4j98s.
    The opening angle ``θ_par`` is defined as the maximum deflection angle of the electron trajectory,
    such that the full width of the angular distribution of the radiated energy in the electron
    oscillation plane is 2×``θ_par``.

    Examples
    --------
    >>> import unyt as u
    >>> from prepic import Plasma, Laser, GaussianBeam
    >>> waist = 15 * u.micrometer
    >>> mylaser = Laser.from_power(power=1 * u.petawatt, ɛL=3 * u.joule,
    ...                            beam=GaussianBeam(w0=waist))
    >>> myplasma = Plasma(n_pe=1e18 / u.cm**3, laser=mylaser, bubble_radius=waist)
    >>> myradiator = Radiator(myplasma)
    >>> myradiator
    <Radiator(<Plasma(1e+18 cm**(-3), <Laser(3.0 J, 2.818311836098954 fs, <GaussianBeam(15.0 µm, 0.8 µm)>)>, 15.0 µm)>)>
    >>> print(myradiator)
    Betatron radiation is emitted at a mean energy of <ħω> = 1475.6 keV, while the critical
    energy is ħωc = 4792.0 keV. The number of photons emitted at <ħω> per betatron period
    and per electron is Nᵧ = 2.6, while the total number of
    betatron photons per laser shot is 4.9e+10.
    The half width of the angular distribution of the radiated energy in the
    electron oscillation plane is θᵣ = 5.9 mrad.
    Radiation-reaction effects are negligible.
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
            f"Betatron radiation is emitted at a mean energy of <ħω> = {self.ħω_avg:.1f}, while the critical\n"
            f"energy is ħωc = {self.ħωc:.1f}. The number of photons emitted at <ħω> per betatron period\n"
            f"and per electron is Nᵧ = {self.Nγ.to_value('dimensionless'):.1f}, while the total number of\n"
            f"betatron photons per laser shot is {self.N_shot.to_value('dimensionless'):.1e}.\n"
            f"The half width of the angular distribution of the radiated energy in the\n"
            f"electron oscillation plane is θᵣ = {self.θ_par:.1f}.\n"
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
