"""
Classes for modelling emitted radiation from laser-plasma interaction

"""
import warnings
from functools import partial

import numpy as np
from scipy.integrate import quad
from scipy.special import kv

import unyt as u

from prepic.base import BaseClass
from prepic.constants import r_e, α

from sliceplots import Plot2D

dim = u.dimensions


@u.returns(dim.dimensionless)
@u.accepts(ħω=dim.energy, θ=dim.angle, ħωc=dim.energy, γ=dim.dimensionless)
def differential_intensity_distribution(ħω, θ, ħωc, γ):
    r"""Computes the synchrotron energy distribution in frequency and solid angle.

    Doubly differential intensity distribution :math:`\frac{d^2I}{d \hbar \omega d \Omega}`, representing the radiated
    energy per unit energy interval :math:`d \hbar \omega` per unit solid angle :math:`d \Omega`.

    .. math:: \frac{d^2I}{d \hbar \omega d \Omega} = \frac{3 \alpha}{4 \pi^2} \left(\frac{\hbar \omega}{\hbar \omega_c}\right)^2 \gamma^6 \left(\frac{1}{\gamma^2} + \theta^2\right)^2 \left[K^2_{2/3}(\xi) + \frac{\theta^2}{1/\gamma^2 + \theta^2} K^2_{1/3}(\xi)\right]

    where

    .. math:: \xi \simeq \frac{\hbar \omega}{2 \hbar \omega_c} \left(1 + \gamma^2 \theta^2 \right)^{3/2}

    Parameters
    ----------
    ħω : float, energy
        Observation energy.
    θ : float, angle
        Observation angle relative to the particle's orbital plane (latitude).
    ħωc : float, energy
        Critical synchrotron energy.
    γ : float, dimensionless
        Electron Lorentz factor.

    Returns
    -------
    d2I : float, dimensionless
        Doubly differential cross section.

    References
    ----------
    Eq. (24) of [DE]_.

    .. [DE] Don Edwards, `"Notes on Synchrotron Radiation" <https://www.researchgate.net/profile/Thierry_De_Mees3/post/Is_there_something_similar_to_synchrotron_radiation_in_gravitoelectromagnetism_GEM/attachment59d62c2e79197b807798a8ee/AS%3A346000725168128%401459504409173/download/syncradnotes.pdf>`_.

    Examples
    --------
    >>> d2I = differential_intensity_distribution(ħω=59.24 * u.kiloelectronvolt, θ=0.1 * u.miliradian, ħωc=197.0 * u.kiloelectronvolt, γ=4956.5 * u.dimensionless)
    >>> print("{:.1f}".format(d2I))
    15364.7 dimensionless
    """  # noqa E501
    γ = γ.to_value(u.dimensionless)
    θ = θ.to_value(u.radian)
    ξ = (ħω / (2 * ħωc) * (1 + γ ** 2 * θ ** 2) ** (3 / 2)).to_value(u.dimensionless)
    d2I = (
        (3 * α)
        / (4 * np.pi ** 2)
        * (ħω / ħωc) ** 2
        * γ ** 6
        * (1 / γ ** 2 + θ ** 2) ** 2
        * (kv(2 / 3, ξ) ** 2 + θ ** 2 / (1 / γ ** 2 + θ ** 2) * kv(1 / 3, ξ) ** 2)
    )
    return d2I.to(u.dimensionless)


@u.returns(dim.energy)
@u.accepts(ωc=1 / dim.time, γ=dim.dimensionless)
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

    Examples
    --------
    >>> I = _total_radiated_energy(ωc=3e5 / u.fs, γ=5e3 * u.dimensionless)
    >>> print("{:.1f}".format(I))
    20119.6 keV
    """
    intensity = 2 / 9 * u.qe ** 2 / (u.eps_0 * u.clight) * ωc * γ
    return intensity.to("keV")


@u.returns(dim.dimensionless)
@u.accepts(y=dim.dimensionless)
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


@u.returns(dim.dimensionless)
@u.accepts(ω=1 / dim.time, ωc=1 / dim.time, γ=dim.dimensionless)
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


@u.returns(dim.dimensionless)
@u.accepts(θ=dim.angle, γ=dim.dimensionless)
def photon_angle_distribution(θ, γ):
    r"""Computes the number of photons per unit solid angle.

    Computes the number of photons observed at an angle
    :math:`\theta` from the electron's plane of oscillation,
    per unit solid angle, per betatron oscillation and per electron,
    integrated over all frequencies.

    .. math::

        \frac{dN}{d\Omega} = \frac{7 e^2}{96 \pi \epsilon_0 \hbar c}
        \frac{\gamma^2}{(1 + \gamma^2 \theta^2)^{5 / 2}}
        \left(1 + \frac{5}{7} \frac{\gamma^2 \theta^2}{1 + \gamma^2 \theta^2}\right)

    Parameters
    ----------
    γ : float, dimensionless
        Electron Lorentz factor.
    θ : float, angle
        Observation angle relative to the particle's orbital plane (latitude).

    Returns
    -------
    dN_over_dΩ : float, dimensionless
        Number of photons :math:`N` per solid angle :math:`\Omega`.

    References
    ----------
    See [DZDS]_, Eq. (18), where we divided by :math:`\hbar \omega_c`

    .. [DZDS] Downer, M. C., et al. Reviews of Modern Physics 90.3 (2018): 035002.

    Examples
    --------
    >>> NΩ = photon_angle_distribution(θ=0.5 * u.degree, γ=5e3 * u.dimensionless)
    >>> print("{:.1e}".format(NΩ))
    5.8e-04 dimensionless
    """
    θ = θ.to_value(u.radian)
    a = 7 * u.qe ** 2 / (96 * np.pi * u.eps_0 * u.hbar * u.clight)  # prefactor
    dN_over_dΩ = (
        a.to(u.dimensionless)
        * γ ** 2
        / ((1 + γ ** 2 * θ ** 2) ** (5 / 2))
        * (1 + 5 / 7 * γ ** 2 * θ ** 2 / (1 + γ ** 2 * θ ** 2))
    )
    return dN_over_dΩ.to(u.dimensionless)


class AnnotationText:
    """
    Examples
    --------
    >>> ann = AnnotationText(text="blarg", xy=(0.6, 0.9), xycoords='axes fraction')
    >>> ann.text
    'blarg'
    """

    def __init__(self, *, text, xy, xycoords="data"):
        self.text = text
        self.xy = xy
        self.xycoords = xycoords


# todo add docstrings
class DifferentialSpectrum:
    def __init__(self, radiator, npoints=50, ħω=None, θ=None):
        self.rad = radiator

        self.dist_func = partial(
            differential_intensity_distribution, ħωc=self.rad.ħωc, γ=self.rad.γ
        )  # = f(ħω, θ)

        if ħω is None:
            self.ħω = np.linspace(1e-5 * self.rad.ħωc, 2 * self.rad.ħωc, npoints)
        else:
            self.ħω = ħω

        if θ is None:
            self.θc = 1e3 / self.rad.γ.to_value(u.dimensionless)
            self.θ = np.linspace(0, 2 * self.θc, npoints) * u.miliradian
        else:
            self.θ = θ

        # pre-allocate
        self._doubly_differential_data = np.array([])
        self._angle_integrated_data = np.array([])
        self._energy_integrated_data = np.array([])

    @property
    def doubly_differential_data(self):
        if self._doubly_differential_data.size == 0:  # empty array?
            # vectorized computation
            ħωħω, θθ = np.meshgrid(self.ħω, self.θ)
            self._doubly_differential_data = self.dist_func(ħωħω, θθ).to_value(
                u.dimensionless
            )
        return self._doubly_differential_data

    @property
    def angle_integrated_data(self):
        if self._angle_integrated_data.size == 0:
            self._angle_integrated_data = np.trapz(
                self.doubly_differential_data, axis=0
            )  # f(ħω) // integrate over θ
        return self._angle_integrated_data

    @property
    def energy_integrated_data(self):
        if self._energy_integrated_data.size == 0:
            self._energy_integrated_data = np.trapz(
                self.doubly_differential_data, axis=1
            )  # f(θ) // integrate over ħω
        return self._energy_integrated_data

    def doubly_differential(self, fig):
        p2d = Plot2D(
            fig=fig,
            arr2d=self.doubly_differential_data,
            h_axis=self.ħω,
            v_axis=self.θ,
            xlabel=r"$\hbar \omega$ [%s]" % self.ħω.units,
            ylabel=r"$\theta$ [%s]" % self.θ.units,
            zlabel=r"$\frac{d^2I}{d \hbar \omega d \Omega}$",
            hslice_val=self.θc * u.miliradian,
            vslice_val=self.rad.ħωc,
            hslice_opts={"color": "#1f77b4", "lw": 1.5, "ls": "-"},
            vslice_opts={"color": "#d62728", "ls": "-"},
            cbar=False,
        )
        return p2d.fig

    def angle_integrated(self, ax, **kwargs):
        color = kwargs.pop("color", "#d62728")

        ax.plot(self.ħω, self.angle_integrated_data, color=color)

        ax.fill_between(
            self.ħω,
            self.angle_integrated_data,
            where=self.ħω <= self.rad.ħωc,
            facecolor=color,
            alpha=0.5,
        )

        ax.set(
            xlim=(self.ħω[0], self.ħω[-1]),
            ylabel=r"$\frac{dI}{d \hbar \omega}$",
            xlabel=r"$\hbar \omega$ [%s]" % self.ħω.units,
        )

        ax.set_ylim(bottom=0)
        ax.ticklabel_format(
            axis="y",
            style="scientific",
            scilimits=(0, 0),
            useOffset=False,
            useMathText=True,
        )

        x_pos = self.rad.ħω_avg.to_value(u.kiloelectronvolt)
        ax.axvline(x=x_pos, linestyle="--", color=color)

        omega_average = AnnotationText(text=r"$\langle \omega \rangle$", xy=(x_pos, 0))
        ax.annotate(
            s=omega_average.text, xy=omega_average.xy, xycoords=omega_average.xycoords
        )

        ax.set_title("Integrated Frequency Spectrum")

        return ax

    def energy_integrated(self, ax, **kwargs):
        color = kwargs.pop("color", "#d62728")

        ax.plot(self.θ, self.energy_integrated_data, color=color)

        ax.fill_between(
            self.θ,
            self.energy_integrated_data,
            where=self.θ <= self.θc,
            facecolor=color,
            alpha=0.5,
        )

        ax.set(
            xlim=(self.θ[0], self.θ[-1]),
            ylabel=r"$\frac{dI}{d \Omega}$",
            xlabel=r"$\theta$ [%s]" % self.θ.units,
        )

        ax.set_ylim(bottom=0)
        ax.ticklabel_format(
            axis="y",
            style="scientific",
            scilimits=(0, 0),
            useOffset=False,
            useMathText=True,
        )

        ax.set_title("Integrated Angular Spectrum")

        return ax


class Radiator(BaseClass):
    """Class for estimating the properties of emitted radiation of a given laser-plasma.

    Attributes
    ----------
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

    References
    ----------
    See [JCK]_, Section 14.6.

    .. [JCK] Jackson, J. D. (1999). Classical electrodynamics.

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

        self.ħωc = (3 / 2 * self.K * self.γ ** 2 * u.h * u.clight / self.λu).to(
            "kiloelectronvolt"
        )
        self.ωc = (self.ħωc / u.hbar).to(1 / u.fs)
        self.ħω_avg = (8 / (15 * np.sqrt(3)) * self.ħωc).to("kiloelectronvolt")
        self.ω_avg = (self.ħω_avg / u.hbar).to(1 / u.fs)
        self.Nγ = 5 * np.sqrt(3) * np.pi * α * self.K / 6
        self.θ_par = (self.K / self.γ * u.radian).to("miliradian")
        self.N_shot = (self.Nγ * self.Nβ * self.plasma.N).to("dimensionless")

        if self.K < 5:
            warnings.warn(
                "K = %.1f < 5. The radiation may be emitted in the undulator regime (K < 1)."
                % self.K.to_value(u.dimensionless)
            )
            # todo implement undulator case
            # raise NotImplementedError("The undulator case is not yet implemented.")

    def __eq__(self, other):
        return super().__eq__(other)

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

    def __repr__(self):
        return f"<{self.__class__.__name__}({repr(self.plasma)})>"
