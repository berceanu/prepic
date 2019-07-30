"""
Classes for modelling emitted radiation from laser-plasma interaction

"""
import warnings
from functools import partial

import numpy as np
import unyt as u
from matplotlib.figure import Figure
from scipy.integrate import quad
from scipy.special import kv
from unyt import accepts, returns
from unyt.dimensions import dimensionless, energy, time, angle

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


@returns(dimensionless)
@accepts(θ=angle, ωc=1 / time, γ=dimensionless)
def photon_angle_distribution(θ, ωc, γ):
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
    ωc : float, 1/time
        Critical synchrotron frequency.
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
    >>> NΩ = photon_angle_distribution(θ=45 * u.degree, ωc=3e5 / u.fs, γ=5e3 * u.dimensionless)
    >>> print("{:.1e}".format(NΩ))
    9.8e-14 dimensionless
    """
    θ = θ.to_value(u.radian)
    a = 7 * u.qe ** 2 / (96 * np.pi * u.eps_0 * u.hbar * u.clight)  # prefactor
    dN_over_dΩ = (
        a.to(dimensionless)
        * γ ** 2
        / ((1 + γ ** 2 * θ ** 2) ** (5 / 2))
        * (1 + 5 / 7 * γ ** 2 * θ ** 2 / (1 + γ ** 2 * θ ** 2))
    )
    return dN_over_dΩ.to("dimensionless")


class Spectrum:
    r"""Base class for holding raw spectrum data and plotting it."""

    def __init__(self, horiz_axis_data, spectrum, horiz_norm=None):
        """Constructs spectrum from raw data.

        Parameters
        ----------
        horiz_axis_data : 1d array_like
            Raw data (incl. units) for the dependent variable, ie. frequency, angle etc.
        horiz_norm : float
            Normalization factor (incl. units) to be applied to `horiz_axis_data`.
        spectrum : 1d array_like
            Raw data (incl. units) for the dependent variable, ie. the spectrum data.

        """
        if horiz_norm is None:
            self.x_data = horiz_axis_data
        else:
            # norm_freq
            self.x_data = (horiz_axis_data / horiz_norm).to_value(dimensionless)

        if spectrum.units == u.dimensionless:
            # distr
            self.spectrum = spectrum.to_value(dimensionless)
        else:
            self.spectrum = spectrum

    def plot(self, ax=None, fig_width=6.4, fill_between=True, mark_val=None):
        if ax is None:
            fig = Figure()
            fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

            fig_height = fig_width / 1.618  # golden ratio
            fig.set_size_inches(fig_width, fig_height)

            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        ax.plot(self.x_data, self.spectrum)

        if fill_between:
            ax.fill_between(
                self.x_data,
                self.spectrum,
                where=self.x_data < 1,
                facecolor="C3",
                alpha=0.5,
            )

        if mark_val is not None:
            ax.axvline(x=mark_val["position"], linestyle="--", color="C3")
            ax.text(mark_val["position"], 0, mark_val["label"])

        ax.set_ylim(bottom=0)

        return ax


class SynchrotronFrequencySpectrum(Spectrum):
    def __init__(self, horiz_axis_data, spectrum, horiz_norm, vline):
        super().__init__(horiz_axis_data, spectrum, horiz_norm)
        self.mark_val = dict(
            position=vline.to_value(dimensionless), label=r"$\langle \omega \rangle$"
        )

    def plot(self, ax=None, fig_width=6.4, **kwargs):
        ax = super().plot(
            ax=ax, fig_width=fig_width, fill_between=True, mark_val=self.mark_val
        )

        ax.set(
            ylabel=r"$\frac{dN}{dy}$",
            xlabel=r"$y = \omega / \omega_c$",
            xlim=[-0.1, 2.0],
        )

        return ax.figure


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
            self.ωc = (self.ħωc / u.hbar).to(1 / u.fs)
            self.ħω_avg = (8 / (15 * np.sqrt(3)) * self.ħωc).to("kiloelectronvolt")
            self.ω_avg = (self.ħω_avg / u.hbar).to(1 / u.fs)
            self.Nγ = 5 * np.sqrt(3) * np.pi * self.α * self.K / 6
            self.θ_par = (self.K / self.γ * u.radian).to("miliradian")
            self.N_shot = (self.Nγ * self.Nβ * self.plasma.N).to("dimensionless")
        else:
            # todo implement undulator case
            raise NotImplementedError("The undulator case is not yet implemented.")

    def frequency_spectrum(self, ω=None):
        r"""Computes photon frequency distribution over a range of frequencies.

        Computes the number of synchrotron photons emitted at each of the frequencies in `ω`.

        Parameters
        ----------
        ω : (N,) unyt_array of floats
            Frequency container, with dimensions of 1/time.
            Defaults to ``np.linspace(1e-5 * self.ωc, 2 * self.ωc, 100)``

        Returns
        -------

        Examples
        --------

        """
        freq_dist = partial(photon_frequency_distribution, ωc=self.ωc, γ=self.γ)

        if ω is None:
            ω = np.linspace(1e-5 * self.ωc, 2 * self.ωc, 100)

        # call once to get unit
        unit_of_spectrum = freq_dist(ω[0]).units

        # pre-allocate
        spectrum = np.empty(ω.size) * unit_of_spectrum

        for i, freq in enumerate(ω):
            spectrum[i] = freq_dist(freq)

        return SynchrotronFrequencySpectrum(
            horiz_axis_data=ω,
            spectrum=spectrum,
            horiz_norm=self.ωc,
            vline=self.ω_avg / self.ωc,
        )

    def angular_spectrum(self):
        raise NotImplementedError

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
