"""
Classes for modelling the laser pulse, without any matching

"""
from prepic.base import BaseClass
from prepic.constants import r_e
from prepic.ionization import helium_ionization_state
import numpy as np
import unyt as u

dim = u.dimensions


@u.accepts(w0=dim.length)
def w0_to_fwhm(w0):
    """Computes Gaussian laser FWHM from its beam waist.

    Args:
        w0 (float, length): beam waist @ 1/e^2 intensity

    Returns:
        fwhm (float, length): beam FWHM @ 1/2 intensity
    """
    return 2 * w0 / np.sqrt(2 / np.log(2))


@u.accepts(fwhm=dim.length)
def fwhm_to_w0(fwhm):
    """Computes Gaussian laser beam waist from its FWHM.

    Args:
        fwhm (float, length): beam FWHM @ 1/2 intensity

    Returns:
        w0 (float, length): beam waist @ 1/e^2 intensity
    """
    return 1 / 2 * np.sqrt(2 / np.log(2)) * fwhm


@u.accepts(a0=dim.dimensionless, λL=dim.length)
def intensity_from_a0(a0, λL=0.8 * u.micrometer):
    """Compute peak laser intensity in the focal plane.

    Args:
        a0 (float, dimensionless): normalized laser vector potential
        λL (float, length): laser wavelength

    Returns:
        I0 (float, energy/time/area): peak laser intensity in the focal plane
    """
    return np.pi / 2 * u.clight / r_e * u.me * u.clight ** 2 / λL ** 2 * a0 ** 2


@u.accepts(i0=dim.flux, λL=dim.length)
def a0_from_intensity(i0, λL=0.8 * u.micrometer):
    """Compute laser normalized vector potential.

    Args:
        i0 (float, energy/time/area): peak laser intensity in the focal plane
        λL (float, dim.length): laser wavelength

    Returns:
        a0 (float, dimensionless): normalized laser vector potential
    """
    return np.sqrt(i0 / (np.pi / 2 * u.clight / r_e * u.me * u.clight ** 2 / λL ** 2))


class GaussianBeam(BaseClass):
    """Contains the (geometric) parameters for a Gaussian laser beam.

    Attributes
    ----------
    w0 : float, length
        Beam waist @ 1/e^2 intensity.
    fwhm : float, length
        Beam FWHM @ 1/2 intensity
    λL : float, length
        Wavelength.
    zR : float, length
        Rayleigh length.

    Examples
    --------
    >>> import unyt as u
    >>> mybeam = GaussianBeam.from_focal_distance(focal_distance=3 * u.m,
    ...                             beam_diameter=200 * u.mm, λL=1.0 * u.micrometer)
    >>> mybeam
    <GaussianBeam(13.504744742356593 µm, 1.0 µm)>
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
            self.zR = (np.pi * self.w0 ** 2 / self.λL).to("milimeter")
        else:
            self.zR = None

    @classmethod
    @u.accepts(f_number=dim.dimensionless, λL=dim.length)
    def from_f_number(cls, f_number, λL=0.8 * u.micrometer):
        """Construct beam by giving the OAP's f/#.

        Args:
            :param f_number: f/# of the off-axis parabolic mirror (float, dimensionless)
            :param λL: laser wavelength (float, length, optional)
        """
        w0 = 2 * np.sqrt(2) / np.pi * λL * f_number
        return cls(w0=w0, λL=λL)

    @classmethod
    @u.accepts(focal_distance=dim.length, beam_diameter=dim.length, λL=dim.length)
    def from_focal_distance(cls, focal_distance, beam_diameter, λL=0.8 * u.micrometer):
        """Constuct beam from OAP's focal distance and beam diameter.

        Args:
            :param focal_distance: focal distance of the off-axis parabolic mirror (float, units of length)
            :param beam_diameter: beam diameter after compressor (float, units of length)
            :param λL: laser wavelength (float, length, optional)
        """
        return cls.from_f_number(f_number=focal_distance / beam_diameter, λL=λL)

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.w0}, {self.λL})>"

    def __str__(self):
        msg = f"beam with λL={self.λL:.2f}"
        if self.w0 and self.fwhm:
            msg = f"beam with w0={self.w0:.1f} (FWHM={self.fwhm:.1f}), zᵣ={self.zR:.2f}, λL={self.λL:.2f}"
        return msg


class Laser(BaseClass):
    """Class containing laser parameters.

    Attributes
    ----------
    beam : :obj:`GaussianBeam`
        Class instance containing beam params.
    ɛL : float, energy
        Pulse energy on target (after compressor
        and beam transport, focused into the FWHM@intensity spot).
    τL : float, time
        Pulse duration at FWHM in intensity
    kL : float, 1/length
        Wavenumber.
    ωL : float, 1/time
        Agular frequency.
    ncrit : float, 1/volume
        Critical plasma density for this laser.
    P0 : float, energy/time
        Power.
    I0 : float, energy/time/area
        Peak intensity in the focal plane.
    a0 : float, dimensionless
        Normalized vector potential.
    E0 : float, energy/charge/length
        Peak electric field.

    Examples
    --------
    >>> import unyt as u
    >>> from prepic import GaussianBeam
    >>> mylaser = Laser.from_power(power=10 * u.petawatt, ɛL=300 * u.joule,
    ...                            beam=GaussianBeam(w0=5 * u.micrometer))
    >>> mylaser
    <Laser(300.0 J, 28.18311836098954 fs, <GaussianBeam(5.0 µm, 0.8 µm)>)>
    """

    def __init__(self, ɛL, τL, beam=GaussianBeam()):
        """Creates laser with given parameters."""
        self.beam = beam
        self.kL = (2 * np.pi / self.beam.λL).to("1/micrometer")
        self.ωL = (u.clight * self.kL).to("1/femtosecond")
        self.ncrit = (np.pi / (r_e * self.beam.λL ** 2)).to("1/cm**3")

        self.ɛL = ɛL.to("joule")
        self.τL = τL.to("femtosecond")
        self.P0 = (2 * np.sqrt(np.log(2) / np.pi) * self.ɛL / self.τL).to("terawatt")

        if self.beam.w0:
            self.I0 = (
                2
                / np.pi
                * np.sqrt(4 * np.log(2) / np.pi)
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
        prefactor = (4 * np.log(2) / np.pi) ** (3 / 2)
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

        Must supply either ɛL or τL and a beam with a defined size.
        """
        if beam.w0 is None:
            raise AssertionError("Must supply laser beam size!")
        prefactor = 2 * np.sqrt(np.log(2) / np.pi)

        if ɛL and (not τL):
            τL = prefactor * ɛL / power
        elif τL and (not ɛL):
            ɛL = power * τL / prefactor
        else:  # either both or none
            raise TypeError("Must supply either ɛL or τL.")

        return cls(ɛL=ɛL, τL=τL, beam=beam)

    def __eq__(self, other):
        return super().__eq__(other)

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
