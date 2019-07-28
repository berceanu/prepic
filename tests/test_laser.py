"""
Tests for prepic.laser module

"""
import pytest
from prepic import GaussianBeam, Laser
from prepic.laser import w0_to_fwhm, fwhm_to_w0, a0_from_intensity, intensity_from_a0
from unyt import assert_allclose_units
import unyt as u


def test_w0_to_fwhm(cet_param):
    """The the beam-waist to full-width half-maximum conversion."""
    fwhm = w0_to_fwhm(cet_param.w0)
    assert_allclose_units(fwhm, cet_param.fwhm)


def test_waist_fwhm(cet_param):
    """Round-trip checking of w0 to fwhm to w0 conversion."""
    assert_allclose_units(fwhm_to_w0(w0_to_fwhm(cet_param.w0)), cet_param.w0)


def test_a0_intensity(cet_param):
    """Round-trip checking of a0 to io to a0 conversion."""
    assert_allclose_units(
        a0_from_intensity(intensity_from_a0(cet_param.a0)), cet_param.a0
    )


def test_beam_constructors(cet_plasma, cet_param):
    g3 = cet_plasma.laser.beam
    g1 = GaussianBeam.from_f_number(f_number=cet_param.f_number)
    g2 = GaussianBeam.from_focal_distance(
        focal_distance=cet_param.focal_distance, beam_diameter=cet_param.beam_diameter
    )
    with pytest.raises(ValueError):
        _ = GaussianBeam(w0=cet_param.w0, fwhm=cet_param.fwhm)
    assert g1 == g2
    assert g2 == g3


def test_rayleigh(cet_plasma, cet_param):
    """Check Rayleigh length."""
    assert_allclose_units(cet_plasma.laser.beam.zR, cet_param.zR)


def test_laser_constructors(cet_plasma, cet_param):
    _ = Laser(ɛL=cet_param.ɛL, τL=cet_param.τL, beam=GaussianBeam())

    with pytest.raises(TypeError):
        _ = Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL)

    cetbeam = GaussianBeam(w0=cet_param.w0)

    with pytest.raises(TypeError):
        _ = Laser.from_power(power=cet_param.power, beam=cetbeam)

    l1 = Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL, τL=cet_param.τL)
    l2 = cet_plasma.laser
    l3 = Laser.from_intensity(
        intensity=cet_param.intensity, ɛL=cet_param.ɛL, τL=cet_param.τL
    )
    l4 = Laser.from_intensity(
        intensity=cet_param.intensity, ɛL=cet_param.ɛL, beam=cetbeam
    )
    l5 = Laser.from_intensity(
        intensity=cet_param.intensity, τL=cet_param.τL, beam=cetbeam
    )
    l6 = Laser.from_power(power=cet_param.power, τL=cet_param.τL, beam=cetbeam)
    l7 = Laser.from_power(power=cet_param.power, ɛL=cet_param.ɛL, beam=cetbeam)
    assert l1 == l2
    assert l2 == l3
    assert l3 == l4
    assert l4 == l5
    assert l5 == l6
    assert l6 == l7


def test_laser(cet_plasma, cet_param):
    """Check Laser class."""
    laser = cet_plasma.laser

    assert_allclose_units(laser.ɛL, cet_param.ɛL)
    assert_allclose_units(laser.ncrit, 1741.95959e18 / u.cm ** 3)

    assert_allclose_units(laser.kL, 7.85398163 * 1 / u.micrometer)
    assert_allclose_units(laser.ωL, 2.35456446 * 1 / u.femtosecond)
    assert_allclose_units(laser.P0, 180.84167615 * u.terawatt)
    assert_allclose_units(laser.I0, 3.5533144e19 * u.watt / u.cm ** 2)
    assert_allclose_units(laser.E0, 16362.40354854 * u.megavolt / u.mm)
