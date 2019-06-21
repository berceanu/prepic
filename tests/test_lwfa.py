#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lwfa` module."""
from collections import namedtuple

import pytest
import unyt as u
from unyt._testing import assert_allclose_units

from prepic import lwfa
from prepic.lwfa import GaussianBeam, Laser, Plasma


@pytest.fixture(scope="module")
def cet_param():
    Params = namedtuple(
        "Cetal",
        [
            "f_number",
            "focal_distance",
            "beam_diameter",
            "w0",
            "fwhm",
            "a0",
            "ɛL",
            "τL",
            "intensity",
            "power",
        ],
    )
    cetal = Params(
        f_number=24.9912 * u.dimensionless,
        focal_distance=3.2 * u.meter,
        beam_diameter=128.045071 * u.mm,
        w0=18 * u.micrometer,
        fwhm=21.193380405278543 * u.micrometer,
        a0=4.076967454355432 * u.dimensionless,
        ɛL=7.7 * u.joule,
        τL=40 * u.femtosecond,
        intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
        power=180.84167614968285 * u.terawatt,
    )
    return cetal


@pytest.fixture(scope="module")
def cet_plasma(cet_param):
    return Plasma(
        n_pe=0.294e18 / u.cm ** 3,
        laser=Laser.from_a0(
            a0=cet_param.a0, ɛL=cet_param.ɛL, beam=GaussianBeam(w0=cet_param.w0)
        ),
    )


def test_plasma_with_laser(cet_plasma):
    """Check Plasma class when given a Laser."""
    assert_allclose_units(cet_plasma.Pc, 100.72555 * u.terawatt)
    assert_allclose_units(cet_plasma.depletion, 71051.20374 * u.micrometer)
    assert_allclose_units(cet_plasma.dephasing, 118514.39895 * u.micrometer)


def test_beam_constructors(cet_param):
    g1 = GaussianBeam.from_f_number(f_number=cet_param.f_number)
    g2 = GaussianBeam.from_focal_distance(
        focal_distance=cet_param.focal_distance, beam_diameter=cet_param.beam_diameter
    )
    g3 = GaussianBeam(w0=cet_param.w0)
    assert (g1 == g2) and (g2 == g3)


def test_laser_constructors(cet_param):
    cetbeam = GaussianBeam(w0=cet_param.w0)
    l1 = Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL, τL=cet_param.τL)
    l2 = Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL, beam=cetbeam)
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
    assert (
        (l1 == l2)
        and (l2 == l3)
        and (l3 == l4)
        and (l4 == l5)
        and (l5 == l6)
        and (l6 == l7)
    )


def test_w0_to_fwhm(cet_param):
    """The the beam-waist to full-width half-maximum conversion."""
    fwhm = lwfa.w0_to_fwhm(cet_param.w0)
    assert_allclose_units(fwhm, cet_param.fwhm)


def test_waist_fwhm(cet_param):
    """Round-trip checking of w0 to fwhm to w0 conversion."""
    assert_allclose_units(lwfa.fwhm_to_w0(lwfa.w0_to_fwhm(cet_param.w0)), cet_param.w0)


def test_a0_intensity():
    """Round-trip checking of a0 to io to a0 conversion."""
    assert_allclose_units(
        lwfa.a0_from_intensity(lwfa.intensity_from_a0(cet_param.a0)), cet_param.a0
    )


def test_rayleigh():
    """Check Rayleigh length."""
    zR = lwfa.GaussianBeam(w0=30.0 * u.micrometer).zR
    assert zR.units.dimensions == dimensions.length
    assert_allclose_units(zR.to_value("micrometer"), 3534.29173, 4)


def test_laser():
    """Check Laser class."""
    laser = lwfa.Laser.from_a0(
        a0=2.343,
        τL=40.0 * u.femtosecond,
        beam=lwfa.GaussianBeam(w0=30.0 * u.micrometer),
    )
    assert_allclose_units(laser.ɛL.to_value("joule"), 7.06412, 4)

    assert_allclose_units(laser.kL.to_value("1/micrometer"), 7.85398, 4)

    assert_allclose_units(laser.ωL.to_value("1/femtosecond"), 2.3546, 4)

    assert_allclose_units(laser.P0.to_value("terawatt"), 165.90753, 4)

    assert_allclose_units(laser.I0.to_value("exawatt/cm**2"), 11.73555, 4)

    assert_allclose_units(laser.E0.to_value("megavolt/mm"), 9403.34009, 4)

    assert_allclose_units(laser.ncrit.to_value("1e18/cm**3"), 1741.95959, 4)


def test_plasma():
    """Check Plasma class."""
    electron_density = 0.294e18 / u.cm ** 3
    plasma = lwfa.Plasma(n_pe=electron_density)

    assert_allclose_units(plasma.λp.to_value("micrometer"), 61.57938, 4)

    assert_allclose_units(plasma.kp.to_value("1/micrometer"), 0.102, 4)

    assert_allclose_units(plasma.ωp.to_value("1/femtosecond"), 0.0305, 4)


def test_matched_laser_plasma():
    """Check laser-plasma matching function."""
    a0 = 2.343
    match = lwfa.matched_laser_plasma(a0)

    assert_allclose_units(match.ΔE.to_value("megaelectronvolt"), 56.35889, 4)

    assert_allclose_units(match.Q.to_value("picocoulomb"), 58.17636, 4)

    assert_allclose_units(match.η, 0.21384, 4)

# def test_simulation():
#     """Check Simulation class."""
# msg = ("simulation with box size ({0.L:.1f})³, Δx={0.Δx:.3f}, Δy={0.Δy:.3f}, "
#        "Δz={0.Δz:.3f}, nx={0.nx}, ny={0.ny}, nz={0.nz}, {0.npart:e} macro-particles, "
#        "{0.nstep:e} time steps")

# todo: add proper regression test for CETAL parameters, with parametrized fixtures
# https://docs.pytest.org/en/latest/fixture.html#scope-sharing-a-fixture-instance-across-tests-in-a-class-module-or-session
# https://docs.pytest.org/en/latest/fixture.html#parametrizing-fixtures

# todo implement an `__eq__` (or "nearly equals") method for your `Laser` class, if you don't already have one,
#  to be able to easily compare objects. Then your tests can try constructing the same object via multiple
#  constructors and see if they come out equal.

# todo publish in JOSS (see unyt's GH page)

# # CETAL laser
# npe_cetal = 1.5e18 / u.cm ** 3
# beam_cetal = GaussianBeam(w0=18 * u.micrometer)
# laser_cetal = Laser(ɛL=7.7 * u.joule, τL=40 * u.femtosecond, beam=beam_cetal)
# Plasma(n_pe=npe_cetal)
# plasma_cetal = Plasma(n_pe=npe_cetal, laser=laser_cetal)
# bubble_R_cetal = (2 * np.sqrt(plasma_cetal.laser.a0) / plasma_cetal.kp).to(
#     "micrometer"
# )
# plasma_cetal = Plasma(
#     n_pe=npe_cetal, laser=laser_cetal, bubble_radius=bubble_R_cetal
# )
# matched_cetal = matched_laser_plasma(a0=4.1)
