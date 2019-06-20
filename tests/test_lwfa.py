#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lwfa` module."""

import unyt as u
from numpy.testing import assert_almost_equal
from unyt import dimensions

import prepic.lwfa as lwfa


def test_w0_to_fwhm():
    """The the beam-waist to full-width half-maximum conversion."""
    res = lwfa.w0_to_fwhm(7.202530529256849)
    assert_almost_equal(res, 8.4803316326207, 4)


def test_waist_fwhm():
    """Round-trip checking of w0 to fwhm to w0 conversion."""
    w0 = 10.0
    assert_almost_equal(lwfa.fwhm_to_w0(lwfa.w0_to_fwhm(w0)), w0, 4)


def test_a0_intensity():
    """Round-trip checking of a0 to io to a0 conversion."""
    a0 = 2.0
    assert_almost_equal(lwfa.a0_from_intensity(lwfa.intensity_from_a0(a0)), a0, 4)


def test_gaussianbeam():
    """Check GaussianBeam class."""
    zR = lwfa.GaussianBeam(w0=30.0 * u.micrometer).zR
    assert zR.units.dimensions == dimensions.length
    assert_almost_equal(zR.to_value('micrometer'), 3534.29173, 4)


def test_laser():
    """Check Laser class."""
    laser = lwfa.Laser.from_a0(
        a0=2.343,
        τL=40.0 * u.femtosecond,
        beam=lwfa.GaussianBeam(w0=30.0 * u.micrometer),
    )
    assert laser.ɛL.units.dimensions == dimensions.energy
    assert_almost_equal(laser.ɛL.to_value('joule'), 7.06412, 4)

    assert laser.kL.units.dimensions == 1 / dimensions.length
    assert_almost_equal(laser.kL.to_value('1/micrometer'), 7.85398, 4)

    assert laser.ωL.units.dimensions == 1 / dimensions.time
    assert_almost_equal(laser.ωL.to_value('1/femtosecond'), 2.3546, 4)

    assert laser.P0.units.dimensions == dimensions.power
    assert_almost_equal(laser.P0.to_value('terawatt'), 165.90753, 4)

    assert laser.I0.units.dimensions == dimensions.flux
    assert_almost_equal(laser.I0.to_value('exawatt/cm**2'), 11.73555, 4)

    assert laser.E0.units.dimensions == dimensions.electric_field_mks
    assert_almost_equal(laser.E0.to_value('megavolt/mm'), 9403.34009, 4)

    assert laser.ncrit.units.dimensions == dimensions.number_density
    assert_almost_equal(laser.ncrit.to_value('1e18/cm**3'), 1741.95959, 4)


def test_plasma():
    """Check Plasma class."""
    electron_density = 0.294e18 / u.cm ** 3
    plasma = lwfa.Plasma(n_pe=electron_density)

    assert plasma.λp.units.dimensions == dimensions.length
    assert_almost_equal(plasma.λp.to_value('micrometer'), 61.57938, 4)

    assert plasma.kp.units.dimensions == 1 / dimensions.length
    assert_almost_equal(plasma.kp.to_value('1/micrometer'), 0.102, 4)

    assert plasma.ωp.units.dimensions == 1 / dimensions.time
    assert_almost_equal(plasma.ωp.to_value('1/femtosecond'), 0.0305, 4)


def test_plasma_with_laser():
    """Check Plasma class when given a Laser."""
    laser = lwfa.Laser.from_a0(
        a0=2.343,
        τL=40.0 * u.femtosecond,
        beam=lwfa.GaussianBeam(w0=30.0 * u.micrometer),
    )
    electron_density = 0.294e18 / u.cm ** 3
    plasma = lwfa.Plasma(n_pe=electron_density, laser=laser)

    assert plasma.Pc.units.dimensions == dimensions.power
    assert_almost_equal(plasma.Pc.to_value('terawatt'), 100.72555, 4)

    assert plasma.depletion.units.dimensions == dimensions.length
    assert_almost_equal(plasma.depletion.to_value('micrometer'), 71051.20374, 4)

    assert plasma.dephasing.units.dimensions == dimensions.length
    assert_almost_equal(plasma.dephasing.to_value('micrometer'), 118514.39895, 4)


def test_matched_laser_plasma():
    """Check laser-plasma matching function."""
    a0 = 2.343
    match = lwfa.matched_laser_plasma(a0)

    assert match.ΔE.units.dimensions == dimensions.energy
    assert_almost_equal(match.ΔE.to_value('megaelectronvolt'), 56.35889, 4)

    assert match.Q.units.dimensions == dimensions.charge_mks
    assert_almost_equal(match.Q.to_value('picocoulomb'), 58.17636, 4)

    assert_almost_equal(match.η, 0.21384, 4)


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

