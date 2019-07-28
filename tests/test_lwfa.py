#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lwfa` module."""


import pytest
import unyt as u
from unyt import assert_allclose_units

from prepic import lwfa


def test_beam_constructors(cet_plasma, cet_param):
    g3 = cet_plasma.laser.beam
    g1 = lwfa.GaussianBeam.from_f_number(f_number=cet_param.f_number)
    g2 = lwfa.GaussianBeam.from_focal_distance(
        focal_distance=cet_param.focal_distance, beam_diameter=cet_param.beam_diameter
    )
    with pytest.raises(ValueError):
        _ = lwfa.GaussianBeam(w0=cet_param.w0, fwhm=cet_param.fwhm)
    assert g1 == g2
    assert g2 == g3


def test_laser_constructors(cet_plasma, cet_param):
    _ = lwfa.Laser(ɛL=cet_param.ɛL, τL=cet_param.τL, beam=lwfa.GaussianBeam())

    with pytest.raises(TypeError):
        _ = lwfa.Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL)

    cetbeam = lwfa.GaussianBeam(w0=cet_param.w0)

    with pytest.raises(TypeError):
        _ = lwfa.Laser.from_power(power=cet_param.power, beam=cetbeam)

    l1 = lwfa.Laser.from_a0(a0=cet_param.a0, ɛL=cet_param.ɛL, τL=cet_param.τL)
    l2 = cet_plasma.laser
    l3 = lwfa.Laser.from_intensity(
        intensity=cet_param.intensity, ɛL=cet_param.ɛL, τL=cet_param.τL
    )
    l4 = lwfa.Laser.from_intensity(
        intensity=cet_param.intensity, ɛL=cet_param.ɛL, beam=cetbeam
    )
    l5 = lwfa.Laser.from_intensity(
        intensity=cet_param.intensity, τL=cet_param.τL, beam=cetbeam
    )
    l6 = lwfa.Laser.from_power(power=cet_param.power, τL=cet_param.τL, beam=cetbeam)
    l7 = lwfa.Laser.from_power(power=cet_param.power, ɛL=cet_param.ɛL, beam=cetbeam)
    assert l1 == l2
    assert l2 == l3
    assert l3 == l4
    assert l4 == l5
    assert l5 == l6
    assert l6 == l7


def test_w0_to_fwhm(cet_param):
    """The the beam-waist to full-width half-maximum conversion."""
    fwhm = lwfa.w0_to_fwhm(cet_param.w0)
    assert_allclose_units(fwhm, cet_param.fwhm)


def test_waist_fwhm(cet_param):
    """Round-trip checking of w0 to fwhm to w0 conversion."""
    assert_allclose_units(lwfa.fwhm_to_w0(lwfa.w0_to_fwhm(cet_param.w0)), cet_param.w0)


def test_a0_intensity(cet_param):
    """Round-trip checking of a0 to io to a0 conversion."""
    assert_allclose_units(
        lwfa.a0_from_intensity(lwfa.intensity_from_a0(cet_param.a0)), cet_param.a0
    )


def test_rayleigh(cet_plasma, cet_param):
    """Check Rayleigh length."""
    assert_allclose_units(cet_plasma.laser.beam.zR, cet_param.zR)


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


def test_plasma(cet_plasma, cet_param):
    """Check Plasma class."""
    assert_allclose_units(cet_plasma.λp, 27.26235669 * u.micrometer)
    assert_allclose_units(cet_plasma.kp, cet_param.kp)
    assert_allclose_units(cet_plasma.ωp, 0.0690935 * 1 / u.femtosecond)


def test_plasma_with_laser(cet_plasma, cet_param):
    """Check Plasma class when given a Laser."""
    # test constructor with no bubble_radius and given propagation_distance
    _ = lwfa.Plasma(
        n_pe=cet_param.npe,
        laser=cet_plasma.laser,
        propagation_distance=13.56555928 * u.mm,
    )

    assert_allclose_units(cet_plasma.Pc, 19.7422087 * u.terawatt)
    assert_allclose_units(cet_plasma.depletion, 13.92603593 * u.mm)
    assert_allclose_units(cet_plasma.dephasing, 13.56555928 * u.mm)


def test_matched_laser_plasma(cet_param):
    """Check laser-plasma matching function."""
    match = lwfa.matched_laser_plasma(cet_param.a0)

    assert_allclose_units(match.ΔE, 1564.41581593 * u.megaelectronvolt)
    assert_allclose_units(match.Q, 533.34275131 * u.picocoulomb)
    assert_allclose_units(match.η, 0.1228936 * u.dimensionless)

    with pytest.raises(ValueError):
        _ = lwfa.matched_laser_plasma(a0=0.5 * u.dimensionless)


def test_simulation(cet_plasma, cet_param):
    """Check Simulation class."""

    with pytest.raises(TypeError):
        _ = lwfa.Simulation(
            lwfa.Plasma(n_pe=cet_param.npe, propagation_distance=13.56555928 * u.mm)
        )

    sim = lwfa.Simulation(cet_plasma)

    assert_allclose_units(sim.L, 109.04942675 * u.micrometer)
    assert_allclose_units(sim.Δx, 0.43389388 * u.micrometer)
    assert_allclose_units(sim.Δz, 0.04 * u.micrometer)

    assert_allclose_units(sim.nx, 251 * u.dimensionless)
    assert_allclose_units(sim.nz, 2726 * u.dimensionless)
    assert_allclose_units(sim.npart, 1373925808 * u.dimensionless)
    assert_allclose_units(sim.nstep, 341865 * u.dimensionless)

    sim2 = lwfa.Simulation(cet_plasma, box_length=4 * cet_plasma.λp, ppc=8)
    assert sim2 == sim
