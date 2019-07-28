"""
Tests for prepic.lwfa module

"""
import pytest
import unyt as u
from unyt import assert_allclose_units

from prepic import matched_laser_plasma


def test_matched_laser_plasma(cet_param):
    """Check laser-plasma matching function."""
    match = matched_laser_plasma(cet_param.a0)

    assert_allclose_units(match.ΔE, 1564.41581593 * u.megaelectronvolt)
    assert_allclose_units(match.Q, 533.34275131 * u.picocoulomb)
    assert_allclose_units(match.η, 0.1228936 * u.dimensionless)

    with pytest.raises(ValueError):
        _ = matched_laser_plasma(a0=0.5 * u.dimensionless)
