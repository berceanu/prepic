"""
Tests for prepic.plasma module

"""
from prepic import Plasma
from unyt import assert_allclose_units
import unyt as u


def test_plasma(cet_plasma, cet_param):
    """Check Plasma class."""
    assert_allclose_units(cet_plasma.λp, 27.26235669 * u.micrometer)
    assert_allclose_units(cet_plasma.kp, cet_param.kp)
    assert_allclose_units(cet_plasma.ωp, 0.0690935 * 1 / u.femtosecond)


def test_plasma_with_laser(cet_plasma, cet_param):
    """Check Plasma class when given a Laser."""
    # test constructor with no bubble_radius and given propagation_distance
    _ = Plasma(
        n_pe=cet_param.npe,
        laser=cet_plasma.laser,
        propagation_distance=13.56555928 * u.mm,
    )

    assert_allclose_units(cet_plasma.Pc, 19.7422087 * u.terawatt)
    assert_allclose_units(cet_plasma.depletion, 13.92603593 * u.mm)
    assert_allclose_units(cet_plasma.dephasing, 13.56555928 * u.mm)
