"""
Tests for prepic.radiation module

"""
import pytest

from prepic import Radiator, Plasma


def test_radiator_without_laser(cet_param):
    """Check Radiator class constructor without passing a Laser."""
    plasma = Plasma(n_pe=cet_param.npe, bubble_radius=cet_param.w0)

    with pytest.raises(TypeError):
        _ = Radiator(plasma)
