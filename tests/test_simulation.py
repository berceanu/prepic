"""
Tests for prepic.simulation module

"""
import pytest
from prepic import Simulation, Plasma
from unyt import assert_allclose_units
import unyt as u


def test_simulation(cet_plasma, cet_param):
    """Check Simulation class."""

    with pytest.raises(TypeError):
        _ = Simulation(
            Plasma(n_pe=cet_param.npe, propagation_distance=13.56555928 * u.mm)
        )

    sim = Simulation(cet_plasma)

    assert_allclose_units(sim.L, 109.04942675 * u.micrometer)
    assert_allclose_units(sim.Δx, 0.43389388 * u.micrometer)
    assert_allclose_units(sim.Δz, 0.04 * u.micrometer)

    assert_allclose_units(sim.nx, 251 * u.dimensionless)
    assert_allclose_units(sim.nz, 2726 * u.dimensionless)
    assert_allclose_units(sim.npart, 1373925808 * u.dimensionless)
    assert_allclose_units(sim.nstep, 341865 * u.dimensionless)

    sim2 = Simulation(cet_plasma, box_length=4 * cet_plasma.λp, ppc=8)
    assert sim2 == sim
