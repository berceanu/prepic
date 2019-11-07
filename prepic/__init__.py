# -*- coding: utf-8 -*-

"""Top-level package for prepic.

The following functions and classes are importable from the
top-level ``prepic`` namespace:

* :class:`prepic.laser.GaussianBeam`
* :class:`prepic.laser.Laser`
* :class:`prepic.plasma.Plasma`
* :class:`prepic.radiation.Radiator`
* :class:`prepic.radiation.DifferentialSpectrum`
* :class:`prepic.simulation.Simulation`
* :func:`prepic.lwfa.matched_laser_plasma`
"""
from prepic.laser import GaussianBeam, Laser  # NOQA: F401
from prepic.plasma import Plasma  # NOQA: F401
from prepic.radiation import Radiator, DifferentialSpectrum  # NOQA: F401
from prepic.simulation import Simulation  # NOQA: F401
from prepic.lwfa import matched_laser_plasma  # NOQA: F401

__author__ = """Andrei Berceanu"""
__email__ = "andreicberceanu@gmail.com"
__version__ = "0.2.4"
