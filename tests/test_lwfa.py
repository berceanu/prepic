#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lwfa` module."""

from prepic import lwfa
from numpy.testing import assert_almost_equal

def test_w0_to_fwhm():
    """The the beam-waist to full-width half-maximum conversion."""
    res = lwfa.w0_to_fwhm(10.0)
    assert_almost_equal(res, 11.77410, 4)

