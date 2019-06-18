#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `prepic` package."""

import pytest


from prepic import prepic


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `lwfa_predictor` package."""

import pytest
from numpy.testing import assert_almost_equal


from lwfa_predictor import lwfa_predictor as lwfa

# todo add more tests

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_w0_to_fwhm():
    """The the beam-waist to full-width half-maximum conversion."""
    res = lwfa.w0_to_fwhm(10.0)
    assert_almost_equal(res, 11.77410, 4)
