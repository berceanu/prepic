=======
pre-PIC
=======


.. image:: https://img.shields.io/pypi/v/prepic.svg
   :target: https://pypi.python.org/pypi/prepic


.. image:: https://img.shields.io/travis/berceanu/prepic.svg
   :target: https://travis-ci.org/berceanu/prepic


.. image:: https://readthedocs.org/projects/prepic/badge/?version=latest
   :target: https://prepic.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://pyup.io/repos/github/berceanu/prepic/shield.svg
   :target: https://pyup.io/repos/github/berceanu/prepic
   :alt: Updates


.. image:: https://codecov.io/gh/berceanu/prepic/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/berceanu/prepic


.. image:: https://img.shields.io/pypi/l/prepic.svg
   :target: https://github.com/berceanu/prepic/blob/master/LICENSE
   :alt: PyPI - License


.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/berceanu/prepic/cf6f61cceb859060fdc6d0af64032c7338f3b3fc?filepath=examples%2Fprepic.ipynb


analytically estimate various laser-plasma parameters for experiments and PIC simulations


* Free software: BSD license
* Documentation: https://prepic.readthedocs.io


Features
--------

* ``lwfa`` module for estimating bunch properties (charge, energy, etc.)
* small codebase
* only depends on `unyt <https://github.com/yt-project/unyt>`_


Quick start
-----------

Install the package via:

.. code-block:: console

        $ pip install prepic

Estimate ideal laser-plasma matching parameters based on scaling laws from [LTJT]_:

.. code-block:: python

    import unyt as u

    from prepic import lwfa

    laser_plasma = lwfa.matched_laser_plasma(a0=4.4 * u.dimensionless)
    print(laser_plasma)

::

    Plasma with nₚ=1.1e+18 cm**(-3) (6.06e-04 × nc), ωₚ=0.058 1/fs, kₚ=0.193 1/µm, λₚ=32.5 µm, Ewb=98.8 MV/mm
    Pc=28.0 TW, Ldeph=23.85 mm, Ldepl=23.85 mm, ΔE=2472.0 MeV over Lacc=23.85 mm
    for laser beam with w0=21.7 µm (FWHM=25.5 µm), zᵣ=1.85 mm, λL=0.80 µm, kL=7.854 1/µm, ωL=2.355 1/fs, ɛL=15.7 J, τL=48.2 fs, P₀=305.9 TW
    I₀=4.1e+19 W/cm**2, a₀=4.4, E₀=1.8e+04 MV/mm
    N=4.5e+09 electrons, Q=723.5 pC, η=0.114

For more info and a fully documented example, see `the usage docs <https://prepic.readthedocs.io/en/latest/usage.html>`_.


.. [LTJT] Lu, Wei, et al. Physical Review Special Topics-Accelerators and Beams 10.6 (2007): 061301.
