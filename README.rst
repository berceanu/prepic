.. image:: ./prepic_logo.png
   :alt: pre-PIC Logo
   :width: 400

=======
pre-PIC
=======

Analytically estimate various laser-plasma parameters for experiments and PIC simulations


Features
--------

* estimation of accelerated bunch properties (charge, energy, etc.)
* estimation of betatron spectrum
* small codebase, with minimal dependencies
* support for SI units via `unyt <https://github.com/yt-project/unyt>`_


Quick start
-----------

Install the package via:

.. code-block:: console

        $ pip install prepic

Estimate ideal laser-plasma matching parameters based on scaling laws from [LTJT]_::

    >>> import unyt as u

    >>> from prepic import matched_laser_plasma

    >>> laser_plasma = matched_laser_plasma(a0=4.4 * u.dimensionless)
    >>> print(laser_plasma)
    Plasma with nₚ=1.1e+18 cm**(-3) (6.06e-04 × nc), ωₚ=0.058 1/fs, kₚ=0.193 1/µm, λₚ=32.5 µm, Ewb=98.8 MV/mm
    Pc=28.0 TW, Ldeph=23.85 mm, Ldepl=23.85 mm, ΔE=2472.0 MeV over Lacc=23.85 mm
    N=4.5e+09 electrons, Q=723.5 pC, η=0.114


.. [LTJT] Lu, Wei, et al. Physical Review Special Topics-Accelerators and Beams 10.6 (2007): 061301.
