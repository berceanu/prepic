=====
Usage
=====


To use ``prepic`` in a project, we first import the necessary Python modules
and then declare the parameters of the laser system we want to \
model. For this example, let's consider the parameters from [CABC]_::

    >>> from collections import namedtuple  # optional, for grouping input parameters

    >>> import numpy as np
    >>> import unyt as u  # for physical units support

    >>> Flame = namedtuple(
    ...     "Flame",
    ...     [
    ...         "npe",  # electron plasma density
    ...         "w0",  # laser beam waist (Gaussian beam assumed)
    ...         "ɛL",  # laser energy on target (focused into the FWHM@intensity spot)
    ...         "τL",  # laser pulse duration (FWHM@intensity)
    ...         "prop_dist",  # laser propagation distance (acceleration length)
    ...     ],
    ... )
    >>> param = Flame(
    ...     npe=6.14e18 / u.cm ** 3,
    ...     w0=6.94 * u.micrometer,
    ...     ɛL=1.0 * u.joule,
    ...     τL=30 * u.femtosecond,
    ...     prop_dist=1.18 * u.mm,
    ... )

Notice the use of physical quantities with associated units throughout.
See `unyt <https://github.com/yt-project/unyt>`_ for further info.

We start by constructing a :py:class:`GaussianBeam <prepic.laser.GaussianBeam>` from the \
given beam waist ``w0``. Note the the beam can also be constructed by giving its FWHM \
size instead. Alternatively, one can use :py:meth:`GaussianBeam.from_f_number <prepic.laser.GaussianBeam.from_f_number>` or \
:py:meth:`GaussianBeam.from_focal_distance <prepic.laser.GaussianBeam.from_focal_distance>`::

    >>> from prepic import GaussianBeam
    >>> flame_beam = GaussianBeam(w0=param.w0)

We see ``prepic`` computed the FWHM spot size and Rayleigh length of our beam::

    >>> print(flame_beam)
    beam with w0=6.9 µm (FWHM=8.2 µm), zᵣ=0.19 mm, λL=0.80 µm

We now initialize a :py:class:`Laser <prepic.laser.Laser>` \
instance using its default constructor. We could have used \
:py:meth:`Laser.from_a0 <prepic.laser.Laser.from_a0>`, :py:meth:`Laser.from_intensity <prepic.laser.Laser.from_intensity>` \
or :py:meth:`Laser.from_power <prepic.laser.Laser.from_power>`, depending on which laser parameters are \
known::

    >>> from prepic import Laser
    >>> flame_laser = Laser(ɛL=param.ɛL, τL=param.τL, beam=flame_beam)
    >>> print(flame_laser)
    laser with kL=7.854 1/µm, ωL=2.355 1/fs, ɛL=1.0 J, τL=30.0 fs, P₀=31.3 TW
    I₀=4.1e+19 W/cm**2, a₀=4.4, E₀=1.8e+04 MV/mm
    Helium ionization state: 2+

The various attributes, such as the critical density ``ncrit`` or peak laser \
electric field ``E0`` can be easily accessed::

    >>> print(f"critical density for this laser is {flame_laser.ncrit:.1e}")
    critical density for this laser is 1.7e+21 cm**(-3)
    >>> print(flame_laser.E0)  # doctest: +FLOAT_CMP
    17659.733275104507 MV/mm

Also notice that ``flame_laser`` contains the ``flame_beam`` instance \
from before. For example, we can access its Rayleigh length via::

    >>> print(flame_laser.beam.zR)  # doctest: +FLOAT_CMP
    0.18913801491304671 mm

We now build the :py:class:`Plasma <prepic.plasma.Plasma>` for our parameters via::

    >>> from prepic import Plasma
    >>> flame_plasma = Plasma(
    ...     n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
    ... )
    >>> print(flame_plasma)
    Plasma with nₚ=6.1e+18 cm**(-3) (3.52e-03 × nc), ωₚ=0.140 1/fs, kₚ=0.466 1/µm, λₚ=13.5 µm, Ewb=238.3 MV/mm
    Pc=4.8 TW, Ldeph=1.70 mm, Ldepl=2.55 mm, ΔE=294.9 MeV over Lacc=1.18 mm

This is the top-level class, which contains all the computed parameters. If, as \
before, we would like to access the Rayleigh length, we can do so via::

    >>> print(flame_plasma.laser.beam.zR)  # doctest: +FLOAT_CMP
    0.18913801491304671 mm

All the computed parameters are stored as attributes.
See :py:class:`Plasma <prepic.plasma.Plasma>` for their description::

    >>> print(f"\nThe dephasing length is {flame_plasma.dephasing:.1f}.")
    The dephasing length is 1.7 mm.

If ``propagation_distance`` is passed, this is used to evaluate the electron \
energy gain ``ΔE``. If not given, the code assumes that the electrons are accelerated for \
a distance equal to the dephasing length.

The ``Plasma`` can also be constructed by passing the \
(optional) ``bubble_radius``, if known from experiments or numerical \
simulations. For now, we can estimate the bubble size from the scaling laws of \
[LTJT]_: :math:`R = 2 \sqrt{a_0} / k_p`. This allows computing the total accelerated \
charge ``Q`` and laser-to-electron energy transfer efficiency ``η``::

    >>> bubble_r = 2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
    >>> print(f"The bubble radius is {bubble_r.to('micrometer'):.1f}.\n")
    The bubble radius is 9.0 µm.
    >>> plasma_with_bubble = Plasma(
    ...     n_pe=param.npe,
    ...     laser=flame_laser,
    ...     bubble_radius=bubble_r,
    ...     propagation_distance=param.prop_dist,
    ... )
    >>> print(plasma_with_bubble.Q)  # doctest: +FLOAT_CMP
    300.1260542601371 pC
    >>> print(plasma_with_bubble.η.to_value('dimensionless'))  # doctest: +FLOAT_CMP
    0.08850500992541349

The ``Plasma`` parameters can also be automagically computed by \
:py:func:`matched_laser_plasma <prepic.lwfa.matched_laser_plasma>`, based on the scaling laws of \
[LTJT]_. The only input parameter in this case is the laser normalized \
vector potential :math:`a_0`::

    >>> from prepic import matched_laser_plasma
    >>> matched_plasma_flame = matched_laser_plasma(a0=flame_laser.a0)
    >>> print(matched_plasma_flame)  # notice density, spot size, etc. changed!
    Plasma with nₚ=1.1e+18 cm**(-3) (6.06e-04 × nc), ωₚ=0.058 1/fs, kₚ=0.193 1/µm, λₚ=32.5 µm, Ewb=98.8 MV/mm
    Pc=28.0 TW, Ldeph=23.86 mm, Ldepl=23.86 mm, ΔE=2472.7 MeV over Lacc=23.86 mm
    N=4.5e+09 electrons, Q=723.7 pC, η=0.114
    >>> print(matched_plasma_flame.Q)  # doctest: +FLOAT_CMP
    723.6933108198331 pC

We see that now the total accelerated charge, final energy, as well as \
efficiency are all improved compared to their previous values. The acceleration \
distance is now longer, and equal to the dephasing and depletion lengths. This \
is possible due to the better matching between laser and plasma parameters.

Finally, the ``prepic`` package also includes a \
:py:class:`Simulation <prepic.simulation.Simulation>` convenience class for estimating the \
recommended parameters for a PIC simulation, based on a particular \
``Plasma``::

    >>> from prepic import Simulation
    >>> sim_flame = Simulation(matched_plasma_flame)
    >>> print(sim_flame)
    3D simulation with box size (130.0 µm)³, Δx=0.517 µm, Δy=0.517 µm, Δz=0.040 µm, nx=251, ny=251, nz=3249, 1.637522e+09 macro-particles, 5.997110e+05 time steps

These values can then be used as inputs for specialized codes such as `PIConGPU`_ or `fbpic`_.


.. [CABC] Curcio, A., et al. Physical Review Accelerators and Beams 20.1 (2017): 012801.
.. _PIConGPU: https://github.com/ComputationalRadiationPhysics/picongpu
.. _fbpic: https://github.com/fbpic/fbpic
