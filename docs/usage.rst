=====
Usage
=====

.. jupyter-kernel::
  :id: prepic_usage

To use ``prepic`` in a project, we first import the necessary Python modules

.. jupyter-execute::

    import pprint  # optional, pretty-print dictionaries
    from collections import namedtuple  # optional, for grouping input parameters

    import numpy as np
    import unyt as u  # for physical units support

    from prepic import lwfa

and then declare the parameters of the laser system we want to \
model. For this example, let's consider the parameters from [CABC]_:

.. jupyter-execute::

    Flame = namedtuple(
        "Flame",
        [
            "npe",  # electron plasma density
            "w0",  # laser beam waist (Gaussian beam assumed)
            "ɛL",  # laser energy on target (focused into the FWHM@intensity spot)
            "τL",  # laser pulse duration (FWHM@intensity)
            "prop_dist",  # laser propagation distance (acceleration length)
        ],
    )
    param = Flame(
        npe=6.14e18 / u.cm ** 3,
        w0=6.94 * u.micrometer,
        ɛL=1.0 * u.joule,
        τL=30 * u.femtosecond,
        prop_dist=1.18 * u.mm,
    )

Notice the use of physical quantities with associated units throughout. In
particular, see :py:class:`unyt.array.unyt_quantity`.

We start by constructing a :py:class:`prepic.lwfa.GaussianBeam` from the \
given beam waist ``w0``. Note the the beam can also be constructed by giving its FWHM \
size instead. Alternatively, one can use :py:meth:`prepic.lwfa.GaussianBeam.from_f_number` or \
:py:meth:`prepic.lwfa.GaussianBeam.from_focal_distance`.

.. jupyter-execute::

    flame_beam = lwfa.GaussianBeam(w0=param.w0)

We see ``prepic`` computed the FWHM spot size and Rayleigh length of our beam:

.. jupyter-execute::

    print(flame_beam)

We now initialize a :py:class:`prepic.lwfa.Laser` \
instance using its default constructor. We could have used \
:py:meth:`prepic.lwfa.Laser.from_a0`, :py:meth:`prepic.lwfa.Laser.from_intensity` \
or :py:meth:`prepic.lwfa.Laser.from_power`, depending on which laser parameters are \
known.

.. jupyter-execute::

    flame_laser = lwfa.Laser(ɛL=param.ɛL, τL=param.τL, beam=flame_beam)
    print(flame_laser)

The various attributes, such as the critical density ``ncrit`` or peak laser \
electric field ``E0`` can be easily accessed

.. jupyter-execute::

    print(f"critical density for this laser is {flame_laser.ncrit:.1e}")
    flame_laser.E0  # each attribute is a ``unyt_quantity``

Also notice that ``flame_laser`` contains the ``flame_beam`` instance \
from before. For example, we can access its Rayleigh length via

.. jupyter-execute::

    print(flame_laser.beam.zR)

We now build the :py:class:`prepic.lwfa.Plasma` for our parameters via

.. jupyter-execute::

    flame_plasma = lwfa.Plasma(
        n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
    )
    print(flame_plasma)

This is the top-level class, which contains all the computed parameters. If, as \
before, we would like to access the Rayleigh length, we can do so via

.. jupyter-execute::

    flame_plasma.laser.beam.zR

All the computed parameters are stored as attributes. Here is the complete list \
(see :py:class:`prepic.lwfa.Plasma` for their description).

.. jupyter-execute::

    pprint.pprint(flame_plasma.__dict__)
    print(f"\nThe dephasing length is {flame_plasma.dephasing:.1f}.")

If ``propagation_distance`` is passed, this is used to evaluate the electron \
energy gain ``ΔE``. If not given, the code assumes that the electrons are accelerated for \
a distance equal to the dephasing length.

The :py:class:`prepic.lwfa.Plasma` can also be constructed by passing the \
(optional) ``bubble_radius``, if known from experiments or numerical \
simulations. For now, we can estimate the bubble size from the scaling laws of \
[LTJT]_: :math:`R = 2 \sqrt{a_0} / k_p`. This allows computing the total accelerated \
charge ``Q`` and laser-to-electron energy transfer efficiency ``η``:

.. jupyter-execute::

    bubble_r = 2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
    print(f"The bubble radius is {bubble_r.to('micrometer'):.1f}.\n")

    plasma_with_bubble = lwfa.Plasma(
        n_pe=param.npe,
        laser=flame_laser,
        bubble_radius=bubble_r,
        propagation_distance=param.prop_dist,
    )
    print(plasma_with_bubble.Q)
    print(plasma_with_bubble.η.to_value('dimensionless'))

The ``Plasma`` parameters can also be automagically computed by \
:py:func:`prepic.lwfa.matched_laser_plasma`, based on the scaling laws of \
[LTJT]_. The only input parameter in this case is the laser normalized \
vector potential :math:`a_0`.

.. jupyter-execute::

    matched_plasma_flame = lwfa.matched_laser_plasma(a0=flame_laser.a0)
    print(matched_plasma_flame)  # notice density, spot size, etc. changed!
    print()
    print(matched_plasma_flame.Q)  # compare to previous value
    print(matched_plasma_flame.ΔE)  # also much improved :)

We see that now the total accelerated charge, final energy, as well as \
efficiency are all improved compared to their previous values. The acceleration \
distance is now longer, and equal to the dephasing and depletion lengths. This \
is possible due to the better matching between laser and plasma parameters.

Finally, the :py:mod:`prepic.lwfa` module also includes a \
:py:class:`prepic.lwfa.Simulation` convenience class for estimating the \
recommended parameters for a PIC simulation, based on a particular \
``Plasma``.

.. jupyter-execute::

    sim_flame = lwfa.Simulation(matched_plasma_flame)
    print(sim_flame)

These values can then be used as inputs for specialized codes such as `PIConGPU`_ or `fbpic`_.


Download this document as a Python script :jupyter-download:script:`prepic_usage` \
or as a Jupyter notebook :jupyter-download:notebook:`prepic_usage`.


.. [CABC] Curcio, A., et al. Physical Review Accelerators and Beams 20.1 (2017): 012801.
.. _PIConGPU: https://github.com/ComputationalRadiationPhysics/picongpu
.. _fbpic: https://github.com/fbpic/fbpic
