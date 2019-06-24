=====
Usage
=====

.. jupyter-kernel::
  :id: prepic_usage

To use ``prepic`` in a project, we first import the necessary Python modules

.. jupyter-execute::

    from collections import namedtuple
    import unyt as u
    import numpy as np
    import pprint
    from prepic import lwfa

and then declare the parameters of the laser system we are interested in \
modelling. For this example, let's consider the parameters from [CABC]_

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

We start by constructing the :py:class:`prepic.lwfa.GaussianBeam` using the \
given beam waist. Note the the beam can also be constructed by giving its FWHM \
size instead, or using :py:meth:`prepic.lwfa.GaussianBeam.from_f_number` or \
:py:meth:`prepic.lwfa.GaussianBeam.from_focal_distance`.

.. jupyter-execute::

    flame_beam = lwfa.GaussianBeam(w0=param.w0)

We see ``prepic`` gives us the FWHM and Rayleigh length of the constructed beam:

.. jupyter-execute::

    print(flame_beam)

Having the beam, we go on and construct a :py:class:`prepic.lwfa.Laser` \
instance using its default constructor. Alternatively, we could also have used \
:py:meth:`prepic.lwfa.Laser.from_a0`, :py:meth:`prepic.lwfa.Laser.from_intensity` \
or :py:meth:`prepic.lwfa.Laser.from_power`, depending on which laser parameters are \
known.

.. jupyter-execute::

    flame_laser = lwfa.Laser(
        ɛL=param.ɛL, τL=param.τL, beam=flame_beam
    )
    print(flame_laser)

The various attributes, such as the critical density ``ncrit`` or peak laser \
electric field ``E0`` can be easily accessed

.. jupyter-execute::

    print(f"critical density for this laser is {flame_laser.ncrit:.1e}")
    flame_laser.E0  # each attribute is a `unyt_quantity` instance

Also notice that ``flame_laser`` contains the ``flame_beam`` instance \
from before. For example, we can access the Rayleigh length via

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

All the computed parameters are stored as attributes. Here is the complete list:

.. jupyter-execute::

    pprint.pprint(flame_plasma.__dict__)  #  see the docs for a description of all attributes
    print(f"\nThe dephasing length is {flame_plasma.dephasing:.1f}.")

If ``propagation_distance`` is passed, this is used to evaluate the electron \
energy gain ``ΔE``. If not, it is assumed that the electrons are accelerated for \
a distance equal to the dephasing length.

The :py:class:`prepic.lwfa.Plasma` can also be constructed by passing the \
(optional) ``bubble_radius``, if known from experiments or numerical \
simulations. For now, we can estimate the bubble size from the scaling laws of \
[LTJT]_. This allows computing the total accelerated charge ``Q`` and \
laser-to-electron energy transfer efficiency ``η``:

.. jupyter-execute::

    bubble_r = (
        2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
    )
    print(f"The bubble radius is {bubble_r.to('micrometer'):.1f}.\n")

    plasma_with_bubble = lwfa.Plasma(
        n_pe=param.npe,
        laser=flame_laser,
        bubble_radius=bubble_r,
        propagation_distance=param.prop_dist,
    )
    print(plasma_with_bubble.Q)
    print(plasma_with_bubble.η.to_value('dimensionless'))

The ``Plasma`` parameters can also be automatically computed using \
:py:func:`prepic.lwfa.matched_laser_plasma`, based on the scaling laws of \
[LTJT]_. The only input parameter in this case is the laser's normalized \
vector potential :math:`a_0`.

.. jupyter-execute::

    matched_plasma_flame = lwfa.matched_laser_plasma(a0=flame_laser.a0)
    print(matched_plasma_flame)
    print(matched_plasma_flame.ΔE)  # much improved :)

The :py:mod:`prepic.lwfa` module also includes the \
:py:class:`prepic.lwfa.Simulation` convenience class for estimating the \
appropriate parameters to chose in a PIC simulation, based on the laser-plasma \
characteristics:

.. jupyter-execute::

    sim_flame = lwfa.Simulation(matched_plasma_flame)
    print(sim_flame)


You can download all the code from this document as a Python script :jupyter-download:script:`prepic_usage` \
or as a Jupyter notebook :jupyter-download:notebook:`prepic_usage`.


.. [CABC] Curcio, A., et al. Physical Review Accelerators and Beams 20.1 (2017): 012801.

