=====
Usage
=====

.. jupyter-kernel::
  :id: prepic

To use ``prepic`` in a project, we first import the necessary modules

.. jupyter-execute::

    from collections import namedtuple
    import unyt as u
    import numpy as np

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
    print(flame_beam)

Having the beam, we go on and construct a :py:class:`prepic.lwfa.Laser` \
instance using the default constructor. This can also be created from the \
normalized vector potential :math:`a_0`, from the intensity or power.

.. jupyter-execute::

    flame_laser = lwfa.Laser(
        ɛL=param.ɛL, τL=param.τL, beam=flame_beam
    )
    print(flame_laser)

We now build the :py:class:`prepic.lwfa.Plasma` for our parameters via

.. jupyter-execute::

    flame_plasma = lwfa.Plasma(
        n_pe=param.npe, laser=flame_laser, propagation_distance=param.prop_dist
    )
    print(flame_plasma)

For the nonlinear LWFA regime, if one knows the bubble radius from simulations \
or experiment, that can be passed to the ``Plasma``. In this case, we can estimate \
the bubble size from the scaling laws of [LTJT]_

.. jupyter-execute::

    bubble_r = (
        2 * np.sqrt(flame_plasma.laser.a0) / flame_plasma.kp
    ).to("micrometer")
    print(f"bubble radius is {bubble_r:.1f}")

    new_plasma = lwfa.Plasma(
        n_pe=param.npe,
        laser=flame_laser,
        bubble_radius=bubble_r,
        propagation_distance=param.prop_dist,
    )
    print(new_plasma)

Finally, we can internally make use of the matching conditions through \
:py:func:`prepic.lwfa.matched_laser_plasma`, which takes the calculated \
:math:`a_0` as input

.. jupyter-execute::

    matched_flame = lwfa.matched_laser_plasma(a0=flame_plasma.laser.a0)
    print(matched_flame)

The :py:mod:`prepic.lwfa` module also includes the \
:py:class:`prepic.lwfa.Simulation` class for suggesting the \
appropriate parameters to chose in a PIC simulation, \
based on the laser-plasma characteristics:

.. jupyter-execute::

    sim_flame = lwfa.Simulation(flame_plasma)
    print(sim_flame)


You can download all the code from this document as a Python script :jupyter-download:script:`prepic` \
or as a Jupyter notebook :jupyter-download:notebook:`prepic`.


.. [Gibb] Gibbon, "Short pulse laser interactions with matter", p. 22.
.. [CABC] Curcio, A., et al. Physical Review Accelerators and Beams 20.1 (2017): 012801.

