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
    from matplotlib import pyplot
    %matplotlib inline

    from prepic import lwfa

and then declare the parameters of the laser system we are interested in modelling

.. jupyter-execute::

    Cetal = namedtuple(
        "Cetal",
        [
            "npe",
            "kp",
            "f_number",
            "focal_distance",
            "beam_diameter",
            "w0",
            "fwhm",
            "zR",
            "a0",
            "ɛL",
            "τL",
            "intensity",
            "power",
        ],
    )
    param = Cetal(
        npe=1.5e18 / u.cm ** 3,
        kp=0.2304711 * 1 / u.micrometer,
        f_number=24.9912 * u.dimensionless,
        focal_distance=3.2 * u.meter,
        beam_diameter=128.045071 * u.mm,
        w0=18 * u.micrometer,
        fwhm=21.193380405278543 * u.micrometer,
        zR=1.27234502 * u.mm,
        a0=4.076967454355432 * u.dimensionless,
            ɛL=7.7 * u.joule,
        τL=40 * u.femtosecond,
        intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
        power=180.84167614968285 * u.terawatt,
    )

We now build the :py:class:`prepic.lwfa.Plasma` class for our parameters

.. jupyter-execute::

    def get_plasma(parameters):
        bubble_r = (2 * np.sqrt(parameters.a0) / parameters.kp)
    
        return lwfa.Plasma(
            n_pe=parameters.npe,
            laser=lwfa.Laser.from_a0(
                a0=parameters.a0, ɛL=parameters.ɛL, beam=lwfa.GaussianBeam(w0=parameters.w0)
            ),
            bubble_radius=bubble_r
        )
    cetal_plasma = get_plasma(param)
    cetal_plasma

We measure charges in units of :math:`e`, masses in units of :math:`m_e`, lengths in \
units of :math:`l_p = c/\omega_p`, times in units of :math:`t_p = 1 /\omega_p`. Momenta \
will then be measured in :math:`m_e c`, velocities in units of :math:`c`, electric fields \
in units of :math:`m_e c \omega_p / e`, magnetic fields in units of :math:`m_e \omega_p / e`.\
Scalar potential :math:`\Phi` in units of :math:`m_e c^2 / e` and vector potential \
:math:`\vec{A}` in units of :math:`m_e c /e`. We will use SI units throughout.

For :math:`a_0 \geq 4-5` we also get self-injection from pure Helium. Helium has the ionization \
energies 24.59 eV :math:`\text{He}^{+}` and 54.42 :math:`\text{He}^{2+}`, corresponding to laser intensities \
:math:`1.4 \times 10^{15}`, respectively :math:`8.8 \times 10^{15}\, \text{W/cm}^{2}` [Gibb]_ , \
and will therefore be easily ionized by the laser prepulse.

The atomic Coulomb field is on the order of :math:`10^{14}\, \text{W/cm}^{2}` and relativistic effects \
become important for laser intensities above :math:`10^{17}\, \text{W/cm}^{2}` (for :math:`a_0 \geq 1`), while \
QED effects such as radiation reaction only become important for intensities beyond \
:math:`\sim 2 \times 10^{21}\, \text{W/cm}^{2}`.

For LWFA, we roughly have :math:`w_0 \approx c \tau_L` and :math:`\tau_L \approx \omega_p^{-1}`.

If we assume the laser energy before the compressor is 20 J, and 30% is lost in the \
compressor and beam transport, we are left with 14 J in the chamber. If 50% of this energy \
can be focused into the FWHM spot of :math:`21 \mu m`, we get 7 J on target.

Need to resolve the smallest length scale: 20-30 cells/wavelength.

- plasma length scale: skin depth :math:`c/\omega_p`
- laser length scale: laser wavelength :math:`\lambda_L = 0.8\, \mu m`


.. [Gibb] Gibbon, "Short pulse laser interactions with matter", p. 22.
