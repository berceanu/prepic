"""
Classes for modelling PIC simulation parameters

"""
from prepic.base import BaseClass
import unyt as u


class Simulation(BaseClass):
    """Class for estimating the recommended simulation parameters.

    Attributes
    ----------

    Δx : float, length
        Transverse spatial resolution.
    Δy : float, length
        Transverse spatial resolution.
    Δz : float, length
        Longitudinal spatial resolution.
    nx : int, dimensionless
        Transverse number of cells.
    ny : int, dimensionless
        Transverse number of cells.
    nz : int, dimensionless
        Longitudinal number of cells.
    L : float, length
        Length of cubic simulation box.
    ppc : int, dimensionless
        Number of particles per cell.
    npart : int, dimensionless
        Total number of (macro-)particles in the simulation box.
    dt : float, time
        Simulation time step per iteration.
    t_interact : float, time
        Time it takes for the moving window to slide across the plasma.
    nstep : int, dimensionless
        Number of iterations to perform.

    Notes
    -----
    Here longitudinal means along the laser propagation direction.
    Recommended number of particles per cell: 64 (1D), 10 (2D), 8 (3D).

    Examples
    --------
    >>> import unyt as u
    >>> from prepic import Plasma, Laser, GaussianBeam
    >>> mylaser = Laser.from_power(power=10 * u.petawatt, ɛL=300 * u.joule,
    ...                            beam=GaussianBeam(w0=5 * u.micrometer))
    >>> myplasma = Plasma(n_pe=1e18 / u.cm**3, laser=mylaser)
    >>> Simulation(myplasma)
    <Simulation(<Plasma(1e+18 cm**(-3), <Laser(300.0 J, 28.18311836098954 fs, <GaussianBeam(5.0 µm, 0.8 µm)>)>, None)>,\
 133.5577261430166 µm, 8 dimensionless)>
    """

    def __init__(self, plasma, box_length=None, ppc=None):
        """Estimate recommended simulation params for given plasma (and laser).

        Parameters
        ----------
        plasma : :obj:`Plasma`
            Instance containing laser and plasma params.
        box_length : float, length, optional
            Length of the cubic simulation box (defaults to 4λₚ).
        ppc : int, dimensionless, optional
            Number of particles per cell (defaults to 8 in 3D).
        """
        if not plasma.laser:
            raise TypeError("Given `Plasma` instance must contain `Laser` instance.")
        else:
            self.plasma = plasma
        if not box_length:
            self.L = 4 * self.plasma.λp
        else:
            self.L = box_length.to("micrometer")
        if not ppc:
            self.ppc = u.unyt_quantity(8, "dimensionless", dtype="int")
        else:
            self.ppc = ppc

        self.Δx = self.plasma.lp / 10
        self.Δy = self.Δx
        self.Δz = self.plasma.laser.beam.λL / 20

        self.nx = u.unyt_quantity(
            (self.L / self.Δx).to_value("dimensionless"), "dimensionless", dtype="int"
        )
        self.ny = self.nx
        self.nz = u.unyt_quantity(
            (self.L / self.Δz).to_value("dimensionless"), "dimensionless", dtype="int"
        )

        self.npart = self.nx * self.ny * self.nz * self.ppc

        self.dt = (self.Δz / u.clight).to("femtosecond")
        self.t_interact = ((self.plasma.Lacc + self.L) / u.clight).to("femtosecond")

        self.nstep = u.unyt_quantity(
            (self.t_interact / self.dt).to_value("dimensionless"),
            "dimensionless",
            dtype="int",
        )

    def __eq__(self, other):
        return super().__eq__(other)

    def __repr__(self):
        return f"<{self.__class__.__name__}({repr(self.plasma)}, {self.L}, {self.ppc})>"

    def __str__(self):
        msg = (
            f"3D simulation with box size ({self.L:.1f})³, Δx={self.Δx:.3f}, Δy={self.Δy:.3f}, "
            f"Δz={self.Δz:.3f}, nx={int(self.nx.to_value('dimensionless'))}, "
            f"ny={int(self.ny.to_value('dimensionless'))}, nz={int(self.nz.to_value('dimensionless'))}, "
            f"{int(self.npart.to_value('dimensionless')):e} macro-particles, "
            f"{int(self.nstep.to_value('dimensionless')):e} time steps"
        )
        return msg
