"""
Classes for modelling PIC simulation parameters

"""
from prepic._base_class import BaseClass
import unyt as u


class Simulation(BaseClass):
    """Class for estimating the recommended simulation parameters.
    Attributes:
        Δx (float, length): transverse spatial resolution
        Δy (float, length): transverse spatial resolution
        Δz (float, length): longitudinal spatial resolution
        nx (int, dimensionless): transverse number of cells
        ny (int, dimensionless): transverse number of cells
        nz (int, dimensionless): longitudinal number of cells
        L (float, length): length of cubic simulation box
        ppc (int, dimensionless): number of particles per cell
        npart (int, dimensionless): total number of (macro-)particles in the
            simulation box
        dt (float, time): simulation time step per iteration
        t_interact (float, time): time it takes for the moving window to slide
            across the plasma
        nstep (int, dimensionless): number of iterations to perform
    Note:
        Here longitudinal means along the laser propagation direction.
        Recommended number of particles per cell: 64 (1D), 10 (2D), 8 (3D).
    """

    def __init__(self, plasma, box_length=None, ppc=None):
        """Estimate recommended simulation params for given plasma (and laser).
        Args:
            plasma (:obj:`Plasma`): instance containing laser and plasma params
            box_length (float, length, optional): length of the cubic
                simulation box. Defaults to 4λₚ.
            ppc (int, dimensionless, optional): number of particles per cell.
                Defaults to 8 (3D).
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
