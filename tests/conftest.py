import pytest
from collections import namedtuple
import unyt as u
import numpy as np
from prepic import GaussianBeam, Laser, Plasma


@pytest.fixture(scope="module")
def cet_param():
    Params = namedtuple(
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
    cetal = Params(
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
    return cetal


@pytest.fixture(scope="module")
def cet_plasma(cet_param):
    bubble_r = 2 * np.sqrt(cet_param.a0) / cet_param.kp

    return Plasma(
        n_pe=cet_param.npe,
        laser=Laser.from_a0(
            a0=cet_param.a0, ɛL=cet_param.ɛL, beam=GaussianBeam(w0=cet_param.w0)
        ),
        bubble_radius=bubble_r,
    )
