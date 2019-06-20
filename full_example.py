import numpy as np
import unyt as u

from prepic.lwfa import GaussianBeam, Laser, Plasma, matched_laser_plasma, Simulation


def main():
    GaussianBeam.from_f_number(f_number=10.0)
    GaussianBeam.from_focal_distance(
        focal_distance=1 * u.meter, beam_diameter=10 * u.cm
    )

    # CETAL params
    Laser(
        ɛL=7.7 * u.joule, τL=40 * u.femtosecond, beam=GaussianBeam(w0=18 * u.micrometer)
    )

    Laser.from_a0(a0=4.076967454355432, ɛL=7.7 * u.joule, τL=40 * u.femtosecond)

    Laser.from_a0(
        a0=4.076967454355432, ɛL=7.7 * u.joule, beam=GaussianBeam(w0=18 * u.micrometer)
    )

    Laser.from_intensity(
        intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
        ɛL=7.7 * u.joule,
        τL=40 * u.femtosecond,
    )
    Laser.from_intensity(
        intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
        ɛL=7.7 * u.joule,
        beam=GaussianBeam(w0=18 * u.micrometer),
    )
    Laser.from_intensity(
        intensity=3.553314404474785e19 * u.watt / u.cm ** 2,
        τL=40 * u.femtosecond,
        beam=GaussianBeam(w0=18 * u.micrometer),
    )
    Laser.from_power(
        power=180.84167614968285 * u.terawatt,
        beam=GaussianBeam(w0=18 * u.micrometer),
        τL=40 * u.femtosecond,
    )
    laser = Laser.from_power(
        power=180.84167614968285 * u.terawatt,
        beam=GaussianBeam(w0=18 * u.micrometer),
        ɛL=7.7 * u.joule,
    )
    print(laser)

    # CETAL laser
    npe_cetal = 1.5e18 / u.cm ** 3
    beam_cetal = GaussianBeam(w0=18 * u.micrometer)
    laser_cetal = Laser(ɛL=7.7 * u.joule, τL=40 * u.femtosecond, beam=beam_cetal)
    Plasma(n_pe=npe_cetal)
    plasma_cetal = Plasma(n_pe=npe_cetal, laser=laser_cetal)
    bubble_R_cetal = (2 * np.sqrt(plasma_cetal.laser.a0) / plasma_cetal.kp).to(
        "micrometer"
    )
    plasma_cetal = Plasma(
        n_pe=npe_cetal, laser=laser_cetal, bubble_radius=bubble_R_cetal
    )
    print(plasma_cetal)

    # Gamma blaster laser
    npe_ong = 6.125e18 / u.cm ** 3
    beam_ong = GaussianBeam(w0=10 * u.micrometer)
    laser_ong = Laser.from_power(
        power=1570.796 * u.terawatt, beam=beam_ong, τL=33 * u.femtosecond
    )
    plasma_ong = Plasma(
        n_pe=npe_ong, laser=laser_ong, propagation_distance=500 * u.micrometer
    )
    print(plasma_ong)

    matched_cetal = matched_laser_plasma(a0=4.1)
    print(matched_cetal)

    # Single-shot non-intercepting profile monitor of plasma-accelerated electron beams with nanometric resolution
    # Appl. Phys. Lett. 111, 133105 (2017)
    print("[...] profile monitor [...]\n")
    npe_monitor = 2e19 / u.cm ** 3
    beam_monitor = GaussianBeam(w0=10 * u.micrometer)
    laser_monitor = Laser(ɛL=1.0 * u.joule, τL=30 * u.femtosecond, beam=beam_monitor)
    plasma_monitor = Plasma(
        n_pe=npe_monitor, laser=laser_monitor, propagation_distance=1 * u.mm
    )

    print()
    print("Parameters for beam monitor paper:")
    print(plasma_monitor)

    matched_monitor = matched_laser_plasma(a0=3.1)
    print()
    print("Matching conditions for beam monitor paper:")
    print(matched_monitor)

    # Trace-space reconstruction of low-emittance electron beams through betatron radiation in laser-plasma accelerators
    # Phys. Rev. ST. Accel. Beams 20, 012801 (2017)
    print("Trace-space reconstruction [...]\n")
    npe_emittance = 6.14e18 / u.cm ** 3
    beam_emittance = GaussianBeam(w0=6.94 * u.micrometer)
    laser_emittance = Laser(
        ɛL=1.0 * u.joule, τL=30 * u.femtosecond, beam=beam_emittance
    )
    plasma_emittance = Plasma(
        n_pe=npe_emittance, laser=laser_emittance, propagation_distance=1.18 * u.mm
    )
    bubble_R_emittance = (
        2 * np.sqrt(plasma_emittance.laser.a0) / plasma_emittance.kp
    ).to(
        "micrometer"
    )  # 9*u.micrometer
    print()
    print(f"bubble radius R={bubble_R_emittance}")
    plasma_emittance = Plasma(
        n_pe=npe_emittance,
        laser=laser_emittance,
        bubble_radius=bubble_R_emittance,
        propagation_distance=1.18 * u.mm,
    )

    print()
    print("Parameters for emmitance paper:")
    print(plasma_emittance)

    matched_emittance = matched_laser_plasma(a0=4.4)
    print()
    print("Matching conditions for emmitance paper:")
    print(matched_emittance)

    # 1. gas-jet with $L_{\text{acc}} = 3$ mm, $n_{\text{pe}} = (1-3) \times 10^{18}$ cm${}^{-3}$
    # 2. capillary with $L_{\text{acc}} = (3-10)$ cm, $n_{\text{pe}} = (3-7) \times 10^{17}$ cm${}^{-3}$

    # - $w_0 = 2 \sigma_{\text{rms}}$, and experiments say $\sigma_{\text{rms}} = 7$ $\mu$m, so $w_0 = 14$ $\mu$m
    # - $\varepsilon_L = 3$ J, $\tau_L = 30$ fs, $I_0 = 3 \times 10^{19}$ W/cm${}^{2}$, $a_0=3.4$

    print()
    print("Frascati future exp [...]\n")

    npe_jet = 3e18 / u.cm ** 3
    l_acc_jet = 3 * u.mm

    npe_capil = 3e17 / u.cm ** 3
    l_acc_capil = 5 * u.cm

    beam_frasc = GaussianBeam(w0=15.56 * u.micrometer)
    laser_frasc = Laser(ɛL=3.0 * u.joule, τL=30 * u.femtosecond, beam=beam_frasc)

    plasma_jet = Plasma(n_pe=npe_jet, laser=laser_frasc, propagation_distance=l_acc_jet)
    plasma_capil = Plasma(
        n_pe=npe_capil, laser=laser_frasc, propagation_distance=l_acc_capil
    )

    print()
    print("Parameters for He gas jet:")
    print(plasma_jet)
    print(Simulation(plasma_jet))

    print()
    print("Parameters for H capillary:")
    print(plasma_capil)
    print(Simulation(plasma_capil))

    matched_frasc = matched_laser_plasma(a0=3.4)
    print()
    print("Matching conditions:")
    print(matched_frasc)


if __name__ == '__main__':
    main()
