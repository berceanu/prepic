import numpy as np
import unyt as u

from prepic.lwfa import GaussianBeam, Laser, Plasma, matched_laser_plasma, Simulation, w0_to_fwhm


def main():
    print(w0_to_fwhm(18 * u.micrometer))
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 1. gas-jet with $L_{\text{acc}} = 3$ mm, $n_{\text{pe}} = (1-3) \times 10^{18}$ cm${}^{-3}$
    # 2. capillary with $L_{\text{acc}} = (3-10)$ cm, $n_{\text{pe}} = (3-7) \times 10^{17}$ cm${}^{-3}$

    # - $w_0 = 2 \sigma_{\text{rms}}$, and experiments say $\sigma_{\text{rms}} = 7$ $\mu$m, so $w_0 = 14$ $\mu$m
    # - $\varepsilon_L = 3$ J, $\tau_L = 30$ fs, $I_0 = 3 \times 10^{19}$ W/cm${}^{2}$, $a_0=3.4$

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

    Simulation(plasma_jet)
    Simulation(plasma_capil)

    matched_frasc = matched_laser_plasma(a0=3.4)

    # todo remove msg "Scaling laws valid up to" and replace with catch/raise

    # We will be using this as a worked-out example for the documentation: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Trace-space reconstruction of low-emittance electron beams through betatron radiation in laser-plasma accelerators
    # Phys. Rev. ST. Accel. Beams 20, 012801 (2017)

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


if __name__ == "__main__":
    main()
