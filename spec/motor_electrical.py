"""
First-principles motor / ESC / battery electrical models — CANDIDATE tier.

The verified tier's first-order speed lag (spec.motor) is the linearization of the
quasi-static DC-motor + quadratic-load model below; these candidates provide the physical
bridge (battery-voltage coupling, current draw, sag) that the identified τ_m hides.
"""

import sympy as sp


def dc_motor_speed_dynamics(W: sp.Expr, V_m: sp.Expr, J_r: sp.Expr, K_q: sp.Expr,
                            K_e: sp.Expr, R_a: sp.Expr, k_m: sp.Expr,
                            b: sp.Expr) -> sp.Expr:
    """
    Quasi-static DC motor (inductance neglected: τ_e = L_a/R_a ≈ 1.4 ms ≪ mechanical) with
    quadratic aerodynamic load:

        J_r·Ω̇ = (K_q/R_a)·(V_m − K_e·Ω) − k_m·Ω² − b·Ω,     i = (V_m − K_e·Ω)/R_a

    Linearized about Ω₀ this is the verified first-order lag with
        τ_m = J_r / (K_q·K_e/R_a + b + 2·k_m·Ω₀)                    (parameter bridge).
    Measured (Bangura ICRA 2014 Table I, mid-size BLDC): R_a = 0.07 Ω, L_a = 0.1 mH,
    K_v = 950 RPM/V (K_e = 0.01005 V·s/rad), K_q0 = 0.0242 N·m/A, J_r = 5.38e-5 kg·m².
    Source: Bangura, Lim, Kim, Mahony, ICRA 2014, Eqs. (11)–(15); Bangura & Mahony ACRA 2012
    Eqs. (6)–(7). JSBSim's torque-balance shaft dynamics is the same structure with table
    aerodynamic power.
    """
    return ((K_q / R_a) * (V_m - K_e * W) - k_m * W**2 - b * W) / J_r


def esc_mean_voltage(u: sp.Expr, V_batt: sp.Expr) -> sp.Expr:
    """Averaged inverter: V_m = u·V_batt, u = duty ∈ [0,1].
    Source: standard; documented verbatim in crazyflie-firmware platform_defaults_cf2.h:58."""
    return u * V_batt


def steady_state_speed(u: sp.Expr, V_batt: sp.Expr, K_q: sp.Expr, K_e: sp.Expr,
                       R_a: sp.Expr, k_m: sp.Expr, b: sp.Expr) -> sp.Expr:
    """
    Battery-coupled steady state of dc_motor_speed_dynamics (Ω̇ = 0), positive root of
    k_m·Ω² + (K_q·K_e/R_a + b)·Ω − (K_q/R_a)·u·V_batt = 0:

        Ω_ss = [−β + √(β² + 4·k_m·(K_q/R_a)·u·V_batt)] / (2·k_m),  β = K_q·K_e/R_a + b

    √-like in u·V_batt when the quadratic load dominates, linear near zero — exactly the
    shape the verified throttle curve √(k·u² + (1−k)·u) emulates.
    """
    beta = K_q * K_e / R_a + b
    return (-beta + sp.sqrt(beta**2 + 4 * k_m * (K_q / R_a) * u * V_batt)) / (2 * k_m)


def crazyflie_thrust_from_voltage(v_m: sp.Expr, C0: sp.Expr, C1: sp.Expr,
                                  C2: sp.Expr, C3: sp.Expr) -> sp.Expr:
    """
    Crazyflie firmware static per-motor model (current master):

        T(v_m) = C0 + C1·v_m + C2·v_m² + C3·v_m³   [N],   v_m = V_batt·PWM/65535

    The firmware's battery compensation inverts this cubic (Cardano) for the required v_m and
    sets PWM = 65535·v_required/V_batt — holding thrust constant as the battery sags. CF2
    2.1+ props: C = (−0.024765, 0.065238, −0.026793, 0.0067768); torque = 0.0069929·T (N·m).
    Legacy quadratic form (tag 2022.01, motors.c:152–173): v_required =
    −0.0006239·thrust_g² + 0.088·thrust_g (thrust in grams).
    Source: bitcraze/crazyflie-firmware, src/drivers/src/motors.c
    (motorsCompensateBatteryVoltage) + platform_defaults_cf2.h:57–91.
    """
    return C0 + C1 * v_m + C2 * v_m**2 + C3 * v_m**3


def thevenin_battery(SOC: sp.Expr, i_batt: sp.Expr, V_TS: sp.Expr, V_TL: sp.Expr,
                     V_OC, R_S, R_TS: sp.Expr, C_TS: sp.Expr,
                     R_TL: sp.Expr, C_TL: sp.Expr, C_cap: sp.Expr) -> tuple:
    """
    Thevenin equivalent-circuit battery (OCV–SoC + series R + two RC branches):

        SȮC = −i/C_cap
        V̇_TS = −V_TS/(R_TS·C_TS) + i/C_TS       V̇_TL = −V_TL/(R_TL·C_TL) + i/C_TL
        V_batt = V_OC(SOC) − i·R_S(SOC) − V_TS − V_TL

    Fitted 850 mAh LiPo cell (Chen & Rincon-Mora 2006, Eqs. (2)–(7)):
    V_OC(SOC) = −1.031·e^(−35·SOC) + 3.685 + 0.2156·SOC − 0.1178·SOC² + 0.3201·SOC³.
    Load coupling: i_batt = Σ motors (u·V_batt − K_e·Ω)/R_a + avionics. The Gazebo
    LinearBatteryPlugin (de-facto drone-sim standard) is this with affine V_OC and the RC
    branches replaced by a low-passed current: V = e0 + e1·(1 − q/c) − r·ī.
    Returns (dSOC/dt, dV_TS/dt, dV_TL/dt, V_batt).
    Source: Chen & Rincon-Mora, IEEE Trans. Energy Conversion 21(2), 2006, Eqs. (1)–(7);
    gazebosim/gz-sim LinearBatteryPlugin.cc.
    """
    soc_dot = -i_batt / C_cap
    vts_dot = -V_TS / (R_TS * C_TS) + i_batt / C_TS
    vtl_dot = -V_TL / (R_TL * C_TL) + i_batt / C_TL
    v_batt = V_OC(SOC) - i_batt * R_S(SOC) - V_TS - V_TL
    return soc_dot, vts_dot, vtl_dot, v_batt


def chen_ocv(SOC: sp.Expr) -> sp.Expr:
    """The Chen & Rincon-Mora fitted single-cell LiPo open-circuit voltage (V)."""
    return (-1.031 * sp.exp(-35 * SOC) + 3.685 + 0.2156 * SOC
            - 0.1178 * SOC**2 + 0.3201 * SOC**3)
