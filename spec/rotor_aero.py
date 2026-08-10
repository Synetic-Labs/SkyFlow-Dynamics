"""
Per-rotor aerodynamics and frame drag.

All airspeeds here are body-frame. The vehicle airspeed at the CoM is
    v_a = R(q)ᵀ · (v − v_wind)                                   (world → body)
and the local airspeed at rotor hub i includes the rigid-body rotation lever arm:
    v_i = v_a + ω × r_i.

Verified sources: RotorPy paper §III (arXiv:2306.04485); Mahony, Kumar, Corke, IEEE RAM 2012
(rotor drag / flapping); SkyDreamer (thrust vs angle-of-attack / advance ratio); Crazyflow
(polynomial thrust/torque curves).
"""

import sympy as sp

from spec.frames import EZ, cross


def local_airspeed(v_a: sp.Matrix, w: sp.Matrix, r_i: sp.Matrix) -> sp.Matrix:
    """Body-frame airspeed at rotor hub i:  v_i = v_a + ω × r_i."""
    return v_a + cross(w, r_i)


def thrust_magnitude(W_i: sp.Expr, ct0: sp.Expr, ct1: sp.Expr, ct2: sp.Expr) -> sp.Expr:
    """
    Rotor thrust magnitude (N) as a polynomial in rotor speed:
        T_i = ct0 + ct1·Ω_i + ct2·Ω_i²
    ct2 ≡ k_η is the classical quadratic coefficient; ct0 = ct1 = 0 recovers T = k_η·Ω².
    ⚠ Crazyflow's identified polynomials are per-RPM — convert ct1 by 60/2π, ct2 by (60/2π)².
    Sources: standard rotor model; Crazyflow first_principles/dynamics.py:121 + params.toml.
    """
    return ct0 + ct1 * W_i + ct2 * W_i**2


def torque_magnitude(W_i: sp.Expr, cq0: sp.Expr, cq1: sp.Expr, cq2: sp.Expr) -> sp.Expr:
    """
    Rotor aerodynamic drag-torque magnitude (N·m):
        Q_i = cq0 + cq1·Ω_i + cq2·Ω_i²
    cq2 ≡ k_m. The torque on the AIRFRAME opposes the rotor spin: τ_yaw,i = −s_i·Q_i·ê_i
    (assembled in spec.wrench). Source: as thrust_magnitude.
    """
    return cq0 + cq1 * W_i + cq2 * W_i**2


def aoa_thrust_factor(v_a: sp.Matrix, W_bar: sp.Expr, r_prop: sp.Expr,
                      k_angle: sp.Expr, k_hor: sp.Expr) -> sp.Expr:
    """
    Multiplicative thrust correction for rotor angle of attack α and advance-ratio angle μ
    (SkyDreamer, identified to racing speeds):

        α = atan2(v_a·ẑ, r_prop·Ω̄),   μ = atan2(‖(v_a·x̂, v_a·ŷ)‖, r_prop·Ω̄),
        factor = 1 + k_angle·α + k_hor·μ,        Ω̄ = mean rotor speed.

    Applied to the BASE thrust polynomial only (before translational lift). Climbing (v_az > 0)
    raises α and increases thrust in this identified model (α, μ signed by atan2).
    ⚠ SkyDreamer's k_w (our ct2) is mass-normalized — multiply by m when porting (finding F-4).
    ⚠ Use either {k_angle, k_hor} or k_h (translational lift), never both — same physics,
    different fidelity (double-count guard).
    Identified: k_angle = 3.145, k_hor = 7.245, r_prop = 0.0635 m (5-inch racer).
    Source: SkyDreamer paper + implementation lines 274–280.
    """
    denom = r_prop * W_bar
    alpha = sp.atan2(v_a[2], denom)
    mu = sp.atan2(sp.sqrt(v_a[0]**2 + v_a[1]**2), denom)
    return 1 + k_angle * alpha + k_hor * mu


def translational_lift(v_i: sp.Matrix, k_h: sp.Expr) -> sp.Expr:
    """
    Translational lift: added thrust from in-plane airspeed over the rotor (N):
        ΔT_i = k_h · (v_i·x̂² + v_i·ŷ²)
    The small-airspeed linearization of the same effect the AoA/advance-ratio factor models.
    Source: RotorPy paper §III.
    """
    return k_h * (v_i[0]**2 + v_i[1]**2)


def rotor_drag_force(W_i: sp.Expr, v_i: sp.Matrix, k_d: sp.Expr, k_z: sp.Expr) -> sp.Matrix:
    """
    Rotor drag ("H-force") at hub i, linear in rotor speed × local airspeed (N):
        H_i = −Ω_i · diag(k_d, k_d, k_z) · v_i
    k_d: in-plane blade-drag coefficient; k_z: induced-inflow coefficient on the rotor axis.
    Strictly dissipative at the hub: H_i · v_i = −Ω_i (k_d(v_x²+v_y²) + k_z v_z²) ≤ 0.
    Sources: Mahony et al. 2012 §III; RotorPy paper §III.
    """
    return -W_i * sp.Matrix([[k_d, 0, 0], [0, k_d, 0], [0, 0, k_z]]) * v_i


def flapping_moment(W_i: sp.Expr, v_i: sp.Matrix, k_flap: sp.Expr) -> sp.Matrix:
    """
    Blade-flapping moment at hub i (N·m):
        M_flap,i = −k_flap · Ω_i · (v_i × ẑ)
    Forward flight (v_x > 0) gives M_y = +k_flap·Ω·v_x — pitch-up, the classical flapping
    response. Source: Mahony et al. 2012 §III; RotorPy paper §III.
    """
    return -k_flap * W_i * cross(v_i, EZ)


def parasitic_drag(v_a: sp.Matrix, c_D: sp.Matrix) -> sp.Matrix:
    """
    Quadratic frame drag at the CoM (N):
        D = −‖v_a‖ · diag(c_Dx, c_Dy, c_Dz) · v_a
    ⚠ Norm convention: this uses the full airspeed magnitude ‖v_a‖ per axis. SkyDreamer instead
    uses per-axis |v_k|·v_k — a structurally different model; the two agree only on a single
    axis. Do not mix identified coefficients across the two forms.
    Source: RotorPy paper §III.
    """
    speed = sp.sqrt(v_a[0]**2 + v_a[1]**2 + v_a[2]**2)
    return -speed * sp.Matrix([[c_D[0], 0, 0], [0, c_D[1], 0], [0, 0, c_D[2]]]) * v_a


def linear_drag(v_a: sp.Matrix, c_L: sp.Matrix) -> sp.Matrix:
    """
    Lumped linear body-frame drag at the CoM (N):
        F = −diag(c_Lx, c_Ly, c_Lz) · v_a,     c_L ≥ 0 in N/(m/s)
    The rotor-drag model of Faessler, Franchi, Scaramuzza (RA-L 2018): in the world frame
    F_W = −R·diag(c_L)·Rᵀ·(v − v_wind), which is what makes quadrotor dynamics with rotor drag
    differentially flat. Crazyflow's first-principles model uses exactly this (their stored
    drag_matrix is the NEGATED diagonal: drag_matrix = −diag(c_L)). It is the Ω-independent
    lumping of the per-rotor H-force −Ω·K·v_i summed over rotors at nominal speed
    (c_L ≈ k_d·ΣΩ_hover in-plane); identify a vehicle against ONE of the two forms, not both.
    Sources: Faessler et al. RA-L 2018; crazyflow first_principles/dynamics.py:127.
    """
    return -sp.Matrix([[c_L[0], 0, 0], [0, c_L[1], 0], [0, 0, c_L[2]]]) * v_a


def vertical_climb_drag(v_a: sp.Matrix, k_v2: sp.Expr) -> sp.Matrix:
    """
    Collective vertical airspeed-squared thrust term at the CoM, body ẑ (N):
        D_z = −k_v2 · v_az · |v_az| · ẑ
    The vertical companion of the AoA thrust model (SkyDreamer; identified k_v2 = 0 for their
    5-inch racer, kept for generality). Source: SkyDreamer paper + implementation.
    """
    return sp.Matrix([0, 0, -k_v2 * v_a[2] * sp.Abs(v_a[2])])
