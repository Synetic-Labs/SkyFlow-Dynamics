"""
Per-rotor aerodynamics and frame drag.

All airspeeds here are body-frame. The vehicle airspeed at the CoM is
    v_a = R(q)ᵀ · (v − v_wind)                                   (world → body)
and the local airspeed at rotor hub i includes the rigid-body rotation lever arm:
    v_i = v_a + ω × r_i.

Verified sources: RotorPy paper §II-B (arXiv:2306.04485); Mahony, Kumar, Corke, IEEE RAM 2012
(rotor drag / flapping); SkyDreamer (thrust vs angle-of-attack / advance ratio); Crazyflow
(polynomial thrust/torque curves).
"""

import sympy as sp

from skyflow_dynamics.spec.frames import EZ, cross


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

    Applied to the BASE thrust polynomial only (before translational lift).
    ⚠ Paper-vs-implementation discrepancies (spec follows the RUNNABLE reference, which the
    golden vectors pin): the released implementation (The-Real-Thisas/dreamerv3
    embodied/envs/skydreamer.py:270–280, explicitly ENU z-up) uses the MEAN rotor speed and
    ‖v_xy‖ exactly as above with k_angle = +3.145 — while the paper's printed equations state
    an NED convention, Ω̄ = Σω (the sum), and a SQUARED in-plane numerator. Note the physical
    implication of the as-implemented ENU form: climbing (v_az > 0) INCREASES modeled thrust,
    opposite to the momentum-theory expectation — treat the α-term sign as part of the
    identification convention and pin it against data before reusing the constants.
    ⚠ SkyDreamer's k_w (our ct2) is mass-normalized — multiply by m when porting (finding F-4).
    ⚠ Use either {k_angle, k_hor} or k_h (translational lift), never both — same physics,
    different fidelity (double-count guard).
    Identified: k_angle = 3.145, k_hor = 7.245, r_prop = 0.0635 m (5-inch racer).
    Source: SkyDreamer paper (arXiv:2510.14783) + reference implementation lines 270–280.
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
    Source: RotorPy reference implementation (multirotor.py, k_h) — the term is in the code
    but NOT in the RotorPy paper, whose aero section (§II-B) lists only parasitic drag, rotor
    drag, and blade flapping.
    """
    return k_h * (v_i[0]**2 + v_i[1]**2)


def rotor_drag_force(W_i: sp.Expr, v_i: sp.Matrix, k_d: sp.Expr, k_z: sp.Expr) -> sp.Matrix:
    """
    Rotor drag ("H-force") at hub i, linear in rotor speed × local airspeed (N):
        H_i = −Ω_i · diag(k_d, k_d, k_z) · v_i
    k_d: in-plane blade-drag coefficient; k_z: induced-inflow coefficient on the rotor axis.
    Strictly dissipative at the hub: H_i · v_i = −Ω_i (k_d(v_x²+v_y²) + k_z v_z²) ≤ 0.
    Sources: Mahony et al. 2012 §III; RotorPy paper §II-B.
    """
    return -W_i * sp.Matrix([[k_d, 0, 0], [0, k_d, 0], [0, 0, k_z]]) * v_i


def flapping_moment(W_i: sp.Expr, v_i: sp.Matrix, k_flap: sp.Expr) -> sp.Matrix:
    """
    Blade-flapping moment at hub i (N·m):
        M_flap,i = −k_flap · Ω_i · (v_i × ẑ)
    Forward flight (v_x > 0) gives M_y = +k_flap·Ω·v_x, which in THIS repo's FLU body frame
    (y left) is a nose-DOWN moment — note the classical flap-back response is nose-up, so the
    SIGN of an identified k_flap is part of its identification convention; pin it against
    data before setting k_flap ≠ 0 (no shipped reference vehicle does). Expression matches
    the verified reference exactly (golden-pinned).
    Source: Mahony et al. 2012 §III; RotorPy reference implementation (multirotor.py).
    """
    return -k_flap * W_i * cross(v_i, EZ)


def rolling_moment(W_i: sp.Expr, s_i: sp.Expr, v_i: sp.Matrix, mu_R: sp.Expr) -> sp.Matrix:
    """
    CANDIDATE — per-rotor rolling moment from advancing/retreating blade lift dissymmetry:
    a hub TORQUE parallel to the in-plane airspeed, sign set by spin direction:

        M_roll,i = −Ω_i · s_i · μ_R · v_⊥,i,     v_⊥,i = v_i − (v_i·ẑ)ẑ

    (PX4 SITL's signed variant of the RotorS gazebo_motor_model rolling moment; RotorS omits
    the spin sign — a bug their forks fixed. Kai et al. 2017 Eq. (7) carries the same physics
    with √T_i scaling instead of Ω_i, but ⚠ under the spin-sign convention his Eq. (8) pins,
    Kai's c_d2 term comes out with the OPPOSITE sign to this classical/PX4 form — only the
    magnitude and scaling map over; the sign here follows PX4.) Cancels pairwise for balanced
    counter-rotating pairs at equal speeds. μ_R: rolling_moment_coefficient, kg·m/rad.
    Source: PX4-SITL_gazebo-classic gazebo_motor_model.cpp; ethz-asl/rotors_simulator;
    Kai, Allibert, Hua, Hamel, IFAC 2017 Eq. (7).
    """
    v_perp = sp.Matrix([v_i[0], v_i[1], 0])
    return -W_i * s_i * mu_R * v_perp


def flapping_force_kai(T_i: sp.Expr, s_i: sp.Expr, v_i: sp.Matrix, w: sp.Matrix,
                       c_av: sp.Expr, c_bv: sp.Expr, c_aw: sp.Expr,
                       c_bw: sp.Expr) -> sp.Matrix:
    """
    CANDIDATE — blade-flapping FORCE with lateral (spin-signed) and body-rate components
    (Kai et al. 2017 Eq. (10)), body frame:

        F_flap,i = −√T_i·c_av·Π_ẑ·v_i + s_i·√T_i·c_bv·(ẑ×v_i)
                   + s_i·√T_i·c_bw·Π_ẑ·ω + √T_i·c_aw·(ẑ×ω)

    Π_ẑ = in-plane projector. Frame conversion note: Kai's Eq. (10) is written in a z-DOWN
    body frame with spin signs about e3 = −ẑ; converting with C = diag(1,−1,−1) (so
    sgn(ω_i) = −s_i, e3×(·) = −ẑ×(·), Π even) flips the two body-rate terms' signs relative
    to a naive transcription — the airspeed terms keep theirs (double sign cancellations).
    The √T (≈√(mg/4) near hover) scaling is what makes the lumped linear-drag model
    (spec.rotor_aero.linear_drag) exact at hover: the s_i-signed terms cancel pairwise for
    counter-rotating rotors (Kai Remark 1), and the c_av term lumps into
    A = 2√(mg)(c_d1 + c_av)·Π_ẑ — the published basis for Faessler's model. The body-rate
    terms (c_aw, c_bw) are rotor-plane damping ABSENT from the verified tier.
    Source: Kai, Allibert, Hua, Hamel, IFAC World Congress 2017, Eqs. (10)–(13).
    """
    v_perp = sp.Matrix([v_i[0], v_i[1], 0])
    w_perp = sp.Matrix([w[0], w[1], 0])
    rt = sp.sqrt(T_i)
    return (-rt * c_av * v_perp + s_i * rt * c_bv * cross(EZ, v_i)
            + s_i * rt * c_bw * w_perp + rt * c_aw * cross(EZ, w))


def flapping_moment_body_rate(W_i: sp.Expr, w: sp.Matrix, k_flap_w: sp.Expr) -> sp.Matrix:
    """
    CANDIDATE — rotor damping moment in roll/pitch: the tip-path plane lags a rolling or
    pitching body (the first-harmonic flap angles carry p_w/Ω and 16·q_w/(γΩ) terms, γ =
    Lock number), and the resulting hub moment opposes the body rate. Minimal one-parameter
    lumping of the JSBSim FGRotor form (NASA TP-1285 eqn 32; rate terms per Amer, NACA
    TN-2136):

        M_flap_w,i = −k_flap_w · Ω_i · Π_ẑ·ω,      Π_ẑ·ω = (ω_x, ω_y, 0)

    Spin-sign-FREE, so it adds pairwise over counter-rotating rotors (a net measurable
    damping derivative, unlike rolling_moment which cancels for balanced pairs). Kai et al.
    2017 Eq. (7) carries the same hub moment with √T_i scaling instead of Ω_i — equivalent
    near hover; both are dissipative on the in-plane rates: ω·M = −k·Ω·(ω_x²+ω_y²) ≤ 0.
    Lumping boundary: the source's flap angles also carry same-order gyroscopic CROSS
    terms (roll moment ∝ −q, pitch moment ∝ +p, relative size γ/16) — dropped here as
    they do no work and, being spin-signed, cancel pairwise on counter-rotating quads.
    For stiff (hingeless) quadrotor props the TP-1285 hinge-offset hub-moment constant
    becomes an effective hub stiffness folded into k_flap_w. k_flap_w: kg·m²/rad².
    Source: JSBSim FGRotor.cpp calc_flapping_angles/body_moments (NASA TP-1285 eqns 32, 43;
    NACA TN-2136); Kai, Allibert, Hua, Hamel, IFAC 2017 Eq. (7).
    """
    w_perp = sp.Matrix([w[0], w[1], 0])
    return -k_flap_w * W_i * w_perp


def blade_profile_drag(C_T: sp.Expr, a: sp.Expr, sigma: sp.Expr) -> sp.Expr:
    """
    CANDIDATE — blade profile-drag coefficient from a simplified Bailey drag polar in the
    MEAN BLADE INCIDENCE ᾱ = 6·C_T/(a·σ) — the mean lift coefficient 6·C_T/σ divided by
    the lift-curve slope (constants hardcoded in JSBSim FGRotor::calc_torque):

        δ = 0.009 + 0.3·(6·C_T/(a·σ))²

    ⚠ Because the polar's argument is incidence, NOT lift coefficient, constants from
    C_L-based literature polars must be rescaled by 1/a² (≈1/36) before transplanting —
    cf. Bailey's original δ = 0.0087 − 0.0216·α + 0.400·α² (NACA Rep. 716).
    a: blade lift-curve slope (1/rad, ≈ 5.7–6.3); σ = b·c/(πR) rotor solidity (b blades of
    chord c). Feeds bramwell_torque.
    Source: JSBSim FGRotor.cpp calc_torque; polar lineage per Bailey, NACA Rep. 716 (1941).
    """
    return 0.009 + 0.3 * (6 * C_T / (a * sigma))**2


def bramwell_torque(T: sp.Expr, H: sp.Expr, mu: sp.Expr, lam: sp.Expr, rho: sp.Expr,
                    blades: sp.Expr, chord: sp.Expr, R: sp.Expr, Omega: sp.Expr,
                    delta: sp.Expr) -> sp.Expr:
    """
    CANDIDATE — rotor shaft (drag) torque with flight-condition dependence:

        Q = ρ·b·c·δ·(ΩR)²·R²·(1 + 4.5μ²)/8 − (T·λ + H·μ)·R

    Nondimensional flight state (from spec.inflow): edgewise advance ratio μ = ‖v_⊥‖/(ΩR);
    inflow ratio λ = −(v_climb + v_i)/(ΩR) with v_climb the axial airspeed along the thrust
    axis (positive climbing) and v_i ≥ 0 the induced velocity — λ < 0 in hover, so
    −T·λ·R = T·(v_climb+v_i)/Ω is exactly the momentum-theory induced + climb power over Ω.
    H: rotor in-plane drag force resolved along −v̂_⊥ (positive opposing edgewise motion,
    the H-force/a_dw of the source). First term: blade profile torque growing with edgewise
    speed as (1+4.5μ²). Generalizes the verified rotor_torque_polynomial, which is this
    model's fixed-flight-condition slice in Ω. Signs follow the source's control-axes
    bookkeeping exactly (the H-force profile power is already inside the 4.5μ² growth);
    autorotation: Q → 0 as descent drives λ positive. Validity envelope: JSBSim clamps
    μ ≤ 0.7 before evaluating these closed forms — do the same (fast multirotors with low
    tip speed can exceed it).
    Source: JSBSim FGRotor.cpp calc_torque (simplified SH79 eqn 36); Bramwell, Helicopter
    Dynamics 2nd ed., eqns 3.43–3.44.
    """
    profile = rho * blades * chord * delta * (Omega * R)**2 * R**2 \
        * (1 + sp.Rational(9, 2) * mu**2) / 8
    return profile - (T * lam + H * mu) * R


def parasitic_drag(v_a: sp.Matrix, c_D: sp.Matrix) -> sp.Matrix:
    """
    Quadratic frame drag at the CoM (N):
        D = −‖v_a‖ · diag(c_Dx, c_Dy, c_Dz) · v_a
    ⚠ Norm convention: this uses the full airspeed magnitude ‖v_a‖ per axis. SkyDreamer instead
    uses per-axis |v_k|·v_k — a structurally different model; the two agree only on a single
    axis. Do not mix identified coefficients across the two forms.
    Source: RotorPy paper §II-B — note its printed Eq. (7) reads −C‖v_a‖²·v_a (cubic), which
    contradicts both the paper's own text ("proportional to the airspeed squared") and the
    reference implementation (−‖v_a‖·C·v_a, quadratic, multirotor.py); the spec follows the
    implementation, which the golden vectors pin.
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
