"""
Rotor inflow in descent — the vortex ring state (VRS) — CANDIDATE tier.

A rotor that descends at a rate near its own induced velocity re-ingests its wake. Momentum
theory has no valid solution there (spec.inflow), and the mean flow through the disk
RISES with descent rate over part of the band: at fixed pitch and rotor speed the thrust
then FALLS as the descent gets faster (negative heave damping — the "settling with power"
of helicopters, the propwash thrust loss of a multirotor).

This module carries Johnson's parametric model of the mean induced velocity (NASA/TP-2005-
213477, read from the report: text pp. 14–15, Table 3, Table 4, Figs. 37–39). It replaces
momentum theory inside the VRS band and joins it outside. spec.bem.vrs_induced_velocity
(the Leishman quartic) is the older alternative for the same band — use ONE of them. The
two differ in kind: with the quartic the net flow through the disk only falls in descent,
so a fixed-pitch rotor never loses thrust; Johnson's curve has the rising part.

Conventions of this module (Johnson's): every velocity is divided by the hover induced
velocity v_h = √(T/(2ρA)) of the CURRENT thrust; V_z is the axial velocity of the rotor
through the air, POSITIVE IN CLIMB (descent is V_z < 0); V_x ≥ 0 is the in-plane speed; v is
the induced velocity, positive down through the disk. The net flow through the disk is
V_z + v. Thrust from it: spec.inflow.inflow_thrust_ratio.
"""

from functools import cache

import sympy as sp

#: Johnson 2005, Table 4 — "VRS model parameters (velocities scaled with v_h)". A, B, C shape
#: the baseline that removes the momentum-theory singularity; D, N, X, E, M shape the VRS
#: increment. (V+v)_N and (V+v)_X are the net-inflow minimum and maximum in vertical descent.
#: Helicopter and tiltrotor values: (V+v)_X = 1.25 comes from flight-dynamics matches (the
#: Dauphin D6075 and V-22 encounters, pp. 15–16); Johnson's Fig. 36 collects measured maxima
#: from about 1.0 to 1.3.
JOHNSON_VRS = {
    "VzA": sp.Rational(-3, 2), "VzB": sp.Rational(-21, 10), "VxC": sp.Rational(3, 4),
    "VzD": sp.Rational(-1, 5), "VzN": sp.Rational(-9, 20), "VvN": sp.Rational(17, 20),
    "VzX": sp.Rational(-3, 2), "VvX": sp.Rational(5, 4), "VzE": sp.Integer(-2),
    "VxM": sp.Rational(19, 20),
}


def momentum_inflow_residual(v: sp.Expr | float, Vz: sp.Expr | float,
                             Vx: sp.Expr | float) -> sp.Expr:
    """
    Momentum theory in Johnson's scaled form: v is a root of

        v²·((V_z + v)² + V_x²) − 1 = 0

    — spec.inflow.oblique_momentum_thrust divided by 2ρA·v_h². In descent it can have three
    positive roots: the LARGEST is the helicopter branch (continuous from hover), the
    SMALLEST the windmill-brake branch.
    """
    return sp.sympify(v ** 2 * ((Vz + v) ** 2 + Vx ** 2) - 1)


def momentum_inflow_slope(v: sp.Expr | float, Vz: sp.Expr | float,
                          Vx: sp.Expr | float) -> sp.Expr:
    """dv/dV_z along a momentum-theory branch (implicit derivative of the residual):

        dv/dV_z = −v·(V_z + v) / ((V_z + v)² + V_x² + v·(V_z + v))
    """
    w = Vz + v
    return sp.sympify(-v * w / (w ** 2 + Vx ** 2 + v * w))


def momentum_inflow_axial(Vz: sp.Expr | float, branch: str = "helicopter") -> sp.Expr:
    """Closed-form axial roots (V_x = 0): helicopter branch ½(−V_z + √(V_z² + 4)), valid for
    V_z + v > 0; windmill-brake branch ½(−V_z − √(V_z² − 4)), which exists for V_z ≤ −2."""
    if branch == "helicopter":
        return (-Vz + sp.sqrt(Vz ** 2 + 4)) / 2
    if branch == "windmill":
        return (-Vz - sp.sqrt(Vz ** 2 - 4)) / 2
    raise ValueError(branch)


def johnson_transition_points(Vx: sp.Expr | float, c: dict = JOHNSON_VRS) -> dict:
    """
    Where the polynomial pieces join, as functions of the in-plane speed (Table 3, steps 2.1
    and 3.1). Returns V_z at A, B (baseline) and D, N, X, E (VRS increment):

        V_zA' = V_zA + 0.2·(V_x/V_xC)²          V_zB' = V_zB + 0.2·(V_x/V_xC)²
        and if V_x/V_xC > 0.5:  V_zB' ← V_zB' + 0.7·(V_zA' − V_zB')·(2·V_x/V_xC − 1)³
        V_zD' = V_zD
        V_zN' = ½(V_zN + V_zX) + ½(V_zN − V_zX)·(1 − (V_x/V_xM)²)^0.2
        V_zX' = ½(V_zN + V_zX) − ½(V_zN − V_zX)·(1 − (V_x/V_xM)²)^1.5
        V_zE' = V_zE + (V_zX' − V_zX)

    The baseline pieces apply for V_x < V_xC, the VRS increment for V_x < V_xM: with forward
    speed N and X close on each other and the unstable region ends at V_x = V_xM = 0.95.
    """
    q = Vx / c["VxC"]
    m = Vx / c["VxM"]
    zA = c["VzA"] + sp.Rational(1, 5) * q ** 2
    zB0 = c["VzB"] + sp.Rational(1, 5) * q ** 2
    zB = sp.Piecewise((zB0 + sp.Rational(7, 10) * (zA - zB0) * (2 * q - 1) ** 3,
                       q > sp.Rational(1, 2)), (zB0, True))
    mid, half = (c["VzN"] + c["VzX"]) / 2, (c["VzN"] - c["VzX"]) / 2
    zN = mid + half * (1 - m ** 2) ** sp.Rational(1, 5)
    zX = mid - half * (1 - m ** 2) ** sp.Rational(3, 2)
    return {"A": sp.sympify(zA), "B": zB, "D": c["VzD"], "N": zN, "X": zX,
            "E": c["VzE"] + (zX - c["VzX"])}


@cache
def _cubic_coefficients(kind: str) -> tuple:
    """Coefficients of p(z) = a + b·z + c·z² + d·z³ from end conditions, solved once.
    "full": value and slope at both ends. "origin": a = 0, value and slope at z0, value at z1
    (Johnson's form where one end slope is left free)."""
    a, b, c, d, z0, y0, s0, z1, y1, s1 = sp.symbols("a b c d z0 y0 s0 z1 y1 s1")
    if kind == "full":
        p = lambda z: a + b * z + c * z ** 2 + d * z ** 3
        dp = lambda z: b + 2 * c * z + 3 * d * z ** 2
        sol = sp.solve([p(z0) - y0, dp(z0) - s0, p(z1) - y1, dp(z1) - s1], [a, b, c, d],
                       dict=True)[0]
        return (z0, y0, s0, z1, y1, s1), tuple(sp.factor(sol[k]) for k in (a, b, c, d))
    if kind == "origin":
        p = lambda z: b * z + c * z ** 2 + d * z ** 3
        dp = lambda z: b + 2 * c * z + 3 * d * z ** 2
        sol = sp.solve([p(z0) - y0, dp(z0) - s0, p(z1) - y1], [b, c, d], dict=True)[0]
        return (z0, y0, s0, z1, y1), (sp.Integer(0), *(sp.factor(sol[k]) for k in (b, c, d)))
    raise ValueError(kind)


def _cubic(kind: str, z, *ends) -> tuple:
    """(p(z), dp/dz) for the cubic through the given end conditions."""
    syms, coef = _cubic_coefficients(kind)
    a, b, c, d = (k.subs(dict(zip(syms, ends)), simultaneous=True) for k in coef)
    return a + b * z + c * z ** 2 + d * z ** 3, b + 2 * c * z + 3 * d * z ** 2


def johnson_baseline_cubic(Vz: sp.Expr | float, zA: sp.Expr | float, vA: sp.Expr | float,
                           dvA: sp.Expr | float, zB: sp.Expr | float,
                           vB: sp.Expr | float) -> tuple:
    """
    Baseline induced velocity between B and A (Table 3, step 2.2): the cubic

        v = b·V_z + c·V_z² + d·V_z³      (no constant term)

    that matches momentum theory's v and dv/dV_z at A (helicopter branch) and its v at B
    (windmill-brake branch). It bridges the band where momentum theory is singular (ideal
    autorotation, V_z + v = 0). Returns (v, dv/dV_z). The momentum values at A and B are
    inputs: for V_x > 0 they are roots of momentum_inflow_residual (no closed form).
    """
    return _cubic("origin", Vz, zA, vA, dvA, zB, vB)


def johnson_vrs_increment(Vz: sp.Expr | float, Vx: sp.Expr | float, slope_N: sp.Expr | float,
                          slope_X: sp.Expr | float, c: dict = JOHNSON_VRS) -> sp.Expr:
    """
    The VRS increment Δ(V_z + v) on top of the baseline (Table 3, step 3) — three cubics in
    V_z between D, N, X and E that put a minimum of the net inflow at N and a maximum at X:

        at D:  Δ = 0, dΔ/dV_z = 0
        at N:  Δ = Δ_N·g,  dΔ/dV_z = −(1 + dv_base/dV_z)      (net inflow has zero slope)
        at X:  Δ = Δ_X·g,  dΔ/dV_z = −(1 + dv_base/dV_z)
        at E:  Δ = 0 (slope free; that cubic has no constant term)
        Δ_N = (V+v)_N − (V_zN + v_mom(V_zN)),  Δ_X = (V+v)_X − (V_zX + v_mom(V_zX)),
        g = √(1 − (V_x/V_xM)⁶)

    and Δ = 0 outside (E, D), for V_z ≥ 0 and for V_x ≥ V_xM. slope_N and slope_X are the
    baseline slopes dv_base/dV_z at the SHIFTED points N', X' (johnson_transition_points) and
    the current V_x. Δ_N and Δ_X use momentum theory in VERTICAL descent at the unshifted
    points — 0.05 and 0.75 with Table 4. (Table 3 prints "v_Nmom" without saying which V_x;
    this reading reproduces Fig. 37's V_x = 0.75 curve — maximum 0.85 near V_z = −1.1,
    minimum 0.58 near −0.55 — and evaluating at the current V_x does not: 1.11.)
    Between N and X the net inflow rises with descent rate: the unstable band.
    ⚠ Not continuous in V_x at V_xM. As V_x → V_xM the values at N' and X' go to zero and
    the two points merge, but the slope conditions stay, so the cubics keep an inflection
    with a horizontal tangent: up to 0.15·v_h of increment remains (near V_z = −1.25) and
    drops to zero when V_x crosses V_xM. Smooth it before differentiating through.
    """
    t = johnson_transition_points(Vx, c)
    g = sp.sqrt(1 - (Vx / c["VxM"]) ** 6)
    dN = (c["VvN"] - (c["VzN"] + momentum_inflow_axial(c["VzN"]))) * g
    dX = (c["VvX"] - (c["VzX"] + momentum_inflow_axial(c["VzX"]))) * g
    sN, sX = -(1 + slope_N), -(1 + slope_X)
    seg_DN = _cubic("full", Vz, t["D"], 0, 0, t["N"], dN, sN)[0]
    seg_NX = _cubic("full", Vz, t["N"], dN, sN, t["X"], dX, sX)[0]
    seg_XE = _cubic("origin", Vz, t["X"], dX, sX, t["E"], 0)[0]
    return sp.Piecewise((0, Vz >= t["D"]), (seg_DN, Vz >= t["N"]), (seg_NX, Vz >= t["X"]),
                        (seg_XE, Vz > t["E"]), (0, True))


def johnson_vrs_inflow_axial(Vz: sp.Expr | float, kappa: sp.Expr | float = 1,
                             f: sp.Expr | float = 1, c: dict = JOHNSON_VRS) -> sp.Expr:
    """
    The whole model in vertical flight (V_x = 0), explicit in V_z:

        v = κ·(v_base + f·Δv_vrs)

    v_base: momentum theory (helicopter branch) for V_z ≥ V_zA, the baseline cubic for
    V_zB < V_z < V_zA, momentum theory (windmill-brake branch) below. κ ≥ 1 is the usual
    induced-power factor (measured hover inflow over the ideal value; the report's
    comparisons use 1.05–1.20); f scales the VRS increment (f = 0 gives the stable baseline).
    With Table 4: net inflow V_z + v/κ = 0.85 at V_z = −0.45 (minimum) and 1.25 at −1.5
    (maximum); momentum theory again below V_z = −2.1.
    """
    zA, zB = c["VzA"], c["VzB"]
    vA = momentum_inflow_axial(zA)
    dvA = momentum_inflow_slope(vA, zA, 0)
    cubic, _ = johnson_baseline_cubic(Vz, zA, vA, dvA, zB, momentum_inflow_axial(zB, "windmill"))
    base = sp.Piecewise((momentum_inflow_axial(Vz), Vz >= zA), (cubic, Vz > zB),
                        (momentum_inflow_axial(Vz, "windmill"), True))
    vN = momentum_inflow_axial(c["VzN"])
    # N lies above A and X coincides with A, so the baseline slope at both is momentum theory's.
    delta = johnson_vrs_increment(Vz, 0, momentum_inflow_slope(vN, c["VzN"], 0), dvA, c)
    return kappa * (base + f * delta)


def johnson_inflow_time_constant(lam_h: sp.Expr | float, Omega: sp.Expr | float) -> sp.Expr:
    """
    Time constant of the inflow's first-order approach to the quasi-steady VRS value,
    τ·dv/dt + v = v_QS (the form of spec.inflow.dynamic_inflow_lag):

        τ = τ_rev·(2π/Ω),   τ_rev = 0.7/λ_h  revolutions,   λ_h = v_h/(ΩR)

    so τ = 0.7·2π·R/v_h ≈ 4.4·R/v_h, independent of rotor speed. Johnson identified τ = 14
    revolutions on the Dauphin (λ_h = 0.05; 2.4 s) and scaled it as R/v_h to get 9
    revolutions for the V-22 (pp. 15–16). No small-rotor identification exists.
    """
    return sp.Rational(7, 10) / lam_h * 2 * sp.pi / Omega


def vrs_fluctuating_increment(delta_vrs: sp.Expr | float, c_n: sp.Expr | float,
                              n: sp.Expr | float) -> sp.Expr:
    """
    Unsteady vortex ring state as a fluctuation of the VRS inflow increment itself:

        Δ(V_z + v)(t) = Δ(V_z + v)·(1 + c_n·n(t))

    n(t): zero-mean, unit-variance, low-frequency noise, INDEPENDENT for each rotor; c_n: the
    relative fluctuation of the recirculating flow. Through the linear thrust law
    (spec.inflow.inflow_thrust_ratio) the thrust fluctuation then scales with the blade's
    inflow sensitivity s, as the mean loss does — and it switches on, peaks and switches off
    with the mean VRS increment, forward speed included.
    DERIVED closure, not a published model. What it rests on (tools/wake_triage.py):
      - rotors with a larger mean loss fluctuate more (Veismann 2023 §4; Veismann 2022): the
        rms over the mean loss at the worst descent rate is 0.33–0.46 on their 4–6 inch
        rotors;
      - the fluctuation map of an isolated 14-inch rotor (Foster & Hartman 2017, Fig. 9) has
        the footprint of Johnson's unstable region: peak at V_x/v_h ≈ 0.4–0.5,
        V_z/v_h ≈ −0.9 to −1.2, gone by V_x/v_h ≈ 1 and V_z/v_h ≈ −1.5;
      - the attitude oscillation of a free-flying quadrotor in descent is explained by
        "asynchronous thrust fluctuations" of its rotors (Veismann et al., VFS Forum 77).
    Known misfit: the measured rms rises near-linearly from hover and is still 0.06–0.08 of
    the mean at a descent ratio of 2, where this envelope has returned to zero.
    Spectrum of n: isolated rotors show reduced frequencies f·R/v_h of 0.02–0.1 (Foster &
    Hartman: 1–3 Hz on a 14-inch rotor; Stack et al.: periods of 20–50 revolutions); on a
    flying 5-inch quadrotor the forcing was flat to 80 Hz and several times the isolated-
    rotor level (Bogaerts 2023). No source gives the rotor-to-rotor correlation.
    """
    return sp.sympify(delta_vrs * (1 + c_n * n))


def vrs_thrust_modulation(A: sp.Expr | float, omega: sp.Expr | float, t: sp.Expr | float,
                          delta: sp.Expr | float) -> sp.Expr:
    """
    Published stand-in for the same effect, a thrust multiplier computed "separately for
    each of the motors":

        C_T,VRS / C_T(μ_x, μ_z) = 1 + A_VRS·cos(ω_VRS·t + δ(t))

    A_VRS: a lookup of the measured relative standard deviation (their Fig. 9, up to 0.18);
    ω_VRS: a constant "chosen to roughly match" the observed 1–3 Hz; δ(t): a random-walk
    phase, independent per motor. The authors call it a prototype that "is not based on the
    changes to induced velocity that more accurately drive real VRS performance".
    Source: Foster & Hartman, "High-Fidelity Multi-Rotor Unmanned Aircraft System Simulation
    Development for Trajectory Prediction Under Off-Nominal Flight Dynamics", AIAA 2017
    (NASA NTRS 20170006073), Eq. (8), read from the paper.
    """
    return sp.sympify(1 + A * sp.cos(omega * t + delta))
