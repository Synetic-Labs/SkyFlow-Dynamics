"""
Motor / actuator dynamics — rotor speed Ω as a first-class state.

Verified sources: RotorPy motor model (first-order lag, Forster 2015 identification);
Crazyflow first-principles dynamics (asymmetric spin-up/down); SkyDreamer (throttle curve);
rpg_flightning (exact-exponential discretization).
"""

import sympy as sp


def first_order_lag(W: sp.Matrix, W_c: sp.Matrix, tau_m: sp.Expr | float) -> sp.Matrix:
    """
    Ω̇ᵢ = (Ω_cᵢ − Ωᵢ) / τ_m   — the standard identified motor model.
    Identified τ_m: Crazyflie 0.072 s, brushless ~0.05 s.
    """
    return (W_c - W) / tau_m


def asymmetric_lag(W: sp.Matrix, W_c: sp.Matrix,
                   ka1: sp.Expr | float, ka2: sp.Expr | float, kd1: sp.Expr | float, kd2: sp.Expr | float) -> sp.Matrix:
    """
    Crazyflow's asymmetric spin-up/spin-down motor model (motors brake slower than they
    accelerate — un-powered deceleration relies on aero drag):

        Ω̇ = ka1·(Ω_c − Ω) + ka2·(Ω_c² − Ω²)   if Ω_c > Ω   (spin-up)
        Ω̇ = kd1·(Ω_c − Ω) + kd2·(Ω_c² − Ω²)   otherwise    (spin-down)

    Reduces exactly to first_order_lag with (ka1, ka2, kd1, kd2) = (1/τ_m, 0, 1/τ_m, 0).
    ⚠ Crazyflow identifies this model on RPM-valued states. Because Ω̇ rescales along with Ω
    (unlike the thrust polynomial, whose LHS is in N), the conversion is one factor lower per
    Ω-power: ka1/kd1 (1/s) carry over UNCHANGED; ka2/kd2 convert by ×60/2π (i.e.
    (60/2π)^(Ω-power − 1)).
    Source: crazyflow/dynamics/first_principles/dynamics.py:115–119 + params.toml.
    """
    def elem(w, wc):
        up = ka1 * (wc - w) + ka2 * (wc**2 - w**2)
        dn = kd1 * (wc - w) + kd2 * (wc**2 - w**2)
        return sp.Piecewise((up, wc > w), (dn, True))
    return sp.Matrix([elem(W[i], W_c[i]) for i in range(W.shape[0])])


def exact_exp_step(W0: sp.Matrix, W_c: sp.Matrix, dt: sp.Expr | float, tau_m: sp.Expr | float) -> sp.Matrix:
    """
    Closed-form (exact) discretization of the linear first-order lag over a step where Ω_c is
    held constant (zero-order hold):

        Ω(dt) = Ω_c + (Ω₀ − Ω_c) · e^(−dt/τ_m)

    Unconditionally stable and monotone for any dt. In a differentiable simulator the per-step
    sensitivity ∂Ω(dt)/∂Ω₀ = e^(−dt/τ_m) ∈ (0,1) is a contraction, unlike the explicit-Euler
    factor (1 − dt/τ_m) which exceeds 1 in magnitude once dt > 2τ_m.
    Implemented for the linear lag only. (The asymmetric branches do admit a logistic/Riccati
    closed form under ZOH — the active branch never switches within a step since Ω cannot
    cross Ω_c — but it is not provided here.)
    Source: rpg_flightning (Heeg, Song, Scaramuzza, ICRA 2025).
    """
    return W_c + (W0 - W_c) * sp.exp(-dt / tau_m)


def throttle_to_speed(u: sp.Expr | float, W_min: sp.Expr | float, W_max: sp.Expr | float, k: sp.Expr | float) -> sp.Expr:
    """
    Normalized throttle u ∈ [0,1] → steady-state rotor speed (the ESC+battery command path):

        Ω_c = (Ω_max − Ω_min) · √(k·u² + (1−k)·u) + Ω_min,   k ∈ [0,1]

    k = 1 is a linear speed map; k = 0 is a square-root map (thrust-linear ESC). Endpoints are
    exact: u=0 → Ω_min, u=1 → Ω_max; monotone in u on [0,1] for k ∈ [0,1].
    Identified k = 0.5 (Ω_min = 341.75, Ω_max = 3100 rad/s) for a 5-inch racer.
    Source: SkyDreamer (arXiv:2510.14783) motor model + implementation lines 253–256.
    """
    return (W_max - W_min) * sp.sqrt(k * u**2 + (1 - k) * u) + W_min


def pwm_quantize(u: sp.Expr | float, pwm_min: sp.Expr | float, pwm_max: sp.Expr | float) -> sp.Expr:
    """
    ESC PWM quantization: snap normalized throttle to the integer PWM grid,
        u_q = round(u · (pwm_max − pwm_min)) / (pwm_max − pwm_min).
    Tie-breaking: written as floor(x + ½), i.e. round-half-UP — differs from banker's
    rounding (numpy/python round) exactly at half-integer ties.
    ⚠ Piecewise-constant (zero gradient a.e.) — exclude from differentiable paths; treat as a
    harness-side command transformation. Source: Crazyflow params.toml (pwm 7000…65535).
    """
    levels = pwm_max - pwm_min
    return sp.floor(u * levels + sp.Rational(1, 2)) / levels


def voltage_to_rpm(V: sp.Expr | float, b0: sp.Expr | float, b1: sp.Expr | float) -> sp.Expr:
    """
    Battery/ESC supply voltage → achievable rotor speed (RPM in the identified source):
        Ω_max(V) = b0 + b1·V.
    Identified (Crazyflow cf2x_L250): vmotor2rpm = [2968.18, 6647.95] (RPM). Field magnitude:
    SkyDreamer measured Ω_max 3200 → 2200 rad/s (−30%) over one battery discharge.
    Used as a multiplicative cap Ω_max(t) on the motor command — a slow parameter drift, not a
    fast dynamic; the battery state itself (SoC, sag) is a harness-side model.
    """
    return sp.sympify(b0 + b1 * V)
