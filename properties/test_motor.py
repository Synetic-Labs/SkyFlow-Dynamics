"""Motor-model identities: reductions, closed forms, limits, and the throttle/PWM curves."""

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import motor
from skyflow_dynamics.spec.discretization import rk4_step


def test_asymmetric_reduces_to_first_order():
    W = sp.Matrix([sp.Symbol("W", nonnegative=True)])
    Wc = sp.Matrix([sp.Symbol("Wc", nonnegative=True)])
    tau = sp.Symbol("tau", positive=True)
    asym = motor.asymmetric_lag(W, Wc, 1 / tau, 0, 1 / tau, 0)[0]
    first = motor.first_order_lag(W, Wc, tau)[0]
    # Identical branches collapse: the Piecewise must reduce to the first-order rate exactly.
    assert sp.simplify(asym - first) == 0


def test_exact_exp_solves_the_ode():
    # d/dt Ω(t) == (Ω_c − Ω(t))/τ  with Ω(0) = Ω₀ — exactness, not approximation.
    t = sp.Symbol("t", nonnegative=True)
    tau = sp.Symbol("tau", positive=True)
    W0 = sp.Matrix([sp.Symbol("W0", nonnegative=True)])
    Wc = sp.Matrix([sp.Symbol("Wc", nonnegative=True)])
    Wt = motor.exact_exp_step(W0, Wc, t, tau)
    residual = sp.diff(Wt[0], t) - (Wc[0] - Wt[0]) / tau
    assert sp.simplify(residual) == 0
    assert sp.simplify(Wt[0].subs(t, 0) - W0[0]) == 0
    assert sp.limit(Wt[0], t, sp.oo) == Wc[0]


def test_exact_exp_euler_consistency():
    # Series in dt:  Ω(dt) = Ω₀ + dt·(Ω_c−Ω₀)/τ + O(dt²) — consistent with the ODE's Euler step.
    dt = sp.Symbol("dt", positive=True)
    tau = sp.Symbol("tau", positive=True)
    W0, Wc = sp.symbols("W0 Wc", nonnegative=True)
    Wt = motor.exact_exp_step(sp.Matrix([W0]), sp.Matrix([Wc]), dt, tau)[0]
    series = sp.series(Wt, dt, 0, 2).removeO()
    assert sp.simplify(series - (W0 + dt * (Wc - W0) / tau)) == 0


def test_exact_exp_gradient_is_contraction():
    # ∂Ω(dt)/∂Ω₀ = e^(−dt/τ) ∈ (0,1) for dt > 0 — the differentiable-sim stability property.
    dt = sp.Symbol("dt", positive=True)
    tau = sp.Symbol("tau", positive=True)
    W0, Wc = sp.symbols("W0 Wc", nonnegative=True)
    g = sp.diff(motor.exact_exp_step(sp.Matrix([W0]), sp.Matrix([Wc]), dt, tau)[0], W0)
    assert sp.simplify(g - sp.exp(-dt / tau)) == 0
    assert bool(g.subs({dt: 1, tau: sp.Rational(1, 100)}) > 0)  # far past Euler's dt=2τ limit


def test_rk4_matches_exact_exp_to_fourth_order():
    tau, Wc, W0, dt = 0.072, 2400.0, 1500.0, 0.01

    def f(t, w):
        return (Wc - w) / tau

    exact = Wc + (W0 - Wc) * np.exp(-dt / tau)
    rk4 = rk4_step(f, W0, dt)
    # One-step truncation ≈ |W0−Wc|·(dt/τ)⁵/5! ≈ 3.9e-4 here.
    assert abs(rk4 - exact) < 2 * abs(W0 - Wc) * (dt / tau) ** 5 / 120

    # Halving the step must shrink the error ~16× (4th order).
    def integrate(h, n):
        w = W0
        for _ in range(n):
            w = rk4_step(f, w, h)
        return w

    e1 = abs(integrate(dt, 1) - exact)
    e2 = abs(integrate(dt / 2, 2) - exact)
    assert 12 < e1 / e2 < 40


def test_asymmetric_crazyflow_values():
    # cf21B_500 identified set, expressed in rad/s units (ka1/kd1 carry over from the RPM
    # identification unchanged; ka2/kd2 were converted by 60/2π — one factor per Ω-power
    # minus one, since Ω̇ rescales too): spin-up strictly faster than spin-down for the same
    # |Δ| at hover-scale speeds.
    W = sp.Matrix([sp.Symbol("W")])
    Wc = sp.Matrix([sp.Symbol("Wc")])
    ka1, ka2, kd1, kd2 = 13.996, 0.00011093, 5.9332, 0.00031951
    expr = motor.asymmetric_lag(W, Wc, ka1, ka2, kd1, kd2)[0]
    up = float(expr.subs({W[0]: 1500.0, Wc[0]: 2000.0}))
    down = float(expr.subs({W[0]: 2000.0, Wc[0]: 1500.0}))
    assert up > 0 > down
    assert up > abs(down)


def test_throttle_curve_endpoints_and_monotonicity():
    u = sp.Symbol("u", nonnegative=True)
    Wmin, Wmax, k = sp.symbols("Wmin Wmax k", positive=True)
    curve = motor.throttle_to_speed(u, Wmin, Wmax, k)
    assert sp.simplify(curve.subs(u, 0) - Wmin) == 0
    assert sp.simplify(curve.subs(u, 1) - Wmax) == 0
    # Monotone on (0,1] for k ∈ [0,1]: dΩ/du > 0 on a dense numeric grid.
    d = sp.lambdify((u, Wmin, Wmax, k), sp.diff(curve, u))
    uu, kk = np.meshgrid(np.linspace(1e-6, 1.0, 101), np.linspace(0.0, 1.0, 11))
    assert np.all(d(uu, 341.75, 3100.0, kk) > 0)


def test_throttle_curve_skydreamer_identified():
    # k = 0.5, Ω ∈ [341.75, 3100]: reproduce the identified curve at u = 0.5 by hand.
    val = motor.throttle_to_speed(sp.Rational(1, 2), 341.75, 3100.0, 0.5)
    by_hand = (3100.0 - 341.75) * np.sqrt(0.5 * 0.25 + 0.5 * 0.5) + 341.75
    assert abs(float(val) - by_hand) < 1e-9


def test_pwm_quantization_grid():
    u = sp.Symbol("u", nonnegative=True)
    q = sp.lambdify(u, motor.pwm_quantize(u, 7000, 65535))
    levels = 65535 - 7000
    for uu in np.linspace(0, 1, 257):
        got = q(uu)
        assert abs(got - uu) <= 0.5 / levels + 1e-15   # nearest grid point
        assert abs(got * levels - round(got * levels)) < 1e-9  # exactly on the grid


def test_voltage_to_rpm_crazyflow_identified():
    # cf2x_L250: vmotor2rpm = [2968.18, 6647.95]; sane full-battery magnitude (~4.2 V).
    rpm = float(motor.voltage_to_rpm(4.2, 2968.18, 6647.95))
    assert abs(rpm - (2968.18 + 6647.95 * 4.2)) < 1e-9
    assert 25000 < rpm < 35000
