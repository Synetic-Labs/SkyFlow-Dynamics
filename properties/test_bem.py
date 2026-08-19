"""Accuracy: symbolic checks for the BEM terms (NeuroBEM / agilicious intake) and the
frame-drag / integrator terms landed with them — INTAKE step 4."""

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import bem
from skyflow_dynamics.spec.discretization import semi_implicit_euler_step
from skyflow_dynamics.spec.inflow import hover_induced_velocity, oblique_momentum_thrust
from skyflow_dynamics.spec.rotor_aero import cubic_drag, per_axis_quadratic_drag

r, psi, R = sp.symbols("r psi R", positive=True)
Omega, rho, A, b = sp.symbols("Omega rho A b", positive=True)
v_hor = sp.symbols("v_hor", nonnegative=True)
v_ver, chi = sp.symbols("v_ver chi", real=True)
v_i, v_h, T, H, Q = sp.symbols("v_i v_h T H Q", positive=True)
theta0, theta1, c_root, c_tip, cl0, cd0 = sp.symbols(
    "theta0 theta1 c_root c_tip cl0 cd0", positive=True)
a0, a1, b1, spin, k_beta, alpha = sp.symbols("a0 a1 b1 s k_beta alpha", real=True)

PARAMS = (R, theta0, theta1, c_root, c_tip, cl0, cd0, b, rho)


def test_section_velocities_match_advance_ratio_form():
    """Code writes U_T = Ω(r + Rμ·sinψ) with μ = v_hor/(ΩR); paper eq. (6) writes
    U_T = Ωr + v_hor·sinψ. The spec adopts the paper form — prove they are identical."""
    U_T, _ = bem.blade_section_velocities(r, psi, Omega, v_hor, v_ver, v_i, a0, a1, b1)
    mu = v_hor / (Omega * R)
    assert sp.simplify(U_T - Omega * (r + R * mu * sp.sin(psi))) == 0


def test_hover_hforce_vanishes():
    """At hover (v_hor = v_ver = 0, zero flapping) the element loads are azimuth-independent,
    so the H integrand carries a bare sinψ and integrates to zero over the disk."""
    _, _, dH = bem.blade_element_integrands(
        r, psi, Omega, 0, 0, v_i, *PARAMS, eps_camber=sp.Rational(7, 100))
    assert sp.integrate(dH, (psi, 0, 2 * sp.pi)) == 0


def test_hover_closure_reduces_to_hover_induced_velocity():
    """At v_hor = v_ver = 0 the closure residual becomes T − 2ρA·v_i², whose positive root is
    spec.inflow.hover_induced_velocity."""
    g = bem.momentum_closure_residual(v_i, T, 0, 0, rho, A)
    assert sp.simplify(g - (T - 2 * rho * A * v_i**2)) == 0
    (root,) = [s for s in sp.solve(g, v_i) if s.is_positive]
    assert sp.simplify(root - hover_induced_velocity(T, rho, A)) == 0


def test_momentum_closure_is_oblique_momentum_thrust():
    """Structural tie: the momentum side of the closure IS the oblique momentum-thrust term
    with V = (v_hor, 0, v_ver) — guards against the two modules drifting apart."""
    g = bem.momentum_closure_residual(v_i, T, v_hor, v_ver, rho, A)
    T_mom = oblique_momentum_thrust(v_i, sp.Matrix([v_hor, 0, v_ver]), rho, A)
    assert sp.simplify(g - (T - T_mom)) == 0


def test_vrs_continuous_with_hover():
    """The empirical VRS polynomial returns v_h at zero descent rate."""
    assert sp.simplify(bem.vrs_induced_velocity(0, v_h) - v_h) == 0


def test_polar_symmetries_and_bound():
    """cl is odd and cd even in α (zero camber); |cl| ≤ cl0·(½ + ε_c) everywhere — the polars
    saturate (stall) instead of growing linearly."""
    eps = sp.Rational(7, 100)
    cl = bem.lift_coefficient(alpha, cl0, 0)
    cd = bem.drag_coefficient(alpha, cd0)
    assert sp.simplify(cl.subs(alpha, -alpha) + cl) == 0
    assert sp.simplify(cd.subs(alpha, -alpha) - cd) == 0
    cl_c = bem.lift_coefficient(alpha, cl0, eps)
    # cl0·(sin·cos + ε) = cl0·(sin(2α)/2 + ε) ⇒ extremal value cl0·(1/2 + ε)
    assert sp.simplify(cl_c.rewrite(sp.sin) - cl0 * (sp.sin(2 * alpha) / 2 + eps)) == 0
    assert sp.maximum(cl_c, alpha, sp.Interval(0, 2 * sp.pi)) == cl0 * (sp.Rational(1, 2) + eps)


def test_integrands_scale_quadratically_with_flow_speed():
    """Element loads go as U² — scaling (Ω, v_hor, v_ver, v_i) by k scales dT, dQ, dH by k²
    (dimensional consistency of the velocity content)."""
    k = sp.symbols("k", positive=True)
    base = bem.blade_element_integrands(r, psi, Omega, v_hor, v_ver, v_i, *PARAMS)
    scaled = bem.blade_element_integrands(
        r, psi, k * Omega, k * v_hor, k * v_ver, k * v_i, *PARAMS)
    for d, dk in zip(base, scaled):
        assert sp.simplify(dk - k**2 * d) == 0


def test_tpp_reduces_to_pure_thrust_and_yaw_torque():
    """Zero flapping and zero H recover the verified-tier composition: f = (0,0,T) and
    τ = −s·Q·ẑ (yaw reaction opposing the physical spin — canonical convention)."""
    f = bem.tpp_rotor_force(T, 0, chi, spin, 0, 0, 0)
    tau = bem.tpp_rotor_torque(Q, chi, spin, k_beta, 0, 0)
    assert sp.simplify(f - sp.Matrix([0, 0, T])) == sp.zeros(3, 1)
    assert sp.simplify(tau - sp.Matrix([0, 0, -spin * Q])) == sp.zeros(3, 1)


def test_tpp_hforce_opposes_inplane_motion():
    """The H-force acts along −x̂_P, i.e. against the in-plane hub velocity direction
    (cos χ, sin χ): its power against that motion is −H·v_hor ≤ 0 (drag, dissipative)."""
    f = bem.tpp_rotor_force(0, H, chi, spin, 0, 0, 0)
    power = f.dot(sp.Matrix([v_hor * sp.cos(chi), v_hor * sp.sin(chi), 0]))
    assert sp.simplify(power + H * v_hor) == 0


def test_axis_drags_dissipative():
    """Per-axis quadratic and cubic drag oppose the airspeed axis-wise: F_k·v_k ≤ 0."""
    v = sp.Matrix(sp.symbols("vx vy vz", real=True))
    kq = sp.Matrix(sp.symbols("kqx kqy kqz", positive=True))
    for F in (per_axis_quadratic_drag(v, kq), cubic_drag(v, kq)):
        for Fk, vk in zip(F, v):
            assert sp.simplify(Fk * vk).is_nonpositive


def test_semi_implicit_euler_is_symplectic_and_consistent():
    """On the harmonic oscillator (ẋ = v, v̇ = −x) the one-step map is
    x⁺ = x + h·v⁺, v⁺ = v − h·x: area-preserving (det = 1 exactly — no energy drift) and
    first-order consistent with the exact flow."""
    h = sp.symbols("h", positive=True)
    x0, v0 = sp.symbols("x0 v0", real=True)
    s = np.array([x0, v0], dtype=object)

    def f(t, state):
        return np.array([state[1], -state[0]], dtype=object)

    out = semi_implicit_euler_step(f, s, h, vel_idx=np.array([1]), pos_idx=np.array([0]))
    assert sp.simplify(out[1] - (v0 - h * x0)) == 0
    assert sp.simplify(out[0] - (x0 + h * (v0 - h * x0))) == 0
    M = sp.Matrix([[sp.diff(out[0], x0), sp.diff(out[0], v0)],
                   [sp.diff(out[1], x0), sp.diff(out[1], v0)]])
    assert sp.simplify(M.det()) == 1
    exact = sp.Matrix([x0 * sp.cos(h) + v0 * sp.sin(h), v0 * sp.cos(h) - x0 * sp.sin(h)])
    err = sp.Matrix([out[0], out[1]]) - exact
    for e in err:
        assert sp.series(e, h, 0, 2).removeO() == 0  # local error O(h²)
