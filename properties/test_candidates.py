"""Symbolic checks for candidate-tier terms: limiting behavior, reductions to verified
terms, identities against their defining equations, and identified-parameter sanity."""

import numpy as np
import sympy as sp

from spec import atmosphere, ground_effect, inflow, motor_electrical, rotor_aero, wind
from spec.frames import EZ

z, R, d, b, V, v_i = sp.symbols("z R d b V v_i", positive=True)


# ---------------- ground effect / downwash ----------------

def test_cheeseman_bennett_limits():
    ratio = ground_effect.cheeseman_bennett(z, R)
    assert sp.limit(ratio, z, sp.oo) == 1                      # OGE recovery
    assert float(ratio.subs({R: 1, z: 1})) > 1                 # IGE gain
    # 1.6% at z = 2R (the published validity anchor).
    assert abs(float(ratio.subs({R: 1, z: 2})) - 1 / (1 - 1 / 64)) < 1e-12


def test_cheeseman_bennett_forward_washout():
    ratio = ground_effect.cheeseman_bennett_forward(z, R, V, v_i)
    assert sp.limit(ratio, V, sp.oo) == 1                      # washes out with speed
    hover = ground_effect.cheeseman_bennett(z, R)
    assert sp.simplify(ratio.subs(V, 0) - hover) == 0          # reduces to hover form


def test_sanchez_cuevas_reduces_to_cheeseman_bennett():
    K_b = sp.Symbol("K_b", nonnegative=True)
    quad = ground_effect.sanchez_cuevas(z, R, d, b, K_b)
    single = ground_effect.cheeseman_bennett(z, R)
    assert sp.simplify(sp.limit(sp.limit(quad, d, sp.oo), b, sp.oo) - single) == 0
    # Interference terms make the multirotor ratio exceed the single-rotor one.
    vals = {z: 0.24, R: 0.12, d: 0.3, b: 0.3 * np.sqrt(2), K_b: 2.0}
    assert float(quad.subs(vals)) > float(single.subs(vals))


def test_pybullet_ground_effect_is_linearized_cb():
    # T·CB(z) − T = T·(R/4z)² + O((R/4z)⁴): the additive increment with G = 1 is the
    # first-order series of the multiplicative model.
    T = sp.Symbol("T", positive=True)
    eps = sp.Symbol("epsilon", positive=True)          # (R/4z)²
    exact_gain = T / (1 - eps) - T
    additive = ground_effect.pybullet_ground_effect(T, z, R, 1).subs((R / (4 * z))**2, eps)
    assert sp.simplify(sp.series(exact_gain, eps, 0, 2).removeO() - additive) == 0


def test_pybullet_downwash_shape():
    c1, c2, c3 = 2267.18, 0.16, -0.11
    dz_s = sp.Symbol("dz", positive=True)
    dxy_s = sp.Symbol("dxy", nonnegative=True)
    F = ground_effect.pybullet_downwash_force(dz_s, dxy_s, 0.0231348, c1, c2, c3)
    fn = sp.lambdify((dz_s, dxy_s), F)
    # Peak directly below; decays with lateral offset; ~29% of CF2 weight at 1 m (sanity
    # anchor from the source audit).
    f0 = fn(1.0, 0.0)
    assert abs(-f0 / 0.265 - 0.286) < 0.02
    assert abs(fn(1.0, 0.5)) < abs(f0)


def test_talbot_inflow_factor_limits():
    h, kge, hs = sp.symbols("h k_ge h_s", positive=True)
    f = ground_effect.talbot_inflow_factor(h, kge, hs, 1)
    assert sp.limit(f, h, sp.oo) == 1                          # OGE recovery
    # Monotone rising toward 1 with height (inflow recovers as ground recedes).
    assert sp.simplify(sp.diff(f, h) - kge * sp.exp(-kge * (h + hs))) == 0
    # load = 0 is the forward-flight kill-switch: exactly OGE.
    assert sp.simplify(ground_effect.talbot_inflow_factor(h, kge, hs, 0) - 1) == 0
    # Near the ground the inflow is REDUCED (f < 1) — that's the thrust-gain mechanism.
    assert 0 < float(f.subs({h: 1e-12, kge: 0.2, hs: 0.3})) < 1


def test_jain_wake_thrust_scaling_and_decay():
    T, rho, A, L, z0 = sp.symbols("T rho A L z0", positive=True)
    r_ = sp.Symbol("r", nonnegative=True)
    Vw = ground_effect.jain_wake_velocity(z, r_, T, rho, A, L, z0, 4.672, 60.808)
    assert sp.limit(Vw, z, sp.oo) == 0                          # far-field decay
    # Scales with the momentum-theory induced velocity: V ∝ √T.
    assert sp.simplify(Vw.subs(T, 4 * T) - 2 * Vw) == 0


# ---------------- inflow ----------------

def test_induced_velocity_hover_and_momentum_identity():
    T, rho, A = sp.symbols("T rho A", positive=True)
    v_h = inflow.hover_induced_velocity(T, rho, A)
    assert sp.simplify(v_h - sp.sqrt(T / (2 * rho * A))) == 0
    # The axial closed form at V_a = 0 reduces to v_h…
    vi0 = inflow.induced_velocity_axial(0, T, rho, A)
    assert sp.simplify(sp.piecewise_fold(vi0 - v_h)) == 0
    # …and satisfies the momentum equation T = 2ρA(V_a + v_i)v_i on the S > 0 branch.
    Va = sp.Symbol("V_a", positive=True)
    vi = ((-Va + sp.sqrt(Va * sp.Abs(Va) + 2 * T / (rho * A))) / 2)
    residual = 2 * rho * A * (Va + vi) * vi - T
    assert sp.simplify(residual.rewrite(sp.Piecewise)) == 0


def test_oblique_thrust_hover_limit():
    rho, A = sp.symbols("rho A", positive=True)
    vi = sp.Symbol("v_i2", positive=True)
    T = inflow.oblique_momentum_thrust(vi, sp.Matrix([0, 0, 0]), rho, A)
    assert sp.simplify(T - 2 * rho * A * vi**2) == 0


def test_dynamic_inflow_is_first_order_lag():
    nu, nu_eq, tau = sp.symbols("nu nu_eq tau", positive=True)
    from spec.motor import exact_exp_step
    rate = inflow.dynamic_inflow_lag(nu, nu_eq, tau)
    assert sp.simplify(rate - (nu_eq - nu) / tau) == 0
    # Exact-exp step applies verbatim (same ODE structure as the verified motor lag).
    dt = sp.Symbol("dt", positive=True)
    step = exact_exp_step(sp.Matrix([nu]), sp.Matrix([nu_eq]), dt, tau)[0]
    assert sp.simplify(sp.diff(step, dt) - (nu_eq - step) / tau) == 0


# ---------------- atmosphere / propeller ----------------

def test_isa_sea_level_and_monotone():
    assert abs(float(atmosphere.density(atmosphere.P0, atmosphere.T0)) - 1.225) < 1e-3
    h = sp.Symbol("h", positive=True)
    rho_h = atmosphere.density(atmosphere.pressure_gradient_layer(h),
                               atmosphere.temperature_troposphere(h))
    f = sp.lambdify(h, rho_h)
    hs = np.linspace(1, 10000, 50)
    vals = f(hs)
    assert np.all(np.diff(vals) < 0)                            # density falls with altitude
    assert abs(f(5000) / f(1e-6) - 0.601) < 0.01                # ~0.6 at 5 km (ISA table)
    a0 = float(atmosphere.speed_of_sound(atmosphere.T0))
    assert abs(a0 - 340.3) < 0.5


def test_propeller_thrust_rho_and_speed_scaling():
    C_T = sp.Function("C_T")
    n, D, rho, Va = sp.symbols("n D rho V_a", positive=True)
    J = atmosphere.advance_ratio(Va, n, D)
    T = atmosphere.propeller_thrust(C_T, J, rho, n, D)
    # Fixed J (scale V with n): T scales as ρ·n²·D⁴ exactly.
    T2 = atmosphere.propeller_thrust(C_T, atmosphere.advance_ratio(2 * Va, 2 * n, D),
                                     3 * rho, 2 * n, D)
    assert sp.simplify(T2 - 12 * T) == 0


# ---------------- rotor aero candidates ----------------

def test_rolling_moment_properties():
    mu_R = sp.Symbol("mu_R", positive=True)
    W = sp.Symbol("W", positive=True)
    vi_ = sp.Matrix(sp.symbols("v1 v2 v3", real=True))
    M = rotor_aero.rolling_moment(W, 1, vi_, mu_R)
    assert M[2] == 0                                           # in-plane torque only
    assert sp.simplify(M + rotor_aero.rolling_moment(W, -1, vi_, mu_R)) == sp.zeros(3, 1)
    assert rotor_aero.rolling_moment(W, 1, sp.Matrix([0, 0, 5]), mu_R) == sp.zeros(3, 1)


def test_flapping_force_pairwise_cancellation():
    # Kai Remark 1: at equal thrust the spin-signed terms cancel for counter-rotating pairs.
    T = sp.Symbol("T", positive=True)
    cav, cbv, caw, cbw = sp.symbols("c_av c_bv c_aw c_bw", positive=True)
    vi_ = sp.Matrix(sp.symbols("v1 v2 v3", real=True))
    w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
    F_ccw = rotor_aero.flapping_force_kai(T, 1, vi_, w, cav, cbv, caw, cbw)
    F_cw = rotor_aero.flapping_force_kai(T, -1, vi_, w, cav, cbv, caw, cbw)
    total = sp.simplify(F_ccw + F_cw)
    # Spin-signed terms gone; what remains is the unsigned drag/damping part, doubled.
    expected = sp.simplify(2 * (-sp.sqrt(T) * cav * sp.Matrix([vi_[0], vi_[1], 0])
                                + sp.sqrt(T) * caw * (EZ.cross(w))))
    assert sp.simplify(total - expected) == sp.zeros(3, 1)


def test_flapping_moment_body_rate_is_dissipative():
    W = sp.Symbol("W", positive=True)
    k = sp.Symbol("k_fw", positive=True)
    w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
    M = rotor_aero.flapping_moment_body_rate(W, w, k)
    assert M[2] == 0                                           # roll/pitch damping only
    # ω·M = −k·Ω·(ω_x² + ω_y²) ≤ 0: strictly dissipative on the in-plane rates.
    power = sp.expand((w.T * M)[0, 0])
    assert sp.simplify(power + k * W * (w[0]**2 + w[1]**2)) == 0
    # Spin-sign-free: a counter-rotating pair at equal speed DOUBLES the moment (contrast
    # rolling_moment, which cancels pairwise).
    assert sp.simplify(2 * M - (M + rotor_aero.flapping_moment_body_rate(W, w, k))) \
        == sp.zeros(3, 1)


def test_bramwell_torque_energy_identities():
    T, H, rho_, bl, ch, R_, Om, delta = sp.symbols(
        "T H rho b_l c_h R_r Omega delta", positive=True)
    vi3, vc = sp.symbols("v_i3 v_c", positive=True)
    P_profile_over_Om = rho_ * bl * ch * delta * (Om * R_)**2 * R_**2 / 8
    # Hover (μ = 0, λ = −v_i/ΩR): Q·Ω = P_profile + T·v_i — momentum-theory induced power.
    lam_h = -vi3 / (Om * R_)
    Q_h = rotor_aero.bramwell_torque(T, H, 0, lam_h, rho_, bl, ch, R_, Om, delta)
    assert sp.simplify(Q_h * Om - (P_profile_over_Om * Om + T * vi3)) == 0
    # Climb by v_c adds exactly the climb power T·v_c.
    lam_c = -(vc + vi3) / (Om * R_)
    Q_c = rotor_aero.bramwell_torque(T, H, 0, lam_c, rho_, bl, ch, R_, Om, delta)
    assert sp.simplify((Q_c - Q_h) * Om - T * vc) == 0
    # Forward flight grows the profile term as (1 + 4.5μ²); H = 0 isolates it.
    mu_ = sp.Rational(3, 10)
    Q_mu = rotor_aero.bramwell_torque(T, 0, mu_, lam_h, rho_, bl, ch, R_, Om, delta)
    Q_0 = rotor_aero.bramwell_torque(T, 0, 0, lam_h, rho_, bl, ch, R_, Om, delta)
    assert sp.simplify((Q_mu - Q_0) - P_profile_over_Om * sp.Rational(9, 2) * mu_**2) == 0
    # Autorotation: descent drives λ positive and the torque through zero.
    lam_auto = P_profile_over_Om / (T * R_)
    assert sp.simplify(rotor_aero.bramwell_torque(
        T, 0, 0, lam_auto, rho_, bl, ch, R_, Om, delta)) == 0


def test_blade_profile_drag_polar():
    a, sig = sp.symbols("a sigma", positive=True)
    CT = sp.Symbol("C_T", positive=True)
    d0 = rotor_aero.blade_profile_drag(0, a, sig)
    assert abs(float(d0) - 0.009) < 1e-15                      # bare profile drag
    # Quadratic growth in blade loading: the increment scales 4× when C_T doubles.
    inc1 = rotor_aero.blade_profile_drag(CT, a, sig) - d0
    inc2 = rotor_aero.blade_profile_drag(2 * CT, a, sig) - d0
    assert sp.simplify(inc2 - 4 * inc1) == 0


# ---------------- motor / battery electrical ----------------

def test_dc_motor_linearization_recovers_tau_m():
    W0, J_r, Kq, Ke, Ra, km, b_ = sp.symbols("W0 J_r K_q K_e R_a k_m b", positive=True)
    W = sp.Symbol("W", positive=True)
    Vm = sp.Symbol("V_m", positive=True)
    rate = motor_electrical.dc_motor_speed_dynamics(W, Vm, J_r, Kq, Ke, Ra, km, b_)
    # ∂Ω̇/∂Ω at Ω₀ = −1/τ_m with τ_m = J_r/(K_qK_e/R_a + b + 2k_mΩ₀)  (the parameter bridge).
    slope = sp.diff(rate, W).subs(W, W0)
    tau_m = J_r / (Kq * Ke / Ra + b_ + 2 * km * W0)
    assert sp.simplify(slope + 1 / tau_m) == 0


def test_steady_state_speed_is_equilibrium():
    u, Vb, Kq, Ke, Ra, km, b_ = sp.symbols("u V_b K_q K_e R_a k_m b", positive=True)
    Wss = motor_electrical.steady_state_speed(u, Vb, Kq, Ke, Ra, km, b_)
    balance = (Kq / Ra) * (u * Vb - Ke * Wss) - km * Wss**2 - b_ * Wss
    assert sp.simplify(balance) == 0


def test_crazyflie_thrust_curve_sane():
    # CF2 2.1+ defaults: monotone increasing over the working voltage range, plausible
    # magnitudes (THRUST_MAX = 0.12 N per motor at ~4.2 V).
    C = (-0.02476537915958403, 0.06523793527519485,
         -0.026792504967750107, 0.006776789303971145)
    v = sp.Symbol("v", positive=True)
    T = motor_electrical.crazyflie_thrust_from_voltage(v, *C)
    f = sp.lambdify(v, T)
    grid = np.linspace(1.0, 4.2, 100)
    vals = f(grid)
    assert np.all(np.diff(vals) > 0)
    # THRUST_MAX = 0.12 N per motor is reached near v_m ≈ 3.1 V (mean voltage under load,
    # not the full 4.2 V battery voltage).
    assert 0.10 < f(3.1) < 0.14


def test_chen_ocv_shape():
    soc = np.linspace(0.02, 1.0, 200)
    f = sp.lambdify(sp.Symbol("SOC"), motor_electrical.chen_ocv(sp.Symbol("SOC")))
    v = f(soc)
    assert np.all(np.diff(v) > 0)          # monotone in SoC
    assert 3.6 < f(1.0) < 4.3 and 2.4 < f(0.02) < 3.6


def test_thevenin_reduces_to_static_sag():
    # With RC states at equilibrium under constant current, V = OCV − i·(R_S + R_TS + R_TL):
    # the Gazebo linear model's structure.
    SOC, i = sp.symbols("SOC i", positive=True)
    R_S, R_TS, C_TS, R_TL, C_TL, C_cap = sp.symbols("R_S R_TS C_TS R_TL C_TL C_cap",
                                                    positive=True)
    V_OC = sp.Function("V_OC")
    _, vts_dot, vtl_dot, v_batt = motor_electrical.thevenin_battery(
        SOC, i, i * R_TS, i * R_TL, V_OC, lambda s_: R_S, R_TS, C_TS, R_TL, C_TL, C_cap)
    assert sp.simplify(vts_dot) == 0 and sp.simplify(vtl_dot) == 0
    assert sp.simplify(v_batt - (V_OC(SOC) - i * (R_S + R_TS + R_TL))) == 0


# ---------------- wind / turbulence ----------------

def test_dryden_filter_magnitude_matches_psd():
    # |H_v(jω)|² must equal the published temporal PSD (÷2π convention aside):
    # σ²(L/πV)(1 + 3(Lω/V)²)/(1 + (Lω/V)²)² — the √3 zero is exactly what squares to the 3.
    sigma, L, V_ = sp.symbols("sigma L V", positive=True)
    omega = sp.Symbol("omega", positive=True)
    H = wind.dryden_filter_vw(sigma, L, V_).subs(wind.s, sp.I * omega)
    mag2 = sp.simplify(sp.Abs(H)**2)
    x = L * omega / V_
    expected = sigma**2 * (L / (sp.pi * V_)) * (1 + 3 * x**2) / (1 + x**2)**2
    assert sp.simplify(mag2 - expected) == 0


def test_dryden_low_altitude_anchors():
    # The fit's built-in anchor: at h = 1000 ft, 0.177 + 0.000823·1000 = 1.0 exactly, so
    # L_u = L_v = h and σ_u = σ_v = σ_w. (The 1000–2000 ft band then interpolates toward the
    # mid-altitude L = 1750 ft constant.)
    L_u, L_v, L_w, s_u, s_v, s_w = wind.dryden_low_altitude_scales(1000.0, 30.0)
    assert abs(float(L_u) - 1000.0) < 1e-9
    assert float(L_w) == 1000.0
    assert abs(float(s_w) - 3.0) < 1e-12          # 0.1·W20
    assert abs(float(s_u) - float(s_w)) < 1e-9
    # Below 1000 ft the u/v intensity exceeds the vertical one.
    _, _, _, s_u2, _, s_w2 = wind.dryden_low_altitude_scales(100.0, 30.0)
    assert float(s_u2) > float(s_w2)


def test_von_karman_psd_limits():
    sigma, L = sp.symbols("sigma L", positive=True)
    Om = sp.Symbol("Omega", positive=True)
    psd = wind.von_karman_psd_u(Om, sigma, L)
    assert sp.simplify(psd.subs(Om, 0) - sigma**2 * 2 * L / sp.pi) == 0
    assert sp.limit(psd, Om, sp.oo) == 0


def test_one_minus_cosine_gust_continuity():
    x, Vm, dm = sp.symbols("x V_m d_m", positive=True)
    g = wind.one_minus_cosine_gust(x, Vm, dm)
    at0 = g.subs(x, 0)
    assert sp.simplify(sp.piecewise_fold(at0)) == 0 or at0 == 0
    assert sp.simplify(g.subs(x, dm) - Vm) == 0 or sp.simplify(
        (Vm / 2 * (1 - sp.cos(sp.pi))) - Vm) == 0