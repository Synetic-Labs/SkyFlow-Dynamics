"""A rotor in its own wake: descent through the vortex ring state, ground effect at constant
rotor speed, rotor-on-rotor interference (spec.descent, spec.inflow, spec.ground_effect;
REFERENCES.md "Own-wake aerodynamics", 2026-11-16).

Part 1 — candidate-tier checks: the published forms and constants, limits, continuity, and
the identities that tie the terms together (one inflow sensitivity s behind ground effect,
descent thrust loss and the k_z inflow term).

Part 2 — findings F-32 to F-34 and the triage sizes the registry quotes
(tools/wake_triage.py). The measured values there are figure and table read-offs; the
tolerances here are the sizes the ledger states, not a validation of the models."""

import math

import pytest
import sympy as sp

from skyflow_dynamics.spec import descent, ground_effect, inflow
from skyflow_dynamics.spec.parameters import CRAZYFLIE, RACER_5IN
from tools import wake_triage as wt

VZ = sp.Symbol("V_z", real=True)
C = descent.JOHNSON_VRS
AXIAL = descent.johnson_vrs_inflow_axial(VZ)
NET = sp.lambdify(VZ, VZ + AXIAL, "math")
DNET = sp.lambdify(VZ, sp.diff(VZ + AXIAL, VZ), "math")
EPS = 1e-9


# ---------------------------------------------------------------- Johnson's VRS inflow

def test_johnson_reproduces_table_4():
    """Net inflow minimum 0.85 at V_z = −0.45 and maximum 1.25 at −1.5, both with zero slope."""
    assert NET(float(C["VzN"])) == pytest.approx(0.85, abs=1e-12)
    assert NET(float(C["VzX"])) == pytest.approx(1.25, abs=1e-12)
    for z in (float(C["VzN"]), float(C["VzX"])):
        assert DNET(z - 1e-7) == pytest.approx(0, abs=1e-5)
        assert DNET(z + 1e-7) == pytest.approx(0, abs=1e-5)


def test_johnson_joins_momentum_theory():
    """Momentum theory above D and below B; hover value 1; continuous everywhere, and with
    a continuous slope at D, N, X and A (E and B match the value only, as the report says)."""
    for z in (0.5, 0.0, -0.1, -0.2):
        assert NET(z) - z == pytest.approx(float(descent.momentum_inflow_axial(z)), abs=1e-12)
    for z in (-2.1, -2.5, -4.0):
        assert NET(z) - z == pytest.approx(float(descent.momentum_inflow_axial(z, "windmill")),
                                           abs=1e-12)
    for key in ("VzD", "VzN", "VzX", "VzE", "VzB"):       # A coincides with X
        z = float(C[key])
        assert NET(z - EPS) == pytest.approx(NET(z + EPS), abs=1e-6), key
    for key in ("VzD", "VzN", "VzX"):
        z = float(C[key])
        assert DNET(z - 1e-7) == pytest.approx(DNET(z + 1e-7), abs=1e-4), key


def test_johnson_unstable_band_is_between_n_and_x():
    """d(V_z + v)/dV_z < 0 — thrust falls as descent gets faster — exactly between N and X."""
    for z in (-0.5, -0.8, -1.2, -1.45):
        assert DNET(z) < 0
    for z in (0.3, -0.1, -0.3, -0.4, -1.6, -1.9, -2.05, -3.0):
        assert DNET(z) > 0


def test_baseline_has_no_unstable_band():
    """f = 0: the baseline alone (singularity removed) has rising net inflow throughout —
    "With the baseline model, the helicopter is stable in vortex ring state"."""
    base = sp.lambdify(VZ, sp.diff(VZ + descent.johnson_vrs_inflow_axial(VZ, f=0), VZ), "math")
    assert all(base(z) > 0 for z in (0.2, -0.3, -1.0, -1.49, -1.6, -2.0, -2.09, -2.5))
    k = sp.Symbol("kappa", positive=True)
    assert sp.simplify(descent.johnson_vrs_inflow_axial(sp.Rational(-1), k)
                       - k * descent.johnson_vrs_inflow_axial(sp.Rational(-1))) == 0


def test_momentum_slope_is_the_implicit_derivative():
    v, z, x = sp.symbols("v z x", real=True)
    res = descent.momentum_inflow_residual(v, z, x)
    assert sp.simplify(-sp.diff(res, z) / sp.diff(res, v)
                       - descent.momentum_inflow_slope(v, z, x)) == 0
    for zz, branch in ((0.7, "helicopter"), (-1.0, "helicopter"), (-2.5, "windmill")):
        vv = descent.momentum_inflow_axial(sp.Float(zz), branch)
        assert float(descent.momentum_inflow_residual(vv, zz, 0)) == pytest.approx(0, abs=1e-12)


def test_transition_points_close_with_forward_speed():
    """V_x = 0 gives Table 4; N and X meet at V_x = V_xM, where the increment also vanishes."""
    t0 = descent.johnson_transition_points(0)
    assert (t0["A"], t0["B"], t0["D"], t0["N"], t0["X"], t0["E"]) == (
        C["VzA"], C["VzB"], C["VzD"], C["VzN"], C["VzX"], C["VzE"])
    tm = descent.johnson_transition_points(C["VxM"])
    assert tm["N"] == tm["X"] == (C["VzN"] + C["VzX"]) / 2
    assert wt.johnson_inflow(-1.0, 0.96) == pytest.approx(wt._momentum(-1.0, 0.96, "helicopter"))
    # the model as printed steps at V_xM: the slope conditions leave up to 0.15 v_h there
    left = max(wt.johnson_inflow(z, 0.9499999) - wt.johnson_baseline(z, 0.9499999)[0]
               for z in (-1.4, -1.3, -1.25, -1.2, -1.1))
    assert left == pytest.approx(0.147, abs=0.005)


def test_johnson_forward_speed_curve_matches_figure_37():
    """V_x/v_h = 0.75, read off the report's Fig. 37: maximum ≈ 0.85 near V_z = −1.1,
    minimum ≈ 0.58 near −0.55 (reading tolerance 0.03 / 0.05)."""
    zs = [-2.0 + 0.005 * i for i in range(400)]
    net = [z + wt.johnson_inflow(z, 0.75) for z in zs]
    i_max = max((i for i, z in enumerate(zs) if z < -0.8), key=lambda i: net[i])
    i_min = min((i for i, z in enumerate(zs) if z > -0.8), key=lambda i: net[i])
    assert net[i_max] == pytest.approx(0.85, abs=0.03) and zs[i_max] == pytest.approx(-1.1, abs=0.05)
    assert net[i_min] == pytest.approx(0.58, abs=0.03) and zs[i_min] == pytest.approx(-0.55, abs=0.05)


def test_johnson_numeric_and_explicit_forms_agree():
    for z in (0.4, -0.3, -0.9, -1.5, -1.8, -2.05, -2.6):
        assert wt.johnson_inflow(z, 1e-12, f=0.999999) == pytest.approx(
            float(wt.johnson_inflow(z)), rel=1e-5)


def test_inflow_time_constant_scales_with_radius_over_vh():
    lam, om = sp.symbols("lam Omega", positive=True)
    tau = descent.johnson_inflow_time_constant(lam, om)
    assert sp.simplify(tau * lam * om / (2 * sp.pi) - sp.Rational(7, 10)) == 0
    # Johnson's identification: 14 revolutions at λ_h = 0.05, 2.4 s at 360 rpm
    assert float(tau.subs({lam: 0.05, om: 360 * 2 * math.pi / 60})) == pytest.approx(2.33, abs=0.1)


def test_fluctuation_forms():
    d, c, n, a, w, t, ph = sp.symbols("d c n A w t ph", real=True)
    assert sp.diff(descent.vrs_fluctuating_increment(d, c, n), n) == c * d
    assert descent.vrs_fluctuating_increment(d, c, 0) == d
    assert descent.vrs_fluctuating_increment(0, c, n) == 0     # off outside the VRS band
    mod = descent.vrs_thrust_modulation(a, w, t, ph)
    assert mod.subs(a, 0) == 1 and sp.simplify(mod.subs({w * t + ph: 0}) - 1 - a) == 0


# ---------------------------------------------------------------- one inflow sensitivity

def test_sensitivity_is_the_blade_element_ratio():
    """C_T ∝ θ₀/3 − λ/2 gives T/T_static = 1 − s·(λ/λ_h − 1) with s = 3λ_h/(2θ₀ − 3λ_h)."""
    th, lam, lam_h = sp.symbols("theta lam lam_h", positive=True)
    ratio = (th / 3 - lam / 2) / (th / 3 - lam_h / 2)
    s = inflow.inflow_thrust_sensitivity(th, lam_h)
    assert sp.simplify(ratio - inflow.inflow_thrust_ratio(lam / lam_h, s)) == 0
    assert inflow.inflow_thrust_ratio(1, s) == 1


def test_ground_effect_forms_are_the_inflow_law():
    z, r, ca, cb, v, vi = sp.symbols("z R C_a C_b V v_i", positive=True)
    exp_form = inflow.inflow_thrust_ratio(
        1 - ground_effect.ground_inflow_fraction_exponential(z, r, cb), ca)
    assert sp.simplify(exp_form - ground_effect.he_leang_exponential(z, r, ca, cb)) == 0
    img_form = inflow.inflow_thrust_ratio(
        1 - ground_effect.ground_inflow_fraction_image(z, r, v, vi), ca)
    assert sp.simplify(img_form - ground_effect.cheeseman_bennett_fixed_pitch(z, r, ca, v, vi)) == 0
    # finite at the ground, 1 out of ground effect
    assert ground_effect.he_leang_exponential(0, r, ca, cb) == 1 + ca
    assert sp.limit(ground_effect.he_leang_exponential(z, r, ca, cb), z, sp.oo) == 1


def test_constant_power_form_is_unit_sensitivity_to_first_order():
    """Cheeseman–Bennett at constant power, 1/(1 − ε) ≈ 1 + ε, is the fixed-pitch form with
    s = 1: a real blade (measured s 0.27–0.63) gains less at fixed rotor speed."""
    z, r, eps = sp.symbols("z R eps", positive=True)
    series = sp.series(1 / (1 - eps), eps, 0, 2).removeO()
    cb = ground_effect.cheeseman_bennett(z, r)
    assert sp.simplify(cb - 1 / (1 - (r / (4 * z)) ** 2)) == 0
    assert sp.simplify(series.subs(eps, (r / (4 * z)) ** 2)
                       - ground_effect.cheeseman_bennett_fixed_pitch(z, r, 1)) == 0


def test_he_leang_published_values():
    """Three-blade 5×3 (Table 2: C_a 0.53, C_b 1.80) and the range of the eleven propellers."""
    gains = [float(ground_effect.he_leang_exponential(zr, 1, 0.53, 1.80)) - 1 for zr in (0.5, 1, 2)]
    assert gains == pytest.approx([0.215, 0.088, 0.014], abs=0.001)
    # their closed form, Eq. (10), with ς = C_lα·σ: the same number as s at the hover solution
    for th_deg, sigma, a in ((15.0, 0.0637, 2 * math.pi), (19.6, 0.149, 5.7), (9.3, 0.103, 5.7)):
        th, zeta = math.radians(th_deg), a * sigma
        root = math.sqrt(192 * zeta * th + 9 * zeta ** 2)
        c_a = (root - 3 * zeta) / (32 * th + 3 * zeta - root)
        ct = sp.Symbol("ct", positive=True)       # hover: C_T = ½σa(θ/3 − ½√(C_T/2)), Eq. (5)
        c_t = float(sp.nsolve(ct - sigma * a / 2 * (th / 3 - sp.sqrt(ct / 2) / 2), ct, 0.005))
        assert float(inflow.inflow_thrust_sensitivity(th, math.sqrt(c_t / 2))) == pytest.approx(c_a, rel=1e-9)


def test_k_z_is_the_frozen_inflow_slope():
    """T = c_T·Ω² − k_z·Ω·V_z is the linear law with v_i frozen: k_z = s·√(2ρA·c_T)/κ; with
    momentum inflow the hover slope of T/T_static per v_h is s/(2 + s)."""
    s, ct, rho, area, om, vz, kappa = sp.symbols("s c_T rho A Omega V_z kappa", positive=True)
    v_static = kappa * om * sp.sqrt(ct / (2 * rho * area))
    frozen = ct * om ** 2 * inflow.inflow_thrust_ratio((vz + v_static) / v_static, s)
    assert sp.simplify(-sp.diff(frozen, vz) / om - s * sp.sqrt(2 * rho * area * ct) / kappa) == 0
    # quasi-static: u² = 1 − s·(x + u·v_mom(x/u) − 1), implicit slope at x = 0, u = 1
    u, x = sp.symbols("u x", real=True)
    g = u ** 2 - inflow.inflow_thrust_ratio(x + u * descent.momentum_inflow_axial(x / u), s)
    du = (-sp.diff(g, x) / sp.diff(g, u)).subs({x: 0, u: 1})
    assert sp.simplify(2 * du + s / (2 + s)) == 0      # d(u²)/dx = −s/(2+s): climb unloads
    # RACER_5IN: backend slope 0.171 per v_h ↔ s = 0.41 with momentum inflow
    slope = wt.backend_descent_ratio(RACER_5IN, 1.0) - 1
    assert slope == pytest.approx(0.171, abs=0.002)
    assert 2 * slope / (1 - slope) == pytest.approx(0.41, abs=0.01)


def test_tandem_interference_factor():
    """Limits and the published curve (their Fig. 6, l/R = 2.83): ≈ 1.9 for a flat wake at
    α = 0, ≈ 0.1 for a wake going straight down; less with wake angle and with incidence."""
    p = sp.Symbol("p", positive=True)
    assert sp.simplify(inflow.tandem_interference_factor(p, 0, 0) - (1 + p / sp.sqrt(1 + p ** 2))) == 0
    assert sp.limit(inflow.tandem_interference_factor(p, 0, 0), p, sp.oo) == 2
    k = lambda a, g: float(inflow.tandem_interference_factor(2.83, math.radians(a), math.radians(g)))
    assert k(0, 0) == pytest.approx(1.94, abs=0.01) and k(0, 89.99) == pytest.approx(0.11, abs=0.01)
    assert k(15, 0) == pytest.approx(1.22, abs=0.01)
    assert k(0, 10) > k(0, 30) > k(0, 60) and k(0, 20) > k(5, 20) > k(15, 20)


# ---------------------------------------------------------------- findings and triage sizes

def test_f32_models_gain_thrust_where_rotors_lose_it():
    """F-32: at constant rotor speed and a descent ratio of 1.2 the backend and the verified
    BEM model (quartic VRS inflow) give MORE thrust than hover; measured 4–6 inch
    fixed-pitch rotors give 0.71–0.78 of it (Veismann 2023)."""
    assert wt.backend_descent_ratio(RACER_5IN, 1.2) == pytest.approx(1.21, abs=0.01)
    assert wt.bem_descent_ratio(1.2) == pytest.approx(1.18, abs=0.02)
    assert all(wt.bem_descent_ratio(x) > 1.0 for x in (0.5, 1.0, 1.5, 2.0))
    measured = [dict(r["curve"]).get(1.2, dict(r["curve"]).get(1.25))
                for r in wt.VEISMANN_2023.values()]
    assert max(measured) <= 0.78 and min(measured) >= 0.70
    # the quartic's net inflow only falls with descent rate; Johnson's rises past 0.45
    q = [wt.quartic_net_inflow(x) for x in (0.0, 0.5, 1.0, 1.5, 2.0)]
    assert q == sorted(q, reverse=True) and len(set(q)) == len(q)
    assert -1.5 + float(wt.johnson_inflow(-1.5)) > -0.45 + float(wt.johnson_inflow(-0.45))


def test_johnson_with_linear_law_against_measured_descent():
    """With Table 4 and s from blade geometry the dip has the right sign and place but about
    half the measured depth; (V+v)_X ≈ 1.55 gives the depth of the two HQProp rotors and
    fits their curves to 0.05 rms up to x = 1.75. Past x = 2 the linear law overshoots."""
    for s, lo, hi in ((0.5, 0.90, 0.92), (1.0, 0.84, 0.86)):
        x_min, y_min = wt.minimum(lambda x, s=s: wt.johnson_descent_ratio(x, s))
        assert lo < y_min < hi and 1.35 < x_min < 1.45
    for name, peak in (("HQ 6x3", 1.54), ("HQ 5x4", 1.57)):
        rotor = wt.VEISMANN_2023[name]
        c = wt.descent_comparison(rotor["curve"], wt.blade_element_sensitivity(rotor))
        assert c["peak"] == pytest.approx(peak, abs=0.02)
        assert c["fitted_rms"] < 0.05 < c["default_rms"]
        assert c["fitted_min"][0] == pytest.approx(c["measured_min"][0], abs=0.13)
        assert c["excess_beyond_2"] > 0.2
    # not every propeller follows it: the APC 10×5 dips earlier and never rises first
    apc = wt.descent_comparison(wt.SHETTY_2011_10X5,
                                wt.blade_element_sensitivity(wt.SHETTY_2011_ROTOR))
    assert apc["fitted_rms"] > 0.1 and apc["measured_min"][0] < 1.1


def test_forward_speed_closes_the_band():
    dips = [wt.minimum(lambda x, vx=vx: wt.johnson_descent_ratio(x, 1.0, vx=vx))[1]
            for vx in (0.0, 0.5, 0.95)]
    assert dips[0] == pytest.approx(0.85, abs=0.01) and dips[1] == pytest.approx(0.95, abs=0.015)
    assert dips[2] > 1.2


def test_descent_fluctuation_sizes():
    ratios = wt.fluctuation_over_loss()
    assert min(ratios.values()) == pytest.approx(0.33, abs=0.01)
    assert max(ratios.values()) == pytest.approx(0.46, abs=0.01)
    hover = wt.descent_roll_disturbance(RACER_5IN, 0.13, 0.78)
    full = wt.descent_roll_disturbance(RACER_5IN, 0.13, 0.78, RACER_5IN["limits"]["rotor_speed_max"])
    idle = wt.descent_roll_disturbance(RACER_5IN, 0.13, 0.78, 300.0)
    assert hover["roll_accel_rms"] == pytest.approx(14.1, abs=0.2)
    assert full["roll_accel_rms"] == pytest.approx(63.4, abs=0.5) and full["v_h"] == pytest.approx(16.4, abs=0.1)
    assert idle["roll_accel_rms"] == pytest.approx(1.1, abs=0.1) and idle["v_h"] == pytest.approx(2.1, abs=0.1)


def test_f33_ground_effect_candidates_against_measurement():
    """F-33: at one rotor radius the candidates span +3 % to +71 %; no measurement at
    constant rotor speed is above +18 %. The Crazyflie-fitted G is a far-field value."""
    m = wt.ground_models(RACER_5IN, 1.0)
    assert m["pybullet (G = 11.37)"] == pytest.approx(0.711, abs=0.002)
    assert m["cheeseman_bennett_fixed_pitch"] == pytest.approx(0.033, abs=0.001)
    assert m["he_leang_exponential"] == pytest.approx(0.088, abs=0.001)
    assert m["sanchez_cuevas (K_b = 2)"] == pytest.approx(0.158, abs=0.002)
    near = [gain for pts in wt.GROUND_MEASURED.values() for z, gain in pts if 0.9 <= z <= 1.2]
    assert 0.08 < min(near) and max(near) <= 0.18
    cf = wt.GROUND_MEASURED["Crazyflie on a load cell (Carter et al., Fig. 2a)"]
    assert wt.pybullet_gain_implied(*cf[0]) == pytest.approx(3.7, abs=0.1)
    assert wt.pybullet_gain_implied(*cf[-1]) == pytest.approx(9.1, abs=0.1)
    # whole vehicles sit above the bare rotor: the body term of sanchez_cuevas
    assert m["he_leang_exponential"] < 0.136 < m["sanchez_cuevas (K_b = 2)"] + 0.03
    # JSBSim's dimensional decay rate is two orders too slow at this scale
    assert wt.talbot_decay_per_radius(RACER_5IN) == pytest.approx(0.021, abs=0.001)
    assert float(ground_effect.talbot_inflow_factor(RACER_5IN["r_prop"], 0.33, 0, 1)) < 0.03


def test_time_scales():
    """Dynamic inflow is 20× faster than the motor; Johnson's VRS lag is as slow as it."""
    racer, cf = wt.time_scales(RACER_5IN), wt.time_scales(CRAZYFLIE)
    assert racer["dynamic_inflow"] == pytest.approx(1.8e-3, abs=1e-4)
    assert racer["vrs_lag"] == pytest.approx(37e-3, abs=1e-3)
    assert racer["motor"] / racer["dynamic_inflow"] > 15
    assert cf["dynamic_inflow"] == pytest.approx(1.1e-3, abs=1e-4)
    assert cf["vrs_lag"] == pytest.approx(23e-3, abs=1e-3)


def test_f34_hub_moment_is_nose_up_and_minor():
    """F-34: a rigid rotor in edgewise flow carries a NOSE-UP hub moment of 0.13–0.22·T·R at
    μ = 0.08–0.16 (isolated SOLO rotor, NASA TM-2018-219758) — k_flap < 0 in this repo's
    flapping_moment; 2–5 % of RACER_5IN's pitch authority at 10 m/s."""
    rows = wt.hub_moment_measured()
    assert [round(r["mu"], 3) for r in rows] == [0.078, 0.159]
    assert [r["raw"] for r in rows] == pytest.approx([0.170, 0.223], abs=0.002)
    assert [r["net"] for r in rows] == pytest.approx([0.133, 0.192], abs=0.002)
    assert all(r["raw"] > 0 and r["net"] > 0 for r in rows)
    lo, hi = wt.hub_moment_on_vehicle(RACER_5IN, 0.9, 10.0), wt.hub_moment_on_vehicle(RACER_5IN, 2.2, 10.0)
    assert lo["fraction_of_authority"] == pytest.approx(0.020, abs=0.002)
    assert hi["fraction_of_authority"] == pytest.approx(0.049, abs=0.002)
    assert hi["k_flap"] < lo["k_flap"] < 0
    # sign in this repo's convention: k_flap < 0 gives −M_y (nose-up in FLU) for v_x > 0
    from skyflow_dynamics.spec.rotor_aero import flapping_moment
    m = flapping_moment(1000.0, sp.Matrix([10.0, 0, 0]), lo["k_flap"])
    assert m[1] < 0 and m[0] == 0


def test_rear_rotor_interference_sizes():
    """Rear rotors lose about 10–12 % at 6–10 m/s near level (Nguyen measured 11 % at
    6 m/s and 5°); the loss is gone at racing speed and pitch."""
    slow = wt.rear_rotor_loss(RACER_5IN, 6.0, 5.0, 0.5)
    cruise = wt.rear_rotor_loss(RACER_5IN, 10.0, 8.0, 0.5)
    fast = wt.rear_rotor_loss(RACER_5IN, 35.0, 60.0, 0.5)
    assert slow["thrust_change"] == pytest.approx(-0.102, abs=0.005)
    assert cruise["thrust_change"] == pytest.approx(-0.124, abs=0.005)
    assert abs(fast["thrust_change"]) < 0.005
