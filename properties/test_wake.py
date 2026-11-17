"""A rotor in its own wake: descent through the vortex ring state, ground effect at constant
rotor speed, rotor-on-rotor interference (spec.descent, spec.inflow, spec.ground_effect;
REFERENCES.md "Own-wake aerodynamics", 2026-11-16).

Part 1 — candidate-tier checks: the published forms and constants, limits, continuity, and
the identities that tie the terms together (one inflow sensitivity s behind ground effect,
descent thrust loss and the k_z inflow term).

Part 2 — findings F-32 to F-34 and the triage sizes the registry quotes
(tools/wake_triage.py). The measured values there are figure and table read-offs; the
tolerances here are the sizes the ledger states, not a validation of the models."""

import json
import math
import pathlib

import numpy as np
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


# ---------------------------------------------------------------- F-35: tabulated descent data
# golden/checkdata/descent_curves.json, frozen by tools/descent_data.py from Shetty's thesis
# tables (UIUC 2011, pinned by sha256) and Veismann's read-offs. The data are not
# redistributed; these tests check the frozen fits and that the spec still reproduces them.

DESCENT = json.loads((pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata"
                      / "descent_curves.json").read_text())
UIUC = [c for c in DESCENT["curves"] if c["lab"] == "UIUC"]
CALTECH = [c for c in DESCENT["curves"] if c["lab"] != "UIUC"]


def test_f35_measured_minimum_comes_earlier_than_table_4():
    """18 APC propellers: the thrust minimum sits at a descent ratio of 0.87–1.24; Johnson's
    Table 4 with blade-element s puts it 0.15–0.42 later on every one that dips."""
    assert len(UIUC) == 18
    for c in UIUC:
        x_meas = c["measured_min"][0]
        if c["name"].endswith("Sport Propeller 10x10"):     # barely dips: 0.98 at the first point
            continue
        assert 0.85 <= x_meas <= 1.25, c["name"]
        assert c["table4_min"][0] > x_meas + 0.1, c["name"]


def test_f35_location_is_a_propeller_family_parameter():
    """Three free parameters (s, k, peak) fit every curve to ≤ 0.09 rms (median ≈ 0.03,
    against ≈ 0.15 with Table 4). The VRS-location scale k is tight within each lab and
    differs between them: 0.69–0.83 for the UIUC APC set, 0.93–0.97 for Veismann's FPV rotors."""
    fits = [c["fit"] for c in DESCENT["curves"]]
    assert max(f["rms"] for f in fits) < 0.09
    assert np.median([f["rms"] for f in fits]) < 0.04
    assert np.median([c["table4_rms"] for c in DESCENT["curves"]]) > 0.13
    assert all(0.65 <= c["fit"]["k"] <= 0.85 for c in UIUC)
    assert all(0.90 <= c["fit"]["k"] <= 1.00 for c in CALTECH)


def test_f35_held_out_beats_backend_within_a_family_not_across():
    """Leave one propeller out, one free parameter each (s; the backend's slope): Johnson
    with the others' median (k, peak) halves the backend's error and wins on 16 of 18. The
    UIUC constants do NOT transfer to the FPV rotors."""
    ho = np.array([[c["held_out"]["rms"], c["held_out"]["backend_rms"]] for c in UIUC])
    assert np.median(ho[:, 0]) < 0.5 * np.median(ho[:, 1])
    assert int((ho[:, 0] < ho[:, 1]).sum()) >= 16
    cross = np.array([[c["cross_lab"]["rms"], c["cross_lab"]["backend_rms"]] for c in CALTECH])
    assert np.median(cross[:, 0]) > 0.9 * np.median(cross[:, 1])


def test_f35_descent_sensitivity_matches_ground_effect():
    """Fitted s (median 0.44) lies in the ground-effect C_a range 0.27–0.63 — one parameter
    behind both, as inflow_thrust_law says. Blade-element s from geometric pitch does not
    (cambered, low-pitch propellers: up to 3.3)."""
    s_fit = np.median([c["fit"]["s"] for c in UIUC])
    assert 0.27 <= s_fit <= 0.63
    assert max(c["s_blade_element"] for c in UIUC) > 3.0


def test_f35_frozen_fits_reproduce_from_the_spec():
    """The spec's Johnson inflow with the frozen (s, k, peak) gives the frozen minimum."""
    from tools import descent_data as dd
    for c in (UIUC[1], UIUC[-1], CALTECH[1]):
        f = c["fit"]
        x, y = dd.model_minimum(f["s"], f["k"], f["peak"])
        assert x == pytest.approx(f["model_min"][0], abs=0.011)
        assert y == pytest.approx(f["model_min"][1], abs=0.002)


def test_neurobem_near_axial_descent_envelope():
    """NeuroBEM reaches a descent ratio of 0.8 near-axially (about 5 s of data past 0.4).
    Thrust per Ω² rises from climb to x ≈ 0.3, dips about 0.04 at 0.4–0.5 (onset of the band;
    the confidence intervals touch), and rises again where frame drag grows. Consistent with
    both k = 0.74 and k = 0.95 — it does not decide the location."""
    bins = {tuple(b["x"]): b for b in DESCENT["neurobem_near_axial"]}
    assert bins[(0.2, 0.3)]["ratio"] - bins[(0.4, 0.5)]["ratio"] == pytest.approx(0.04, abs=0.015)
    assert bins[(-0.6, -0.4)]["ratio"] < bins[(0.0, 0.1)]["ratio"] < bins[(0.2, 0.3)]["ratio"]
    assert sum(b["n"] for b in DESCENT["neurobem_near_axial"] if b["x"][0] >= 0.4) < 1500


# ---------------------------------------------------------------- F-36: whole-quadcopter ground effect
# golden/checkdata/ground_ceiling.json, frozen by tools/ground_data.py from Du Mutel et al.
# 2024 (Zenodo 13166292, pinned by sha256): a quadcopter on a load cell, 10 in and 12 in
# propellers, a panel at 5–100 cm, rotor speed measured. The ratio is at matched rotor speed.

GROUND = json.loads((pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata"
                     / "ground_ceiling.json").read_text())


def test_f36_single_propeller_constants_predict_a_whole_quadcopter():
    """Measured evidence for ground_effect_fixed_rpm: He & Leang's constants for one 5×3
    propeller (C_a 0.53, C_b 1.80; another lab, not fitted here) give 0.020 rms (12 in) and
    0.024 (10 in) against 0.14 and 0.12 for the backend, which has no ground effect. Fixed-
    pitch Cheeseman–Bennett with the F-35 descent s = 0.44 gives 0.035 and 0.033."""
    for prop in ("12in", "10in"):
        m = GROUND["ground"][prop]["model_rms"]
        assert m["he_leang_5x3"] < 0.03, prop
        assert m["he_leang_5x3"] < 0.25 * m["backend"], prop
        assert m["cheeseman_bennett_fixed_pitch_s044"] < 0.3 * m["backend"], prop
    near = GROUND["ground"]["12in"]["points"][0]
    assert near["z_over_R"] == pytest.approx(0.33, abs=0.01)
    assert near["ratio"] == pytest.approx(1.26, abs=0.01)


def test_f36_fitted_constants_are_single_propeller_values():
    """Fitted freely, the whole vehicle gives C_a 0.47/0.53 and C_b 1.74/2.08 — inside He &
    Leang's single-propeller ranges (0.27–0.63, 1.18–2.87). Fixed-pitch s 0.46/0.58 sits with
    the descent fit (F-35 median 0.44): one inflow sensitivity behind both."""
    for prop in ("12in", "10in"):
        f = GROUND["ground"][prop]["fit_he_leang"]
        assert 0.27 <= f["C_a"] <= 0.63 and 1.18 <= f["C_b"] <= 2.87, prop
        assert 0.27 <= GROUND["ground"][prop]["fit_fixed_pitch"]["s"] <= 0.63, prop
    assert GROUND["ground"]["12in"]["fit_fixed_pitch"]["s"] == pytest.approx(0.44, abs=0.05)


def test_f36_constant_power_and_fitted_g_forms_overshoot():
    """Near the ground the constant-power forms overshoot: Cheeseman–Bennett 2.39 and
    Sanchez-Cuevas (images only) 2.55 against 1.26 measured at z/R = 0.33; worse than no
    model below their z/R ≥ 0.5 envelope, within 0.03 inside it. pybullet's additive form
    needs G ≤ 1.5 here, not 11.37."""
    for prop in ("12in", "10in"):
        g = GROUND["ground"][prop]
        assert g["model_rms"]["cheeseman_bennett"] > g["model_rms"]["backend"]
        assert g["model_rms"]["sanchez_cuevas_kb0"] > g["model_rms"]["backend"]
        for p in g["points"]:
            if p["z_over_R"] >= 0.5:
                assert abs(p["models"]["cheeseman_bennett"] - p["ratio"]) < 0.03
            if p["z_over_R"] <= 2.0:
                assert p["pybullet_G_implied"] < 1.6
    near = GROUND["ground"]["12in"]["points"][0]["models"]
    assert near["cheeseman_bennett"] == pytest.approx(2.39, abs=0.01)


def test_f36_ceiling_is_weaker_near_and_reaches_further():
    """12 in, same rotor speed: a ceiling adds 15 % at z/R = 0.33 (ground: 26 %) but more
    than the ground beyond 1.6 R (+3.8 / +1.9 % at 1.6 / 2.3 R, ground +1.9 / −1.3 %)."""
    g = {round(p["D_cm"]): p["ratio"] for p in GROUND["ground"]["12in"]["points"]}
    c = {round(p["D_cm"]): p["ratio"] for p in GROUND["ceiling"]["12in"]["points"]}
    assert c[5] == pytest.approx(1.146, abs=0.01) and c[5] < g[5]
    assert c[25] > g[25] and c[35] > g[35]
    assert c[35] > 1.01


def test_f36_tachometer_quality():
    """The 12 in tachometers agree to 1.5 %; the 10 in ones disagree by up to 10 %, so the
    10 in ratios carry a ±10 % band and the 12 in runs are the reliable set."""
    assert max(p["tach_gap"] for p in GROUND["ground"]["12in"]["points"]) < 0.015
    assert max(p["tach_gap"] for p in GROUND["ground"]["10in"]["points"]) > 0.05


def test_f36_frozen_models_reproduce_from_the_spec():
    from tools import ground_data as gd
    for prop in ("12in", "10in"):
        g = GROUND["ground"][prop]
        for p in g["points"]:
            fresh = gd.candidates(p["z_over_R"], g["R_m"])
            for name, value in p["models"].items():
                assert fresh[name] == pytest.approx(value, rel=1e-9), (prop, name)


# ---------------------------------------------------------------- F-38: inter-vehicle downwash
# golden/checkdata/downwash_maps.json, frozen by tools/downwash_data.py from Kiran, Ayanian &
# Breuer's RSS 2025 data (Hugging Face ajkiran18/QuadrotorDownwash, pinned by sha256): two
# Crazyflies on load cells, the lower one fixed, both at a fixed hover command on 4 V (rotor
# speed not held). The data are not redistributed; these tests check the frozen numbers and
# that the spec still reproduces its side of the comparison.

DOWNWASH = json.loads((pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata"
                       / "downwash_maps.json").read_text())
DW_ROWS = DOWNWASH["force_rows"]
DW_VALID = [r for r in DW_ROWS if "pybullet" in r]          # Δz ≥ 0.7 m, the fit's range


def test_f38_pybullet_downwash_overpredicts_the_centreline():
    """In its own range (Δz 0.7–1.1 m) the pybullet fit takes 24–58 % of the lower
    Crazyflie's weight; the measured loss is 1.8–3.6 % (12–16×). The measured loss steps
    down 3.5× between 0.60 and 0.70 m; against the near block's own trend extended past the
    step, and at 0.5–0.6 m just outside the fit's range, the fit is still 6–8× too strong."""
    assert len(DW_VALID) == 5
    for r in DW_VALID:
        assert 0.015 <= r["loss_centre"] <= 0.04, r["dz_m"]
        assert 12 <= r["pybullet"]["ratio_centre"] <= 17, r["dz_m"]
    for e in DOWNWASH["near_block"]["extrapolated"] + DOWNWASH["near_block"]["outside_range"]:
        assert 6 <= e.get("pybullet_ratio", e.get("ratio")) <= 8, e["dz_m"]


def test_f38_pybullet_core_is_too_narrow():
    """The fit's Gaussian half-width at half maximum is 2–78 mm over 0.7–1.1 m; the
    measured loss falls to half at 0.11–0.14 m off the axis. Integrated over the plane the
    fit swings from 0.003 to 3.4 times the measured impulse, so no rescaling of c1 fixes it:
    the form (a CoM point force with β = 0.16·Δz − 0.11) is wrong, not only the size."""
    for r in DW_VALID:
        assert r["pybullet"]["half_max_x_m"] < 0.08
        assert 0.11 <= r["half_loss_x_m"] <= 0.15
    ratios = [r["pybullet"]["integral_Nm2"] / r["pybullet"]["measured_integral_Nm2"] for r in DW_VALID]
    assert min(ratios) < 0.01 and max(ratios) > 3


def test_f38_measured_step_is_not_in_the_wake():
    """Centreline loss 12.7 % at 0.60 m, 3.6 % at 0.70 m. The single-vehicle wake speed at the
    lower rotors (PIV, front-rotor chords) falls only 12 % over the same step: the step is in
    the force map, not in the flow, and is unexplained."""
    by = {round(r["dz_m"], 2): r for r in DW_ROWS}
    assert by[0.6]["loss_centre"] / by[0.7]["loss_centre"] > 3.0
    pv = {round(e["dz_m"], 2): e["u_rotor_over_Ui"] for e in DOWNWASH["inflow_sensitivity"]}
    assert pv[0.7] / pv[0.6] > 0.85


def test_f38_wake_is_a_round_jet_with_a_far_virtual_origin():
    """The merged wake (z/l 6.5–23.8) fits B/(z − z0)·exp(−c·x²/(z − z0)²) to 0.03 U_i rms with
    the virtual origin 0.37 m ABOVE the rotor plane and c = 91 (spread rate 0.087, the
    classical round-jet value). Jain's published z0 (0.03–0.11 m below) misses the centreline
    by up to 70 %, and his c_rad 25–61 is a 1.2–1.9× wider jet. The mean wake carries 83–98 %
    of the thrust momentum only if U_i ≈ 8.2 m/s (the paper's Re example); with its Eq. (1)
    (U_i = v_h ≈ 4.05 m/s) it would carry about 20 %, so the field is read with U_i ≈ 2·v_h."""
    jet = DOWNWASH["jet"]
    assert jet["rms"] < 0.03
    assert jet["z0_m"] < -0.3
    assert 0.08 <= jet["S_half_width"] <= 0.095
    assert jet["c_rad"] > 60.8
    assert jet["centreline_fits"]["free"]["max_rel_err"] < 0.15
    assert jet["centreline_fits"]["z0_jain_0.07m"]["max_rel_err"] > 0.5
    mom = jet["momentum_over_W_at_z_l_8_15_22"]
    assert all(0.8 <= m <= 1.0 for m in mom["re_example"])
    assert all(m < 0.3 for m in mom["eq1_n4"])


def test_f38_frozen_comparison_reproduces_from_the_spec():
    """spec.ground_effect.pybullet_downwash_force gives the frozen pybullet column, and
    jain_wake_velocity with the frozen fit gives the frozen centreline speed."""
    c = DOWNWASH["constants"]
    p = c["pybullet"]
    for r in (DW_VALID[0], DW_VALID[-1]):
        F = ground_effect.pybullet_downwash_force(r["dz_m"], 0.0, p["R"], p["c1"], p["c2"], p["c3"])
        assert -float(F) / c["W_N"] == pytest.approx(r["pybullet"]["loss_centre"], rel=1e-9)
    jet = DOWNWASH["jet"]
    U = c["U_i_cases"]["re_example"]
    A_p = math.pi * c["R_prop_m"] ** 2
    T_prop = 2 * c["rho"] * A_p * c["v_h"] ** 2                 # per-rotor hover thrust
    l = c["l_m"]
    for s in jet["slices"][4:]:
        z = s["z_l"] * l
        V = ground_effect.jain_wake_velocity(z, 0.0, T_prop, c["rho"], A_p, 2 * l,
                                             jet["z0_m"], jet["c_ax"]["re_example"], jet["c_rad"])
        assert float(V) / U == pytest.approx(s["model_u_max"], rel=0.02)


# ---------------------------------------------------------------- F-37: rotor-on-rotor interference
# golden/checkdata/rotor_interference.json, frozen by tools/rotor_interference_data.py from the
# NASA MTB2 data tables (pinned by sha256; 24.5 in two-blade rotors, hub spacing 2.69R, back
# rotors raised 0 / 0.33 / 0.57 R). Metric: slope of thrust against ρ·rpm² over 2000–3000 rpm
# against the same rotor alone (the load cells carry run-dependent additive offsets).

RI = json.loads((pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata"
                 / "rotor_interference.json").read_text())


def _row(table, layout, rotor, vbin):
    return next(t for t in RI["tables"][table]
                if (t["layout"], t["rotor"], t["vbin"]) == (layout, rotor, vbin))


def test_f37_front_rotor_feels_no_back_rotor():
    """In line, the front rotor is unchanged to ±1.5 % (median) at 20 and 40 ft/s in every
    layout, and blade phase (0 against −90 deg) moves thrust by under 1 % — the +6 to +13 %
    in the raw point ratios is a load-cell offset, gone in the slope metric."""
    for lay in ("equal", "concept", "tall"):
        for v in (20, 40):
            assert abs(_row("tandem", lay, 1, v)["median"]) < 0.015, (lay, v)
    assert abs(RI["phase_0_vs_m90"]["median"]) < 0.01
    assert RI["phase_0_vs_m90"]["rms"] < 0.03
    assert _row("tandem", "equal", 1, 20)["point_2500_median"] > 0.10


def test_f37_back_rotor_loss_grows_with_speed_and_shrinks_with_height():
    """Back rotor in line: −6.5 % / −28 % at 20 / 40 ft/s with both rotors at one height;
    raising it 0.33R gives −2 % / −11 %, 0.57R gives 0 % / −9 %. Torque rises as thrust
    falls (+5 to +17 %): the rotor works against air already moving down, as in a climb."""
    t = {(lay, v): _row("tandem", lay, 3, v)["median"]
         for lay in ("equal", "concept", "tall") for v in (20, 40)}
    assert t[("equal", 20)] == pytest.approx(-0.065, abs=0.02)
    assert t[("equal", 40)] == pytest.approx(-0.28, abs=0.04)
    assert t[("equal", 40)] < t[("concept", 40)] < -0.08
    assert t[("tall", 40)] == pytest.approx(-0.09, abs=0.02)
    assert abs(t[("tall", 20)]) < 0.02 and abs(t[("concept", 20)]) < 0.04
    for lay in ("equal", "concept", "tall"):
        for v in (20, 40):
            assert _row("tandem", lay, 3, v)["dQ_median"] > 0.03


def test_f37_spec_factor_carries_the_size_with_descent_s():
    """tandem_interference_factor + the linear inflow law, one free s over 48 back-rotor
    conditions: s = 0.50 (0.37–0.56 by layout) — the descent median 0.44 (F-35) does as well
    (rms 0.051 against 0.144 for no interference). Quadrotor back rotors (concept, tall): 0.53."""
    f = RI["fits"]
    assert f["tandem_back_all"]["s"] == pytest.approx(0.50, abs=0.05)
    for lay in ("equal", "concept", "tall"):
        assert 0.3 <= f[f"tandem_back_{lay}"]["s"] <= 0.65
    assert f["tandem_back_all"]["rms_s044"] < 0.4 * f["tandem_back_all"]["rms_zero"]
    assert f["quad_back_concept_tall"]["s"] == pytest.approx(0.53, abs=0.08)


def test_f37_speed_trend_is_too_weak_in_the_factor():
    """The factor's loss grows less than 2× from 20 to 40 ft/s (V/v_h ≈ 1.4 to 2.8); the
    measured loss grows 4× (equal height) to over 10× (raised). Seen as a local s: 0.04–0.29
    at 20 ft/s, 0.42–0.63 at 40 ft/s. At V/v_h ≈ 1.4 the steep front wake passes under a
    raised back rotor and the factor still predicts a loss."""
    loc = {(l["layout"], l["vbin"]): l["s_median"] for l in RI["local_s"]}
    for lay in ("equal", "concept", "tall"):
        assert loc[(lay, 20)] < 0.3, lay
        assert 0.4 <= loc[(lay, 40)] <= 0.65, lay
    assert loc[("tall", 20)] < 0.1


def test_f37_side_by_side_and_front_rotors_gain_a_little():
    """Side by side: +0.5 to +1.4 % at 20 ft/s, +3.6 to +3.9 % at 40 ft/s (upwash from the
    neighbour). Front rotors of the quadrotor: +2 to +7 % (the lateral neighbour)."""
    for r in (1, 2):
        assert 0.0 < _row("sbs", "concept", r, 40)["median"] < 0.05
        assert abs(_row("sbs", "concept", r, 20)["median"]) < 0.02
    for lay in ("concept", "tall"):
        for r in (1, 2):
            assert 0.01 < _row("quad", lay, r, 40)["median"] < 0.09, (lay, r)


def test_f37_frozen_factor_reproduces_from_the_spec():
    """The spec's factor at the frozen hub-line geometry gives the frozen k."""
    for t in RI["tandem_back_by_pitch"][::7]:
        k = float(inflow.tandem_interference_factor(t["p"], math.radians(t["alpha_deg"]),
                                                    math.radians(t["gamma_deg"])))
        assert k == pytest.approx(t["k"], rel=0.02)
