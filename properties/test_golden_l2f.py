"""Authenticity: the symbolic spec must reproduce the frozen rl-tools L2F vectors
(kind == "l2f_terms", golden/generate/gen_l2f.py — the EXECUTED header-only C++ library at
the commit pinned by rl-tools/hyperdrone).

Division of labor: the spec carries the math (exact Ornstein–Uhlenbeck bias step, additive
IMU corruption, strapdown specific force, exact-exponential and asymmetric motor lag, RK4);
this test carries the reference's conventions as explicit harness details —
  - white noise as a per-sample standard deviation plus a constant offset, where the spec
    takes a noise density σ_nd and a turn-on bias: σ_nd = std·√dt, b_on = offset;
  - τ ≤ 0 meaning a pure random walk (the spec's τ → ∞ limit, proven symbolically here);
  - the accelerometer's finite-difference specific force (v⁺ − v)/dt at the new attitude;
  - the motor setpoint each path takes — the CLOSED_FORM path uses the raw action, not the
    mapped command (finding F-31; the vectors carry the setpoint as executed);
  - rotor state in normalized command units (the lag is unit-free).
The unit-normal draws are the reference's own, replayed on a copy of its generator state
(see the generator); nothing here is recovered from the outputs it is checked against."""

import json
import math
import pathlib

import numpy as np
import pytest
import sympy as sp

from skyflow_dynamics.spec import motor, sensors
from skyflow_dynamics.spec.discretization import rk4_step

DOC = json.loads((pathlib.Path(__file__).resolve().parent.parent / "golden" / "vectors"
                  / "l2f_imu.json").read_text())
STEP_FAMILIES = ("imu_step_ou", "imu_step_random_walk", "imu_step_closed_form_motor")

_B = sp.Matrix(sp.symbols("b1 b2 b3", real=True))
_W = sp.Matrix(sp.symbols("w1 w2 w3", real=True))
_TAU, _SIG, _DT = sp.symbols("tau sigma dt", positive=True)
_BIAS_EXPR = sensors.imu_bias_gauss_markov_step(_B, _TAU, _SIG, _DT, _W)
_BIAS = sp.lambdify((*_B, *_W, _TAU, _SIG, _DT), _BIAS_EXPR, "numpy")
#: the τ → ∞ limit of the spec step — what the reference runs for τ ≤ 0
_WALK_EXPR = _BIAS_EXPR.applyfunc(lambda e: sp.limit(e, _TAU, sp.oo))
_WALK = sp.lambdify((*_B, *_W, _SIG, _DT), _WALK_EXPR, "numpy")

_Y = sp.Matrix(sp.symbols("y1 y2 y3", real=True))
_BON = sp.Matrix(sp.symbols("o1 o2 o3", real=True))
_N = sp.Matrix(sp.symbols("n1 n2 n3", real=True))
_SND = sp.symbols("sigma_nd", nonnegative=True)
_CORRUPT = sp.lambdify((*_Y, *_B, *_BON, *_N, _SND, _DT),
                       sensors.imu_corrupt(_Y, _B, _BON, _SND, _DT, _N), "numpy")

_Q = sp.Matrix(sp.symbols("q0 q1 q2 q3", real=True))
_VD = sp.Matrix(sp.symbols("vd1 vd2 vd3", real=True))
_OM = sp.Matrix(sp.symbols("om1 om2 om3", real=True))
_G = sp.symbols("g", positive=True)
_ACC_EXPR, _GYRO_EXPR = sensors.imu(_Q, _VD, _OM, sp.zeros(3, 1), sp.zeros(3, 1), sp.eye(3), _G)
_ACC = sp.lambdify((*_Q, *_VD, _G), _ACC_EXPR, "numpy")
_GYRO = sp.lambdify(tuple(_OM), _GYRO_EXPR, "numpy")

_U, _C = sp.symbols("u c", real=True)
_EXP_STEP = sp.lambdify((_U, _C, _DT, _TAU),
                        motor.exact_exp_step(sp.Matrix([_U]), sp.Matrix([_C]), _DT, _TAU)[0],
                        "numpy")
_KA, _KD = sp.symbols("ka kd", positive=True)
_RK4_LAG = sp.lambdify(
    (_U, _C, _DT, _KA, _KD),
    rk4_step(lambda t, s: motor.asymmetric_lag(s, sp.Matrix([_C]), _KA, 0, _KD, 0),
             sp.Matrix([_U]), _DT)[0], "math")   # scalar Piecewise: plain conditionals


def _bias_step(b, w, tau, sigma, dt):
    """The spec step under the reference's parameter conventions."""
    if sigma == 0:
        return np.asarray(b, float)          # no driving noise: the process is not run
    if tau > 0:
        return np.asarray(_BIAS(*b, *w, tau, sigma, dt), float).ravel()
    return np.asarray(_WALK(*b, *w, sigma, dt), float).ravel()


def _corrupt(y_true, bias, draw, mean, std, dt):
    """imu_corrupt with σ_nd = std·√dt and b_on = the constant offset; n is the unit
    normal behind the executed draw."""
    n = (np.asarray(draw) - mean) / std
    return np.asarray(_CORRUPT(*y_true, *bias, *([mean] * 3), *n, std * math.sqrt(dt), dt),
                      float).ravel()


def test_random_walk_is_the_spec_limit():
    """τ → ∞ of the exact OU step is b + σ·√dt·w — the branch the reference takes for τ ≤ 0."""
    for i in range(3):
        assert sp.simplify(_WALK_EXPR[i] - (_B[i] + _SIG * sp.sqrt(_DT) * _W[i])) == 0


@pytest.mark.parametrize("case", DOC["bias_process"],
                         ids=lambda c: f"tau={c['tau']:g},sigma={c['sigma']:g}")
def test_bias_process(case):
    """imu_bias_gauss_markov_step against the executed l2f::bias_process_step: OU at
    τ = 0.05 / 2 / 300 s, random walk, and the held bias."""
    moved = False
    for step in case["steps"]:
        got = _bias_step(step["b"], step["w"], case["tau"], case["sigma"], case["dt"])
        np.testing.assert_allclose(got, step["b_next"], rtol=1e-12, atol=1e-15)
        moved = moved or step["b_next"] != step["b"]
    assert moved == (case["sigma"] != 0)


def test_bias_process_covers_every_branch():
    kinds = {("ou" if c["tau"] > 0 else "walk") if c["sigma"] != 0 else "hold"
             for c in DOC["bias_process"]}
    assert kinds == {"ou", "walk", "hold"}


@pytest.mark.parametrize("family", STEP_FAMILIES)
def test_imu_measurement_in_full_step(family):
    """Gyro and accelerometer rows of the executed rl_tools::step: the bias states advance
    by the spec step, and each measurement is imu_corrupt over the spec's strapdown truth
    (identity mount, no lever arm)."""
    case = DOC[family]
    dt, grav = case["dt"], -case["gravity"][2]
    assert case["gravity"][:2] == [0, 0]
    for s in case["steps"]:
        gyro_bias = _bias_step(s["gyro_bias"], s["gyro_w"], case["gyro_bias_tau"],
                               case["gyro_bias_sigma"], dt)
        accel_bias = _bias_step(s["accel_bias"], s["accel_w"], case["accel_bias_tau"],
                                case["accel_bias_sigma"], dt)
        np.testing.assert_allclose(gyro_bias, s["gyro_bias_next"], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(accel_bias, s["accel_bias_next"], rtol=1e-12, atol=1e-15)

        gyro_true = np.asarray(_GYRO(*s["omega_next"]), float).ravel()
        gyro = _corrupt(gyro_true, gyro_bias, s["gyro_draw"], case["gyro_noise_mean"],
                        case["gyro_noise_std"], dt)
        np.testing.assert_allclose(gyro, s["gyro"], rtol=1e-12, atol=1e-13)

        v_dot = (np.asarray(s["v_next"]) - np.asarray(s["v"])) / dt   # harness: finite difference
        accel_true = np.asarray(_ACC(*s["q_next"], *v_dot, grav), float).ravel()
        accel = _corrupt(accel_true, accel_bias, s["accel_draw"], case["accel_noise_mean"],
                         case["accel_noise_std"], dt)
        np.testing.assert_allclose(accel, s["accel"], rtol=1e-12, atol=1e-12)


def test_asymmetric_closed_form_motor_step():
    """motor_exact_exp_discretization with the time constant chosen by direction — exact
    under a held command, since an exponential approach never crosses its setpoint."""
    case = DOC["imu_step_closed_form_motor"]
    assert case["closed_form"] and case["tau_rising"] != case["tau_falling"]
    rising = falling = 0
    for s in case["steps"]:
        for i in range(4):
            u, c = s["rpm"][i], s["motor_setpoint"][i]
            up = c >= u
            tau = case["tau_rising"][i] if up else case["tau_falling"][i]
            got = np.clip(_EXP_STEP(u, c, case["dt"], tau), case["action_min"], case["action_max"])
            assert got == pytest.approx(s["rpm_next"][i], rel=1e-12, abs=1e-15)
            rising, falling = rising + up, falling + (not up)
    assert rising > 5 and falling > 5


@pytest.mark.parametrize("family", ("imu_step_ou", "imu_step_random_walk"))
def test_asymmetric_integrated_motor_step(family):
    """motor_asymmetric_lag (ka2 = kd2 = 0) through the spec's RK4 against the integrated
    path of the executed step."""
    case = DOC[family]
    assert not case["closed_form"]
    for s in case["steps"]:
        for i in range(4):
            got = _RK4_LAG(s["rpm"][i], s["motor_setpoint"][i], case["dt"],
                           1 / case["tau_rising"][i], 1 / case["tau_falling"][i])
            got = np.clip(got, case["action_min"], case["action_max"])
            assert got == pytest.approx(s["rpm_next"][i], rel=1e-12, abs=1e-15)


def test_f31_closed_form_takes_the_raw_action():
    """Finding F-31: with CLOSED_FORM the executed setpoint is clamp(a, min, max); the
    integrated path maps the same action to (a + 1)/2. A negative action therefore commands
    zero in one path and a positive speed in the other."""
    cf, ou = DOC["imu_step_closed_form_motor"], DOC["imu_step_ou"]
    for case, mapped in ((cf, False), (ou, True)):
        lo, hi = case["action_min"], case["action_max"]
        assert (lo, hi) == (0, 1)
        for s in case["steps"]:
            for a, c in zip(s["action"], s["motor_setpoint"]):
                want = (a + 1) / 2 if mapped else min(hi, max(lo, a))
                assert c == pytest.approx(want, abs=1e-15)
    gaps = [abs(c - (a + 1) / 2) for s in cf["steps"]
            for a, c in zip(s["action"], s["motor_setpoint"])]
    assert max(gaps) > 0.4
