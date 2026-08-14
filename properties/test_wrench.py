"""Wrench-assembly checks: signs, hand-computed moments, and structural symmetries.
Sign conventions here guard against the exact failure modes found in reference sims
(F-3 gyro-x sign, F-6 spin-vs-torque-sign)."""

import numpy as np
import sympy as sp

from properties.helpers import flat_params, make_inputs, params_dict, statedot_fn
from skyflow_dynamics.spec import wrench
from skyflow_dynamics.spec.parameters import substitution
from skyflow_dynamics.spec.symbols import param_symbols, state_symbols

P4 = param_symbols(4)
S4 = state_symbols(4)


def _wrench_fn(values):
    w = sp.Matrix(sp.symbols("wb_1 wb_2 wb_3", real=True))
    va = sp.Matrix(sp.symbols("va_1 va_2 va_3", real=True))
    F, M = wrench.body_wrench(w, S4.W, va, P4)
    sub = substitution(P4, values)
    F, M = F.subs(sub), M.subs(sub)
    args = (w.flat(), va.flat(), S4.W.flat())
    return (sp.lambdify(args, F, modules="numpy"),
            sp.lambdify(args, M, modules="numpy"))


def _inertia_fn(values):
    w = sp.Matrix(sp.symbols("wb_1 wb_2 wb_3", real=True))
    Wd = sp.Matrix(sp.symbols("Wd_1:5", real=True))
    M = wrench.rotor_inertia_moment(w, S4.W, Wd, P4).subs(substitution(P4, values))
    return sp.lambdify((w.flat(), S4.W.flat(), Wd.flat()), M, modules="numpy")


def test_hover_wrench_is_pure_vertical_thrust():
    vals = params_dict()
    Ff, Mf = _wrench_fn(vals)
    w_h = 1788.53
    F = np.asarray(Ff([0, 0, 0], [0, 0, 0], [w_h] * 4), float).ravel()
    M = np.asarray(Mf([0, 0, 0], [0, 0, 0], [w_h] * 4), float).ravel()
    np.testing.assert_allclose(F[:2], 0, atol=1e-12)
    assert abs(F[2] - 4 * vals["ct2"][0] * w_h**2) < 1e-9
    np.testing.assert_allclose(M, 0, atol=1e-12)  # balanced speeds: zero net moment


def test_thrust_moment_hand_computed():
    # Asymmetric per-rotor ct2: M must equal Σ rᵢ × (Tᵢ ẑ) computed independently.
    ct2 = [2.0e-8, 2.6e-8, 2.2e-8, 2.4e-8]
    vals = params_dict(ct2=ct2, k_d=0.0, k_z=0.0)
    _, Mf = _wrench_fn(vals)
    W = np.array([1800.0, 1900.0, 1700.0, 2000.0])
    M = np.asarray(Mf([0, 0, 0], [0, 0, 0], list(W)), float).ravel()
    r = np.array(vals["rotor_pos"])
    T = np.array(ct2) * W**2
    expected = np.sum([np.cross(r[i], [0, 0, T[i]]) for i in range(4)], axis=0)
    # Yaw component additionally carries the reaction torques −sᵢ·cq2·Ω²:
    expected[2] += sum(-vals["spin"][i] * vals["cq2"][i] * W[i]**2 for i in range(4))
    np.testing.assert_allclose(M, expected, rtol=1e-12)


def test_yaw_reaction_opposes_spin():
    # A single CCW rotor (s=+1, spinning about +ẑ) must torque the airframe about −ẑ (F-6).
    vals = params_dict(spin=[1, 1, 1, 1], k_d=0.0, k_z=0.0)
    _, Mf = _wrench_fn(vals)
    M = np.asarray(Mf([0, 0, 0], [0, 0, 0], [1800.0, 0.0, 0.0, 0.0]), float).ravel()
    assert M[2] < 0


def test_gyroscopic_signs_opposite():
    # −ω×h: with net rotor momentum h_z ẑ, roll rate p gives +y moment, pitch rate q gives
    # −x moment — opposite leading signs (the exact Crazyflow F-3 failure mode).
    vals = params_dict(I_rot=3.4e-9, spin=[1, 1, 1, 1])  # all same spin → net momentum
    Mi = _inertia_fn(vals)
    W = [2000.0] * 4
    h_z = vals["I_rot"] * sum(vals["spin"][i] * W[i] for i in range(4))
    assert h_z > 0
    M_roll = np.asarray(Mi([1.0, 0, 0], W, [0] * 4), float).ravel()
    M_pitch = np.asarray(Mi([0, 1.0, 0], W, [0] * 4), float).ravel()
    np.testing.assert_allclose(M_roll, [0, +h_z, 0], rtol=1e-12)
    np.testing.assert_allclose(M_pitch, [-h_z, 0, 0], rtol=1e-12)


def test_gyroscopic_is_minus_omega_cross_h():
    vals = params_dict(I_rot=3.452e-8)
    Mi = _inertia_fn(vals)
    rng = np.random.default_rng(2)
    for _ in range(10):
        w = rng.uniform(-3, 3, 3)
        W = rng.uniform(500, 2500, 4)
        Wd = rng.uniform(-3000, 3000, 4)
        got = np.asarray(Mi(list(w), list(W), list(Wd)), float).ravel()
        s = np.array(vals["spin"], float)
        h = vals["I_rot"] * np.array([0, 0, np.sum(s * W)])
        expected = -np.cross(w, h) + np.array([0, 0, -vals["I_rot"] * np.sum(s * Wd)])
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-18)


def test_balanced_counter_rotation_cancels_rotor_inertia():
    vals = params_dict(I_rot=3.452e-8)  # crazyflie spin pattern (-1,1,-1,1)
    Mi = _inertia_fn(vals)
    M = np.asarray(Mi([1.0, -2.0, 0.5], [2000.0] * 4, [500.0] * 4), float).ravel()
    np.testing.assert_allclose(M, 0, atol=1e-18)


def test_flapping_moment_sign_in_forward_flight():
    # M = −k_flap·Ω·(v×ẑ) gives +M_y for v_x > 0 (nose-DOWN in this FLU frame; the sign of
    # an identified k_flap is part of its identification convention — see the docstring).
    vals = params_dict(k_flap=1.5e-7, k_d=0.0, k_z=0.0)
    _, Mf = _wrench_fn(vals)
    M = np.asarray(Mf([0, 0, 0], [3.0, 0, 0], [1800.0] * 4), float).ravel()
    assert M[1] > 0            # +M_y, matching the golden-pinned reference expression
    assert abs(M[0]) < 1e-12   # no roll from pure-x airspeed


def test_tilted_axis_hand_computed():
    # Tilt rotor 1 by θ toward +x: F gains T·sinθ in x; yaw torque scales by cosθ along ẑ
    # and leaks into x; moments follow r × (T·ê).
    theta = 0.05
    e1 = [np.sin(theta), 0.0, np.cos(theta)]
    vals = params_dict(axis=[e1, [0, 0, 1], [0, 0, 1], [0, 0, 1]], k_d=0.0, k_z=0.0)
    Ff, Mf = _wrench_fn(vals)
    W = [1800.0, 0.0, 0.0, 0.0]
    F = np.asarray(Ff([0, 0, 0], [0, 0, 0], W), float).ravel()
    M = np.asarray(Mf([0, 0, 0], [0, 0, 0], W), float).ravel()
    T = vals["ct2"][0] * 1800.0**2
    Q = vals["cq2"][0] * 1800.0**2
    e = np.array(e1)
    r = np.array(vals["rotor_pos"][0])
    np.testing.assert_allclose(F, T * e, rtol=1e-12)
    np.testing.assert_allclose(M, np.cross(r, T * e) - vals["spin"][0] * Q * e, rtol=1e-12)


def test_wind_enters_only_through_airspeed():
    # Equal v and v_wind → zero airspeed: aero terms must vanish identically.
    vals = params_dict(c_D=[0.1, 0.1, 0.15], k_flap=1e-7)
    p = flat_params(vals)
    f = statedot_fn()
    v = np.array([2.0, -1.0, 0.5])
    s_still = np.concatenate([[0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1800.0] * 4])
    s_moving = s_still.copy()
    s_moving[3:6] = v
    out_still = f(s_still, make_inputs(), p)
    out_moving = f(s_moving, make_inputs(v_wind=v), p)
    # Same airspeed (zero) → same v̇ and ω̇; only ẋ differs (by v).
    np.testing.assert_allclose(out_moving[3:6], out_still[3:6], atol=1e-12)
    np.testing.assert_allclose(out_moving[10:13], out_still[10:13], atol=1e-12)
