"""IMU measurement model: defaults, static reading, lever-arm, and mount rotation — the
symbolic guards for the F-1/F-2 frame-mixing defect class."""

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import quaternion as Q
from skyflow_dynamics.spec import sensors


def _sym(name, n=3):
    return sp.Matrix(sp.symbols(f"{name}_1:{n+1}", real=True))


Qs = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))


def test_default_mount_reduces_to_specific_force():
    # p_BS = 0, R_BS = I:  accel = Rᵀ(v̇ − g_W), gyro = ω — exactly.
    vdot, w, wdot = _sym("vdot"), _sym("w"), _sym("wdot")
    g = sp.Symbol("g", positive=True)
    accel, gyro = sensors.imu(Qs, vdot, w, wdot, sp.zeros(3, 1), sp.eye(3), g)
    expected = Q.rotation_matrix(Qs).T * (vdot - sp.Matrix([0, 0, -g]))
    assert sp.expand(accel - expected) == sp.zeros(3, 1)
    assert gyro == w


def test_static_level_reads_plus_g():
    g = sp.Symbol("g", positive=True)
    accel, _ = sensors.imu(sp.Matrix([1, 0, 0, 0]), sp.zeros(3, 1), sp.zeros(3, 1),
                           sp.zeros(3, 1), sp.zeros(3, 1), sp.eye(3), g)
    assert sp.simplify(accel - sp.Matrix([0, 0, g])) == sp.zeros(3, 1)


def test_gyro_respects_mount_rotation():
    # F-2 guard: gyro = R_BSᵀ·ω, not ω.
    w = _sym("w")
    g = sp.Symbol("g", positive=True)
    th = sp.Symbol("theta", real=True)
    R_BS = sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
    _, gyro = sensors.imu(Qs, _sym("vdot"), w, _sym("wdot"), sp.zeros(3, 1), R_BS, g)
    assert sp.simplify(gyro - R_BS.T * w) == sp.zeros(3, 1)


def test_lever_arm_centripetal():
    # Level vehicle spinning at ω = r·ẑ with the IMU at p_BS = (d,0,0): the sensor point
    # runs a circle of radius d → centripetal specific force −r²·d·x̂ plus gravity +g·ẑ.
    g_val, r_val, d_val = 9.81, 3.0, 0.05
    vdot, w, wdot = _sym("vdot"), _sym("w"), _sym("wdot")
    p = _sym("p")
    g = sp.Symbol("g", positive=True)
    accel, _ = sensors.imu(sp.Matrix([1, 0, 0, 0]), vdot, w, wdot, p, sp.eye(3), g)
    fn = sp.lambdify((vdot.flat(), w.flat(), wdot.flat(), p.flat(), g), accel, modules="numpy")
    got = np.asarray(fn([0, 0, 0], [0, 0, r_val], [0, 0, 0], [d_val, 0, 0], g_val), float).ravel()
    np.testing.assert_allclose(got, [-r_val**2 * d_val, 0, g_val], atol=1e-14)


def test_lever_arm_frames_consistent():
    # F-1 guard: lever-arm terms are body-frame quantities rotated ONCE. Rotating the whole
    # scenario (attitude) must not change the body-frame IMU reading.
    rng = np.random.default_rng(17)
    vdot_w, w_b, wdot_b = _sym("vdot"), _sym("w"), _sym("wdot")
    p = _sym("p")
    g = sp.Symbol("g", positive=True)
    accel, _ = sensors.imu(Qs, vdot_w, w_b, wdot_b, p, sp.eye(3), g)
    fn = sp.lambdify((Qs.flat(), vdot_w.flat(), w_b.flat(), wdot_b.flat(), p.flat(), g),
                     accel, modules="numpy")

    from properties.helpers import random_unit_quaternion
    w_val = rng.uniform(-3, 3, 3)
    wdot_val = rng.uniform(-10, 10, 3)
    p_val = [0.02, -0.01, 0.005]

    # Free fall (v̇ = g_W) isolates the lever-arm terms: reading must be attitude-invariant.
    for _ in range(10):
        q1 = random_unit_quaternion(rng)
        q2 = random_unit_quaternion(rng)
        a1 = np.asarray(fn(q1, [0, 0, -9.81], w_val, wdot_val, p_val, 9.81), float).ravel()
        a2 = np.asarray(fn(q2, [0, 0, -9.81], w_val, wdot_val, p_val, 9.81), float).ravel()
        np.testing.assert_allclose(a1, a2, atol=1e-12)
        expected = np.cross(wdot_val, p_val) + np.cross(w_val, np.cross(w_val, p_val))
        np.testing.assert_allclose(a1, expected, atol=1e-12)
