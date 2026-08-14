"""Point-mass surrogate model: exact special cases, unit-norm preservation, and consistency
with its own continuous form as dt → 0."""

import numpy as np
import sympy as sp

from properties.helpers import random_unit_quaternion
from skyflow_dynamics.spec import simplified


def _sym(name, n=3):
    return sp.Matrix(sp.symbols(f"{name}_1:{n+1}", real=True))


def test_ballistic_exact():
    # c = 0, ω_cmd = 0: pure projectile step, q unchanged (exp(0) = identity — checked
    # numerically as the symbolic form has the removable 0/0 at θ=0).
    x, v = _sym("x"), _sym("v")
    dt, g = sp.symbols("dt g", positive=True)
    q = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))
    x2, v2, _, _ = simplified.step(x, v, q, 0, sp.Matrix([0, 0, 0]) + sp.Matrix(_sym("eps")) * 0, dt, g)
    assert sp.simplify(x2 - (x + dt * v)) == sp.zeros(3, 1)
    assert sp.simplify(v2 - (v + dt * sp.Matrix([0, 0, -g]))) == sp.zeros(3, 1)


def test_hover_exact():
    # Identity attitude and c = g: velocity unchanged.
    x, v = _sym("x"), _sym("v")
    dt, g = sp.symbols("dt g", positive=True)
    q = sp.Matrix([1, 0, 0, 0])
    _, v2, _, _ = simplified.step(x, v, q, g, _sym("w"), dt, g)
    assert sp.simplify(v2 - v) == sp.zeros(3, 1)


def test_quaternion_stays_unit_numeric():
    rng = np.random.default_rng(9)
    x, v = _sym("x"), _sym("v")
    q = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))
    w = _sym("w")
    dt, g, c = sp.symbols("dt g c", positive=True)
    _, _, q2, _ = simplified.step(x, v, q, c, w, dt, g)
    fn = sp.lambdify((q.flat(), w.flat(), dt), q2.T * q2, modules="numpy")
    for _ in range(20):
        qn = random_unit_quaternion(rng)
        wn = rng.uniform(-6, 6, 3)
        assert abs(np.asarray(fn(qn, wn, 0.01)).item() - 1.0) < 1e-14


def test_step_consistent_with_continuous_dynamics():
    # (s⁺ − s)/dt → ṡ as dt → 0, for every block including the quaternion exponential.
    rng = np.random.default_rng(31)
    x, v = _sym("x"), _sym("v")
    q = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))
    w = _sym("w")
    dt, g, c = sp.symbols("dt g c", positive=True)

    step_out = simplified.step(x, v, q, c, w, dt, g)
    cont_out = simplified.dynamics(v, q, c, w, g)

    args = (x.flat(), v.flat(), q.flat(), w.flat(), c, g, dt)
    fd = {name: sp.lambdify(args, (step_out[i] - [x, v, q][i]) / dt, modules="numpy")
          for name, i in (("x", 0), ("v", 1), ("q", 2))}
    ct = {name: sp.lambdify(args[:-1], cont_out[i], modules="numpy")
          for name, i in (("x", 0), ("v", 1), ("q", 2))}

    for _ in range(5):
        vals = (rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3), random_unit_quaternion(rng),
                rng.uniform(-4, 4, 3), rng.uniform(1, 20), 9.81)
        for name in ("x", "v", "q"):
            approx = np.asarray(fd[name](*vals, 1e-7), float).ravel()
            exact = np.asarray(ct[name](*vals), float).ravel()
            np.testing.assert_allclose(approx, exact, atol=1e-5)


def test_rate_command_is_body_frame():
    # ω_cmd about body z from a 90°-rolled attitude must rotate about the WORLD x-axis
    # direction consistent with q ⊗ exp(dt·ω): body-frame right multiplication.
    roll90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])  # +90° about body x
    xs, vs = _sym("x"), _sym("v")
    qs = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))
    ws = _sym("w")
    dt, g, c = sp.symbols("dt g c", positive=True)
    _, _, q2, _ = simplified.step(xs, vs, qs, c, ws, dt, g)
    fn = sp.lambdify((qs.flat(), ws.flat(), dt), q2, modules="numpy")
    got = np.asarray(fn(roll90, [0, 0, 2.0], 0.01), float).ravel()
    # Independent check: q ⊗ exp — body-frame composition (right multiply).
    half = 0.5 * 0.02  # ½·‖dt·ω‖
    dq = np.array([np.cos(half), 0, 0, np.sin(half)])
    aw, ax, ay, az = roll90
    bw, bx, by, bz = dq
    expected = np.array([aw*bw - ax*bx - ay*by - az*bz,
                         aw*bx + ax*bw + ay*bz - az*by,
                         aw*by - ax*bz + ay*bw + az*bx,
                         aw*bz + ax*by - ay*bx + az*bw])
    np.testing.assert_allclose(got, expected, atol=1e-14)
