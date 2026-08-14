"""Rigid-body invariants: equilibria and conservation laws."""

import numpy as np
import sympy as sp

from properties.helpers import (
    P,
    S,
    U,
    flat_params,
    hover_speed,
    make_inputs,
    params_dict,
    statedot_fn,
)
from skyflow_dynamics.spec import quaternion as Q
from skyflow_dynamics.spec.discretization import rk4_step
from skyflow_dynamics.spec.frames import cross, hat


def test_hover_is_exact_equilibrium():
    vals = params_dict()
    w_h = hover_speed(vals)
    s = np.concatenate([[0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], np.full(4, w_h)])
    out = statedot_fn()(s, make_inputs(W_c=np.full(4, w_h)), flat_params(vals))
    np.testing.assert_allclose(out, 0.0, atol=1e-12)


def test_hover_speed_symbolic():
    # Solve Σᵢ ct2·Ω² = m·g for the symmetric quadratic vehicle out of the actual v̇_z expression.
    from skyflow_dynamics.spec.dynamics import statedot
    w_sym = sp.Symbol("Omega_h", positive=True)
    subs = {}
    for i in range(4):
        subs[S.W[i]] = w_sym
        subs[U.W_c[i]] = w_sym
        subs[P.ct0[i]] = 0
        subs[P.ct1[i]] = 0
        subs[P.ct2[i]] = P.ct2[0]
        subs[P.axis[i][0]] = 0
        subs[P.axis[i][1]] = 0
        subs[P.axis[i][2]] = 1
    for sym in (*S.x, *S.v, *S.w, *U.v_wind, *U.F_ext, *U.tau_ext):
        subs[sym] = 0
    for sym, v in zip(S.q.flat(), (1, 0, 0, 0)):
        subs[sym] = v
    vz_dot = statedot(S, U, P).flat()[5].subs(subs)
    sols = sp.solve(sp.Eq(vz_dot, 0), w_sym)
    expected = sp.sqrt(P.mass * P.grav / (4 * P.ct2[0]))
    assert any(sp.simplify(s - expected) == 0 for s in sols)


def test_torque_free_angular_momentum_conserved_symbolically():
    # With zero wrench, L_W = R(q)·I·ω is conserved:  d/dt L_W = R(ω̂·I·ω + I·ω̇) = 0
    # when ω̇ = I⁻¹(−ω × I·ω). Proves Euler's equation and the attitude kinematics cohere.
    q = sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True))
    w = sp.Matrix(sp.symbols("w_1 w_2 w_3", real=True))
    I = P.inertia
    w_dot = I.inv() * (-cross(w, I * w))
    L_dot = Q.rotation_matrix(q) * (hat(w) * (I * w) + I * w_dot)
    assert sp.simplify(L_dot) == sp.zeros(3, 1)


def test_torque_free_energy_conserved_numerically():
    # Kill every wrench source; integrate tumbling with fine RK4; E = ½ωᵀIω and ‖L_W‖ hold.
    vals = params_dict(ct0=[0.0] * 4, ct1=[0.0] * 4, ct2=[0.0] * 4,
                       cq0=[0.0] * 4, cq1=[0.0] * 4, cq2=[0.0] * 4,
                       k_d=0.0, k_z=0.0)
    p = flat_params(vals)
    inputs = make_inputs(W_c=np.zeros(4))
    f = statedot_fn()
    I = np.array(vals["inertia"])

    s = np.concatenate([[0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [1.5, -2.0, 1.0], np.zeros(4)])

    def energy(s):
        w = s[10:13]
        return 0.5 * w @ I @ w

    def ang_mom_world(s):
        qw, qx, qy, qz = s[6:10] / np.linalg.norm(s[6:10])
        R = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)]])
        return R @ (I @ s[10:13])

    E0, L0 = energy(s), ang_mom_world(s)
    dt = 1e-4
    for _ in range(2000):  # 0.2 s of tumbling
        s = rk4_step(lambda t, x: f(x, inputs, p), s, dt)
        s[6:10] /= np.linalg.norm(s[6:10])
    assert abs(energy(s) - E0) / E0 < 1e-10
    np.testing.assert_allclose(ang_mom_world(s), L0, rtol=1e-8)


def test_gravity_only_ballistic():
    # No thrust/aero: v̇ = (0,0,−g) exactly, ẋ = v.
    vals = params_dict(ct0=[0.0] * 4, ct1=[0.0] * 4, ct2=[0.0] * 4,
                       cq0=[0.0] * 4, cq1=[0.0] * 4, cq2=[0.0] * 4,
                       k_d=0.0, k_z=0.0)
    rng = np.random.default_rng(3)
    s = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3),
                        [1, 0, 0, 0], [0, 0, 0], np.zeros(4)])
    out = statedot_fn()(s, make_inputs(W_c=np.zeros(4)), flat_params(vals))
    np.testing.assert_allclose(out[0:3], s[3:6], atol=1e-15)
    np.testing.assert_allclose(out[3:6], [0, 0, -vals["grav"]], atol=1e-15)


def test_external_wrench_injection_exact():
    # F_ext adds F/m to v̇ (world); τ_ext adds I⁻¹τ to ω̇ — nothing else changes.
    vals = params_dict()
    p = flat_params(vals)
    rng = np.random.default_rng(11)
    from properties.helpers import random_state
    s = random_state(rng)
    F, tau = np.array([0.02, -0.01, 0.03]), np.array([1e-4, -2e-4, 5e-5])
    base = statedot_fn()(s, make_inputs(), p)
    pert = statedot_fn()(s, make_inputs(F_ext=F, tau_ext=tau), p)
    d = pert - base
    np.testing.assert_allclose(d[3:6], F / vals["mass"], rtol=1e-12)
    np.testing.assert_allclose(d[10:13],
                               np.linalg.inv(np.array(vals["inertia"])) @ tau, rtol=1e-12)
    np.testing.assert_allclose(np.delete(d, [3, 4, 5, 10, 11, 12]), 0.0, atol=1e-15)


def test_yaw_symmetry():
    # The model has no world-frame preference in the horizontal plane: yawing the whole
    # problem (attitude, velocity, wind) by ψ yaws the derivative identically.
    vals = params_dict(c_D=[0.05, 0.05, 0.08], k_flap=1e-7)  # exercise aero too
    p = flat_params(vals)
    f = statedot_fn()
    rng = np.random.default_rng(5)
    from properties.helpers import random_state
    psi = 0.7
    c, s_ = np.cos(psi), np.sin(psi)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    q_psi = np.array([np.cos(psi / 2), 0, 0, np.sin(psi / 2)])

    def quat_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([aw * bw - ax * bx - ay * by - az * bz,
                         aw * bx + ax * bw + ay * bz - az * by,
                         aw * by - ax * bz + ay * bw + az * bx,
                         aw * bz + ax * by - ay * bx + az * bw])

    for _ in range(5):
        st = random_state(rng)
        wind = rng.uniform(-2, 2, 3)
        out = f(st, make_inputs(v_wind=wind), p)

        st2 = st.copy()
        st2[0:3] = Rz @ st[0:3]
        st2[3:6] = Rz @ st[3:6]
        st2[6:10] = quat_mul(q_psi, st[6:10])
        out2 = f(st2, make_inputs(v_wind=Rz @ wind), p)

        np.testing.assert_allclose(out2[0:3], Rz @ out[0:3], atol=1e-10)
        np.testing.assert_allclose(out2[3:6], Rz @ out[3:6], atol=1e-10)
        np.testing.assert_allclose(out2[6:10], quat_mul(q_psi, out[6:10]), atol=1e-10)
        np.testing.assert_allclose(out2[10:13], out[10:13], atol=1e-10)  # body-frame: unchanged
