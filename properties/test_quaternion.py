"""Algebraic identities of the quaternion layer — proved symbolically where the expressions
are polynomial (using the unit-norm substitution), numerically at machine precision otherwise."""

import numpy as np
import sympy as sp

from spec import quaternion as Q
from spec.frames import hat
from properties.helpers import random_unit_quaternion, unit_subs


def _sym_quat(name):
    return sp.Matrix(sp.symbols(f"{name}_w {name}_x {name}_y {name}_z", real=True))


def test_product_associative():
    p, q, r = _sym_quat("p"), _sym_quat("q"), _sym_quat("r")
    lhs = Q.product(Q.product(p, q), r)
    rhs = Q.product(p, Q.product(q, r))
    assert sp.expand(lhs - rhs) == sp.zeros(4, 1)


def test_product_norm_multiplicative():
    p, q = _sym_quat("p"), _sym_quat("q")
    lhs = Q.norm_squared(Q.product(p, q))
    rhs = Q.norm_squared(p) * Q.norm_squared(q)
    assert sp.expand(lhs - rhs) == 0


def test_conjugate_is_inverse_for_unit():
    q = _sym_quat("q")
    prod = Q.product(q, Q.conjugate(q))
    identity = sp.Matrix([Q.norm_squared(q), 0, 0, 0])
    assert sp.expand(prod - identity) == sp.zeros(4, 1)


def test_rotation_matrix_orthogonal_and_special():
    q = _sym_quat("q")
    R = Q.rotation_matrix(q)
    should_be_identity = unit_subs(R.T * R, q)
    assert should_be_identity == sp.eye(3)
    assert unit_subs(R.det(), q) == 1


def test_rotation_matrix_matches_sandwich():
    q = _sym_quat("q")
    r = sp.Matrix(sp.symbols("r_1 r_2 r_3", real=True))
    diff = Q.rotation_matrix(q) * r - Q.rotate(q, r)
    assert all(unit_subs(d, q) == 0 for d in diff)


def test_kinematics_preserves_norm():
    # d/dt ‖q‖² = 2 qᵀ q̇ = qᵀ (q ⊗ (0,ω)) = 0 — holds for ANY q, not just unit.
    q = _sym_quat("q")
    w = sp.Matrix(sp.symbols("w_1 w_2 w_3", real=True))
    assert sp.expand((q.T * Q.kinematics(q, w))[0, 0]) == 0


def test_rotation_matrix_derivative_is_R_hat_omega():
    # Along q̇ = ½ q ⊗ (0,ω):  Ṙ(q) = R(q)·hat(ω)  for unit q — ties the quaternion
    # kinematics to the rotation kinematics; a sign error anywhere breaks it.
    q = _sym_quat("q")
    w = sp.Matrix(sp.symbols("w_1 w_2 w_3", real=True))
    q_dot = Q.kinematics(q, w)
    R = Q.rotation_matrix(q)
    R_dot = sp.Matrix(3, 3, lambda i, j: sum(sp.diff(R[i, j], q[k]) * q_dot[k]
                                             for k in range(4)))
    diff = R_dot - R * hat(w)
    assert all(unit_subs(d, q) == 0 for d in diff)


def test_from_rotation_vector_unit_norm():
    phi = sp.Matrix(sp.symbols("phi_1 phi_2 phi_3", real=True, positive=True))
    assert sp.simplify(Q.norm_squared(Q.from_rotation_vector(phi)) - 1) == 0


def test_from_rotation_vector_axis_rotations():
    theta = sp.Symbol("theta", positive=True)
    # Rotation about ẑ by θ must reproduce the standard Rz(θ).
    Rz = Q.rotation_matrix(Q.from_rotation_vector(sp.Matrix([0, 0, 1]) * theta))
    expected = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                          [sp.sin(theta), sp.cos(theta), 0],
                          [0, 0, 1]])
    assert sp.simplify(Rz - expected) == sp.zeros(3, 3)


def test_body_to_world_convention_numeric():
    # 90° about world/body z maps body x̂ to world ŷ: r_W = R(q) r_B.
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    fn = sp.lambdify(sp.symbols("q_w q_x q_y q_z"), Q.rotation_matrix(_sym_quat("q")))
    R = np.asarray(fn(*q), dtype=float)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-15)


def test_double_cover_numeric():
    rng = np.random.default_rng(7)
    q_syms = _sym_quat("q")
    fn = sp.lambdify(list(q_syms), Q.rotation_matrix(q_syms))
    for _ in range(20):
        q = random_unit_quaternion(rng)
        np.testing.assert_allclose(np.asarray(fn(*q), float),
                                   np.asarray(fn(*(-q)), float), atol=1e-14)
