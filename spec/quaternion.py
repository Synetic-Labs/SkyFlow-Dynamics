"""
Quaternion algebra — scalar-first (w, x, y, z), Hamilton convention (i·j = k).

A unit quaternion q represents the body→world rotation: for a vector with body coordinates
r_B, its world coordinates are  r_W = R(q)·r_B, equivalently  (0, r_W) = q ⊗ (0, r_B) ⊗ q*.

Kinematics for body-frame angular velocity ω_B:   q̇ = ½ · q ⊗ (0, ω_B).

All functions take and return sympy column Matrices; quaternions are 4×1 (w, x, y, z),
vectors 3×1. None of them assume unit norm unless stated.
"""

import sympy as sp


def product(p: sp.Matrix, q: sp.Matrix) -> sp.Matrix:
    """Hamilton product p ⊗ q (scalar-first)."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return sp.Matrix([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def conjugate(q: sp.Matrix) -> sp.Matrix:
    """q* = (w, −x, −y, −z). For unit q this is the inverse."""
    return sp.Matrix([q[0], -q[1], -q[2], -q[3]])


def norm_squared(q: sp.Matrix) -> sp.Expr:
    return q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2


def rotation_matrix(q: sp.Matrix) -> sp.Matrix:
    """
    Body→world rotation matrix R(q) for a UNIT quaternion:  r_W = R(q)·r_B.

        R = (q_w² − q_v·q_v)·I + 2·q_v q_vᵀ + 2·q_w·hat(q_v),   q_v = (q_x, q_y, q_z).
    """
    w, x, y, z = q
    return sp.Matrix([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x**2 + y**2)],
    ])


def rotate(q: sp.Matrix, r: sp.Matrix) -> sp.Matrix:
    """Rotate a 3-vector by the quaternion sandwich: vector part of q ⊗ (0, r) ⊗ q*."""
    return product(product(q, sp.Matrix([0, r[0], r[1], r[2]])), conjugate(q))[1:4, 0]


def kinematics(q: sp.Matrix, omega_body: sp.Matrix) -> sp.Matrix:
    """q̇ = ½ · q ⊗ (0, ω_B) — attitude rate for body-frame angular velocity."""
    return product(q, sp.Matrix([0, omega_body[0], omega_body[1], omega_body[2]])) / 2


def from_rotation_vector(phi: sp.Matrix) -> sp.Matrix:
    """
    Exponential map: rotation vector φ (rad, axis·angle) → unit quaternion
        q = (cos(θ/2), sin(θ/2)·φ/θ),  θ = ‖φ‖.
    Written with sinc-style structure so the θ→0 limit is (1, φ/2) — but note the symbolic
    expression contains θ in denominators; evaluate the series form for tiny θ numerically.
    """
    theta = sp.sqrt(phi[0]**2 + phi[1]**2 + phi[2]**2)
    return sp.Matrix([
        sp.cos(theta / 2),
        sp.sin(theta / 2) * phi[0] / theta,
        sp.sin(theta / 2) * phi[1] / theta,
        sp.sin(theta / 2) * phi[2] / theta,
    ])
