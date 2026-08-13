"""
Rigid-body equations of motion (Newton–Euler) with quaternion attitude.

    ẋ = v
    v̇ = g_W + ( R(q)·F_B + F_ext_W ) / m
    q̇ = ½ · q ⊗ (0, ω)
    ω̇ = I⁻¹ · ( M_B + τ_ext_B − ω × (I·ω) )

x, v in the world frame; ω, F_B, M_B, τ_ext in the body frame; F_ext in the world frame.
I is the full 3×3 body inertia matrix (products of inertia supported).
"""

import sympy as sp

from skyflow_dynamics.spec import quaternion
from skyflow_dynamics.spec.frames import cross, gravity_world


def translational(v: sp.Matrix, q: sp.Matrix, F_B: sp.Matrix, F_ext: sp.Matrix,
                  mass: sp.Expr, grav: sp.Expr) -> tuple:
    """(ẋ, v̇): world-frame kinematics and Newton's second law."""
    R = quaternion.rotation_matrix(q)
    return v, gravity_world(grav) + (R * F_B + F_ext) / mass


def rotational(q: sp.Matrix, w: sp.Matrix, M_B: sp.Matrix, tau_ext: sp.Matrix,
               inertia: sp.Matrix) -> tuple:
    """(q̇, ω̇): quaternion kinematics and Euler's equation in the body frame."""
    q_dot = quaternion.kinematics(q, w)
    w_dot = inertia.inv() * (M_B + tau_ext - cross(w, inertia * w))
    return q_dot, w_dot
