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
    pw, px, py, pz = p.flat()
    qw, qx, qy, qz = q.flat()
    return sp.Matrix([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def conjugate(q: sp.Matrix) -> sp.Matrix:
    """q* = (w, −x, −y, −z). For unit q this is the inverse."""
    w, x, y, z = q.flat()
    return sp.Matrix([w, -x, -y, -z])


def norm_squared(q: sp.Matrix) -> sp.Expr:
    w, x, y, z = q.flat()
    return w**2 + x**2 + y**2 + z**2


def rotation_matrix(q: sp.Matrix) -> sp.Matrix:
    """
    Body→world rotation matrix R(q):  r_W = R(q)·r_B, in the scale-invariant homogeneous form

        R = [ (q_w² − q_v·q_v)·I + 2·q_v q_vᵀ + 2·q_w·hat(q_v) ] / ‖q‖²,  q_v = (q_x, q_y, q_z).

    On the unit manifold this is the textbook unit-quaternion matrix. The ‖q‖² division makes
    R exact for any q ≠ 0 (R(λq) = R(q)) — deliberately: fixed-step integrators evaluate the
    dynamics at stage states whose quaternion has drifted off unit norm, and reference
    implementations (scipy Rotation, and most libraries) normalize there. Without the division
    the spec's RK4 step disagrees with the references at O(‖q‖²−1) — found by the rk4-step
    golden vectors.
    """
    w, x, y, z = q.flat()
    n = w**2 + x**2 + y**2 + z**2
    return sp.Matrix([
        [w**2 + x**2 - y**2 - z**2, 2*(x*y - w*z),             2*(x*z + w*y)],
        [2*(x*y + w*z),             w**2 - x**2 + y**2 - z**2, 2*(y*z - w*x)],
        [2*(x*z - w*y),             2*(y*z + w*x),             w**2 - x**2 - y**2 + z**2],
    ]) / n


def rotate(q: sp.Matrix, r: sp.Matrix) -> sp.Matrix:
    """
    Rotate a 3-vector: vector part of q ⊗ (0, r) ⊗ q*, divided by ‖q‖².
    The raw sandwich scales by ‖q‖² for non-unit q; the division makes rotate(q, r) equal
    rotation_matrix(q)·r for ALL q ≠ 0, consistent with rotation_matrix's deliberate
    scale-invariance (off-manifold RK stage states — see rotation_matrix's docstring).
    """
    rx, ry, rz = r.flat()
    _, sx, sy, sz = product(product(q, sp.Matrix([0, rx, ry, rz])), conjugate(q)).flat()
    return sp.Matrix([sx, sy, sz]) / norm_squared(q)


def kinematics(q: sp.Matrix, omega_body: sp.Matrix) -> sp.Matrix:
    """q̇ = ½ · q ⊗ (0, ω_B) — attitude rate for body-frame angular velocity."""
    wx, wy, wz = omega_body.flat()
    return product(q, sp.Matrix([0, wx, wy, wz])) / 2


def kinematics_norm_corrected(q: sp.Matrix, omega_body: sp.Matrix,
                              K: sp.Expr | float) -> sp.Matrix:
    """
    Kinematics with Lagrange-multiplier norm-drift stabilization (gain K ≥ 0, units 1/s):

        q̇ = ½ · q ⊗ (0, ω_B) + K·ε·q,    ε = 1 − ‖q‖²

    The ⊗ term is norm-orthogonal (qᵀ(q ⊗ (0, ω)) = 0), so d‖q‖²/dt = 2K·ε·‖q‖²: under
    integration error the norm no longer drifts but decays back to 1 (rate ≈ 2K near the
    unit manifold). Reduces to kinematics() exactly on ‖q‖ = 1. This is the smooth,
    differentiable alternative to post-step renormalization (harness-side today) — the
    correction stays inside the ODE, so backends that differentiate through the integrator
    see a single smooth vector field. Choose K·dt ≪ 1 or the correction dominates the step.
    Sources: standard flight-simulation practice (Stevens & Lewis; Zipfel); exact form as
    printed in the MathWorks Aerospace Blockset '6DOF (Quaternion)' block documentation.
    """
    return kinematics(q, omega_body) + K * (1 - norm_squared(q)) * q


def from_rotation_vector(phi: sp.Matrix) -> sp.Matrix:
    """
    Exponential map: rotation vector φ (rad, axis·angle) → unit quaternion
        q = (cos(θ/2), sin(θ/2)·φ/θ),  θ = ‖φ‖.
    ⚠ The singularity at θ = 0 is removable (the limit is (1, φ/2)) but NOT removed here:
    the expression literally contains θ in denominators, so naive codegen/autodiff evaluates
    0/0 (NaN value and NaN gradients) at φ = 0 — an ordinary input (hover, zero rate
    command; spec.simplified.step composes this at dt·ω_cmd). Backends MUST substitute the
    Taylor form for small θ:  q ≈ (1 − θ²/8,  φ·(1/2 − θ²/48)).
    """
    px, py, pz = phi.flat()
    theta = sp.sqrt(px**2 + py**2 + pz**2)
    return sp.Matrix([
        sp.cos(theta / 2),
        sp.sin(theta / 2) * px / theta,
        sp.sin(theta / 2) * py / theta,
        sp.sin(theta / 2) * pz / theta,
    ])
