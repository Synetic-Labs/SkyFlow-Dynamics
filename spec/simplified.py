"""
Simplified point-mass model — the surrogate used for differentiable-simulation gradients.

A point mass with kinematic attitude, commanded by mass-normalized collective thrust c (m/s²,
along body ẑ) and body rate ω_cmd (rad/s), assuming instantaneous motors and perfect rate
tracking. This is rpg_flightning's `quadrotor_dyn`: its Jacobian is smooth, cheap, and free of
the stiff motor/drag modes, which is exactly why it substitutes for the full model's Jacobian
in the surrogate-gradient scheme (see below).

Discrete step (the form used for backprop-through-time):
    x⁺ = x + dt·v
    v⁺ = v + dt·( g_W + R(q)·(0, 0, c) )
    q⁺ = q ⊗ Exp(dt·ω_cmd)          Exp = rotation-vector → unit-quaternion map,
                                     Exp(φ) = (cos(θ/2), sin(θ/2)·φ/θ) — the HALF-angle form
                                     (quaternion.from_rotation_vector), not the literal
                                     quaternion exponential of (0, φ), which would double the
                                     rotation angle.
    ω⁺ = ω_cmd

⚠ Backend guard: Exp has a removable 0/0 at ω_cmd = 0 (hover!) — generated code must use the
Taylor form near zero (see quaternion.from_rotation_vector).

Surrogate gradient (straight-through estimator, framework-agnostic):
    y_out = y_simplified + stop_gradient( y_full − y_simplified )
Value: y_out ≡ y_full (full-fidelity forward rollout). Gradient: ∂y_out/∂θ ≡ ∂y_simplified/∂θ
(the simplified Jacobian replaces the true one). This is the reverse-mode (VJP) equivalent of
flightning's forward-mode `custom_jvp` substitution. Input mapping: c = Σᵢ T(Ω_c,i)/m from the
commanded speeds; ω_cmd from the body-rate command when the control interface provides one,
else the current ω. Motor state and wind pass through detached (the point mass has neither).

Source: Heeg, Song, Scaramuzza, "Learning Quadrotor Control From Visual Features Using
Differentiable Simulation", ICRA 2025 (rpg_flightning).
"""

import sympy as sp

from spec import quaternion
from spec.frames import gravity_world


def step(x: sp.Matrix, v: sp.Matrix, q: sp.Matrix, c: sp.Expr, w_cmd: sp.Matrix,
         dt: sp.Expr, grav: sp.Expr) -> tuple:
    """One discrete step of the point-mass model → (x⁺, v⁺, q⁺, ω⁺)."""
    R = quaternion.rotation_matrix(q)
    x_next = x + dt * v
    v_next = v + dt * (gravity_world(grav) + R * sp.Matrix([0, 0, c]))
    q_next = quaternion.product(q, quaternion.from_rotation_vector(dt * w_cmd))
    return x_next, v_next, q_next, w_cmd


def dynamics(v: sp.Matrix, q: sp.Matrix, c: sp.Expr, w_cmd: sp.Matrix,
             grav: sp.Expr) -> tuple:
    """Continuous form → (ẋ, v̇, q̇): rates follow the command exactly (ω ≡ ω_cmd)."""
    R = quaternion.rotation_matrix(q)
    return v, gravity_world(grav) + R * sp.Matrix([0, 0, c]), quaternion.kinematics(q, w_cmd)
