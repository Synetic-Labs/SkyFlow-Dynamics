"""
The canonical continuous-time model:  ṡ = f(s, u, θ)   (state, inputs, parameters).

Assembles motor dynamics, the body wrench, and the rigid-body equations into the full state
derivative. This single symbolic function is what golden vectors pin down and what backends
are generated from.

State ordering (must match symbols.State.flat() and the golden files):
    ṡ = (ẋ(3), v̇(3), q̇(4, wxyz), ω̇(3), Ω̇(n))
"""

import sympy as sp

from spec import motor, quaternion, rigid_body, wrench
from spec.symbols import Inputs, Params, State

#: Selectable motor models for statedot().
MOTOR_MODELS = ("first_order", "asymmetric")


def body_airspeed(state: State, inputs: Inputs) -> sp.Matrix:
    """CoM airspeed in the body frame: v_a = R(q)ᵀ · (v − v_wind)."""
    R = quaternion.rotation_matrix(state.q)
    return R.T * (state.v - inputs.v_wind)


def rotor_acceleration(state: State, inputs: Inputs, p: Params,
                       motor_model: str = "first_order") -> sp.Matrix:
    """Ω̇ per the selected motor model."""
    if motor_model == "first_order":
        return motor.first_order_lag(state.W, inputs.W_c, p.tau_m)
    if motor_model == "asymmetric":
        return motor.asymmetric_lag(state.W, inputs.W_c, p.ka1, p.ka2, p.kd1, p.kd2)
    raise ValueError(f"motor_model must be one of {MOTOR_MODELS}")


def statedot(state: State, inputs: Inputs, p: Params,
             motor_model: str = "first_order") -> sp.Matrix:
    """Full state derivative, stacked (13 + n) × 1."""
    v_a = body_airspeed(state, inputs)
    W_dot = rotor_acceleration(state, inputs, p, motor_model)

    F_B, M_B = wrench.body_wrench(state.w, state.W, v_a, p)
    M_B = M_B + wrench.rotor_inertia_moment(state.w, state.W, W_dot, p)

    x_dot, v_dot = rigid_body.translational(state.v, state.q, F_B, inputs.F_ext,
                                            p.mass, p.grav)
    q_dot, w_dot = rigid_body.rotational(state.q, state.w, M_B, inputs.tau_ext, p.inertia)

    return sp.Matrix.vstack(x_dot, v_dot, q_dot, w_dot, W_dot)
