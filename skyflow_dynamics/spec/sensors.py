"""
Sensor measurement models.

The deterministic part is the exact measurement given the true state and its derivative.
For the stochastic error processes, the spec carries the exact propagation equations and
noise-scaling conventions (the class of math where scaling bugs hide — cf. the Dryden
√(π/dt) findings); the unit-normal draws themselves, the once-per-power-up turn-on-bias
draw, sample-rate / zero-order-hold behavior, and artifact spikes are harness-side.
"""

import sympy as sp

from skyflow_dynamics.spec import quaternion
from skyflow_dynamics.spec.frames import cross, gravity_world


def imu(q: sp.Matrix, v_dot: sp.Matrix, w: sp.Matrix, w_dot: sp.Matrix,
        p_BS: sp.Matrix, R_BS: sp.Matrix, grav: sp.Expr) -> tuple:
    """
    IMU at body-frame offset p_BS with mounting rotation R_BS (sensor→body):

        a_S,W  = v̇ + R_WB · ( ω̇ × p_BS + ω × (ω × p_BS) )     (world accel of the sensor point)
        accel  = R_BSᵀ · R_WBᵀ · ( a_S,W − g_W )                (specific force, sensor frame)
        gyro   = R_BSᵀ · ω                                       (sensor frame)

    The lever-arm cross products are evaluated in the BODY frame before rotating to world —
    mixing frames there is a real observed defect class (findings F-1/F-2: body-frame ω crossed
    with a world-frame lever arm, and a gyro that ignored R_BS; invisible at the default
    p_BS = 0, R_BS = I).
    Returns (accel, gyro).
    """
    R_WB = quaternion.rotation_matrix(q)
    a_sensor_world = v_dot + R_WB * (cross(w_dot, p_BS) + cross(w, cross(w, p_BS)))
    accel = R_BS.T * R_WB.T * (a_sensor_world - gravity_world(grav))
    gyro = R_BS.T * w
    return accel, gyro


def imu_bias_gauss_markov_step(b: sp.Matrix, tau: sp.Expr, sigma_b: sp.Expr,
                               dt: sp.Expr, w: sp.Matrix) -> sp.Matrix:
    """
    Exact one-step update of a first-order Gauss–Markov (Ornstein–Uhlenbeck) sensor bias
    ḃ = −b/τ + σ_b·ẇ, driven by a unit-normal sample vector w (harness-side draw):

        b⁺ = e^(−dt/τ) · b  +  σ_b · √( (τ/2) · (1 − e^(−2·dt/τ)) ) · w

    Transition and driving standard deviation are the exact discretization of the OU
    process (Maybeck Vol. 1, Eq. 4-114, as cited by the RotorS implementation).
    Limits: stationary variance σ_b²·τ/2 as dt→∞; pure random walk σ_b·√dt as τ→∞.
    """
    phi = sp.exp(-dt / tau)
    sigma_d = sigma_b * sp.sqrt(tau / 2 * (1 - sp.exp(-2 * dt / tau)))
    return phi * b + sigma_d * w


def imu_corrupt(y_true: sp.Matrix, b: sp.Matrix, b_on: sp.Matrix,
                sigma_nd: sp.Expr, dt: sp.Expr, n: sp.Matrix) -> sp.Matrix:
    """
    Additive measurement corruption for one sensor triad:

        y = y_true + b + b_on + (σ_nd/√dt) · n

    σ_nd is the continuous noise density ([unit]/√Hz); σ_nd/√dt is the discrete standard
    deviation of an integrating sampler over dt. b is the Gauss–Markov bias state, b_on
    the turn-on bias (drawn once at power-up), n a unit-normal sample vector — both draws
    harness-side.
    """
    return y_true + b + b_on + sigma_nd / sp.sqrt(dt) * n
