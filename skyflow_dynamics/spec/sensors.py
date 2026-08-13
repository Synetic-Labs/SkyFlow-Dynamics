"""
Sensor measurement models (deterministic part).

Noise processes (white noise density, bias random walks, artifact spikes) and sample-rate /
zero-order-hold behavior are harness-side; the math here is the exact measurement given the
true state and its derivative.
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
