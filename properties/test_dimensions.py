"""Dimensional consistency via scaling invariance.

A physically consistent equation is invariant under a change of units: rescaling every
quantity by λ_L^a·λ_M^b·λ_T^c (its length/mass/time dimensions) must rescale each output by
exactly its own dimensions. Run over the FULL dynamics with every term active, this catches
unit errors (mass-normalized coefficients, missing 1/m or I⁻¹, mixed-frame products) that
point tests miss. Angles are dimensionless (rad).
"""

import numpy as np

from properties.helpers import N, flat_params, make_inputs, params_dict, random_state, statedot_fn

# (L, M, T) exponents.
DIM_STATE = [(1, 0, 0)] * 3 + [(1, 0, -1)] * 3 + [(0, 0, 0)] * 4 + [(0, 0, -1)] * 3 + [(0, 0, -1)] * N
DIM_INPUT = [(0, 0, -1)] * N + [(1, 0, -1)] * 3 + [(1, 1, -2)] * 3 + [(2, 1, -2)] * 3
DIM_OUT = [(1, 0, -1)] * 3 + [(1, 0, -2)] * 3 + [(0, 0, -1)] * 4 + [(0, 0, -2)] * 3 + [(0, 0, -2)] * N
DIM_PARAM = (
    [(0, 1, 0), (1, 0, -2)]                    # mass, grav
    + [(2, 1, 0)] * 6                          # inertia components
    + [(1, 0, 0)] * (3 * N)                    # rotor positions
    + [(0, 0, 0)] * N                          # spin signs
    + [(0, 0, 0)] * (3 * N)                    # unit axes
    + [(1, 1, -2)] * N + [(1, 1, -1)] * N + [(1, 1, 0)] * N     # ct0, ct1, ct2
    + [(2, 1, -2)] * N + [(2, 1, -1)] * N + [(2, 1, 0)] * N     # cq0, cq1, cq2
    + [(0, 0, 1), (0, 0, -1), (0, 0, 0), (0, 0, -1), (0, 0, 0)]  # tau_m, ka1, ka2, kd1, kd2
    + [(2, 1, 0)]                              # I_rot
    + [(-1, 1, 0)] * 3                         # c_D  (kg/m)
    + [(0, 1, 0), (0, 1, 0)]                   # k_d, k_z  (kg)
    + [(1, 1, 0)]                              # k_flap (kg·m)
    + [(-1, 1, 0)]                             # k_h (kg/m)
    + [(0, 0, 0), (0, 0, 0)]                   # k_angle, k_hor
    + [(-1, 1, 0)]                             # k_v2 (kg/m)
    + [(1, 0, 0)]                              # r_prop
)


def _scale(dims, lL, lM, lT):
    return np.array([lL**a * lM**b * lT**c for a, b, c in dims])


def _everything_on(motor_model):
    kw = dict(ct0=[1e-4] * 4, ct1=[1e-6] * 4,
              cq0=[1e-6] * 4, cq1=[1e-8] * 4,
              c_D=[0.02, 0.03, 0.05], k_flap=1.5e-7, I_rot=3.452e-8,
              k_h=0.0, k_angle=3.145, k_hor=7.245, k_v2=1e-4)
    if motor_model == "asymmetric":
        kw.update(ka1=13.996, ka2=1.1e-4, kd1=5.933, kd2=3.2e-4)
    return params_dict(**kw)


def test_scaling_invariance():
    for motor_model in ("first_order", "asymmetric"):
        f = statedot_fn(motor_model)
        vals = _everything_on(motor_model)
        p = flat_params(vals)
        rng = np.random.default_rng(77)

        lL, lM, lT = 3.7, 0.6, 2.3
        sS = _scale(DIM_STATE, lL, lM, lT)
        sU = _scale(DIM_INPUT, lL, lM, lT)
        sP = _scale(DIM_PARAM, lL, lM, lT)
        sO = _scale(DIM_OUT, lL, lM, lT)
        assert len(sP) == len(p)

        for _ in range(5):
            s = random_state(rng)
            u = make_inputs(W_c=rng.uniform(900, 2300, 4), v_wind=rng.uniform(-2, 2, 3),
                            F_ext=rng.uniform(-0.05, 0.05, 3), tau_ext=rng.uniform(-1e-3, 1e-3, 3))
            out = f(s, u, p)
            out_scaled = f(s * sS, u * sU, p * sP)
            np.testing.assert_allclose(out_scaled, out * sO, rtol=1e-9,
                                       err_msg=f"unit inconsistency ({motor_model})")
