"""
Shared machinery for the property tests: canonical n=4 symbols, cached lambdified dynamics,
parameter bindings, and seeded random states.
"""

from functools import lru_cache

import numpy as np
import sympy as sp

from skyflow_dynamics.spec.dynamics import statedot
from skyflow_dynamics.spec.parameters import CRAZYFLIE, substitution
from skyflow_dynamics.spec.symbols import input_symbols, param_symbols, state_symbols

N = 4
S = state_symbols(N)
U = input_symbols(N)
P = param_symbols(N)


@lru_cache(maxsize=None)
def statedot_expr(motor_model: str = "first_order") -> sp.Matrix:
    return statedot(S, U, P, motor_model)


@lru_cache(maxsize=None)
def statedot_fn(motor_model: str = "first_order"):
    """Numeric f(s_flat, u_flat, p_flat) -> (13+N,) ndarray."""
    fn = sp.lambdify((S.flat(), U.flat(), P.flat()), statedot_expr(motor_model),
                     modules="numpy", cse=True)
    return lambda s, u, p: np.asarray(fn(s, u, p), dtype=float).ravel()


def params_dict(**overrides) -> dict:
    vals = {k: v for k, v in CRAZYFLIE.items() if k != "limits"}
    vals.update(overrides)
    return vals


def flat_params(values: dict) -> np.ndarray:
    sub = substitution(P, values)
    return np.array([float(sub[sym]) for sym in P.flat()])


def random_unit_quaternion(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def random_state(rng, w_lo=800.0, w_hi=2400.0) -> np.ndarray:
    """State in S.flat() order: x(3), v(3), q(4 wxyz), w(3), Omega(N)."""
    return np.concatenate([
        rng.uniform(-5, 5, 3),
        rng.uniform(-3, 3, 3),
        random_unit_quaternion(rng),
        rng.uniform(-2, 2, 3),
        rng.uniform(w_lo, w_hi, N),
    ])


def make_inputs(W_c=None, v_wind=(0, 0, 0), F_ext=(0, 0, 0), tau_ext=(0, 0, 0)) -> np.ndarray:
    W_c = np.full(N, 1788.53) if W_c is None else np.asarray(W_c, dtype=float)
    return np.concatenate([W_c, v_wind, F_ext, tau_ext]).astype(float)


def hover_speed(values: dict) -> float:
    """Hover rotor speed for a symmetric pure-quadratic vehicle: sqrt(mg / Σ ct2)."""
    return float(np.sqrt(values["mass"] * values["grav"] / sum(values["ct2"])))


def unit_subs(expr, q):
    """Substitute the unit-norm constraint q_w² = 1 − q_x² − q_y² − q_z² and expand.
    Sound for expressions polynomial in q (degree ≤ 2 in q_w after expand)."""
    w, x, y, z = q
    return sp.expand(sp.expand(expr).subs(w**2, 1 - x**2 - y**2 - z**2))
