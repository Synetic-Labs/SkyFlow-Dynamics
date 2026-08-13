"""
Parameter schema and reference vehicles.

Numeric parameter sets are plain dicts keyed by the canonical names below — the same format
golden-vector files embed. Per-rotor entries are lists of length n. Everything is SI (rad/s
speeds); `spin` is the PHYSICAL spin sign about +ẑ_B (⚠ RotorPy's `rotor_directions` is the
yaw-torque sign — the negative of this).
"""

import math

import sympy as sp

from skyflow_dynamics.spec.symbols import Params

#: name → (units, description). Shapes: () scalar, (n,) per-rotor, (3,) vector, (n,3) per-rotor vector.
SCHEMA = {
    "mass":    ("kg",            "vehicle mass"),
    "grav":    ("m/s^2",         "gravitational acceleration magnitude"),
    "inertia": ("kg*m^2",        "3x3 body inertia matrix (products of inertia allowed)"),
    "rotor_pos": ("m (n,3)",     "rotor hub positions in the body frame"),
    "spin":    ("- (n,)",        "physical rotor spin sign about +z_B (+1 CCW seen from above)"),
    "axis":    ("- (n,3)",       "unit thrust axes in the body frame (default z; tilt = misalignment)"),
    "ct0":     ("N (n,)",        "thrust polynomial constant term"),
    "ct1":     ("N/(rad/s) (n,)", "thrust polynomial linear term"),
    "ct2":     ("N/(rad/s)^2 (n,)", "thrust polynomial quadratic term (k_eta)"),
    "cq0":     ("N*m (n,)",      "drag-torque polynomial constant term"),
    "cq1":     ("N*m/(rad/s) (n,)", "drag-torque polynomial linear term"),
    "cq2":     ("N*m/(rad/s)^2 (n,)", "drag-torque polynomial quadratic term (k_m)"),
    "tau_m":   ("s",             "motor first-order time constant"),
    "ka1":     ("1/s",           "asymmetric motor: spin-up linear coefficient"),
    "ka2":     ("1/rad",         "asymmetric motor: spin-up quadratic coefficient"),
    "kd1":     ("1/s",           "asymmetric motor: spin-down linear coefficient"),
    "kd2":     ("1/rad",         "asymmetric motor: spin-down quadratic coefficient"),
    "I_rot":   ("kg*m^2",        "rotor spin-axis moment of inertia (per rotor)"),
    "c_D":     ("N/(m/s)^2 (3,)", "parasitic drag diagonal (c_Dx, c_Dy, c_Dz)"),
    "c_L":     ("N/(m/s) (3,)",  "lumped linear body-frame drag diagonal (Faessler rotor-drag form)"),
    "k_d":     ("kg/rad",        "rotor in-plane drag (H-force) coefficient"),
    "k_z":     ("kg/rad",        "rotor axial induced-inflow coefficient"),
    "k_flap":  ("kg*m/rad",      "blade-flapping moment coefficient"),
    "k_h":     ("kg/m",          "translational lift coefficient"),
    "k_angle": ("1/rad",         "thrust vs rotor angle-of-attack slope"),
    "k_hor":   ("1/rad",         "thrust vs advance-ratio-angle slope"),
    "k_v2":    ("kg/m",          "vertical airspeed^2 collective term"),
    "r_prop":  ("m",             "propeller radius (forms the inflow angles)"),
}


def validate(values: dict) -> None:
    """Reject physically inconsistent parameter sets."""
    if values.get("k_h", 0.0) != 0.0 and (
            values.get("k_angle", 0.0) != 0.0 or values.get("k_hor", 0.0) != 0.0):
        raise ValueError(
            "k_h (translational lift) and k_angle/k_hor (AoA/advance-ratio thrust) model the "
            "same in-plane airspeed effect — enabling both double-counts it. Use one.")
    for s in values["spin"]:
        if s not in (1, -1, 1.0, -1.0):
            raise ValueError(f"spin entries must be +1 or -1, got {s}")
    for e in values["axis"]:
        if abs(math.sqrt(sum(c * c for c in e)) - 1.0) > 1e-9:
            raise ValueError(f"thrust axis {e} is not unit-norm")


def substitution(p: Params, values: dict) -> dict:
    """Map a numeric parameter dict onto the Params symbols → {symbol: value} for subs/lambdify."""
    validate(values)
    n = p.n
    sub = {p.mass: values["mass"], p.grav: values["grav"],
           p.tau_m: values["tau_m"], p.ka1: values["ka1"], p.ka2: values["ka2"],
           p.kd1: values["kd1"], p.kd2: values["kd2"], p.I_rot: values["I_rot"],
           p.k_d: values["k_d"], p.k_z: values["k_z"], p.k_flap: values["k_flap"],
           p.k_h: values["k_h"], p.k_angle: values["k_angle"], p.k_hor: values["k_hor"],
           p.k_v2: values["k_v2"], p.r_prop: values["r_prop"]}
    I = values["inertia"]
    for a in range(3):
        for b in range(3):
            sub[p.inertia[a, b]] = I[a][b]
    for k in range(3):
        sub[p.c_D[k]] = values["c_D"][k]
        sub[p.c_L[k]] = values["c_L"][k]
    for i in range(n):
        sub[p.spin[i]] = values["spin"][i]
        for k in range(3):
            sub[p.rotor_pos[i][k]] = values["rotor_pos"][i][k]
            sub[p.axis[i][k]] = values["axis"][i][k]
        for name in ("ct0", "ct1", "ct2", "cq0", "cq1", "cq2"):
            sub[getattr(p, name)[i]] = values[name][i]
    return sub


def _sym(x):
    return [round(v, 12) for v in x]


#: Crazyflie 2.0 reference vehicle, via RotorPy rotorpy/vehicles/crazyflie_params.py.
#: Provenance is mixed (per that file's own header): thrust coefficient inferred from
#: 14.5 g at 2500 rad/s (bitcraze measurements / Forster 2015 lineage); k_d, k_z are RotorPy
#: placeholder values ("k_drag is mostly made up"). spin = −rotor_directions (F-6 flip).
_d = 0.043  # arm length, m
_c45 = 0.70710678118
CRAZYFLIE = {
    "mass": 0.03, "grav": 9.81,
    "inertia": [[1.43e-5, 0.0, 0.0], [0.0, 1.43e-5, 0.0], [0.0, 0.0, 2.89e-5]],
    "rotor_pos": [_sym([_d * _c45, _d * _c45, 0.0]), _sym([_d * _c45, -_d * _c45, 0.0]),
                  _sym([-_d * _c45, -_d * _c45, 0.0]), _sym([-_d * _c45, _d * _c45, 0.0])],
    "spin": [-1, 1, -1, 1],
    "axis": [[0.0, 0.0, 1.0]] * 4,
    "ct0": [0.0] * 4, "ct1": [0.0] * 4, "ct2": [2.3e-8] * 4,
    "cq0": [0.0] * 4, "cq1": [0.0] * 4, "cq2": [7.8e-10] * 4,
    "tau_m": 0.072, "ka1": 0.0, "ka2": 0.0, "kd1": 0.0, "kd2": 0.0,
    "I_rot": 0.0,
    "c_D": [0.0, 0.0, 0.0],
    "c_L": [0.0, 0.0, 0.0],
    "k_d": 1.02506e-6, "k_z": 7.553e-7, "k_flap": 0.0, "k_h": 0.0,
    "k_angle": 0.0, "k_hor": 0.0, "k_v2": 0.0, "r_prop": 0.0225,
    # Harness-side operating limits (not part of the continuous dynamics):
    "limits": {"rotor_speed_min": 0.0, "rotor_speed_max": 2500.0},
}
