"""
Golden-vector generator: RotorPy (branch research-additions).

Run from the RotorPy repo root (its venv has numpy/scipy):

    cd /path/to/rotorpy
    uv run --extra all python /path/to/SkyFlow-Dynamics/golden/generate/gen_rotorpy.py \
        --out /path/to/SkyFlow-Dynamics/golden/vectors

Freezes (state, input, params) → state-derivative vectors from `Multirotor._s_dot_fn` (the
canonical NumPy dynamics), plus full fixed-step RK4 and exact-exp discretization steps from
`Multirotor.step`, across configurations that exercise every verified term. Output is written
in SkyFlow spec conventions:

  quaternion xyzw → wxyz;  spin = −rotor_directions (F-6);  wind/external wrench → inputs;
  k_eta/k_m + thrust_c*/torque_c* → ct*/cq* polynomials;  aero=False → aero coefficients 0.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.cwd()))

from rotorpy.vehicles.crazyflie_params import quad_params as CF  # type: ignore
from rotorpy.vehicles.multirotor import Multirotor  # type: ignore

RNG_SEED = 20260810
N_CASES = 5


def spec_params(m: Multirotor, aero: bool) -> dict:
    """Map a constructed Multirotor's parameters into SkyFlow canonical form."""
    n = m.num_rotors
    on = aero
    coef = m.rotor_dyn_coef
    return {
        "mass": float(m.mass), "grav": float(m.g),
        "inertia": np.asarray(m.inertia, float).tolist(),
        "rotor_pos": np.asarray(m.rotor_geometry, float).tolist(),
        "spin": (-np.asarray(m.rotor_dir)).astype(int).tolist(),
        "axis": np.asarray(m.thrust_dir, float).tolist(),
        "ct0": np.broadcast_to(m.thrust_c0, (n,)).tolist(),
        "ct1": np.broadcast_to(m.thrust_c1, (n,)).tolist(),
        "ct2": np.broadcast_to(m.k_eta, (n,)).tolist(),
        "cq0": np.broadcast_to(m.torque_c0, (n,)).tolist(),
        "cq1": np.broadcast_to(m.torque_c1, (n,)).tolist(),
        "cq2": np.broadcast_to(m.k_m, (n,)).tolist(),
        "tau_m": float(m.tau_m),
        "ka1": float(coef[0]) if coef is not None else 0.0,
        "ka2": float(coef[1]) if coef is not None else 0.0,
        "kd1": float(coef[2]) if coef is not None else 0.0,
        "kd2": float(coef[3]) if coef is not None else 0.0,
        "I_rot": float(m.rotor_inertia),
        "c_D": [float(m.c_Dx) if on else 0.0, float(m.c_Dy) if on else 0.0,
                float(m.c_Dz) if on else 0.0],
        "c_L": [0.0, 0.0, 0.0],  # RotorPy has no lumped linear drag term
        "k_d": float(m.k_d) if on else 0.0,
        "k_z": float(m.k_z) if on else 0.0,
        "k_flap": float(m.k_flap) if on else 0.0,
        "k_h": float(m.k_h) if on else 0.0,
        "k_angle": float(m.k_angle) if on else 0.0,
        "k_hor": float(m.k_hor) if on else 0.0,
        "k_v2": float(m.k_v2) if on else 0.0,
        "r_prop": float(m.r_prop) if m.r_prop else 0.0225,
        "limits": {"rotor_speed_min": float(m.rotor_speed_min),
                   "rotor_speed_max": float(m.rotor_speed_max)},
    }


def xyzw_to_wxyz(q):
    q = np.asarray(q, float)
    return np.array([q[3], q[0], q[1], q[2]])


def random_state(rng, n):
    q = rng.standard_normal(4)
    return {
        "x": rng.uniform(-5, 5, 3), "v": rng.uniform(-3, 3, 3),
        "q": q / np.linalg.norm(q),                      # rotorpy xyzw
        "w": rng.uniform(-2, 2, 3),
        "wind": np.zeros(3),
        "rotor_speeds": rng.uniform(800, 2400, n),
    }


def pack(state, n):
    s = np.zeros(16 + n)
    s[0:3], s[3:6], s[6:10] = state["x"], state["v"], state["q"]
    s[10:13], s[13:16], s[16:] = state["w"], state["wind"], state["rotor_speeds"]
    return s


def case_json(state, cmd, F_ext, tau_ext):
    return {
        "state": {"x": state["x"].tolist(), "v": state["v"].tolist(),
                  "q_wxyz": xyzw_to_wxyz(state["q"]).tolist(), "w": state["w"].tolist(),
                  "rotor_speeds": state["rotor_speeds"].tolist()},
        "inputs": {"cmd_rotor_speeds": cmd.tolist(), "v_wind": state["wind"].tolist(),
                   "F_ext": F_ext.tolist(), "tau_ext": tau_ext.tolist()},
    }


def gen_statedot(name, overrides, mkw, wind=False, wrench=False, motor_model="first_order"):
    p = {**CF, **overrides, "motor_noise_std": 0.0}
    m = Multirotor(p, control_abstraction="cmd_motor_speeds", **mkw)
    n = m.num_rotors
    rng = np.random.default_rng(RNG_SEED)
    cases = []
    for _ in range(N_CASES):
        st = random_state(rng, n)
        if wind:
            st["wind"] = rng.uniform(-3, 3, 3)
        F_ext = rng.uniform(-0.05, 0.05, 3) if wrench else np.zeros(3)
        tau_ext = rng.uniform(-1e-3, 1e-3, 3) if wrench else np.zeros(3)
        m.external_force, m.external_torque = F_ext, tau_ext
        cmd = np.clip(rng.uniform(700, 2600, n), m.rotor_speed_min, m.rotor_speed_max)
        s_dot = m._s_dot_fn(0.0, pack(st, n), cmd)
        c = case_json(st, cmd, F_ext, tau_ext)
        c["expected"] = {"xdot": s_dot[0:3].tolist(), "vdot": s_dot[3:6].tolist(),
                         "qdot_wxyz": xyzw_to_wxyz(s_dot[6:10]).tolist(),
                         "wdot": s_dot[10:13].tolist(), "Wdot": s_dot[16:].tolist()}
        cases.append(c)
    return {"kind": "statedot", "name": name, "motor_model": motor_model,
            "params": spec_params(m, mkw.get("aero", True)), "cases": cases}


def gen_step(name, overrides, mkw, dt, motor_discretization):
    p = {**CF, **overrides, "motor_noise_std": 0.0}
    m = Multirotor(p, control_abstraction="cmd_motor_speeds",
                   integrator_kwargs={"method": "rk4"},
                   motor_discretization=motor_discretization, **mkw)
    n = m.num_rotors
    rng = np.random.default_rng(RNG_SEED + 1)
    cases = []
    for _ in range(N_CASES):
        st = random_state(rng, n)
        cmd = np.clip(rng.uniform(700, 2600, n), m.rotor_speed_min, m.rotor_speed_max)
        out = m.step({k: v.copy() for k, v in st.items()}, {"cmd_motor_speeds": cmd}, dt)
        c = case_json(st, cmd, np.zeros(3), np.zeros(3))
        c["expected"] = {"x": out["x"].tolist(), "v": out["v"].tolist(),
                         "q_wxyz": xyzw_to_wxyz(out["q"]).tolist(), "w": out["w"].tolist(),
                         "rotor_speeds": out["rotor_speeds"].tolist()}
        cases.append(c)
    return {"kind": f"step_{motor_discretization}", "name": name, "dt": dt,
            "motor_model": "first_order",
            "params": spec_params(m, mkw.get("aero", True)), "cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    provenance = {
        "generator": "golden/generate/gen_rotorpy.py",
        "source": "RotorPy (spencerfolk/rotorpy) — Multirotor NumPy dynamics, canonical path",
        "source_branch": branch, "source_commit": commit,
        "date": datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
        "conventions": "converted to SkyFlow canonical: quaternion wxyz, spin=-rotor_directions,"
                       " wind/external wrench as inputs, aero=False -> aero coefficients zeroed",
    }

    tilt = 0.035
    docs = [
        gen_statedot("base_no_aero", {}, {"aero": False}),
        gen_statedot("aero_default_wind", {}, {"aero": True}, wind=True),
        gen_statedot("aero_full_wrench",
                     {"c_Dx": 0.02, "c_Dy": 0.03, "c_Dz": 0.05, "k_flap": 1.5e-7},
                     {"aero": True}, wind=True, wrench=True),
        gen_statedot("aoa_advance_ratio",
                     {"k_angle": 3.145, "k_hor": 7.245, "k_v2": 1e-4, "r_prop": 0.0635},
                     {"aero": True}, wind=True),
        gen_statedot("translational_lift", {"k_h": 1e-5}, {"aero": True}, wind=True),
        gen_statedot("polynomial_curves",
                     {"thrust_c0": 1e-4, "thrust_c1": 1e-6,
                      "torque_c0": 1e-6, "torque_c1": 1e-8}, {"aero": True}),
        gen_statedot("asymmetric_motor",
                     {"rotor_dyn_coef": [13.996001897562685, 0.00011093207920685363,
                                         5.933168530682111, 0.00031951312393561264]},
                     {"aero": True}, motor_model="asymmetric"),
        gen_statedot("per_rotor_coefficients",
                     {"k_eta": [2.0e-8, 2.6e-8, 2.2e-8, 2.4e-8],
                      "k_m": [7.0e-10, 8.6e-10, 7.4e-10, 8.2e-10]}, {"aero": True}),
        gen_statedot("thrust_axis_tilt",
                     {"thrust_dir": [[np.sin(tilt), 0, np.cos(tilt)],
                                     [0, -np.sin(tilt), np.cos(tilt)],
                                     [-np.sin(tilt), 0, np.cos(tilt)],
                                     [0, 0, 1]]}, {"aero": True}),
        gen_statedot("rotor_inertia", {"rotor_inertia": 3.452e-8}, {"aero": True}),
        gen_step("rk4_step", {"c_Dx": 0.02, "c_Dy": 0.03, "c_Dz": 0.05, "k_flap": 1.5e-7},
                 {"aero": True}, 0.01, "ode"),
        gen_step("exact_exp_step", {"rotor_inertia": 3.452e-8}, {"aero": True},
                 0.01, "exact_exp"),
    ]
    for doc in docs:
        doc["schema"] = 1
        doc["provenance"] = provenance
        path = out_dir / f"rotorpy_{doc['name']}.json"
        path.write_text(json.dumps(doc, indent=1))
        print(f"wrote {path.name}: {len(doc['cases'])} cases ({doc['kind']})")


if __name__ == "__main__":
    main()
