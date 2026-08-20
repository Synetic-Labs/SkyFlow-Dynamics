"""
Golden-vector generator: RotorPy (spencerfolk/rotorpy, upstream main).

Run from the RotorPy repo root:

    cd /path/to/rotorpy
    uv run --with numpy --with scipy python \
        /path/to/SkyFlow-Dynamics/golden/generate/gen_rotorpy.py \
        --out /path/to/SkyFlow-Dynamics/golden/vectors

Freezes (state, input, params) → state-derivative vectors from `Multirotor._s_dot_fn` (the
canonical NumPy dynamics) across configurations that exercise every upstream-implemented
term: Newton–Euler + quaternion kinematics, first-order motor lag, quadratic thrust/torque
(k_eta/k_m), the Svacha per-rotor drag matrix (k_d, k_z), blade-flapping moment (k_flap),
translational lift (k_h), and parasitic drag (c_D*). Output is written in SkyFlow spec
conventions:

  quaternion xyzw → wxyz;  spin = −rotor_directions (F-6);  wind as input;
  k_eta/k_m → ct2/cq2 (upstream curves are pure quadratics);  aero=False → aero
  coefficients zeroed.

Terms absent upstream (general polynomial curves, per-rotor coefficient arrays, thrust-axis
tilt, rotor-inertia gyroscopic moments, asymmetric motor lag, exact-exp/fixed-RK4 stepping,
AoA corrections, external wrench inputs) are golden-verified against the other reference
implementations (Crazyflow, SkyDreamer, agilicious agilib) and by exact property tests —
see the registry entries.
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
    """Map a constructed upstream Multirotor's parameters into SkyFlow canonical form."""
    n = m.num_rotors
    on = aero
    return {
        "mass": float(m.mass), "grav": float(m.g),
        "inertia": np.asarray(m.inertia, float).tolist(),
        "rotor_pos": np.asarray(m.rotor_geometry, float).tolist(),
        "spin": (-np.asarray(m.rotor_dir)).astype(int).tolist(),
        "axis": [[0.0, 0.0, 1.0]] * n,          # upstream thrust is always body-z
        "ct0": [0.0] * n, "ct1": [0.0] * n,     # upstream curves are pure quadratics
        "ct2": np.broadcast_to(m.k_eta, (n,)).tolist(),
        "cq0": [0.0] * n, "cq1": [0.0] * n,
        "cq2": np.broadcast_to(m.k_m, (n,)).tolist(),
        "tau_m": float(m.tau_m),
        "ka1": 0.0, "ka2": 0.0, "kd1": 0.0, "kd2": 0.0,   # no asymmetric lag upstream
        "I_rot": 0.0,                                      # no gyroscopic term upstream
        "c_D": [float(m.c_Dx) if on else 0.0, float(m.c_Dy) if on else 0.0,
                float(m.c_Dz) if on else 0.0],
        "c_L": [0.0, 0.0, 0.0],  # RotorPy has no lumped linear drag term
        "k_d": float(m.k_d) if on else 0.0,
        "k_z": float(m.k_z) if on else 0.0,
        "k_flap": float(m.k_flap) if on else 0.0,
        "k_h": float(m.k_h) if on else 0.0,
        "k_angle": 0.0, "k_hor": 0.0, "k_v2": 0.0,        # no AoA corrections upstream
        "r_prop": 0.0225,
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


def case_json(state, cmd):
    return {
        "state": {"x": state["x"].tolist(), "v": state["v"].tolist(),
                  "q_wxyz": xyzw_to_wxyz(state["q"]).tolist(), "w": state["w"].tolist(),
                  "rotor_speeds": state["rotor_speeds"].tolist()},
        "inputs": {"cmd_rotor_speeds": cmd.tolist(), "v_wind": state["wind"].tolist(),
                   "F_ext": [0.0, 0.0, 0.0], "tau_ext": [0.0, 0.0, 0.0]},
    }


def gen_statedot(name, overrides, mkw, wind=False):
    p = {**CF, **overrides, "motor_noise_std": 0.0}
    m = Multirotor(p, control_abstraction="cmd_motor_speeds", **mkw)
    n = m.num_rotors
    rng = np.random.default_rng(RNG_SEED)
    cases = []
    for _ in range(N_CASES):
        st = random_state(rng, n)
        if wind:
            st["wind"] = rng.uniform(-3, 3, 3)
        cmd = np.clip(rng.uniform(700, 2600, n), m.rotor_speed_min, m.rotor_speed_max)
        s_dot = m._s_dot_fn(0.0, pack(st, n), cmd)
        c = case_json(st, cmd)
        c["expected"] = {"xdot": s_dot[0:3].tolist(), "vdot": s_dot[3:6].tolist(),
                         "qdot_wxyz": xyzw_to_wxyz(s_dot[6:10]).tolist(),
                         "wdot": s_dot[10:13].tolist(), "Wdot": s_dot[16:].tolist()}
        cases.append(c)
    return {"kind": "statedot", "name": name, "motor_model": "first_order",
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
                       " wind as input, aero=False -> aero coefficients zeroed",
    }

    docs = [
        gen_statedot("base_no_aero", {}, {"aero": False}),
        gen_statedot("aero_default_wind", {}, {"aero": True}, wind=True),
        gen_statedot("aero_drag_flap",
                     {"c_Dx": 0.02, "c_Dy": 0.03, "c_Dz": 0.05, "k_flap": 1.5e-7},
                     {"aero": True}, wind=True),
        gen_statedot("translational_lift", {"k_h": 1e-5}, {"aero": True}, wind=True),
    ]
    for doc in docs:
        doc["schema"] = 1
        doc["provenance"] = provenance
        path = out_dir / f"rotorpy_{doc['name']}.json"
        path.write_text(json.dumps(doc, indent=1))
        print(f"wrote {path.name}: {len(doc['cases'])} cases ({doc['kind']})")


if __name__ == "__main__":
    main()
