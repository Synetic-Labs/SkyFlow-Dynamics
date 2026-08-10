"""
Golden-vector generator: Crazyflow first-principles dynamics (Synetic-Labs fork).

Runs crazyflow's actual `dynamics()` (crazyflow/dynamics/first_principles/dynamics.py) via a
bare-package import shim (bypassing the mujoco-importing package __init__) with lightweight
stubs for jax/casadi/flax (imported but unused on the numpy evaluation path). Requires numpy,
scipy, and array-api-compat:

    uv run --with scipy --with array-api-compat python golden/generate/gen_crazyflow.py \
        --crazyflow /path/to/crazyflow --out golden/vectors

Conversions to SkyFlow canonical form:
  quat xyzw → wxyz · rotor speeds/commands RPM → rad/s · thrust/torque polynomials per-RPM →
  per-(rad/s) (÷(2π/60) per Ω power) · rotor positions from the mixing matrix: their torque map
  is M_x = L·Σ mix_x,i·T_i, M_y = L·Σ mix_y,i·T_i ⇒ r_i = (−L·mix_y,i, +L·mix_x,i, 0) ·
  spin = −mix_z (mix_z is the yaw-TORQUE sign) · drag_matrix = −diag(c_L) (lumped linear drag).

Gyroscopic term: INCLUDED (prop_inertia from params.toml). Crazyflow's original gyro had the
roll-row sign flipped (finding F-3); fixed upstream by learnsyslab/crazyflow PR #86 (merged
2026-07-13) — generate only from a checkout at/after that merge.

Deliberately excluded:
  quat_dot — their Ξ(ω) construction is scalar-first but applied to xyzw storage; their own
  docstring forbids integrating it, so it is vestigial. Compared blocks: xdot, vdot, wdot, Wdot.
"""

import argparse
import dataclasses
import datetime
import json
import pathlib
import subprocess
import sys
import tomllib
import types

import numpy as np

RPM2RAD = 2 * np.pi / 60
RNG_SEED = 20260810
N_CASES = 8


def install_stubs():
    jax = types.ModuleType("jax")
    jax.numpy = types.ModuleType("jax.numpy")
    jax.Array = type("Array", (), {})
    jax.device_put = lambda x, device=None: x
    jax.jit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules.setdefault("jax", jax)
    sys.modules.setdefault("jax.numpy", jax.numpy)
    flax = types.ModuleType("flax")
    flax.struct = types.ModuleType("flax.struct")
    flax.struct.dataclass = dataclasses.dataclass
    sys.modules.setdefault("flax", flax)
    sys.modules.setdefault("flax.struct", flax.struct)
    from unittest.mock import MagicMock
    sys.modules.setdefault("casadi", MagicMock())  # symbolic path only; absorbs import-time use


def import_crazyflow_dynamics(root: pathlib.Path):
    """Import ONLY the dynamics module: bare parent packages skip every __init__.py, and the
    two casadi-at-import modules (symbols, utils.rotation) are mocked — dynamics() touches
    them only on paths we exclude (symbolic_dynamics, quat_dot)."""
    from unittest.mock import MagicMock
    sys.path.insert(0, str(root))

    def bare(name: str, path: pathlib.Path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        sys.modules[name] = mod
        return mod

    bare("crazyflow", root / "crazyflow")
    bare("crazyflow.dynamics", root / "crazyflow/dynamics")
    bare("crazyflow.dynamics.first_principles", root / "crazyflow/dynamics/first_principles")
    utils_pkg = bare("crazyflow.dynamics.utils", root / "crazyflow/dynamics/utils")
    rotation = MagicMock()  # quat_dot output is discarded (see module docstring)
    sys.modules["crazyflow.dynamics.utils.rotation"] = rotation
    utils_pkg.rotation = rotation
    sys.modules["crazyflow.dynamics.symbols"] = MagicMock()  # symbolic path only

    from crazyflow.dynamics.first_principles.dynamics import dynamics
    return dynamics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crazyflow", default="/home/james/CODE/crazyflow-fork")
    ap.add_argument("--drone", default="cf2x_L250")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = pathlib.Path(args.crazyflow)

    install_stubs()
    cf_dynamics = import_crazyflow_dynamics(root)

    toml = tomllib.loads((root / "crazyflow/drones/params.toml").read_text())
    cfp = toml[args.drone]
    mass, L = cfp["mass"], cfp["L"]
    J = np.array(cfp["J"])
    rpm2thrust = np.array(cfp["rpm2thrust"])
    rpm2torque = np.array(cfp["rpm2torque"])
    mix = np.array(cfp["mixing_matrix"])
    drag = np.array(cfp["drag_matrix"])
    coef = np.array(cfp["rotor_dyn_coef"])

    spec_params = {
        "mass": mass, "grav": 9.81, "inertia": J.tolist(),
        "rotor_pos": [[-L * mix[1][i], L * mix[0][i], 0.0] for i in range(4)],
        "spin": (-mix[2]).astype(int).tolist(),
        "axis": [[0.0, 0.0, 1.0]] * 4,
        "ct0": [rpm2thrust[0]] * 4, "ct1": [rpm2thrust[1] / RPM2RAD] * 4,
        "ct2": [rpm2thrust[2] / RPM2RAD**2] * 4,
        "cq0": [rpm2torque[0]] * 4, "cq1": [rpm2torque[1] / RPM2RAD] * 4,
        "cq2": [rpm2torque[2] / RPM2RAD**2] * 4,
        "tau_m": 1.0 / coef[0],  # unused (motor_model=asymmetric)
        "ka1": coef[0], "ka2": coef[1] / RPM2RAD,
        "kd1": coef[2], "kd2": coef[3] / RPM2RAD,
        "I_rot": cfp["prop_inertia"],
        "c_D": [0.0, 0.0, 0.0],
        "c_L": (-np.diag(drag)).tolist(),
        "k_d": 0.0, "k_z": 0.0, "k_flap": 0.0, "k_h": 0.0,
        "k_angle": 0.0, "k_hor": 0.0, "k_v2": 0.0, "r_prop": 0.0225,
        "limits": {"rotor_speed_min": 0.0, "rotor_speed_max": 3000.0},
    }

    rng = np.random.default_rng(RNG_SEED)
    cases = []
    for _ in range(N_CASES):
        pos = rng.uniform(-5, 5, 3)
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)                      # crazyflow xyzw
        vel = rng.uniform(-3, 3, 3)
        ang = rng.uniform(-2, 2, 3)
        rotor_rpm = rng.uniform(8000, 22000, 4)
        cmd_rpm = rng.uniform(8000, 24000, 4)

        pos_dot, _quat_dot, vel_dot, ang_dot, rotor_dot = cf_dynamics(
            pos, q, vel, ang, cmd_rpm, rotor_vel=rotor_rpm,
            mass=mass, L=L, prop_inertia=cfp["prop_inertia"],
            gravity_vec=np.array([0.0, 0.0, -9.81]),
            J=J, J_inv=np.linalg.inv(J), rpm2thrust=rpm2thrust, rpm2torque=rpm2torque,
            mixing_matrix=mix, drag_matrix=drag, rotor_dyn_coef=coef)

        cases.append({
            "state": {"x": pos.tolist(), "v": vel.tolist(),
                      "q_wxyz": [q[3], q[0], q[1], q[2]], "w": ang.tolist(),
                      "rotor_speeds": (rotor_rpm * RPM2RAD).tolist()},
            "inputs": {"cmd_rotor_speeds": (cmd_rpm * RPM2RAD).tolist(),
                       "v_wind": [0.0] * 3, "F_ext": [0.0] * 3, "tau_ext": [0.0] * 3},
            "expected": {"xdot": np.asarray(pos_dot, float).tolist(),
                         "vdot": np.asarray(vel_dot, float).tolist(),
                         "wdot": np.asarray(ang_dot, float).tolist(),
                         "Wdot": (np.asarray(rotor_dot, float) * RPM2RAD).tolist()},
        })

    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    doc = {
        "schema": 1, "kind": "statedot", "name": f"crazyflow_{args.drone}",
        "motor_model": "asymmetric",
        "compare": ["xdot", "vdot", "wdot", "Wdot"],
        "provenance": {
            "generator": "golden/generate/gen_crazyflow.py",
            "source": "Crazyflow first-principles dynamics (Synetic-Labs/crazyflow fork), "
                      "actual running code via bare-package shim",
            "source_commit": commit,
            "date": datetime.date.today().isoformat(),
            "notes": "prop_inertia ACTIVE: gyro term correct at/after learnsyslab/crazyflow "
                     "PR #86 (merged 2026-07-13; F-3 roll-row sign fix). quat_dot excluded "
                     "(scalar-first Xi applied to xyzw storage, unused by their integrator). "
                     "RPM->rad/s and polynomial conversions applied.",
        },
        "params": spec_params, "cases": cases,
    }
    out = pathlib.Path(args.out) / f"crazyflow_{args.drone}.json"
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out.name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
