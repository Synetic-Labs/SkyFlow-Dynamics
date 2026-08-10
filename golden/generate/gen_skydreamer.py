"""
Golden-vector generator: SkyDreamer identified dynamics (compute_dynamics_jit).

Executes the ACTUAL SkyDreamer dynamics source (a copy of embodied/envs/skydreamer.py) by
slicing the @njit function block out of the file and exec'ing it with a no-op njit stub —
their literal code, no numba required:

    uv run python golden/generate/gen_skydreamer.py \
        --skydreamer /path/to/skydreamer.py --out golden/vectors

Scope (structural, documented):
  Only the FORCE side and rotor dynamics are comparable. SkyDreamer's moments are per-rotor
  identified k_p/k_q/k_r coefficients (not geometry-derived r × F) — never comparable to a
  structural model; its quadratic drag is per-axis |v_k|·v_k (not ‖v‖-scaled) — a different
  model, so k_x2/k_y2 are zeroed here. Cases use identity attitude and zero body rates (their
  model has no ω×r lever arm on the drag), varied airspeed and rotor speeds.
  Compared blocks: vdot, Wdot.

Conversions (finding F-4): SkyDreamer emits mass-normalized accelerations — spec coefficients
are ct2 = m·k_w, k_d = m·k_x, k_v2 = m·k_v2 for an arbitrary mass. Rotor states are normalized
w_n ∈ [−1,1] over [0, 3000] rad/s: Ω = (w_n+1)/2·3000, Ω̇ = ẇ_n·1500. Their quaternion is
already scalar-first wxyz. Zero action → commanded speed = throttle_curve(0) = w_min.
"""

import argparse
import datetime
import hashlib
import json
import pathlib

import numpy as np

RNG_SEED = 20260810
MASS = 0.6  # arbitrary spec mass; SkyDreamer's coefficients are per-unit-mass (F-4)
W_NORM_MAX = 3000.0


def load_skydreamer(path: pathlib.Path):
    src = path.read_text()
    start = src.index("@njit(fastmath=True, cache=True)\ndef quat_normalize")
    end = src.index("@njit(fastmath=True, cache=True)\ndef compute_step_logic_jit")
    p_start = src.index("SKYDREAMER_PARAMS = {")
    p_end = src.index("}", p_start) + 1
    header = ("import numpy as np\n"
              "def njit(*a, **k):\n"
              "    return a[0] if a and callable(a[0]) else (lambda f: f)\n")
    ns = {}
    exec(header + src[p_start:p_end] + "\n" + src[start:end], ns)
    return ns["compute_dynamics_jit"], ns["SKYDREAMER_PARAMS"]


PARAM_KEYS = ["k_x", "k_y", "k_w", "k_x2", "k_y2", "k_angle", "k_hor", "k_v2",
              "k_p1", "k_p2", "k_p3", "k_p4", "k_q1", "k_q2", "k_q3", "k_q4",
              "k_r1", "k_r2", "k_r3", "k_r4", "k_r5", "k_r6", "k_r7", "k_r8",
              "J_x", "J_y", "J_z", "tau", "k", "w_min", "w_max", "r_prop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skydreamer", default="/home/james/CODE/nav-jax/sims/skydreamer/skydreamer.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = pathlib.Path(args.skydreamer)

    compute_dynamics_jit, sky_params = load_skydreamer(path)
    sp = dict(sky_params)
    sp["k_x2"] = 0.0  # per-axis |v|·v quadratic drag: structurally different model — excluded
    sp["k_y2"] = 0.0
    params_row = np.array([[sp[k] for k in PARAM_KEYS]], dtype=np.float64)

    spec_params = {
        "mass": MASS, "grav": 9.81,
        "inertia": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # wdot not compared
        "rotor_pos": [[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, -0.1, 0.0], [-0.1, 0.1, 0.0]],
        "spin": [-1, 1, -1, 1],
        "axis": [[0.0, 0.0, 1.0]] * 4,
        "ct0": [0.0] * 4, "ct1": [0.0] * 4, "ct2": [MASS * sp["k_w"]] * 4,
        "cq0": [0.0] * 4, "cq1": [0.0] * 4, "cq2": [0.0] * 4,
        "tau_m": sp["tau"], "ka1": 0.0, "ka2": 0.0, "kd1": 0.0, "kd2": 0.0,
        "I_rot": 0.0,
        "c_D": [0.0, 0.0, 0.0], "c_L": [0.0, 0.0, 0.0],
        "k_d": MASS * sp["k_x"], "k_z": 0.0, "k_flap": 0.0, "k_h": 0.0,
        "k_angle": sp["k_angle"], "k_hor": sp["k_hor"], "k_v2": MASS * sp["k_v2"],
        "r_prop": sp["r_prop"],
        "limits": {"rotor_speed_min": sp["w_min"], "rotor_speed_max": sp["w_max"]},
    }
    assert abs(sp["k_x"] - sp["k_y"]) < 1e-12, "spec k_d assumes k_x == k_y"

    rng = np.random.default_rng(RNG_SEED)
    # Zero action → their env maps a ∈ [−1,1] to throttle U = (a+1)/2 = 0.5 → the curve:
    U = 0.5
    cmd_speed = float((sp["w_max"] - sp["w_min"]) * np.sqrt(sp["k"] * U**2 + (1 - sp["k"]) * U)
                      + sp["w_min"])
    cases = []
    for _ in range(6):
        # Their state array is float32 — round first, then freeze the exact rounded values.
        v = rng.uniform(-8, 8, 3).astype(np.float32)
        W = rng.uniform(500, 2800, 4)          # rad/s, inside their [0, 3000] norm range
        w_norm = (2 * W / W_NORM_MAX - 1).astype(np.float32)
        W_frozen = (w_norm.astype(np.float64) + 1) / 2 * W_NORM_MAX
        states = np.zeros((1, 17), dtype=np.float32)
        states[0, 3:6] = v                     # world velocity; identity attitude
        states[0, 6] = 1.0                     # quat wxyz identity
        states[0, 13:17] = w_norm
        actions = np.zeros((1, 4), dtype=np.float32)

        d = compute_dynamics_jit(states, actions, params_row)[0]
        vdot = d[3:6]
        Wdot = d[13:17] * (W_NORM_MAX / 2)     # normalized rate → rad/s²

        cases.append({
            "state": {"x": [0.0] * 3, "v": v.astype(np.float64).tolist(),
                      "q_wxyz": [1.0, 0.0, 0.0, 0.0],
                      "w": [0.0] * 3, "rotor_speeds": W_frozen.tolist()},
            "inputs": {"cmd_rotor_speeds": [cmd_speed] * 4, "v_wind": [0.0] * 3,
                       "F_ext": [0.0] * 3, "tau_ext": [0.0] * 3},
            "expected": {"vdot": np.asarray(vdot, float).tolist(),
                         "Wdot": np.asarray(Wdot, float).tolist()},
        })

    doc = {
        "schema": 1, "kind": "statedot", "name": "skydreamer_forces",
        "motor_model": "first_order",
        "compare": ["vdot", "Wdot"],
        "tolerance": 2e-5,  # their atan2 uses a +1e-6 denominator epsilon
        "provenance": {
            "generator": "golden/generate/gen_skydreamer.py",
            "source": "SkyDreamer (arXiv:2510.14783) compute_dynamics_jit — literal source "
                      "exec'd with njit stubbed",
            "source_file": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "date": datetime.date.today().isoformat(),
            "notes": "Force side + rotor lag only. Moments are per-rotor identified k_p/k_q/k_r "
                     "(not structural r x F) — not comparable. Per-axis quadratic drag "
                     "(k_x2/k_y2) zeroed: different model than |v|-scaled parasitic drag. "
                     "Coefficients mass-scaled per finding F-4; rotor states denormalized from "
                     "[-1,1] over [0,3000] rad/s.",
        },
        "params": spec_params, "cases": cases,
    }
    out = pathlib.Path(args.out) / "skydreamer_forces.json"
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out.name}: {len(cases)} cases")


if __name__ == "__main__":
    main()
