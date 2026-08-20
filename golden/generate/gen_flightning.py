"""
Golden-vector generator: rpg_flightning (ICRA 2025) executed-code vectors.

Runs the ACTUAL flightning JAX source (github.com/uzh-rpg/rpg_flightning, commit pinned in
the provenance block) with float64 enabled, on CPU, and freezes:

    flightning_simple_jvp.json   quadrotor_dyn (the point-mass surrogate) — primal steps AND
                                 jax.jvp tangents (the surrogate-gradient scheme), plus
                                 Quadrotor.step() custom_jvp cases that pin the surrogate
                                 wiring (c = f_d/m, dt-tangent = 0) from executed code.
    flightning_full_step.json    Quadrotor._dynamics single low-level substeps (physics only,
                                 commanded motor speeds given directly) and full Quadrotor.step
                                 calls (low-level P controller in the loop, N substeps).

        uv run --python 3.12 --with "jax==0.4.30" --with jax-dataclasses --with chex \\
            --with "flax<0.9" --with pyyaml python golden/generate/gen_flightning.py \\
            --flightning /path/to/rpg_flightning --out golden/vectors

jax is PINNED to the era version 0.4.30: the reference's custom-JVP rule returns the primal
dr_key (uint32/typed key) in the tangent slot (quadrotor_obj.py:303), which current JAX
(checked on 0.11.0) rejects — Quadrotor.step is un-differentiable there, in both forward and
reverse mode (finding F-28). On 0.4.30 the rule executes as designed.

Domain randomization (harness-side, per episode): _dynamics draws a thrust-map multiplier
(±5%) and per-axis drag-coefficient multipliers (±50%) from state.dr_key, which is never
advanced — the draws are constant within an episode. The generator replays the identical
key splits (their code path) and records the DRAWN values per case; the spec consumes them
as effective parameters.

Reference numerics pinned as recorded params (not re-derived):
  - allocation matrix and its inverse are built in float32 (quadrotor_obj.py:207,222) inside
    an otherwise float64 pipeline — the executed float32 entries are recorded (finding F-27).
  - rotation_matrix_from_vector biases the Rodrigues angle: theta = ||abs(v) + 1e-5||
    (math.py:25), so the "exact" attitude step is not the true exp map (finding F-25). The
    biased form is replicated by the consumer as a harness detail; the deviation of the true
    exp map is bounded there.

Every frozen expected block is re-derived here in plain numpy (float64) and asserted against
the executed JAX outputs before writing, so a wrong DR replay or convention slip fails loudly.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

RNG_SEED = 20260819
EPS_RODRIGUES = 1e-5

SOURCE_FILES = [
    "flightning/objects/quadrotor_obj.py",
    "flightning/objects/quadrotor_simple_obj.py",
    "flightning/simulation/model_body_drag.py",
    "flightning/utils/math.py",
]


# ---------------- numpy replicas (pre-freeze assertions only) ----------------

def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def quat_to_R(q):
    """Standard Hamilton wxyz body->world rotation matrix (matches spec.quaternion)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rod_biased(v, eps=EPS_RODRIGUES):
    """flightning's rotation_matrix_from_vector, verbatim math (biased angle, F-25)."""
    K = skew(v)
    theta = np.linalg.norm(np.abs(v) + eps)
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * (K @ K)


def replica_dynamics(qd, st, motor_omega_d, dt, tm_drawn, cd_drawn):
    """Plain-numpy replica of Quadrotor._dynamics with the DR draws as inputs."""
    p, R, v, omega, mot = st["p"], st["R"], st["v"], st["omega"], st["motor_omega"]
    g = np.array([0.0, 0.0, -9.81])
    p_new = p + dt * v
    R_new = R @ rod_biased(dt * omega)
    f = tm_drawn * mot**2
    v_body = R.T @ v
    area = np.array([qd["frontarea_x"], qd["frontarea_y"], qd["frontarea_z"]])
    f_drag = -0.5 * qd["rho"] * cd_drawn * area * v_body * np.abs(v_body)
    f_vec = np.array([0.0, 0.0, f.sum()]) + f_drag
    acc = g + R @ f_vec / qd["mass"]
    v_new = v + dt * acc
    dmot = (motor_omega_d - mot) / qd["motor_tau"]
    dirs = np.array([-1.0, -1.0, 1.0, 1.0])
    tau_inertia = np.array([0.0, 0.0, (dmot * dirs).sum() * qd["motor_inertia"]])
    J = np.diag(qd["inertia"])
    alloc = np.array(qd["allocation_matrix"])
    tau = alloc[1:] @ f
    domega = np.linalg.solve(J, tau - np.cross(omega, J @ omega) + tau_inertia)
    omega_new = omega + dt * domega
    mot_new = (mot - motor_omega_d) * np.exp(-dt / qd["motor_tau"]) + motor_omega_d
    mot_new = np.clip(mot_new, qd["motor_omega_min"], qd["motor_omega_max"])
    return {"p": p_new, "R": R_new, "v": v_new, "omega": omega_new,
            "motor_omega": mot_new, "domega": domega, "acc": acc}


def replica_controller(qd, omega, f_T, omega_cmd):
    """Plain-numpy replica of Quadrotor._low_level_controller."""
    K = np.diag(qd["controller_K"])
    J = np.diag(qd["inertia"])
    torque_cmd = J @ K @ (omega_cmd - omega) + np.cross(omega, J @ omega)
    alpha = np.concatenate([[f_T], torque_cmd])
    f_cmd = np.array(qd["allocation_matrix_inv"]) @ alpha
    f_cmd = np.clip(f_cmd, qd["thrust_min_effective"], qd["thrust_max"])
    mot_d = np.sqrt(f_cmd / qd["thrust_map"][0])
    return np.clip(mot_d, qd["motor_omega_min"], qd["motor_omega_max"])


def replica_step(qd, st, f_d, omega_d, dt, tm_drawn, cd_drawn):
    n = round(dt / qd["dt_low_level"])
    cur = dict(st)
    for _ in range(n):
        mot_d = replica_controller(qd, cur["omega"], f_d, omega_d)
        cur = replica_dynamics(qd, cur, mot_d, qd["dt_low_level"], tm_drawn, cd_drawn)
    return cur


def random_state(rng, hover_speed):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return {
        "q_wxyz": q,
        "p": rng.uniform(-5, 5, 3),
        "R": quat_to_R(q),
        "v": rng.uniform(-10, 10, 3),
        "omega": rng.uniform(-5, 5, 3),
        "motor_omega": hover_speed * rng.uniform(0.5, 1.6, 4),
    }


def tolist(tree):
    if isinstance(tree, dict):
        return {k: tolist(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [tolist(v) for v in tree]
    if isinstance(tree, np.ndarray):
        return tree.tolist()
    if isinstance(tree, (np.floating, np.integer)):
        return tree.item()
    return tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flightning", required=True,
                    help="path to a checkout of github.com/uzh-rpg/rpg_flightning")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = pathlib.Path(args.flightning).resolve()
    out = pathlib.Path(args.out)

    sys.path.insert(0, str(repo))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp
    from flightning.objects.quadrotor_obj import Quadrotor
    from flightning.objects.quadrotor_simple_obj import quadrotor_dyn

    # provenance
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    shas = {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    prov_base = {
        "source": "rpg_flightning (Heeg, Song, Scaramuzza, ICRA 2025) — executed JAX code, "
                  "float64 (jax_enable_x64), CPU",
        "repo": "https://github.com/uzh-rpg/rpg_flightning",
        "commit": commit,
        "source_sha256": shas,
        "jax_version": jax.__version__,
        "date": datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
    }

    # constructor defaults = the Kolibri platform the repo documents its maps for
    quad = Quadrotor()
    alloc = np.asarray(quad.allocation_matrix, dtype=np.float64)          # float32-valued
    alloc_inv = np.asarray(quad.allocation_matrix_inv, dtype=np.float64)  # float32 inverse
    qd = {
        "mass": 0.75,
        "tbm": [[0.04, -0.04, 0.0], [-0.04, 0.04, 0.0],
                [-0.04, -0.04, 0.0], [0.04, 0.04, 0.0]],
        "tbm_order": ["fr", "bl", "br", "fl"],
        "inertia": [0.00014, 0.00016, 0.0002],
        "motor_omega_min": 150.0,
        "motor_omega_max": 4400.0,
        "motor_tau": 0.033,
        "motor_inertia": 2.6e-7,
        "thrust_map": [2e-7, 0.0, 0.0],
        "kappa": 0.008,
        "thrust_min_effective": float(quad._thrust_min),  # 0 -> + c_T*Omega_min^2
        "thrust_max": 3.5,
        "dt_low_level": 0.001,
        "motor_directions": [-1, -1, 1, 1],
        "spin_canonical": [1, 1, -1, -1],  # s_i = -direction_i (kappa row: -s_i*kappa*f_i)
        "gravity": [0.0, 0.0, -9.81],
        "controller_K": [20.0, 20.0, 41.0],
        "allocation_matrix": alloc,
        "allocation_matrix_inv": alloc_inv,
        "rho": 1.2,
        "cd_horizontal": 1.04,
        "cd_vertical": 1.04,
        "frontarea_x": 1.0e-3,
        "frontarea_y": 1.0e-3,
        "frontarea_z": 1.0e-2,
        "eps_rodrigues": EPS_RODRIGUES,
    }
    hover = float(quad.hovering_motor_speed)

    def dr_draws(key):
        """Replay _dynamics' exact DR key splits and draws (their code path, x64)."""
        key_thrust, key_drag = jax.random.split(key)
        tm = jnp.asarray(quad._thrust_map[0])
        tm_drawn = jax.random.uniform(key_thrust, tm.shape,
                                      minval=0.95 * tm, maxval=1.05 * tm)
        cd = jnp.array([1.04, 1.04, 1.04])
        cd_drawn = jax.random.uniform(key_drag, cd.shape,
                                      minval=0.5 * cd, maxval=1.5 * cd)
        return float(tm_drawn), np.asarray(cd_drawn, dtype=np.float64)

    rng = np.random.default_rng(RNG_SEED)

    # ---------------- 1) quadrotor_dyn primal + JVP ----------------
    simple_cases = []
    ortho_worst = 0.0  # measured F-25 magnitude, recorded in provenance notes
    for i in range(8):
        if i == 0:  # hover: omega = 0 exercises the eps guard (K = 0 -> R_delta = I exactly)
            st = random_state(rng, hover)
            st["v"] = np.zeros(3)
            st["omega"] = np.zeros(3)
            a = 9.81
            dt = 0.001
        else:
            st = random_state(rng, hover)
            a = float(rng.uniform(0.0, 25.0))
            dt = [0.001, 0.01, 0.02][i % 3]
        primals = (jnp.asarray(st["p"]), jnp.asarray(st["R"]), jnp.asarray(st["v"]),
                   jnp.asarray(a), jnp.asarray(st["omega"]), dt)
        p_n, R_n, v_n = quadrotor_dyn(*primals)
        p_n, R_n, v_n = (np.asarray(x, dtype=np.float64) for x in (p_n, R_n, v_n))

        # pre-freeze replica assertion
        np.testing.assert_allclose(p_n, st["p"] + dt * st["v"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(R_n, st["R"] @ rod_biased(dt * st["omega"]),
                                   rtol=1e-12, atol=1e-12)
        g = np.array([0.0, 0.0, -9.81])
        np.testing.assert_allclose(
            v_n, st["v"] + dt * (g + st["R"] @ np.array([0.0, 0.0, a])),
            rtol=1e-12, atol=1e-12)

        rb = rod_biased(dt * st["omega"])
        ortho_worst = max(ortho_worst, float(np.abs(rb.T @ rb - np.eye(3)).max()))

        case = {"p": st["p"], "q_wxyz": st["q_wxyz"], "R": st["R"], "v": st["v"],
                "a": a, "omega": st["omega"], "dt": dt,
                "expected": {"p": p_n, "R": R_n, "v": v_n}}

        if i > 0:  # JVP cases need omega != 0 (abs kink at exactly zero)
            u = rng.uniform(-1, 1, 3)
            tang = {"p_dot": rng.uniform(-1, 1, 3), "u_rot": u,
                    "R_dot": st["R"] @ skew(u), "v_dot": rng.uniform(-1, 1, 3),
                    "a_dot": float(rng.uniform(-1, 1)),
                    "omega_dot": rng.uniform(-1, 1, 3)}
            tangents = (jnp.asarray(tang["p_dot"]), jnp.asarray(tang["R_dot"]),
                        jnp.asarray(tang["v_dot"]), jnp.asarray(tang["a_dot"]),
                        jnp.asarray(tang["omega_dot"]), 0.0)
            _, tan_out = jax.jvp(quadrotor_dyn, primals, tangents)
            dp, dR, dv = (np.asarray(x, dtype=np.float64) for x in tan_out)
            case["jvp"] = {"tangents": tang, "expected": {"dp": dp, "dR": dR, "dv": dv}}
        simple_cases.append(case)

    # ---------------- 2) Quadrotor.step() custom_jvp (surrogate wiring) ----------------
    step_jvp_cases = []
    for i in range(3):
        st = random_state(rng, hover)
        st["omega"] = rng.uniform(-3, 3, 3)
        key = jax.random.PRNGKey(1000 + i)
        tm_drawn, cd_drawn = dr_draws(key)
        f_d = float(rng.uniform(3.0, 12.0))
        omega_d = rng.uniform(-3, 3, 3)
        dt = 0.02
        state = quad.create_state(
            jnp.asarray(st["p"]), jnp.asarray(st["R"]), jnp.asarray(st["v"]),
            omega=jnp.asarray(st["omega"]), motor_omega=jnp.asarray(st["motor_omega"]),
            dr_key=key)

        tang = {"p_dot": rng.uniform(-1, 1, 3),
                "u_rot": rng.uniform(-1, 1, 3),
                "v_dot": rng.uniform(-1, 1, 3),
                "f_d_dot": float(rng.uniform(-1, 1)),
                "omega_d_dot": rng.uniform(-1, 1, 3)}
        tang["R_dot"] = st["R"] @ skew(tang["u_rot"])

        state_tan = state.replace(
            p=jnp.asarray(tang["p_dot"]), R=jnp.asarray(tang["R_dot"]),
            v=jnp.asarray(tang["v_dot"]), omega=jnp.zeros(3), domega=jnp.zeros(3),
            motor_omega=jnp.zeros(4), acc=jnp.zeros(3),
            dr_key=np.zeros(key.shape, dtype=jax.dtypes.float0))
        primal_out, tan_out = jax.jvp(
            lambda s, f, o, dt=dt: quad.step(s, f, o, dt),
            (state, jnp.asarray(f_d), jnp.asarray(omega_d)),
            (state_tan, jnp.asarray(tang["f_d_dot"]), jnp.asarray(tang["omega_d_dot"])))

        exp_primal = {k: np.asarray(getattr(primal_out, k), dtype=np.float64)
                      for k in ("p", "R", "v", "omega", "motor_omega")}
        # replica assertion: controller-in-the-loop rollout
        rep = replica_step(qd, st, f_d, omega_d, dt, tm_drawn, cd_drawn)
        for k, val in exp_primal.items():
            np.testing.assert_allclose(val, rep[k], rtol=1e-9, atol=1e-9,
                                       err_msg=f"step replica mismatch [{k}]")
        # surrogate tangent must equal the simple-model JVP with c = f_d/m, dt-tangent 0
        _, tan_ref = jax.jvp(
            quadrotor_dyn,
            (jnp.asarray(st["p"]), jnp.asarray(st["R"]), jnp.asarray(st["v"]),
             jnp.asarray(f_d / qd["mass"]), jnp.asarray(omega_d), dt),
            (jnp.asarray(tang["p_dot"]), jnp.asarray(tang["R_dot"]),
             jnp.asarray(tang["v_dot"]), jnp.asarray(tang["f_d_dot"] / qd["mass"]),
             jnp.asarray(tang["omega_d_dot"]), 0.0))
        for got, ref in zip((tan_out.p, tan_out.R, tan_out.v), tan_ref):
            np.testing.assert_allclose(np.asarray(got), np.asarray(ref),
                                       rtol=1e-12, atol=1e-12)

        step_jvp_cases.append({
            "state": {k: st[k] for k in ("p", "q_wxyz", "R", "v", "omega", "motor_omega")},
            "dr": {"key_seed": 1000 + i, "thrust_map_drawn": tm_drawn,
                   "drag_coeff_drawn": cd_drawn},
            "f_d": f_d, "omega_d": omega_d, "dt": dt, "tangents": tang,
            "expected_primal": exp_primal,
            "expected_tangent": {"dp": np.asarray(tan_out.p, dtype=np.float64),
                                 "dR": np.asarray(tan_out.R, dtype=np.float64),
                                 "dv": np.asarray(tan_out.v, dtype=np.float64)},
        })

    # ---------------- 3) _dynamics single substeps (physics, no controller) ----------------
    dynamics_cases = []
    specs = [
        # (description-free tweaks): hover, generic x4, fast spin, high speed, clip-high, clip-low
        {"i": 0, "hoverish": True, "dt": 0.001},
        {"i": 1, "dt": 0.001}, {"i": 2, "dt": 0.001}, {"i": 3, "dt": 0.001},
        {"i": 4, "dt": 0.0005}, {"i": 5, "dt": 0.002},
        {"i": 6, "fast": True, "dt": 0.001},
        {"i": 7, "swift": True, "dt": 0.001},
        {"i": 8, "clip_high": True, "dt": 0.001},
        {"i": 9, "clip_low": True, "dt": 0.001},
    ]
    for spec in specs:
        st = random_state(rng, hover)
        dt = spec["dt"]
        if spec.get("hoverish"):
            st["q_wxyz"] = np.array([1.0, 0.0, 0.0, 0.0])
            st["R"] = np.eye(3)
            st["v"] = np.zeros(3)
            st["omega"] = np.zeros(3)
            st["motor_omega"] = np.full(4, hover)
            mot_d = np.full(4, hover)
        elif spec.get("fast"):
            st["omega"] = np.array([5.5, -5.5, 3.8])
            mot_d = hover * rng.uniform(0.8, 1.2, 4)
        elif spec.get("swift"):
            st["v"] = np.array([14.0, -9.0, 4.0])
            mot_d = hover * rng.uniform(0.8, 1.2, 4)
        elif spec.get("clip_high"):
            st["motor_omega"] = np.full(4, 4380.0)
            mot_d = np.full(4, 4400.0)  # exact-exp + clip pins the upper limit
            dt = 0.05  # long substep drives Omega into the clip
        elif spec.get("clip_low"):
            st["motor_omega"] = np.full(4, 120.0)  # below Omega_min: clip pulls up to 150
            mot_d = np.full(4, 150.0)
        else:
            mot_d = hover * rng.uniform(0.6, 1.4, 4)
        key = jax.random.PRNGKey(2000 + spec["i"])
        tm_drawn, cd_drawn = dr_draws(key)
        state = quad.create_state(
            jnp.asarray(st["p"]), jnp.asarray(st["R"]), jnp.asarray(st["v"]),
            omega=jnp.asarray(st["omega"]), motor_omega=jnp.asarray(st["motor_omega"]),
            dr_key=key)
        st_out = quad._dynamics(state, jnp.asarray(mot_d), dt)
        exp = {k: np.asarray(getattr(st_out, k), dtype=np.float64)
               for k in ("p", "R", "v", "omega", "motor_omega", "domega", "acc")}
        rep = replica_dynamics(qd, st, mot_d, dt, tm_drawn, cd_drawn)
        for k, val in exp.items():
            np.testing.assert_allclose(val, rep[k], rtol=1e-11, atol=1e-11,
                                       err_msg=f"_dynamics replica mismatch [{k}]")
        dynamics_cases.append({
            "state": {k: st[k] for k in ("p", "q_wxyz", "R", "v", "omega", "motor_omega")},
            "motor_omega_d": mot_d, "dt": dt,
            "dr": {"key_seed": 2000 + spec["i"], "thrust_map_drawn": tm_drawn,
                   "drag_coeff_drawn": cd_drawn},
            "expected": exp,
        })

    # ---------------- write ----------------
    simple_doc = {
        "schema": 1, "kind": "flightning_terms", "name": "flightning_simple_jvp",
        "provenance": {
            **prov_base, "generator": "golden/generate/gen_flightning.py",
            "notes": "quadrotor_dyn primal + jax.jvp tangents (surrogate-gradient scheme); "
                     "step() custom_jvp cases pin the wiring: primal = N low-level substeps "
                     "(controller in loop), tangent = simple-model JVP at c = f_d/m with "
                     "dt-tangent 0. Tangent slots omega/domega/motor_omega/acc pass the "
                     "INPUT tangents through unchanged (not recorded). R tangents are seeded "
                     "as right-translated so(3) vectors: R_dot = R@skew(u_rot). "
                     f"Measured F-25 magnitudes over these cases: max |R_delta^T R_delta - I| "
                     f"= {ortho_worst:.3e} (biased-angle Rodrigues is not exactly SO(3)).",
        },
        "gravity": [0.0, 0.0, -9.81],
        "quad": {"mass": qd["mass"], "eps_rodrigues": EPS_RODRIGUES},
        "cases": simple_cases,
        "step_jvp_cases": step_jvp_cases,
    }
    full_doc = {
        "schema": 1, "kind": "flightning_terms", "name": "flightning_full_step",
        "provenance": {
            **prov_base, "generator": "golden/generate/gen_flightning.py",
            "notes": "_dynamics substeps take commanded motor speeds directly (physics only); "
                     "step cases run the low-level P controller (harness detail, gains "
                     "recorded) at 1 kHz. DR draws replayed from the recorded key seeds and "
                     "stored as effective parameters (thrust map x[0.95,1.05], drag coeff "
                     "x[0.5,1.5]; dr_key never advances, so draws are per-episode constants). "
                     "allocation_matrix / _inv recorded AS EXECUTED (float32 entries/inverse "
                     "in a float64 pipeline, finding F-27). thrust_min_effective = "
                     "c_T*Omega_min^2 (constructor promotes thrust_min=0).",
        },
        "quad": qd,
        "dynamics_cases": dynamics_cases,
        "step_cases": [  # primal-only view of the step_jvp cases, kept with the full model
            {k: c[k] for k in ("state", "dr", "f_d", "omega_d", "dt", "expected_primal")}
            for c in step_jvp_cases
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in [("flightning_simple_jvp.json", simple_doc),
                      ("flightning_full_step.json", full_doc)]:
        (out / name).write_text(json.dumps(tolist(doc), indent=1))
        print(f"wrote {out / name}")
    print(f"F-25 measured: max orthogonality defect {ortho_worst:.3e}, "
          f"biased-vs-exact handled by consumer deviation test")


if __name__ == "__main__":
    main()
