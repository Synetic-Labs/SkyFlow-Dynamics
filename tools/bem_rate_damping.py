"""
Body-rate damping of the verified BEM rotor model (finding F-29) — the evidence behind the
registry entries rotor_rate_damping_nonlinear and flapping_moment_body_rate.

    uv run python tools/bem_rate_damping.py

Runs the float-exact agilib BEM replica in golden/generate/gen_agilicious.py (the code path
the verified bem_* terms reproduce) with the measured 5.1-inch three-blade prop of
golden/vectors/agilicious_bem.json, at hover, with a pure body rate about one axis, and
reports the damping derivative D(p) = −M(p)/p for roll, pitch and yaw.

Two hub-velocity variants:
  - "agilib": the executed code as-is. Its hub velocity mixes an FLU lever arm with an FRD
    body rate (finding F-24), so a roll rate gives the wrong-signed vertical flow at each
    rotor.
  - "canonical": the spec's v + ω × r, with both factors in FRD before the cross product.
    This is the variant the spec adopts (F-24) and the one the registry numbers quote.

Geometry: symmetric X with a 0.13 m arm and 0.772 kg — the NeuroBEM flight-data platform
(setupKingfisher.m in the dataset code; the mass the dataset labels use). The golden file's
own quad (agilib sim_kingfisher.yaml: 0.15 m arm, 0.752 kg) differs only in these two values.
"""

import importlib.util
import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPLICA = ROOT / "golden" / "generate" / "gen_agilicious.py"
VECTORS = ROOT / "golden" / "vectors" / "agilicious_bem.json"
MASS, ARM = 0.772, 0.13
RATES = (1.0, 10.0, 13.0, 50.0, 100.0, 200.0)  # INTAKE operating points + 13 rad/s, the NeuroBEM data maximum


def load(variant: str = "canonical"):
    """A private module instance of the replica (patching it leaves other imports alone)."""
    spec = importlib.util.spec_from_file_location(f"_bem_replica_{variant}", REPLICA)
    assert spec is not None and spec.loader is not None, REPLICA
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    if variant == "canonical":
        update = ga.PState.update

        def update_canonical(self, v_W, q, w_B, mot, t_BM):
            update(self, v_W, q, w_B, mot, t_BM)
            v_frd = ga.FLU * (self.rot.T @ np.asarray(v_W, float))
            for i in range(4):
                self.velocity[:, i] = v_frd + np.cross(self.w_frd, ga.FLU * t_BM[:, i])
            self.vhor = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2) + 1e-6
            self.vver = self.velocity[2].copy()
            self.vtot = np.sqrt((self.velocity ** 2).sum(axis=0)) + 1e-6
            self.alpha_s = np.array([ga.approx_atan2(self.vver[i], self.vhor[i])
                                     for i in range(4)])
            self.mu = self.vhor / (self.omega_mot * self.bp["r_prop"])

        ga.PState.update = update_canonical
    elif variant != "agilib":
        raise ValueError(variant)
    doc = json.loads(VECTORS.read_text())
    quad = dict(doc["quad"])
    a = ARM / math.sqrt(2)
    quad["t_BM"] = [[a, -a, 0.0], [-a, a, 0.0], [-a, -a, 0.0], [a, a, 0.0]]  # FLU: fr bl br fl
    quad["mass"] = MASS
    return ga, dict(doc["bem_params"]), quad


def wrench(model, w=(0.0, 0.0, 0.0), W=1100.0):
    """Body force and torque (FLU) of the four BEM rotors at rest in still air."""
    ga, bp, quad = model
    case = {"v_W": [0.0, 0.0, 0.0], "q_wxyz": [1, 0, 0, 0], "w_B": list(w), "mot": [W] * 4}
    r = ga.run_bem(bp, quad, case, np.full(4, 5.0), np.full(4, 5.0))
    force = r["dvel"] * MASS + np.array([0.0, 0.0, quad["G"] * MASS])
    return force, np.asarray(quad["J_diag"]) * r["dome"]


def hover_speed(model) -> float:
    lo, hi = 600.0, 2000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wrench(model, W=mid)[0][2] < MASS * 9.81 else (lo, mid)
    return 0.5 * (lo + hi)


def damping(variant: str = "canonical", rates=RATES) -> dict:
    """D(p) = −M_axis/p [N·m·s/rad] per axis at hover; positive = damping."""
    model = load(variant)
    W_h = hover_speed(model)
    out = {"variant": variant, "hover_rotor_speed": W_h, "roll": {}, "pitch": {}, "yaw": {}}
    for p in rates:
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            w = [0.0, 0.0, 0.0]
            w[axis] = p
            out[name][p] = float(-wrench(model, w, W_h)[1][axis] / p)
    return out


def main() -> None:
    for variant in ("canonical", "agilib"):
        d = damping(variant)
        print(f"{variant}: hover {d['hover_rotor_speed']:.0f} rad/s")
        print("   rate rad/s    roll       pitch      yaw     [N·m·s/rad]")
        for p in RATES:
            print(f"   {p:6.0f}     {d['roll'][p]: .5f}   {d['pitch'][p]: .5f}   "
                  f"{d['yaw'][p]: .5f}")


if __name__ == "__main__":
    main()
