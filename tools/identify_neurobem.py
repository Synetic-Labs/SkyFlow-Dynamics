"""
Identify the 5-inch racer reference vehicle (spec.parameters.RACER_5IN) from the public
NeuroBEM flight data — the evidence behind its values and behind finding F-30.

    uv run python tools/identify_neurobem.py --data /path/to/neurobem [--write]

Data (not redistributed): https://download.ifi.uzh.ch/rpg/NeuroBEM/ — processed_data.zip
(sha256 a3e2e83d…) and testset.txt (sha256 d59cd589…, the authors' 13 held-out segments),
both in --data. 1h15 of agile flight of the UZH-RPG Kingfisher quad at 400 Hz, 247 segments.
Body frame FLU, world z up; "acc" is the specific force (+9.81 on z in hover), so the force on
the airframe is m·acc. Motor columns: back-right, front-right, back-left, front-left. Labels
as the authors make them: m = 0.772 kg, J = diag(0.0025, 0.0021, 0.0043), τ = J ω̇ + ω × Jω.

Model: the backend's per-rotor terms (rotor_thrust_polynomial, rotor_drag_hforce,
parasitic_drag), hub airspeed v_i = v + ω × r_i, still air:
    F = Σᵢ [c_T Ωᵢ² ẑ − Ωᵢ diag(k_d, k_d, k_z) v_i] − ‖v‖ diag(c_Dxy, c_Dxy, c_Dz) v
Geometry: symmetric X with a 0.13 m arm (the dataset code's setupKingfisher.m); FR and BL
spin counter-clockwise.

What the data identify:
  - FORCES identify c_T, k_d and the frame drag (95 % bootstrap over training segments,
    checked on the held-out set). k_z is not separable from c_Dz (both oppose v_z), so k_z
    comes from BEM (tools/bem_rate_damping.py): the per-rotor value whose lever-arm roll
    damping 4·Ω_h·k_z·a² equals BEM's at 13 rad/s, the largest body rate in the data.
  - MOMENTS do not close (finding F-30): the roll torque the motor-speed differentials imply
    is several times the measured J ω̇, so coefficients on motor signals are biased toward
    zero. c_Q, I_rot, J and the 8.5 N thrust cap stay at the agilib config values.

--write freezes the result in golden/checkdata/neurobem_racer.json, which
properties/test_findings.py checks spec.parameters.RACER_5IN against.
"""

import argparse
import hashlib
import io
import json
import math
import pathlib
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import bem_rate_damping as brd

OUT = ROOT / "golden" / "checkdata" / "neurobem_racer.json"
SHA256 = {"processed_data.zip": "a3e2e83d156b48a95c346c4fd9b3ded0b62781331dc613f1cb9ab285be2b36a8",
          "testset.txt": "d59cd589105cf4cd3fcfe0a34a0b4824d6ae2560db13d34b9a7acd169deeb367"}
M_DATA = 0.772
J_DATA = np.array([0.0025, 0.0021, 0.0043])
SPIN = np.array([-1.0, 1.0, 1.0, -1.0])     # BR, FR, BL, FL (+1 = counter-clockwise)
SIGN_X = np.array([-1.0, 1.0, -1.0, 1.0])   # rotor x = SIGN_X·a
SIGN_Y = np.array([-1.0, -1.0, 1.0, 1.0])   # rotor y = SIGN_Y·a
ARM = 0.13 / math.sqrt(2)
STRIDE = 4          # 400 Hz → 100 Hz: neighbouring samples are filtered, not independent
K_Z_RATE = 13.0     # rad/s: where k_z is matched to BEM's roll damping
NAMES = ("c_T", "k_d", "c_Dxy", "c_Dz")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(data: pathlib.Path):
    """Every processed segment (strided), cached as one npz next to the data."""
    for name, digest in SHA256.items():
        if sha256(data / name) != digest:
            raise SystemExit(f"{name}: sha256 mismatch — not the pinned dataset")
    cache = data / f"segments_stride{STRIDE}.npz"
    if cache.exists():
        z = np.load(cache)
        segs = {k: z[k] for k in z.files}
    else:
        segs = {}
        with zipfile.ZipFile(data / "processed_data.zip") as zf:
            for member in sorted(zf.namelist()):
                if member.endswith(".csv"):
                    name = pathlib.Path(member).stem.removeprefix("merged_")
                    text = io.TextIOWrapper(zf.open(member), encoding="utf-8")
                    segs[name] = np.loadtxt(text, delimiter=",", skiprows=1)[::STRIDE]
        np.savez(cache, **segs)
    test = {s.strip() for s in (data / "testset.txt").read_text().splitlines() if s.strip()}
    return segs, test


def unpack(d):
    return {"w_dot": d[:, 1:4], "w": d[:, 4:7], "acc": d[:, 11:14], "v": d[:, 14:17],
            "W": d[:, 20:24]}


def force_rows(s):
    """(A, b) for F = A·θ, θ = (c_T, k_d, k_z, c_Dxy, c_Dz); rows x, y, z stacked."""
    v, w, W = s["v"], s["w"], s["W"]
    rx, ry = SIGN_X * ARM, SIGN_Y * ARM
    vx = v[:, None, 0] - w[:, None, 2] * ry
    vy = v[:, None, 1] + w[:, None, 2] * rx
    vz = v[:, None, 2] + w[:, None, 0] * ry - w[:, None, 1] * rx
    speed = np.linalg.norm(v, axis=1)
    A = np.zeros((3, len(v), 5))
    A[0, :, 1] = -(W * vx).sum(1)
    A[1, :, 1] = -(W * vy).sum(1)
    A[2, :, 0] = (W ** 2).sum(1)
    A[2, :, 2] = -(W * vz).sum(1)
    A[0, :, 3] = -speed * v[:, 0]
    A[1, :, 3] = -speed * v[:, 1]
    A[2, :, 4] = -speed * v[:, 2]
    return A.reshape(-1, 5), (M_DATA * s["acc"]).T.reshape(-1)


def fit(normal_eqs, k_z):
    """Least squares for (c_T, k_d, c_Dxy, c_Dz) with k_z held at the BEM value."""
    AtA = np.sum([e[0] for e in normal_eqs], axis=0)
    Atb = np.sum([e[1] for e in normal_eqs], axis=0)
    free = [0, 1, 3, 4]
    rhs = Atb[free] - AtA[free, 2] * k_z
    return np.linalg.solve(AtA[np.ix_(free, free)], rhs)


def force_rmse(segs, names, theta):
    err = []
    for k in names:
        A, b = force_rows(unpack(segs[k]))
        err.append((A @ theta - b).reshape(3, -1))
    err = np.concatenate(err, axis=1)
    return {"F_xy": float(np.sqrt((err[:2] ** 2).sum(0).mean())),
            "F_z": float(np.sqrt((err[2] ** 2).mean()))}


def bem_k_z() -> tuple[float, float]:
    """k_z whose lever-arm roll damping 4·Ω_h·k_z·a² equals BEM's at K_Z_RATE; and Ω_h."""
    d = brd.damping("canonical", (K_Z_RATE,))
    W_h = d["hover_rotor_speed"]
    return d["roll"][K_Z_RATE] / (4 * W_h * ARM * ARM), W_h


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--write", action="store_true", help=f"freeze the result in {OUT.name}")
    args = ap.parse_args(argv)

    segs, test = load(args.data)
    train = sorted(k for k in segs if k not in test)
    held = sorted(k for k in segs if k in test)
    print(f"{len(train)} training / {len(held)} held-out segments")

    k_z, W_h_bem = bem_k_z()
    eqs = [(A.T @ A, A.T @ b) for A, b in (force_rows(unpack(segs[k])) for k in train)]
    est = fit(eqs, k_z)
    rng = np.random.default_rng(0)
    boots = np.array([fit([eqs[i] for i in rng.integers(0, len(eqs), len(eqs))], k_z)
                      for _ in range(args.boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ident = dict(zip(NAMES, map(float, est), strict=True))
    ci95 = {n: [float(a), float(b)] for n, a, b in zip(NAMES, lo, hi, strict=True)}
    print(f"k_z (BEM roll damping at {K_Z_RATE:.0f} rad/s): {k_z:.4e}")
    for n in NAMES:
        print(f"  {n:6s} {ident[n]: .4e}   95 % [{ci95[n][0]: .4e}, {ci95[n][1]: .4e}]")

    # F-30: the roll lever arm that the thrust differential implies against measured J ω̇
    num = den = 0.0
    for k in train:
        s = unpack(segs[k])
        u = ident["c_T"] * (SIGN_Y * s["W"] ** 2).sum(1)
        tau_x = J_DATA[0] * s["w_dot"][:, 0] + (J_DATA[2] - J_DATA[1]) * s["w"][:, 1] * s["w"][:, 2]
        num, den = num + u @ tau_x, den + u @ u
    implied_arm = num / den
    print(f"F-30: roll torque vs thrust differential implies a {implied_arm * 1e3:.1f} mm "
          f"lever arm (geometry {ARM * 1e3:.1f} mm)")

    theta = np.array([ident["c_T"], ident["k_d"], k_z, max(ident["c_Dxy"], 0.0), ident["c_Dz"]])
    rmse = {"zero force": force_rmse(segs, held, np.zeros(5)),
            "identified (RACER_5IN)": force_rmse(segs, held, theta)}
    rmse["NeuroBEM Table II: BEM"] = {"F_xy": 0.803, "F_z": 1.265}
    rmse["NeuroBEM Table II: polynomial fit"] = {"F_xy": 1.536, "F_z": 1.381}
    for label, r in rmse.items():
        print(f"  held-out RMSE  {label:34s} F_xy {r['F_xy']:.3f} N  F_z {r['F_z']:.3f} N")

    allw = np.concatenate([unpack(segs[k])["w"] for k in train])
    allv = np.concatenate([unpack(segs[k])["v"] for k in train])
    allW = np.concatenate([unpack(segs[k])["W"] for k in train])
    speed = np.linalg.norm(allv, axis=1)
    envelope = {"body_rate_max": np.abs(allw).max(0).tolist(),
                "body_rate_p99": np.percentile(np.abs(allw), 99, axis=0).tolist(),
                "speed_max": float(speed.max()),
                "hover_rotor_speed": float(np.median(allW[speed < 0.5]))}
    print(f"  envelope: |ω| max {np.round(envelope['body_rate_max'], 1)} rad/s, speed max "
          f"{envelope['speed_max']:.1f} m/s, hover {envelope['hover_rotor_speed']:.0f} rad/s "
          f"(BEM {W_h_bem:.0f})")

    if args.write:
        OUT.write_text(json.dumps({
            "kind": "neurobem_racer_identification",
            "provenance": {"data": "https://download.ifi.uzh.ch/rpg/NeuroBEM/",
                           "sha256": SHA256, "stride": STRIDE,
                           "train_segments": len(train), "test_segments": len(held),
                           "generator": "tools/identify_neurobem.py"},
            "fixed": {"mass": M_DATA, "inertia_diag": J_DATA.tolist(), "arm": 0.13},
            "k_z_bem": k_z, "k_z_rate": K_Z_RATE, "force_fit": ident, "ci95": ci95,
            "implied_roll_arm": float(implied_arm), "held_out_rmse": rmse,
            "envelope": envelope}, indent=1) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
