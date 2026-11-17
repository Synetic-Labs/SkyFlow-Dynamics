"""
Whole-quadcopter ground and ceiling effect at measured rotor speed, from public data — the
evidence behind finding F-36.

    uv run --with unrar-cffi python tools/ground_data.py --data /path/to/cache [--write]

Data (fetched into --data when missing, checked by sha256, not redistributed):
  - David Du Mutel, Scanavino, Conyers et al., "Ground, ceiling and wall effect evaluation of
    small quadcopters in pressure-controlled environments", J. Intell. Robot. Syst. 110:125
    (2024), doi 10.1007/s10846-024-02155-7. Supplementary data: Zenodo record 13166292
    (CC-BY 4.0), one RAR5 archive. unrar-cffi extracts the four CSV files used here.

Setup (read from the paper, §2.1.1, §3.1, Table 1): a quadcopter frame in 'x' layout, 500 mm
between diagonal motors, all motors in one plane, on a six-axis load cell in a climatic
chamber (20 °C, 1000 hPa). T-Motor 2212 920 kV, 10×4.7 and 12×5 propellers. Open-loop PWM,
the same to all four ESCs, four levels, each held 50 s and run twice. Two tachometers log
two of the four rotors. A 1 m² panel above the vehicle at D = 5, 15, 25, 35 and 100 cm from
the propeller plane: with the propellers mounted normally it is a ceiling; for ground
effect the propellers are reversed so that the wake blows up onto it.

What the data measure, and what they do not:
  - The ratio at matched rotor speed, (F/Ω̄²)(D)/(F/Ω̄²)(100 cm), is the constant-speed
    thrust ratio of ground_effect_fixed_rpm: four rotors, tip gap 0.32 R (12 in) and
    0.78 R (10 in).
  - In the ground runs the frame is UPSTREAM of the rotors, so no body sits in the
    fountain: the body lift of ground_effect_sanchez_cuevas cannot appear.
  - Each propeller's 100 cm row is identical in the ground and ceiling files, so the ground
    runs share a reference taken with the propellers mounted normally, where the arms sit in
    the wake (inferred from the files; the paper does not say). The ground ratios carry that
    arm download as a bias of about +1 to +3 % (estimated, not measured).
  - The 10 in tachometers disagree by 4–8 % at every height below 100 cm (agree to 1 % at
    100 cm); the 12 in ones agree to about 1 %. The ratio is reported with the mean speed
    and with each tachometer alone; the 12 in runs are the reliable set.

--write freezes the results in golden/checkdata/ground_ceiling.json, which
properties/test_wake.py checks.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import urllib.request

import numpy as np
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skyflow_dynamics.spec import ground_effect

OUT = ROOT / "golden" / "checkdata" / "ground_ceiling.json"
ARCHIVE_URL = "https://zenodo.org/api/records/13166292/files/SI.rar/content"
ARCHIVE = "SI.rar"
SHA256 = {
    "SI.rar": "bd8675ec8f36a7adf8a6bc2c5f001dfa0d0f9affef4a368c433b75cca4140e26",
    "GroundEffect_10_txc.csv": "448615f5a6df96f826361cbece3bffffef98c58728ab2d38dc24ff88f0995108",
    "GroundEffect_12_txc.csv": "bf772304e0ec22757889f8b2f81b95ba0bdc075626d62f6504050bb09c7479c1",
    "CeilingEffect_10_txc.csv": "c23930c4336953e75aaeaf5c1ae4c54c0af5cb31faf60f3d6bfb696aa1ce0819",
    "CeilingEffect_12_txc.csv": "be78c1021890214711c9fe97d0d40050f24ea969d02339fac9f9081bf74e2c51",
}
INCH = 0.0254
DIAGONAL = 0.500                       # m, between non-consecutive motors (paper §2.1.1)
ADJACENT = DIAGONAL / np.sqrt(2)       # m, square 'x' layout
REFERENCE_CM = 100.0
#: Candidate constants under test (REFERENCES.md F-33, F-35).
HE_LEANG_5X3 = (0.53, 1.80)            # C_a, C_b: He & Leang Table 2, three-blade 5×3
S_DESCENT = 0.44                       # F-35 median inflow sensitivity
G_PYBULLET = 11.36859
K_B_SANCHEZ = 2.0


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(data: pathlib.Path) -> None:
    """Download the archive, then extract ONLY the pinned CSV files, by base name."""
    data.mkdir(parents=True, exist_ok=True)
    wanted = [n for n in SHA256 if n != ARCHIVE]
    if all((data / n).exists() for n in wanted):
        for n in wanted:
            if sha256(data / n) != SHA256[n]:
                raise SystemExit(f"{n}: sha256 mismatch — not the pinned version")
        return
    arc = data / ARCHIVE
    if not arc.exists():
        req = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "skyflow-dynamics"})
        arc.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    if sha256(arc) != SHA256[ARCHIVE]:
        raise SystemExit(f"{ARCHIVE}: sha256 mismatch — not the pinned version")
    try:
        from unrar.cffi import rarfile
    except ImportError:
        raise SystemExit("RAR5 archive: run with `uv run --with unrar-cffi python "
                         "tools/ground_data.py ...`, or extract the four CSV files of "
                         f"{arc} into {data} by hand")
    rf = rarfile.RarFile(str(arc))
    for info in rf.infolist():
        base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
        if base in wanted:
            (data / base).write_bytes(rf.read(info))
    for n in wanted:
        if not (data / n).exists() or sha256(data / n) != SHA256[n]:
            raise SystemExit(f"{n}: missing from the archive or sha256 mismatch")


def load(path: pathlib.Path) -> list[dict]:
    """Rows of D (cm), PWM, Ω1, Ω2 (rpm; None where the sensor failed), F_z (N)."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.reader(fh):
            try:
                dist = float(r[0])
            except (ValueError, IndexError):
                continue
            w = [float(x) if x.strip() not in ("-", "") else None for x in r[4:6]]
            rows.append({"D_cm": dist, "pwm": int(float(r[3])), "w1": w[0], "w2": w[1],
                         "F": float(r[6])})
    return rows


def ratios(rows: list[dict], radius: float) -> list[dict]:
    """Per height: thrust ratio at matched rotor speed against the 100 cm row of the same
    PWM level, with the mean speed and with each tachometer alone."""
    ref = {r["pwm"]: r for r in rows if r["D_cm"] == REFERENCE_CM}

    def k(r, which):
        w = [r[x] for x in which if r[x] is not None]
        return r["F"] / np.mean(w) ** 2 if w else np.nan

    def mean_finite(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else None

    out = []
    for dist in sorted({r["D_cm"] for r in rows if r["D_cm"] != REFERENCE_CM}):
        level = sorted((r for r in rows if r["D_cm"] == dist), key=lambda r: r["pwm"])
        per = {name: [k(r, cols) / k(ref[r["pwm"]], cols) for r in level]
               for name, cols in (("mean", ("w1", "w2")), ("w1", ("w1",)), ("w2", ("w2",)))}
        mean = np.array(per["mean"])
        out.append({"D_cm": dist, "z_over_R": dist / 100 / radius,
                    "pwm": [r["pwm"] for r in level],
                    "thrust_N": [r["F"] for r in level],
                    "ratio_per_pwm": [float(x) for x in mean],
                    "ratio": float(np.mean(mean)),
                    "ratio_range": [float(mean.min()), float(mean.max())],
                    "ratio_tach1": mean_finite(per["w1"]),
                    "ratio_tach2": mean_finite(per["w2"]),
                    "tach_gap": max((abs(r["w1"] - r["w2"]) / r["w1"] for r in level
                                     if r["w2"] is not None), default=None)})
    return out


def fmt(x) -> str:
    return "  n/a" if x is None else f"{x:5.3f}"


# ---------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------

def candidates(z_over_r: float, radius: float) -> dict:
    """T/T_∞ of each ground-effect candidate at this height, with its published constants."""
    z = z_over_r * radius
    return {
        "backend": 1.0,                 # the backend carries no ground effect
        "cheeseman_bennett": float(ground_effect.cheeseman_bennett(z, radius)),
        "cheeseman_bennett_fixed_pitch_s044": float(
            ground_effect.cheeseman_bennett_fixed_pitch(z, radius, S_DESCENT)),
        "he_leang_5x3": float(ground_effect.he_leang_exponential(z, radius, *HE_LEANG_5X3)),
        "sanchez_cuevas_kb2": float(ground_effect.sanchez_cuevas(
            z, radius, ADJACENT, DIAGONAL, K_B_SANCHEZ)),
        "sanchez_cuevas_kb0": float(ground_effect.sanchez_cuevas(
            z, radius, ADJACENT, DIAGONAL, 0.0)),
        "pybullet": 1 + G_PYBULLET * (1 / (4 * z_over_r)) ** 2,
    }


def he_leang(x, c_a, c_b):
    return 1 + c_a * np.exp(-c_b * np.asarray(x))


def fit_he_leang(points: list[dict]) -> dict:
    x = np.array([p["z_over_R"] for p in points])
    y = np.array([p["ratio"] for p in points])
    sol = least_squares(lambda q: he_leang(x, *q) - y, [0.5, 1.8],
                        bounds=([0.0, 0.1], [3.0, 10.0]))
    return {"C_a": float(sol.x[0]), "C_b": float(sol.x[1]),
            "rms": float(np.sqrt(np.mean(sol.fun ** 2)))}


def fit_fixed_pitch_s(points: list[dict], radius: float) -> dict:
    """s of Cheeseman & Bennett's fixed-pitch form, T/T_∞ = 1 + s·(R/4z)², by least squares."""
    x = np.array([p["z_over_R"] for p in points])
    y = np.array([p["ratio"] for p in points])
    f = (1 / (4 * x)) ** 2
    s = float(f @ (y - 1) / (f @ f))
    return {"s": s, "rms": float(np.sqrt(np.mean((1 + s * f - y) ** 2)))}


def rms(model: list[float], points: list[dict]) -> float:
    return float(np.sqrt(np.mean([(m - p["ratio"]) ** 2 for m, p in zip(model, points)])))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True, help="cache for the sources")
    ap.add_argument("--write", action="store_true", help=f"freeze the result in {OUT.name}")
    args = ap.parse_args(argv)
    fetch(args.data)

    result = {"kind": "ground_ceiling",
              "provenance": {
                  "paper": "David Du Mutel et al., J. Intell. Robot. Syst. 110:125 (2024), "
                           "doi 10.1007/s10846-024-02155-7",
                  "data": "Zenodo 13166292 (CC-BY 4.0)", "url": ARCHIVE_URL,
                  "sha256": SHA256, "generator": "tools/ground_data.py",
                  "geometry": {"diagonal_m": DIAGONAL, "adjacent_m": float(ADJACENT)},
                  "reference_cm": REFERENCE_CM},
              "ground": {}, "ceiling": {}}
    for inch in (12, 10):
        radius = inch * INCH / 2
        name = f"{inch}in"
        g = ratios(load(args.data / f"GroundEffect_{inch}_txc.csv"), radius)
        c = ratios(load(args.data / f"CeilingEffect_{inch}_txc.csv"), radius)
        for p in g:
            p["models"] = candidates(p["z_over_R"], radius)
            p["pybullet_G_implied"] = (p["ratio"] - 1) * (4 * p["z_over_R"]) ** 2
        model_rms = {m: rms([p["models"][m] for p in g], g) for m in g[0]["models"]}
        result["ground"][name] = {
            "R_m": radius, "tip_gap_over_R": float((ADJACENT - 2 * radius) / radius),
            "points": g, "model_rms": model_rms,
            "fit_he_leang": fit_he_leang(g), "fit_fixed_pitch": fit_fixed_pitch_s(g, radius)}
        result["ceiling"][name] = {"R_m": radius, "points": c}

        print(f"\n{name}: R = {radius*100:.2f} cm, tip gap {(ADJACENT - 2*radius)/radius:.2f} R")
        print(f"  ground  {'z/R':>5s} {'ratio':>6s} {'range':>13s} {'tach1':>6s} {'tach2':>6s} "
              f"{'gap':>5s} | " + " ".join(f"{m[:14]:>14s}" for m in g[0]["models"]))
        for p in g:
            print(f"  {p['D_cm']:4.0f}cm {p['z_over_R']:5.2f} {p['ratio']:6.3f} "
                  f"[{p['ratio_range'][0]:.3f},{p['ratio_range'][1]:.3f}] "
                  f"{fmt(p['ratio_tach1'])} {fmt(p['ratio_tach2'])} {fmt(p['tach_gap'])} | "
                  + " ".join(f"{v:14.3f}" for v in p["models"].values())
                  + f"   G needed {p['pybullet_G_implied']:.1f}")
        print("  rms vs data: " + ", ".join(f"{m} {v:.3f}" for m, v in model_rms.items()))
        fh, fs = result["ground"][name]["fit_he_leang"], result["ground"][name]["fit_fixed_pitch"]
        print(f"  fit He & Leang: C_a {fh['C_a']:.2f}, C_b {fh['C_b']:.2f}, rms {fh['rms']:.3f};"
              f"  fit fixed-pitch C-B: s {fs['s']:.2f}, rms {fs['rms']:.3f}")
        for p in c:
            print(f"  ceiling {p['D_cm']:4.0f}cm {p['z_over_R']:5.2f} {p['ratio']:6.3f} "
                  f"[{p['ratio_range'][0]:.3f},{p['ratio_range'][1]:.3f}] "
                  f"{fmt(p['ratio_tach1'])} {fmt(p['ratio_tach2'])} gap {fmt(p['tach_gap'])}")
    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
