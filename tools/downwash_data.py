"""
Inter-vehicle downwash on a Crazyflie from public measurements — the evidence behind
finding F-38.

    uv run python tools/downwash_data.py --data /path/to/cache [--write]

Data (fetched into --data when missing, checked by sha256, not redistributed):
  - Kiran, Ayanian, Breuer, "Influence of Static and Dynamic Downwash Interactions on
    Multi-Quadrotor Systems", RSS 2025, doi 10.15607/RSS.2025.XXI.102. Data: Hugging Face
    ajkiran18/QuadrotorDownwash (MIT), RSS_2025_data.zip at revision c7bb7a2. Members are read
    from the zip in memory; nothing is extracted to disk.

Setup (read from the paper, Sec. II-A and III-A): two Crazyflies, each on an ATI Nano-17.
The lower one (B) is fixed; the upper one (A) moves on an x–z stage, 0.1–0.6 m across and
0.1–1.1 m down. Both run at a fixed hover bit-PWM command on a 4 V tether, about 320 Hz;
the paper states the voltage is held, NOT the rotor speed. W ≈ 0.265 N. l = 32.5 mm (body
centre to motor) normalises every length. Force maps: mean F_z/W of each vehicle, three
trials of 30 s. PIV: time-averaged (u, v)/U_i under ONE hovering Crazyflie, in a vertical
plane through the two front rotors (X configuration: 0.707·l off the vehicle axis).

Checks:
  1. the lower vehicle's thrust loss map against spec.ground_effect.pybullet_downwash_force
     in its stated range (Δz ≥ 0.7 m): centreline magnitude, lateral width, lateral integral;
  2. the single-vehicle wake against spec.ground_effect.jain_wake_velocity: centreline
     B/(z − z0), Gaussian width, and the momentum flux the mean wake carries;
  3. the loss against the PIV wake speed at the lower rotors through the inflow law
     (spec.inflow.inflow_thrust_ratio with momentum-theory induced velocity: the small-speed
     loss is s·X/(2 + s), X = wake speed / v_h) — an estimate of s, at constant voltage.

--write freezes the results in golden/checkdata/downwash_maps.json, which
properties/test_wake.py checks.
"""

import argparse
import hashlib
import io
import json
import math
import pathlib
import sys
import urllib.request
import zipfile

import numpy as np
import scipy.io as sio
import sympy as sp
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skyflow_dynamics.spec import ground_effect

OUT = ROOT / "golden" / "checkdata" / "downwash_maps.json"
HF_REV = "c7bb7a2454a1590806b634778d345b7cb031e45e"
ZIP_URL = ("https://huggingface.co/datasets/ajkiran18/QuadrotorDownwash/resolve/"
           f"{HF_REV}/RSS_2025_data.zip")
SHA256 = {"RSS_2025_data.zip": "b495b7a224b3c1db9f15efd5a15815c586e4e11ea29d4df6edcbc25bf50554fa"}

# --- constants: read from the paper unless marked ---
L_ARM = 0.0325          # m, body centre to motor (paper Sec. III-A; data README)
W = 0.265               # N, Crazyflie weight (paper Sec. III-A, "≈ 0.265 N")
ROTOR_HZ = 320.0        # paper Sec. II-A, "approximately 320 Hz"
R_PROP = 0.0231348      # m, cf2x.urdf (gym-pybullet-drones) — not stated in the paper
RHO = 1.204             # kg/m³, not stated in the paper (20 °C sea level assumed)
PYBULLET = {"c1": 2267.18, "c2": 0.16, "c3": -0.11, "R": R_PROP}
JAIN_PUBLISHED = {"c_ax": [4.7, 4.9], "c_rad": [24.9, 60.8], "z0_m": [0.033, 0.110]}
PAPER_TABLE_I = {"Bd": 9.508, "S": 0.0667, "z0_l": -6.0585}
#: v_h of one rotor at hover (momentum theory, paper Eq. (1) with n = 4).
V_H = math.sqrt(W / (2 * RHO * 4 * math.pi * R_PROP**2))
#: The paper's Re example (Eq. (8)) puts u_c = 4.25 m/s at z/l = 6.5, where the data give
#: u_c ≈ 0.52 U_i: U_i ≈ 8.2 m/s ≈ 2·V_H, i.e. Eq. (1) evaluated with n = 1. Which value
#: normalised the released field is not stated; both are carried.
U_I_CASES = {"eq1_n4": V_H, "re_example": 4.25 / 0.52}
VALID_DZ = 0.7          # m, pybullet fit's stated lower limit (β ≤ 0 below 0.6875 m)
JET_Z = (6.5, 23.8)     # z/l: merged-jet region (paper: z/l > 6.5) to the end of the field


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(data: pathlib.Path) -> pathlib.Path:
    data.mkdir(parents=True, exist_ok=True)
    path = data / "RSS_2025_data.zip"
    if not path.exists():
        req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "skyflow-dynamics"})
        path.write_bytes(urllib.request.urlopen(req, timeout=300).read())
    if sha256(path) != SHA256[path.name]:
        raise SystemExit(f"{path.name}: sha256 mismatch — not the pinned version")
    return path


def load_static(zpath: pathlib.Path) -> dict:
    """Every static .mat variable, read from the zip in memory, keyed by variable name."""
    out = {}
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            base = name.rsplit("/", 1)[-1]
            if "/static/" not in f"/{name}" or not base.endswith(".mat") or base.startswith("slanCM"):
                continue
            m = sio.loadmat(io.BytesIO(zf.read(name)))
            out.update({k: v for k, v in m.items() if not k.startswith("__")})
    return out


# ---------------------------------------------------------------------------------------
# 1. Force map against the pybullet fit
# ---------------------------------------------------------------------------------------

def _pybullet():
    dz, dxy = sp.symbols("dz dxy", positive=True)
    p = PYBULLET
    F = ground_effect.pybullet_downwash_force(dz, dxy, p["R"], p["c1"], p["c2"], p["c3"])
    return sp.lambdify((dz, dxy), F, "numpy")


def _half_point(x, y):
    """First x where y falls to half of y[0] (linear interpolation); None if it never does."""
    h = y[0] / 2
    for i in range(1, len(x)):
        if y[i] <= h:
            return float(x[i - 1] + (h - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return None


def force_rows(d: dict) -> list[dict]:
    x_l, z_l, F, Fs = d["x_l"], d["z_l"], d["lower_F_W"], d["lower_Fstd_W"]
    Fu = d["upper_F_W"]
    fn = _pybullet()
    rows = []
    for i in range(F.shape[0]):
        x = x_l[i] * L_ARM
        dz = float(z_l[i, 0] * L_ARM)
        base = float(F[i, -1])                      # Δx = 0.6 m: outside the wake
        loss = base - F[i]                          # fraction of W
        r = {"dz_l": float(z_l[i, 0]), "dz_m": dz, "baseline": base,
             "loss_centre": float(loss[0]), "std_centre": float(Fs[i, 0]),
             "upper_loss_centre": float(Fu[i, -1] - Fu[i, 0]),
             "loss_profile": [float(v) for v in loss], "x_m": [float(v) for v in x],
             "half_loss_x_m": _half_point(x, loss)}
        if dz >= VALID_DZ - 1e-9:
            beta = PYBULLET["c2"] * dz + PYBULLET["c3"]
            pb0 = -float(fn(dz, 0.0)) / W
            meas_int = float(np.trapezoid(loss * W * 2 * np.pi * x, x))     # N·m²
            r["pybullet"] = {"loss_centre": pb0, "ratio_centre": pb0 / r["loss_centre"],
                             "beta_m": beta, "half_max_x_m": beta * math.sqrt(2 * math.log(2)),
                             "integral_Nm2": pb0 * W * 2 * math.pi * beta**2,
                             "measured_integral_Nm2": meas_int}
        rows.append(r)
    return rows


def near_block_check(rows: list[dict]) -> dict:
    """The measured centreline loss drops 3.5× between Δz = 0.60 and 0.70 m (z/l 18.5 →
    21.5), while the single-vehicle wake speed at the lower rotors drops 12 % (PIV). The
    step is not explained in the paper. Robustness: extend the near block's own trend
    (log-linear in Δz over 0.35–0.60 m) to 0.7–1.1 m and compare the pybullet fit with that
    too; and evaluate the fit at 0.5–0.6 m, just outside its stated range."""
    fn = _pybullet()
    near = [r for r in rows if 0.34 <= r["dz_m"] <= 0.61]
    a, b = np.polyfit([r["dz_m"] for r in near], np.log([r["loss_centre"] for r in near]), 1)
    out = {"loglinear_slope_per_m": float(a), "extrapolated": [], "outside_range": []}
    for r in rows:
        pb0 = -float(fn(r["dz_m"], 0.0)) / W
        if r["dz_m"] >= VALID_DZ - 1e-9:
            ext = float(math.exp(a * r["dz_m"] + b))
            out["extrapolated"].append({"dz_m": r["dz_m"], "loss_extrapolated": ext,
                                        "pybullet_ratio": pb0 / ext})
        elif r["dz_m"] >= 0.49:
            out["outside_range"].append({"dz_m": r["dz_m"], "pybullet_loss": pb0,
                                         "ratio": pb0 / r["loss_centre"]})
    return out


# ---------------------------------------------------------------------------------------
# 2. Single-vehicle wake against the turbulent-jet model
# ---------------------------------------------------------------------------------------

def piv(d: dict):
    """PIV arrays are stored x-major; return (x/l row, z/l column, u/U_i[z, x])."""
    return d["x_piv"][:, 0], d["z_piv"][0, :], d["u_piv"].T


def jet_fit(d: dict) -> dict:
    """Fit u/U_i = B/(z−z0)·exp(−c·(x−x_c)² /(z−z0)²) to every slice in JET_Z, all in units
    of l, with one (B, z0, c) and a per-fit plane-centre offset x_c. Also the per-slice
    half-width and centreline, for the record."""
    xs, zs, u = piv(d)
    sel = (zs >= JET_Z[0]) & (zs <= JET_Z[1])
    Z, X = np.meshgrid(zs[sel], xs, indexing="ij")
    U = u[sel]

    def model(p, X, Z):
        B, z0, c, xc = p
        zeta = Z - z0
        return B / zeta * np.exp(-c * ((X - xc) / zeta) ** 2)

    fit = least_squares(lambda p: (model(p, X, Z) - U).ravel(), [10.0, -5.0, 60.0, 0.0],
                        bounds=([0.1, -40, 1, -2], [100, 6.4, 1e4, 2]))
    B, z0, c, xc = (float(v) for v in fit.x)
    rms = float(np.sqrt(np.mean(fit.fun ** 2)))
    slices = []
    for zz in (6.5, 8, 10, 12.5, 15, 17.5, 20, 21.5, 23.5):
        i = int(np.argmin(abs(zs - zz)))
        row = u[i]
        half = xs[row >= row.max() / 2]
        slices.append({"z_l": float(zs[i]), "u_max": float(row.max()),
                       "u_axis": float(row[np.argmin(abs(xs))]),
                       "half_width_l": float((half.max() - half.min()) / 2),
                       "model_u_max": float(B / (zs[i] - z0))})
    # The plane is 0.707·l off the axis: the in-plane peak understates the axis speed by
    # exp(−c·0.5/ζ²). Jain's form uses L = 2·l and per-rotor v_i: c_ax = B·(U_i/v_h)/2.
    out = {"B_l": B, "z0_l": z0, "z0_m": z0 * L_ARM, "c_rad": c, "x_c_l": xc, "rms": rms,
           "S_half_width": math.sqrt(math.log(2) / c), "slices": slices,
           "c_ax": {k: B * (U / V_H) / 2 for k, U in U_I_CASES.items()}}
    # Momentum flux of the mean wake (axisymmetric Gaussian, axis speed corrected for the
    # plane offset) over the vehicle's thrust W = 4·2ρA·v_h².
    mom = {}
    for k, U in U_I_CASES.items():
        ratios = []
        for zz in (8.0, 15.0, 22.0):
            zeta = zz - z0
            uc = B / zeta * math.exp(c * 0.5 / zeta**2) * U          # m/s on the axis
            J = RHO * math.pi * uc**2 * (zeta * L_ARM) ** 2 / (2 * c)
            ratios.append(J / W)
        mom[k] = ratios
    out["momentum_over_W_at_z_l_8_15_22"] = mom
    # Centreline only (peak of each slice), virtual origin held at Jain's published z0
    # (mid-range 0.07 m below the rotor plane): least-squares B and the misfit, against the
    # free fit on the same points.
    zz = zs[sel]
    um = u[sel].max(axis=1)
    out["centreline_fits"] = {}
    for name, z0c in (("free", z0), ("z0_jain_0.07m", 0.07 / L_ARM)):
        g = 1 / (zz - z0c)
        Bc = float(g @ um / (g @ g))
        out["centreline_fits"][name] = {
            "z0_l": float(z0c), "B_l": Bc,
            "rms": float(np.sqrt(np.mean((Bc * g - um) ** 2))),
            "max_rel_err": float(np.max(abs(Bc * g - um) / um)),
            "c_ax": {k: Bc * (U / V_H) / 2 for k, U in U_I_CASES.items()}}
    return out


# ---------------------------------------------------------------------------------------
# 3. Thrust loss against the wake speed at the lower rotors
# ---------------------------------------------------------------------------------------

def inflow_sensitivity(d: dict, rows: list[dict]) -> list[dict]:
    """Stacked case. The lower vehicle's front rotors sit at x = ±0.707·l in the PIV plane;
    average u over each disk's chord (±R) at z = Δz. X = that speed / v_h. With momentum-
    theory induced velocity the small-X loss is s·X/(2+s), so s = 2·loss/(X − loss).
    Constant voltage, not constant speed: the rotor speeds up a little as the load drops, so
    this s is a lower bound on the constant-speed s."""
    xs, zs, u = piv(d)
    rp = R_PROP / L_ARM
    chord = (abs(abs(xs) - 0.707) <= rp)
    out = []
    for r in rows:
        if r["dz_l"] > zs.max():
            continue
        i = int(np.argmin(abs(zs - r["dz_l"])))
        ubar = float(u[i, chord].mean())
        e = {"dz_l": r["dz_l"], "dz_m": r["dz_m"], "u_rotor_over_Ui": ubar,
             "loss": r["loss_centre"]}
        for k, U in U_I_CASES.items():
            X = ubar * U / V_H
            L = r["loss_centre"]
            e[k] = {"X": X, "s": 2 * L / (X - L) if X > L else None}
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True, help="cache for the sources")
    ap.add_argument("--write", action="store_true", help=f"freeze the result in {OUT.name}")
    args = ap.parse_args(argv)
    d = load_static(fetch(args.data))

    rows = force_rows(d)
    print(f"v_h (Eq. 1, n = 4) = {V_H:.2f} m/s; U_i from the paper's Re example = "
          f"{U_I_CASES['re_example']:.2f} m/s")
    print(f"\n{'dz m':>5s} {'dz/l':>5s} {'base':>5s} {'loss':>6s} {'std':>5s} {'half-loss x':>11s} | "
          f"{'pybullet':>8s} {'ratio':>6s} {'β m':>6s} {'∫ pb/meas':>9s}")
    for r in rows:
        pb = r.get("pybullet")
        hx = f"{r['half_loss_x_m']:.3f}" if r["half_loss_x_m"] is not None else "  —"
        tail = (f"{pb['loss_centre']:8.3f} {pb['ratio_centre']:6.1f} {pb['beta_m']:6.3f} "
                f"{pb['integral_Nm2'] / pb['measured_integral_Nm2']:9.2f}") if pb else ""
        print(f"{r['dz_m']:5.2f} {r['dz_l']:5.1f} {r['baseline']:5.3f} {r['loss_centre']:6.3f} "
              f"{r['std_centre']:5.3f} {hx:>11s} | {tail}")

    near = near_block_check(rows)
    print(f"\nnear-block trend (0.35–0.60 m): loss ∝ exp({near['loglinear_slope_per_m']:.2f}·Δz)")
    for e in near["extrapolated"]:
        print(f"  {e['dz_m']:.2f} m  extrapolated loss {e['loss_extrapolated']:.3f}  "
              f"pybullet ratio {e['pybullet_ratio']:.1f}")
    for e in near["outside_range"]:
        print(f"  {e['dz_m']:.2f} m (outside the fit's range)  pybullet {e['pybullet_loss']:.3f}"
              f"  ratio {e['ratio']:.1f}")

    jet = jet_fit(d)
    print(f"\njet fit z/l {JET_Z}: B = {jet['B_l']:.2f} l, z0 = {jet['z0_l']:.2f} l "
          f"({jet['z0_m']:.3f} m), c_rad = {jet['c_rad']:.1f} (S = {jet['S_half_width']:.3f}), "
          f"rms {jet['rms']:.4f}")
    print("  c_ax (L = 2l): " + ", ".join(f"{k} {v:.2f}" for k, v in jet["c_ax"].items())
          + f"   published {JAIN_PUBLISHED}")
    print("  mean-wake momentum / W at z/l 8, 15, 22: " + "; ".join(
        f"{k} " + " ".join(f"{v:.2f}" for v in vs)
        for k, vs in jet["momentum_over_W_at_z_l_8_15_22"].items()))
    for name, cf in jet["centreline_fits"].items():
        print(f"  centreline {name:14s} z0 {cf['z0_l']:6.2f} l  B {cf['B_l']:6.2f} l  rms {cf['rms']:.4f}"
              f"  max err {cf['max_rel_err']:.0%}  c_ax " + ", ".join(f"{k} {v:.2f}" for k, v in cf["c_ax"].items()))
    for s in jet["slices"]:
        print(f"  z/l {s['z_l']:5.2f}  u_max {s['u_max']:.3f} (model {s['model_u_max']:.3f})"
              f"  half-width {s['half_width_l']:.2f} l")

    sens = inflow_sensitivity(d, rows)
    print("\nloss vs wake at the lower rotors (stacked): dz/l  u/U_i  loss  | X, s per U_i case")
    for e in sens:
        print(f"  {e['dz_l']:5.1f} {e['u_rotor_over_Ui']:.3f} {e['loss']:.3f} | " + "  ".join(
            f"{k}: X {e[k]['X']:.2f} s {e[k]['s'] if e[k]['s'] is None else round(e[k]['s'], 2)}"
            for k in U_I_CASES))

    result = {"kind": "downwash_maps",
              "provenance": {"paper": "Kiran, Ayanian, Breuer, RSS 2025, doi 10.15607/RSS.2025.XXI.102",
                             "data_url": ZIP_URL, "sha256": SHA256,
                             "generator": "tools/downwash_data.py"},
              "constants": {"l_m": L_ARM, "W_N": W, "rotor_hz": ROTOR_HZ, "R_prop_m": R_PROP,
                            "rho": RHO, "v_h": V_H, "U_i_cases": U_I_CASES,
                            "pybullet": PYBULLET, "jain_published": JAIN_PUBLISHED,
                            "paper_table_I": PAPER_TABLE_I},
              "force_rows": rows, "near_block": near, "jet": jet, "inflow_sensitivity": sens}
    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
