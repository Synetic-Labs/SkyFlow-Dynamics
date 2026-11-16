"""
Constant-speed thrust in axial descent from public tabulated data — the evidence behind
finding F-35 — and the near-axial descent envelope of the NeuroBEM flight data.

    uv run python tools/descent_data.py --data /path/to/cache [--neurobem /path/to/processed_data] [--write]

Data (fetched into --data when missing, checked by sha256, not redistributed):
  - Shetty, "Measurements of Small-Scale Propellers Operating in the Vortex Ring State",
    M.S. thesis, UIUC 2011 (hdl 2142/18400). Appendix B tabulates C_T(J) at about 4000 rpm
    for 24 APC propellers from J ≈ −0.13 to −0.8; Appendix A tabulates chord and twist.
  - The UIUC Propeller Data Site (same lab and tunnel), Volume 1: static C_T(rpm) of the same
    propellers — the hover normaliser. 19 of the 24 have a static file.
Text is extracted with poppler's `pdftotext -raw`.

x = descent rate / v_h0 = −J/√(2·C_T0/π) (propeller C_T = T/(ρn²D⁴)); y = C_T(J)/C_T0.

Models compared on each curve, over the VRS band x ≤ 1.75 (beyond it the blade stalls):
  - backend: T/T_static = 1 + a·x (the k_z channel), slope a free per propeller;
  - Johnson + the linear inflow thrust law (spec.descent, spec.inflow) with Table 4 and the
    blade-element s from the measured geometry;
  - the same with three parameters: s, k (scales Johnson's VRS points V_zD, V_zN, V_zX, V_zE)
    and peak = (V+v)_X.
Held-out test (leave one propeller out): (k, peak) = the median of the other propellers'
fits, s fitted on the held-out curve — one free parameter per propeller, like the backend.
Cross-lab: the UIUC (k, peak) applied to Veismann's 4–6 in FPV rotors (tools/wake_triage.py).

--write freezes the results in golden/checkdata/descent_curves.json, which
properties/test_wake.py checks.
"""

import argparse
import glob
import hashlib
import itertools
import json
import math
import pathlib
import re
import subprocess
import sys
import urllib.request
import warnings

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skyflow_dynamics.spec import descent
from tools import wake_triage as wt

OUT = ROOT / "golden" / "checkdata" / "descent_curves.json"
THESIS_URL = "https://www.ideals.illinois.edu/items/18490/bitstreams/66576/data"
UIUC_URL = "https://m-selig.ae.illinois.edu/props/volume-1/data/"
SHA256 = {
    "shetty_thesis.pdf": "1f197023b630f4df731251f9b17a96f5bddbe44acab2537a6274a79cb9b36f9c",
    "apce_10x5_static_pg0819.txt": "9d4f0434c5e31f6c53082e8551426d2dba40def4a60ee5aca5eeee60a514f2dd",
    "apce_9x4.5_static_rd0995.txt": "f700c8d9a77fb3025e727ff97de88a5a414610f0ae56df063ce0f43ce62858d9",
    "apce_9x6_static_rd0987.txt": "03e5466a2dced023cceb78c93628e4950d804cb05bc426822d856c79480e17d2",
    "apce_9x7.5_static_2545rd.txt": "f8ab0339a0f3d6023ccc7ed00f09ed0c1169408ec72e8c8af927309386cfbb40",
    "apcsf_10x4.7_static_kt0835.txt": "20519deb7e9678b0674005c38fdb8a23e916800bf5fc1b5714612f3f4c13d9a0",
    "apcsf_11x3.8_static_kt0542.txt": "7d52d517dc42d38fe32f04f7108aedd20ec4c1891b8e7caa56f7eab8975fda33",
    "apcsf_11x4.7_static_pg0526.txt": "ef65895140092ffed2e5064b5b5058d267ed0b3563c747bf067ed80edb3198fa",
    "apcsf_9x3.8_static_2553rd.txt": "b5758cda62b5cbcc387b03874e0deef30413efb035721683e9313c5fd6cd8ba5",
    "apcsf_9x4.7_static_kt1032.txt": "3745a63634d65a830f5ca5860288e7eb152c0939511ed5108d21d0c6aa787f61",
    "apcsf_9x6_static_kt0979.txt": "28691ef497f67bfd3df72825b66c6f1cf9f1541e48a5017332e8ec75a7fed982",
    "apcsf_9x7.5_static_2567rd.txt": "01f28ec7fee4e1ec15f1b3f856dfda4b9bceb9a92788fcc722ec190eeb46fdf1",
    "apcsp_10x10_static_2665rd.txt": "285059bf457ba07a47c96f1830fabfe8f413e037a30776825b01cab780f90f31",
    "apcsp_10x4_static_2644rd.txt": "661674cf0a7c9f45d10fd03a38f0be89f24c1eedd13ae91fe7a58b5f7b442e7a",
    "apcsp_10x5_static_2651rd.txt": "da4ee0ad7179471e3ea5227baef4af47614b68f1f2ab1adace2bf15be764b3c3",
    "apcsp_10x7_static_2654rd.txt": "2cae7ec86100f1784f1ab448e7de9f1cf0c1fe29517708a45ebc5991cb6ae797",
    "apcsp_11x5_static_rd0496.txt": "e555dec99e6c710dffa01c1233dec586cdcebc9d0f2da367d669f11f76df8eff",
    "apcsp_9x10_static_2513rd.txt": "611831d75ec1c7611d07a7d0d8a6ee80cf2d0d58d66a66cebd314e3613de068b",
    "apcsp_9x7_static_kt1009.txt": "9058c9bfabdf37b3379e8991a9e4900d6d8053a611ca1a531a634a086eec5822",
    "apcsp_9x8_static_2530rd.txt": "43410a9550ff4a2c6398893ad2c583f7c278cd4d69b279c4a8cae8101f10b398",
}
PREFIX = {"Slow Flyer": "apcsf", "Sport Propeller": "apcsp", "Thin Electric": "apce"}
#: Excluded: the 11×3.8's descent C_T sits above its static C_T at every J (ratio 1.07–2.5),
#: so the two tables are not from comparable conditions.
EXCLUDE = {("Slow Flyer", "11x3.8")}
BAND = 1.75


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(data: pathlib.Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    for name, digest in SHA256.items():
        path = data / name
        if not path.exists():
            url = THESIS_URL if name.endswith(".pdf") else UIUC_URL + name
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; skyflow-dynamics)"})
            path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        if sha256(path) != digest:
            raise SystemExit(f"{name}: sha256 mismatch — not the pinned version")


_FAMILY = r"(Slow Flyer|Sport Propeller|Sport Prop|Thin Electric)\n(\S+)\n"


def _family(name: str) -> str:
    return "Sport Propeller" if name.startswith("Sport") else name


def parse_thesis(pdf: pathlib.Path) -> tuple[dict, dict]:
    """(geometry, tables) keyed by (family, 'DxP'). Geometry: θ.75 (deg) and solidity of a
    two-blade propeller from Appendix A; tables: rpm, J, C_T from Appendix B."""
    raw = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], check=True,
                         capture_output=True, text=True).stdout
    a0 = raw.find("Tabulated Geometry\nIn this appendix")
    b0 = raw.rfind("Tabulated Performance Data")
    geometry = {}
    hdr = list(re.finditer(_FAMILY + r"r/R c/R β\n", raw[a0:b0]))
    for i, m in enumerate(hdr):
        seg = raw[a0:b0][m.end(): hdr[i + 1].start() if i + 1 < len(hdr) else None]
        rows = re.findall(r"^(\d\.\d+) (\d\.\d+) (\d+\.\d+)$", seg, re.MULTILINE)
        r, c, b = (np.array(v, float) for v in zip(*rows))
        geometry[(_family(m.group(1)), m.group(2).replace("×", "x"))] = {
            "theta75_deg": float(np.interp(0.75, r, b)),
            "solidity": float(2 * np.trapezoid(c, r) / math.pi)}
    tables = {}
    body = raw[b0:]
    hdr = list(re.finditer(_FAMILY + r"Run: (\S+)\nAverage RPM: (\d+)\nJ CT\n", body))
    # One header (Sport 9×5) is printed as "Average RPM: 4020)" and is caught below.
    for i, m in enumerate(hdr):
        seg = body[m.end(): hdr[i + 1].start() if i + 1 < len(hdr) else None]
        pts = []
        for j, ct in re.findall(r"^(-?\d\.\d+) (\d\.\d+)$", seg, re.MULTILINE):
            if (float(j), float(ct)) not in pts:
                pts.append((float(j), float(ct)))
        key = (_family(m.group(1)), m.group(2).replace("×", "x"))
        if key == ("Sport Propeller", "9x4") and len(pts) > 17:
            tables[("Sport Propeller", "9x5")] = {"rpm": 4020.0, "pts": pts[17:]}
            pts = pts[:17]
        tables[key] = {"rpm": float(m.group(4)), "pts": pts}
    return geometry, tables


def curves(data: pathlib.Path) -> list[dict]:
    geometry, tables = parse_thesis(data / "shetty_thesis.pdf")
    out = []
    for (family, size), tab in tables.items():
        static = glob.glob(str(data / f"{PREFIX[family]}_{size}_static_*.txt"))
        if (family, size) in EXCLUDE or not static:
            continue
        rpm, ct = np.loadtxt(static[0], skiprows=1, usecols=(0, 1), unpack=True)
        ct0 = float(np.interp(tab["rpm"], rpm, ct))
        scale = math.sqrt(2 * ct0 / math.pi)
        # C_T > 0.03 drops one misprint (Slow Flyer 10×4.7, J = −0.735: 0.0122)
        curve = [(-j / scale, c / ct0) for j, c in tab["pts"] if c > 0.03]
        geo = geometry.get((family, size))
        if geo is None:     # four Sport propellers: no Appendix A table parsed
            d, p = map(float, size.split("x"))
            geo = {"theta75_deg": math.degrees(math.atan(p / (0.75 * math.pi * d))),
                   "solidity": None}
        out.append({"name": f"APC {family} {size}", "rpm": tab["rpm"], "ct0": ct0,
                    "static_file": pathlib.Path(static[0]).name, **geo,
                    "ct_hover": ct0 * 4 / math.pi ** 3, "curve": curve})
    return out


# ---------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------

_U = np.linspace(0.3, 1.8, 3001)


def _inflow():
    vz, peak, k = sp.symbols("vz peak k", real=True)
    c = dict(descent.JOHNSON_VRS)
    c.update({"VvX": peak, **{n: c[n] * k for n in ("VzD", "VzN", "VzX", "VzE")}})
    return sp.lambdify((vz, peak, k), descent.johnson_vrs_inflow_axial(vz, c=c), "numpy")


_F = _inflow()


def johnson_ratio(xs, s: float, k: float = 1.0, peak: float = 1.25) -> np.ndarray:
    """T/T_static at fixed rotor speed (the lowest root u² of wake_triage's equation,
    linear interpolation on a 0.0005 grid in u): Johnson's inflow with its VRS points
    scaled by k and (V+v)_X = peak, through spec.inflow.inflow_thrust_ratio."""
    xs = np.asarray(xs, float)[:, None]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore")
        v = _F(-xs / _U[None, :], peak, k)
    g = _U ** 2 - (1 - s * (-xs + _U * v - 1))
    change = g[:, :-1] * g[:, 1:] <= 0
    i = np.argmax(change, axis=1)
    rows = np.arange(len(i))
    g0, g1 = g[rows, i], g[rows, i + 1]
    u = _U[i] + (_U[i + 1] - _U[i]) * g0 / (g0 - g1)
    return np.where(change[rows, i], u ** 2, np.nan)


def _band(curve):
    xs, ys = map(np.array, zip(*[(x, y) for x, y in curve if x <= BAND]))
    return xs, ys


def _rms(r) -> float:
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def fit_three(curve, s0: float) -> dict:
    xs, ys = _band(curve)

    def res(p):
        return np.nan_to_num(johnson_ratio(xs, *p) - ys, nan=1.0)

    best = None
    for k0 in (0.7, 0.85, 1.0):
        for p0 in (1.3, 1.6):
            r = least_squares(res, [min(max(s0, 0.2), 3.0), k0, p0], diff_step=1e-3,
                              bounds=([0.05, 0.4, 0.9], [4.0, 1.4, 3.0]))
            best = r if best is None or r.cost < best.cost else best
    s, k, peak = (float(v) for v in best.x)
    return {"s": s, "k": k, "peak": peak, "rms": _rms(res(best.x)),
            "model_min": model_minimum(s, k, peak)}


def fit_s(curve, k: float, peak: float) -> tuple[float, float]:
    xs, ys = _band(curve)
    r = least_squares(lambda p: np.nan_to_num(johnson_ratio(xs, p[0], k, peak) - ys, nan=1.0),
                      [0.5], bounds=([0.05], [4.0]), diff_step=1e-3)
    return float(r.x[0]), _rms(r.fun)


def fit_backend(curve) -> tuple[float, float]:
    """Best slope a of T/T_static = 1 + a·x (least squares, any sign)."""
    xs, ys = _band(curve)
    a = float(np.sum(xs * (ys - 1)) / np.sum(xs * xs))
    return a, _rms(1 + a * xs - ys)


def model_minimum(s: float, k: float, peak: float) -> tuple[float, float]:
    xx = np.arange(0.4, 2.0, 0.01)
    yy = johnson_ratio(xx, s, k, peak)
    j = int(np.nanargmin(yy))
    return float(xx[j]), float(yy[j])


# ---------------------------------------------------------------------------------------
# NeuroBEM: thrust per Ω² against the axial descent ratio
# ---------------------------------------------------------------------------------------

def neurobem_descent(processed: pathlib.Path) -> list[dict]:
    """Bins of x (descent along the thrust axis / v_h at the current rotor speeds), samples
    with in-plane speed < 0.3·v_h and |dΩ/dt|/Ω < 5 1/s, at 100 Hz. m·a_z/(c_T·ΣΩ²) median
    with a 95 % bootstrap over segments. c_T, mass and radius: RACER_5IN's identification.
    The frame's vertical drag is NOT removed (it pushes up in descent; F-30 notes c_Dz and
    k_z are not separable here)."""
    ident = json.loads((ROOT / "golden" / "checkdata" / "neurobem_racer.json").read_text())
    c_t, mass, radius, rho = ident["force_fit"]["c_T"], 0.772, 0.0635, wt.RHO
    segs = []
    for f in sorted(processed.glob("*.csv")):
        a = np.loadtxt(f, delimiter=",", skiprows=1,
                       usecols=(13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27))[::4]
        w, dw, v = a[:, 4:8], a[:, 8:12], a[:, 1:4]
        thrust = c_t * (w ** 2).sum(1)
        v_h = np.sqrt(thrust / 4 / (2 * rho * math.pi * radius ** 2))
        keep = (np.hypot(v[:, 0], v[:, 1]) / v_h < 0.3) & (np.abs(dw).max(1) / w.mean(1) < 5)
        segs.append((-v[keep, 2] / v_h[keep], mass * a[keep, 0] / thrust[keep]))
    rng = np.random.default_rng(0)
    out = []
    edges = (-0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
    for lo, hi in itertools.pairwise(edges):
        per = [r[(x >= lo) & (x < hi)] for x, r in segs]
        per = [p for p in per if p.size]
        allr = np.concatenate(per)
        boot = [np.median(np.concatenate([per[i] for i in rng.integers(0, len(per), len(per))]))
                for _ in range(300)]
        out.append({"x": [lo, hi], "n": int(allr.size), "segments": len(per),
                    "ratio": float(np.median(allr)),
                    "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])]})
    return out


# ---------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True, help="cache for the sources")
    ap.add_argument("--neurobem", type=pathlib.Path, help="NeuroBEM processed_data folder")
    ap.add_argument("--write", action="store_true", help=f"freeze the result in {OUT.name}")
    args = ap.parse_args(argv)
    fetch(args.data)

    props = curves(args.data)
    print(f"{len(props)} UIUC propellers (Shetty 2011 App. B ÷ UIUC static C_T)")
    rows = []
    for p in props:
        s_be = wt.blade_element_sensitivity(p)
        xs, ys = _band(p["curve"])
        mx, my = min(p["curve"], key=lambda q: q[1])
        rows.append({"name": p["name"], "lab": "UIUC", "rpm": p["rpm"], "ct0": p["ct0"],
                     "theta75_deg": p["theta75_deg"], "s_blade_element": s_be,
                     "x_range": [p["curve"][0][0], p["curve"][-1][0]],
                     "measured_min": [mx, my], "points_in_band": int(xs.size),
                     "table4_rms": _rms(johnson_ratio(xs, s_be) - ys),
                     "table4_min": model_minimum(s_be, 1.0, 1.25),
                     "fit": fit_three(p["curve"], s_be), "_curve": p["curve"]})
    for name, r in wt.VEISMANN_2023.items():
        s_be = wt.blade_element_sensitivity(r)
        xs, ys = _band(r["curve"])
        mx, my = min(r["curve"], key=lambda q: q[1])
        rows.append({"name": name, "lab": "Caltech (figure read-off)",
                     "s_blade_element": s_be, "measured_min": [mx, my],
                     "points_in_band": int(xs.size),
                     "table4_rms": _rms(johnson_ratio(xs, s_be) - ys),
                     "table4_min": model_minimum(s_be, 1.0, 1.25),
                     "fit": fit_three(r["curve"], s_be), "_curve": r["curve"]})

    uiuc = [r for r in rows if r["lab"] == "UIUC"]
    for r in uiuc:      # leave one out: shared (k, peak) from the others
        others = [o["fit"] for o in uiuc if o is not r]
        k = float(np.median([o["k"] for o in others]))
        peak = float(np.median([o["peak"] for o in others]))
        s, rms = fit_s(r["_curve"], k, peak)
        a, rms_b = fit_backend(r["_curve"])
        r["held_out"] = {"k": k, "peak": peak, "s": s, "rms": rms,
                         "backend_slope": a, "backend_rms": rms_b}
    k_u = float(np.median([r["fit"]["k"] for r in uiuc]))
    p_u = float(np.median([r["fit"]["peak"] for r in uiuc]))
    for r in rows:
        if r["lab"] != "UIUC":
            s, rms = fit_s(r["_curve"], k_u, p_u)
            a, rms_b = fit_backend(r["_curve"])
            r["cross_lab"] = {"k": k_u, "peak": p_u, "s": s, "rms": rms,
                              "backend_slope": a, "backend_rms": rms_b}

    print(f"\n{'curve':30s} {'meas min@x':>11s} {'T4 rms':>6s} | {'s':>5s} {'k':>5s} "
          f"{'peak':>5s} {'rms':>6s} | held-out/cross-lab rms: {'Johnson':>7s} {'backend':>7s}")
    for r in rows:
        f, h = r["fit"], r.get("held_out") or r.get("cross_lab")
        print(f"{r['name']:30s} {r['measured_min'][1]:5.2f}@{r['measured_min'][0]:4.2f} "
              f"{r['table4_rms']:6.3f} | {f['s']:5.2f} {f['k']:5.2f} {f['peak']:5.2f} "
              f"{f['rms']:6.3f} | {'':24s}{h['rms']:7.3f} {h['backend_rms']:7.3f}")
    ho = np.array([[r["held_out"]["rms"], r["held_out"]["backend_rms"]] for r in uiuc])
    print(f"\nUIUC median fits: k {k_u:.2f}, peak {p_u:.2f}, s "
          f"{np.median([r['fit']['s'] for r in uiuc]):.2f}")
    print(f"held out (UIUC, n = {len(uiuc)}): Johnson rms median {np.median(ho[:, 0]):.3f}, "
          f"backend {np.median(ho[:, 1]):.3f}; Johnson better on {int((ho[:, 0] < ho[:, 1]).sum())}")

    result = {"kind": "descent_curves",
              "provenance": {"thesis": "Shetty, M.S. thesis, UIUC 2011, hdl 2142/18400",
                             "thesis_url": THESIS_URL, "static_url": UIUC_URL,
                             "sha256": SHA256, "generator": "tools/descent_data.py",
                             "band": BAND, "excluded": [" ".join(e) for e in EXCLUDE]},
              "uiuc_median": {"k": k_u, "peak": p_u},
              "curves": [{k: v for k, v in r.items() if k != "_curve"} for r in rows]}
    if args.neurobem:
        result["neurobem_near_axial"] = neurobem_descent(args.neurobem)
        print("\nNeuroBEM near-axial, slow rotor-speed change: m·a_z/(c_T·ΣΩ²)")
        for b in result["neurobem_near_axial"]:
            print(f"  x {b['x'][0]:+.1f}..{b['x'][1]:+.1f}  n {b['n']:6d}  segs {b['segments']:3d}"
                  f"  {b['ratio']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
