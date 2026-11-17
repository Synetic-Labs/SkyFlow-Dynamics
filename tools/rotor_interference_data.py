"""
Rotor-on-rotor thrust interference from the NASA Multirotor Test Bed 2 (MTB2) wind-tunnel
tables — the evidence behind finding F-37 — against spec.inflow.tandem_interference_factor
(rotor_wake_interference) and the linear inflow thrust law (inflow_thrust_law).

    uv run python tools/rotor_interference_data.py --data /path/to/cache [--write]

Data (fetched into --data when missing, checked by sha256, not redistributed):
  - Conley, Russell, Kallstrom, Shirazi, Pereyra, Wright — MTB2 data tables
    (mtbii_data_tables_v2.xlsx): 2,235 points, six loads per rotor (hub frame, tared),
    rotor speed, tunnel speed, density, model pitch, rotor tilt, rotor phase. NASA Ames,
    7- by 10-Foot Wind Tunnel, 2022 entry. Read with the standard library (zipfile + XML).
  - Geometry from the data report, NASA/TM-20250004795 (May 2025), read: two-blade rotors,
    R = 12.245 in (24.5 in diameter), E387 airfoil, −16° linear twist, 6.7° at 0.75R,
    solidity 0.08 (Tables 7.2.1–7.2.2). Appendix A, pitch 0: hubs at x = −16.60 in (front,
    R1 starboard, R2 port) and +16.40 in (back, R3 starboard, R4 port), y = +16.73 / −15.98 in;
    back rotors raised dz = 0, 4, 7 in ("equal height", "concept", "tall"; dz/R = 0, 0.33,
    0.57). X+ downstream; pitch and tilt negative nose down.

What is compared (same rotor hardware at the same position, matched tilt, model pitch,
tunnel-speed bin and rotor-speed set point; thrust as T/(ρ·rpm²) so small density and
speed differences cancel):
  - tandem (R1 + R3 in line, phase 0) against each rotor running alone;
  - quadrotor (all four, two blades, phase 0) against each rotor alone;
  - side by side (R1 + R2, concept only) against each alone.
The front rotor alone was run only in the concept layout (run 102); the front hub does not
move between layouts, so run 102 is its reference in all three (flagged in the output).

Spec prediction for the back rotor: k from tandem_interference_factor with the hub line
in place of the disk plane (p = along-stream hub spacing / R, tan α = height of the back hub
above the front hub across the stream / along-stream spacing, γ = angle between the front
wake and the hub line); the front wake from momentum theory with the measured front thrust;
ΔT = −s·k·v_front/v_h·T_static (the back rotor's own induced velocity frozen, as
tools/wake_triage.rear_rotor_loss). With dz = 0 this is Nguyen's coplanar geometry exactly.

--write freezes the results in golden/checkdata/rotor_interference.json, which
properties/test_wake.py checks.
"""

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skyflow_dynamics.spec import inflow

OUT = ROOT / "golden" / "checkdata" / "rotor_interference.json"
PAGE = "https://rotorcraft.arc.nasa.gov/Research/Programs/MTB2.html"
BASE = "https://rotorcraft.arc.nasa.gov/Research/Programs/"
REPORT_URL = "https://rotorcraft.arc.nasa.gov/Publications/files/MTB2_Data_Report_05222025.pdf"
SHA256 = {
    "mtbii_data_tables_v2.xlsx": "ab82069b7d3e1131d67233a44928b51019530ab395612d21755d9ec39141749c",
}
REPORT_SHA256 = "8c034a713441dba7f45353d4ed7f5ecbc5bd421916a5f85d76ebec6fb652febf"  # not fetched

IN, LB = 0.0254, 4.4482216152605
SLUG_FT3, FT = 515.378818, 0.3048
R = 12.245 * IN
AREA = math.pi * R ** 2
L_HUB = 33.00 * IN                      # front-to-back hub spacing along the body (App. A)
DZ = {"equal": 0.0, "concept": 4.0 * IN, "tall": 7.0 * IN}

#: Run → (layout, kind). Two-blade runs only; 3- and 6-blade quadrotor runs (128–135) and
#: tares, hover sweeps and transients (105, 107, 111, 120, 122 and 124 hover parts are
#: dropped by the speed bins) are left out. From MTB2_Test_Matrix_Configuration.xlsx.
RUNS = {
    102: ("concept", "single", 1), 101: ("concept", "single", 2),
    103: ("concept", "single", 3), 104: ("concept", "single", 4),
    141: ("equal", "single", 3), 140: ("equal", "single", 4),
    147: ("tall", "single", 3), 148: ("tall", "single", 4),
    113: ("concept", "tandem", None), 116: ("concept", "tandem", None),
    117: ("concept", "tandem", None), 119: ("concept", "tandem", None),
    143: ("equal", "tandem", None), 144: ("equal", "tandem", None),
    151: ("tall", "tandem", None),
    121: ("concept", "quad", None), 122: ("concept", "quad", None),
    123: ("concept", "quad", None), 124: ("concept", "quad", None),
    125: ("concept", "quad", None), 126: ("concept", "quad", None),
    127: ("concept", "quad", None),
    152: ("equal", "quad", None), 153: ("equal", "quad", None),
    149: ("tall", "quad", None), 150: ("tall", "quad", None),
    106: ("concept", "sbs", None), 108: ("concept", "sbs", None),
    109: ("concept", "sbs", None), 112: ("concept", "sbs", None),
}
ACTIVE = {"tandem": (1, 3), "quad": (1, 2, 3, 4), "sbs": (1, 2)}
SET_RPM = (2000, 2500, 3000)


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
            req = urllib.request.Request(BASE + name, headers={
                "User-Agent": "Mozilla/5.0 (compatible; skyflow-dynamics)"})
            path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        if sha256(path) != digest:
            raise SystemExit(f"{name}: sha256 mismatch — not the pinned version")


_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: pathlib.Path) -> list[list]:
    """First worksheet as rows of cell values (str or float), standard library only."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find("m:sheetData", _NS).findall("m:row", _NS):
        out = []
        for c in row.findall("m:c", _NS):
            i = _col(c.get("r"))
            v = c.find("m:v", _NS)
            if v is None:
                isv = c.find("m:is", _NS)
                val = "".join(t.text or "" for t in isv.iter(f"{{{_NS['m']}}}t")) if isv is not None else None
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            elif c.get("t") in ("str", "inlineStr"):
                val = v.text
            else:
                val = float(v.text)
            out += [None] * (i + 1 - len(out))
            out[i] = val
        rows.append(out)
    return rows


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _speed_bin(vfps: float) -> int | None:
    if vfps < 8:
        return 0
    if 15 <= vfps <= 30:
        return 20
    if vfps > 30:
        return 40
    return None


def points(xlsx: pathlib.Path) -> list[dict]:
    rows = read_xlsx(xlsx)
    head = next(i for i, r in enumerate(rows) if r and r[0] == "Run")
    cols = {name: j for j, name in enumerate(rows[head]) if name}
    out = []
    for r in rows[head + 1:]:
        run = _num(r[0]) if r else None
        if run is None or int(run) not in RUNS:
            continue
        g = lambda k: _num(r[cols[k]]) if cols[k] < len(r) else None
        layout, kind, single = RUNS[int(run)]
        rpm = max(g(f"RPM_{i}") or 0 for i in range(1, 5))
        rpm_set = min(SET_RPM, key=lambda s: abs(s - rpm))
        vbin = _speed_bin(g("VFPS"))
        if abs(rpm - rpm_set) > 30 or vbin is None:
            continue
        phases = [_num(r[cols[f"Phase_{i}"]]) for i in range(1, 5)]
        phase = next((p for p in phases if p not in (None, 0.0)), 0.0)
        rho = g("RHO") * SLUG_FT3
        active = (single,) if kind == "single" else ACTIVE[kind]
        for i in active:
            out.append({"run": int(run), "point": int(g("Point")), "layout": layout,
                        "kind": kind, "rotor": i, "phase": phase,
                        "tilt": round(g("Tilt")), "pitch": round(g("Pitch")),
                        "vbin": vbin, "V": g("VFPS") * FT, "rpm_set": rpm_set, "rpm": rpm,
                        "rho": rho, "T": g(f"Fz{i}_H") * LB, "Q": g(f"Mz{i}_H") * LB * IN,
                        "C": g(f"Fz{i}_H") * LB / (rho * rpm ** 2),
                        "CQ": g(f"Mz{i}_H") * LB * IN / (rho * rpm ** 2)})
    return out


# ---------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------

def _key(p):
    return (p["tilt"], p["pitch"], p["vbin"], p["rpm_set"])


def reference(pts: list[dict], field: str = "C") -> dict:
    """(layout, rotor) → {key: mean T/(ρ·rpm²)} (field "C"; "CQ" for torque) for rotors
    running alone. Front rotors have only the concept run; their hub is the same in every
    layout."""
    ref = {}
    for p in pts:
        if p["kind"] == "single":
            ref.setdefault((p["layout"], p["rotor"]), {}).setdefault(_key(p), []).append(p[field])
    ref = {k: {kk: float(np.mean(v)) for kk, v in d.items()} for k, d in ref.items()}
    for lay in ("equal", "tall"):
        ref[(lay, 1)] = ref[("concept", 1)]
        ref[(lay, 2)] = ref[("concept", 2)]
    return ref


def static_coefficient(pts: list[dict], rotor_layout=("concept", 1)) -> float:
    """T/(ρ·rpm²) with the tunnel off (V < 8 ft/s), single front rotor."""
    c = [p["C"] for p in pts if p["kind"] == "single" and (p["layout"], p["rotor"]) == rotor_layout
         and p["vbin"] == 0]
    return float(np.mean(c))


def front_inflow(T: float, rho: float, V: float, a_disk: float) -> float:
    """Momentum-theory induced velocity at incidence a_disk (nose down positive)."""
    vh = math.sqrt(T / (2 * rho * AREA))
    v = vh
    for _ in range(200):
        v = 0.5 * v + 0.5 * vh ** 2 / math.hypot(V * math.cos(a_disk), V * math.sin(a_disk) + v)
    return v


def spec_k(layout: str, pitch_deg: float, tilt_deg: float, V: float, v: float) -> dict:
    """Nguyen's factor with the hub line as reference (see module docstring)."""
    a_disk = math.radians(-(pitch_deg + tilt_deg))
    th = math.radians(-pitch_deg)
    dz = DZ[layout]
    along = L_HUB * math.cos(th) - dz * math.sin(th)          # downstream spacing
    across = L_HUB * math.sin(th) + dz * math.cos(th)         # back hub above front hub
    beta = math.atan2(across, along)
    delta = math.atan2(v * math.cos(a_disk), V + v * math.sin(a_disk))  # wake below stream
    gamma = beta + delta
    k = float(inflow.tandem_interference_factor(along / R, beta, gamma))
    return {"k": k, "p": along / R, "alpha_deg": math.degrees(beta),
            "gamma_deg": math.degrees(gamma), "wake_drop_R": (along * math.tan(delta) - across) / R}


def compare(pts: list[dict], ref: dict, refq: dict, kind: str, phase: float = 0.0) -> list[dict]:
    """Ratio of each rotor's C in `kind` to the same rotor alone, at matched key."""
    rows = []
    for p in pts:
        if p["kind"] != kind or p["phase"] != phase:
            continue
        r = ref.get((p["layout"], p["rotor"]), {}).get(_key(p))
        if r is None:
            continue
        rq = refq[(p["layout"], p["rotor"])][_key(p)]
        rows.append({**{k: p[k] for k in ("run", "point", "layout", "rotor", "tilt", "pitch",
                                          "vbin", "V", "rpm_set", "rho", "T", "Q")},
                     "ratio": p["C"] / r, "C_alone": r, "CQ_alone": rq})
    return rows


def _slope(x, y) -> float:
    A = np.vstack([x, np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def slope_compare(rows: list[dict]) -> list[dict]:
    """Offset-free comparison. The load cells carry run-to-run additive offsets of a few
    tenths of a pound (an idle rotor reads −0.2 to −0.9 lb whenever the motors run), so the
    point ratio mixes an offset into the interference. Fit T = a·ρ·rpm² + b over the three
    set points (2000/2500/3000) of one run and condition, and compare a with the same fit
    on the rotor alone; the same for torque. Prediction: the slope of T_alone·(1 + s·per_s)
    over the slope of T_alone, per unit s."""
    groups = {}
    for r in rows:
        groups.setdefault((r["run"], r["layout"], r["rotor"], r["tilt"], r["pitch"], r["vbin"]),
                          {})[r["rpm_set"]] = r
    out = []
    for key, g in groups.items():
        if sorted(g) != list(SET_RPM):
            continue
        rs = [g[k] for k in SET_RPM]
        x = np.array([r["rho"] * r["rpm_set"] ** 2 for r in rs])
        T = np.array([r["T"] for r in rs])
        Ta = np.array([r["C_alone"] for r in rs]) * x
        Q = np.array([r["Q"] for r in rs])
        Qa = np.array([r["CQ_alone"] for r in rs]) * x
        a, aa = _slope(x, T), _slope(x, Ta)
        row = dict(zip(("run", "layout", "rotor", "tilt", "pitch", "vbin"), key))
        row.update({"V": float(np.mean([r["V"] for r in rs])), "dT": a / aa - 1,
                    "dQ": _slope(x, Q) / _slope(x, Qa) - 1,
                    "dT_point_2500": g[2500]["ratio"] - 1})
        if all("per_s" in r for r in rs):
            row["per_s"] = _slope(x, Ta * np.array([r["per_s"] for r in rs])) / aa
            for f in ("k", "gamma_deg", "wake_drop_R", "p", "alpha_deg"):   # at 2500 rpm
                row[f] = g[2500][f]
        out.append(row)
    return out


def predict(rows: list[dict], ref: dict, c0: float) -> None:
    """Attach the spec's k and the s-free prediction −k·v_f/v_h·T_static/T_alone to each
    back-rotor row (thrust change per unit s)."""
    for r in rows:
        if r["rotor"] not in (3, 4):
            continue
        front = ref[(r["layout"], 1 if r["rotor"] == 3 else 2)].get(
            (r["tilt"], r["pitch"], r["vbin"], r["rpm_set"]))
        if front is None:
            continue
        rpm = r["rpm_set"]
        T_f = front * r["rho"] * rpm ** 2
        T_s = c0 * r["rho"] * rpm ** 2
        a_disk = math.radians(-(r["pitch"] + r["tilt"]))
        v_f = front_inflow(T_f, r["rho"], r["V"], a_disk)
        v_h = math.sqrt(T_s / (2 * r["rho"] * AREA))
        g = spec_k(r["layout"], r["pitch"], r["tilt"], r["V"], v_f)
        T_alone = r["C_alone"] * r["rho"] * rpm ** 2
        r.update(g)
        r["v_front"], r["v_h"] = v_f, v_h
        r["per_s"] = -g["k"] * v_f / v_h * T_s / T_alone      # predicted ΔT/T_alone ÷ s


def fit_s(rows: list[dict], field: str = "dT") -> dict:
    """Least-squares s for measured = s·per_s; with the rms of s, of s = 0.44 and of zero."""
    x = np.array([r["per_s"] for r in rows])
    y = np.array([r[field] for r in rows])
    s = float(np.dot(x, y) / np.dot(x, x))
    return {"s": s, "n": int(x.size), "rms": float(np.sqrt(np.mean((y - s * x) ** 2))),
            "rms_s044": float(np.sqrt(np.mean((y - 0.44 * x) ** 2))),
            "rms_zero": float(np.sqrt(np.mean(y ** 2)))}


def summarise(rows: list[dict], by: tuple, field: str = "dT") -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[b] for b in by), []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        v = np.array([r[field] for r in rs])
        row = {**dict(zip(by, k)), "n": int(v.size), "median": float(np.median(v)),
               "min": float(v.min()), "max": float(v.max()),
               "dQ_median": float(np.median([r["dQ"] for r in rs]))}
        if all("per_s" in r for r in rs):
            for f in ("per_s", "k", "gamma_deg", "wake_drop_R", "p", "alpha_deg"):
                row[f] = float(np.median([r[f] for r in rs]))
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().splitlines()[0])
    ap.add_argument("--data", type=pathlib.Path, required=True, help="cache for the sources")
    ap.add_argument("--write", action="store_true", help="freeze the result")
    args = ap.parse_args(argv)
    fetch(args.data)

    pts = points(args.data / "mtbii_data_tables_v2.xlsx")
    ref, refq = reference(pts), reference(pts, "CQ")
    c0 = static_coefficient(pts)
    print(f"{len(pts)} rotor-points; static T/(ρ·rpm²) front rotor {c0:.4e} N/(kg/m³·rpm²)")

    comps = {}
    for name, kind, phase in (("tandem", "tandem", 0.0), ("tandem_phase_m90", "tandem", -90.0),
                              ("quad", "quad", 0.0), ("sbs", "sbs", 0.0)):
        rows = compare(pts, ref, refq, kind, phase)
        predict(rows, ref, c0)
        comps[name] = {"points": rows, "slopes": slope_compare(rows)}

    # Repeatability and phase: the same tandem condition at blade phase 0 and −90 deg.
    sl = {}
    for name in ("tandem", "tandem_phase_m90"):
        for r in comps[name]["slopes"]:
            sl.setdefault((r["layout"], r["rotor"], r["tilt"], r["pitch"], r["vbin"]), {})[name] = r["dT"]
    rep = np.array([v["tandem"] - v["tandem_phase_m90"] for v in sl.values() if len(v) == 2])
    print(f"tandem phase 0 against −90 deg, same condition (slope ΔT/T difference): n {rep.size}, "
          f"median {np.median(rep):+.3f}, rms {np.sqrt(np.mean(rep ** 2)):.3f}")

    tables = {}
    print("\nOffset-free ΔT/T (slope over 2000–3000 rpm) against the same rotor alone; "
          "median [min, max]; torque ΔQ/Q median; point ratio at 2500 rpm for contrast")
    for name, c in comps.items():
        tables[name] = summarise(c["slopes"], ("layout", "rotor", "vbin"))
        pt = summarise([dict(r, dT=r["dT_point_2500"]) for r in c["slopes"]],
                       ("layout", "rotor", "vbin"))
        print(f"  {name}")
        for t, q in zip(tables[name], pt):
            t["point_2500_median"] = q["median"]
            print(f"    {t['layout']:8s} R{t['rotor']} V {t['vbin']:2d} ft/s n {t['n']:3d}  "
                  f"ΔT {t['median']:+.3f} [{t['min']:+.3f}, {t['max']:+.3f}]  ΔQ "
                  f"{t['dQ_median']:+.3f}   point {q['median']:+.3f}")

    back = [r for r in comps["tandem"]["slopes"] if r["rotor"] == 3 and "per_s" in r]
    by_pitch = summarise([r for r in back if r["tilt"] == 0], ("layout", "vbin", "pitch"))
    print("\nTandem back rotor R3, tilt 0: measured ΔT/T (slope), spec k, γ, wake centre "
          "below the back hub, spec prediction per unit s")
    for t in by_pitch:
        print(f"  {t['layout']:8s} V {t['vbin']:2d} pitch {t['pitch']:+3d}: {t['median']:+.3f} "
              f"(n {t['n']})  ΔQ {t['dQ_median']:+.3f}  k {t['k']:.2f}  γ {t['gamma_deg']:5.1f}°  "
              f"drop {t['wake_drop_R']:+.2f}R  spec/s {t['per_s']:+.3f}")

    fits = {"tandem_back_all": fit_s(back)}
    for lay in DZ:
        fits[f"tandem_back_{lay}"] = fit_s([r for r in back if r["layout"] == lay])
    #: Equal-height quadrotor runs 152/153: rotor 3 reads 42 % less thrust than alone in
    #: wind-off hover at unchanged torque, and twice the tandem loss at the same geometry —
    #: a load-cell problem, not aerodynamics. Left out of the fit, kept in the tables.
    qb = [r for r in comps["quad"]["slopes"] if r["rotor"] in (3, 4) and "per_s" in r
          and r["layout"] != "equal"]
    fits["quad_back_concept_tall"] = fit_s(qb)
    local = {}
    for r in back:
        local.setdefault((r["layout"], r["vbin"]), []).append(r["dT"] / r["per_s"])
    local = [{"layout": k[0], "vbin": k[1], "s_median": float(np.median(v)),
              "s_min": float(np.min(v)), "s_max": float(np.max(v)), "n": len(v)}
             for k, v in sorted(local.items())]
    print("\nlocal s = measured / (spec per unit s), tandem back rotor:")
    for l in local:
        print(f"  {l['layout']:8s} V {l['vbin']:2d}: median {l['s_median']:+.2f} "
              f"[{l['s_min']:+.2f}, {l['s_max']:+.2f}] n {l['n']}")
    print("\ns fitted to the back-rotor slope change (spec: −s·k·v_f/v_h·T_s/T):")
    for k, f in fits.items():
        print(f"  {k:22s} n {f['n']:3d}  s {f['s']:+.2f}  rms {f['rms']:.3f}  "
              f"(s = 0.44: {f['rms_s044']:.3f}; no interference: {f['rms_zero']:.3f})")

    # Idle-rotor offsets: what an unbladed / idle position reads while the motors run.
    offs = {}
    xl = read_xlsx(args.data / "mtbii_data_tables_v2.xlsx")
    head = next(i for i, r in enumerate(xl) if r and r[0] == "Run")
    cols = {n: j for j, n in enumerate(xl[head]) if n}
    for r in xl[head + 1:]:
        run = _num(r[0]) if r else None
        if run is None or int(run) not in RUNS or max(_num(r[cols[f"RPM_{i}"]]) or 0 for i in range(1, 5)) < 1900:
            continue
        lay, kind, single = RUNS[int(run)]
        act = (single,) if kind == "single" else ACTIVE[kind]
        for i in range(1, 5):
            if i not in act:
                offs.setdefault(kind, []).append(_num(r[cols[f"Fz{i}_H"]]) * LB)
    idle = {k: {"n": len(v), "median_N": float(np.median(v)), "min_N": float(np.min(v)),
                "max_N": float(np.max(v))} for k, v in offs.items()}
    print("\nidle-rotor Fz while the others run (N): " + ", ".join(
        f"{k} {v['median_N']:+.2f} [{v['min_N']:+.2f}, {v['max_N']:+.2f}]" for k, v in idle.items()))

    result = {"kind": "rotor_interference",
              "provenance": {"page": PAGE, "data_url": BASE + "mtbii_data_tables_v2.xlsx",
                             "report": "NASA/TM-20250004795 (May 2025)", "report_url": REPORT_URL,
                             "report_sha256": REPORT_SHA256, "sha256": SHA256,
                             "generator": "tools/rotor_interference_data.py",
                             "geometry": {"R_m": R, "l_hub_m": L_HUB,
                                          "dz_over_R": {k: v / R for k, v in DZ.items()}},
                             "front_reference": "run 102 (concept) for every layout",
                             "metric": "slope of T against ρ·rpm² over 2000/2500/3000 rpm, "
                                       "ratio to the same rotor alone, minus 1"},
              "static_C": c0, "idle_offsets": idle,
              "phase_0_vs_m90": {"n": int(rep.size), "median": float(np.median(rep)),
                                  "rms": float(np.sqrt(np.mean(rep ** 2)))},
              "local_s": local,
              "tables": tables, "tandem_back_by_pitch": by_pitch, "fits": fits}
    if args.write:
        OUT.write_text(json.dumps(result, indent=1) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
