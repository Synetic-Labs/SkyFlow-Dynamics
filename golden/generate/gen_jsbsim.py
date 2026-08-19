"""
Golden-vector generator: JSBSim (pip wheel, runnable reference) term-level vectors.

Runs the ACTUAL JSBSim flight dynamics engine (the official `jsbsim` PyPI wheel — same
C++ sources as github.com/JSBSim-Team/jsbsim at the wheel's release tag) and freezes
per-step tied-property sequences for the JSBSim-derived candidate terms:

    jsbsim_isa_atmosphere.json   FGStandardAtmosphere      spec.atmosphere (USSA-1976)
    jsbsim_dryden_lowalt.json    FGWinds ttTustin          spec.wind Dryden filters +
                                                           low-altitude closures (EXECUTED)
    jsbsim_prop_bldc.json        FGBrushLessDCMotor +      spec.motor_electrical DC-motor ODE,
                                 FGPropeller               spec.atmosphere advance-ratio model,
                                                           spec.inflow induced velocity
    jsbsim_rotor_inflow.json     FGRotor (SH79/Bramwell)   spec.inflow.dynamic_inflow_lag,
                                                           spec.rotor_aero.bramwell_torque

        uv run --with jsbsim python golden/generate/gen_jsbsim.py --out golden/vectors

Experiment rig (all families): `forces/hold-down` freezes position and ZEROES vehicle
velocity, so airspeed is produced with a steady WIND (atmosphere/wind-*-fps) — the
aerodynamic state is identical to flying at that airspeed, and (h, V) stay constant up to
turbulence feedback, which is recorded per step. Two custom minimal aircraft are built in a
scratch JSBSim root: `sfd_prop` (DJI_E305 brushless_dc_motor with noloadcurrent zeroed +
DJI_9450 propeller — both engine files shipped inside the wheel, sha256-recorded) and
`sfd_rotor` (FGRotor with the wheel's ah1s main-rotor geometry, ExternalRPM -1 so rotor
speed is commanded exactly, gearratio 1, no ground effect configured).

Noise recovery (Dryden): the reference's Gaussian stream is not reproducible outside its
binary, but the ttTustin difference equations (FGWinds.cpp, Yeager/CR-1998-206937 eqs
18-20) are affine in the current noise sample, so the generator back-solves nu_k from the
recorded outputs using the transcribed equations with each step's recorded (h, V). Two
checks make this non-circular: (a) replaying the recovered noise reproduces the recorded
outputs exactly (transcription consistency), (b) the recovered stream is plausibly white
N(0,1) (mean/variance/lag-1 bounds) — a transcription or closure error would leave
structure in the residual stream. properties/test_dryden_authenticity.py holds the
symbolic proof that these difference equations are the (C-prewarped) bilinear
discretization of spec.wind's filters; note JSBSim uses the u-axis prewarp constant
C_BL(tau_u) for all three axes (the proof is parametric in C, so this is still an exact
discretization of the spec filters).

Every family self-checks its transcription against the executed sequences (asserts) BEFORE
writing vectors. Transcribed equations verified against the release-tag sources
(src/models/atmosphere/FGWinds.cpp, src/models/propulsion/{FGPropeller,
FGBrushLessDCMotor}.cpp, src/models/propulsion/FGRotor.cpp) — line references in the
per-family provenance notes.
"""

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import shutil
import tempfile

import jsbsim

PKG = pathlib.Path(jsbsim.__file__).parent
DT = 1.0 / 120.0

# --- exact / defined conversions (NIST): 1 ft = 0.3048 m, 1 lbf = 4.4482216152605 N ---
FT = 0.3048
LBF = 4.4482216152605
FTLB = LBF * FT                       # N*m per ft*lbf = 1.3558179483314003
SLUG = LBF / FT                       # kg per slug = 14.593902937206364
SLUGFT3 = SLUG / FT**3                # kg/m^3 per slug/ft^3
SLUGFT2 = SLUG * FT**2                # kg*m^2 per slug*ft^2 (= FTLB numerically)
PSF = LBF / FT**2                     # Pa per lbf/ft^2
R_PER_K = 9.0 / 5.0
# --- JSBSim internal constants (transcribed) ---
NM_TO_FTLB_JSB = 1.3558               # FGBrushLessDCMotor.h:73 (rounded!)
WATT_PER_RPM_TO_FTLB = 60.0 / (2.0 * math.pi * NM_TO_FTLB_JSB)
EARTH_RADIUS_FT = 6356766.0 / 0.3048  # FGStandardAtmosphere geopotential constant
KGM2_LOADER = 1.35594                 # FGXMLElement.cpp:106 — the config loader's ROUNDED
#                                       SLUG*FT2<->KG*M2 constant (exact is 1.35581795);
#                                       the executed prop inertia is ixx_xml/1.35594 slug*ft2


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ custom root
ROTOR_XML = """<?xml version="1.0"?>
<rotor name="sfd_rotor_thruster">
  <!-- ah1s main-rotor geometry (wheel aircraft/ah1s/Engines/ah1s_rotor.xml), with:
       ExternalRPM -1 (rotor speed commanded via propulsion/engine[0]/x-rpm-dict),
       gearratio 1, and NO groundeffectexp (ground effect disabled). -->
  <ExternalRPM> -1 </ExternalRPM>
  <diameter unit="FT">                  44.0 </diameter>
  <numblades>                              2 </numblades>
  <gearratio>                            1.0 </gearratio>
  <nominalrpm>                         324.0 </nominalrpm>
  <chord unit="FT">                     2.25 </chord>
  <liftcurveslope>                       6.0 </liftcurveslope>
  <flappingmoment unit="SLUG*FT2">    1382.0 </flappingmoment>
  <twist>                             -0.175 </twist>
  <massmoment>                          85.0 </massmoment>
  <tiplossfactor>                        1.0 </tiplossfactor>
  <polarmoment unit="SLUG*FT2">       2900.0 </polarmoment>
  <inflowlag>                           0.09 </inflowlag>
  <hingeoffset unit="FT">               3.30 </hingeoffset>
</rotor>
"""

AIRCRAFT_TMPL = """<?xml version="1.0"?>
<fdm_config name="{name}" version="2.0" release="ALPHA">
  <fileheader><author>SkyFlow-Dynamics gen_jsbsim.py</author>
    <description>Minimal golden-vector testbed</description></fileheader>
  <metrics>
    <wingarea unit="FT2"> 1.0 </wingarea>
    <wingspan unit="FT"> 1.0 </wingspan>
    <chord unit="FT"> 1.0 </chord>
    <location name="AERORP" unit="IN"> 0 0 0 </location>
    <location name="EYEPOINT" unit="IN"> 0 0 0 </location>
    <location name="VRP" unit="IN"> 0 0 0 </location>
  </metrics>
  <mass_balance>
    <ixx unit="SLUG*FT2"> 10.0 </ixx>
    <iyy unit="SLUG*FT2"> 10.0 </iyy>
    <izz unit="SLUG*FT2"> 10.0 </izz>
    <emptywt unit="LBS"> 100.0 </emptywt>
    <location name="CG" unit="IN"> 0 0 0 </location>
  </mass_balance>
  <ground_reactions/>
  <propulsion>
{propulsion}
  </propulsion>
  <flight_control name="FCS: none"/>
  <aerodynamics/>
</fdm_config>
"""

ENGINE_BLOCK = """    <engine file="{engine}">
      <location unit="IN"> 0 0 0 </location>
      <orient unit="DEG"> 0 0 0 </orient>
      <thruster file="{thruster}">
        <location unit="IN"> 0 0 0 </location>
        <orient unit="DEG"> 0 0 0 </orient>
        <sense> 1 </sense>
      </thruster>
    </engine>"""


def build_root(root: pathlib.Path) -> dict:
    """Build the custom JSBSim root; return sha256 provenance of every data file used."""
    if root.exists():
        shutil.rmtree(root)
    (root / "aircraft" / "sfd_prop").mkdir(parents=True)
    (root / "aircraft" / "sfd_rotor").mkdir(parents=True)
    (root / "engine").mkdir()
    (root / "systems").mkdir()

    shutil.copy(PKG / "engine" / "DJI_9450.xml", root / "engine" / "DJI_9450.xml")
    e305 = (PKG / "engine" / "DJI_E305.xml").read_text()
    assert "<noloadcurrent>0.45</noloadcurrent>" in e305
    (root / "engine" / "sfd_e305_i0zero.xml").write_text(
        e305.replace('name="DJI E305"', 'name="sfd e305 i0zero"')
            .replace("<noloadcurrent>0.45</noloadcurrent>",
                     "<noloadcurrent>0.0</noloadcurrent>"))
    (root / "engine" / "sfd_dummy_electric.xml").write_text(
        '<electric_engine name="sfd dummy electric">\n'
        '  <power unit="HP"> 1.00 </power>\n</electric_engine>\n')
    (root / "engine" / "sfd_rotor_thruster.xml").write_text(ROTOR_XML)

    (root / "aircraft" / "sfd_prop" / "sfd_prop.xml").write_text(AIRCRAFT_TMPL.format(
        name="sfd_prop",
        propulsion=ENGINE_BLOCK.format(engine="sfd_e305_i0zero", thruster="DJI_9450")))
    (root / "aircraft" / "sfd_rotor" / "sfd_rotor.xml").write_text(AIRCRAFT_TMPL.format(
        name="sfd_rotor",
        propulsion=ENGINE_BLOCK.format(engine="sfd_dummy_electric",
                                       thruster="sfd_rotor_thruster")))
    return {
        "wheel_DJI_9450.xml_sha256": sha256(PKG / "engine" / "DJI_9450.xml"),
        "wheel_DJI_E305.xml_sha256": sha256(PKG / "engine" / "DJI_E305.xml"),
        "wheel_ah1s_rotor.xml_sha256":
            sha256(PKG / "aircraft" / "ah1s" / "Engines" / "ah1s_rotor.xml"),
    }


def new_fdm(root, aircraft):
    fdm = jsbsim.FGFDMExec(str(root))
    fdm.set_debug_level(0)
    fdm.load_model(aircraft)
    fdm.set_dt(DT)
    return fdm


def provenance(notes: str, extra: dict | None = None) -> dict:
    p = {
        "generator": "golden/generate/gen_jsbsim.py",
        "source": "JSBSim flight dynamics engine, official PyPI wheel (executed)",
        "jsbsim_version": jsbsim.FGJSBBase().get_version(),
        "source_tag": "v" + jsbsim.FGJSBBase().get_version().split()[0],
        "date": datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
        "rig": "forces/hold-down (position frozen, vehicle velocity zeroed); airspeed "
               "produced by steady wind; dt = 1/120 s",
        "units_note": "exact conversions: ft=0.3048 m, lbf=4.4482216152605 N; "
                      "slug*ft2->kg*m2 and ft*lbf->N*m factor 1.3558179483314003",
        "notes": notes,
    }
    if extra:
        p.update(extra)
    return p


# ------------------------------------------------------------------ ISA atmosphere
def gen_atmosphere(root, out):
    fdm = new_fdm(PKG, "ball")
    cases = []
    for h_ft in (0.0, 250.0, 1000.0, 2500.0, 5000.0, 8000.0, 12000.0, 16000.0,
                 20000.0, 25000.0, 30000.0, 35000.0):
        fdm["ic/h-sl-ft"] = h_ft
        fdm.run_ic()
        T_K = fdm["atmosphere/T-R"] * 5.0 / 9.0
        P_Pa = fdm["atmosphere/P-psf"] * PSF
        rho = fdm["atmosphere/rho-slugs_ft3"] * SLUGFT3
        a_ms = fdm["atmosphere/a-fps"] * FT
        H_m = h_ft * EARTH_RADIUS_FT / (EARTH_RADIUS_FT + h_ft) * FT
        # transcription check: JSBSim troposphere is exactly T = 288.15 K - 6.5 K/km * H
        assert abs(T_K - (288.15 - 0.0065 * H_m)) < 1e-6, (h_ft, T_K)
        cases.append({"h_geometric_ft": h_ft, "h_geopotential_m": H_m,
                      "expected": {"T_K": T_K, "P_Pa": P_Pa,
                                   "rho_kg_m3": rho, "a_m_s": a_ms}})
    doc = {
        "schema": 1, "kind": "jsbsim_terms", "name": "jsbsim_isa_atmosphere",
        "terms": ["isa_atmosphere"],
        "tolerance": 5e-4,  # JSBSim USSA-1976 constants (R*, P0) vs spec ICAO SI values
        "provenance": provenance(
            "FGStandardAtmosphere (USSA-1976): T/P/rho/a read from tied properties over "
            "an altitude grid, troposphere only. h_geopotential_m computed with JSBSim's "
            "geopotential conversion H = h*R_E/(R_E+h), R_E = 6356766 m / 0.3048; the "
            "conversion is asserted against the executed T(h) to 1e-6 K. Spec-side inputs "
            "are geopotential altitude in m; tolerance covers the gas-constant and P0 "
            "differences between USSA-1976 imperial constants and the spec's ICAO SI "
            "values (<= 1e-5 relative each)."),
        "cases": cases,
    }
    (out / "jsbsim_isa_atmosphere.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote jsbsim_isa_atmosphere.json: {len(cases)} altitudes")


# ------------------------------------------------------------------ Dryden low-altitude
def dryden_coeffs(h_ft, V, W20):
    """Transcribed FGWinds.cpp ttTustin coefficients for one step (lines 269-338).

    Low-altitude closures (lines 277-281): L_u = h/(0.177+0.000823h)^1.2, L_w = h,
    sig_w = 0.1*W20, sig_u = sig_w/(0.177+0.000823h)^0.4.  C_BL (line 316) is computed
    from tau_u and reused for the v and w axes.
    """
    h = max(h_ft, 10.0)
    assert h <= 1000.0, "low-altitude branch only"
    f = 0.177 + 0.000823 * h
    L_u = h / f**1.2
    L_w = h
    sig_w = 0.1 * W20
    sig_u = sig_w / f**0.4
    tau_u = L_u / V
    C = 1.0 / tau_u / math.tan(DT / 2.0 / tau_u)
    om_v = V / L_u
    om_w = V / L_w
    return sig_u, sig_w, tau_u, C, om_v, om_w


def second_order_terms(sig, om, C):
    a1 = 2.0 * (om**2 - C**2) / (om + C) ** 2
    a2 = (om - C) ** 2 / (om + C) ** 2
    g = sig * math.sqrt(3.0 * om / DT) / (om + C) ** 2
    b0 = C + om / math.sqrt(3.0)
    b1 = 2.0 * om / math.sqrt(3.0)
    b2 = om / math.sqrt(3.0) - C
    return a1, a2, g, b0, b1, b2


def gen_dryden(root, out):
    cases = []
    for h_ft, W20, wind, seed in ((200.0, 25.0, 60.0, 20260818),
                                  (500.0, 25.0, 40.0, 20260819),
                                  (900.0, 35.0, 80.0, 20260820),
                                  (50.0, 15.0, 30.0, 20260821)):
        fdm = new_fdm(PKG, "ball")
        fdm["ic/h-sl-ft"] = h_ft
        fdm.run_ic()
        fdm["forces/hold-down"] = 1
        fdm["atmosphere/wind-north-fps"] = wind      # psiw = 0: north==xi_u, east==xi_v
        for _ in range(3):
            fdm.run()                                # warm up: vt settles to the wind
        assert abs(fdm["velocities/vt-fps"] - wind) < 1e-9  # (turbulence still off,
        fdm["atmosphere/randomseed"] = seed                  # filter histories at zero)
        fdm["atmosphere/turb-type"] = 4              # ttTustin
        fdm["atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps"] = W20
        fdm["atmosphere/turbulence/milspec/severity"] = 3
        hs, Vs, tu, tv, tw = [], [], [], [], []
        for _ in range(400):
            hs.append(fdm["position/h-sl-ft"])
            Vs.append(fdm["velocities/vt-fps"])      # value FGWinds consumes this step
            fdm.run()
            tu.append(fdm["atmosphere/turb-north-fps"])
            tv.append(fdm["atmosphere/turb-east-fps"])
            tw.append(fdm["atmosphere/turb-down-fps"])

        # --- back-solve the noise, then replay to prove transcription consistency ---
        nu_u, nu_v, nu_w = [], [], []
        xiu = nuu = 0.0
        xv1 = xv2 = nv1 = nv2 = 0.0
        xw1 = xw2 = nw1 = nw2 = 0.0
        for k in range(400):
            sig_u, sig_w, tau_u, C, om_v, om_w = dryden_coeffs(hs[k], Vs[k], W20)
            # u axis, eq(18)
            g_u = sig_u * math.sqrt(2.0 * tau_u / DT) / (1.0 + C * tau_u)
            a_u = (1.0 - C * tau_u) / (1.0 + C * tau_u)
            n = (tu[k] + a_u * xiu) / g_u - nuu
            nu_u.append(n)
            xiu, nuu = tu[k], n
            # v axis, eq(20) with om_v, sig_u
            a1, a2, g, b0, b1, b2 = second_order_terms(sig_u, om_v, C)
            n = (tv[k] + a1 * xv1 + a2 * xv2 - g * (b1 * nv1 + b2 * nv2)) / (g * b0)
            nu_v.append(n)
            xv2, xv1 = xv1, tv[k]
            nv2, nv1 = nv1, n
            # w axis, eq(20) with om_w, sig_w
            a1, a2, g, b0, b1, b2 = second_order_terms(sig_w, om_w, C)
            n = (tw[k] + a1 * xw1 + a2 * xw2 - g * (b1 * nw1 + b2 * nw2)) / (g * b0)
            nu_w.append(n)
            xw2, xw1 = xw1, tw[k]
            nw2, nw1 = nw1, n
        for ns in (nu_u, nu_v, nu_w):
            m = sum(ns) / len(ns)
            var = sum((x - m) ** 2 for x in ns) / (len(ns) - 1)
            r1 = (sum((ns[i] - m) * (ns[i + 1] - m) for i in range(len(ns) - 1))
                  / ((len(ns) - 1) * var))
            assert abs(m) < 0.2 and 0.75 < var < 1.35 and abs(r1) < 0.2, \
                ("recovered noise not white N(0,1)", h_ft, m, var, r1)

        cases.append({
            "params": {"W20_fps": W20, "dt_s": DT, "wind_north_fps": wind, "seed": seed},
            "sequence": {"h_ft": hs, "V_fps": Vs,
                         "noise_u": nu_u, "noise_v": nu_v, "noise_w": nu_w},
            "expected": {"turb_north_fps": tu, "turb_east_fps": tv,
                         "turb_down_fps": tw},
        })
    doc = {
        "schema": 1, "kind": "jsbsim_terms", "name": "jsbsim_dryden_lowalt",
        "terms": ["dryden_turbulence (low-altitude closures)"],
        "tolerance": 1e-9,
        "provenance": provenance(
            "FGWinds.cpp ttTustin (lines 258-345): NASA CR-1998-206937 (Yeager) Tustin "
            "difference equations eqs 18-20 with the MIL-F-8785C low-altitude closures "
            "ACTIVE (h <= 1000 ft, lines 277-281) — the closures' first executed-code "
            "exercise (the CR-206937 check data pins L = 1750 ft only). Sequences are in "
            "the reference's native ft/s and FEET (the closures' published units). The "
            "driving noise is back-solved per step from the recorded outputs (affine "
            "inversion of the difference equations, per-step recorded h and V) and "
            "whiteness-checked: mean, variance, lag-1 autocorrelation of each recovered "
            "stream within N(0,1) bounds (asserted at generation). psiw = 0 so "
            "north/east/down = xi_u/xi_v/xi_w. C_BL prewarp constant is computed from "
            "tau_u and reused for v/w axes (JSBSim's choice; the spec-filter equivalence "
            "proof in properties/test_dryden_authenticity.py is parametric in C)."),
        "cases": cases,
    }
    (out / "jsbsim_dryden_lowalt.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote jsbsim_dryden_lowalt.json: {len(cases)} cases x 400 steps")


# ------------------------------------------------------------------ prop + BLDC motor
D_FT = 9.4 / 12.0                     # DJI_9450 diameter
IXX_KGM2 = 6.05e-05                   # DJI_9450 <ixx unit="KG*M2">
KV_RPM_V = 960.0                      # DJI_E305
RM_OHM = 0.117
VMAX = 14.63


def gen_prop_bldc(root, out):
    ixx_slugft2 = IXX_KGM2 / KGM2_LOADER   # what JSBSim actually loaded and integrated with
    cases = []
    for wind, throttle in ((0.0, 0.35), (0.0, 0.65), (0.0, 0.95),
                           (30.0, 0.65), (60.0, 0.85), (15.0, 0.5)):
        fdm = new_fdm(root, "sfd_prop")
        fdm["ic/h-sl-ft"] = 1000.0
        fdm.run_ic()
        fdm["forces/hold-down"] = 1
        fdm["atmosphere/wind-north-fps"] = -wind     # headwind: axial Vel = +wind
        fdm["propulsion/set-running"] = -1
        fdm["fcs/throttle-cmd-norm"] = throttle
        rho = fdm["atmosphere/rho-slugs_ft3"]
        seq = {k: [] for k in ("rpm", "J", "CT", "vi_fps", "P_req_ftlbps",
                               "thrust_lbf", "current_A")}
        for _ in range(150):
            fdm.run()
            e = "propulsion/engine[0]/"
            seq["rpm"].append(fdm[e + "propeller-rpm"])
            seq["J"].append(fdm[e + "advance-ratio"])
            seq["CT"].append(fdm[e + "thrust-coefficient"])
            seq["vi_fps"].append(fdm[e + "prop-induced-velocity_fps"])
            seq["P_req_ftlbps"].append(fdm[e + "propeller-power-ftlbps"])
            seq["thrust_lbf"].append(fdm[e + "thrust-lbs"])
            seq["current_A"].append(fdm[e + "current-amperes"])

        # --- transcription self-check over the whole tail (native units) ---
        area = 0.25 * D_FT * D_FT * math.pi
        for k in range(1, 150):
            rps = seq["rpm"][k - 1] / 60.0           # entry RPM of step k
            omega = rps * 2.0 * math.pi
            assert abs(seq["J"][k] - (wind / (D_FT * rps) if rps > 0.01
                                      else wind / D_FT)) < 1e-9
            T = seq["CT"][k] * rps * rps * D_FT**4 * rho
            assert abs(T - seq["thrust_lbf"][k]) < 1e-9 * max(1.0, abs(T))
            s = wind * abs(wind) + 2.0 * T / (rho * area)
            vi = 0.5 * (-wind + math.sqrt(s)) if s > 0 else \
                 0.5 * (-wind - math.sqrt(-s))
            assert abs(vi - seq["vi_fps"][k]) < 1e-9
            cur = (VMAX * throttle - seq["rpm"][k - 1] / KV_RPM_V) / RM_OHM
            assert abs(cur - seq["current_A"][k]) < 1e-9
            q_m = cur / KV_RPM_V * WATT_PER_RPM_TO_FTLB   # noloadcurrent = 0
            p_eng = 2.0 * math.pi * max(seq["rpm"][k - 1], 1e-4) * q_m / 60.0
            exc = (p_eng - seq["P_req_ftlbps"][k]) / (omega if omega > 0.01 else 1.0)
            rpm_next = (rps + ((exc / ixx_slugft2) / (2.0 * math.pi)) * DT) * 60.0
            assert abs(rpm_next - seq["rpm"][k]) < 1e-6 * max(1.0, seq["rpm"][k])

        cases.append({
            "params": {"V_axial_m_s": wind * FT, "throttle": throttle,
                       "rho_kg_m3": rho * SLUGFT3, "dt_s": DT},
            "sequence_si": {
                "Omega_rad_s": [r / 60.0 * 2.0 * math.pi for r in seq["rpm"]],
                "J": seq["J"], "CT": seq["CT"],
                "vi_m_s": [v * FT for v in seq["vi_fps"]],
                "P_req_W": [p * FTLB for p in seq["P_req_ftlbps"]],
                "thrust_N": [t * LBF for t in seq["thrust_lbf"]],
                "current_A": seq["current_A"],
            },
        })
    doc = {
        "schema": 1, "kind": "jsbsim_terms", "name": "jsbsim_prop_bldc",
        "terms": ["advance_ratio_tables", "momentum_induced_velocity",
                  "dc_motor_quasistatic"],
        "tolerance": 1e-7,
        "params": {
            "D_m": D_FT * FT, "disk_area_m2": 0.25 * math.pi * (D_FT * FT) ** 2,
            # SI equivalent of the inertia the reference EXECUTED: the config loader
            # converts <ixx unit="KG*M2"> with the rounded 1.35594 constant (9.0e-5 high),
            # so the simulated prop's J_r is xml_value * (1.35581795/1.35594).
            "J_r_kg_m2": (IXX_KGM2 / KGM2_LOADER) * SLUGFT2,
            "K_e_V_s_rad": 60.0 / (2.0 * math.pi * KV_RPM_V),
            "K_q_Nm_A": (WATT_PER_RPM_TO_FTLB / KV_RPM_V) * FTLB,
            "R_a_ohm": RM_OHM, "V_max_V": VMAX,
        },
        "provenance": provenance(
            "FGBrushLessDCMotor::Calculate (Drela QPROP three-constant model, "
            "noloadcurrent set to 0 so the torque law is exactly Q = K_q(V - K_e*Omega)/"
            "R_a) driving FGPropeller::Calculate (advance ratio J = V/(nD) at lines "
            "232-233, T = C_T(J) rho n^2 D^4 at line 249, McCormick sign-safe induced "
            "velocity at lines 256-261, P_req = C_P(J) rho n^3 D^5, explicit-Euler shaft "
            "update RPM_(k+1) = (RPS_k + (P_avail/omega/Ixx)/(2pi) dt)*60 at line 294). "
            "Engine/prop data: wheel DJI_E305 (Kv 960 RPM/V, Rm 0.117 ohm, Vmax 14.63 V) "
            "+ DJI_9450 (APC 9x4.5E C_T/C_P tables; the tables are the term's measured "
            "parameters, interpolated 1-D). Every recorded step is asserted against this "
            "transcription to 1e-9 before writing. K_q carries JSBSim's rounded "
            "1.3558 N*m/(ft*lbf) constant folded back through the exact factor (1.3e-5 "
            "relative wart, documented); J_r_kg_m2 likewise folds back the config "
            "loader's rounded KG*M2->SLUG*FT2 constant 1.35594 (FGXMLElement.cpp:106) so "
            "it is exactly the inertia the reference integrated with — diagnosed from a "
            "9.0e-5 residual in the Euler-update self-check. Axial airspeed is the exact "
            "case parameter "
            "(steady headwind, turbulence off). Static cases (V=0): J identically 0, "
            "C_P(0)/C_T(0) constant -> the shaft ODE is exactly the spec's quadratic-"
            "load DC-motor ODE; vi reduces to the hover form sqrt(T/(2 rho A)). "
            "Windmilling (J<0 / C_T<0) is NOT exercised (table domain J>=0). The b*Omega "
            "viscous term and the I0 friction deadband are NOT exercised (b=0, I0=0)."),
        "cases": cases,
    }
    (out / "jsbsim_prop_bldc.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote jsbsim_prop_bldc.json: {len(cases)} cases x 150 steps")


# ------------------------------------------------------------------ rotor inflow/torque
ROTOR = {"R_ft": 22.0, "blades": 2, "chord_ft": 2.25, "a_slope": 6.0,
         "twist_rad": -0.175, "tiploss_B": 1.0, "tau_s": 0.09, "rpm": 324.0}


def bailey_c0(theta0, mu, lam_prev):
    """Transcribed FGRotor::calc_flow_and_thrust lines 420-430: the Glauert equilibrium
    inflow c0 = C_T(lambda_prev)/(2 sqrt(mu^2+lambda_prev^2)+1e-15), Bailey C_T."""
    B = ROTOR["tiploss_B"]
    sigma = ROTOR["blades"] * ROTOR["chord_ft"] / (math.pi * ROTOR["R_ft"])
    mu2 = mu * mu
    ct_t0 = (B**3 / 3.0 + B * mu2 / 2.0 - 4.0 / (9.0 * math.pi) * mu * mu2) * theta0
    ct_t1 = (B**4 / 4.0 + B * B * mu2 / 4.0) * ROTOR["twist_rad"]
    ct_l = (B * B / 2.0 + mu2 / 4.0) * lam_prev
    c0 = (ROTOR["a_slope"] / 2.0) * (ct_l + ct_t0 + ct_t1) * sigma
    return c0 / (2.0 * math.sqrt(mu2 + lam_prev * lam_prev) + 1e-15)


def gen_rotor(root, out):
    sigma = ROTOR["blades"] * ROTOR["chord_ft"] / (math.pi * ROTOR["R_ft"])
    omega = ROTOR["rpm"] / 60.0 * 2.0 * math.pi
    e_exp = math.exp(-DT / ROTOR["tau_s"])
    cases = []
    for name, wn, we, theta0 in (("hover", 0.0, 0.0, 0.25),
                                 ("axial_inflow", -40.0, 0.0, 0.25),
                                 ("axial_outflow", 25.0, 0.0, 0.25),
                                 ("edgewise", 0.0, 50.0, 0.20),
                                 ("oblique", -20.0, 40.0, 0.25)):
        fdm = new_fdm(root, "sfd_rotor")
        fdm["ic/h-sl-ft"] = 3000.0
        fdm.run_ic()
        fdm["forces/hold-down"] = 1
        fdm["atmosphere/wind-north-fps"] = wn
        fdm["atmosphere/wind-east-fps"] = we
        fdm["propulsion/set-running"] = -1
        fdm["propulsion/engine[0]/x-rpm-dict"] = ROTOR["rpm"]
        fdm["propulsion/engine[0]/collective-ctrl-rad"] = theta0
        rho = fdm["atmosphere/rho-slugs_ft3"]
        seq = {k: [] for k in ("nu", "lambda", "mu", "CT", "thrust_lbf",
                               "torque_lbsft")}
        for _ in range(160):
            fdm.run()
            e = "propulsion/engine[0]/"
            assert abs(fdm[e + "rotor-rpm"] - ROTOR["rpm"]) < 1e-9
            seq["nu"].append(fdm[e + "induced-inflow-ratio"])
            seq["lambda"].append(fdm[e + "inflow-ratio"])
            seq["mu"].append(fdm[e + "advance-ratio"])
            seq["CT"].append(fdm[e + "thrust-coefficient"])
            seq["thrust_lbf"].append(fdm[e + "thrust-lbs"])
            seq["torque_lbsft"].append(fdm[e + "torque-lbsft"])
        if name in ("edgewise", "oblique"):
            assert seq["mu"][-1] > 0.01, (name, seq["mu"][-1])
        if name == "hover":
            assert seq["thrust_lbf"][-1] > 0 and seq["nu"][-1] > 0, "hover regime sane"

        # --- derived inputs + transcription self-check ---
        nu_eq, H_lbf = [None], [None]            # step 0 has no recorded predecessor
        t075 = theta0 + 0.75 * ROTOR["twist_rad"]
        for k in range(1, 160):
            mu, lam, ct = seq["mu"][k], seq["lambda"][k], seq["CT"][k]
            c0 = bailey_c0(theta0, mu, seq["lambda"][k - 1])
            nu_pred = (seq["nu"][k - 1] - c0) * e_exp + c0
            assert abs(nu_pred - seq["nu"][k]) < 1e-12, (name, k)
            nu_eq.append(c0)
            # H-force from the flapping downwash coefficient (pqr = 0 under hold-down):
            a_dw = (2.0 * lam + 8.0 / 3.0 * t075) * mu / (1.0 - mu * mu / 2.0)
            H = seq["thrust_lbf"][k] * a_dw
            H_lbf.append(H)
            delta = 0.009 + 0.3 * (6.0 * ct / (ROTOR["a_slope"] * sigma)) ** 2
            q = (rho * ROTOR["blades"] * ROTOR["chord_ft"] * delta
                 * (omega * ROTOR["R_ft"]) ** 2 * ROTOR["R_ft"] ** 2
                 * (1.0 + 4.5 * mu * mu) / 8.0
                 - (seq["thrust_lbf"][k] * lam + H * mu) * ROTOR["R_ft"])
            assert abs(q - seq["torque_lbsft"][k]) < 1e-9 * max(1.0, abs(q)), (name, k)

        cases.append({
            "name": name, "params": {
                "theta0_rad": theta0, "rho_kg_m3": rho * SLUGFT3, "dt_s": DT,
                "wind_north_fps": wn, "wind_east_fps": we,
            },
            "sequence": {
                "nu": seq["nu"], "lambda": seq["lambda"], "mu": seq["mu"],
                "CT": seq["CT"], "nu_eq": nu_eq,
                "thrust_N": [t * LBF for t in seq["thrust_lbf"]],
                "H_N": [None if h is None else h * LBF for h in H_lbf],
                "torque_N_m": [q * FTLB for q in seq["torque_lbsft"]],
            },
        })
    doc = {
        "schema": 1, "kind": "jsbsim_terms", "name": "jsbsim_rotor_inflow",
        "terms": ["dynamic_inflow_lag", "bramwell_rotor_torque"],
        "tolerance": 1e-7,
        "params": {
            "R_m": ROTOR["R_ft"] * FT, "blades": ROTOR["blades"],
            "chord_m": ROTOR["chord_ft"] * FT, "a_slope": ROTOR["a_slope"],
            "solidity": sigma, "tau_s": ROTOR["tau_s"],
            "Omega_rad_s": omega, "tiploss_B": ROTOR["tiploss_B"],
            "twist_rad": ROTOR["twist_rad"],
        },
        "provenance": provenance(
            "FGRotor (NASA TP-1285 / Bramwell model, ah1s geometry, ExternalRPM so "
            "Omega is exact, ground effect not configured -> flow_scale = 1, hold-down "
            "-> body rates exactly 0). Recorded ties per step: nu (induced-inflow-ratio),"
            " lambda (inflow-ratio), mu (advance-ratio), C_T, thrust, torque. nu_eq is "
            "the Glauert equilibrium c0 = C_T_Bailey(lambda_prev)/(2 sqrt(mu^2+"
            "lambda_prev^2)+1e-15) computed from the recorded state via the transcribed "
            "calc_flow_and_thrust (lines 420-430) — asserted at generation: the recorded "
            "nu sequence satisfies nu_k = (nu_(k-1) - nu_eq_k) exp(-dt/tau) + nu_eq_k to "
            "1e-12 (the exact-exponential step of dnu/dt = (nu_eq - nu)/tau, line 436). "
            "H is Thrust*a_dw with a_dw from calc_flapping_angles lines 495-499 at zero "
            "body rates; recorded torque is asserted against calc_torque lines 536-540 "
            "(Bramwell profile + induced/climb decomposition with the "
            "delta = 0.009 + 0.3(6 C_T/(a sigma))^2 drag polar) to 1e-9 relative. "
            "Body-rate flapping contributions (p,q terms) are NOT exercised (held at 0). "
            "Sequences: nondimensional ratios as recorded; forces/torques converted to "
            "SI exactly."),
        "cases": cases,
    }
    (out / "jsbsim_rotor_inflow.json").write_text(json.dumps(doc, indent=1))
    print(f"wrote jsbsim_rotor_inflow.json: {len(cases)} cases x 160 steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(tempfile.mkdtemp(prefix="sfd_jsbsim_root_"))
    try:
        shas = build_root(root)
        print("jsbsim:", jsbsim.FGJSBBase().get_version())
        for k, v in shas.items():
            print(f"  {k} = {v}")
        gen_atmosphere(root, out)
        gen_dryden(root, out)
        gen_prop_bldc(root, out)
        gen_rotor(root, out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
