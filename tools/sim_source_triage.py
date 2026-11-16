"""
Triage numbers for the 2026-11-15 intake of libfpv / SimITL and rl-tools l2f (REFERENCES.md,
"libfpv + SimITL" and "rl-tools l2f") — the sizes quoted there and in the registry entries
imu_rotor_vibration, propwash_thrust_unsteadiness and rotor_drag_hforce.

    uv run python tools/sim_source_triage.py

Five checks, each against data already in this repo:
  1. rotor-drag prior: libfpv sets k_d = κ·√(ρ·A·c_T/2) with κ = 0.6 (patch 0011). κ is
     evaluated on the reference vehicles; RACER_5IN's k_d is measured.
  2. libfpv's pitch-based thrust falloff C_T(J) = C_T0·(1 − (J/J0)²), J0 = 0.9·pitch/D
     (patch 0011), against the executed JSBSim APC 9x4.5E table
     (golden/vectors/jsbsim_prop_bldc.json).
  3. gust gradient across the airframe (libfpv samples the wind at each hub, patch 0017):
     rms roll disturbance when the left and right rotor pairs see different vertical gusts,
     from the von Kármán transverse correlation at the MIL-F-8785C low-altitude scales.
  4. SimITL's propwash thrust loss, at the INTAKE axial operating points.
  5. SimITL's rotor-synchronous gyro vibration against its own white gyro noise, and its
     angular damping against BEM (F-29).

Checks 4 and 5 evaluate SimITL's hand-tuned constants (transcribed below with file
locations). They size what that simulator does; they are not measurements.
"""

import json
import math
import pathlib

import mpmath as mp
import numpy as np

from skyflow_dynamics.spec.parameters import CRAZYFLIE, RACER_5IN

ROOT = pathlib.Path(__file__).resolve().parent.parent
RHO = 1.225
VEHICLES = {"CRAZYFLIE": CRAZYFLIE, "RACER_5IN": RACER_5IN}

#: 1 knot in m/s; MIL-F-8785C names W20 = 15 / 30 / 45 kt light / moderate / severe.
KNOT = 0.514444
W20_LEVELS = {"5 m/s": 5.0, "light": 15 * KNOT, "moderate": 30 * KNOT, "severe": 45 * KNOT}

#: SimITL c881d55 src/sim/physics.cpp with libfpv 08d140a's "Freestyler 5" preset
#: (assets/presets/quad/freestyler-5.json, motor/generic-2207-1950kV.json,
#: propeller/generic-5.1x3x3.json).
SIMITL_5IN = {
    "max_propwash_speed": 20.0,    # m/s   physics.cpp:518,522  speedFactor = speed / this
    "propwash_aoa": 0.1,           # -     physics.cpp:516,540  cosine threshold
    "propwash_factor": 1.1,        # -     physics.cpp:552
    "gyro_white_amp": 0.005,       # rad/s physics.cpp:379      uniform half-range
    "imbalance_x": 8.0,            # rad/s physics.cpp:419      per-motor amplitude scale
    "damage_floor": 0.05,          # -     physics.cpp:404      dmgFactor of an undamaged prop
    "kv": 1950.0, "cells": 6,      # physics.cpp:389-393        maxRpm = kV · 4.2 V · cells
    "prop_max_rpm": 33000.0,       # rpm at full static thrust
    "max_thrust": 15.5,            # N per motor
    "mass": 0.58,                  # kg ready to fly
    "ang_drag_quad": 1e-5,         # physics.cpp:807            × frameDragArea × |ω|ω
    "frame_drag_area": 0.0095,     # m²
}

#: BEM roll damping D = −M/p on the racer, N·m·s/rad (F-29, tools/bem_rate_damping.py).
BEM_ROLL_DAMPING = {10.0: 0.00132, 200.0: 0.00402}


def hover(v: dict) -> tuple:
    """(rotor speed rad/s, induced velocity m/s) at hover."""
    ct = v["ct2"][0]
    area = math.pi * v["r_prop"] ** 2
    W = math.sqrt(v["mass"] * v["grav"] / (len(v["ct2"]) * ct))
    return W, W * math.sqrt(ct / (2 * RHO * area))


def rotor_drag_kappa(v: dict) -> float:
    """κ = k_d / √(ρ·A·c_T/2): the H-force as a fraction of the hover momentum drag ṁ·v."""
    area = math.pi * v["r_prop"] ** 2
    return v["k_d"] / math.sqrt(RHO * area * v["ct2"][0] / 2)


def pitch_closure_error() -> dict:
    """libfpv's C_T(J) closure against the executed APC 9x4.5E table (pitch/D = 0.5)."""
    doc = json.loads((ROOT / "golden" / "vectors" / "jsbsim_prop_bldc.json").read_text())
    pts = {}
    for case in doc["cases"]:
        seq = case["sequence_si"]
        for J, CT in zip(seq["J"], seq["CT"]):
            if 1e-3 < J < 0.84:       # spin-up samples at Ω ≈ 0 give meaningless J
                pts[round(J, 3)] = CT
    J = np.array(sorted(pts))
    CT = np.array([pts[j] for j in J])
    CT0 = max(c for case in doc["cases"] for c in case["sequence_si"]["CT"])
    J0 = 0.9 * 4.5 / 9.0
    model = np.maximum(0.0, CT0 * (1 - (J / J0) ** 2))
    err = (model - CT) / CT0
    worst = int(np.argmax(np.abs(err)))
    return {"n": len(J), "J_range": (float(J.min()), float(J.max())), "CT0": CT0, "J0": J0,
            "rms": float(np.sqrt(np.mean(err ** 2))), "worst": float(abs(err[worst])),
            "worst_J": float(J[worst]), "table_CT_at_J0": float(np.interp(J0, J, CT)) / CT0}


def von_karman_transverse(r: float, L: float) -> float:
    """Transverse velocity correlation g(r) of isotropic von Kármán turbulence, scale L."""
    def f(z):
        return 2 ** (2 / 3) / mp.gamma(mp.mpf(1) / 3) * z ** (mp.mpf(1) / 3) * \
            mp.besselk(mp.mpf(1) / 3, z)
    z = mp.mpf(r) / (1.339 * L)
    return float(f(z) + z / 2 * mp.diff(f, z))


def gust_gradient(v: dict, W20: float, h: float) -> dict:
    """Roll disturbance from the vertical-gust difference between the left and right rotor
    pairs, through the k_z thrust channel: τ = 2·a_y·Ω_h·k_z·Δw. MIL low-altitude closure:
    σ_w = 0.1·W20, L_w = h."""
    a_y = abs(v["rotor_pos"][0][1])
    W, _ = hover(v)
    sigma_w = 0.1 * W20
    dw = sigma_w * math.sqrt(2 * (1 - von_karman_transverse(2 * a_y, h)))
    torque = 2 * a_y * W * v["k_z"] * dw
    authority = 2 * a_y * v["ct2"][0] * v["limits"]["rotor_speed_max"] ** 2
    return {"dw_rms": dw, "roll_accel_rms": torque / v["inertia"][0][0],
            "roll_authority": authority / v["inertia"][0][0],
            "uniform_gust_force_fraction": 4 * W * v["k_z"] * sigma_w / (v["mass"] * v["grav"])}


def simitl_propwash_loss(airspeed: float, cos_into_wake: float = 1.0,
                         p: dict = SIMITL_5IN) -> float:
    """Upper bound of SimITL's per-rotor thrust loss: its noise factor (low-passed
    |simplex|, 45 Hz, drawn per rotor) is at most 1. cos_into_wake is the cosine between
    the airspeed and the reversed thrust axis — 1 in a straight axial descent."""
    speed_factor = min(airspeed / p["max_propwash_speed"], 1.0)
    reverse = max(0.0, (cos_into_wake - p["propwash_aoa"]) / (1.0 - p["propwash_aoa"])) ** 2
    return min(1.0, speed_factor * reverse * p["propwash_factor"])


def simitl_vibration(p: dict = SIMITL_5IN) -> dict:
    """Per-motor amplitude of SimITL's rotor-frequency gyro term (roll axis, undamaged
    prop) at full throttle and at hover, against its white-noise half-range."""
    no_load_rpm = p["kv"] * 4.2 * p["cells"]
    full = (p["prop_max_rpm"] / no_load_rpm) ** 2
    twr = 4 * p["max_thrust"] / (p["mass"] * 9.81)
    amp = p["imbalance_x"] * p["damage_floor"]
    return {"full": amp * full, "hover": amp * full / twr, "white": p["gyro_white_amp"],
            "thrust_to_weight": twr}


def simitl_angular_damping_fraction(rate: float, p: dict = SIMITL_5IN) -> float:
    """SimITL's quadratic roll damping torque as a fraction of BEM's at the same rate."""
    return p["ang_drag_quad"] * p["frame_drag_area"] * rate ** 2 / (BEM_ROLL_DAMPING[rate] * rate)


def main():
    print("1. rotor-drag prior κ = k_d/√(ρ·A·c_T/2)   (libfpv assumes 0.6 for open props)")
    for name, v in VEHICLES.items():
        print(f"   {name}: κ = {rotor_drag_kappa(v):.3f}")

    e = pitch_closure_error()
    print(f"2. libfpv C_T(J) closure, J0 = {e['J0']:.2f}, vs APC 9x4.5E table "
          f"({e['n']} points, J {e['J_range'][0]:.2f}–{e['J_range'][1]:.2f}):")
    print(f"   rms error {e['rms']:.3f} of static thrust, worst {e['worst']:.2f} at "
          f"J = {e['worst_J']:.2f}; at J0 the table still gives {e['table_CT_at_J0']:.2f}")

    print("3. gust gradient: rms roll acceleration, rad/s² (authority in brackets)")
    for name, v in VEHICLES.items():
        for h in (10.0, 2.0):
            row = "  ".join(f"{lab} {gust_gradient(v, W, h)['roll_accel_rms']:.2f}"
                            for lab, W in W20_LEVELS.items())
            print(f"   {name} h = {h:.0f} m: {row}  "
                  f"[{gust_gradient(v, 5.0, h)['roll_authority']:.0f}]")

    _, v_h = hover(RACER_5IN)
    print(f"4. SimITL propwash thrust loss per rotor, upper bound, racer v_h = {v_h:.2f} m/s:")
    print(f"   descent 0.5 v_h: {simitl_propwash_loss(0.5 * v_h):.2f}   "
          f"2 v_h: {simitl_propwash_loss(2 * v_h):.2f}")

    vib = simitl_vibration()
    print(f"5. SimITL gyro vibration per motor: {vib['full']:.3f} rad/s at full throttle, "
          f"{vib['hover']:.4f} at hover; white-noise half-range {vib['white']} rad/s")
    print("   SimITL angular damping / BEM roll damping: "
          + ", ".join(f"{simitl_angular_damping_fraction(r):.4f} at {r:.0f} rad/s"
                      for r in BEM_ROLL_DAMPING))


if __name__ == "__main__":
    main()
