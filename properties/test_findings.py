"""Findings from use (REFERENCES.md, "Findings from use"): each finding that a registry entry
cites as 'derived' evidence is reproduced here, so the numbers in the ledger stay tied to code.

F-29 — body-rate damping of the verified BEM rotor model (tools/bem_rate_damping.py): with the
canonical hub velocity v + ω×r it damps and stiffens with rate; with agilib's frame mixing
(F-24) roll is anti-damped and yaw damping all but vanishes.

RACER_5IN / F-30 — the reference vehicle's values match the frozen identification
(tools/identify_neurobem.py → golden/checkdata/neurobem_racer.json); its BEM-derived k_z
reproduces without the flight data; the flight data's moments do not close.

Source triage 2026-11-15 (libfpv / SimITL, rl-tools l2f; tools/sim_source_triage.py) — the
sizes quoted in REFERENCES.md and in the registry entries imu_rotor_vibration,
propwash_thrust_unsteadiness and rotor_drag_hforce."""

import json
import math
import pathlib

import pytest

from skyflow_dynamics.spec.parameters import RACER_5IN
from tools import bem_rate_damping as brd
from tools import sim_source_triage as triage

RATES = (1.0, 13.0, 200.0)


@pytest.fixture(scope="module")
def canonical():
    return brd.damping("canonical", RATES)


@pytest.fixture(scope="module")
def agilib():
    return brd.damping("agilib", RATES)


def test_hover_speed_matches_flight_data(canonical):
    # NeuroBEM flight data: hover motor speed ≈ 1120 rad/s at 0.772 kg
    assert canonical["hover_rotor_speed"] == pytest.approx(1117.0, abs=2.0)


def test_canonical_damps_on_every_axis(canonical):
    for axis in ("roll", "pitch", "yaw"):
        assert all(d > 0 for d in canonical[axis].values()), axis


def test_roll_damping_stiffens_with_rate(canonical):
    roll = canonical["roll"]
    assert roll[1.0] < roll[13.0] < roll[200.0]
    assert roll[1.0] == pytest.approx(0.00086, rel=0.02)
    assert roll[13.0] == pytest.approx(0.00145, rel=0.02)
    assert roll[200.0] == pytest.approx(0.00402, rel=0.02)
    # a linear term matched at 13 rad/s is a third of BEM at 200 rad/s
    assert roll[13.0] / roll[200.0] == pytest.approx(0.36, abs=0.02)


def test_f24_frame_mixing_flips_roll_and_drops_yaw(canonical, agilib):
    for p in RATES:
        assert agilib["roll"][p] == pytest.approx(-canonical["roll"][p], rel=1e-6)
        assert agilib["pitch"][p] == pytest.approx(canonical["pitch"][p], rel=1e-6)
        assert abs(agilib["yaw"][p]) < 0.05 * canonical["yaw"][p]


def test_roll_damping_is_all_lever_arm():
    """Zeroing the hub spring moment leaves roll unchanged and makes pitch equal roll: BEM's
    roll damping is the lever-arm thrust differential (the k_z channel), and the pitch excess
    comes only from the vehicle-specific flapping fits."""
    model = brd.load("canonical")
    W_h = brd.hover_speed(model)
    with_spring = [-brd.wrench(model, w, W_h)[1][a] / 13.0
                   for a, w in ((0, (13.0, 0, 0)), (1, (0, 13.0, 0)))]
    model[1]["k_spring"] = 0.0
    no_spring = [-brd.wrench(model, w, W_h)[1][a] / 13.0
                 for a, w in ((0, (13.0, 0, 0)), (1, (0, 13.0, 0)))]
    assert no_spring[0] == pytest.approx(with_spring[0], rel=1e-9)
    assert no_spring[1] == pytest.approx(no_spring[0], rel=1e-6)
    assert with_spring[1] - with_spring[0] == pytest.approx(0.00083, rel=0.05)


# ---- RACER_5IN reference vehicle and F-30 (tools/identify_neurobem.py) ----

RACER_JSON = pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata" / \
    "neurobem_racer.json"


def test_racer_matches_frozen_identification():
    doc = json.loads(RACER_JSON.read_text())
    fitted = doc["force_fit"]
    assert RACER_5IN["ct2"] == [pytest.approx(fitted["c_T"], rel=1e-9)] * 4
    assert RACER_5IN["k_d"] == pytest.approx(fitted["k_d"], rel=1e-9)
    assert RACER_5IN["c_D"] == [0.0, 0.0, pytest.approx(fitted["c_Dz"], rel=1e-9)]
    lo, hi = doc["ci95"]["c_Dxy"]
    assert lo < 0.0 < hi  # horizontal frame drag is not distinguishable from zero
    assert RACER_5IN["k_z"] == pytest.approx(doc["k_z_bem"], rel=1e-9)
    assert RACER_5IN["mass"] == doc["fixed"]["mass"]
    rmse = doc["held_out_rmse"]["identified (RACER_5IN)"]
    assert rmse["F_xy"] < 0.6 and rmse["F_z"] < 1.2
    assert max(doc["envelope"]["body_rate_max"]) == pytest.approx(13.1, abs=0.1)


def test_racer_k_z_reproduces_from_bem():
    """k_z needs no flight data: it is BEM's roll damping at 13 rad/s on the racer's arm."""
    W_h = brd.hover_speed(brd.load("canonical"))
    a = RACER_5IN["rotor_pos"][0][0]
    roll = brd.damping("canonical", (13.0,))["roll"][13.0]
    assert roll / (4 * W_h * a * a) == pytest.approx(RACER_5IN["k_z"], rel=1e-6)


def test_f30_moments_do_not_close():
    doc = json.loads(RACER_JSON.read_text())
    geometric = 0.13 / math.sqrt(2)
    assert doc["implied_roll_arm"] < 0.2 * geometric


# ---- source triage 2026-11-15: libfpv / SimITL and rl-tools l2f (tools/sim_source_triage.py) ----

def test_rotor_drag_prior_overstates_measured_racer():
    """libfpv's k_d = 0.6·√(ρ·A·c_T/2) against the racer's measured k_d (κ = 0.40)."""
    kappa = triage.rotor_drag_kappa(RACER_5IN)
    assert kappa == pytest.approx(0.397, abs=0.002)
    assert 0.6 / kappa == pytest.approx(1.51, abs=0.02)
    assert triage.rotor_drag_kappa(triage.CRAZYFLIE) == pytest.approx(0.217, abs=0.002)


def test_pitch_thrust_closure_fails_against_executed_table():
    """libfpv's C_T(J) = C_T0·(1 − (J/J0)²), J0 = 0.9·pitch/D, puts zero thrust where the
    executed APC 9x4.5E table still gives half the static value."""
    e = triage.pitch_closure_error()
    assert e["J0"] == pytest.approx(0.45)
    assert e["rms"] == pytest.approx(0.288, abs=0.005)
    assert e["worst"] == pytest.approx(0.49, abs=0.01)
    assert e["table_CT_at_J0"] == pytest.approx(0.50, abs=0.02)


def test_gust_gradient_is_minor():
    """Per-rotor gust differences (libfpv patch 0017; the multirotor form of the skipped
    Dryden angular gusts): under 1 % of roll authority up to severe turbulence at 2 m."""
    assert triage.gust_gradient(RACER_5IN, 5.0, 10.0)["roll_accel_rms"] == \
        pytest.approx(0.59, abs=0.01)
    worst = triage.gust_gradient(RACER_5IN, triage.W20_LEVELS["severe"], 2.0)
    assert worst["roll_accel_rms"] == pytest.approx(4.63, abs=0.05)
    assert worst["roll_accel_rms"] < 0.01 * worst["roll_authority"]
    cf = triage.gust_gradient(triage.CRAZYFLIE, triage.W20_LEVELS["severe"], 2.0)
    assert cf["roll_accel_rms"] < 0.01 * cf["roll_authority"]


def test_simitl_propwash_and_vibration_sizes():
    """SimITL's hand-tuned constants at the INTAKE operating points (sizes, not measurements)."""
    _, v_h = triage.hover(RACER_5IN)
    assert v_h == pytest.approx(7.66, abs=0.01)
    assert triage.simitl_propwash_loss(0.5 * v_h) == pytest.approx(0.21, abs=0.005)
    assert triage.simitl_propwash_loss(2 * v_h) == pytest.approx(0.84, abs=0.005)
    assert triage.simitl_propwash_loss(2 * v_h, cos_into_wake=0.1) == 0.0
    vib = triage.simitl_vibration()
    assert vib["full"] / vib["white"] == pytest.approx(36, abs=1)
    assert vib["hover"] / vib["white"] == pytest.approx(3.3, abs=0.1)
    assert triage.simitl_angular_damping_fraction(200.0) < 0.005
