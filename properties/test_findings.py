"""Findings from use (REFERENCES.md, "Findings from use"): each finding that a registry entry
cites as 'derived' evidence is reproduced here, so the numbers in the ledger stay tied to code.

F-29 — body-rate damping of the verified BEM rotor model (tools/bem_rate_damping.py): with the
canonical hub velocity v + ω×r it damps and stiffens with rate; with agilib's frame mixing
(F-24) roll is anti-damped and yaw damping all but vanishes.

RACER_5IN / F-30 — the reference vehicle's values match the frozen identification
(tools/identify_neurobem.py → golden/checkdata/neurobem_racer.json); its BEM-derived k_z
reproduces without the flight data; the flight data's moments do not close."""

import json
import math
import pathlib

import pytest

from skyflow_dynamics.spec.parameters import RACER_5IN
from tools import bem_rate_damping as brd

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
