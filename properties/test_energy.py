"""Dissipativity of the drag terms — drag must never inject energy.
(Blade flapping and thrust-airspeed corrections are energy-coupled to the rotor and are
NOT sign-definite in isolation; they are deliberately excluded.)"""

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import rotor_aero


def test_parasitic_drag_power_nonpositive_symbolic():
    va1, va2, va3 = sp.symbols("va_1 va_2 va_3", real=True)
    cD1, cD2, cD3 = sp.symbols("cD_1 cD_2 cD_3", nonnegative=True)
    va, cD = sp.Matrix([va1, va2, va3]), sp.Matrix([cD1, cD2, cD3])
    power = (rotor_aero.parasitic_drag(va, cD).T * va)[0, 0]
    # P = −‖v‖·(cD₁v₁² + cD₂v₂² + cD₃v₃²): the negation of a product of nonnegatives.
    speed = sp.sqrt(va1**2 + va2**2 + va3**2)
    quad = cD1*va1**2 + cD2*va2**2 + cD3*va3**2
    assert sp.simplify(power + speed * quad) == 0


def test_linear_drag_power_nonpositive_symbolic():
    va1, va2, va3 = sp.symbols("va_1 va_2 va_3", real=True)
    cL1, cL2, cL3 = sp.symbols("cL_1 cL_2 cL_3", nonnegative=True)
    va, cL = sp.Matrix([va1, va2, va3]), sp.Matrix([cL1, cL2, cL3])
    power = (rotor_aero.linear_drag(va, cL).T * va)[0, 0]
    assert sp.simplify(power + cL1*va1**2 + cL2*va2**2 + cL3*va3**2) == 0


def test_rotor_drag_power_nonpositive_symbolic():
    vi1, vi2, vi3 = sp.symbols("vi_1 vi_2 vi_3", real=True)
    vi = sp.Matrix([vi1, vi2, vi3])
    W, kd, kz = sp.symbols("W k_d k_z", nonnegative=True)
    power = (rotor_aero.rotor_drag_force(W, vi, kd, kz).T * vi)[0, 0]
    assert sp.simplify(power + W * (kd * (vi1**2 + vi2**2) + kz * vi3**2)) == 0


def test_total_drag_power_sampled():
    # Total mechanical power of {parasitic + per-rotor H-forces at their hubs} ≤ 0 across
    # random states — the sampled version of the two proofs above, through the full assembly.
    rng = np.random.default_rng(42)
    kd, kz = 1.02506e-6, 7.553e-7
    cD = np.array([0.05, 0.06, 0.08])
    r = np.array([[0.03, 0.03, 0], [0.03, -0.03, 0], [-0.03, -0.03, 0], [-0.03, 0.03, 0]])
    for _ in range(200):
        va = rng.uniform(-8, 8, 3)
        w = rng.uniform(-4, 4, 3)
        W = rng.uniform(0, 2500, 4)
        P = -np.linalg.norm(va) * (cD * va) @ va
        for i in range(4):
            vi = va + np.cross(w, r[i])
            H = -W[i] * np.diag([kd, kd, kz]) @ vi
            P += H @ vi
        assert P <= 1e-15


def test_climb_drag_power_nonpositive_on_axis():
    # −k_v2·v_z·|v_z| along ẑ: power against the vertical airspeed is −k_v2·v_z²·|v_z| ≤ 0.
    vz = sp.Symbol("vz", real=True)
    va = sp.Matrix([0, 0, vz])
    kv2 = sp.Symbol("k_v2", nonnegative=True)
    power = (rotor_aero.vertical_climb_drag(va, kv2).T * va)[0, 0]
    assert sp.simplify(power + kv2 * vz**2 * sp.Abs(vz)) == 0
