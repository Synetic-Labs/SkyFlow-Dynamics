"""
SkyFlow-Dynamics symbolic specification.

The math is organized one module per physics domain; `spec.dynamics.statedot` assembles the
canonical continuous-time model  ṡ = f(s, u, w)  with

    state      s = (x_W, v_W, q_WB, ω_B, Ω₁…Ω_n)
    input      u = Ω_c (commanded rotor speeds; command abstractions map onto this)
    exogenous  w = (v_wind_W, F_ext_W, τ_ext_B)

Conventions (see README.md for the full statement):
  world ENU-like (ẑ up, gravity −ẑ) · body FLU (x forward, z up) · quaternion wxyz scalar-first,
  Hamilton, body→world · rotor spin sign s_i about +ẑ_B (yaw reaction torque = −s_i·Q_i) · SI units.

`spec.registry` lists every term with its tier (verified/candidate), sources, and tests.
"""

from spec import quaternion, frames, symbols, motor, rotor_aero, wrench, rigid_body  # noqa: F401
from spec import dynamics, simplified, discretization, sensors, parameters, registry  # noqa: F401
