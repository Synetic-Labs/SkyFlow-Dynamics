"""
Total body wrench assembly: forces and moments in the body frame.

    F_B = Σᵢ (T_i·ê_i + H_i)  +  D_parasitic + D_linear + D_climb
    M_B = Σᵢ r_i × (T_i·ê_i + H_i)  +  Σᵢ (−s_i)·Q_i·ê_i  +  Σᵢ M_flap,i  +  M_rotor-inertia

with D_linear = −diag(c_L)·v_a the lumped (Faessler-form) linear drag. ⚠ c_L and the
per-rotor H-force coefficients {k_d, k_z} lump the same physics — identify a vehicle against
ONE of the two, never both (no structural guard exists; c_L defaults to zero).

with T_i the (aero-corrected) thrust magnitude along the per-rotor unit axis ê_i (default ẑ),
H_i the rotor-drag force, Q_i the drag-torque magnitude, and s_i the physical spin sign — the
aerodynamic yaw torque on the airframe opposes the rotor spin.

Order of aero corrections on the thrust magnitude (matches the verified reference exactly):
base polynomial → × (1 + k_angle·α + k_hor·μ) → + translational lift. The AoA factor scales
only the base thrust; the lift term adds after.
"""

import sympy as sp

from skyflow_dynamics.spec import rotor_aero
from skyflow_dynamics.spec.frames import EZ, cross


def body_wrench(w: sp.Matrix, W: sp.Matrix, v_a: sp.Matrix, p) -> tuple:
    """
    (F_B, M_B) from body rates w, rotor speeds W, and CoM body airspeed v_a, excluding the
    rotor-inertia moments (those need Ω̇ — see rotor_inertia_moment) and gravity (applied in
    world frame by the rigid-body equations). p is a spec.symbols.Params.
    """
    n = p.n
    Ws = W.flat()
    W_bar = sum(Ws) / n
    factor = rotor_aero.aoa_thrust_factor(v_a, W_bar, p.r_prop, p.k_angle, p.k_hor)

    F = rotor_aero.parasitic_drag(v_a, p.c_D) + rotor_aero.linear_drag(v_a, p.c_L) \
        + rotor_aero.vertical_climb_drag(v_a, p.k_v2)
    M = sp.zeros(3, 1)
    for i in range(n):
        v_i = rotor_aero.local_airspeed(v_a, w, p.rotor_pos[i])
        T_i = rotor_aero.thrust_magnitude(Ws[i], p.ct0[i], p.ct1[i], p.ct2[i]) * factor \
            + rotor_aero.translational_lift(v_i, p.k_h)
        H_i = rotor_aero.rotor_drag_force(Ws[i], v_i, p.k_d, p.k_z)
        Q_i = rotor_aero.torque_magnitude(Ws[i], p.cq0[i], p.cq1[i], p.cq2[i])

        thrust_vec = T_i * p.axis[i]
        F += thrust_vec + H_i
        M += cross(p.rotor_pos[i], thrust_vec + H_i)          # thrust/drag moments
        M += -p.spin[i] * Q_i * p.axis[i]                      # yaw reaction (opposes spin)
        M += rotor_aero.flapping_moment(Ws[i], v_i, p.k_flap)  # blade flapping
    return F, M


def rotor_inertia_moment(w: sp.Matrix, W: sp.Matrix, W_dot: sp.Matrix, p) -> sp.Matrix:
    """
    Moments from the angular momentum of the spinning rotors, h = I_rot·(Σᵢ s_i·Ω_i)·ẑ:

        M_gyro     = −ω × h                      (roll/pitch precession)
        M_reaction = −I_rot·(Σᵢ s_i·Ω̇_i)·ẑ      (yaw reaction to rotor acceleration)

    Both vanish for balanced counter-rotating rotors at constant speed. The x and y components
    of M_gyro must carry opposite signs (−ω×h = (−h_z·ω_y, +h_z·ω_x, 0)) — a known failure mode
    in reference sims (Crazyflow's gyro roll-row sign was flipped; finding F-3, fixed upstream
    by their PR #86). Derivation: τ_body = −d/dt(h)|inertial = −ḣ_body − ω × h.
    ⚠ Assumes every rotor spins about ẑ_B: unlike the thrust/yaw terms, this term does NOT
    follow a tilted per-rotor axis ê_i (matching the verified references, which share the
    restriction). For ê_i ≠ ẑ with I_rot ≠ 0 the momentum would be h = I_rot·Σ s_i·Ω_i·ê_i.
    """
    h = p.I_rot * sum(p.spin[i] * W[i] for i in range(p.n)) * EZ
    reaction = -p.I_rot * sum(p.spin[i] * W_dot[i] for i in range(p.n)) * EZ
    return -cross(w, h) + reaction


def rotor_imbalance_wrench(W: sp.Matrix, theta: sp.Matrix, U: sp.Matrix, p) -> tuple:
    """
    (F_B, M_B) from rotor mass imbalance — the source of the rotation-frequency lines in the
    accelerometer and gyro. A rotor whose centre of mass sits off its axis (imbalance
    U_i = m_u·r_u, kg·m) pulls on its shaft with a force that turns with the rotor:

        F_i = U_i · Ω_i² · (cos θ_i, sin θ_i, 0)        body frame, in the rotor plane
        F_B = Σᵢ F_i ,   M_B = Σᵢ r_i × F_i

    θ_i is the rotor's phase about ẑ_B (θ̇_i = s_i·Ω_i — a harness-side state; the spin sign
    only sets the sense of rotation). With c_T·Ω² = T the force is (U_i/c_T)·T_i: it scales
    with thrust. Each F_i has constant magnitude and zero mean over a revolution, so the
    term leaves the mean flight path alone and adds lines at Ω_i/2π: an in-plane
    accelerometer amplitude U_i·Ω_i²/m, and — for hubs in the plane of the centre of mass —
    a pure yaw torque of amplitude |r_i|·U_i·Ω_i², i.e. a yaw-rate amplitude
    |r_i|·U_i·Ω_i/J_zz. Roll and pitch lines need a hub height r_i,z ≠ 0.

    Source: Ghalamchi, Jia, Mueller (2020) eq. (4) for F_i and eq. (5) for the accelerometer;
    they assume a rigid body and an accelerometer at the centre of mass, and model no
    moment. The r_i × F_i moment here is plain rigid-body mechanics, not their result.
    ⚠ Rigid-frame, first harmonic only: frame flexibility changed their estimate of one
    damaged propeller by a factor of two between two airframes, and higher harmonics and
    blade-pass aerodynamic loads are not modelled. Assumes rotor planes normal to ẑ_B (like
    rotor_inertia_moment, it does not follow a tilted axis ê_i). Stiff: resolving the line
    needs a step well below 1/Ω.
    """
    F = sp.zeros(3, 1)
    M = sp.zeros(3, 1)
    for i in range(p.n):
        F_i = U[i] * W[i] ** 2 * sp.Matrix([sp.cos(theta[i]), sp.sin(theta[i]), 0])
        F += F_i
        M += cross(p.rotor_pos[i], F_i)
    return F, M
