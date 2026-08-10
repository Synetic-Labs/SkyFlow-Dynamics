"""
The term registry: every physics term in the spec, with tier, provenance, and tests.

Tiers:
  verified  — cross-validated against a runnable reference implementation; covered by golden
              vectors and property tests in this repo.
  candidate — credible published model, symbolically checked and cited; awaiting numeric
              validation against a runnable reference.

Domains: rigid_body · actuator · rotor_aero · frame_aero · sensor · disturbance ·
discretization · differentiation · harness (not physics — timing/stateful machinery, listed so
nothing is lost).

The INTAKE.md protocol appends candidates here; promotion to verified requires golden vectors.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str
    citation: str
    url: str = ""


@dataclass(frozen=True)
class Term:
    key: str
    tier: str            # 'verified' | 'candidate'
    domain: str
    summary: str
    expression: str      # dotted path to the defining spec function(s)
    sources: tuple
    parameters: tuple = ()
    tests: tuple = ()
    notes: str = ""


SOURCES = {s.key: s for s in [
    Source("rotorpy", "Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator "
           "with Aerodynamics for Education and Research (arXiv:2306.04485); reference "
           "implementation branch research-additions", "https://github.com/spencerfolk/rotorpy"),
    Source("mahony2012", "Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, "
           "Estimation, and Control of Quadrotor, IEEE RAM 2012"),
    Source("crazyflow", "Crazyflow first-principles dynamics "
           "(crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml",
           "https://github.com/learnsyslab/crazyflow"),
    Source("skydreamer", "SkyDreamer (arXiv:2510.14783) + reference implementation "
           "embodied/envs/skydreamer.py", "https://github.com/The-Real-Thisas/dreamerv3"),
    Source("flightning", "Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual "
           "Features Using Differentiable Simulation, ICRA 2025",
           "https://github.com/uzh-rpg/rpg_flightning"),
    Source("forster2015", "Förster — System Identification of the Crazyflie 2.0 Nano "
           "Quadrocopter, ETH Zürich, 2015"),
    Source("graf", "Graf — Quaternions and Dynamics (quaternion kinematics)"),
    Source("eschmann2024", "Eschmann et al. — Data-Driven System Identification of Quadrotors "
           "Subject to Motor Delays, 2024 (arXiv:2404.07837)"),
    Source("faessler2018", "Faessler, Franchi, Scaramuzza — Differential Flatness of Quadrotor "
           "Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories, "
           "IEEE RA-L 2018"),
]}


TERMS = (
    # ---------------- rigid body ----------------
    Term("newton_euler", "verified", "rigid_body",
         "Rigid-body translational + rotational EOM with full inertia matrix",
         "spec.rigid_body.translational, spec.rigid_body.rotational",
         ("rotorpy", "mahony2012"), ("mass", "grav", "inertia"),
         ("properties/test_rigid_body.py", "properties/test_golden.py")),
    Term("quaternion_kinematics", "verified", "rigid_body",
         "q̇ = ½ q ⊗ (0, ω); wxyz scalar-first Hamilton, body→world",
         "spec.quaternion.kinematics", ("graf", "rotorpy"),
         (), ("properties/test_quaternion.py",),
         "Unit norm preserved exactly by the continuous equation; discrete integrators "
         "renormalize post-step (harness)."),

    # ---------------- actuator ----------------
    Term("motor_first_order_lag", "verified", "actuator",
         "Ω̇ = (Ω_c − Ω)/τ_m", "spec.motor.first_order_lag",
         ("rotorpy", "forster2015"), ("tau_m",),
         ("properties/test_motor.py", "properties/test_golden.py")),
    Term("motor_asymmetric_lag", "verified", "actuator",
         "Separate spin-up/spin-down linear+quadratic rates",
         "spec.motor.asymmetric_lag", ("crazyflow",), ("ka1", "ka2", "kd1", "kd2"),
         ("properties/test_motor.py", "properties/test_golden.py"),
         "Crazyflow coefficients are RPM-units — convert (60/2π per Ω power). Reduces to "
         "first-order at (1/τ, 0, 1/τ, 0)."),
    Term("motor_exact_exp_discretization", "verified", "discretization",
         "Closed-form Ω(dt) = Ω_c + (Ω₀−Ω_c)e^(−dt/τ); operator-split from the RK stages",
         "spec.motor.exact_exp_step", ("flightning",), ("tau_m",),
         ("properties/test_motor.py", "properties/test_golden.py"),
         "Unconditionally stable; per-step gradient factor e^(−dt/τ) ∈ (0,1). Linear lag only."),
    Term("throttle_curve", "verified", "actuator",
         "Ω_c = (Ω_max−Ω_min)√(k·u² + (1−k)u) + Ω_min",
         "spec.motor.throttle_to_speed", ("skydreamer",), (),
         ("properties/test_motor.py",),
         "Identified k = 0.5 for a 5-inch racer (Ω_min 341.75, Ω_max 3100 rad/s)."),
    Term("pwm_quantization", "verified", "actuator",
         "Throttle snapped to the integer PWM grid",
         "spec.motor.pwm_quantize", ("crazyflow",), (),
         ("properties/test_motor.py",),
         "Piecewise-constant — exclude from differentiable paths."),
    Term("battery_voltage_speed_cap", "verified", "actuator",
         "Supply voltage → achievable Ω_max (linear map); slow drift over discharge",
         "spec.motor.voltage_to_rpm", ("crazyflow", "skydreamer"), (),
         ("properties/test_motor.py",),
         "Battery state evolution (SoC, sag dynamics) is harness-side."),
    Term("rotor_thrust_polynomial", "verified", "rotor_aero",
         "T_i = ct0 + ct1·Ω + ct2·Ω² per rotor (per-rotor coefficient asymmetry supported)",
         "spec.rotor_aero.thrust_magnitude", ("rotorpy", "crazyflow", "skydreamer"),
         ("ct0", "ct1", "ct2"), ("properties/test_wrench.py", "properties/test_golden.py"),
         "Crazyflow identifies RPM-unit polynomials; SkyDreamer's k_w is mass-normalized "
         "(multiply by m; finding F-4)."),
    Term("rotor_torque_polynomial", "verified", "rotor_aero",
         "Q_i = cq0 + cq1·Ω + cq2·Ω²; yaw torque on airframe = −s_i·Q_i·ê_i (opposes spin)",
         "spec.rotor_aero.torque_magnitude", ("rotorpy", "crazyflow"),
         ("cq0", "cq1", "cq2", "spin"), ("properties/test_wrench.py", "properties/test_golden.py"),
         "⚠ RotorPy's rotor_directions = torque sign = −spin (finding F-6)."),
    Term("thrust_axis_misalignment", "verified", "rotor_aero",
         "Per-rotor unit thrust axis ê_i ≠ ẑ from assembly tolerance → parasitic forces/moments",
         "spec.wrench.body_wrench", ("rotorpy",), ("axis",),
         ("properties/test_wrench.py", "properties/test_golden.py")),
    Term("rotor_inertia_moments", "verified", "rotor_aero",
         "Gyroscopic precession −ω×h and yaw reaction −I_rot·Σ s_i Ω̇_i·ẑ",
         "spec.wrench.rotor_inertia_moment", ("rotorpy", "crazyflow", "skydreamer"),
         ("I_rot", "spin"), ("properties/test_wrench.py", "properties/test_golden.py"),
         "Signs re-derived from τ = −d/dt(h); Crazyflow's gyro-x sign is flipped (finding F-3, "
         "confirmed against their running code — our sign wins)."),

    # ---------------- aerodynamics ----------------
    Term("rotor_drag_hforce", "verified", "rotor_aero",
         "H_i = −Ω_i·diag(k_d,k_d,k_z)·v_i at each hub, v_i incl. ω×r_i lever arm",
         "spec.rotor_aero.rotor_drag_force", ("rotorpy", "mahony2012", "skydreamer"),
         ("k_d", "k_z"), ("properties/test_energy.py", "properties/test_golden.py"),
         "SkyDreamer's lumped −k_x·v·ΣΩ is this summed over rotors (k_d = m·k_x; F-4)."),
    Term("blade_flapping_moment", "verified", "rotor_aero",
         "M_flap,i = −k_flap·Ω_i·(v_i × ẑ); pitch-up in forward flight",
         "spec.rotor_aero.flapping_moment", ("rotorpy", "mahony2012"), ("k_flap",),
         ("properties/test_wrench.py", "properties/test_golden.py")),
    Term("translational_lift", "verified", "rotor_aero",
         "ΔT_i = k_h·(v_i,x² + v_i,y²)",
         "spec.rotor_aero.translational_lift", ("rotorpy",), ("k_h",),
         ("properties/test_golden.py",),
         "Small-airspeed linearization of the AoA/advance-ratio model — mutually exclusive "
         "with k_angle/k_hor (validation rule)."),
    Term("aoa_advance_ratio_thrust", "verified", "rotor_aero",
         "T × (1 + k_angle·atan2(v_az, rΩ̄) + k_hor·atan2(‖v_axy‖, rΩ̄))",
         "spec.rotor_aero.aoa_thrust_factor", ("skydreamer",),
         ("k_angle", "k_hor", "r_prop"), ("properties/test_golden.py",),
         "Identified to racing speeds: k_angle 3.145, k_hor 7.245 (mass-normalized k_w; F-4)."),
    Term("vertical_climb_drag", "verified", "rotor_aero",
         "−k_v2·v_az·|v_az|·ẑ collective at CoM",
         "spec.rotor_aero.vertical_climb_drag", ("skydreamer",), ("k_v2",),
         ("properties/test_golden.py",)),
    Term("linear_drag", "verified", "frame_aero",
         "Lumped linear body-frame drag F = −diag(c_L)·v_a (Faessler differential-flatness form)",
         "spec.rotor_aero.linear_drag", ("faessler2018", "crazyflow"), ("c_L",),
         ("properties/test_energy.py", "properties/test_golden.py"),
         "Ω-independent lumping of the per-rotor H-force; identify against c_L OR k_d, "
         "not both. Crazyflow stores the negated diagonal (drag_matrix = −diag(c_L))."),
    Term("parasitic_drag", "verified", "frame_aero",
         "D = −‖v_a‖·diag(c_D)·v_a at CoM",
         "spec.rotor_aero.parasitic_drag", ("rotorpy",), ("c_D",),
         ("properties/test_energy.py", "properties/test_golden.py"),
         "⚠ ‖v‖-scaled, NOT per-axis |v_k|·v_k (SkyDreamer's form) — structurally different; "
         "don't transplant coefficients between the two."),

    # ---------------- disturbances / inputs ----------------
    Term("external_wrench_inputs", "verified", "disturbance",
         "Exogenous F_ext (world) and τ_ext (body) enter the EOM directly",
         "spec.rigid_body.translational, spec.rigid_body.rotational",
         ("skydreamer", "crazyflow"), (), ("properties/test_golden.py",),
         "The two-band resample-and-hold schedule that drives these in training (SkyDreamer "
         "Table III: ±3 m/s²+±3 rad/s² @1 Hz, ±125 rad/s² @90 Hz, ε_u ±0.2) is harness-side."),

    # ---------------- sensors ----------------
    Term("imu_measurement", "verified", "sensor",
         "Specific force + body rate at offset/rotated mount, lever-arm terms in body frame",
         "spec.sensors.imu", ("rotorpy",), (),
         ("properties/test_sensors.py",),
         "Frame-mixing defects F-1/F-2 found and fixed in the reference; equations here are "
         "the corrected form."),

    # ---------------- differentiable simulation ----------------
    Term("point_mass_surrogate", "verified", "differentiation",
         "Point mass + kinematic attitude; surrogate Jacobian for BPTT via straight-through",
         "spec.simplified.step, spec.simplified.dynamics", ("flightning",), (),
         ("properties/test_simplified.py",)),
    Term("rk4_fixed_step", "verified", "discretization",
         "Classical RK4; the differentiable reference integrator (adaptive solvers are not "
         "cleanly differentiable)", "spec.discretization.rk4_step", ("flightning", "rotorpy"),
         (), ("properties/test_motor.py", "properties/test_golden.py")),

    # ---------------- harness (tracked, not physics) ----------------
    Term("command_transport_delay", "verified", "harness",
         "u_applied(t) = u_cmd(t − t_d); ring buffer of round(t_d/dt) steps",
         "(harness — no symbolic form)", ("skydreamer", "eschmann2024"), (),
         (), "SkyDreamer trains with t_d = 11 ms."),
    Term("control_rate_zoh", "verified", "harness",
         "Controller decides at a lower rate than physics; command zero-order-held between",
         "(harness — no symbolic form)", ("crazyflow",), ()),
    Term("ground_contact_heuristic", "candidate", "harness",
         "Normal-force cancellation + velocity clamps at z ≤ 0 — bookkeeping, not contact physics",
         "(harness — no symbolic form)", ("rotorpy",), (),
         (), "Do not port as physics; a real contact model would be a new candidate."),
)


#: Sources reviewed and rejected, so intake doesn't re-litigate them.
EXCLUSIONS = (
    ("genesis", "Genesis (Genesis-Embodied-AI): props are fixed joints, KF·rpm² force + "
     "KM·rpm² yaw torque only, no gyroscopic effects — strict subset of this spec."),
)


def by_key(key: str) -> Term:
    for t in TERMS:
        if t.key == key:
            return t
    raise KeyError(key)


def validate_registry() -> None:
    """Structural invariants checked by properties/test_registry.py."""
    from spec.parameters import SCHEMA
    keys = [t.key for t in TERMS]
    assert len(keys) == len(set(keys)), "duplicate term keys"
    for t in TERMS:
        assert t.tier in ("verified", "candidate"), t.key
        assert t.sources, f"{t.key} has no sources"
        for s in t.sources:
            assert s in SOURCES, f"{t.key}: unknown source {s}"
        for p in t.parameters:
            assert p in SCHEMA, f"{t.key}: unknown parameter {p}"
        if t.tier == "verified" and t.domain not in ("harness",):
            assert t.tests or t.expression.startswith("(harness"), \
                f"{t.key} is verified but lists no tests"
