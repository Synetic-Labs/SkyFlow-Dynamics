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
    Source("jsbsim", "JSBSim flight dynamics engine (FGPropeller, FGRotor, FGStandardAtmosphere), "
           "commit 9a0b028", "https://github.com/JSBSim-Team/jsbsim"),
    Source("mccormick", "McCormick — Aerodynamics, Aeronautics, and Flight Mechanics, 1st ed. "
           "(momentum-theory induced velocity, Eq. 6.15)"),
    Source("sanchez2017", "Sanchez-Cuevas, Heredia, Ollero — Characterization of the Aerodynamic "
           "Ground Effect and Its Influence in Multirotor Control, Int. J. Aerospace Eng. 2017, "
           "doi 10.1155/2017/1823056"),
    Source("cheeseman1955", "Cheeseman & Bennett — The Effect of the Ground on a Helicopter "
           "Rotor in Forward Flight, ARC R&M 3021, 1955"),
    Source("pybullet_drones", "utiasDSL/gym-pybullet-drones BaseAviary (_groundEffect, "
           "_downwash) + cf2x.urdf identified constants",
           "https://github.com/utiasDSL/gym-pybullet-drones"),
    Source("jain2019", "Jain, Fortmuller, Byun, Makiharju, Mueller — Modeling of aerodynamic "
           "disturbances for proximity flight of multirotors, ICUAS 2019, Eqs. (1)-(8)"),
    Source("bangura", "Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony "
           "(ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for "
           "Quadrotors (arXiv:1601.00733)"),
    Source("kai2017", "Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors "
           "exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13)"),
    Source("rotors_px4", "ethz-asl/rotors_simulator gazebo_motor_model.cpp + "
           "PX4/PX4-SITL_gazebo-classic variant (signed rolling moment)",
           "https://github.com/ethz-asl/rotors_simulator"),
    Source("chen2006", "Chen & Rincon-Mora — Accurate Electrical Battery Model Capable of "
           "Predicting Runtime and I-V Performance, IEEE Trans. Energy Conversion 21(2), 2006"),
    Source("crazyflie_fw", "bitcraze/crazyflie-firmware — motors.c "
           "motorsCompensateBatteryVoltage + platform_defaults_cf2.h (master and tag 2022.01)",
           "https://github.com/bitcraze/crazyflie-firmware"),
    Source("gazebo_battery", "gazebosim/gz-sim LinearBatteryPlugin.cc (linear OCV + internal "
           "resistance + current low-pass)", "https://github.com/gazebosim/gz-sim"),
    Source("mil8785c", "MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: "
           "Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust"),
    Source("ussa1976", "US Standard Atmosphere 1976 (NASA-TM-X-74335): layered T(h), P(h), "
           "ideal-gas density"),
    Source("neurobem", "Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid "
           "Aerodynamic Quadrotor Model, RSS 2021 (Agilicious high-fidelity BEM option)"),
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

    # ---------------- candidates: ground effect / downwash ----------------
    Term("ground_effect_cheeseman_bennett", "candidate", "rotor_aero",
         "Single-rotor IGE thrust ratio 1/(1−(R/4z)²); forward-flight washout variant",
         "spec.ground_effect.cheeseman_bennett, spec.ground_effect.cheeseman_bennett_forward",
         ("cheeseman1955", "sanchez2017"), (),
         ("properties/test_candidates.py",),
         "Valid 0.5 ≤ z/R ≤ 2; singular at z = R/4 (clamp). Under-predicts for multirotors."),
    Term("ground_effect_sanchez_cuevas", "candidate", "rotor_aero",
         "Quadrotor IGE with mirrored-rotor images + fountain body-lift (K_b ≈ 2)",
         "spec.ground_effect.sanchez_cuevas", ("sanchez2017",), (),
         ("properties/test_candidates.py",),
         "Significant to z ≈ 5R. Reduces to Cheeseman-Bennett as d, b → ∞, K_b → 0."),
    Term("ground_effect_pybullet", "candidate", "rotor_aero",
         "Per-rotor additive increment ΔT = T·G·(R/4z)² (linearized CB, identified G)",
         "spec.ground_effect.pybullet_ground_effect", ("pybullet_drones",), (),
         ("properties/test_candidates.py",),
         "G = 11.37 identified for CF2 (fountain amplification folded in); needs height clip."),
    Term("downwash_pybullet", "candidate", "disturbance",
         "Inter-vehicle Gaussian downwash force fit (DSL/SiQi Zhou)",
         "spec.ground_effect.pybullet_downwash_force", ("pybullet_drones",), (),
         ("properties/test_candidates.py",),
         "CF2-specific fit, thrust-independent, untrustworthy below Δz ≈ 0.7 m."),
    Term("downwash_jain_jet", "candidate", "disturbance",
         "Turbulent-jet wake velocity field + frame drag + per-rotor thrust loss",
         "spec.ground_effect.jain_wake_velocity", ("jain2019",), (),
         ("properties/test_candidates.py",),
         "Thrust-scaled, gives force AND moment; ZEF only (z > 3L). Route wake velocity "
         "through ONE rotor-inflow path to avoid double-counting with AoA thrust terms."),

    # ---------------- candidates: inflow / propeller / atmosphere ----------------
    Term("momentum_induced_velocity", "candidate", "rotor_aero",
         "Actuator-disk induced velocity: hover v_h = √(T/2ρA); sign-safe axial closed form",
         "spec.inflow.hover_induced_velocity, spec.inflow.induced_velocity_axial",
         ("mccormick", "jsbsim", "bangura"), (),
         ("properties/test_candidates.py",),
         "The physical input behind ground effect / downwash / climb corrections."),
    Term("oblique_momentum_thrust", "candidate", "rotor_aero",
         "Nonlinear T(airspeed): T = 2ρA·v_i·U, U = √(Vx²+Vy²+(v_i−Vz)²) (implicit v_i)",
         "spec.inflow.oblique_momentum_thrust", ("bangura",), (),
         ("properties/test_candidates.py",),
         "Principled model the identified k_v2/k_angle/k_hor terms linearize. VRS validity "
         "band excluded (descent 0.5–2 v_h; spec.inflow VRS constants)."),
    Term("dynamic_inflow_lag", "candidate", "rotor_aero",
         "First-order induced-inflow lag to Glauert equilibrium, τ ≈ 16/(γΩ); exact-exp step",
         "spec.inflow.dynamic_inflow_lag", ("jsbsim",), (),
         ("properties/test_candidates.py",),
         "Same operator-split pattern as the verified motor exact-exp discretization."),
    Term("advance_ratio_tables", "candidate", "rotor_aero",
         "T = C_T(J)·ρ·n²·D⁴, P = C_P(J)·ρ·n³·D⁵ with measured tables; windmilling via sign",
         "spec.atmosphere.advance_ratio, spec.atmosphere.propeller_thrust",
         ("jsbsim", "neurobem"), (),
         ("properties/test_candidates.py",),
         "Generalizes the polynomial T(Ω) (a fixed-J slice). J uses axial inflow only; "
         "UIUC/APC databases supply tables for small UAV props. NeuroBEM's BEM model is the "
         "higher-fidelity per-element variant."),
    Term("isa_atmosphere", "candidate", "environment",
         "USSA-1976 layered T(h), P(h); ρ = P/RT; thrust/torque scale linearly with ρ",
         "spec.atmosphere.temperature_troposphere, spec.atmosphere.pressure_gradient_layer, "
         "spec.atmosphere.density, spec.atmosphere.speed_of_sound",
         ("ussa1976", "jsbsim"), (),
         ("properties/test_candidates.py",),
         "Verified-tier coefficients absorb ρ at identification altitude — scale by ρ/ρ_ident."),

    # ---------------- candidates: rotor aero extensions ----------------
    Term("rolling_moment", "candidate", "rotor_aero",
         "Per-rotor rolling moment −Ω·s·μ_R·v_⊥ (advancing/retreating dissymmetry)",
         "spec.rotor_aero.rolling_moment", ("rotors_px4", "kai2017"), (),
         ("properties/test_candidates.py",),
         "RotorS omits the spin sign (bug); PX4 variant adopted. Cancels for balanced pairs."),
    Term("flapping_force_body_rate", "candidate", "rotor_aero",
         "Kai Eq. (10) flapping force incl. spin-signed lateral and body-rate damping terms",
         "spec.rotor_aero.flapping_force_kai", ("kai2017", "faessler2018"), (),
         ("properties/test_candidates.py",),
         "The published basis for the lumped linear drag; adds rotor-plane damping (B·ω) "
         "absent from the verified tier."),

    # ---------------- candidates: motor / battery electrical ----------------
    Term("dc_motor_quasistatic", "candidate", "actuator",
         "J_r·Ω̇ = (K_q/R_a)(V_m − K_e·Ω) − k_m·Ω² − b·Ω; τ_m bridge via linearization",
         "spec.motor_electrical.dc_motor_speed_dynamics", ("bangura", "jsbsim"), (),
         ("properties/test_candidates.py",),
         "τ_m = J_r/(K_qK_e/R_a + b + 2k_mΩ₀) connects to the verified first-order lag."),
    Term("esc_battery_coupling", "candidate", "actuator",
         "V_m = u·V_batt; battery-coupled steady-state speed (√-like in u·V_batt)",
         "spec.motor_electrical.esc_mean_voltage, spec.motor_electrical.steady_state_speed",
         ("bangura", "crazyflie_fw"), (),
         ("properties/test_candidates.py",),
         "The physical origin of the verified throttle curve's √(k·u²+(1−k)·u) shape."),
    Term("crazyflie_battery_compensation", "candidate", "actuator",
         "Firmware cubic T(v_m) with Cardano inversion (+ legacy quadratic), PWM/V_batt scaling",
         "spec.motor_electrical.crazyflie_thrust_from_voltage", ("crazyflie_fw",), (),
         ("properties/test_candidates.py",),
         "Identified C0–C3 per prop variant; thrust held constant under sag by design."),
    Term("thevenin_battery", "candidate", "actuator",
         "OCV(SoC) + series R(SoC) + two RC branches; Gazebo linear model as special case",
         "spec.motor_electrical.thevenin_battery, spec.motor_electrical.chen_ocv",
         ("chen2006", "gazebo_battery"), (),
         ("properties/test_candidates.py",),
         "Load coupling i = Σ(u·V_batt − K_e·Ω)/R_a closes the sag loop physically."),

    # ---------------- candidates: wind / turbulence ----------------
    Term("dryden_turbulence", "candidate", "environment",
         "Dryden forming filters H_u/H_v/H_w + low-altitude scale/intensity closures",
         "spec.wind.dryden_filter_u, spec.wind.dryden_filter_vw, "
         "spec.wind.dryden_low_altitude_scales", ("mil8785c",), (),
         ("properties/test_candidates.py",),
         "⚠ discrete driving noise must be N(0, π/dt) for the published gains; 8785C vs "
         "1797 length-scale factor-of-2 trap. Low-altitude fit is in FEET."),
    Term("von_karman_turbulence", "candidate", "environment",
         "von Kármán spectra (5/6, 11/6 exponents) + standard rational filter approximations",
         "spec.wind.von_karman_psd_u", ("mil8785c",), (),
         ("properties/test_candidates.py",),
         "Measurement-preferred; no exact finite filter."),
    Term("discrete_gust", "candidate", "environment",
         "1-cosine discrete gust ramp per axis",
         "spec.wind.one_minus_cosine_gust", ("mil8785c",), (),
         ("properties/test_candidates.py",)),

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
