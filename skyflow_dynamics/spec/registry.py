"""
The term registry: every physics term in the spec, with tier, provenance, and tests.

Tiers:
  verified  — cross-validated against a runnable reference implementation; covered by golden
              vectors and property tests in this repo.
  candidate — credible published model, symbolically checked and cited; awaiting numeric
              validation against a runnable reference.

Domains: rigid_body · actuator · rotor_aero · frame_aero · sensor · disturbance ·
discretization · differentiation · environment · harness (not physics — timing/stateful
machinery, listed so nothing is lost).

The INTAKE.md protocol appends candidates here; promotion to verified requires golden vectors.
"""

from dataclasses import dataclass


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
    Source("jsbsim", "JSBSim flight dynamics engine (FGPropeller, FGRotor, "
           "FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit "
           "9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 "
           "(golden/generate/gen_jsbsim.py)", "https://github.com/JSBSim-Team/jsbsim"),
    Source("mccormick", "McCormick — Aerodynamics, Aeronautics, and Flight Mechanics, 1st ed. "
           "(momentum-theory induced velocity, Eq. 6.15)"),
    Source("sh79", "Shaughnessy, Deaux, Yenni — Development and Validation of a Piloted "
           "Simulation of a Helicopter and External Sling Load, NASA TP-1285, 1979 (JSBSim "
           "FGRotor's model basis); body-rate flap terms per Amer, NACA TN-2136, 1950"),
    Source("bramwell", "Bramwell — Helicopter Dynamics, 2nd ed., eqns 3.43-3.44 (rotor torque "
           "decomposition: profile + induced/climb components)"),
    Source("talbot1977", "Talbot & Corliss — A Mathematical Force and Moment Model of a UH-1H "
           "Helicopter for Flight Dynamics Simulations, NASA TM-73,254, 1977 (eqn 10a "
           "ground-effect inflow factor)"),
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
           "Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component "
           "eqs. (5)-(19), Kingfisher platform §IV-B", "https://arxiv.org/abs/2106.08015"),
    Source("agilicious", "Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, "
           "GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at "
           "commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 "
           "(Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib "
           "(golden/generate/gen_agilicious.py)",
           "https://github.com/alibabasomeone/agilicious_internal_mine"),
    Source("hoffmann2007", "Hoffmann, Huang, Waslander, Tomlin — Quadrotor Helicopter Flight "
           "Dynamics and Control: Theory and Experiment, AIAA GNC 2007 (VRS empirical quartic; "
           "hinged-blade spring model)"),
    Source("gill2017", "Gill & D'Andrea — Propeller Thrust and Drag in Forward Flight, IEEE "
           "CCTA 2017 (sinusoidal high-incidence lift/drag polars; with Ducard & Hua, CCA 2014)"),
    Source("mathworks_aeroblks", "MathWorks Aerospace Blockset documentation — block-equation "
           "pages only (implementations are proprietary and were not consulted); each block "
           "cites the public standard it implements",
           "https://www.mathworks.com/help/aeroblks/"),
    Source("cr206937", "Yeager — Implementation and Testing of Turbulence Models for the "
           "F18-HARV Simulation, NASA CR-1998-206937, 1998. Pinned-document golden source "
           "(sha256 4f63d46d…, NTRS 19980028448): GUSTMDL ACSL listing + Tables 2-7 run "
           "statistics", "https://ntrs.nasa.gov/citations/19980028448"),
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
         "Crazyflow coefficients are RPM-based; because Ω̇ rescales with Ω, ka1/kd1 carry "
         "over unchanged and ka2/kd2 convert by ×60/2π — i.e. (60/2π)^(Ω-power − 1). Reduces "
         "to first-order at (1/τ, 0, 1/τ, 0)."),
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
         "Signs re-derived from τ = −d/dt(h). Crazyflow's gyro roll-row sign was flipped "
         "(finding F-3, confirmed against their running code); fixed upstream by "
         "learnsyslab/crazyflow PR #86 (merged 2026-07-13) — post-fix Crazyflow golden "
         "vectors now cross-validate this term."),

    # ---------------- aerodynamics ----------------
    Term("rotor_drag_hforce", "verified", "rotor_aero",
         "H_i = −Ω_i·diag(k_d,k_d,k_z)·v_i at each hub, v_i incl. ω×r_i lever arm",
         "spec.rotor_aero.rotor_drag_force", ("rotorpy", "mahony2012", "skydreamer"),
         ("k_d", "k_z"), ("properties/test_energy.py", "properties/test_golden.py"),
         "SkyDreamer's lumped −k_x·v·ΣΩ is this summed over rotors (k_d = m·k_x; F-4)."),
    Term("blade_flapping_moment", "verified", "rotor_aero",
         "M_flap,i = −k_flap·Ω_i·(v_i × ẑ); +M_y (nose-down in FLU) for v_x > 0, k_flap > 0",
         "spec.rotor_aero.flapping_moment", ("rotorpy", "mahony2012"), ("k_flap",),
         ("properties/test_wrench.py", "properties/test_golden.py")),
    Term("translational_lift", "verified", "rotor_aero",
         "ΔT_i = k_h·(v_i,x² + v_i,y²)",
         "spec.rotor_aero.translational_lift", ("rotorpy", "agilicious"), ("k_h",),
         ("properties/test_golden.py", "properties/test_golden_agilicious.py"),
         "Also executed as agilib ModelLinCubDrag's induced_lift_coeff (2026-08-19). "
         "Small-airspeed linearization of the AoA/advance-ratio model — mutually exclusive "
         "with k_angle/k_hor (validation rule)."),
    Term("aoa_advance_ratio_thrust", "verified", "rotor_aero",
         "T × (1 + k_angle·atan2(v_az, rΩ̄) + k_hor·atan2(‖v_axy‖, rΩ̄))",
         "spec.rotor_aero.aoa_thrust_factor", ("skydreamer",),
         ("k_angle", "k_hor", "r_prop"), ("properties/test_golden.py",),
         "Identified to racing speeds: k_angle 3.145, k_hor 7.245 (mass-normalized k_w; F-4). "
         "Spec follows the runnable reference (ENU, mean Ω̄, hypot); the paper's printed "
         "equations differ (NED, ΣΩ, squared numerator) — see the function docstring."),
    Term("vertical_climb_drag", "verified", "rotor_aero",
         "−k_v2·v_az·|v_az|·ẑ collective at CoM",
         "spec.rotor_aero.vertical_climb_drag", ("skydreamer",), ("k_v2",),
         ("properties/test_golden.py",)),
    Term("linear_drag", "verified", "frame_aero",
         "Lumped linear body-frame drag F = −diag(c_L)·v_a (Faessler differential-flatness form)",
         "spec.rotor_aero.linear_drag", ("faessler2018", "crazyflow", "agilicious"),
         ("c_L",),
         ("properties/test_energy.py", "properties/test_golden.py",
          "properties/test_golden_agilicious.py"),
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
    Term("momentum_induced_velocity", "verified", "rotor_aero",
         "Actuator-disk induced velocity: hover v_h = √(T/2ρA); sign-safe axial closed form",
         "spec.inflow.hover_induced_velocity, spec.inflow.induced_velocity_axial",
         ("mccormick", "jsbsim", "bangura"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "The physical input behind ground effect / downwash / climb corrections. "
         "Verified 2026-08-18 against executed JSBSim FGPropeller "
         "(golden/vectors/jsbsim_prop_bldc.json): hover form at V_a = 0 and axial form "
         "at V_a up to 18 m/s. V_a < 0 (descent/reverse-flow branch) unexercised."),
    Term("oblique_momentum_thrust", "verified", "rotor_aero",
         "Nonlinear T(airspeed): T = 2ρA·v_i·U, U = √(Vx²+Vy²+(v_i−Vz)²) (implicit v_i)",
         "spec.inflow.oblique_momentum_thrust",
         ("bangura", "neurobem", "agilicious"), (),
         ("properties/test_candidates.py", "properties/test_golden_agilicious.py"),
         "Principled model the identified k_v2/k_angle/k_hor terms linearize. VRS validity "
         "band excluded (descent 0.5–2 v_h; spec.inflow VRS constants). Verified "
         "2026-08-19: executed verbatim as agilib's ThrustFunction momentum side "
         "(NeuroBEM eq. 5) and pinned via the BEM closure vectors — the implicit-v_i "
         "root regime; explicit T(v_i) evaluation is the same expression."),
    Term("dynamic_inflow_lag", "verified", "rotor_aero",
         "First-order induced-inflow lag to Glauert equilibrium, τ ≈ 16/(γΩ); exact-exp step",
         "spec.inflow.dynamic_inflow_lag", ("jsbsim",), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Same operator-split pattern as the verified motor exact-exp discretization. "
         "Verified 2026-08-18 against executed JSBSim FGRotor "
         "(golden/vectors/jsbsim_rotor_inflow.json): the recorded ν sequences satisfy "
         "the exact-exp step to 1e-10 across hover/axial/edgewise/oblique conditions; "
         "ν_eq is the Glauert equilibrium with the reference's Bailey C_T (transcribed, "
         "self-checked at 1e-12 in the generator — the Bailey closed form itself is not "
         "a spec term)."),
    Term("advance_ratio_tables", "verified", "rotor_aero",
         "T = C_T(J)·ρ·n²·D⁴, P = C_P(J)·ρ·n³·D⁵ with measured tables; windmilling via sign",
         "spec.atmosphere.advance_ratio, spec.atmosphere.propeller_thrust",
         ("jsbsim", "neurobem"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Generalizes the polynomial T(Ω) (a fixed-J slice). J uses axial inflow only; "
         "UIUC/APC databases supply tables for small UAV props. NeuroBEM's BEM model is the "
         "higher-fidelity per-element variant. Verified 2026-08-18 against executed "
         "JSBSim FGPropeller with the wheel's APC 9x4.5E tables "
         "(golden/vectors/jsbsim_prop_bldc.json): J and T pinned per step over J up to "
         "0.42; the P form is exercised through the shaft-ODE load. Windmilling "
         "(J < 0 / C_T < 0) unexercised — the shipped table domain is J ≥ 0."),
    Term("isa_atmosphere", "verified", "environment",
         "USSA-1976 layered T(h), P(h); ρ = P/RT; thrust/torque scale linearly with ρ",
         "spec.atmosphere.temperature_troposphere, spec.atmosphere.pressure_gradient_layer, "
         "spec.atmosphere.density, spec.atmosphere.speed_of_sound",
         ("ussa1976", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Verified-tier coefficients absorb ρ at identification altitude — scale by ρ/ρ_ident. "
         "Verified 2026-08-18 against executed JSBSim FGStandardAtmosphere "
         "(golden/vectors/jsbsim_isa_atmosphere.json): T/P/ρ/a at 12 altitudes to 35 kft, "
         "5e-4 relative (imperial-vs-ICAO constant sets); inputs are geopotential "
         "altitude. Stratosphere (isothermal layer) unexercised."),

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
    Term("flapping_moment_body_rate", "candidate", "rotor_aero",
         "Rotor roll/pitch damping moment −k_flap_w·Ω·Π_ẑ·ω (tip-path plane lags body rate)",
         "spec.rotor_aero.flapping_moment_body_rate", ("sh79", "jsbsim", "kai2017"), (),
         ("properties/test_candidates.py",),
         "Spin-sign-free: adds (not cancels) pairwise — a net damping derivative. Kai "
         "Eq. (7) carries the same hub moment with √T scaling; JSBSim derives it from flap "
         "angles + hinge-offset hub moments."),
    Term("bramwell_rotor_torque", "verified", "rotor_aero",
         "Q = ρbcδ(ΩR)²R²(1+4.5μ²)/8 − (Tλ+Hμ)R: profile + induced/climb torque vs flight "
         "state",
         "spec.rotor_aero.bramwell_torque, spec.rotor_aero.blade_profile_drag",
         ("bramwell", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "The flight-condition dependence (yaw authority and power rise with μ, fall in "
         "descent) that the verified torque polynomial — its fixed-condition slice — lacks. "
         "Needs λ, μ: adopt together with dynamic_inflow_lag. Verified 2026-08-18 against "
         "executed JSBSim FGRotor (golden/vectors/jsbsim_rotor_inflow.json): torque "
         "identity with the δ = 0.009 + 0.3(6C_T/(aσ))² polar at 1e-7 incl. edgewise "
         "μ ≈ 0.07 (the (1+4.5μ²) term is live). H from zero-body-rate flapping; "
         "body-rate flapping contributions to H unexercised (rig holds rates at 0)."),
    Term("ground_effect_talbot_inflow", "candidate", "rotor_aero",
         "IGE inflow factor v_i ← (1 − load·e^{−k_ge(h+h₀)})·v_i, exponential in height",
         "spec.ground_effect.talbot_inflow_factor", ("talbot1977", "jsbsim"), (),
         ("properties/test_candidates.py",),
         "Acts on induced velocity (composes with dynamic_inflow_lag); the thrust-ratio "
         "family (cheeseman_bennett, sanchez_cuevas, pybullet) acts on T directly — use "
         "one route, never both."),

    # ---------------- candidates: blade-element-momentum (NeuroBEM / agilicious) ----------------
    Term("bem_blade_element_loads", "verified", "rotor_aero",
         "BEM disk-load integrands dT/dQ/dH: sinusoidal stall-capable polars "
         "cl=cl0·(sinα·cosα+ε_c), cd=cd0·sin²α over U_T=Ωr+v_hor·sinψ, U_P=(v_ver−v_i)−…",
         "spec.bem.blade_element_integrands, spec.bem.blade_section_velocities, "
         "spec.bem.inflow_angle, spec.bem.section_aoa, spec.bem.lift_coefficient, "
         "spec.bem.drag_coefficient, spec.bem.chord",
         ("neurobem", "agilicious", "gill2017"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib "
         "(golden/vectors/agilicious_bem.json): T/Q/H disk integrals reproduced at the "
         "recorded induced velocity across hover, forward flight to 18 m/s, climb, and "
         "descent; the pure-spec exact-atan2 form deviates <=2.3% from the reference's "
         "float32 atan2 (finding F-22, bounded in the test). Nonzero flapping inside the "
         "integrands unexercised (the reference zeroes it while integrating). "
         "Valid at any incidence/advance ratio (unlike the small-angle verified-tier rotor "
         "terms); smooth throughout. Camber offset ε_c = 0.07 and the H-force correction 3.0 "
         "are executed-code identifications absent from the paper. ⚠ paper eq. (7) prints "
         "+v_ver·β·cosψ where the code has −v_ver·β·cosψ (inert: flapping zeroed during "
         "integration; code form adopted). Reference quadrature: single 15-point "
         "Gauss-Kronrod per axis (generator/consumer detail, not spec)."),

    Term("bem_momentum_inflow_closure", "verified", "rotor_aero",
         "Induced velocity as root of T_BEM(v_i) = 2ρA·v_i·√(v_hor²+(v_ver−v_i)²)",
         "spec.bem.momentum_closure_residual", ("neurobem", "agilicious"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: the recorded Brent roots "
         "satisfy the residual within solver tolerance in 11/12 cases; in deep descent "
         "(25 m/s) the residual provably has NO root in the solver range and the "
         "reference silently returns range-max 30 m/s (finding F-21) — pinned as such. "
         "Momentum side IS spec.inflow.oblique_momentum_thrust with V=(v_hor,0,v_ver). "
         "Reference solves by warm-started vectorized Brent (tol 1e-3); differentiable "
         "backends: fixed smooth iterations, or v_i from dynamic_inflow_lag state (in-ODE)."),

    Term("vrs_empirical_inflow", "verified", "rotor_aero",
         "Vortex-ring-state induced velocity: ṽ_i = v_h·(1+1.125x−1.372x²+1.718x³−0.655x⁴), "
         "x = v_ver/v_h, gated on v_ver/v_i ∈ (0.01, 2)",
         "spec.bem.vrs_induced_velocity", ("hoffmann2007", "neurobem", "agilicious"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: quartic + executed blend replayed "
         "exactly on 6 gated cases (shallow/deep/oblique/fast descent, mixed-regime ANY-gate) "
         "and 6 ungated. Fills the descent band where momentum theory fails (spec.inflow VRS constants). "
         "Blend variants differ: paper max(ṽ_i, v_h); executed agilib max(v_i^mom, ṽ_i) then "
         "clamp ≤ 2·v_h — and its gate fires on ANY-rotor predicates (finding F-20). "
         "Non-smooth (gate + max/min): document surrogate before differentiating through."),

    Term("bem_tpp_wrench", "verified", "rotor_aero",
         "Per-rotor force/torque from tip-path-plane tilt: f = Rz(χ)·(−(H+T·sin a1), "
         "s·T·sin b1, T·cos a0); τ = Rz(χ)·(−s·k_β·b1, −k_β·a1, −s·Q) + r×f",
         "spec.bem.tpp_rotor_force, spec.bem.tpp_rotor_torque",
         ("neurobem", "agilicious", "hoffmann2007"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: composition replayed from "
         "recorded (T, Q, H, a0, a1, b1) through to the executed acceleration/omega-dot "
         "contributions (live flapping angles in all 12 cases, spring moments, 0.9575 "
         "z-obstruction). χ = atan2(v_y, v_x) aligns H with the in-plane hub velocity (drag, rearward); "
         "hinge-spring moments k_β per Hoffmann's hinged-blade model. Flapping angles "
         "(a0, a1, b1) are INPUTS: the reference's machine-generated vehicle-specific "
         "rational fits are rejected (REFERENCES.md) — general closures per Prouty pp. 463 "
         "remain future work. Reduces to (0,0,T) / −s·Q·ẑ at zero flapping and H. The "
         "executed reference also scales the collective z-force by 0.9575 (frame "
         "obstruction) — an assembly-level identified constant."),

    Term("per_axis_quadratic_drag", "verified", "frame_aero",
         "F_k = −k_Q,k·v_a,k·|v_a,k| per body axis (k_Q = ½ρ·c_k·A_k physical packing)",
         "spec.rotor_aero.per_axis_quadratic_drag", ("agilicious", "skydreamer"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib ModelBodyDrag "
         "(golden/vectors/agilicious_simple_models.json). Per-axis |v|·v form (SkyDreamer convention), NOT parasitic_drag's ‖v‖·v — don't mix "
         "coefficients. vertical_climb_drag is its z-restriction: enable one, not both. "
         "⚠ agilib's ModelBodyDrag adds the force to the acceleration slot without dividing "
         "by mass (finding F-19); vectors pin the force expression."),

    Term("cubic_axis_drag", "verified", "frame_aero",
         "F_k = −k_C,k·v_a,k³ per body axis — cubic companion of linear_drag (PolyFit model)",
         "spec.rotor_aero.cubic_drag", ("agilicious", "neurobem"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib ModelLinCubDrag. "
         "The NeuroBEM 'PolyFit' baseline is linear_drag + this + translational_lift "
         "(agilib ModelLinCubDragIndLift, induced_lift_coeff ≡ k_h). Smooth odd polynomial."),

    # ---------------- candidates: motor / battery electrical ----------------
    Term("dc_motor_quasistatic", "verified", "actuator",
         "J_r·Ω̇ = (K_q/R_a)(V_m − K_e·Ω) − k_m·Ω² − b·Ω; τ_m bridge via linearization",
         "spec.motor_electrical.dc_motor_speed_dynamics", ("bangura", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "τ_m = J_r/(K_qK_e/R_a + b + 2k_mΩ₀) connects to the verified first-order lag. "
         "Verified 2026-08-18 against executed JSBSim FGBrushLessDCMotor + FGPropeller "
         "(golden/vectors/jsbsim_prop_bldc.json): static spin-ups at three throttles, "
         "Euler-replayed through the reference's own discretization at 1e-7, with "
         "K_q = K_e = 60/(2πKv) (Drela) and k_m = C_P(0)ρD⁵/(8π³) from the APC table. "
         "The b·Ω viscous term and the I0 friction deadband are unexercised (b = 0, "
         "I0 = 0 in the rig; the deadband is not a spec term)."),
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
         "Load coupling i_batt = Σ u·(u·V_batt − K_e·Ω)/R_a (duty-reflected motor current, "
         "ideal-inverter power balance) closes the sag loop physically."),

    # ---------------- candidates: wind / turbulence ----------------
    Term("dryden_turbulence", "verified", "environment",
         "Dryden forming filters H_u/H_v/H_w + low-altitude scale/intensity closures",
         "spec.wind.dryden_filter_u, spec.wind.dryden_filter_vw, "
         "spec.wind.dryden_low_altitude_scales", ("mil8785c", "cr206937"), (),
         ("properties/test_candidates.py", "properties/test_dryden_authenticity.py",
          "properties/test_golden_jsbsim.py"),
         "⚠ discrete driving noise must be N(0, π/dt) for the published gains; 8785C vs "
         "1797 length-scale factor-of-2 trap. Low-altitude fit is in FEET. Verified via "
         "the ARCHAIC-SOURCE EXCEPTION (2026-08-11): the golden data are NASA "
         "CR-1998-206937's published run statistics (pinned PDF, transcribed tables + "
         "listing), not vectors from executed code — the reference is ACSL (no runnable "
         "interpreter) with an irreproducible RNG. Proven: the report's Tustin difference "
         "equations are exactly the prewarped bilinear of these filters, and re-simulating "
         "them reproduces Tables 2-7 calibration (σ, u/v/w) within the published spread at "
         "L = 1750 ft, V ∈ {100, 1000} ft/s. UPGRADE 2026-08-18: the low-altitude "
         "closures (L(h), σ(h, W20), h < 1000 ft) are now ALSO pinned by executed-code "
         "vectors — JSBSim FGWinds ttTustin implements the same CR-206937 difference "
         "equations with the closures active, and its seeded runs are reproduced "
         "sample-exactly via recovered driving noise "
         "(golden/vectors/jsbsim_dryden_lowalt.json)."),
    Term("von_karman_turbulence", "candidate", "environment",
         "von Kármán spectra (5/6, 11/6 exponents) + standard rational filter approximations",
         "spec.wind.von_karman_psd_u", ("mil8785c",), (),
         ("properties/test_candidates.py",),
         "Measurement-preferred; no exact finite filter."),
    Term("discrete_gust", "candidate", "environment",
         "1-cosine discrete gust ramp per axis",
         "spec.wind.one_minus_cosine_gust", ("mil8785c",), (),
         ("properties/test_candidates.py",)),
    Term("wind_shear_log", "candidate", "environment",
         "MIL-F-8785C mean-wind log profile u_w = W20·ln(h/z0)/ln(20/z0) (h, z0 in ft)",
         "spec.wind.log_wind_shear", ("mil8785c", "mathworks_aeroblks"), (),
         ("properties/test_candidates.py",),
         "The deterministic member of the 8785C wind triad (shear + turbulence + gust); "
         "superposes onto v_wind. Valid 3–1000 ft AGL; z0 = 0.15 ft (Category C landing) "
         "or 2.0 ft (otherwise). Anchored to the same W20 as the Dryden closures, so mean "
         "wind and turbulence intensity stay mutually calibrated."),

    # ---------------- candidates: discretization / integration ----------------
    Term("semi_implicit_euler", "verified", "discretization",
         "Symplectic (semi-implicit) Euler: velocities first with f(s), then positions with "
         "f evaluated at the velocity-updated state",
         "spec.discretization.semi_implicit_euler_step", ("agilicious", "neurobem"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib IntegratorSymplecticEuler. "
         "NeuroBEM's evaluation integrator (1 ms steps, §IV-D): first-order like explicit "
         "Euler but symplectic on the mechanical part — bounded energy error instead of "
         "drift. agilib groups (v, ω, Ω) as velocities and (x, q) as positions; the "
         "quaternion advances with the NEW ω against the OLD q. Single smooth composition, "
         "cleanly differentiable."),

    Term("quaternion_norm_correction", "candidate", "discretization",
         "q̇ = ½ q ⊗ (0, ω) + K·(1−‖q‖²)·q — smooth Lagrange-style norm stabilization",
         "spec.quaternion.kinematics_norm_corrected", ("mathworks_aeroblks",), (),
         ("properties/test_candidates.py",),
         "Differentiable alternative to the harness's post-step renormalization: the "
         "correction lives inside the ODE (single smooth vector field for backends that "
         "differentiate through the integrator). d‖q‖²/dt = 2K·ε·‖q‖², ε = 1−‖q‖² → norm "
         "error decays at rate ≈ 2K; exactly quaternion.kinematics on the unit manifold. "
         "Textbook basis: Stevens & Lewis; Zipfel. Choose K·dt ≪ 1."),

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
    ("genesis", ("Genesis (Genesis-Embodied-AI): props are fixed joints, KF·rpm² force + "
                 "KM·rpm² yaw torque only, no gyroscopic effects — strict subset of this spec.")),
)


def by_key(key: str) -> Term:
    for t in TERMS:
        if t.key == key:
            return t
    raise KeyError(key)


def validate_registry() -> None:
    """Structural invariants checked by properties/test_registry.py."""
    from skyflow_dynamics.spec.parameters import SCHEMA
    keys = [t.key for t in TERMS]
    assert len(keys) == len(set(keys)), "duplicate term keys"
    domains = ("rigid_body", "actuator", "rotor_aero", "frame_aero", "sensor", "disturbance",
               "discretization", "differentiation", "environment", "harness")
    for t in TERMS:
        assert t.tier in ("verified", "candidate"), t.key
        assert t.domain in domains, f"{t.key}: unknown domain {t.domain}"
        assert t.sources, f"{t.key} has no sources"
        for s in t.sources:
            assert s in SOURCES, f"{t.key}: unknown source {s}"
        for p in t.parameters:
            assert p in SCHEMA, f"{t.key}: unknown parameter {p}"
        if t.tier == "verified" and t.domain not in ("harness",):
            assert t.tests or t.expression.startswith("(harness"), \
                f"{t.key} is verified but lists no tests"
