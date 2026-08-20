# SkyFlow-Dynamics — equation catalog

*Generated from `spec/registry.py` by `tools/render_docs.py` — do not edit by hand.*

The authoritative math lives in the `spec/` modules (each function's docstring carries the
full equations, unit statements, and pitfalls); this catalog is the index. Conventions are
stated in [README.md](../README.md).

## The canonical model

State s = (x, v, q, ω, Ω); input Ω_c; exogenous (v_wind, F_ext, τ_ext):

```
v_a  = R(q)ᵀ (v − v_wind)                        body airspeed
v_i  = v_a + ω × r_i                              local airspeed at rotor hub i
T_i  = (ct0 + ct1·Ω_i + ct2·Ω_i²)·(1 + k_angle·α + k_hor·μ) + k_h·(v_i,x² + v_i,y²)
H_i  = −Ω_i · diag(k_d, k_d, k_z) · v_i
Q_i  = cq0 + cq1·Ω_i + cq2·Ω_i²
F_B  = Σᵢ (T_i·ê_i + H_i) − ‖v_a‖·diag(c_D)·v_a − diag(c_L)·v_a − k_v2·v_az|v_az|·ẑ
M_B  = Σᵢ r_i×(T_i·ê_i + H_i) − Σᵢ s_i·Q_i·ê_i − Σᵢ k_flap·Ω_i·(v_i × ẑ)
       − ω×h + (−I_rot·Σᵢ s_i·Ω̇_i)·ẑ,          h = I_rot·(Σᵢ s_i·Ω_i)·ẑ

ẋ = v                    v̇ = (0,0,−g) + (R(q)·F_B + F_ext)/m
q̇ = ½·q ⊗ (0, ω)        ω̇ = I⁻¹(M_B + τ_ext − ω×(I·ω))
Ω̇ = motor model          (first-order lag or asymmetric spin-up/down)
```

Tier legend: **verified** = golden-tested against a reference implementation's running code;
*candidate* = published model, symbolically checked and cited, awaiting numeric validation.


## Rigid body

### `newton_euler` — **verified**

Rigid-body translational + rotational EOM with full inertia matrix.

- **Defined in:** `spec.rigid_body.translational, spec.rigid_body.rotational`
- **Parameters:** `mass`, `grav`, `inertia`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor, IEEE RAM 2012
- **Tests:** `properties/test_rigid_body.py`, `properties/test_golden.py`

### `quaternion_kinematics` — **verified**

q̇ = ½ q ⊗ (0, ω); wxyz scalar-first Hamilton, body→world.

- **Defined in:** `spec.quaternion.kinematics`
- **Sources:** Graf — Quaternions and Dynamics (quaternion kinematics); Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Tests:** `properties/test_quaternion.py`
- **Notes:** Unit norm preserved exactly by the continuous equation; discrete integrators renormalize post-step (harness).


## Actuators (motor / ESC / battery)

### `motor_first_order_lag` — **verified**

Ω̇ = (Ω_c − Ω)/τ_m.

- **Defined in:** `spec.motor.first_order_lag`
- **Parameters:** `tau_m`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Förster — System Identification of the Crazyflie 2.0 Nano Quadrocopter, ETH Zürich, 2015
- **Tests:** `properties/test_motor.py`, `properties/test_golden.py`

### `motor_asymmetric_lag` — **verified**

Separate spin-up/spin-down linear+quadratic rates.

- **Defined in:** `spec.motor.asymmetric_lag`
- **Parameters:** `ka1`, `ka2`, `kd1`, `kd2`
- **Sources:** Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml
- **Tests:** `properties/test_motor.py`, `properties/test_golden.py`
- **Notes:** Crazyflow coefficients are RPM-based; because Ω̇ rescales with Ω, ka1/kd1 carry over unchanged and ka2/kd2 convert by ×60/2π — i.e. (60/2π)^(Ω-power − 1). Reduces to first-order at (1/τ, 0, 1/τ, 0).

### `throttle_curve` — **verified**

Ω_c = (Ω_max−Ω_min)√(k·u² + (1−k)u) + Ω_min.

- **Defined in:** `spec.motor.throttle_to_speed`
- **Sources:** SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_motor.py`
- **Notes:** Identified k = 0.5 for a 5-inch racer (Ω_min 341.75, Ω_max 3100 rad/s).

### `pwm_quantization` — **verified**

Throttle snapped to the integer PWM grid.

- **Defined in:** `spec.motor.pwm_quantize`
- **Sources:** Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml
- **Tests:** `properties/test_motor.py`
- **Notes:** Piecewise-constant — exclude from differentiable paths.

### `battery_voltage_speed_cap` — **verified**

Supply voltage → achievable Ω_max (linear map); slow drift over discharge.

- **Defined in:** `spec.motor.voltage_to_rpm`
- **Sources:** Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml; SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_motor.py`
- **Notes:** Battery state evolution (SoC, sag dynamics) is harness-side.

### `dc_motor_quasistatic` — **verified**

J_r·Ω̇ = (K_q/R_a)(V_m − K_e·Ω) − k_m·Ω² − b·Ω; τ_m bridge via linearization.

- **Defined in:** `spec.motor_electrical.dc_motor_speed_dynamics`
- **Sources:** Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony (ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for Quadrotors (arXiv:1601.00733); JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** τ_m = J_r/(K_qK_e/R_a + b + 2k_mΩ₀) connects to the verified first-order lag. Verified 2026-08-18 against executed JSBSim FGBrushLessDCMotor + FGPropeller (golden/vectors/jsbsim_prop_bldc.json): static spin-ups at three throttles, Euler-replayed through the reference's own discretization at 1e-7, with K_q = K_e = 60/(2πKv) (Drela) and k_m = C_P(0)ρD⁵/(8π³) from the APC table. The b·Ω viscous term and the I0 friction deadband are unexercised (b = 0, I0 = 0 in the rig; the deadband is not a spec term).

### `esc_battery_coupling` — *candidate*

V_m = u·V_batt; battery-coupled steady-state speed (√-like in u·V_batt).

- **Defined in:** `spec.motor_electrical.esc_mean_voltage, spec.motor_electrical.steady_state_speed`
- **Sources:** Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony (ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for Quadrotors (arXiv:1601.00733); bitcraze/crazyflie-firmware — motors.c motorsCompensateBatteryVoltage + platform_defaults_cf2.h (master and tag 2022.01)
- **Tests:** `properties/test_candidates.py`
- **Notes:** The physical origin of the verified throttle curve's √(k·u²+(1−k)·u) shape.

### `crazyflie_battery_compensation` — *candidate*

Firmware cubic T(v_m) with Cardano inversion (+ legacy quadratic), PWM/V_batt scaling.

- **Defined in:** `spec.motor_electrical.crazyflie_thrust_from_voltage`
- **Sources:** bitcraze/crazyflie-firmware — motors.c motorsCompensateBatteryVoltage + platform_defaults_cf2.h (master and tag 2022.01)
- **Tests:** `properties/test_candidates.py`
- **Notes:** Identified C0–C3 per prop variant; thrust held constant under sag by design.

### `thevenin_battery` — *candidate*

OCV(SoC) + series R(SoC) + two RC branches; Gazebo linear model as special case.

- **Defined in:** `spec.motor_electrical.thevenin_battery, spec.motor_electrical.chen_ocv`
- **Sources:** Chen & Rincon-Mora — Accurate Electrical Battery Model Capable of Predicting Runtime and I-V Performance, IEEE Trans. Energy Conversion 21(2), 2006; gazebosim/gz-sim LinearBatteryPlugin.cc (linear OCV + internal resistance + current low-pass)
- **Tests:** `properties/test_candidates.py`
- **Notes:** Load coupling i_batt = Σ u·(u·V_batt − K_e·Ω)/R_a (duty-reflected motor current, ideal-inverter power balance) closes the sag loop physically.


## Rotor aerodynamics

### `rotor_thrust_polynomial` — **verified**

T_i = ct0 + ct1·Ω + ct2·Ω² per rotor (per-rotor coefficient asymmetry supported).

- **Defined in:** `spec.rotor_aero.thrust_magnitude`
- **Parameters:** `ct0`, `ct1`, `ct2`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml; SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_wrench.py`, `properties/test_golden.py`
- **Notes:** Crazyflow identifies RPM-unit polynomials; SkyDreamer's k_w is mass-normalized (multiply by m; finding F-4).

### `rotor_torque_polynomial` — **verified**

Q_i = cq0 + cq1·Ω + cq2·Ω²; yaw torque on airframe = −s_i·Q_i·ê_i (opposes spin).

- **Defined in:** `spec.rotor_aero.torque_magnitude`
- **Parameters:** `cq0`, `cq1`, `cq2`, `spin`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml
- **Tests:** `properties/test_wrench.py`, `properties/test_golden.py`
- **Notes:** ⚠ RotorPy's rotor_directions = torque sign = −spin (finding F-6).

### `thrust_axis_misalignment` — **verified**

Per-rotor unit thrust axis ê_i ≠ ẑ from assembly tolerance → parasitic forces/moments.

- **Defined in:** `spec.wrench.body_wrench`
- **Parameters:** `axis`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Tests:** `properties/test_wrench.py`, `properties/test_golden.py`

### `rotor_inertia_moments` — **verified**

Gyroscopic precession −ω×h and yaw reaction −I_rot·Σ s_i Ω̇_i·ẑ.

- **Defined in:** `spec.wrench.rotor_inertia_moment`
- **Parameters:** `I_rot`, `spin`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml; SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py; Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual Features Using Differentiable Simulation, ICRA 2025; reference implementation executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28)
- **Tests:** `properties/test_wrench.py`, `properties/test_golden.py`, `properties/test_golden_flightning.py`
- **Notes:** Signs re-derived from τ = −d/dt(h). Crazyflow's gyro roll-row sign was flipped (finding F-3, confirmed against their running code); fixed upstream by learnsyslab/crazyflow PR #86 (merged 2026-07-13) — post-fix Crazyflow golden vectors now cross-validate this term. flightning implements ONLY the yaw-reaction half (+I_m·Σ dir_i·Ω̇_i·ẑ with dir_i = −s_i ≡ this term at ω = 0, executed-code verified) with the continuous rate Ω̇ = (Ω_c−Ω)/τ; it omits the −ω×h precession.

### `rotor_drag_hforce` — **verified**

H_i = −Ω_i·diag(k_d,k_d,k_z)·v_i at each hub, v_i incl. ω×r_i lever arm.

- **Defined in:** `spec.rotor_aero.rotor_drag_force`
- **Parameters:** `k_d`, `k_z`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor, IEEE RAM 2012; SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_energy.py`, `properties/test_golden.py`
- **Notes:** SkyDreamer's lumped −k_x·v·ΣΩ is this summed over rotors (k_d = m·k_x; F-4).

### `blade_flapping_moment` — **verified**

M_flap,i = −k_flap·Ω_i·(v_i × ẑ); +M_y (nose-down in FLU) for v_x > 0, k_flap > 0.

- **Defined in:** `spec.rotor_aero.flapping_moment`
- **Parameters:** `k_flap`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor, IEEE RAM 2012
- **Tests:** `properties/test_wrench.py`, `properties/test_golden.py`

### `translational_lift` — **verified**

ΔT_i = k_h·(v_i,x² + v_i,y²).

- **Defined in:** `spec.rotor_aero.translational_lift`
- **Parameters:** `k_h`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork); Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py)
- **Tests:** `properties/test_golden.py`, `properties/test_golden_agilicious.py`
- **Notes:** Also executed as agilib ModelLinCubDrag's induced_lift_coeff (2026-08-19). Small-airspeed linearization of the AoA/advance-ratio model — mutually exclusive with k_angle/k_hor (validation rule).

### `aoa_advance_ratio_thrust` — **verified**

T × (1 + k_angle·atan2(v_az, rΩ̄) + k_hor·atan2(‖v_axy‖, rΩ̄)).

- **Defined in:** `spec.rotor_aero.aoa_thrust_factor`
- **Parameters:** `k_angle`, `k_hor`, `r_prop`
- **Sources:** SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_golden.py`
- **Notes:** Identified to racing speeds: k_angle 3.145, k_hor 7.245 (mass-normalized k_w; F-4). Spec follows the runnable reference (ENU, mean Ω̄, hypot); the paper's printed equations differ (NED, ΣΩ, squared numerator) — see the function docstring.

### `vertical_climb_drag` — **verified**

−k_v2·v_az·|v_az|·ẑ collective at CoM.

- **Defined in:** `spec.rotor_aero.vertical_climb_drag`
- **Parameters:** `k_v2`
- **Sources:** SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py
- **Tests:** `properties/test_golden.py`

### `ground_effect_cheeseman_bennett` — *candidate*

Single-rotor IGE thrust ratio 1/(1−(R/4z)²); forward-flight washout variant.

- **Defined in:** `spec.ground_effect.cheeseman_bennett, spec.ground_effect.cheeseman_bennett_forward`
- **Sources:** Cheeseman & Bennett — The Effect of the Ground on a Helicopter Rotor in Forward Flight, ARC R&M 3021, 1955; Sanchez-Cuevas, Heredia, Ollero — Characterization of the Aerodynamic Ground Effect and Its Influence in Multirotor Control, Int. J. Aerospace Eng. 2017, doi 10.1155/2017/1823056
- **Tests:** `properties/test_candidates.py`
- **Notes:** Valid 0.5 ≤ z/R ≤ 2; singular at z = R/4 (clamp). Under-predicts for multirotors.

### `ground_effect_sanchez_cuevas` — *candidate*

Quadrotor IGE with mirrored-rotor images + fountain body-lift (K_b ≈ 2).

- **Defined in:** `spec.ground_effect.sanchez_cuevas`
- **Sources:** Sanchez-Cuevas, Heredia, Ollero — Characterization of the Aerodynamic Ground Effect and Its Influence in Multirotor Control, Int. J. Aerospace Eng. 2017, doi 10.1155/2017/1823056
- **Tests:** `properties/test_candidates.py`
- **Notes:** Significant to z ≈ 5R. Reduces to Cheeseman-Bennett as d, b → ∞, K_b → 0.

### `ground_effect_pybullet` — *candidate*

Per-rotor additive increment ΔT = T·G·(R/4z)² (linearized CB, identified G).

- **Defined in:** `spec.ground_effect.pybullet_ground_effect`
- **Sources:** utiasDSL/gym-pybullet-drones BaseAviary (_groundEffect, _downwash) + cf2x.urdf identified constants
- **Tests:** `properties/test_candidates.py`
- **Notes:** G = 11.37 identified for CF2 (fountain amplification folded in); needs height clip.

### `momentum_induced_velocity` — **verified**

Actuator-disk induced velocity: hover v_h = √(T/2ρA); sign-safe axial closed form.

- **Defined in:** `spec.inflow.hover_induced_velocity, spec.inflow.induced_velocity_axial`
- **Sources:** McCormick — Aerodynamics, Aeronautics, and Flight Mechanics, 1st ed. (momentum-theory induced velocity, Eq. 6.15); JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py); Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony (ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for Quadrotors (arXiv:1601.00733)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** The physical input behind ground effect / downwash / climb corrections. Verified 2026-08-18 against executed JSBSim FGPropeller (golden/vectors/jsbsim_prop_bldc.json): hover form at V_a = 0 and axial form at V_a up to 18 m/s. V_a < 0 (descent/reverse-flow branch) unexercised.

### `oblique_momentum_thrust` — **verified**

Nonlinear T(airspeed): T = 2ρA·v_i·U, U = √(Vx²+Vy²+(v_i−Vz)²) (implicit v_i).

- **Defined in:** `spec.inflow.oblique_momentum_thrust`
- **Sources:** Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony (ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for Quadrotors (arXiv:1601.00733); Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_agilicious.py`
- **Notes:** Principled model the identified k_v2/k_angle/k_hor terms linearize. VRS validity band excluded (descent 0.5–2 v_h; spec.inflow VRS constants). Verified 2026-08-19: executed verbatim as agilib's ThrustFunction momentum side (NeuroBEM eq. 5) and pinned via the BEM closure vectors — the implicit-v_i root regime; explicit T(v_i) evaluation is the same expression.

### `dynamic_inflow_lag` — **verified**

First-order induced-inflow lag to Glauert equilibrium, τ ≈ 16/(γΩ); exact-exp step.

- **Defined in:** `spec.inflow.dynamic_inflow_lag`
- **Sources:** JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** Same operator-split pattern as the verified motor exact-exp discretization. Verified 2026-08-18 against executed JSBSim FGRotor (golden/vectors/jsbsim_rotor_inflow.json): the recorded ν sequences satisfy the exact-exp step to 1e-10 across hover/axial/edgewise/oblique conditions; ν_eq is the Glauert equilibrium with the reference's Bailey C_T (transcribed, self-checked at 1e-12 in the generator — the Bailey closed form itself is not a spec term).

### `advance_ratio_tables` — **verified**

T = C_T(J)·ρ·n²·D⁴, P = C_P(J)·ρ·n³·D⁵ with measured tables; windmilling via sign.

- **Defined in:** `spec.atmosphere.advance_ratio, spec.atmosphere.propeller_thrust`
- **Sources:** JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py); Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** Generalizes the polynomial T(Ω) (a fixed-J slice). J uses axial inflow only; UIUC/APC databases supply tables for small UAV props. NeuroBEM's BEM model is the higher-fidelity per-element variant. Verified 2026-08-18 against executed JSBSim FGPropeller with the wheel's APC 9x4.5E tables (golden/vectors/jsbsim_prop_bldc.json): J and T pinned per step over J up to 0.42; the P form is exercised through the shaft-ODE load. Windmilling (J < 0 / C_T < 0) unexercised — the shipped table domain is J ≥ 0.

### `rolling_moment` — *candidate*

Per-rotor rolling moment −Ω·s·μ_R·v_⊥ (advancing/retreating dissymmetry).

- **Defined in:** `spec.rotor_aero.rolling_moment`
- **Sources:** ethz-asl/rotors_simulator gazebo_motor_model.cpp + PX4/PX4-SITL_gazebo-classic variant (signed rolling moment); Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13)
- **Tests:** `properties/test_candidates.py`
- **Notes:** RotorS omits the spin sign (bug); PX4 variant adopted. Cancels for balanced pairs.

### `flapping_force_body_rate` — *candidate*

Kai Eq. (10) flapping force incl. spin-signed lateral and body-rate damping terms.

- **Defined in:** `spec.rotor_aero.flapping_force_kai`
- **Sources:** Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13); Faessler, Franchi, Scaramuzza — Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories, IEEE RA-L 2018
- **Tests:** `properties/test_candidates.py`
- **Notes:** The published basis for the lumped linear drag; adds rotor-plane damping (B·ω) absent from the verified tier.

### `flapping_moment_body_rate` — *candidate*

Rotor roll/pitch damping moment −k_flap_w·Ω·Π_ẑ·ω (tip-path plane lags body rate).

- **Defined in:** `spec.rotor_aero.flapping_moment_body_rate`
- **Sources:** Shaughnessy, Deaux, Yenni — Development and Validation of a Piloted Simulation of a Helicopter and External Sling Load, NASA TP-1285, 1979 (JSBSim FGRotor's model basis); body-rate flap terms per Amer, NACA TN-2136, 1950; JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py); Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13)
- **Tests:** `properties/test_candidates.py`
- **Notes:** Spin-sign-free: adds (not cancels) pairwise — a net damping derivative. Kai Eq. (7) carries the same hub moment with √T scaling; JSBSim derives it from flap angles + hinge-offset hub moments.

### `bramwell_rotor_torque` — **verified**

Q = ρbcδ(ΩR)²R²(1+4.5μ²)/8 − (Tλ+Hμ)R: profile + induced/climb torque vs flight state.

- **Defined in:** `spec.rotor_aero.bramwell_torque, spec.rotor_aero.blade_profile_drag`
- **Sources:** Bramwell — Helicopter Dynamics, 2nd ed., eqns 3.43-3.44 (rotor torque decomposition: profile + induced/climb components); JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** The flight-condition dependence (yaw authority and power rise with μ, fall in descent) that the verified torque polynomial — its fixed-condition slice — lacks. Needs λ, μ: adopt together with dynamic_inflow_lag. Verified 2026-08-18 against executed JSBSim FGRotor (golden/vectors/jsbsim_rotor_inflow.json): torque identity with the δ = 0.009 + 0.3(6C_T/(aσ))² polar at 1e-7 incl. edgewise μ ≈ 0.07 (the (1+4.5μ²) term is live). H from zero-body-rate flapping; body-rate flapping contributions to H unexercised (rig holds rates at 0).

### `ground_effect_talbot_inflow` — *candidate*

IGE inflow factor v_i ← (1 − load·e^{−k_ge(h+h₀)})·v_i, exponential in height.

- **Defined in:** `spec.ground_effect.talbot_inflow_factor`
- **Sources:** Talbot & Corliss — A Mathematical Force and Moment Model of a UH-1H Helicopter for Flight Dynamics Simulations, NASA TM-73,254, 1977 (eqn 10a ground-effect inflow factor); JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py)
- **Tests:** `properties/test_candidates.py`
- **Notes:** Acts on induced velocity (composes with dynamic_inflow_lag); the thrust-ratio family (cheeseman_bennett, sanchez_cuevas, pybullet) acts on T directly — use one route, never both.

### `bem_blade_element_loads` — **verified**

BEM disk-load integrands dT/dQ/dH: sinusoidal stall-capable polars cl=cl0·(sinα·cosα+ε_c), cd=cd0·sin²α over U_T=Ωr+v_hor·sinψ, U_P=(v_ver−v_i)−….

- **Defined in:** `spec.bem.blade_element_integrands, spec.bem.blade_section_velocities, spec.bem.inflow_angle, spec.bem.section_aoa, spec.bem.lift_coefficient, spec.bem.drag_coefficient, spec.bem.chord`
- **Sources:** Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py); Gill & D'Andrea — Propeller Thrust and Drag in Forward Flight, IEEE CCTA 2017 (sinusoidal high-incidence lift/drag polars; with Ducard & Hua, CCA 2014)
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib (golden/vectors/agilicious_bem.json): T/Q/H disk integrals reproduced at the recorded induced velocity across hover, forward flight to 18 m/s, climb, and descent; the pure-spec exact-atan2 form deviates <=2.3% from the reference's float32 atan2 (finding F-22, bounded in the test). Nonzero flapping inside the integrands unexercised (the reference zeroes it while integrating). Valid at any incidence/advance ratio (unlike the small-angle verified-tier rotor terms); smooth throughout. Camber offset ε_c = 0.07 and the H-force correction 3.0 are executed-code identifications absent from the paper. ⚠ paper eq. (7) prints +v_ver·β·cosψ where the code has −v_ver·β·cosψ (inert: flapping zeroed during integration; code form adopted). Reference quadrature: single 15-point Gauss-Kronrod per axis (generator/consumer detail, not spec).

### `bem_momentum_inflow_closure` — **verified**

Induced velocity as root of T_BEM(v_i) = 2ρA·v_i·√(v_hor²+(v_ver−v_i)²).

- **Defined in:** `spec.bem.momentum_closure_residual`
- **Sources:** Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py)
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib: the recorded Brent roots satisfy the residual within solver tolerance in 11/12 cases; in deep descent (25 m/s) the residual provably has NO root in the solver range and the reference silently returns range-max 30 m/s (finding F-21) — pinned as such. Momentum side IS spec.inflow.oblique_momentum_thrust with V=(v_hor,0,v_ver). Reference solves by warm-started vectorized Brent (tol 1e-3); differentiable backends: fixed smooth iterations, or v_i from dynamic_inflow_lag state (in-ODE).

### `vrs_empirical_inflow` — **verified**

Vortex-ring-state induced velocity: ṽ_i = v_h·(1+1.125x−1.372x²+1.718x³−0.655x⁴), x = v_ver/v_h, gated on v_ver/v_i ∈ (0.01, 2).

- **Defined in:** `spec.bem.vrs_induced_velocity`
- **Sources:** Hoffmann, Huang, Waslander, Tomlin — Quadrotor Helicopter Flight Dynamics and Control: Theory and Experiment, AIAA GNC 2007 (VRS empirical quartic; hinged-blade spring model); Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py)
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib: quartic + executed blend replayed exactly on 6 gated cases (shallow/deep/oblique/fast descent, mixed-regime ANY-gate) and 6 ungated. Fills the descent band where momentum theory fails (spec.inflow VRS constants). Blend variants differ: paper max(ṽ_i, v_h); executed agilib max(v_i^mom, ṽ_i) then clamp ≤ 2·v_h — and its gate fires on ANY-rotor predicates (finding F-20). Non-smooth (gate + max/min): document surrogate before differentiating through.

### `bem_tpp_wrench` — **verified**

Per-rotor force/torque from tip-path-plane tilt: f = Rz(χ)·(−(H+T·sin a1), s·T·sin b1, T·cos a0); τ = Rz(χ)·(−s·k_β·b1, −k_β·a1, −s·Q) + r×f.

- **Defined in:** `spec.bem.tpp_rotor_force, spec.bem.tpp_rotor_torque`
- **Sources:** Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py); Hoffmann, Huang, Waslander, Tomlin — Quadrotor Helicopter Flight Dynamics and Control: Theory and Experiment, AIAA GNC 2007 (VRS empirical quartic; hinged-blade spring model)
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib: composition replayed from recorded (T, Q, H, a0, a1, b1) through to the executed acceleration/omega-dot contributions (live flapping angles in all 12 cases, spring moments, 0.9575 z-obstruction). χ = atan2(v_y, v_x) aligns H with the in-plane hub velocity (drag, rearward); hinge-spring moments k_β per Hoffmann's hinged-blade model. Flapping angles (a0, a1, b1) are INPUTS: the reference's machine-generated vehicle-specific rational fits are rejected (REFERENCES.md) — general closures per Prouty pp. 463 remain future work. Reduces to (0,0,T) / −s·Q·ẑ at zero flapping and H. The executed reference also scales the collective z-force by 0.9575 (frame obstruction) — an assembly-level identified constant.


## Frame aerodynamics

### `linear_drag` — **verified**

Lumped linear body-frame drag F = −diag(c_L)·v_a (Faessler differential-flatness form).

- **Defined in:** `spec.rotor_aero.linear_drag`
- **Parameters:** `c_L`
- **Sources:** Faessler, Franchi, Scaramuzza — Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories, IEEE RA-L 2018; Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml; Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py)
- **Tests:** `properties/test_energy.py`, `properties/test_golden.py`, `properties/test_golden_agilicious.py`
- **Notes:** Ω-independent lumping of the per-rotor H-force; identify against c_L OR k_d, not both. Crazyflow stores the negated diagonal (drag_matrix = −diag(c_L)).

### `parasitic_drag` — **verified**

D = −‖v_a‖·diag(c_D)·v_a at CoM.

- **Defined in:** `spec.rotor_aero.parasitic_drag`
- **Parameters:** `c_D`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Tests:** `properties/test_energy.py`, `properties/test_golden.py`
- **Notes:** ⚠ ‖v‖-scaled, NOT per-axis |v_k|·v_k (SkyDreamer's form) — structurally different; don't transplant coefficients between the two.

### `per_axis_quadratic_drag` — **verified**

F_k = −k_Q,k·v_a,k·|v_a,k| per body axis (k_Q = ½ρ·c_k·A_k physical packing).

- **Defined in:** `spec.rotor_aero.per_axis_quadratic_drag`
- **Sources:** Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py); SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py; Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual Features Using Differentiable Simulation, ICRA 2025; reference implementation executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28)
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`, `properties/test_golden_flightning.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib ModelBodyDrag (golden/vectors/agilicious_simple_models.json) and the EXECUTED flightning compute_drag_force (same ½ρ·c_k·A_k packing, correctly divided by mass — no F-19 analog; its ±50% coefficient randomization is harness-side, replayed as effective params). Per-axis |v|·v form (SkyDreamer convention), NOT parasitic_drag's ‖v‖·v — don't mix coefficients. vertical_climb_drag is its z-restriction: enable one, not both. ⚠ agilib's ModelBodyDrag adds the force to the acceleration slot without dividing by mass (finding F-19); vectors pin the force expression.

### `cubic_axis_drag` — **verified**

F_k = −k_C,k·v_a,k³ per body axis — cubic companion of linear_drag (PolyFit model).

- **Defined in:** `spec.rotor_aero.cubic_drag`
- **Sources:** Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py); Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib ModelLinCubDrag. The NeuroBEM 'PolyFit' baseline is linear_drag + this + translational_lift (agilib ModelLinCubDragIndLift, induced_lift_coeff ≡ k_h). Smooth odd polynomial.


## Disturbances & interaction

### `external_wrench_inputs` — **verified**

Exogenous F_ext (world) and τ_ext (body) enter the EOM directly.

- **Defined in:** `spec.rigid_body.translational, spec.rigid_body.rotational`
- **Sources:** SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py; Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml
- **Tests:** `properties/test_golden.py`
- **Notes:** The two-band resample-and-hold schedule that drives these in training (SkyDreamer Table III: ±3 m/s²+±3 rad/s² @1 Hz, ±125 rad/s² @90 Hz, ε_u ±0.2) is harness-side.

### `downwash_pybullet` — *candidate*

Inter-vehicle Gaussian downwash force fit (DSL/SiQi Zhou).

- **Defined in:** `spec.ground_effect.pybullet_downwash_force`
- **Sources:** utiasDSL/gym-pybullet-drones BaseAviary (_groundEffect, _downwash) + cf2x.urdf identified constants
- **Tests:** `properties/test_candidates.py`
- **Notes:** CF2-specific fit, thrust-independent, untrustworthy below Δz ≈ 0.7 m.

### `downwash_jain_jet` — *candidate*

Turbulent-jet wake velocity field + frame drag + per-rotor thrust loss.

- **Defined in:** `spec.ground_effect.jain_wake_velocity`
- **Sources:** Jain, Fortmuller, Byun, Makiharju, Mueller — Modeling of aerodynamic disturbances for proximity flight of multirotors, ICUAS 2019, Eqs. (1)-(8)
- **Tests:** `properties/test_candidates.py`
- **Notes:** Thrust-scaled, gives force AND moment; ZEF only (z > 3L). Route wake velocity through ONE rotor-inflow path to avoid double-counting with AoA thrust terms.


## Sensors

### `imu_measurement` — **verified**

Specific force + body rate at offset/rotated mount, lever-arm terms in body frame.

- **Defined in:** `spec.sensors.imu`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Tests:** `properties/test_sensors.py`
- **Notes:** Frame-mixing defects F-1/F-2 found and fixed in the reference; equations here are the corrected form.


## Discretization

### `motor_exact_exp_discretization` — **verified**

Closed-form Ω(dt) = Ω_c + (Ω₀−Ω_c)e^(−dt/τ); operator-split from the RK stages.

- **Defined in:** `spec.motor.exact_exp_step`
- **Parameters:** `tau_m`
- **Sources:** Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual Features Using Differentiable Simulation, ICRA 2025; reference implementation executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28)
- **Tests:** `properties/test_motor.py`, `properties/test_golden.py`, `properties/test_golden_flightning.py`
- **Notes:** Unconditionally stable; per-step gradient factor e^(−dt/τ) ∈ (0,1). Linear lag only. Verified 2026-08-19 against the EXECUTED flightning quadrotor_obj.py ((Ω−Ω_c)e^(−dt/τ)+Ω_c, post-step clip to [Ω_min, Ω_max] as harness detail).

### `rk4_fixed_step` — **verified**

Classical RK4; the differentiable reference integrator (adaptive solvers are not cleanly differentiable).

- **Defined in:** `spec.discretization.rk4_step`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Tests:** `properties/test_motor.py`, `properties/test_golden.py`
- **Notes:** flightning attribution removed 2026-08-19: its executed integrator is explicit Euler at 1 kHz (exact attitude/motor substeps), no RK4 anywhere in the repo.

### `semi_implicit_euler` — **verified**

Symplectic (semi-implicit) Euler: velocities first with f(s), then positions with f evaluated at the velocity-updated state.

- **Defined in:** `spec.discretization.semi_implicit_euler_step`
- **Sources:** Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py); Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B
- **Tests:** `properties/test_bem.py`, `properties/test_golden_agilicious.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED agilib IntegratorSymplecticEuler. NeuroBEM's evaluation integrator (1 ms steps, §IV-D): first-order like explicit Euler but symplectic on the mechanical part — bounded energy error instead of drift. agilib groups (v, ω, Ω) as velocities and (x, q) as positions; the quaternion advances with the NEW ω against the OLD q. Single smooth composition, cleanly differentiable.

### `quaternion_norm_correction` — *candidate*

q̇ = ½ q ⊗ (0, ω) + K·(1−‖q‖²)·q — smooth Lagrange-style norm stabilization.

- **Defined in:** `spec.quaternion.kinematics_norm_corrected`
- **Sources:** MathWorks Aerospace Blockset documentation — block-equation pages only (implementations are proprietary and were not consulted); each block cites the public standard it implements
- **Tests:** `properties/test_candidates.py`
- **Notes:** Differentiable alternative to the harness's post-step renormalization: the correction lives inside the ODE (single smooth vector field for backends that differentiate through the integrator). d‖q‖²/dt = 2K·ε·‖q‖², ε = 1−‖q‖² → norm error decays at rate ≈ 2K; exactly quaternion.kinematics on the unit manifold. Textbook basis: Stevens & Lewis; Zipfel. Choose K·dt ≪ 1.


## Differentiable simulation

### `point_mass_surrogate` — **verified**

Point mass + kinematic attitude; surrogate Jacobian for BPTT via straight-through.

- **Defined in:** `spec.simplified.step, spec.simplified.dynamics`
- **Sources:** Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual Features Using Differentiable Simulation, ICRA 2025; reference implementation executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28)
- **Tests:** `properties/test_simplified.py`, `properties/test_golden_flightning.py`
- **Notes:** Verified 2026-08-19 against the EXECUTED flightning quadrotor_dyn: primal steps, jax.jvp tangents, AND the step()-level custom_jvp wiring (c = f_d/m, dt-tangent 0) — executed-code confirmation of the surrogate-gradient scheme. Their attitude step is a biased-angle Rodrigues, not the exp map (finding F-25, deviation bounded in the golden test); their custom_jvp is broken on JAX ≥ 0.11 (finding F-28).


## Environment (atmosphere / turbulence)

### `isa_atmosphere` — **verified**

USSA-1976 layered T(h), P(h); ρ = P/RT; thrust/torque scale linearly with ρ.

- **Defined in:** `spec.atmosphere.temperature_troposphere, spec.atmosphere.pressure_gradient_layer, spec.atmosphere.density, spec.atmosphere.speed_of_sound`
- **Sources:** US Standard Atmosphere 1976 (NASA-TM-X-74335): layered T(h), P(h), ideal-gas density; JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py)
- **Tests:** `properties/test_candidates.py`, `properties/test_golden_jsbsim.py`
- **Notes:** Verified-tier coefficients absorb ρ at identification altitude — scale by ρ/ρ_ident. Verified 2026-08-18 against executed JSBSim FGStandardAtmosphere (golden/vectors/jsbsim_isa_atmosphere.json): T/P/ρ/a at 12 altitudes to 35 kft, 5e-4 relative (imperial-vs-ICAO constant sets); inputs are geopotential altitude. Stratosphere (isothermal layer) unexercised.

### `dryden_turbulence` — **verified**

Dryden forming filters H_u/H_v/H_w + low-altitude scale/intensity closures.

- **Defined in:** `spec.wind.dryden_filter_u, spec.wind.dryden_filter_vw, spec.wind.dryden_low_altitude_scales`
- **Sources:** MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust; Yeager — Implementation and Testing of Turbulence Models for the F18-HARV Simulation, NASA CR-1998-206937, 1998. Pinned-document golden source (sha256 4f63d46d…, NTRS 19980028448): GUSTMDL ACSL listing + Tables 2-7 run statistics
- **Tests:** `properties/test_candidates.py`, `properties/test_dryden_authenticity.py`, `properties/test_golden_jsbsim.py`
- **Notes:** ⚠ discrete driving noise must be N(0, π/dt) for the published gains; 8785C vs 1797 length-scale factor-of-2 trap. Low-altitude fit is in FEET. Verified via the ARCHAIC-SOURCE EXCEPTION (2026-08-11): the golden data are NASA CR-1998-206937's published run statistics (pinned PDF, transcribed tables + listing), not vectors from executed code — the reference is ACSL (no runnable interpreter) with an irreproducible RNG. Proven: the report's Tustin difference equations are exactly the prewarped bilinear of these filters, and re-simulating them reproduces Tables 2-7 calibration (σ, u/v/w) within the published spread at L = 1750 ft, V ∈ {100, 1000} ft/s. UPGRADE 2026-08-18: the low-altitude closures (L(h), σ(h, W20), h < 1000 ft) are now ALSO pinned by executed-code vectors — JSBSim FGWinds ttTustin implements the same CR-206937 difference equations with the closures active, and its seeded runs are reproduced sample-exactly via recovered driving noise (golden/vectors/jsbsim_dryden_lowalt.json).

### `von_karman_turbulence` — *candidate*

von Kármán spectra (5/6, 11/6 exponents) + standard rational filter approximations.

- **Defined in:** `spec.wind.von_karman_psd_u`
- **Sources:** MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust
- **Tests:** `properties/test_candidates.py`
- **Notes:** Measurement-preferred; no exact finite filter.

### `discrete_gust` — *candidate*

1-cosine discrete gust ramp per axis.

- **Defined in:** `spec.wind.one_minus_cosine_gust`
- **Sources:** MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust
- **Tests:** `properties/test_candidates.py`

### `wind_shear_log` — *candidate*

MIL-F-8785C mean-wind log profile u_w = W20·ln(h/z0)/ln(20/z0) (h, z0 in ft).

- **Defined in:** `spec.wind.log_wind_shear`
- **Sources:** MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust; MathWorks Aerospace Blockset documentation — block-equation pages only (implementations are proprietary and were not consulted); each block cites the public standard it implements
- **Tests:** `properties/test_candidates.py`
- **Notes:** The deterministic member of the 8785C wind triad (shear + turbulence + gust); superposes onto v_wind. Valid 3–1000 ft AGL; z0 = 0.15 ft (Category C landing) or 2.0 ft (otherwise). Anchored to the same W20 as the Dryden closures, so mean wind and turbulence intensity stay mutually calibrated.


## Harness (timing & stateful machinery — not physics)

### `command_transport_delay` — **verified**

u_applied(t) = u_cmd(t − t_d); ring buffer of round(t_d/dt) steps.

- **Defined in:** `(harness — no symbolic form)`
- **Sources:** SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py; Eschmann et al. — Data-Driven System Identification of Quadrotors Subject to Motor Delays, 2024 (arXiv:2404.07837)
- **Notes:** SkyDreamer trains with t_d = 11 ms.

### `control_rate_zoh` — **verified**

Controller decides at a lower rate than physics; command zero-order-held between.

- **Defined in:** `(harness — no symbolic form)`
- **Sources:** Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml

### `ground_contact_heuristic` — *candidate*

Normal-force cancellation + velocity clamps at z ≤ 0 — bookkeeping, not contact physics.

- **Defined in:** `(harness — no symbolic form)`
- **Sources:** Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork)
- **Notes:** Do not port as physics; a real contact model would be a new candidate.


## Sources

- **rotorpy**: Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator with Aerodynamics for Education and Research (arXiv:2306.04485); reference implementation branch research-additions (Synetic-Labs/rotorpy fork) — <https://github.com/Synetic-Labs/rotorpy>
- **mahony2012**: Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, Estimation, and Control of Quadrotor, IEEE RAM 2012
- **crazyflow**: Crazyflow first-principles dynamics (crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml — <https://github.com/learnsyslab/crazyflow>
- **skydreamer**: SkyDreamer (arXiv:2510.14783) + reference implementation embodied/envs/skydreamer.py — <https://github.com/The-Real-Thisas/dreamerv3>
- **flightning**: Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual Features Using Differentiable Simulation, ICRA 2025; reference implementation executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28) — <https://github.com/uzh-rpg/rpg_flightning>
- **forster2015**: Förster — System Identification of the Crazyflie 2.0 Nano Quadrocopter, ETH Zürich, 2015
- **graf**: Graf — Quaternions and Dynamics (quaternion kinematics)
- **eschmann2024**: Eschmann et al. — Data-Driven System Identification of Quadrotors Subject to Motor Delays, 2024 (arXiv:2404.07837)
- **faessler2018**: Faessler, Franchi, Scaramuzza — Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories, IEEE RA-L 2018
- **jsbsim**: JSBSim flight dynamics engine (FGPropeller, FGRotor, FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit 9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 (golden/generate/gen_jsbsim.py) — <https://github.com/JSBSim-Team/jsbsim>
- **mccormick**: McCormick — Aerodynamics, Aeronautics, and Flight Mechanics, 1st ed. (momentum-theory induced velocity, Eq. 6.15)
- **sh79**: Shaughnessy, Deaux, Yenni — Development and Validation of a Piloted Simulation of a Helicopter and External Sling Load, NASA TP-1285, 1979 (JSBSim FGRotor's model basis); body-rate flap terms per Amer, NACA TN-2136, 1950
- **bramwell**: Bramwell — Helicopter Dynamics, 2nd ed., eqns 3.43-3.44 (rotor torque decomposition: profile + induced/climb components)
- **talbot1977**: Talbot & Corliss — A Mathematical Force and Moment Model of a UH-1H Helicopter for Flight Dynamics Simulations, NASA TM-73,254, 1977 (eqn 10a ground-effect inflow factor)
- **sanchez2017**: Sanchez-Cuevas, Heredia, Ollero — Characterization of the Aerodynamic Ground Effect and Its Influence in Multirotor Control, Int. J. Aerospace Eng. 2017, doi 10.1155/2017/1823056
- **cheeseman1955**: Cheeseman & Bennett — The Effect of the Ground on a Helicopter Rotor in Forward Flight, ARC R&M 3021, 1955
- **pybullet_drones**: utiasDSL/gym-pybullet-drones BaseAviary (_groundEffect, _downwash) + cf2x.urdf identified constants — <https://github.com/utiasDSL/gym-pybullet-drones>
- **jain2019**: Jain, Fortmuller, Byun, Makiharju, Mueller — Modeling of aerodynamic disturbances for proximity flight of multirotors, ICUAS 2019, Eqs. (1)-(8)
- **bangura**: Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony (ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for Quadrotors (arXiv:1601.00733)
- **kai2017**: Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13)
- **rotors_px4**: ethz-asl/rotors_simulator gazebo_motor_model.cpp + PX4/PX4-SITL_gazebo-classic variant (signed rolling moment) — <https://github.com/ethz-asl/rotors_simulator>
- **chen2006**: Chen & Rincon-Mora — Accurate Electrical Battery Model Capable of Predicting Runtime and I-V Performance, IEEE Trans. Energy Conversion 21(2), 2006
- **crazyflie_fw**: bitcraze/crazyflie-firmware — motors.c motorsCompensateBatteryVoltage + platform_defaults_cf2.h (master and tag 2022.01) — <https://github.com/bitcraze/crazyflie-firmware>
- **gazebo_battery**: gazebosim/gz-sim LinearBatteryPlugin.cc (linear OCV + internal resistance + current low-pass) — <https://github.com/gazebosim/gz-sim>
- **mil8785c**: MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust
- **ussa1976**: US Standard Atmosphere 1976 (NASA-TM-X-74335): layered T(h), P(h), ideal-gas density
- **neurobem**: Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component eqs. (5)-(19), Kingfisher platform §IV-B — <https://arxiv.org/abs/2106.08015>
- **agilicious**: Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 (Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib (golden/generate/gen_agilicious.py) — <https://github.com/alibabasomeone/agilicious_internal_mine>
- **hoffmann2007**: Hoffmann, Huang, Waslander, Tomlin — Quadrotor Helicopter Flight Dynamics and Control: Theory and Experiment, AIAA GNC 2007 (VRS empirical quartic; hinged-blade spring model)
- **gill2017**: Gill & D'Andrea — Propeller Thrust and Drag in Forward Flight, IEEE CCTA 2017 (sinusoidal high-incidence lift/drag polars; with Ducard & Hua, CCA 2014)
- **mathworks_aeroblks**: MathWorks Aerospace Blockset documentation — block-equation pages only (implementations are proprietary and were not consulted); each block cites the public standard it implements — <https://www.mathworks.com/help/aeroblks/>
- **cr206937**: Yeager — Implementation and Testing of Turbulence Models for the F18-HARV Simulation, NASA CR-1998-206937, 1998. Pinned-document golden source (sha256 4f63d46d…, NTRS 19980028448): GUSTMDL ACSL listing + Tables 2-7 run statistics — <https://ntrs.nasa.gov/citations/19980028448>

## Reviewed and excluded

- **genesis**: Genesis (Genesis-Embodied-AI): props are fixed joints, KF·rpm² force + KM·rpm² yaw torque only, no gyroscopic effects — strict subset of this spec.

See [REFERENCES.md](../REFERENCES.md) for the full per-source evaluation ledger including skipped models.
