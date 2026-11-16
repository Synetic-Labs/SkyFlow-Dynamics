"""
Ground effect and vehicle-to-vehicle downwash — CANDIDATE tier.

Credible published models, symbolically checked and cited; not yet validated against a
runnable reference by golden vectors (the reference implementations we golden against model
neither effect). Promotion path: bench or flight data at constant rotor speed.

All heights z are the rotor-plane height above ground along the ground normal (m); R is the
rotor radius (m). WHAT IS HELD CONSTANT differs between the forms and decides which one
applies: cheeseman_bennett, cheeseman_bennett_forward and sanchez_cuevas are at constant
POWER; he_leang_exponential and cheeseman_bennett_fixed_pitch are at constant ROTOR SPEED
— the condition of an ESC-driven fixed-pitch propeller, and the forms measurement supports
there (REFERENCES.md, F-33). The ground acts on the induced velocity: the constant-speed
forms are spec.inflow.inflow_thrust_ratio over a ground_inflow_fraction_*. Use ONE route
for a rotor, never two.
"""

import sympy as sp


def cheeseman_bennett(z: sp.Expr | float, R: sp.Expr | float) -> sp.Expr:
    """
    Single-rotor in-ground-effect thrust ratio (potential flow, method of images):

        T_IGE / T_OGE = 1 / (1 − (R/(4z))²)

    Valid experimentally for 0.5 ≤ z/R ≤ 2 (≲1.6% above 2R); singular at z = R/4 — clamp
    z ≥ ~0.5R in any implementation. Applied per rotor with each rotor's own z it also yields
    the partial-ground-effect moment near ledges.
    Source: Cheeseman & Bennett, ARC R&M 3021, 1955 (as Eqs. (1)–(2) of Sanchez-Cuevas 2017).
    ⚠ Known to UNDER-predict for a full multirotor (fountain effect) — see sanchez_cuevas.
    """
    return sp.sympify(1 / (1 - (R / (4 * z))**2))


def cheeseman_bennett_forward(z: sp.Expr | float, R: sp.Expr | float, V: sp.Expr | float, v_i: sp.Expr | float) -> sp.Expr:
    """
    Forward-flight generalization — ground effect washes out with horizontal speed V:

        T_IGE / T_OGE = 1 / (1 − (R/(4z))² · 1/(1 + (V/v_i)²)),   v_i = √(T/(2ρA)) hover
    At CONSTANT POWER; Z is the height of the rotor hub; the washout comes from a directional
    image source whose strength is fixed to agree with the hover form. The report's own
    flight tests "agree fairly well with the theory" above Z/R = 0.6, but the measured
    effect falls off faster with speed at first.
    Source: Cheeseman & Bennett, ARC R&M 3021 (1955), Eq. (4), read from the report
    (2026-11-16). For a fixed-pitch rotor at fixed speed use cheeseman_bennett_fixed_pitch.
    """
    return sp.sympify(1 / (1 - (R / (4 * z))**2 / (1 + (V / v_i)**2)))


def ground_inflow_fraction_image(z: sp.Expr | float, R: sp.Expr | float,
                                 V: sp.Expr | float = 0, v_i: sp.Expr | float = 1) -> sp.Expr:
    """
    Fraction of the induced velocity that the ground cancels at the rotor, image-source model:

        δv_i / v_i = (R/(4z))² / (1 + (V/v_i)²)

    (δv_i = A·v_i/(16πZ²) at hover.) Feed it to spec.inflow.inflow_thrust_ratio as
    λ/λ_h = 1 − δv_i/v_i. Source: Cheeseman & Bennett, ARC R&M 3021 (1955), §2.1 and Eq. (3)
    with k = 1/3, read from the report.
    """
    return sp.sympify((R / (4 * z))**2 / (1 + (V / v_i)**2))


def ground_inflow_fraction_exponential(z: sp.Expr | float, R: sp.Expr | float,
                                       C_b: sp.Expr | float) -> sp.Expr:
    """
    The same fraction, exponential in height and finite at the ground:

        δv_i / v_i = exp(−C_b·z/R)

    with δv_i = v_i at z = 0 (no flow through a rotor lying on the ground) — He & Leang's
    Assumption 1 and Eq. (16). C_b is fitted: 1.18–2.87 on eleven fixed-pitch propellers
    (mean 2.3; 1.80 for a three-blade 5×3); a rotor inside a quadrotor reaches higher (C_b up
    to 30 % lower, their §IV.C). Source: He & Leang, AIAA J. 2020, Eq. (16), Table 2.
    """
    return sp.exp(-C_b * z / R)


def he_leang_exponential(z: sp.Expr | float, R: sp.Expr | float, C_a: sp.Expr | float,
                         C_b: sp.Expr | float) -> sp.Expr:
    """
    In-ground-effect thrust ratio of a fixed-pitch rotor at CONSTANT ROTOR SPEED — the
    condition of an ESC-driven propeller, where the ratios above are at constant power:

        K_G = T(z)/T_∞ = 1 + C_a·exp(−C_b·z/R)

    Finite everywhere: 1 + C_a at the ground. C_a is the inflow sensitivity of the blade
    (spec.inflow.inflow_thrust_sensitivity), so this IS spec.inflow.inflow_thrust_ratio over
    ground_inflow_fraction_exponential. Measured on a thrust stand at constant speed and 13
    heights, eleven propellers: C_a 0.27–0.63, C_b 1.18–2.87; thrust rose monotonically to
    the ground with no drop. Three-blade 5×3 (R = 0.064 m): C_a = 0.53, C_b = 1.80, i.e.
    +21.5 % / +8.8 % / +1.4 % at z/R = 0.5 / 1 / 2.
    Source: He & Leang, "Quasi-Steady In-Ground-Effect Model for Single and Multirotor Aerial
    Vehicles", AIAA J. 2020, doi 10.2514/1.J059223, Eqs. (3), (15), (16), Table 2 — read from
    the author-hosted paper. Not covered: the body of a real vehicle (which adds lift, see
    sanchez_cuevas) and the thrust-LOSS region the paper reports for bare rotor pairs at
    tip gaps of 1.2–2.8 R and heights of 1.1–2.3 R (fountain re-ingestion, up to 8 %).
    """
    return sp.sympify(1 + C_a * ground_inflow_fraction_exponential(z, R, C_b))


def cheeseman_bennett_fixed_pitch(z: sp.Expr | float, R: sp.Expr | float, s: sp.Expr | float,
                                  V: sp.Expr | float = 0, v_i: sp.Expr | float = 1) -> sp.Expr:
    """
    Cheeseman & Bennett's own FIXED-PITCH, fixed-rotor-speed form — in the report, dropped by
    the literature that cites it:

        T_g/T_∞ = 1 + s·(R/(4z))² / (1 + (V/v_i)²),   s = ¼·(ρ·a·b·c·Ω²R³/T_∞)·λ

    with T_∞ = ½·ρ·a·b·c·Ω²R³·(θ/3 − λ/2), so s = λ/(2θ/3 − λ): the same inflow sensitivity
    as He & Leang's C_a. The blade-loading dependence the constant-power form lacks: a
    lightly loaded high-pitch rotor (small s) gains less. Singular only at z = 0; the
    report warns that the point-source image is poor at small Z/R.
    Source: Cheeseman & Bennett, ARC R&M 3021 (1955), Eqs. (5)–(6), read from the report.
    """
    return sp.sympify(1 + s * ground_inflow_fraction_image(z, R, V, v_i))


def sanchez_cuevas(z: sp.Expr | float, R: sp.Expr | float, d: sp.Expr | float, b: sp.Expr | float, K_b: sp.Expr | float) -> sp.Expr:
    """
    Quadrotor ground effect with mirrored-rotor interference and body lift (fountain effect):

        T_IGE/T_OGE = 1 / ( 1 − (R/(4z))² − R²·z/(d² + 4z²)^{3/2}
                              − (R²/2)·z/(2d² + 4z²)^{3/2} − 2R²·K_b·z/(b² + 4z²)^{3/2} )

    d: adjacent-rotor axis spacing (m); b: diagonal spacing (= √2·d square layout); K_b ≈ 2
    (fitted, PQUAD quadrotor R = 0.12 m). Significant to z ≈ 5R (vs 2R single-rotor).
    K_b = 0 recovers their Eq. (3) (pure image model); d, b → ∞ recovers Cheeseman–Bennett.
    Source: Sanchez-Cuevas, Heredia, Ollero, Int. J. Aerospace Eng. 2017, doi
    10.1155/2017/1823056, Eqs. (3)–(4). Validated hover, all rotors coplanar.
    """
    return 1 / (1 - (R / (4 * z))**2
                - R**2 * z / (d**2 + 4 * z**2)**sp.Rational(3, 2)
                - (R**2 / 2) * z / (2 * d**2 + 4 * z**2)**sp.Rational(3, 2)
                - 2 * R**2 * K_b * z / (b**2 + 4 * z**2)**sp.Rational(3, 2))


def talbot_inflow_factor(h: sp.Expr | float, k_ge: sp.Expr | float, h_shift: sp.Expr | float,
                         load: sp.Expr | float) -> sp.Expr:
    """
    In-ground-effect INFLOW reduction, exponential in height (JSBSim FGRotor form):

        f_GE = 1 − load · exp(−k_ge·(h + h_shift)),      applied as  v_i ← f_GE · v_i

    Unlike the thrust-ratio models above, this scales the induced velocity: reduced inflow
    raises the effective blade incidence, so thrust at fixed Ω rises — the more physical
    route. Apply f_GE ONCE to the quasi-steady inflow (the ν_eq equilibrium of
    spec.inflow.dynamic_inflow_lag, or the output v_i). ⚠ Do NOT copy JSBSim's variant,
    which multiplies the persistent inflow STATE every step — compounding through the lag
    filter, that is timestep-dependent with fixed point ν* = f·ν_eq·(1−e)/(1−f·e),
    e = e^(−dt/τ), far below f·ν_eq for small dt. load is a rotor-loading × runtime scale
    (JSBSim: RPM/NominalRPM — a startup guard that can exceed 1, with the f_GE clamp doing
    the real bounding — times a property used as a forward-flight kill-switch); take
    load ∈ [0, 1] here, load = 0 giving f_GE = 1 (OGE). k_ge ≈ 0.13–0.33 1/m (source docs:
    0.04–0.1 1/ft, small rotors at the high end; effect vanishes ~2–3 rotor diameters up);
    h_shift ≈ hub height above the ground contact point. Implementation lessons that
    transfer: clamp f_GE to [0.5, 1] and low-pass h so terrain steps don't produce force
    discontinuities (JSBSim's comment says 1 Hz, but Filter(1.0, dt) places the pole at
    1 rad/s ≈ 0.16 Hz — match the behavior, not the comment).
    Source: JSBSim FGRotor.cpp CalcRotorState (master @ 9a0b028, lines 639–646); the
    inflow-scaling idea credited there to Talbot & Corliss, NASA TM-73,254 (1977), eqn 10a.
    """
    return 1 - load * sp.exp(-k_ge * (h + h_shift))


def pybullet_ground_effect(T_i: sp.Expr | float, z_i: sp.Expr | float, R: sp.Expr | float, G: sp.Expr | float) -> sp.Expr:
    """
    gym-pybullet-drones per-rotor additive thrust increment (linearized Cheeseman–Bennett):

        ΔT_i = T_i · G · (R/(4·z_i))²,       along the rotor axis at hub i.

    The first-order expansion of a Cheeseman-Bennett-type ratio with the multirotor
    amplification folded into G (identified G = 11.37 for the Crazyflie 2.x — ≫1, encoding
    the fountain effect). Needs a height clip (theirs caps ΔT at max-thrust/15); disabled
    beyond |roll|,|pitch| ≥ π/2.
    Source: utiasDSL/gym-pybullet-drones BaseAviary._groundEffect (L688–715) + cf2x.urdf;
    model form from Shi et al. 2019 (Neural-Lander), Eq. (15).
    """
    return sp.sympify(T_i * G * (R / (4 * z_i))**2)


def pybullet_downwash_force(dz: sp.Expr | float, dxy: sp.Expr | float, R: sp.Expr | float,
                            c1: sp.Expr | float, c2: sp.Expr | float, c3: sp.Expr | float) -> sp.Expr:
    """
    Empirical vehicle-to-vehicle downwash: downward force on the lower vehicle's CoM from a
    vehicle hovering Δz above at horizontal offset Δxy:

        F_z = −α · exp(−½ (Δxy/β)²),   α = c1·(R/(4Δz))²,   β = c2·Δz + c3

    Identified (Crazyflie 2.x): c1 = 2267.18 N, c2 = 0.16, c3 = −0.11 m. Fit is NOT
    trustworthy below Δz ≈ 0.7 m (β ≤ 0 for Δz ≤ 0.69 m) and is independent of the upper
    vehicle's thrust — CF2-specific.
    Source: gym-pybullet-drones BaseAviary._downwash (L760–779), DSL experiments (SiQi Zhou).
    """
    alpha = c1 * (R / (4 * dz))**2
    beta = c2 * dz + c3
    return -alpha * sp.exp(-sp.Rational(1, 2) * (dxy / beta)**2)


def jain_wake_velocity(z: sp.Expr | float, r: sp.Expr | float, T: sp.Expr | float, rho: sp.Expr | float, A_p: sp.Expr | float,
                       L: sp.Expr | float, z0: sp.Expr | float, c_ax: sp.Expr | float, c_rad: sp.Expr | float) -> sp.Expr:
    """
    Physically grounded downwash: the upper vehicle's wake as an axisymmetric turbulent jet
    (zone of established flow, z > 3L), axial velocity at axial/radial separation (z, r):

        V(z, r) = √(T/(2ρA_p)) · c_ax·L/(z − z0) · exp(−c_rad·(r/(z − z0))²)

    T: per-propeller thrust of the UPPER vehicle (thrust-scaling, unlike the pybullet fit);
    L = vehicle size (2× arm length); fitted c_ax ≈ 4.7–4.9 (matches published turbulent-jet
    values), c_rad ≈ 25–61, z0 ≈ 0.03–0.11 m (vehicle-dependent).
    Coupling to the lower vehicle: (a) flat-plate frame drag ½·C_D·ρ·V²·A (C_D = 1.18) with
    its lever-arm moment, and (b) per-rotor thrust loss ΔT_i = −b_v·V(z_i, r_i)·Ω_i² — route
    V through ONE inflow path (add to the local air velocity seen by the existing rotor
    models) to avoid double-counting with the AoA/advance-ratio thrust correction.
    Source: Jain, Fortmuller, Byun, Makiharju, Mueller, ICUAS 2019, Eqs. (1)–(8).
    """
    v_i = sp.sqrt(T / (2 * rho * A_p))
    return v_i * c_ax * L / (z - z0) * sp.exp(-c_rad * (r / (z - z0))**2)
