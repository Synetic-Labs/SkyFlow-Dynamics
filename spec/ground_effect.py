"""
Ground effect and vehicle-to-vehicle downwash — CANDIDATE tier.

Credible published models, symbolically checked and cited; not yet validated against a
runnable reference by golden vectors (the reference implementations we golden against model
neither effect). Promotion path: freeze vectors from gym-pybullet-drones or bench data.

All heights z are the rotor-plane height above ground along the ground normal (m); R is the
rotor radius (m). Ratios are multiplicative on thrust at fixed rotor speed/power.
"""

import sympy as sp


def cheeseman_bennett(z: sp.Expr, R: sp.Expr) -> sp.Expr:
    """
    Single-rotor in-ground-effect thrust ratio (potential flow, method of images):

        T_IGE / T_OGE = 1 / (1 − (R/(4z))²)

    Valid experimentally for 0.5 ≤ z/R ≤ 2 (≲1.6% above 2R); singular at z = R/4 — clamp
    z ≥ ~0.5R in any implementation. Applied per rotor with each rotor's own z it also yields
    the partial-ground-effect moment near ledges.
    Source: Cheeseman & Bennett, ARC R&M 3021, 1955 (as Eqs. (1)–(2) of Sanchez-Cuevas 2017).
    ⚠ Known to UNDER-predict for a full multirotor (fountain effect) — see sanchez_cuevas.
    """
    return 1 / (1 - (R / (4 * z))**2)


def cheeseman_bennett_forward(z: sp.Expr, R: sp.Expr, V: sp.Expr, v_i: sp.Expr) -> sp.Expr:
    """
    Forward-flight generalization — ground effect washes out with horizontal speed V:

        T_IGE / T_OGE = 1 / (1 − (R/(4z))² · 1/(1 + (V/v_i)²)),   v_i = √(T/(2ρA)) hover
    Source: Cheeseman & Bennett 1955 (commonly cited form; verify against R&M 3021 before
    promotion — flagged by the intake sweep).
    """
    return 1 / (1 - (R / (4 * z))**2 / (1 + (V / v_i)**2))


def sanchez_cuevas(z: sp.Expr, R: sp.Expr, d: sp.Expr, b: sp.Expr, K_b: sp.Expr) -> sp.Expr:
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


def pybullet_ground_effect(T_i: sp.Expr, z_i: sp.Expr, R: sp.Expr, G: sp.Expr) -> sp.Expr:
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
    return T_i * G * (R / (4 * z_i))**2


def pybullet_downwash_force(dz: sp.Expr, dxy: sp.Expr, R: sp.Expr,
                            c1: sp.Expr, c2: sp.Expr, c3: sp.Expr) -> sp.Expr:
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


def jain_wake_velocity(z: sp.Expr, r: sp.Expr, T: sp.Expr, rho: sp.Expr, A_p: sp.Expr,
                       L: sp.Expr, z0: sp.Expr, c_ax: sp.Expr, c_rad: sp.Expr) -> sp.Expr:
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
