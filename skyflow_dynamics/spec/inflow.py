"""
Rotor inflow — momentum (actuator-disk) theory — CANDIDATE tier.

The induced velocity v_i is the physically correct input for ground effect, downwash
coupling, and climb/descent thrust corrections (the principled model behind the identified
k_v2/k_angle surrogates in the verified tier). Momentum theory is INVALID in the vortex-ring
band (descent rates between ≈0.5·v_h and ≈2·v_h) — see vrs_boundaries.
"""

import sympy as sp


def hover_induced_velocity(T: sp.Expr | float, rho: sp.Expr | float,
                           A: sp.Expr | float) -> sp.Expr:
    """v_h = √(T/(2ρA)) — hover induced velocity; A = πR² disk area.
    Source: classical momentum theory (Leishman; Bangura et al. arXiv:1601.00733 Eq. (4))."""
    return sp.sqrt(T / (2 * rho * A))


def induced_velocity_axial(V_a: sp.Expr | float, T: sp.Expr | float, rho: sp.Expr | float,
                           A: sp.Expr | float) -> sp.Expr:
    """
    Induced velocity for axial flow through the disk (JSBSim's sign-safe form):

        S = V_a·|V_a| + 2T/(ρA)
        v_i = ½(−V_a + √S)   if S > 0,   else   ½(−V_a − √(−S))

    V_a: axial airspeed into the disk (m/s), v_i positive downstream. ⚠ This is the EXACT
    root of the momentum equation T = 2ρA(V_a + v_i)v_i only for V_a ≥ 0 (climb/inflow);
    for V_a < 0 the V_a·|V_a| term makes it a sign-symmetric continuation that deviates from
    the momentum equation by O((V_a/v_h)²) — acceptable in slow descent, where momentum
    theory itself is failing anyway (VRS band). Smooth except at S = 0 (use a smooth-min
    patch in differentiable backends).
    Source: McCormick, "Aerodynamics, Aeronautics, and Flight Mechanics", Eq. 6.15, as
    implemented in JSBSim FGPropeller.cpp:251–261.
    """
    S = V_a * sp.Abs(V_a) + 2 * T / (rho * A)
    return sp.Piecewise(
        ((-V_a + sp.sqrt(S)) / 2, S > 0),
        ((-V_a - sp.sqrt(-S)) / 2, True),
    )


def oblique_momentum_thrust(v_i: sp.Expr | float, V: sp.Matrix, rho: sp.Expr | float,
                            A: sp.Expr | float) -> sp.Expr:
    """
    Momentum thrust in oblique flight (implicit in v_i — solve by Newton iteration):

        T = 2ρA·v_i·U,     U = √(V_x² + V_y² + (v_i − V_z)²)

    V: rotor velocity relative to the air, body/hub frame (incl. the ω×r lever arm), with the
    source's z-down sign convention noted; hover limit U = v_i = v_h recovers T = 2ρA·v_h².
    Aerodynamic power P_a = 2ρA·v_i·U·(v_i − V_z); shaft coupling P_m = P_a/FoM + I_r·Ω·Ω̇
    (FoM ≈ 0.6–0.7 measured). This is the nonlinear T(airspeed) correction that the verified
    tier's k_v2·v_z|v_z| and AoA-factor terms linearize.
    Source: Bangura & Mahony, ACRA 2012 Eqs. (10a),(10b),(11); Bangura et al. ICRA 2014
    Eqs. (3)–(10); arXiv:1601.00733.
    """
    Vx, Vy, Vz = V.flat()
    U = sp.sqrt(Vx**2 + Vy**2 + (v_i - Vz)**2)
    return 2 * rho * A * v_i * U


def dynamic_inflow_lag(nu: sp.Expr | float, nu_eq: sp.Expr | float,
                       tau: sp.Expr | float) -> sp.Expr:
    """
    First-order dynamic inflow: the induced-inflow ratio ν relaxes to its momentum
    equilibrium ν_eq = C_T / (2√(μ² + λ²)) with time constant τ ≈ 16/(γΩ) (γ = Lock number):

        ν̇ = (ν_eq − ν)/τ,    exact step  ν⁺ = (ν − ν_eq)·e^(−dt/τ) + ν_eq

    — the same exact-exponential discretization pattern as the motor lag (spec.motor).
    Source: JSBSim FGRotor.cpp (Glauert equilibrium + SH79 blade-element C_T, GE49 time
    constant); Gaonkar & Peters dynamic-inflow literature.
    """
    return sp.sympify((nu_eq - nu) / tau)


def inflow_thrust_sensitivity(theta0: sp.Expr | float, lam_h: sp.Expr | float) -> sp.Expr:
    """
    Sensitivity of a fixed-pitch rotor's thrust to the flow through its disk, from blade-
    element theory with constant pitch and chord, C_T = ½·σ·C_lα·(θ₀/3 − λ/2):

        s = λ_h / (2θ₀/3 − λ_h) = 3λ_h / (2θ₀ − 3λ_h)

    θ₀: blade pitch from zero lift (rad); λ_h = v_i/(ΩR): static inflow ratio out of ground
    effect. s is the fraction of static thrust lost per unit relative rise of the inflow. A
    low-pitch or heavily loaded rotor (λ_h close to 2θ₀/3) is the most sensitive; s does not
    depend on rotor speed for a fixed-pitch rotor (λ_h is then a constant of the blade).
    Source: He & Leang, AIAA J. 2020 (doi 10.2514/1.J059223), Eqs. (4) and (15), where
    s = C_a; the same coefficient is ¼·(ρ·a·b·c·Ω²R³/T_∞)·λ in Cheeseman & Bennett, ARC R&M
    3021 (1955), Eqs. (5)–(6). Measured C_a for eleven fixed-pitch propellers of 3.5 to 16
    inches: 0.27–0.63 (He & Leang Table 2; 0.53 for a three-blade 5×3).
    """
    return sp.sympify(3 * lam_h / (2 * theta0 - 3 * lam_h))


def inflow_thrust_ratio(inflow_ratio: sp.Expr | float, s: sp.Expr | float) -> sp.Expr:
    """
    Thrust of a fixed-pitch rotor at FIXED rotor speed when the flow through its disk
    changes, over its static thrust out of ground effect:

        T / T_static = 1 − s·(λ/λ_h − 1),     λ/λ_h = (V_z + v_i) / v_i,static

    inflow_ratio = λ/λ_h: the net axial flow through the disk (axial airspeed V_z of the rotor,
    positive in climb, plus the induced velocity v_i) over the static induced velocity at
    the same rotor speed. s: inflow_thrust_sensitivity. One law, several effects — each is
    a model of the inflow, and they add in the inflow (never multiply thrust ratios):
      - ground effect: λ/λ_h = 1 − δv_i/v_i (spec.ground_effect.ground_inflow_fraction_*);
      - climb and slow descent: momentum theory (induced_velocity_axial);
      - descent through the vortex ring state: spec.descent (net inflow above 1, thrust loss);
      - another vehicle's downwash or a vertical gust: added to V_z.
    The verified linear inflow term T = c_T·Ω² − k_z·Ω·V_z (rotor_drag_hforce, k_z channel) is
    this law with the induced velocity FROZEN at its static value: k_z = s·√(2ρA·c_T)/κ, κ the
    ratio of real to ideal induced velocity. With the induced velocity following momentum
    theory the small-speed slope is s/(2 + s) of thrust per v_h instead of s (κ = 1).
    Linear in λ: valid while the blade is not stalled. A fast climb unloads the blade and is
    inside the law; reduced inflow (ground effect, the first part of a descent) loads it
    and stall then caps the gain — He & Leang warn of this below z/R = 0.5.
    Source: as inflow_thrust_sensitivity; Veismann, Dougherty, Gharib, Flow 3 E7 (2023),
    Eqs. (2.4)–(2.5) use the same blade-element relation to turn measured thrust into inflow.
    """
    return sp.sympify(1 - s * (inflow_ratio - 1))


def tandem_interference_factor(p: sp.Expr | float, alpha: sp.Expr | float,
                               gamma: sp.Expr | float) -> sp.Expr:
    """
    Downwash that a FRONT rotor's wake adds at the centre of the rotor behind it, over the
    front rotor's own induced velocity (horseshoe-vortex wake, Biot–Savart):

        k = [√(1 + p²/cos²α) + p·cos γ − p·tan α·sin γ]
            / [√(1 + p²/cos²α) · (1 + p²·cos²γ·(tan α + tan γ)²)]

    p = l/R: hub spacing along the flight direction over the rotor radius; tan α = h/l: the
    rear hub's height above the front disk plane measured across the wake (nose-down disk
    incidence in level flight); γ: the front rotor's wake angle from its disk plane,
    tan γ = λ/μ (γ = 90° − χ, χ the skew angle; γ = 90° in hover and axial flight). The
    rear rotor then sees the inflow λ' = (V·sin α + (k + 1)·v)/(ΩR): add k·v to its axial
    flow and use spec.inflow.inflow_thrust_ratio. For a flat wake through the rear disk
    (γ, α → 0) k = 1 + p/√(1 + p²), which tends to 2 (the far-wake doubling); for a wake that
    goes straight down (γ = 90°) k = 1/√(1 + p²) at α = 0 — 1.94 and 0.11 at l/R = 2.83.
    Measured (wind tunnel, R = 0.12 m, V = 6 m/s, α = −5°): rear-to-front thrust 0.89 on a
    quadrotor at l/R = 2.83 against 0.86 from this factor; "good approximation" for l/R of
    2.8–3.6, too strong a spacing trend below; no interference in axial climb. Validated on
    thrust only — the rear rotor's hub moment also changes (about −50 % in CFD, Healy et al.
    2022) and nothing here models that.
    Source: Nguyen, Liu, Mori, "Experimental Study for Aerodynamic Performance of Quadrotor
    Helicopter", Trans. Japan Soc. Aero. Space Sci. 61(1), 2018, Eqs. (35)–(40), read from
    the paper.
    """
    root = sp.sqrt(1 + p ** 2 / sp.cos(alpha) ** 2)
    num = root + p * sp.cos(gamma) - p * sp.tan(alpha) * sp.sin(gamma)
    den = root * (1 + p ** 2 * sp.cos(gamma) ** 2 * (sp.tan(alpha) + sp.tan(gamma)) ** 2)
    return num / den


#: Descent-regime boundaries in units of the hover induced velocity v_h. Between VRS onset
#: and the windmill-brake state momentum theory is invalid (thrust fluctuation, uncommanded
#: roll/pitch); above 2·v_h descent it applies again with T = −2ρA(−V_z + v_i)·v_i.
#: Source: Bangura & Mahony ACRA 2012 §5.2; arXiv:1601.00733 §2 (VRS/TWS/WBS).
VRS_ONSET_DESCENT_RATE = sp.Rational(1, 2)   # × v_h
WINDMILL_BRAKE_DESCENT_RATE = 2              # × v_h
