"""
Rotor inflow — momentum (actuator-disk) theory — CANDIDATE tier.

The induced velocity v_i is the physically correct input for ground effect, downwash
coupling, and climb/descent thrust corrections (the principled model behind the identified
k_v2/k_angle surrogates in the verified tier). Momentum theory is INVALID in the vortex-ring
band (descent rates between ≈0.5·v_h and ≈2·v_h) — see vrs_boundaries.
"""

import sympy as sp


def hover_induced_velocity(T: sp.Expr, rho: sp.Expr, A: sp.Expr) -> sp.Expr:
    """v_h = √(T/(2ρA)) — hover induced velocity; A = πR² disk area.
    Source: classical momentum theory (Leishman; Bangura et al. arXiv:1601.00733 Eq. (4))."""
    return sp.sqrt(T / (2 * rho * A))


def induced_velocity_axial(V_a: sp.Expr, T: sp.Expr, rho: sp.Expr, A: sp.Expr) -> sp.Expr:
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


def oblique_momentum_thrust(v_i: sp.Expr, V: sp.Matrix, rho: sp.Expr, A: sp.Expr) -> sp.Expr:
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
    U = sp.sqrt(V[0]**2 + V[1]**2 + (v_i - V[2])**2)
    return 2 * rho * A * v_i * U


def dynamic_inflow_lag(nu: sp.Expr, nu_eq: sp.Expr, tau: sp.Expr) -> sp.Expr:
    """
    First-order dynamic inflow: the induced-inflow ratio ν relaxes to its momentum
    equilibrium ν_eq = C_T / (2√(μ² + λ²)) with time constant τ ≈ 16/(γΩ) (γ = Lock number):

        ν̇ = (ν_eq − ν)/τ,    exact step  ν⁺ = (ν − ν_eq)·e^(−dt/τ) + ν_eq

    — the same exact-exponential discretization pattern as the motor lag (spec.motor).
    Source: JSBSim FGRotor.cpp (Glauert equilibrium + SH79 blade-element C_T, GE49 time
    constant); Gaonkar & Peters dynamic-inflow literature.
    """
    return (nu_eq - nu) / tau


#: Descent-regime boundaries in units of the hover induced velocity v_h. Between VRS onset
#: and the windmill-brake state momentum theory is invalid (thrust fluctuation, uncommanded
#: roll/pitch); above 2·v_h descent it applies again with T = −2ρA(−V_z + v_i)·v_i.
#: Source: Bangura & Mahony ACRA 2012 §5.2; arXiv:1601.00733 §2 (VRS/TWS/WBS).
VRS_ONSET_DESCENT_RATE = sp.Rational(1, 2)   # × v_h
WINDMILL_BRAKE_DESCENT_RATE = 2              # × v_h
