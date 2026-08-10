"""
Atmosphere and advance-ratio propeller model — CANDIDATE tier.

Air density enters rotor thrust/torque/power LINEARLY (T ∝ ρ); the verified tier's constant
coefficients (ct2 ≡ k_η etc.) absorb ρ at the identification altitude. For altitude-varying
simulation, scale rotor coefficients by ρ(h)/ρ_ident.
Sources: US Standard Atmosphere 1976 (as implemented in JSBSim FGStandardAtmosphere);
JSBSim FGPropeller (advance-ratio model, explicit ρ scaling).
"""

import sympy as sp

#: SI constants (USSA-1976).
R_DRY = 287.0528    # J/(kg·K)
GAMMA_AIR = sp.Rational(14, 10)
G0 = 9.80665        # m/s²
T0 = 288.15         # K sea level
P0 = 101325.0       # Pa sea level
LAPSE_TROPO = -0.0065  # K/m (troposphere, h < 11 km)


def temperature_troposphere(h: sp.Expr) -> sp.Expr:
    """T(h) = T0 + L·h, L = −6.5 K/km, valid to 11 km (geopotential altitude)."""
    return T0 + LAPSE_TROPO * h


def pressure_gradient_layer(h: sp.Expr, T_b: sp.Expr = T0, P_b: sp.Expr = P0,
                            L_b: sp.Expr = LAPSE_TROPO, h_b: sp.Expr = 0) -> sp.Expr:
    """USSA-1976 Eq. 33a (gradient layer): P = P_b·(T_b/(T_b + L_b(h−h_b)))^(g0/(R·L_b))."""
    return P_b * (T_b / (T_b + L_b * (h - h_b)))**(G0 / (R_DRY * L_b))


def density(P: sp.Expr, T: sp.Expr) -> sp.Expr:
    """Ideal gas: ρ = P/(R_dry·T). Sea level: 1.225 kg/m³."""
    return P / (R_DRY * T)


def speed_of_sound(T: sp.Expr) -> sp.Expr:
    """a = √(γ·R_dry·T)."""
    return sp.sqrt(GAMMA_AIR * R_DRY * T)


def advance_ratio(V_a: sp.Expr, n: sp.Expr, D: sp.Expr) -> sp.Expr:
    """J = V_a/(n·D): axial hub airspeed over (rev/s × diameter). JSBSim guards n ≥ 0.01
    rev/s (its J = V/D fallback below that is a numerical hack — do not copy)."""
    return V_a / (n * D)


def propeller_thrust(C_T, J: sp.Expr, rho: sp.Expr, n: sp.Expr, D: sp.Expr) -> sp.Expr:
    """
    Classic fixed-pitch propeller parameterization with measured coefficient tables:

        T = C_T(J) · ρ · n² · D⁴        P = C_P(J) · ρ · n³ · D⁵,   Q = P/Ω

    C_T, C_P from wind-tunnel tables (UIUC/APC databases cover small UAV props). Windmilling
    falls out of the table sign (C_T < 0 at high/negative J) — no special-casing. This
    generalizes the verified tier's polynomial T(Ω) (a fixed-J slice) and the AoA-factor's
    axial part; note J uses ONLY axial inflow — edgewise flow needs the k_hor-style term.
    Table lookup is JAX-friendly (interpolation). Pass C_T as a sympy Function or callable.
    Source: JSBSim FGPropeller.cpp:220–249, 307–372 (commit 9a0b028); Stevens & Lewis §8.2.
    """
    return C_T(J) * rho * n**2 * D**4
