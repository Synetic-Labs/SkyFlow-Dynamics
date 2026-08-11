"""
Atmospheric wind — Dryden / von Kármán turbulence, discrete gusts, mean-wind shear —
CANDIDATE tier.

Wind is an exogenous velocity process: the outputs below superpose onto v_wind and enter
the dynamics only through the airspeed v_a = Rᵀ(v − v_wind) (verified tier). The
stochastic generation itself (white-noise driving, resample-and-hold) is harness-side; the
math here is the published filter/spectrum definitions with all unit pitfalls stated.

Convention note: MIL-F-8785C and MIL-HDBK-1797 differ by a factor of 2 in the length-scale
definition (L_1797 = L_8785C/2 for v, w). Everything below states which convention it uses.
"""

import sympy as sp

#: Laplace variable for the forming filters.
s = sp.Symbol("s")


def dryden_filter_u(sigma_u: sp.Expr, L_u: sp.Expr, V: sp.Expr) -> sp.Expr:
    """
    Longitudinal forming filter (MIL-F-8785C), unit-PSD white noise → gust velocity u_g:

        H_u(s) = σ_u·√(2L_u/(πV)) · 1/(1 + (L_u/V)·s)

    V is the vehicle airspeed carrying it through the frozen turbulence field.
    Source: MIL-F-8785C; Beard & McLain 2012 §4.4 (monic small-UAV form is identical).
    """
    return sigma_u * sp.sqrt(2 * L_u / (sp.pi * V)) / (1 + (L_u / V) * s)


def dryden_filter_vw(sigma: sp.Expr, L: sp.Expr, V: sp.Expr) -> sp.Expr:
    """
    Lateral/vertical forming filter (MIL-F-8785C):

        H(s) = σ·√(L/(πV)) · (1 + √3·(L/V)·s) / (1 + (L/V)·s)²

    |H(jω)|² reproduces the published Φ_v,w spectra exactly (|1 + j√3·x|² = 1 + 3x²).
    ⚠ Driving-noise scaling: for these gains the discrete driving samples must be
    N(0, π/dt) — i.i.d. unit-variance noise UNDERSCALES the output variance (a real
    implementation trap; the σ² normalization audit is part of this candidate).
    Source: MIL-F-8785C / MIL-HDBK-1797 (2L convention differs — see module docstring).
    """
    return sigma * sp.sqrt(L / (sp.pi * V)) * (1 + sp.sqrt(3) * (L / V) * s) \
        / (1 + (L / V) * s)**2


def dryden_low_altitude_scales(h: sp.Expr, W20: sp.Expr) -> tuple:
    """
    Low-altitude closures (10 ft < h < 1000 ft, h in FEET — the published fit is imperial):

        L_w = h,   L_u = L_v = h/(0.177 + 0.000823·h)^1.2
        σ_w = 0.1·W20,   σ_u = σ_v = σ_w/(0.177 + 0.000823·h)^0.4

    W20 = mean wind at 20 ft AGL (light 15 kt, moderate 30 kt, severe 45 kt). Above 2000 ft:
    L = 1750 ft; 1000–2000 ft: interpolate. Returns (L_u, L_v, L_w, σ_u, σ_v, σ_w).
    Source: MIL-F-8785C low-altitude model.
    """
    f = (0.177 + 0.000823 * h)
    L_u = h / f**sp.Rational(6, 5)
    sigma_w = 0.1 * W20
    sigma_u = sigma_w / f**sp.Rational(2, 5)
    return L_u, L_u, h, sigma_u, sigma_u, sigma_w


def von_karman_psd_u(omega_spatial: sp.Expr, sigma_u: sp.Expr, L_u: sp.Expr) -> sp.Expr:
    """
    von Kármán longitudinal spatial PSD (MIL-F-8785C convention, a = 1.339):

        Φ_u(Ω) = σ_u²·(2L_u/π) / (1 + (1.339·L_u·Ω)²)^(5/6)

    The measurement-preferred spectrum; irrational exponents (5/6, 11/6) mean no exact
    finite-dimensional filter — use the standard rational approximations for generation.
    Source: MIL-F-8785C.
    """
    return sigma_u**2 * (2 * L_u / sp.pi) / (1 + (1.339 * L_u * omega_spatial)**2)**sp.Rational(5, 6)


def log_wind_shear(h: sp.Expr, W20: sp.Expr, z0: sp.Expr) -> sp.Expr:
    """
    MIL-F-8785C mean-wind (shear) profile — magnitude logarithmic in height AGL:

        u_w(h) = W20 · ln(h/z0) / ln(20/z0),   valid 3 ft < h < 1000 ft (h, z0 in FEET —
                                               the published fit is imperial, like the
                                               low-altitude turbulence closures)

    W20 = mean wind at 20 ft AGL — the same severity parameter that anchors the Dryden
    intensities (σ_w = 0.1·W20), so shear and turbulence stay mutually calibrated.
    z0 = surface roughness length: 0.15 ft (Category C landing phase), 2.0 ft otherwise.
    Direction is constant with height; u_w superposes onto v_wind together with the
    turbulence and gust outputs (the deterministic member of the 8785C triad).
    Source: MIL-F-8785C para 3.7.3.3; reproduced in the MathWorks Aerospace Blockset
    'Wind Shear Model' block documentation.
    """
    return W20 * sp.log(h / z0) / sp.log(20 / z0)


def one_minus_cosine_gust(x: sp.Expr, V_m: sp.Expr, d_m: sp.Expr) -> sp.Expr:
    """
    MIL-F-8785C discrete gust, per axis; x = distance penetrated into the gust:

        V(x) = 0 (x<0);  (V_m/2)(1 − cos(πx/d_m)) (0≤x≤d_m);  V_m (x>d_m)

    C¹-continuous ramp to amplitude V_m over gradient distance d_m.
    """
    return sp.Piecewise((0, x < 0),
                        (V_m / 2 * (1 - sp.cos(sp.pi * x / d_m)), x <= d_m),
                        (V_m, True))
