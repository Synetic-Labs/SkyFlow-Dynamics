"""
Blade-element-momentum (BEM) single-rotor model — NeuroBEM / agilicious `agilib`.

The stall-capable, high-advance-ratio rotor model of Bauersfeld et al. (RSS 2021): each
blade element carries sinusoidal lift/drag polars valid at any angle of attack, the disk
loads are azimuth/radius integrals of the element loads, and the induced velocity is the
root of the blade-element thrust against the oblique momentum-theory thrust
(spec.inflow.oblique_momentum_thrust). Everything here is smooth (sin/cos/atan2/sqrt), so
the terms fit the differentiable-model goal directly; the only non-smooth pieces are the
vortex-ring-state gate and its max/min blend, which are documented on the term.

Frames and signs (the reference's propeller frame P, restated for canonical FLU):

- Per-rotor hub velocity relative to the air, body FLU: v = R(q)ᵀ·(v_W − w_W) + ω × r_i.
- v_hor = √(v_x² + v_y²) ≥ 0; v_ver = −v_z (descent-positive, i.e. positive when the hub
  moves downward through the air); v_i = induced velocity, positive down through the disk.
- Blade azimuth ψ is measured from the downwind blade in the direction of rotation; the
  in-plane flow direction enters the composition via χ = atan2(v_y, v_x) (FLU).
- Flapping angles (a0 coning, a1 longitudinal, b1 lateral) follow the reference's P-frame
  definitions and are INPUTS here: the reference computes them from machine-generated
  vehicle-specific rational fits that are deliberately NOT adopted (see REFERENCES.md);
  the executed reference also zeroes them while integrating the disk loads.

Executed reference: agilib ModelPropellerBEM + bem/ (GPLv3), BEM sources unchanged since
the RPG init commit 2d78b81; numerics of the reference are 15-point Gauss–Kronrod disk
quadrature, a vectorized Brent solve (tol 1e-3) for v_i, and a float32 atan2
approximation — all reference-implementation details that live in the golden generator and
consumer test, not in the spec.
"""

import sympy as sp

from skyflow_dynamics.spec.inflow import oblique_momentum_thrust

#: Identified constants of the executed agilib configuration (not in the RSS 2021 paper):
#: camber offset in the lift polar, the drone-level H-force correction ("For the drone, the
#: BEM underestimates the drag which is corrected by the factor 3.0" — flight-identified),
#: and the airframe-obstruction factor on the collective z-force ("due to frame obscurring
#: parts of the area below").
CAMBER_OFFSET_AGILIB = 0.07
H_FORCE_CORRECTION_AGILIB = 3.0
Z_OBSTRUCTION_FACTOR_AGILIB = 0.9575

#: Executed vortex-ring-state gate on v_ver/v_i (paper eq. (18) uses the open band (0, 2)).
VRS_GATE_LO = 0.01
VRS_GATE_HI = 2.0


def blade_section_velocities(r: sp.Expr | float, psi: sp.Expr | float,
                             Omega: sp.Expr | float, v_hor: sp.Expr | float,
                             v_ver: sp.Expr | float, v_i: sp.Expr | float,
                             a0: sp.Expr | float = 0, a1: sp.Expr | float = 0,
                             b1: sp.Expr | float = 0) -> tuple[sp.Expr, sp.Expr]:
    """
    Tangential and perpendicular airspeed at a blade element (radius r, azimuth ψ):

        U_T = Ω·r + v_hor·sin ψ
        U_P = (v_ver − v_i) − v_ver·cos ψ·β − Ω·r·(a1·sin ψ + b1·cos ψ),
        β   = a0 − a1·cos ψ − b1·sin ψ

    Source: NeuroBEM (arXiv:2106.08015) eqs. (6)–(7); agilib bem/functions.cpp:14–24.
    ⚠ The executed code carries −v_ver·β·cos ψ where paper eq. (7) prints +v_ver·β·cos ψ;
    the term is inert in the executed configuration (flapping is zeroed during the disk
    integration), and the code form is adopted here.
    """
    beta = a0 - a1 * sp.cos(psi) - b1 * sp.sin(psi)
    U_T = Omega * r + v_hor * sp.sin(psi)
    U_P = (v_ver - v_i) - v_ver * sp.cos(psi) * beta \
        - Omega * r * (a1 * sp.sin(psi) + b1 * sp.cos(psi))
    return U_T, U_P


def inflow_angle(U_P: sp.Expr | float, U_T: sp.Expr | float) -> sp.Expr:
    """φ = atan2(U_P, U_T) — inflow angle at the element (paper eq. (8)). The executed
    reference evaluates this with a float32 fast-atan2 approximation (max error ≈ 5e-3 rad);
    the spec form is the exact atan2."""
    return sp.atan2(U_P, U_T)


def section_aoa(phi: sp.Expr | float, r: sp.Expr | float, R: sp.Expr | float,
                theta0: sp.Expr | float, theta1: sp.Expr | float) -> sp.Expr:
    """α = θ0 + θ1·(r/R) + φ — geometric pitch with linear twist plus inflow angle
    (paper eq. (9); θ1 is the total root-to-tip twist in rad, applied over r/R ∈ [0,1])."""
    return sp.sympify(theta0 + theta1 * r / R + phi)


def lift_coefficient(alpha: sp.Expr | float, cl0: sp.Expr | float,
                     eps_camber: sp.Expr | float = 0) -> sp.Expr:
    """cl(α) = cl0·(sin α·cos α + ε_c) — sinusoidal, stall-capable lift polar, valid at any
    angle of attack. Paper eq. (12) (Gill & D'Andrea 2017; Ducard & Hua 2014) has ε_c = 0;
    the executed agilib adds the camber offset ε_c = 0.07 (CAMBER_OFFSET_AGILIB)."""
    return sp.sympify(cl0 * (sp.sin(alpha) * sp.cos(alpha) + eps_camber))


def drag_coefficient(alpha: sp.Expr | float, cd0: sp.Expr | float) -> sp.Expr:
    """cd(α) = cd0·sin²α — sinusoidal drag polar (paper eq. (12))."""
    return sp.sympify(cd0 * sp.sin(alpha) ** 2)


def chord(r: sp.Expr | float, R: sp.Expr | float, c_root: sp.Expr | float,
          c_tip: sp.Expr | float) -> sp.Expr:
    """Linear chord taper c(r) = c_root + (c_tip − c_root)·r/R (agilib functions.cpp:38–40)."""
    return sp.sympify(c_root + (c_tip - c_root) * r / R)


def blade_element_integrands(r: sp.Expr | float, psi: sp.Expr | float,
                             Omega: sp.Expr | float, v_hor: sp.Expr | float,
                             v_ver: sp.Expr | float, v_i: sp.Expr | float,
                             R: sp.Expr | float, theta0: sp.Expr | float,
                             theta1: sp.Expr | float, c_root: sp.Expr | float,
                             c_tip: sp.Expr | float, cl0: sp.Expr | float,
                             cd0: sp.Expr | float, b: sp.Expr | float,
                             rho: sp.Expr | float,
                             eps_camber: sp.Expr | float = 0,
                             a0: sp.Expr | float = 0, a1: sp.Expr | float = 0,
                             b1: sp.Expr | float = 0
                             ) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Disk-load integrands (dT, dQ, dH) over r ∈ [0, R], ψ ∈ [0, 2π], such that

        T = ∬ dT dψ dr,   Q = ∬ dQ dψ dr,   H = ∬ dH dψ dr,

    with the b·ρ/(4π) prefactor already included (paper eqs. (10)–(15): the element loads
    dL = c·cl·U², dD = c·cd·U² carry no ½ρ — it is part of the prefactor, b/(4π) = number
    of blades × azimuth average × ½):

        dT = (bρ/4π)·(dL·cos φ + dD·sin φ)
        dQ = (bρ/4π)·r·(−dL·sin φ + dD·cos φ)
        dH = (bρ/4π)·(−dL·sin φ + dD·cos φ)·sin ψ

    H is the in-plane force along the hub-velocity direction x̂_P (rearward, i.e. drag).
    Source: agilib bem/functions.cpp (IntegrandPsi::evaluate, IntegrandR::evaluate).
    """
    U_T, U_P = blade_section_velocities(r, psi, Omega, v_hor, v_ver, v_i, a0, a1, b1)
    phi = inflow_angle(U_P, U_T)
    alpha = section_aoa(phi, r, R, theta0, theta1)
    U_sq = U_T ** 2 + U_P ** 2
    c = chord(r, R, c_root, c_tip)
    dL = c * lift_coefficient(alpha, cl0, eps_camber) * U_sq
    dD = c * drag_coefficient(alpha, cd0) * U_sq
    pre = b * rho / (4 * sp.pi)
    dT = pre * (dL * sp.cos(phi) + dD * sp.sin(phi))
    dQ = pre * r * (-dL * sp.sin(phi) + dD * sp.cos(phi))
    dH = pre * (-dL * sp.sin(phi) + dD * sp.cos(phi)) * sp.sin(psi)
    return dT, dQ, dH


def momentum_closure_residual(v_i: sp.Expr | float, T_bem: sp.Expr | float,
                              v_hor: sp.Expr | float, v_ver: sp.Expr | float,
                              rho: sp.Expr | float, A: sp.Expr | float) -> sp.Expr:
    """
    The BEM–momentum coupling: the induced velocity is the root v_i of

        g(v_i) = T_BEM(v_i) − 2ρA·v_i·√(v_hor² + (v_ver − v_i)²) = 0

    (paper eq. (5) against eq. (13), algorithm step 2). The momentum side is exactly
    spec.inflow.oblique_momentum_thrust with V = (v_hor, 0, v_ver). The reference solves
    this with a warm-started Brent iteration (tol 1e-3 on v_i); a differentiable backend
    can instead run fixed smooth iterations or supply v_i from the dynamic-inflow state
    (spec.inflow.dynamic_inflow_lag), making the disk loads explicit in-ODE forms.
    """
    return T_bem - oblique_momentum_thrust(v_i, sp.Matrix([v_hor, 0, v_ver]), rho, A)


def vrs_induced_velocity(x: sp.Expr | float, v_h: sp.Expr | float) -> sp.Expr:
    """
    Empirical vortex-ring-state induced velocity (paper eq. (19); Hoffmann et al., AIAA GNC
    2007), with x = v_ver/v_h the descent rate in units of the horizontal-flight induced
    velocity v_h (the closure root recomputed with v_ver = 0):

        ṽ_i = v_h·(1 + 1.125·x − 1.372·x² + 1.718·x³ − 0.655·x⁴)

    Applied when the momentum solution is invalid, gated on v_ver/v_i ∈ (VRS_GATE_LO,
    VRS_GATE_HI). ⚠ Blend variants differ: the paper takes v_i = max(ṽ_i, v_h); the
    executed agilib takes v_i = max(v_i^mom, ṽ_i) and then clamps v_i ≤ 2·v_h for the
    gated rotors — and its gate fires on ANY-rotor predicates (see REFERENCES.md, F-20).
    """
    return sp.sympify(v_h * (1 + sp.Rational(1125, 1000) * x
                             - sp.Rational(1372, 1000) * x ** 2
                             + sp.Rational(1718, 1000) * x ** 3
                             - sp.Rational(655, 1000) * x ** 4))


def tpp_rotor_force(T: sp.Expr | float, H: sp.Expr | float, chi: sp.Expr | float,
                    spin: sp.Expr | float, a0: sp.Expr | float, a1: sp.Expr | float,
                    b1: sp.Expr | float) -> sp.Matrix:
    """
    Per-rotor force in canonical body FLU, tilted by the tip-path-plane angles and aligned
    with the in-plane hub velocity direction χ = atan2(v_y, v_x):

        f = Rz(χ)·( −(H + T·sin a1),  s·T·sin b1,  T·cos a0 )

    with s the canonical physical spin sign about +ẑ_B (the reference's clockwise flag
    cw = −s). Zero flapping and H reduce this to (0, 0, T). Source: paper algorithm step 5;
    agilib model_propeller_bem.cpp:96–110 (converted FRD→FLU).
    """
    c, s_ = sp.cos(chi), sp.sin(chi)
    Rz = sp.Matrix([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return Rz * sp.Matrix([-(H + T * sp.sin(a1)), spin * T * sp.sin(b1), T * sp.cos(a0)])


def tpp_rotor_torque(Q: sp.Expr | float, chi: sp.Expr | float, spin: sp.Expr | float,
                     k_beta: sp.Expr | float, a1: sp.Expr | float,
                     b1: sp.Expr | float) -> sp.Matrix:
    """
    Per-rotor hub torque in canonical body FLU (add r_i × f_i separately):

        τ = Rz(χ)·( −s·k_β·b1,  −k_β·a1,  −s·Q )

    — hinge-spring moments k_β from the flapping deflection (paper eq. (17)) plus the
    aerodynamic drag torque opposing the spin (−s·Q about +ẑ_B, our canonical convention).
    Source: paper algorithm step 5; agilib model_propeller_bem.cpp:112–121 (FRD→FLU).
    """
    c, s_ = sp.cos(chi), sp.sin(chi)
    Rz = sp.Matrix([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return Rz * sp.Matrix([-spin * k_beta * b1, -k_beta * a1, -spin * Q])
