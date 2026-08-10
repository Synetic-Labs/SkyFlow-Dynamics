"""
Canonical symbol containers for the multirotor model.

The model is a pure function of three groups:

    State   s = (x_W, v_W, q_WB, ω_B, Ω)          — 13 + n scalars
    Inputs  u = (Ω_c, v_wind_W, F_ext_W, τ_ext_B)  — commanded speeds + exogenous signals
    Params  θ = physical parameters (see Params)

Everything is SI. Rotor count n is a construction argument (default 4). The `flat()` methods
define the canonical scalar ordering used by lambdify'd numeric evaluations and by the golden
vector files — that ordering is part of the spec.
"""

from dataclasses import dataclass

import sympy as sp


def _vec(name: str, n: int = 3, **assumptions) -> sp.Matrix:
    return sp.Matrix(sp.symbols(f"{name}_1:{n + 1}", real=True, **assumptions))


@dataclass(frozen=True)
class State:
    """x: world position (m) · v: world velocity (m/s) · q: body→world unit quaternion (wxyz)
    · w: body angular rate (rad/s) · W: rotor speeds Ω ≥ 0 (rad/s)."""
    x: sp.Matrix
    v: sp.Matrix
    q: sp.Matrix
    w: sp.Matrix
    W: sp.Matrix

    def flat(self) -> tuple:
        return (*self.x, *self.v, *self.q, *self.w, *self.W)


@dataclass(frozen=True)
class Inputs:
    """W_c: commanded rotor speeds Ω_c ≥ 0 (rad/s) · v_wind: wind velocity, world frame (m/s)
    · F_ext: external force, world frame (N) · tau_ext: external torque, body frame (N·m)."""
    W_c: sp.Matrix
    v_wind: sp.Matrix
    F_ext: sp.Matrix
    tau_ext: sp.Matrix

    def flat(self) -> tuple:
        return (*self.W_c, *self.v_wind, *self.F_ext, *self.tau_ext)


@dataclass(frozen=True)
class Params:
    """Physical parameters. Per-rotor quantities are tuples of length n (index = rotor).

    mass (kg) · grav g>0 (m/s²) · inertia: 3×3 symmetric body inertia (kg·m²) ·
    rotor_pos: rotor hub positions r_i, body frame (m) · spin: s_i ∈ {+1,−1}, sign of the rotor
    angular velocity about +ẑ_B · axis: unit thrust axes ê_i, body frame (default ẑ) ·
    ct0/ct1/ct2: per-rotor thrust polynomial  T_i = ct0 + ct1·Ω + ct2·Ω²  (N; ct2 ≡ k_η) ·
    cq0/cq1/cq2: per-rotor drag-torque magnitude polynomial  Q_i = cq0 + cq1·Ω + cq2·Ω²
    (N·m; cq2 ≡ k_m) · tau_m: motor time constant (s) · ka1,ka2,kd1,kd2: asymmetric motor
    coefficients (1/s, 1/(rad)) · I_rot: rotor spin-axis inertia (kg·m²) ·
    c_D: parasitic drag diagonal (c_Dx, c_Dy, c_Dz) (N/(m/s)²) · k_d/k_z: rotor in-plane/axial
    drag (N/(m/s) per rad/s) · k_flap: blade-flapping moment coefficient (N·m/(m/s) per rad/s) ·
    k_h: translational-lift coefficient (N/(m/s)²) · k_angle/k_hor: thrust vs angle-of-attack /
    advance-ratio slopes (1/rad) · k_v2: vertical airspeed² thrust loss (N/(m/s)²) ·
    r_prop: propeller radius (m).

    Validity rule: k_h and {k_angle, k_hor} model the same in-plane airspeed effect at different
    fidelities — a vehicle uses one or the other, never both (double-counting).
    """
    mass: sp.Symbol
    grav: sp.Symbol
    inertia: sp.Matrix
    rotor_pos: tuple
    spin: tuple
    axis: tuple
    ct0: tuple
    ct1: tuple
    ct2: tuple
    cq0: tuple
    cq1: tuple
    cq2: tuple
    tau_m: sp.Symbol
    ka1: sp.Symbol
    ka2: sp.Symbol
    kd1: sp.Symbol
    kd2: sp.Symbol
    I_rot: sp.Symbol
    c_D: sp.Matrix
    k_d: sp.Symbol
    k_z: sp.Symbol
    k_flap: sp.Symbol
    k_h: sp.Symbol
    k_angle: sp.Symbol
    k_hor: sp.Symbol
    k_v2: sp.Symbol
    r_prop: sp.Symbol

    @property
    def n(self) -> int:
        return len(self.spin)

    def flat(self) -> tuple:
        I = self.inertia
        return (
            self.mass, self.grav,
            I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2],
            *(c for r in self.rotor_pos for c in r),
            *self.spin,
            *(c for e in self.axis for c in e),
            *self.ct0, *self.ct1, *self.ct2,
            *self.cq0, *self.cq1, *self.cq2,
            self.tau_m, self.ka1, self.ka2, self.kd1, self.kd2,
            self.I_rot,
            *self.c_D,
            self.k_d, self.k_z, self.k_flap, self.k_h,
            self.k_angle, self.k_hor, self.k_v2, self.r_prop,
        )


def state_symbols(n: int = 4) -> State:
    return State(
        x=_vec("x"), v=_vec("v"),
        q=sp.Matrix(sp.symbols("q_w q_x q_y q_z", real=True)),
        w=_vec("omega"),
        W=sp.Matrix(sp.symbols(f"Omega_1:{n + 1}", nonnegative=True)),
    )


def input_symbols(n: int = 4) -> Inputs:
    return Inputs(
        W_c=sp.Matrix(sp.symbols(f"Omegac_1:{n + 1}", nonnegative=True)),
        v_wind=_vec("vw"), F_ext=_vec("Fe"), tau_ext=_vec("taue"),
    )


def param_symbols(n: int = 4) -> Params:
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = sp.symbols("Ixx Iyy Izz Ixy Ixz Iyz", real=True)
    per_rotor = lambda name, **a: tuple(sp.symbols(f"{name}_1:{n + 1}", real=True, **a))
    return Params(
        mass=sp.Symbol("m", positive=True),
        grav=sp.Symbol("g", positive=True),
        inertia=sp.Matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]]),
        rotor_pos=tuple(_vec(f"r{i + 1}") for i in range(n)),
        spin=per_rotor("s"),
        axis=tuple(_vec(f"e{i + 1}") for i in range(n)),
        ct0=per_rotor("ct0"), ct1=per_rotor("ct1"), ct2=per_rotor("ct2", nonnegative=True),
        cq0=per_rotor("cq0"), cq1=per_rotor("cq1"), cq2=per_rotor("cq2", nonnegative=True),
        tau_m=sp.Symbol("tau_m", positive=True),
        ka1=sp.Symbol("ka1", real=True), ka2=sp.Symbol("ka2", real=True),
        kd1=sp.Symbol("kd1", real=True), kd2=sp.Symbol("kd2", real=True),
        I_rot=sp.Symbol("I_rot", nonnegative=True),
        c_D=sp.Matrix(sp.symbols("c_Dx c_Dy c_Dz", nonnegative=True)),
        k_d=sp.Symbol("k_d", nonnegative=True),
        k_z=sp.Symbol("k_z", nonnegative=True),
        k_flap=sp.Symbol("k_flap", real=True),
        k_h=sp.Symbol("k_h", real=True),
        k_angle=sp.Symbol("k_angle", real=True),
        k_hor=sp.Symbol("k_hor", real=True),
        k_v2=sp.Symbol("k_v2", real=True),
        r_prop=sp.Symbol("r_prop", positive=True),
    )
