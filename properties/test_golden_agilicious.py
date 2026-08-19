"""Authenticity: the symbolic spec must reproduce the frozen agilicious agilib vectors
(kind == "agilicious_terms", golden/generate/gen_agilicious.py — the EXECUTED compiled
agilib, GPLv3 mirror pinned by commit + file sha256s).

Division of labor: the spec carries the physics (blade-element integrands, momentum
closure, VRS polynomial, tip-path-plane wrench, drag forms, integrator schemes); this test
carries the reference's NUMERICS as explicit harness details — the single 15-point
Gauss-Kronrod rule per axis, and the float32 fast-atan2 the reference uses for the inflow
angle. The spec's inflow angle is the exact atan2; the deviation that induces is measured
and bounded here (test_exact_atan2_spec_deviation) rather than hidden."""

import json
import math
import pathlib

import numpy as np
import pytest
import sympy as sp

from skyflow_dynamics.spec import bem
from skyflow_dynamics.spec.discretization import rk4_step, semi_implicit_euler_step
from skyflow_dynamics.spec.motor import first_order_lag
from skyflow_dynamics.spec.quaternion import kinematics, rotation_matrix
from skyflow_dynamics.spec.rotor_aero import (
    cubic_drag,
    linear_drag,
    per_axis_quadratic_drag,
    thrust_magnitude,
    torque_magnitude,
    translational_lift,
)

VECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "golden" / "vectors"
BEM_DOC = json.loads((VECTOR_DIR / "agilicious_bem.json").read_text())
SIMPLE_DOC = json.loads((VECTOR_DIR / "agilicious_simple_models.json").read_text())

# ---------------- reference numerics (agilib bem/gauss_kronrod.hpp, fast_atan2.hpp) ----------------

GK_X = np.array([
    -9.914553711208126392068546975263285e-01, -9.491079123427585245261896840478513e-01,
    -8.648644233597690727897127886409262e-01, -7.415311855993944398638647732807884e-01,
    -5.860872354676911302941448382587296e-01, -4.058451513773971669066064120769615e-01,
    -2.077849550078984676006894037732449e-01, 0.0,
    2.077849550078984676006894037732449e-01, 4.058451513773971669066064120769615e-01,
    5.860872354676911302941448382587296e-01, 7.415311855993944398638647732807884e-01,
    8.648644233597690727897127886409262e-01, 9.491079123427585245261896840478513e-01,
    9.914553711208126392068546975263285e-01])
GK_W = np.array([
    2.293532201052922496373200805896959e-02, 6.309209262997855329070066318920429e-02,
    1.047900103222501838398763225415180e-01, 1.406532597155259187451895905102379e-01,
    1.690047266392679028265834265985503e-01, 1.903505780647854099132564024210137e-01,
    2.044329400752988924141619992346491e-01, 2.094821410847278280129991748917143e-01,
    2.044329400752988924141619992346491e-01, 1.903505780647854099132564024210137e-01,
    1.690047266392679028265834265985503e-01, 1.406532597155259187451895905102379e-01,
    1.047900103222501838398763225415180e-01, 6.309209262997855329070066318920429e-02,
    2.293532201052922496373200805896959e-02])

F32 = np.float32
_N1, _N2 = F32(0.97239411), F32(-0.19194795)
_PI_F, _PI2_F = F32(math.pi), F32(math.pi / 2)


def approx_atan2(y, x):
    """The reference's float32 fast-atan2 (agilib fast_atan2.hpp), op-order exact."""
    xf, yf = F32(x), F32(y)
    if xf != F32(0.0):
        if abs(xf) >= abs(yf):
            offset = F32(math.copysign(float(_PI_F), float(yf))) if xf < 0 else F32(0.0)
            z = F32(yf / xf)
            return float(F32(offset + F32(F32(_N1 + F32(F32(_N2 * z) * z)) * z)))
        offset = F32(math.copysign(float(_PI2_F), float(yf)))
        z = F32(xf / yf)
        return float(F32(offset - F32(F32(_N1 + F32(F32(_N2 * z) * z)) * z)))
    return float(_PI2_F) if yf > 0 else (float(-_PI2_F) if yf < 0 else 0.0)


# ---------------- spec expressions, lambdified once ----------------

_r, _psi, _phi = sp.symbols("r psi phi", real=True)
_Om, _vhor, _vver, _vi = sp.symbols("Omega v_hor v_ver v_i", real=True)
_R, _th0, _th1, _ci, _co = sp.symbols("R theta0 theta1 c_i c_o", real=True)
_cl0, _cd0, _eps, _b, _rho = sp.symbols("cl0 cd0 eps b rho", real=True)

_UT, _UP = bem.blade_section_velocities(_r, _psi, _Om, _vhor, _vver, _vi, 0, 0, 0)


def _phi_param_integrands():
    """The spec integrands with the inflow angle φ as a free input, so the reference's
    approximate atan2 can be injected as a harness detail."""
    alpha = bem.section_aoa(_phi, _r, _R, _th0, _th1)
    U_sq = _UT**2 + _UP**2
    c = bem.chord(_r, _R, _ci, _co)
    dL = c * bem.lift_coefficient(alpha, _cl0, _eps) * U_sq
    dD = c * bem.drag_coefficient(alpha, _cd0) * U_sq
    pre = _b * _rho / (4 * sp.pi)
    dT = pre * (dL * sp.cos(_phi) + dD * sp.sin(_phi))
    dQ = pre * _r * (-dL * sp.sin(_phi) + dD * sp.cos(_phi))
    dH = pre * (-dL * sp.sin(_phi) + dD * sp.cos(_phi)) * sp.sin(_psi)
    return dT, dQ, dH


_DT_E, _DQ_E, _DH_E = _phi_param_integrands()
_ARGS = (_r, _psi, _phi, _Om, _vhor, _vver, _vi, _R, _th0, _th1, _ci, _co,
         _cl0, _cd0, _eps, _b, _rho)
_LAM_UT = sp.lambdify((_r, _psi, _Om, _vhor, _vver, _vi), _UT, "numpy")
_LAM_UP = sp.lambdify((_r, _psi, _Om, _vhor, _vver, _vi), _UP, "numpy")
_LAM_DT = sp.lambdify(_ARGS, _DT_E, "numpy")
_LAM_DQ = sp.lambdify(_ARGS, _DQ_E, "numpy")
_LAM_DH = sp.lambdify(_ARGS, _DH_E, "numpy")


def test_phi_assembly_is_the_spec_term():
    """Substituting φ = atan2(U_P, U_T) into the φ-parameterized assembly recovers
    spec.bem.blade_element_integrands exactly — the harness split changes nothing."""
    spec = bem.blade_element_integrands(_r, _psi, _Om, _vhor, _vver, _vi, _R, _th0, _th1,
                                        _ci, _co, _cl0, _cd0, _b, _rho, eps_camber=_eps)
    phi_true = sp.atan2(_UP, _UT)
    for assembled, reference in zip((_DT_E, _DQ_E, _DH_E), spec):
        assert sp.simplify(assembled.subs(_phi, phi_true) - reference) == 0


def _disk_loads(bp, Omega, vhor, vver, vind, atan2_fn):
    """T, Q, H per rotor via the reference quadrature over the spec integrands."""
    r_nodes = GK_X * (bp["r_prop"] / 2) + bp["r_prop"] / 2
    p_nodes = GK_X * math.pi + math.pi
    rg, pg = np.meshgrid(r_nodes, p_nodes, indexing="ij")
    w2 = np.outer(GK_W, GK_W) * (bp["r_prop"] / 2) * math.pi
    out = np.zeros((3, 4))
    for i in range(4):
        ut = _LAM_UT(rg, pg, Omega[i], vhor[i], vver[i], vind[i])
        up = np.broadcast_to(_LAM_UP(rg, pg, Omega[i], vhor[i], vver[i], vind[i]), rg.shape)
        phi = np.array([[atan2_fn(up[j, k], ut[j, k]) for k in range(15)]
                        for j in range(15)])
        args = (rg, pg, phi, Omega[i], vhor[i], vver[i], vind[i], bp["r_prop"],
                bp["theta0"], bp["theta1"], bp["chord_inner"], bp["chord_outer"],
                bp["cl0"], bp["cd0"], 0.07, bp["num_blades"], bp["rho"])
        out[0, i] = float((_LAM_DT(*args) * w2).sum())
        out[1, i] = float((_LAM_DQ(*args) * w2).sum())
        out[2, i] = float((_LAM_DH(*args) * w2).sum())
    return out


def _bp():
    return BEM_DOC["bem_params"]


def test_blade_element_disk_integrals():
    """Spec integrands + reference numerics reproduce the executed thrust, drag torque, and
    (×3.0-corrected) H-force integrals at the final induced velocity."""
    bp = _bp()
    for c in BEM_DOC["cases"]:
        loads = _disk_loads(bp, np.array(c["omega_mot"]), np.array(c["v_hor"]),
                            np.array(c["v_ver"]), np.array(c["vind"]), approx_atan2)
        np.testing.assert_allclose(loads[0], c["thrust"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(loads[1], c["torque"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            loads[2] * bem.H_FORCE_CORRECTION_AGILIB, c["hforce"], rtol=1e-9, atol=1e-9)


def test_exact_atan2_spec_deviation():
    """The pure-spec form (exact atan2) deviates from the executed reference only through
    the reference's float32 atan2 approximation: bound that deviation explicitly instead
    of letting it hide inside a loose tolerance."""
    # the approximation itself: max error of the minimax polynomial is < 5.2e-3 rad
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 50, size=(4096, 2))
    err = max(abs(approx_atan2(y, x) - math.atan2(y, x)) for y, x in pts)
    assert err < 5.2e-3

    bp = _bp()
    worst = 0.0
    for c in BEM_DOC["cases"]:
        loads = _disk_loads(bp, np.array(c["omega_mot"]), np.array(c["v_hor"]),
                            np.array(c["v_ver"]), np.array(c["vind"]), math.atan2)
        scale = np.maximum(1e-3, np.abs(c["thrust"]))
        worst = max(worst, float(np.max(np.abs(loads[0] - c["thrust"]) / scale)))
    # measured 2.3e-2 across the vector set: the reference's float32 atan2 moves the disk
    # loads by up to ~2.3% relative — the price agilib pays for the fast approximation
    assert worst < 3e-2, f"exact-atan2 thrust deviation {worst:.3e} exceeds documented bound"


def test_momentum_closure_residual():
    """At the recorded pre-VRS root the closure residual g(v_i) = T_BEM − T_mom vanishes to
    within the reference Brent tolerance (1e-3 on v_i) times the local slope."""
    bp = _bp()
    lam_res = sp.lambdify(
        (_vi, sp.Symbol("T_bem"), _vhor, _vver, _rho, sp.Symbol("A")),
        bem.momentum_closure_residual(_vi, sp.Symbol("T_bem"), _vhor, _vver, _rho,
                                      sp.Symbol("A")), "numpy")

    def residual(c, v):
        t_bem = _disk_loads(bp, np.array(c["omega_mot"]), np.array(c["v_hor"]),
                            np.array(c["v_ver"]), v, approx_atan2)[0]
        return np.array([lam_res(v[i], t_bem[i], c["v_hor"][i], c["v_ver"][i],
                                 bp["rho"], bp["prop_area"]) for i in range(4)])

    n_roots = n_failed = 0
    for c in BEM_DOC["cases"]:
        v = np.array(c["vind_momentum"])
        failed = v == 30.0  # the reference Brent's range max
        if failed.any():
            # Deep descent: the closure residual has NO root in the solver range — momentum
            # theory is inapplicable (the very regime the VRS fit exists for). The executed
            # reference's Brent silently returns range-max 30 m/s on bracket failure and the
            # VRS clamp rescues the value (finding F-21). Pin the no-root claim itself.
            assert failed.all()
            n_failed += 1
            for probe in (-20.0, 0.0, 10.0, 20.0, 30.0):
                assert (residual(c, np.full(4, probe)) > 0).all(), \
                    f"expected one-signed residual across the Brent range at v={probe}"
            continue
        n_roots += 1
        g = residual(c, v)
        h = 1e-4
        slope = (residual(c, v + h) - residual(c, v - h)) / (2 * h)
        bound = 4e-3 * np.abs(slope) + 1e-9
        assert (np.abs(g) <= bound).all(), f"residual {g} exceeds Brent-tol bound {bound}"
    assert n_roots >= 10 and n_failed >= 1, "vector set must exercise both solver outcomes"


def test_vrs_blend_replay():
    """The executed VRS path is exactly spec.bem.vrs_induced_velocity plus the reference's
    blend: ṽ from the horizontal-equivalent root, max against the momentum root, and the
    2·v_h clamp for in-window rotors (ANY-rotor gate semantics, finding F-20)."""
    lam_vrs = sp.lambdify((sp.Symbol("x"), sp.Symbol("v_h")),
                          bem.vrs_induced_velocity(sp.Symbol("x"), sp.Symbol("v_h")),
                          "numpy")
    n_gated = n_free = 0
    for c in BEM_DOC["cases"]:
        v_mom = np.array(c["vind_momentum"])
        if not c["vrs_any_gate"]:
            n_free += 1
            np.testing.assert_allclose(c["vind"], v_mom, rtol=1e-13)
            continue
        n_gated += 1
        v_h = np.array(c["vind_h"])
        v_tilde = lam_vrs(np.array(c["v_ver"]) / v_h, v_h)
        np.testing.assert_allclose(v_tilde, c["vind_vrs_candidate"], rtol=1e-12)
        blended = np.maximum(v_mom, v_tilde)
        for i in range(4):
            if c["vrs_in_window"][i]:
                blended[i] = min(blended[i], 2 * v_h[i])
        np.testing.assert_allclose(blended, c["vind"], rtol=1e-12)
    assert n_gated >= 3 and n_free >= 3, "vector set must exercise both VRS branches"


def test_tpp_wrench_composition():
    """Recorded per-rotor loads and flapping angles, pushed through the spec tip-path-plane
    wrench (canonical FLU, physical spin signs) plus lever arms, the z-obstruction factor,
    and the rigid-body projection, reproduce the executed derivative contributions."""
    quad = BEM_DOC["quad"]
    bp = _bp()
    spin = BEM_DOC["spin"]
    t_BM = np.array(quad["t_BM"]).T
    syms = sp.symbols("T H Q chi s a0 a1 b1 k_beta", real=True)
    T_, H_, Q_, chi_, s_, a0_, a1_, b1_, kb_ = syms
    lam_f = sp.lambdify((T_, H_, chi_, s_, a0_, a1_, b1_),
                        bem.tpp_rotor_force(T_, H_, chi_, s_, a0_, a1_, b1_), "numpy")
    lam_tau = sp.lambdify((Q_, chi_, s_, kb_, a1_, b1_),
                          bem.tpp_rotor_torque(Q_, chi_, s_, kb_, a1_, b1_), "numpy")
    q_syms = sp.Matrix(sp.symbols("qw qx qy qz", real=True))
    lam_R = sp.lambdify(tuple(q_syms), rotation_matrix(q_syms), "numpy")

    for c in BEM_DOC["cases"]:
        force = np.zeros(3)
        torque = np.zeros(3)
        for i in range(4):
            v_frd = np.array(c["hub_velocity_frd"][i])
            v_flu = np.array([v_frd[0], -v_frd[1], -v_frd[2]])
            chi = math.atan2(v_flu[1], v_flu[0])
            f = lam_f(c["thrust"][i], c["hforce"][i], chi, spin[i],
                      c["a0"][i], c["a1s"][i], c["b1s"][i]).flatten()
            tau = lam_tau(c["torque"][i], chi, spin[i], bp["k_spring"],
                          c["a1s"][i], c["b1s"][i]).flatten()
            force += f
            torque += tau + np.cross(t_BM[:, i], f)
        force[2] *= bem.Z_OBSTRUCTION_FACTOR_AGILIB
        R = lam_R(*c["q_wxyz"])
        dvel = R @ force / quad["mass"] + np.array([0.0, 0.0, -quad["G"]])
        dome = torque / np.array(quad["J_diag"])
        np.testing.assert_allclose(dvel, c["dvel"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(dome, c["dome"], rtol=1e-9, atol=1e-9)


# ---------------- simple models + integrators ----------------

def _lam_simple():
    W = sp.Matrix(sp.symbols("W1 W2 W3 W4", real=True))
    Wc = sp.Matrix(sp.symbols("Wc1 Wc2 Wc3 Wc4", real=True))
    tau = sp.Symbol("tau_m", positive=True)
    v = sp.Matrix(sp.symbols("vx vy vz", real=True))
    k3 = sp.Matrix(sp.symbols("kx ky kz", real=True))
    q = sp.Matrix(sp.symbols("qw qx qy qz", real=True))
    w = sp.Matrix(sp.symbols("wx wy wz", real=True))
    kh = sp.Symbol("k_h", real=True)
    return {
        "motor": sp.lambdify((*W, *Wc, tau), first_order_lag(W, Wc, tau), "numpy"),
        "quad_drag": sp.lambdify((*v, *k3), per_axis_quadratic_drag(v, k3), "numpy"),
        "cub_drag": sp.lambdify((*v, *k3), cubic_drag(v, k3), "numpy"),
        "lin_drag": sp.lambdify((*v, *k3), linear_drag(v, k3), "numpy"),
        "lift": sp.lambdify((*v, kh), translational_lift(v, kh), "numpy"),
        "R": sp.lambdify(tuple(q), rotation_matrix(q), "numpy"),
        "qdot": sp.lambdify((*q, *w), kinematics(q, w), "numpy"),
    }


LAM = _lam_simple()
POS, ATT, VEL, OME, MOT, MOTDES, SIZE = 0, 3, 7, 10, 31, 35, 39


def _spec_pipeline_f(quad):
    """agilib's default pipeline (motor + quadratic thrust/torque maps + rigid body) built
    from spec expressions, on the reference's 39-slot state layout."""
    t_BM = np.array(quad["t_BM"]).T
    Jd = np.array(quad["J_diag"])
    tm = quad["thrust_map"]
    kappa = quad["kappa"]
    spin = np.array(SIMPLE_DOC["spin"], float)

    def f(t, s):
        d = np.zeros(SIZE)
        q, w, mot = s[ATT:ATT + 4], s[OME:OME + 3], s[MOT:MOT + 4]
        d[POS:POS + 3] = s[VEL:VEL + 3]
        d[ATT:ATT + 4] = LAM["qdot"](*q, *w).flatten()
        T = np.array([thrust_magnitude(mot[i], tm[2], tm[1], tm[0]) for i in range(4)],
                     float)
        Q = np.array([torque_magnitude(mot[i], 0.0, 0.0, kappa * tm[0])
                      for i in range(4)], float)
        force = np.array([0.0, 0.0, T.sum()])
        torque = sum(np.cross(t_BM[:, i], [0.0, 0.0, T[i]])
                     + np.array([0.0, 0.0, -spin[i] * Q[i]]) for i in range(4))
        d[VEL:VEL + 3] = LAM["R"](*q) @ force / quad["mass"] + [0, 0, -quad["G"]]
        d[OME:OME + 3] = (torque - np.cross(w, Jd * w)) / Jd
        d[MOT:MOT + 4] = LAM["motor"](*mot, *s[MOTDES:MOTDES + 4],
                                      quad["motor_tau"]).flatten()
        return d
    return f


@pytest.mark.parametrize("k", range(len(SIMPLE_DOC["cases"])))
def test_agilicious_simple_models(k):
    c = SIMPLE_DOC["cases"][k]
    quad = SIMPLE_DOC["quad"]
    bd, lc = SIMPLE_DOC["body_drag_params"], SIMPLE_DOC["lin_cub_params"]
    q, v, w = np.array(c["q_wxyz"]), np.array(c["v_W"]), np.array(c["w_B"])
    mot = np.array(c["mot"])
    R = LAM["R"](*q)
    vb = R.T @ v

    np.testing.assert_allclose(
        LAM["motor"](*mot, *c["motdes"], quad["motor_tau"]).flatten(),
        c["motor_dmot"], rtol=1e-12, atol=1e-12)

    # per-axis quadratic body drag, physical packing k = ½ρ·c·A. The reference adds this
    # FORCE straight into the acceleration slot (no /m — finding F-19): the vector pins the
    # force expression, so no mass division here either.
    kq = 0.5 * bd["rho"] * np.array([bd["cxy"] * bd["ax"], bd["cxy"] * bd["ay"],
                                     bd["cz"] * bd["az"]])
    np.testing.assert_allclose(R @ LAM["quad_drag"](*vb, *kq).flatten(),
                               c["bodydrag_dvel"], rtol=1e-12, atol=1e-12)

    # linear + cubic drag + translational lift (the NeuroBEM PolyFit family), force/m
    F = (LAM["lin_drag"](*vb, *lc["lin_drag_coeff"]).flatten()
         + LAM["cub_drag"](*vb, *lc["cub_drag_coeff"]).flatten()
         + np.array([0, 0, LAM["lift"](*vb, lc["induced_lift_coeff"])]))
    np.testing.assert_allclose(R @ F / quad["mass"], c["lincub_dvel"],
                               rtol=1e-12, atol=1e-12)

    f = _spec_pipeline_f(quad)
    s = np.zeros(SIZE)
    s[ATT:ATT + 4], s[VEL:VEL + 3], s[OME:OME + 3] = q, v, w
    s[MOT:MOT + 4], s[MOTDES:MOTDES + 4] = mot, c["motdes"]
    d = f(0.0, s)
    np.testing.assert_allclose(d[VEL:VEL + 3], c["pipeline_dvel"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d[OME:OME + 3], c["pipeline_dome"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d[MOT:MOT + 4], c["pipeline_dmot"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", range(len(SIMPLE_DOC["cases"])))
def test_agilicious_integrator_steps(k):
    """spec.discretization.semi_implicit_euler_step (agilib velocity/position grouping),
    rk4_step, and plain Euler reproduce the executed integrator steps."""
    c = SIMPLE_DOC["cases"][k]
    quad = SIMPLE_DOC["quad"]
    f = _spec_pipeline_f(quad)
    s = np.zeros(SIZE)
    s[ATT:ATT + 4], s[VEL:VEL + 3], s[OME:OME + 3] = c["q_wxyz"], c["v_W"], c["w_B"]
    s[MOT:MOT + 4], s[MOTDES:MOTDES + 4] = c["mot"], c["motdes"]
    dt = c["dt"]

    vel_idx = np.r_[VEL:VEL + 3, OME:OME + 3, MOT:MOT + 4]
    pos_idx = np.r_[POS:POS + 3, ATT:ATT + 4]
    blocks = [("pos", slice(POS, POS + 3)), ("att", slice(ATT, ATT + 4)),
              ("vel", slice(VEL, VEL + 3)), ("ome", slice(OME, OME + 3)),
              ("mot", slice(MOT, MOT + 4))]

    sym = semi_implicit_euler_step(f, s, dt, vel_idx, pos_idx)
    eul = s + dt * f(0.0, s)
    rk4 = rk4_step(f, s, dt)
    for name, sl in blocks:
        np.testing.assert_allclose(sym[sl], c[f"sym_{name}"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(eul[sl], c[f"euler_{name}"], rtol=1e-12, atol=1e-14)
        # four RK stages amplify Eigen-vs-numpy summation-order noise: one notch looser
        np.testing.assert_allclose(rk4[sl], c[f"rk4_{name}"], rtol=1e-9, atol=1e-12)
