"""Authenticity: the symbolic spec must reproduce the frozen rpg_flightning vectors
(kind == "flightning_terms", golden/generate/gen_flightning.py — the EXECUTED JAX code,
commit-pinned, era jax 0.4.30 per finding F-28).

Division of labor: the spec carries the physics (point-mass surrogate step, exact-exp motor
discretization, quadratic thrust map, per-axis quadratic frame drag, rotor-inertia yaw
reaction) and the true SO(3) exponential map; this test carries the reference's NUMERICS as
explicit harness details — the biased-angle Rodrigues (theta = ||abs(phi) + 1e-5||, finding
F-25) and its exact symbolic derivative for the JVP rows, the float32 allocation matrix and
its float32 inverse (finding F-27), the explicit-Euler substep composition, the low-level
P body-rate controller, and the post-step rotor-speed clip. The deviation the true exp map
induces against the executed biased form is measured and bounded here
(test_biased_rodrigues_deviation) rather than hidden in a loose tolerance."""

import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from skyflow_dynamics.spec import simplified
from skyflow_dynamics.spec.motor import exact_exp_step
from skyflow_dynamics.spec.quaternion import from_rotation_vector, rotation_matrix
from skyflow_dynamics.spec.rotor_aero import per_axis_quadratic_drag, thrust_magnitude
from skyflow_dynamics.spec.wrench import rotor_inertia_moment

VECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "golden" / "vectors"
SIMPLE_DOC = json.loads((VECTOR_DIR / "flightning_simple_jvp.json").read_text())
FULL_DOC = json.loads((VECTOR_DIR / "flightning_full_step.json").read_text())
EPS_ROD = SIMPLE_DOC["quad"]["eps_rodrigues"]
GRAV = 9.81


def _skew_sym(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rod_biased_sym(phi):
    """flightning rotation_matrix_from_vector (math.py:20-33), symbolically verbatim:
    Rodrigues with the biased angle theta = ||abs(phi) + eps|| (finding F-25)."""
    K = _skew_sym(phi)
    theta = sp.sqrt(sum((sp.Abs(c) + EPS_ROD) ** 2 for c in phi))
    return sp.eye(3) + sp.sin(theta) / theta * K + (1 - sp.cos(theta)) / theta**2 * (K * K)


# harness numerics, lambdified once: the biased Rodrigues and its exact s-derivative along
# phi(s) = dt*(omega + s*omega_dot) at s = 0 (the R-block of the reference JVP)
_PH = sp.Matrix(sp.symbols("ph1 ph2 ph3", real=True))
_ROD = sp.lambdify(tuple(_PH), _rod_biased_sym(_PH), "numpy")
_W = sp.Matrix(sp.symbols("w1 w2 w3", real=True, nonzero=True))
_WD = sp.Matrix(sp.symbols("wd1 wd2 wd3", real=True))
_S, _DT = sp.symbols("s dt", real=True, positive=True)
_DROD = sp.lambdify((*_W, *_WD, _DT),
                    _rod_biased_sym(_DT * (_W + _S * _WD)).diff(_S).subs(_S, 0), "numpy")

# spec point-mass step, lambdified once (quaternion form)
_X = sp.Matrix(sp.symbols("x1 x2 x3", real=True))
_V = sp.Matrix(sp.symbols("v1 v2 v3", real=True))
_Q = sp.Matrix(sp.symbols("qw qx qy qz", real=True))
_C = sp.Symbol("c", real=True)
_STEP = simplified.step(_X, _V, _Q, _C, _W, _DT, GRAV)
_ARGS = (*_X, *_V, *_Q, _C, *_W, _DT)
_LAM_X = sp.lambdify(_ARGS, _STEP[0], "numpy")
_LAM_V = sp.lambdify(_ARGS, _STEP[1], "numpy")
_LAM_Q = sp.lambdify(_ARGS, _STEP[2], "numpy")
_LAM_R = sp.lambdify(tuple(_Q), rotation_matrix(_Q), "numpy")


def _spec_step(c):
    args = (*c["p"], *c["v"], *c["q_wxyz"], c["a"], *c["omega"], c["dt"])
    return (_LAM_X(*args).ravel(), _LAM_V(*args).ravel(), _LAM_Q(*args).ravel())


def _simple_tangent(p_dot, R, R_dot, v_dot, a, a_dot, omega, omega_dot, dt):
    """The reference JVP rows, assembled from the recorded tangents. The dp/dv rows contain
    no Rodrigues term — they are literally the spec's surrogate tangent algebra
    (dv = v_dot + dt*(R_dot ez a + R ez a_dot)); only dR needs the biased-Rodrigues
    derivative as a harness detail."""
    ez = np.array([0.0, 0.0, 1.0])
    dp = p_dot + dt * v_dot
    dv = v_dot + dt * (R_dot @ ez * a + R @ ez * a_dot)
    dR = R_dot @ _ROD(*(dt * omega)) + R @ _DROD(*omega, *omega_dot, dt)
    return dp, dR, dv


# ---------------- point-mass surrogate (spec.simplified) ----------------

@pytest.mark.parametrize("k", range(len(SIMPLE_DOC["cases"])))
def test_simple_step_primal(k):
    c = SIMPLE_DOC["cases"][k]
    R, e = np.array(c["R"]), c["expected"]
    x_n, v_n, q_n = _spec_step(c) if np.any(c["omega"]) else (
        np.array(c["p"]) + c["dt"] * np.array(c["v"]),
        None,  # v computed below for the hover case too
        np.array(c["q_wxyz"]))  # from_rotation_vector's removable 0/0: limit q+ = q
    if v_n is None:
        v_n = (np.array(c["v"])
               + c["dt"] * (np.array([0, 0, -GRAV]) + R @ np.array([0, 0, c["a"]])))
    np.testing.assert_allclose(x_n, e["p"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_n, e["v"], rtol=1e-12, atol=1e-12)

    # attitude, executed form: biased Rodrigues as harness detail — exact match
    np.testing.assert_allclose(R @ _ROD(*(c["dt"] * np.array(c["omega"]))), e["R"],
                               rtol=1e-12, atol=1e-12)
    # attitude, spec form: true exp map — bounded by test_biased_rodrigues_deviation
    np.testing.assert_allclose(_LAM_R(*q_n), e["R"], rtol=0, atol=2e-7)


@pytest.mark.parametrize("k", [k for k, c in enumerate(SIMPLE_DOC["cases"]) if "jvp" in c])
def test_simple_step_jvp(k):
    """The executed jax.jvp of quadrotor_dyn — the surrogate-gradient tangent map."""
    c = SIMPLE_DOC["cases"][k]
    t = c["jvp"]["tangents"]
    e = c["jvp"]["expected"]
    dp, dR, dv = _simple_tangent(
        np.array(t["p_dot"]), np.array(c["R"]), np.array(t["R_dot"]),
        np.array(t["v_dot"]), c["a"], t["a_dot"],
        np.array(c["omega"]), np.array(t["omega_dot"]), c["dt"])
    np.testing.assert_allclose(dp, e["dp"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dv, e["dv"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dR, e["dR"], rtol=1e-11, atol=1e-11)


def test_step_surrogate_wiring():
    """Quadrotor.step()'s custom_jvp, executed under jax.jvp: the tangent must be the
    point-mass tangent map at c = f_d/m with dt-tangent zero — pinning the surrogate
    input mapping (spec/simplified.py) from executed code."""
    mass = SIMPLE_DOC["quad"]["mass"]
    for c in SIMPLE_DOC["step_jvp_cases"]:
        st, t, e = c["state"], c["tangents"], c["expected_tangent"]
        dp, dR, dv = _simple_tangent(
            np.array(t["p_dot"]), np.array(st["R"]), np.array(t["R_dot"]),
            np.array(t["v_dot"]), c["f_d"] / mass, t["f_d_dot"] / mass,
            np.array(c["omega_d"]), np.array(t["omega_d_dot"]), c["dt"])
        np.testing.assert_allclose(dp, e["dp"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dv, e["dv"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(dR, e["dR"], rtol=1e-11, atol=1e-11)


def test_biased_rodrigues_deviation():
    """F-25 quantified: the executed attitude step is Rodrigues with theta = ||abs(phi)+eps||,
    NOT the exp map — its output is not exactly SO(3), and the spec's true exp map deviates
    from it measurably. Bound both effects instead of hiding them in tolerances."""
    worst_dev, worst_ortho = 0.0, 0.0
    for c in SIMPLE_DOC["cases"]:
        phi = np.array(c["dt"]) * np.array(c["omega"])
        rb = _ROD(*phi)
        worst_ortho = max(worst_ortho, np.abs(rb.T @ rb - np.eye(3)).max())
        if not np.any(phi):
            # eps guard at zero: K = 0, so the executed form returns I exactly
            np.testing.assert_array_equal(rb, np.eye(3))
            continue
        q = np.array(from_rotation_vector(sp.Matrix(phi)), float).ravel()
        worst_dev = max(worst_dev, np.abs(_LAM_R(*q) - rb).max())
    assert 1e-10 < worst_dev < 2e-7, f"biased-Rodrigues deviation {worst_dev:.3e}"
    assert 1e-12 < worst_ortho < 5e-8, f"orthogonality defect {worst_ortho:.3e}"


# ---------------- full model (Quadrotor._dynamics / step) ----------------

QD = FULL_DOC["quad"]
ALLOC = np.array(QD["allocation_matrix"])
ALLOC_INV = np.array(QD["allocation_matrix_inv"])
J_DIAG = np.array(QD["inertia"])
ROTOR_P = SimpleNamespace(I_rot=QD["motor_inertia"], spin=QD["spin_canonical"], n=4)


def _dynamics_spec(st, mot_d, dt, tm_drawn, cd_drawn):
    """One low-level substep composed from spec terms; harness details: explicit-Euler
    updates, biased Rodrigues, executed float32 allocation rows (F-27), post-step clip."""
    p, R, v = np.array(st["p"]), np.array(st["R"]), np.array(st["v"])
    omega, mot = np.array(st["omega"]), np.array(st["motor_omega"])
    mot_d = np.asarray(mot_d, float)

    T = np.array([float(thrust_magnitude(mot[i], 0.0, 0.0, tm_drawn)) for i in range(4)])
    kq = 0.5 * QD["rho"] * np.asarray(cd_drawn) * np.array(
        [QD["frontarea_x"], QD["frontarea_y"], QD["frontarea_z"]])
    f_drag = np.array(per_axis_quadratic_drag(sp.Matrix(R.T @ v), sp.Matrix(kq)),
                      float).ravel()
    acc = (np.array([0, 0, -GRAV])
           + R @ (np.array([0.0, 0.0, T.sum()]) + f_drag) / QD["mass"])

    # rotor-inertia yaw reaction (spec term at w = 0: flightning omits the -w x h
    # precession half) with the continuous motor rate the reference uses
    w_dot = (mot_d - mot) / QD["motor_tau"]
    tau_r = np.array(rotor_inertia_moment(sp.zeros(3, 1), sp.Matrix(mot),
                                          sp.Matrix(w_dot), ROTOR_P), float).ravel()
    tau = ALLOC[1:] @ T
    domega = (tau - np.cross(omega, J_DIAG * omega) + tau_r) / J_DIAG

    mot_new = np.array(exact_exp_step(sp.Matrix(mot), sp.Matrix(mot_d), dt,
                                      QD["motor_tau"]), float).ravel()
    mot_new = np.clip(mot_new, QD["motor_omega_min"], QD["motor_omega_max"])
    return {"p": p + dt * v, "R": R @ _ROD(*(dt * omega)), "v": v + dt * acc,
            "omega": omega + dt * domega, "motor_omega": mot_new,
            "domega": domega, "acc": acc}


def _controller(omega, f_T, omega_cmd):
    """The reference low-level P body-rate controller — control, harness-side by this
    repo's rules; replicated with the recorded gains and executed float32 inverse."""
    torque_cmd = (J_DIAG * (np.array(QD["controller_K"]) * (omega_cmd - omega))
                  + np.cross(omega, J_DIAG * omega))
    f_cmd = ALLOC_INV @ np.concatenate([[f_T], torque_cmd])
    f_cmd = np.clip(f_cmd, QD["thrust_min_effective"], QD["thrust_max"])
    return np.clip(np.sqrt(f_cmd / QD["thrust_map"][0]),
                   QD["motor_omega_min"], QD["motor_omega_max"])


@pytest.mark.parametrize("k", range(len(FULL_DOC["dynamics_cases"])))
def test_full_dynamics_substep(k):
    c = FULL_DOC["dynamics_cases"][k]
    got = _dynamics_spec(c["state"], c["motor_omega_d"], c["dt"],
                         c["dr"]["thrust_map_drawn"], c["dr"]["drag_coeff_drawn"])
    for key, val in c["expected"].items():
        np.testing.assert_allclose(got[key], val, rtol=1e-11, atol=1e-11,
                                   err_msg=f"case {k} [{key}]")


@pytest.mark.parametrize("k", range(len(FULL_DOC["step_cases"])))
def test_full_step_controller_loop(k):
    """Quadrotor.step primal: N 1 kHz substeps with the controller in the loop."""
    c = FULL_DOC["step_cases"][k]
    n = round(c["dt"] / QD["dt_low_level"])
    cur = {key: np.array(val) for key, val in c["state"].items()}
    for _ in range(n):
        mot_d = _controller(cur["omega"], c["f_d"], np.array(c["omega_d"]))
        cur = _dynamics_spec(cur, mot_d, QD["dt_low_level"],
                             c["dr"]["thrust_map_drawn"], c["dr"]["drag_coeff_drawn"])
    for key, val in c["expected_primal"].items():
        np.testing.assert_allclose(cur[key], val, rtol=1e-9, atol=1e-9,
                                   err_msg=f"case {k} [{key}]")


def test_allocation_is_structural():
    """The executed allocation matrix IS the canonical structural composition — torque rows
    are the r x F lever arms (tau_x = sum y_i f_i, tau_y = -sum x_i f_i) and the yaw row is
    -s_i*kappa*f_i — quantized to float32 (finding F-27)."""
    tbm = np.array(QD["tbm"])
    spin = np.array(QD["spin_canonical"], float)
    canonical = np.stack([np.ones(4), tbm[:, 1], -tbm[:, 0], -spin * QD["kappa"]])
    np.testing.assert_allclose(ALLOC, canonical, rtol=0, atol=1.5e-9)
    # float32 inverse of a float32 matrix: pin the executed inverse's quality explicitly
    np.testing.assert_allclose(ALLOC_INV @ ALLOC, np.eye(4), rtol=0, atol=1e-6)
    resid = np.abs(ALLOC_INV @ ALLOC - np.eye(4)).max()
    assert resid > 1e-12, "inverse unexpectedly exact — float32 provenance claim is stale"
