"""Backend authenticity: the generated JAX functions (backends/jax.py) must reproduce the
same frozen golden vectors as the symbolic spec, and must agree with the NumPy-lambdified
spec to float64 precision on random states — including under jit and vmap, whose op fusion
may only reassociate floating point, never change the math.

x64 is enabled here because the golden tolerances (1e-9) are below float32 resolution; the
backend itself follows the ambient JAX config."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (needs the x64 flag set first)

from backends import jax as backend
from properties.helpers import (N, flat_params, hover_speed, make_inputs, params_dict,
                                random_state, statedot_fn)
from properties.test_golden import (BLOCKS, FILES, _check, _flat_inputs, _flat_state,
                                    _load, _params)

#: Parameter overrides that light up every aero/motor path of the model at once
#: (k_h stays 0 — it is mutually exclusive with k_angle/k_hor by the validation rule).
FULL_MODEL = dict(I_rot=1e-8, k_flap=1e-7, k_angle=3.145, k_hor=7.245, k_v2=1e-4,
                  c_D=[1e-4, 1e-4, 2e-4], c_L=[1e-3, 1e-3, 0.0],
                  ka1=9.0, ka2=2e-4, kd1=6.0, kd2=1e-4)

RICH_INPUTS = make_inputs(v_wind=(1.0, -2.0, 0.5), F_ext=(0.01, 0.0, -0.02),
                          tau_ext=(1e-5, -2e-5, 3e-5))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_golden_jax(path):
    doc = _load(path)
    p = jnp.asarray(_params(doc))
    motor_model = doc.get("motor_model", "first_order")
    f = backend.statedot_fn(N, motor_model)
    kind = doc["kind"]
    lim = doc["params"]["limits"]
    tol = doc.get("tolerance", 1e-9)
    compare = doc.get("compare", list(BLOCKS))

    for k, c in enumerate(doc["cases"]):
        s = jnp.asarray(_flat_state(c))
        u = jnp.asarray(_flat_inputs(c))
        e = c["expected"]
        ctx = f"{path.stem} case {k} [jax]"

        if kind == "statedot":
            out = np.asarray(f(s, u, p))
            for block in compare:
                _check(out[BLOCKS[block]], np.asarray(e[block], float),
                       f"{ctx} [{block}]", tol)
            continue

        if kind == "step_ode":
            step = backend.rk4_step_fn(N, motor_model)
        elif kind == "step_exact_exp":
            step = backend.exact_exp_step_fn(N)
        else:
            raise AssertionError(f"unknown kind {kind}")

        s_next = backend.post_step(step(s, u, p, doc["dt"]),
                                   lim["rotor_speed_min"], lim["rotor_speed_max"])
        expected = np.concatenate([e["x"], e["v"], e["q_wxyz"], e["w"], e["rotor_speeds"]])
        _check(np.asarray(s_next), expected, ctx, tol)


@pytest.mark.parametrize("motor_model", ("first_order", "asymmetric"))
def test_matches_numpy_reference(motor_model):
    """Full-model random-state equivalence with the NumPy-lambdified spec, jitted."""
    rng = np.random.default_rng(7)
    p = flat_params(params_dict(**FULL_MODEL))
    f_ref = statedot_fn(motor_model)
    f_jax = jax.jit(backend.statedot_fn(N, motor_model))
    for k in range(20):
        s = random_state(rng)
        _check(np.asarray(f_jax(s, RICH_INPUTS, p)), f_ref(s, RICH_INPUTS, p),
               f"statedot {motor_model} case {k}", tol=1e-10)


def test_vmap_batches_match_loop():
    rng = np.random.default_rng(11)
    p = flat_params(params_dict(**FULL_MODEL))
    f = backend.statedot_fn(N, "first_order")
    batch = np.stack([random_state(rng) for _ in range(16)])
    out = jax.vmap(f, in_axes=(0, None, None))(batch, RICH_INPUTS, p)
    ref = np.stack([statedot_fn("first_order")(s, RICH_INPUTS, p) for s in batch])
    _check(np.asarray(out), ref, "vmap batch", tol=1e-10)


def test_pack_params_matches_reference():
    vals = params_dict(**FULL_MODEL)
    np.testing.assert_array_equal(np.asarray(backend.pack_params(vals)), flat_params(vals))


def test_pack_state_and_inputs_layout():
    s = backend.pack_state([1, 2, 3], [4, 5, 6], [1, 0, 0, 0], [7, 8, 9], [10, 11, 12, 13])
    u = backend.pack_inputs([1, 2, 3, 4], v_wind=(5, 6, 7), F_ext=(8, 9, 10),
                            tau_ext=(11, 12, 13))
    sl = backend.state_slices(N)
    np.testing.assert_array_equal(np.asarray(s)[sl["q_wxyz"]], [1, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(s)[sl["rotor_speeds"]], [10, 11, 12, 13])
    il = backend.input_slices(N)
    np.testing.assert_array_equal(np.asarray(u)[il["cmd_rotor_speeds"]], [1, 2, 3, 4])
    np.testing.assert_array_equal(np.asarray(u)[il["tau_ext"]], [11, 12, 13])


def test_rollout_hover_equilibrium():
    """A jitted scan rollout at exact hover speeds stays at hover: the physics, the
    integrator, and the post-step all agree with the spec's equilibrium."""
    vals = params_dict()
    p = backend.pack_params(vals)
    w_h = hover_speed(vals)
    s0 = backend.pack_state(jnp.zeros(3), jnp.zeros(3), jnp.array([1.0, 0, 0, 0]),
                            jnp.zeros(3), jnp.full(N, w_h))
    u = backend.pack_inputs(jnp.full(N, w_h))
    T, dt = 200, 0.0025
    rollout = jax.jit(backend.make_rollout(backend.rk4_step_fn(N), 0.0, 2500.0))
    traj = np.asarray(rollout(s0, jnp.tile(u, (T, 1)), p, dt))
    assert traj.shape == (T, 13 + N)
    assert np.isfinite(traj).all()
    np.testing.assert_allclose(np.linalg.norm(traj[:, 6:10], axis=1), 1.0, atol=1e-12)
    assert np.abs(traj[-1, 3:6]).max() < 1e-9, "hover equilibrium drifted"
    # One scan step equals step + post_step applied by hand.
    manual = backend.post_step(backend.rk4_step_fn(N)(s0, u, p, dt), 0.0, 2500.0)
    _check(traj[0], np.asarray(manual), "scan vs manual step", tol=1e-12)


def test_jacobian_is_finite():
    """Differentiability smoke: the generated code must admit finite forward-mode Jacobians
    at generic states (guards the planned differentiable variant)."""
    rng = np.random.default_rng(3)
    p = flat_params(params_dict(**FULL_MODEL))
    f = backend.statedot_fn(N, "first_order")
    J = jax.jacfwd(f)(jnp.asarray(random_state(rng)), jnp.asarray(RICH_INPUTS),
                      jnp.asarray(p))
    assert np.isfinite(np.asarray(J)).all()
