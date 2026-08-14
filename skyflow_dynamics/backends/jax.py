"""
JAX backend — generated from the SymPy spec, never handwritten.

Every dynamics function here is emitted from spec/ by sympy.lambdify's JAX printer on first
use (cached per rotor count and motor model): the math stays in spec/, this module only fixes
the target framework and the calling convention. Authenticity is enforced by
properties/test_backend_jax.py, which replays the full golden-vector suite through these
functions and cross-checks them against the NumPy-lambdified spec.

Calling convention — flat float arrays in the canonical spec.symbols ordering (the same
ordering the golden files use):

    state  s : (13+n,)  = (x_W(3), v_W(3), q_wxyz(4), ω_B(3), Ω(n))
    inputs u : (9+n,)   = (Ω_c(n), v_wind_W(3), F_ext_W(3), τ_ext_B(3))
    params p : (·,)     = spec.symbols.Params.flat() order — build with pack_params()

All returned functions are single-vehicle and transformation-closed: compose with jax.jit for
speed, jax.vmap for fleets, jax.grad / jax.jacfwd for derivatives (the spec keeps the ODE
right-hand side smooth apart from the asymmetric motor's Ω_c = Ω branch point — see the
registry's discretization/differentiation notes). Precision follows the ambient JAX config:
enable x64 (jax.config.update("jax_enable_x64", True)) to reproduce the golden vectors at
their 1e-9 tolerances; float32 is the usual choice for RL-scale rollouts.

Harness boundary: post_step() and make_rollout() replicate only the reference harness
operations the golden step vectors pin (quaternion renormalization, rotor-speed clipping,
zero-order-held inputs). Command transport delay, control-rate ZOH scheduling, disturbance
resampling and RNG stay harness-side (see README) and belong to the consuming simulator.
"""

from functools import cache

import jax
import jax.numpy as jnp
import sympy as sp

from skyflow_dynamics.spec import dynamics, motor, parameters, sensors
from skyflow_dynamics.spec.discretization import rk4_step
from skyflow_dynamics.spec.symbols import input_symbols, param_symbols, state_symbols


def state_slices(n: int = 4) -> dict:
    """Block layout of the flat state vector (13+n,)."""
    return {"x": slice(0, 3), "v": slice(3, 6), "q_wxyz": slice(6, 10),
            "w": slice(10, 13), "rotor_speeds": slice(13, 13 + n)}


def input_slices(n: int = 4) -> dict:
    """Block layout of the flat input vector (9+n,)."""
    return {"cmd_rotor_speeds": slice(0, n), "v_wind": slice(n, n + 3),
            "F_ext": slice(n + 3, n + 6), "tau_ext": slice(n + 6, n + 9)}


def pack_state(x, v, q_wxyz, w, rotor_speeds) -> jnp.ndarray:
    """Flat state from blocks (world position/velocity, wxyz quaternion, body rate, Ω)."""
    return jnp.concatenate([jnp.atleast_1d(jnp.asarray(b)) for b in
                            (x, v, q_wxyz, w, rotor_speeds)])


def pack_inputs(cmd_rotor_speeds, v_wind=(0.0, 0.0, 0.0), F_ext=(0.0, 0.0, 0.0),
                tau_ext=(0.0, 0.0, 0.0)) -> jnp.ndarray:
    """Flat inputs from blocks (Ω_c, wind velocity W-frame, external force W / torque B)."""
    return jnp.concatenate([jnp.atleast_1d(jnp.asarray(b)) for b in
                            (cmd_rotor_speeds, v_wind, F_ext, tau_ext)])


def pack_params(values: dict) -> jnp.ndarray:
    """
    Numeric parameter dict (spec.parameters.SCHEMA keys, the golden-file `params` format;
    a harness-side "limits" entry is ignored) → flat parameter vector in Params.flat() order.
    Runs spec.parameters.validate() — rejects double-counted aero terms, bad spin signs,
    non-unit thrust axes.
    """
    n = len(values["spin"])
    P = param_symbols(n)
    sub = parameters.substitution(P, values)
    return jnp.asarray([float(sub[sym]) for sym in P.flat()])


@cache
def _tau_m_index(n: int) -> int:
    """Position of τ_m in the flat parameter vector (needed by the exact-exp split)."""
    P = param_symbols(n)
    return P.flat().index(P.tau_m)


@cache
def statedot_fn(n: int = 4, motor_model: str = "first_order"):
    """
    The continuous model ṡ = f(s, u, p) as a JAX function (13+n,) — code-generated from
    spec.dynamics.statedot for `n` rotors and the selected motor model
    (spec.dynamics.MOTOR_MODELS). First call per (n, motor_model) pays the symbolic
    build + printing cost once; the returned function is pure and traceable.
    """
    S, U, P = state_symbols(n), input_symbols(n), param_symbols(n)
    expr = dynamics.statedot(S, U, P, motor_model)
    raw = sp.lambdify((S.flat(), U.flat(), P.flat()), expr, modules="jax", cse=True)

    def f(s, u, p):
        return jnp.asarray(raw(s, u, p)).reshape(-1)

    return f


@cache
def rk4_step_fn(n: int = 4, motor_model: str = "first_order"):
    """
    One fixed-step RK4 step s⁺ = step(s, u, p, dt) over the full model — the reference
    integrator (spec.discretization.rk4_step; inputs zero-order-held across the step).
    Returns the raw integrator output: apply post_step() afterwards to match the reference
    harness (and the golden step vectors).
    """
    f = statedot_fn(n, motor_model)

    def step(s, u, p, dt):
        return rk4_step(lambda t, x: f(x, u, p), s, dt)

    return step


@cache
def exact_exp_step_fn(n: int = 4):
    """
    RK4 step with the exact-exponential motor splitting (first-order lag only —
    spec.motor.exact_exp_step): inside the stages the rotor speed follows the closed form
    Ω(t) = Ω_c + (Ω₀−Ω_c)·e^(−t/τ_m) at the stage time and its ODE slot is held; after the
    step Ω is set to Ω(dt) analytically. Removes the stiffest mode from the integrator with
    a per-step gradient factor e^(−dt/τ) ∈ (0,1). Matches the step_exact_exp golden vectors
    (rpg_flightning discretization); apply post_step() afterwards as with rk4_step_fn.
    """
    f = statedot_fn(n, "first_order")
    rotors = slice(13, 13 + n)
    i_tau = _tau_m_index(n)

    def step(s, u, p, dt):
        tau = p[i_tau]
        W0, Wc = s[rotors], u[:n]

        def f_split(t, x):
            x = x.at[rotors].set(Wc + (W0 - Wc) * jnp.exp(-t / tau))
            return f(x, u, p).at[rotors].set(0.0)

        s_next = rk4_step(f_split, s, dt)
        return s_next.at[rotors].set(Wc + (W0 - Wc) * jnp.exp(-dt / tau))

    return step


def post_step(s, rotor_speed_min, rotor_speed_max):
    """
    The reference harness post-step the golden step vectors include: renormalize the
    quaternion, clip rotor speeds to the vehicle's operating limits (the `limits` entry of
    the parameter dicts). Batch-safe (operates on the last axis).
    """
    q = s[..., 6:10]
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    W = jnp.clip(s[..., 13:], rotor_speed_min, rotor_speed_max)
    return jnp.concatenate([s[..., :6], q, s[..., 10:13], W], axis=-1)


@cache
def param_slices(n: int = 4) -> dict:
    """
    SCHEMA name → index array into the flat parameter vector (pack_params order).
    Multi-entry names map to all their positions (inertia → its 6 stored entries,
    per-rotor names → n entries, vectors → 3). Intended for harness-side parameter
    randomization masks and diagnostics — the layout is derived from the symbols,
    never hardcoded.
    """
    import numpy as np
    P = param_symbols(n)
    pos = {s: i for i, s in enumerate(P.flat())}
    out = {
        "mass": [pos[P.mass]], "grav": [pos[P.grav]],
        "inertia": [pos[P.inertia[a, b]] for a, b in
                    ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))],
        "rotor_pos": [pos[c] for r in P.rotor_pos for c in r],
        "spin": [pos[s] for s in P.spin],
        "axis": [pos[c] for e in P.axis for c in e],
        "tau_m": [pos[P.tau_m]], "ka1": [pos[P.ka1]], "ka2": [pos[P.ka2]],
        "kd1": [pos[P.kd1]], "kd2": [pos[P.kd2]], "I_rot": [pos[P.I_rot]],
        "c_D": [pos[c] for c in P.c_D], "c_L": [pos[c] for c in P.c_L],
        "k_d": [pos[P.k_d]], "k_z": [pos[P.k_z]], "k_flap": [pos[P.k_flap]],
        "k_h": [pos[P.k_h]], "k_angle": [pos[P.k_angle]], "k_hor": [pos[P.k_hor]],
        "k_v2": [pos[P.k_v2]], "r_prop": [pos[P.r_prop]],
    }
    for name in ("ct0", "ct1", "ct2", "cq0", "cq1", "cq2"):
        out[name] = [pos[s] for s in getattr(P, name)]
    return {k: np.asarray(v) for k, v in out.items()}


@cache
def throttle_to_speed_fn():
    """
    Normalized throttle → commanded rotor speed, generated from the verified throttle
    curve (spec.motor.throttle_to_speed):  Ω_c = (Ω_max−Ω_min)·√(k·u² + (1−k)·u) + Ω_min.
    Returns f(u, w_min, w_max, k) — elementwise, broadcasts over any batch shape.
    """
    u, w_min, w_max, k = sp.symbols("u w_min w_max k", real=True)
    return sp.lambdify((u, w_min, w_max, k),
                       motor.throttle_to_speed(u, w_min, w_max, k), modules="jax")


@cache
def imu_fn(n: int = 4, motor_model: str = "first_order"):
    """
    Exact IMU measurement (spec.sensors.imu) with v̇, ω̇ substituted from the full model:

        f(s, u, p, p_BS, R_BS_flat) -> (6,) = (accel_S(3) specific force, gyro_S(3))

    p_BS: sensor offset in the body frame (3,); R_BS_flat: sensor→body mounting rotation,
    row-major (9,). Noise, bias, and sample-rate behavior stay harness-side (the spec's
    sensor boundary). Single-vehicle; compose jax.vmap for fleets.
    """
    S, U, P = state_symbols(n), input_symbols(n), param_symbols(n)
    sd = dynamics.statedot(S, U, P, motor_model)
    v_dot, w_dot = sp.Matrix(sd[3:6, 0]), sp.Matrix(sd[10:13, 0])
    p_BS = sp.Matrix(sp.symbols("pBS_1:4", real=True))
    R_BS = sp.Matrix(3, 3, sp.symbols("RBS_1:10", real=True))
    accel, gyro = sensors.imu(S.q, v_dot, S.w, w_dot, p_BS, R_BS, P.grav)
    raw = sp.lambdify((S.flat(), U.flat(), P.flat(), tuple(p_BS.flat()), tuple(R_BS.flat())),
                      sp.Matrix.vstack(accel, gyro), modules="jax", cse=True)

    def f(s, u, p, p_BS, R_BS_flat):
        return jnp.asarray(raw(s, u, p, p_BS, R_BS_flat)).reshape(-1)

    return f


def make_rollout(step, rotor_speed_min=0.0, rotor_speed_max=jnp.inf):
    """
    Fixed-step trajectory rollout via lax.scan — the fast path for simulation:

        rollout(s0, u_seq, p, dt) -> (T, 13+n) states after each step,

    with u_seq (T, 9+n) zero-order-held per step and post_step() applied after every step,
    exactly as the reference harnesses integrate. `step` is rk4_step_fn(...) or
    exact_exp_step_fn(...). jit it (optionally vmap over (s0, u_seq) for fleets); any
    richer scheduling (command delay lines, control-rate ZOH, disturbance resampling) is
    harness-side and composes on top by generating u_seq.
    """
    def rollout(s0, u_seq, p, dt):
        def body(s, u):
            s_next = post_step(step(s, u, p, dt), rotor_speed_min, rotor_speed_max)
            return s_next, s_next

        return jax.lax.scan(body, s0, u_seq)[1]

    return rollout
