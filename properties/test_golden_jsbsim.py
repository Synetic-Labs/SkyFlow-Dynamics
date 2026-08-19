"""Authenticity: JSBSim executed-code golden vectors (kind == "jsbsim_terms").

golden/generate/gen_jsbsim.py runs the actual JSBSim engine (official PyPI wheel) and
freezes per-step tied-property sequences; every file's transcription was asserted against
the executed values at generation time. Here the SPEC side reproduces them:

  jsbsim_isa_atmosphere   spec.atmosphere USSA-1976 expressions at geopotential altitude.
  jsbsim_dryden_lowalt    spec.wind Dryden filters, discretized by the pole-prewarped
                          bilinear map (the route proven exact in
                          test_dryden_authenticity.py), with sigma/L from
                          spec.wind.dryden_low_altitude_scales — the low-altitude
                          closures' executed-code exercise — driven by the recovered
                          noise streams.
  jsbsim_prop_bldc        spec.atmosphere.advance_ratio/propeller_thrust identities,
                          spec.inflow induced velocity (axial + hover forms), and the
                          spec.motor_electrical DC-motor ODE Euler-replayed through the
                          reference's own discretization.
  jsbsim_rotor_inflow     spec.inflow.dynamic_inflow_lag exact-exponential step and
                          spec.rotor_aero.bramwell_torque/blade_profile_drag identities.
"""

import json
import math
import pathlib

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import atmosphere, inflow, motor_electrical, rotor_aero, wind
from skyflow_dynamics.spec.motor import exact_exp_step

VECTOR_DIR = pathlib.Path(__file__).resolve().parent.parent / "golden" / "vectors"


def _doc(name):
    d = json.loads((VECTOR_DIR / f"{name}.json").read_text())
    assert d["kind"] == "jsbsim_terms"
    assert d["provenance"]["jsbsim_version"], "executed-reference provenance missing"
    return d


# ---------------------------------------------------------------- ISA atmosphere

def test_jsbsim_isa_atmosphere():
    doc = _doc("jsbsim_isa_atmosphere")
    tol = doc["tolerance"]
    h_s = sp.Symbol("h", positive=True)
    T_e = atmosphere.temperature_troposphere(h_s)
    P_e = atmosphere.pressure_gradient_layer(h_s)
    f = sp.lambdify(h_s, [T_e, P_e, atmosphere.density(P_e, T_e),
                          atmosphere.speed_of_sound(T_e)])
    for c in doc["cases"]:
        T, P, rho, a = f(c["h_geopotential_m"])
        e = c["expected"]
        for got, ref in ((T, e["T_K"]), (P, e["P_Pa"]), (rho, e["rho_kg_m3"]),
                         (a, e["a_m_s"])):
            assert abs(got - ref) < tol * abs(ref), (c["h_geometric_ft"], got, ref)


# ---------------------------------------------------------------- Dryden low-altitude

def _bilinear_coeffs(H):
    """Coefficients (ascending powers of z^-1) of the prewarped-bilinear discretization
    of a spec forming filter, with the N(0, pi/dt) -> N(0,1) noise bridge folded in.
    test_dryden_authenticity.py proves this construction IS the reference's difference
    equation (Yeager eqs 18/20), parametrically in the prewarp constant C."""
    C_s, Tv_s, zi = sp.symbols("C T_v zeta", positive=True)
    Hd = H.subs(wind.s, C_s * (1 - zi) / (1 + zi)) * sp.sqrt(sp.pi / Tv_s)
    num, den = sp.fraction(sp.together(Hd))
    args = list(H.free_symbols - {wind.s}) + [C_s, Tv_s]
    b = [sp.lambdify(args, co) for co in reversed(sp.Poly(num, zi).all_coeffs())]
    a = [sp.lambdify(args, co) for co in reversed(sp.Poly(den, zi).all_coeffs())]
    return args, b, a


def test_jsbsim_dryden_low_altitude_closures():
    doc = _doc("jsbsim_dryden_lowalt")
    h_s, W20_s = sp.symbols("h W20", positive=True)
    scales = sp.lambdify((h_s, W20_s), wind.dryden_low_altitude_scales(h_s, W20_s))

    sig_s, L_s, V_s = sp.symbols("sigma_f L_f V_f", positive=True)
    args_u, bu, au = _bilinear_coeffs(wind.dryden_filter_u(sig_s, L_s, V_s))
    args_vw, bvw, avw = _bilinear_coeffs(wind.dryden_filter_vw(sig_s, L_s, V_s))

    def order(args, sig, L, V, C, Tv):
        m = {"sigma_f": sig, "L_f": L, "V_f": V, "C": C, "T_v": Tv}
        return [m[str(s)] for s in args]

    for c in doc["cases"]:
        dt = c["params"]["dt_s"]
        W20 = c["params"]["W20_fps"]
        seqs = c["sequence"]
        n = len(seqs["h_ft"])
        got = {"u": np.zeros(n), "v": np.zeros(n), "w": np.zeros(n)}
        y = {k: [0.0, 0.0] for k in "uvw"}      # y_{k-1}, y_{k-2}
        x = {k: [0.0, 0.0] for k in "uvw"}      # noise_{k-1}, noise_{k-2}
        for k in range(n):
            h = max(seqs["h_ft"][k], 10.0)      # the reference's height clip
            V = seqs["V_fps"][k]
            L_u, L_v, L_w, s_u, s_v, s_w = scales(h, W20)
            tau_u = L_u / V
            C = 1.0 / tau_u / math.tan(dt / 2.0 / tau_u)   # u-axis prewarp, all axes
            # u: first order
            a = [f(*order(args_u, s_u, L_u, V, C, dt)) for f in au]
            b = [f(*order(args_u, s_u, L_u, V, C, dt)) for f in bu]
            nu = seqs["noise_u"][k]
            yu = (b[0] * nu + b[1] * x["u"][0] - a[1] * y["u"][0]) / a[0]
            x["u"] = [nu, x["u"][0]]
            y["u"] = [yu, y["u"][0]]
            got["u"][k] = yu
            # v, w: second order (v uses the u-axis scales per the closures)
            for comp, sig, L in (("v", s_v, L_v), ("w", s_w, L_w)):
                a = [f(*order(args_vw, sig, L, V, C, dt)) for f in avw]
                b = [f(*order(args_vw, sig, L, V, C, dt)) for f in bvw]
                nk = seqs[f"noise_{comp}"][k]
                yk = (b[0] * nk + b[1] * x[comp][0] + b[2] * x[comp][1]
                      - a[1] * y[comp][0] - a[2] * y[comp][1]) / a[0]
                x[comp] = [nk, x[comp][0]]
                y[comp] = [yk, y[comp][0]]
                got[comp][k] = yk
        exp = c["expected"]
        for comp, key in (("u", "turb_north_fps"), ("v", "turb_east_fps"),
                          ("w", "turb_down_fps")):
            np.testing.assert_allclose(
                got[comp], np.asarray(exp[key]), rtol=1e-7, atol=1e-9,
                err_msg=f"h={c['sequence']['h_ft'][0]:.0f}ft {comp}")


# ---------------------------------------------------------------- prop + BLDC motor

def test_jsbsim_prop_advance_ratio_thrust_and_induced_velocity():
    doc = _doc("jsbsim_prop_bldc")
    p = doc["params"]
    D, A = p["D_m"], p["disk_area_m2"]
    Va_s, T_s, rho_s, n_s = sp.symbols("V_a T rho n", real=True)
    J_fn = sp.lambdify((Va_s, n_s), atmosphere.advance_ratio(Va_s, n_s, D))
    vi_fn = sp.lambdify((Va_s, T_s, rho_s),
                        inflow.induced_velocity_axial(Va_s, T_s, rho_s, A))
    vh_fn = sp.lambdify((T_s, rho_s), inflow.hover_induced_velocity(T_s, rho_s, A))
    CT_s = sp.Symbol("C_T0", real=True)
    T_fn = sp.lambdify((CT_s, rho_s, n_s),
                       atmosphere.propeller_thrust(lambda j: CT_s, None, rho_s, n_s, D))
    for c in doc["cases"]:
        rho = c["params"]["rho_kg_m3"]
        Va = c["params"]["V_axial_m_s"]
        s = c["sequence_si"]
        # errstate: the lambdified Piecewise evaluates its dead branch (sqrt of negative)
        with np.errstate(invalid="ignore"):
            for k in range(1, len(s["Omega_rad_s"])):
                n_rev = s["Omega_rad_s"][k - 1] / (2 * math.pi)  # entry speed, rev/s
                if n_rev * 0.3048 <= 0.01:  # reference's J fallback region (rps<=0.01)
                    continue
                T = T_fn(s["CT"][k], rho, n_rev)
                assert abs(T - s["thrust_N"][k]) < 1e-7 * max(1.0, abs(T)), (k, T)
                assert abs(J_fn(Va, n_rev) - s["J"][k]) < 1e-9, (k, s["J"][k])
                vi = vi_fn(Va, T, rho)
                assert abs(vi - s["vi_m_s"][k]) < 1e-7, (k, vi, s["vi_m_s"][k])
                if Va == 0.0 and T > 0:
                    assert abs(vh_fn(T, rho) - s["vi_m_s"][k]) < 1e-7, "hover form"


def test_jsbsim_dc_motor_ode_euler_replay():
    # Static cases (V_a = 0): J = 0 identically, so the propeller load is exactly
    # k_m*Omega^2 with k_m = C_P(0)*rho*D^5/(8 pi^3), and the reference's shaft update is
    # the explicit-Euler step of spec.motor_electrical.dc_motor_speed_dynamics (b = 0).
    doc = _doc("jsbsim_prop_bldc")
    p = doc["params"]
    W_s, Vm_s, km_s = sp.symbols("W V_m k_m", positive=True)
    rate = motor_electrical.dc_motor_speed_dynamics(
        W_s, Vm_s, p["J_r_kg_m2"], p["K_q_Nm_A"], p["K_e_V_s_rad"], p["R_a_ohm"],
        km_s, 0)
    rate_fn = sp.lambdify((W_s, Vm_s, km_s), rate)
    static = [c for c in doc["cases"] if c["params"]["V_axial_m_s"] == 0.0]
    assert len(static) >= 3, "static spin-up cases missing"
    for c in static:
        s = c["sequence_si"]
        dt = c["params"]["dt_s"]
        Vm = c["params"]["throttle"] * p["V_max_V"]
        om = np.asarray(s["Omega_rad_s"])
        # k_m from the recorded load power: Q_aero = P_req/Omega_entry = k_m*Omega^2
        kms = np.asarray(s["P_req_W"][5:]) / om[4:-1] ** 3
        assert np.ptp(kms) < 1e-6 * kms.mean(), "C_P(0) load not constant"
        km = float(kms.mean())
        w = om[5]
        for k in range(6, len(om)):
            w = w + dt * rate_fn(w, Vm, km)
            assert abs(w - om[k]) < 1e-7 * max(1.0, om[k]), \
                (c["params"]["throttle"], k, w, om[k])


# ---------------------------------------------------------------- rotor inflow / torque

def test_jsbsim_dynamic_inflow_exact_exp_step():
    doc = _doc("jsbsim_rotor_inflow")
    tau = doc["params"]["tau_s"]
    nu_s, nueq_s, dt_s = sp.symbols("nu nu_eq dt_", positive=True)
    step = exact_exp_step(sp.Matrix([nu_s]), sp.Matrix([nueq_s]), dt_s, tau)[0]
    # the step integrates exactly the spec lag ODE (proven in test_candidates)
    assert sp.simplify(sp.diff(step, dt_s)
                       - inflow.dynamic_inflow_lag(step, nueq_s, tau)) == 0
    for c in doc["cases"]:
        dt = c["params"]["dt_s"]
        f = sp.lambdify((nu_s, nueq_s), step.subs(dt_s, dt))
        s = c["sequence"]
        for k in range(1, len(s["nu"])):
            pred = f(s["nu"][k - 1], s["nu_eq"][k])
            assert abs(pred - s["nu"][k]) < 1e-10, (c["name"], k, pred, s["nu"][k])


def test_jsbsim_bramwell_torque_identity():
    doc = _doc("jsbsim_rotor_inflow")
    p = doc["params"]
    T_s, H_s, mu_s, lam_s, rho_s, CT_s = sp.symbols("T H mu lam rho C_T", real=True)
    delta = rotor_aero.blade_profile_drag(CT_s, p["a_slope"], p["solidity"])
    Q = rotor_aero.bramwell_torque(T_s, H_s, mu_s, lam_s, rho_s, p["blades"],
                                   p["chord_m"], p["R_m"], p["Omega_rad_s"], delta)
    Q_fn = sp.lambdify((T_s, H_s, mu_s, lam_s, rho_s, CT_s), Q)
    mus = []
    for c in doc["cases"]:
        rho = c["params"]["rho_kg_m3"]
        s = c["sequence"]
        for k in range(1, len(s["nu"])):
            got = Q_fn(s["thrust_N"][k], s["H_N"][k], s["mu"][k], s["lambda"][k],
                       rho, s["CT"][k])
            ref = s["torque_N_m"][k]
            assert abs(got - ref) < 1e-7 * max(1.0, abs(ref)), (c["name"], k, got, ref)
        mus.append(s["mu"][-1])
    # the (1 + 4.5 mu^2) forward-flight growth is actually exercised
    assert max(mus) > 0.01, "no edgewise case"
