"""
Triage numbers for the 2026-11-16 evidence review of a rotor in its own wake — descent
(vortex ring state, "propwash"), ground effect, and rotor-on-rotor interference
(REFERENCES.md, "Own-wake aerodynamics"). The sizes quoted there and in the registry
entries vrs_johnson_inflow, inflow_thrust_law, propwash_thrust_unsteadiness,
ground_effect_fixed_rpm, rotor_wake_interference and blade_flapping_moment.

    uv run python tools/wake_triage.py

Checks:
  1. F-32: thrust at CONSTANT rotor speed against descent rate — the backend model and the
     verified BEM model (with the quartic VRS inflow) against the measured curve.
  2. Johnson's VRS inflow (spec.descent) through the linear thrust law (spec.inflow): the
     curve it predicts, and the two parameters fitted to each measured data set.
  3. Thrust fluctuation in descent: size relative to the mean loss; the roll disturbance it
     makes on RACER_5IN.
  4. F-33: the ground-effect candidates against measurements at constant rotor speed.
  5. Time scales: dynamic inflow, VRS development.
  6. F-34: the hub pitching moment of a rigid propeller in edgewise flow, and rear-rotor
     interference.

MEASURED values below are read from published figures and tables. Each block says which,
who read it, and how well a figure can be read. They size effects; they are not a dataset.
"""

import functools
import math

import numpy as np
import sympy as sp

from skyflow_dynamics.spec import descent, ground_effect, inflow
from skyflow_dynamics.spec.parameters import CRAZYFLIE, RACER_5IN
from tools import bem_rate_damping as brd

RHO = 1.204   # kg/m³, the density of the BEM reference case (golden/vectors/agilicious_bem.json)

# --------------------------------------------------------------------------------------
# Measured data (figure read-offs unless stated). x = descent rate / v_h0, v_h0 the hover
# induced velocity at the test rotor speed; y = mean thrust / hover thrust, rotor speed held.
# --------------------------------------------------------------------------------------

#: Veismann, Dougherty, Gharib, Flow 3 E7 (2023). 6×3: Fig. 5b (S = 2, three rotor speeds
#: collapsed); 5×4 and 4×4.5: Fig. 6. Read from the page images for this file, ±0.02.
#: Table 1 gives σ, θ.75 (deg) and hover C_T = T/(ρA(ΩR)²). sigma_peak: Fig. 7, the peak of
#: σ(T)/T̄ over the single-rotor and S ≥ 1.5 series.
VEISMANN_2023 = {
    "HQ 6x3": {"solidity": 0.1070, "theta75_deg": 10.82, "ct_hover": 0.0077,
               "sigma_peak": 0.16,
               "curve": [(0.25, 1.06), (0.5, 1.07), (0.75, 0.99), (1.0, 0.88), (1.2, 0.71),
                         (1.5, 0.88), (1.75, 1.03), (2.0, 1.13), (2.4, 1.33)]},
    "HQ 5x4": {"solidity": 0.1261, "theta75_deg": 18.15, "ct_hover": 0.0116,
               "sigma_peak": 0.13,
               "curve": [(0.25, 1.02), (0.5, 1.03), (0.75, 1.00), (1.0, 0.90), (1.2, 0.78),
                         (1.5, 0.90), (1.75, 0.97), (2.0, 0.98)]},
    "Gemfan 4x4.5": {"solidity": 0.1221, "theta75_deg": 15.91, "ct_hover": 0.0063,
                     "sigma_peak": 0.14,
                     "curve": [(0.25, 1.05), (0.5, 1.07), (0.75, 1.02), (1.0, 0.90),
                               (1.25, 0.70), (1.5, 0.93), (1.75, 1.08), (2.0, 1.15)]},
}

#: Shetty & Selig, AIAA 2011-1254, Figs. 4 and 11: APC Thin Electric 10×5 at 4000 rpm
#: (θ.75 = 13.39°). Read by the literature search (±0.005 in C_T), checked against both
#: figures for this file. x = J/0.242 with hover C_T = 0.093 (propeller convention).
#: Table 1 of the paper: σ = 0.0806, θ.75 = 13.39°; C_T converted to T/(ρA(ΩR)²) by 4/π³.
SHETTY_2011_ROTOR = {"solidity": 0.0806, "theta75_deg": 13.39, "ct_hover": 0.093 * 4 / math.pi ** 3}
SHETTY_2011_10X5 = [(0.62, 0.93), (0.79, 0.93), (0.95, 0.82), (1.07, 0.81), (1.24, 0.87),
                    (1.36, 0.95), (1.53, 1.05), (1.65, 1.14), (1.78, 1.22), (1.94, 1.34),
                    (2.07, 1.42), (2.23, 1.51), (2.52, 1.70), (2.98, 1.97)]

#: Veismann, Yos, Gharib, Phys. Fluids 34, 035124 (2022), Figs. 7–8: 6-inch printed
#: NACA0012 reference rotor. Read by the literature search; not re-read for this file.
VEISMANN_2022_REF = [(0.4, 1.02), (0.75, 0.95), (1.0, 0.87), (1.2, 0.80), (1.5, 0.93),
                     (1.75, 1.05), (2.0, 1.15)]

#: Thrust gain in ground effect at constant rotor speed: (z/R, T/T_∞ − 1).
GROUND_MEASURED = {
    # He & Leang, AIAA J. 2020, Table 2 fit of thrust-stand data (constant speed, 13 heights)
    "single 5x3 three-blade prop (He & Leang, Table 2 fit)":
        [(z, 0.53 * math.exp(-1.80 * z)) for z in (0.5, 1.0, 2.0, 4.0)],
    # Kan et al., RA-L 2019, Eq. (12): hover thrust needed T_IGE = T_h·(1 − 3R/(25z)); the
    # gain at fixed speed is its reciprocal. AscTec Hummingbird (R = 0.1 m) in free flight.
    "Hummingbird quadrotor in flight (Kan et al., Eq. 12)":
        [(z, 1 / (1 - 3 / (25 * z)) - 1) for z in (0.5, 1.0, 2.0, 4.0)],
    # Carter, Bouchard, Quinn, AIAA J. 2021, Fig. 2a: tethered Crazyflie 2.0 on a load cell.
    # Read from the figure by the literature search (±0.01); not re-read for this file.
    "Crazyflie on a load cell (Carter et al., Fig. 2a)":
        [(1.14, 0.18), (1.5, 0.13), (1.8, 0.11), (2.2, 0.086), (2.85, 0.07)],
}

#: Isolated 3DR SOLO rotor (10 in), NASA/TM-2018-219758 Appendix C, read from the tables:
#: C68a point 31 and C69a point 18 (0° pitch, 5700 rpm), C84a point 11 (hover, 5600 rpm).
#: Fz in lb, My (nose-up positive, about the hub) in in-lb, speed in ft/s.
RUSSELL_SOLO_ROTOR = {"radius_in": 5.0, "rpm": 5700.0, "hover_my": 0.188,
                      "edgewise": [{"speed_fts": 19.34, "fz": 1.032, "my": 0.875},
                                   {"speed_fts": 39.59, "fz": 1.218, "my": 1.358}]}


# --------------------------------------------------------------------------------------
# 1. F-32: constant-speed thrust in descent, model against measurement
# --------------------------------------------------------------------------------------

def hover_induced_velocity(v: dict, rotor_speed: float | None = None, rho: float = RHO) -> float:
    """v_h = √(T/(2ρA)) per rotor; at hover thrust unless a rotor speed is given."""
    area = math.pi * v["r_prop"] ** 2
    n = len(v["ct2"])
    thrust = v["mass"] * v["grav"] / n if rotor_speed is None else v["ct2"][0] * rotor_speed ** 2
    return math.sqrt(thrust / (2 * rho * area))


def backend_descent_ratio(v: dict, x: float, rho: float = RHO) -> float:
    """T/T_static at fixed rotor speed in the backend: c_T·Ω² − k_z·Ω·V_z with V_z = −x·v_h.
    Linear in x and independent of rotor speed: slope k_z/√(2ρA·c_T)."""
    area = math.pi * v["r_prop"] ** 2
    return 1 + x * v["k_z"] / math.sqrt(2 * rho * area * v["ct2"][0])


@functools.cache
def _bem():
    model = brd.load("canonical")
    return model, brd.hover_speed(model)


def bem_descent_ratio(x: float) -> float:
    """T/T_hover of the verified BEM family (quartic VRS inflow and its executed blend
    included) at the hover rotor speed, vehicle level and descending at x·v_h."""
    (ga, bp, quad), w_h = _bem()

    def thrust(v_down):
        case = {"v_W": [0.0, 0.0, -v_down], "q_wxyz": [1, 0, 0, 0], "w_B": [0, 0, 0],
                "mot": [w_h] * 4}
        return float(ga.run_bem(bp, quad, case, np.full(4, 5.0), np.full(4, 5.0))["thrust"][0])

    t0 = thrust(0.0)
    v_h = math.sqrt(t0 / (2 * bp["rho"] * math.pi * bp["r_prop"] ** 2))
    return thrust(x * v_h) / t0


def quartic_net_inflow(x: float) -> float:
    """(V_z + v)/v_h of the Leishman quartic (spec.bem.vrs_induced_velocity) at descent x."""
    return -x + 1 + 1.125 * x - 1.372 * x ** 2 + 1.718 * x ** 3 - 0.655 * x ** 4


# --------------------------------------------------------------------------------------
# 2. Johnson's inflow, evaluated numerically for any V_x from the spec's pieces
# --------------------------------------------------------------------------------------

def _momentum(vz: float, vx: float, branch: str) -> float:
    roots = np.roots([1.0, 2 * vz, vx ** 2 + vz ** 2, 0.0, -1.0])
    real = np.sort(roots[(abs(roots.imag) < 1e-9) & (roots.real > 0)].real)
    return float(real[-1] if branch == "helicopter" else real[0])


def _slope(v: float, vz: float, vx: float) -> float:
    w = vz + v
    return -v * w / (w ** 2 + vx ** 2 + v * w)      # spec.descent.momentum_inflow_slope


@functools.cache
def _compiled() -> dict:
    """The spec's symbolic pieces compiled once: (V+v)_X stays a free parameter."""
    vz, vx, peak, s_n, s_x = sp.symbols("vz vx peak s_n s_x", real=True)
    z_a, v_a, dv_a, z_b, v_b = sp.symbols("z_a v_a dv_a z_b v_b", real=True)
    c = {**descent.JOHNSON_VRS, "VvX": peak}
    points = descent.johnson_transition_points(vx, c)
    cubic = descent.johnson_baseline_cubic(vz, z_a, v_a, dv_a, z_b, v_b)
    return {
        "axial": sp.lambdify((vz, peak), descent.johnson_vrs_inflow_axial(vz, c=c), "numpy"),
        "points": sp.lambdify(vx, [points[k] for k in "ABDNXE"], "math"),
        "cubic": sp.lambdify((vz, z_a, v_a, dv_a, z_b, v_b), list(cubic), "math"),
        "increment": sp.lambdify((vz, vx, s_n, s_x, peak),
                                 descent.johnson_vrs_increment(vz, vx, s_n, s_x, c), "math"),
    }


def johnson_baseline(vz: float, vx: float) -> tuple:
    """(v_base, dv_base/dV_z), Table 3 step 2, momentum roots found numerically."""
    c = descent.JOHNSON_VRS
    if vz >= 0 or vx >= c["VxC"]:
        v = _momentum(vz, vx, "helicopter")
        return v, _slope(v, vz, vx)
    z_a, z_b = _compiled()["points"](vx)[:2]
    if vz >= z_a:
        v = _momentum(vz, vx, "helicopter")
        return v, _slope(v, vz, vx)
    if vz <= z_b:
        v = _momentum(vz, vx, "windmill")
        return v, _slope(v, vz, vx)
    v_a = _momentum(z_a, vx, "helicopter")
    v, dv = _compiled()["cubic"](vz, z_a, v_a, _slope(v_a, z_a, vx), z_b,
                                 _momentum(z_b, vx, "windmill"))
    return float(v), float(dv)


def johnson_inflow(vz, vx: float = 0.0, peak: float | None = None, kappa: float = 1.0,
                   f: float = 1.0):
    """v/v_h of the whole model. peak overrides (V+v)_X, Table 4's net-inflow maximum.
    Vertical flight (vx = 0) with f = 1 takes arrays and uses the explicit spec form."""
    peak = float(descent.JOHNSON_VRS["VvX"]) if peak is None else peak
    if vx == 0 and f == 1:
        with np.errstate(invalid="ignore"):     # the unselected momentum branch
            return kappa * _compiled()["axial"](np.asarray(vz, float), peak)
    base = johnson_baseline(vz, vx)[0]
    if not (vz < 0 and vx < descent.JOHNSON_VRS["VxM"]):
        return kappa * base
    _, _, _, z_n, z_x, _ = _compiled()["points"](vx)
    delta = _compiled()["increment"](vz, vx, johnson_baseline(z_n, vx)[1],
                                     johnson_baseline(z_x, vx)[1], peak)
    return kappa * (base + f * float(delta))


_US = np.linspace(0.05, 2.6, 256)


def johnson_descent_ratio(x: float, s: float, peak: float | None = None, vx: float = 0.0,
                          f: float = 1.0) -> float:
    """
    T/T_static at fixed rotor speed, descending at x·v_h0 with in-plane speed vx·v_h0: the
    root u² of  u² = 1 − s·((−x + u·ṽ(−x/u, vx/u)) − 1),  u = v_h/v_h0 = √(T/T_static),
    ṽ Johnson's inflow in units of the CURRENT v_h (κ cancels in the ratio). The lowest
    root is taken when the unstable band gives three.
    """
    def g(u):
        if vx == 0 and f == 1:
            v = johnson_inflow(-x / u, 0.0, peak)
        else:
            v = np.array([johnson_inflow(-x / ui, vx / ui, peak, f=f)
                          for ui in np.atleast_1d(u)])
        return u * u - (1 - s * (-x + u * v - 1))    # spec.inflow.inflow_thrust_ratio

    gs = np.atleast_1d(g(_US))
    i = int(np.argmax(gs[:-1] * gs[1:] <= 0))
    if not gs[i] * gs[i + 1] <= 0:
        raise ValueError("no root")
    lo, hi = _US[i], _US[i + 1]
    g_lo = gs[i]
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        g_mid = float(np.atleast_1d(g(np.array([mid])))[0])
        lo, hi, g_lo = (mid, hi, g_mid) if g_lo * g_mid > 0 else (lo, mid, g_lo)
    return (0.5 * (lo + hi)) ** 2


def blade_element_sensitivity(rotor: dict, kappa: float = 1.0, tip_loss: float = 1.0) -> float:
    """s from a rotor's measured geometry: λ_h = κ·√(C_T/2), pitch θ.75 (no zero-lift
    correction — the published tables give the geometric angle only)."""
    lam = kappa * math.sqrt(rotor["ct_hover"] / 2)
    return float(inflow.inflow_thrust_sensitivity(tip_loss * math.radians(rotor["theta75_deg"]),
                                                  lam))


def fit_peak_to_minimum(curve: list, s: float) -> float:
    """The net-inflow maximum (V+v)_X that makes the model's thrust minimum as deep as the
    measured one, for a given blade sensitivity s (bisection; the depth grows with it)."""
    target = min(y for _, y in curve)
    lo, hi = 1.0, 3.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        depth = minimum(lambda x, m=mid: johnson_descent_ratio(x, s, m))[1]
        lo, hi = (mid, hi) if depth > target else (lo, mid)
    return 0.5 * (lo + hi)


def descent_comparison(curve: list, s: float) -> dict:
    """Model against one measured curve: with Table 4's (V+v)_X = 1.25, and with (V+v)_X set
    to the measured depth."""
    peak = fit_peak_to_minimum(curve, s)
    default = lambda x: johnson_descent_ratio(x, s)
    fitted = lambda x: johnson_descent_ratio(x, s, peak)
    measured_x, measured_y = min(curve, key=lambda pt: pt[1])
    band = [pt for pt in curve if pt[0] <= 1.75]     # the VRS band; beyond it the blade stalls
    beyond = [fitted(x) - y for x, y in curve if x >= 2.0]
    return {"s": s, "measured_min": (measured_x, measured_y),
            "default_min": minimum(default), "default_rms": curve_rms(band, default),
            "peak": peak, "fitted_min": minimum(fitted), "fitted_rms": curve_rms(band, fitted),
            "rise_at_half": default(0.5), "excess_beyond_2": max(beyond) if beyond else None}


def curve_rms(curve: list, model) -> float:
    return math.sqrt(sum((model(x) - y) ** 2 for x, y in curve) / len(curve))


def minimum(model, lo: float = 0.6, hi: float = 2.0) -> tuple:
    xs = np.arange(lo, hi + 1e-9, 0.02)
    ys = [model(x) for x in xs]
    i = int(np.argmin(ys))
    return float(xs[i]), float(ys[i])


# --------------------------------------------------------------------------------------
# 3. Fluctuation
# --------------------------------------------------------------------------------------

def fluctuation_over_loss() -> dict:
    """rms thrust fluctuation over the mean thrust LOSS at the worst descent rate, both as
    fractions of hover thrust, for the Veismann 2023 rotors."""
    out = {}
    for name, rotor in VEISMANN_2023.items():
        t_min = min(y for _, y in rotor["curve"])
        out[name] = rotor["sigma_peak"] * t_min / (1 - t_min)
    return out


def descent_roll_disturbance(v: dict, sigma: float, mean_ratio: float,
                             rotor_speed: float | None = None) -> dict:
    """rms roll acceleration if the four rotors fluctuate independently with relative rms
    sigma about a mean thrust mean_ratio × static, and the descent speed of the peak."""
    n = len(v["ct2"])
    static = v["mass"] * v["grav"] / n if rotor_speed is None else v["ct2"][0] * rotor_speed ** 2
    a_y = abs(v["rotor_pos"][0][1])
    torque = a_y * math.sqrt(n) * sigma * mean_ratio * static
    return {"roll_accel_rms": torque / v["inertia"][0][0],
            "v_h": hover_induced_velocity(v, rotor_speed)}


# --------------------------------------------------------------------------------------
# 4. F-33: ground effect
# --------------------------------------------------------------------------------------

def ground_models(v: dict, z_over_r: float, s: float = 0.53, c_b: float = 1.80) -> dict:
    """Thrust gain T/T_∞ − 1 of each candidate at rotor height z = z_over_r·R."""
    r = v["r_prop"]
    z = z_over_r * r
    pos = v["rotor_pos"]
    d, b = math.dist(pos[0], pos[1]), math.dist(pos[0], pos[2])
    return {
        "cheeseman_bennett (constant power)": float(ground_effect.cheeseman_bennett(z, r)) - 1,
        "cheeseman_bennett_fixed_pitch": float(ground_effect.cheeseman_bennett_fixed_pitch(z, r, s)) - 1,
        "he_leang_exponential": float(ground_effect.he_leang_exponential(z, r, s, c_b)) - 1,
        "sanchez_cuevas (K_b = 2)": float(ground_effect.sanchez_cuevas(z, r, d, b, 2.0)) - 1,
        "pybullet (G = 11.37)": 11.36859 * (r / (4 * z)) ** 2,
    }


def pybullet_gain_implied(z_over_r: float, gain: float) -> float:
    """The G that the additive form G·(R/4z)² needs to give a measured gain."""
    return gain * (4 * z_over_r) ** 2


def talbot_decay_per_radius(v: dict, k_ge: float = 0.33) -> float:
    """k_ge·R: what JSBSim's dimensional decay rate (1/m, full-scale) becomes per rotor
    radius on a small vehicle — against the measured C_b of 1.2–2.9."""
    return k_ge * v["r_prop"]


# --------------------------------------------------------------------------------------
# 5. Time scales
# --------------------------------------------------------------------------------------

def time_scales(v: dict) -> dict:
    """Seconds, at hover. Dynamic inflow: Pitt–Peters uniform state, perturbation form
    τ = (2/(3π))·R/v_h (M11 = 8/(3π), L11 = 1/(4·v0); R.T.N. Chen, NASA TM-102219,
    Eqs. 17–21). VRS: Johnson's τ_rev = 0.7/λ_h revolutions, i.e. 0.7·2π·R/v_h."""
    v_h = hover_induced_velocity(v, rho=1.225)
    r = v["r_prop"]
    w_h = math.sqrt(v["mass"] * v["grav"] / (len(v["ct2"]) * v["ct2"][0]))
    tau_vrs = float(descent.johnson_inflow_time_constant(v_h / (w_h * r), w_h))
    return {"v_h": v_h, "revolution": 2 * math.pi / w_h,
            "dynamic_inflow": 2 / (3 * math.pi) * r / v_h, "vrs_lag": tau_vrs,
            "motor": v["tau_m"]}


# --------------------------------------------------------------------------------------
# 6. F-34: hub pitching moment; rear-rotor interference
# --------------------------------------------------------------------------------------

def hub_moment_measured(data: dict = RUSSELL_SOLO_ROTOR) -> list:
    """Per condition: advance ratio μ, M_y/(T·R) raw and with the hover offset removed, and
    the slope (M_y/(T·R))/μ — which is −k_flap/c_T in this repo's flapping_moment."""
    out = []
    tip = data["rpm"] * 2 * math.pi / 60 * data["radius_in"] / 12   # ft/s
    for row in data["edgewise"]:
        mu = row["speed_fts"] / tip
        raw = row["my"] / (row["fz"] * data["radius_in"])
        net = (row["my"] - data["hover_my"]) / (row["fz"] * data["radius_in"])
        out.append({"mu": mu, "raw": raw, "net": net, "slope_raw": raw / mu,
                    "slope_net": net / mu})
    return out


def hub_moment_on_vehicle(v: dict, slope: float, speed: float) -> dict:
    """Four rotors at hover thrust and speed (level, edgewise flow): pitch moment and its
    share of the pitch authority 2·a_x·T_max·(n/2)."""
    n = len(v["ct2"])
    w_h = math.sqrt(v["mass"] * v["grav"] / (n * v["ct2"][0]))
    mu = speed / (w_h * v["r_prop"])
    moment = n * slope * mu * (v["mass"] * v["grav"] / n) * v["r_prop"]
    a_x = abs(v["rotor_pos"][0][0])
    authority = n * a_x * v["ct2"][0] * v["limits"]["rotor_speed_max"] ** 2
    return {"mu": mu, "moment": moment, "k_flap": -slope * v["ct2"][0],
            "pitch_accel": moment / v["inertia"][1][1], "fraction_of_authority": moment / authority}


def rear_rotor_loss(v: dict, speed: float, pitch_down_deg: float, s: float) -> dict:
    """Rear-rotor thrust change at fixed speed from the front rotor's wake: k from
    spec.inflow.tandem_interference_factor, thrust from the linear law with the rear
    rotor's own induced velocity frozen (−s·k). Hover thrust, momentum inflow."""
    v_h = hover_induced_velocity(v, rho=1.225)
    a = math.radians(pitch_down_deg)
    v_axial, v_plane = speed * math.sin(a), speed * math.cos(a)
    v_i = v_h
    for _ in range(60):
        v_i = v_h ** 2 / math.sqrt(v_plane ** 2 + (v_axial + v_i) ** 2)
    gamma = math.atan2(v_axial + v_i, v_plane)
    spacing = abs(v["rotor_pos"][0][0] - v["rotor_pos"][2][0]) / v["r_prop"]
    k = float(inflow.tandem_interference_factor(spacing, a, gamma))
    return {"gamma_deg": math.degrees(gamma), "k": k, "thrust_change": -s * k * v_i / v_h}


def main() -> None:
    print("1. F-32 — thrust at constant rotor speed in vertical descent, T/T_hover")
    print("   x = descent/v_h    backend    BEM+quartic    measured 5x4 (Veismann 2023)")
    measured = dict(VEISMANN_2023["HQ 5x4"]["curve"])
    for x in (0.5, 1.0, 1.2, 1.5, 2.0):
        print(f"   {x:4.2f}               {backend_descent_ratio(RACER_5IN, x):5.2f}      "
              f"{bem_descent_ratio(x):5.2f}          {measured[x]:4.2f}")
    print(f"   quartic net inflow at x = 0.5 / 1.0 / 1.5: "
          f"{quartic_net_inflow(0.5):.2f} / {quartic_net_inflow(1.0):.2f} / "
          f"{quartic_net_inflow(1.5):.2f} (only falls); Johnson: "
          f"{-0.5 + johnson_inflow(-0.5):.2f} / {-1.0 + johnson_inflow(-1.0):.2f} / "
          f"{-1.5 + johnson_inflow(-1.5):.2f}")

    print("2. Johnson inflow + linear thrust law against the measured curves (s from geometry)")
    sets = {name: (r["curve"], blade_element_sensitivity(r)) for name, r in VEISMANN_2023.items()}
    sets["APC 10x5 (Shetty & Selig 2011)"] = (SHETTY_2011_10X5,
                                              blade_element_sensitivity(SHETTY_2011_ROTOR))
    for name, (curve, s_geo) in sets.items():
        c = descent_comparison(curve, s_geo)
        excess = "" if c["excess_beyond_2"] is None else f"; excess past x = 2: {c['excess_beyond_2']:+.2f}"
        print(f"   {name:31s} s {c['s']:.2f}; measured minimum {c['measured_min'][1]:.2f} at "
              f"{c['measured_min'][0]:.2f}; Table 4: {c['default_min'][1]:.2f} at "
              f"{c['default_min'][0]:.2f}, rms to x = 1.75 {c['default_rms']:.3f}; (V+v)_X = "
              f"{c['peak']:.2f}: minimum at {c['fitted_min'][0]:.2f}, rms {c['fitted_rms']:.3f}"
              f"{excess}")
    for s in (0.5, 1.0):
        x_min, y_min = minimum(lambda x, s=s: johnson_descent_ratio(x, s))
        print(f"   Table 4 defaults, s = {s}: minimum {y_min:.2f} at x = {x_min:.2f}; "
              f"at x = 0.5: {johnson_descent_ratio(0.5, s):.2f}")
    print("   forward speed closes the band (s = 1, (V+v)_X = 1.25): minimum T/T_static at "
          + ", ".join(f"V_x/v_h = {vx}: {minimum(lambda x, vx=vx: johnson_descent_ratio(x, 1.0, vx=vx))[1]:.2f}"
                      for vx in (0.0, 0.5, 0.95)))

    print("3. Thrust fluctuation in descent")
    for name, ratio in fluctuation_over_loss().items():
        print(f"   {name}: rms / mean loss at the worst descent rate = {ratio:.2f}")
    for label, speed in (("hover thrust", None), ("idle, 300 rad/s", 300.0),
                         ("full thrust", RACER_5IN["limits"]["rotor_speed_max"])):
        d = descent_roll_disturbance(RACER_5IN, 0.13, 0.78, speed)
        print(f"   RACER_5IN at {label}: v_h {d['v_h']:.1f} m/s, band 0.45–1.5 v_h = "
              f"{0.45 * d['v_h']:.1f}–{1.5 * d['v_h']:.1f} m/s, roll rms "
              f"{d['roll_accel_rms']:.1f} rad/s² (σ 0.13, independent rotors)")

    print("4. F-33 — ground effect at constant rotor speed, thrust gain")
    for z in (0.5, 1.0, 2.0, 4.0):
        m = ground_models(RACER_5IN, z)
        print(f"   z/R = {z}: " + ", ".join(f"{k} {100 * val:+.1f} %" for k, val in m.items()))
    for name, pts in GROUND_MEASURED.items():
        print(f"   measured, {name}: "
              + ", ".join(f"z/R {z:g}: {100 * gain:+.1f} %" for z, gain in pts))
    cf = GROUND_MEASURED["Crazyflie on a load cell (Carter et al., Fig. 2a)"]
    print(f"   G implied by the Crazyflie points: {pybullet_gain_implied(*cf[0]):.1f} at z/R "
          f"{cf[0][0]}, {pybullet_gain_implied(*cf[-1]):.1f} at {cf[-1][0]} (shipped: 11.37)")
    print(f"   Talbot/JSBSim k_ge = 0.33 /m is {talbot_decay_per_radius(RACER_5IN):.3f} per "
          f"radius on RACER_5IN; measured C_b 1.2–2.9 per radius")

    print("5. Time scales at hover")
    for name, v in (("RACER_5IN", RACER_5IN), ("CRAZYFLIE", CRAZYFLIE)):
        t = time_scales(v)
        print(f"   {name}: v_h {t['v_h']:.1f} m/s, revolution {1e3 * t['revolution']:.1f} ms, "
              f"dynamic inflow {1e3 * t['dynamic_inflow']:.1f} ms, VRS lag "
              f"{1e3 * t['vrs_lag']:.0f} ms, motor {1e3 * t['motor']:.0f} ms")

    print("6. F-34 — hub pitching moment (nose-up) of a rigid rotor in edgewise flow")
    for row in hub_moment_measured():
        print(f"   SOLO rotor, μ = {row['mu']:.3f}: M/(T·R) = {row['raw']:.3f} raw, "
              f"{row['net']:.3f} less the hover offset; slope {row['slope_raw']:.2f} / "
              f"{row['slope_net']:.2f}")
    for slope in (0.9, 1.4, 2.2):
        h = hub_moment_on_vehicle(RACER_5IN, slope, 10.0)
        print(f"   RACER_5IN at 10 m/s, slope {slope}: {h['moment']:.3f} N·m nose-up, "
              f"{h['pitch_accel']:.0f} rad/s², {100 * h['fraction_of_authority']:.1f} % of "
              f"pitch authority; k_flap = {h['k_flap']:.2e}")
    for speed, pitch in ((6.0, 5.0), (10.0, 8.0), (20.0, 30.0), (35.0, 60.0)):
        r = rear_rotor_loss(RACER_5IN, speed, pitch, 0.5)
        print(f"   rear rotors, {speed:.0f} m/s at {pitch:.0f}° nose-down: wake angle "
              f"{r['gamma_deg']:.0f}°, k = {r['k']:.2f}, thrust {100 * r['thrust_change']:+.1f} % "
              f"(s = 0.5)")


if __name__ == "__main__":
    main()
