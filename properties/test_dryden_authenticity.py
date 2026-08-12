"""
Authenticity tests for the Dryden turbulence filters against NASA CR-1998-206937
(Yeager, F18-HARV turbulence; pinned-document check data — the archaic-source exception).

The reference cannot be executed (ACSL, discontinued) and its noise stream cannot be
reproduced, so there are no sample-exact vectors. The evidence chain instead has two links:

  1. SYMBOLIC — the report's published difference equations (Eqs 18/20, the code's
     FILK1..FILK9 constants) are *exactly* the pole-prewarped bilinear discretization of
     this spec's forming filters under the documented noise conventions. This ties the
     reference implementation to spec.wind with zero numerical slack.
  2. STATISTICAL — re-simulating those difference equations with numpy noise under the
     report's exact protocol (10 runs x 1000 s @ 80 Hz, zero initial state, N-1 variance)
     reproduces the published Tables 2-7 ensemble statistics within the published
     run-to-run spread. The table numbers came from the authors running their actual
     code, so this validates the CALIBRATION (output std == sigma — what the driving-
     noise-scaling trap corrupts; a wrongly-scaled implementation lands ~66 standard
     errors away).

Noise-convention bridge (why sqrt(pi/T_v) appears): spec filters are stated for
unit-two-sided-PSD continuous white noise, realized discretely as N(0, pi/dt) samples;
the report drives with N(0,1) samples at T_v and absorbs the conversion into its gains —
sigma*sqrt(2L/(pi*V)) * sqrt(pi/T_v) == sigma*sqrt(2*tau/T_v), the report's gain.

Check data: golden/checkdata/dryden_cr206937_tables.json (see golden/generate/
gen_dryden_cr206937.py for provenance, the pinned PDF sha256, and the transcription
audit trail). u,v,w only — the spec carries no p,q,r gust terms (skipped per ledger).
"""

import json
import pathlib

import numpy as np
import sympy as sp

from spec import wind

DATA = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / "golden" / "checkdata" /
     "dryden_cr206937_tables.json").read_text())

SIGMA = DATA["constants"]["sigma_u_v_w_ftps"]
L_FT = DATA["constants"]["L_u_v_w_ft"]
T_V = DATA["constants"]["T_v_sample_s"]
N_RUNS = DATA["constants"]["n_runs"]
N_SAMPLES = DATA["constants"]["n_samples_per_run"]


# ---------------- symbolic: report difference equations == discretized spec filters ----

def test_cr206937_tustin_u_is_prewarped_bilinear_of_spec_filter():
    # Report Eq 18 with Eq 19's C_BL is the bilinear transform s -> C*(1-z⁻¹)/(1+z⁻¹) of
    # spec.wind.dryden_filter_u (prewarped so the pole 1/tau maps exactly), with the
    # N(0, pi/dt) -> N(0,1) driving-noise conversion folded into the gain.
    sigma, L, V, Tv, C = sp.symbols("sigma L V T_v C", positive=True)
    zi = sp.Symbol("zeta")  # z^{-1}
    tau = L / V
    H_disc = wind.dryden_filter_u(sigma, L, V).subs(wind.s, C * (1 - zi) / (1 + zi)) \
        * sp.sqrt(sp.pi / Tv)
    gain = sigma * sp.sqrt(2 * tau / Tv) / (1 + C * tau)
    H_18 = gain * (1 + zi) / (1 + (1 - C * tau) / (1 + C * tau) * zi)
    assert sp.simplify(H_disc - H_18) == 0


def test_cr206937_tustin_vw_is_prewarped_bilinear_of_spec_filter():
    # Report Eq 20 (code constants FILK4..FILK9, omega = 1/tau) against
    # spec.wind.dryden_filter_vw under the same substitution and noise conversion.
    sigma, L, V, Tv, C = sp.symbols("sigma L V T_v C", positive=True)
    zi = sp.Symbol("zeta")
    tau = L / V
    w = 1 / tau
    H_disc = wind.dryden_filter_vw(sigma, L, V).subs(wind.s, C * (1 - zi) / (1 + zi)) \
        * sp.sqrt(sp.pi / Tv)
    A1 = 2 * (w**2 - C**2) / (w + C) ** 2
    A2 = (w - C) ** 2 / (w + C) ** 2
    num = (C + w / sp.sqrt(3)) + (2 * w / sp.sqrt(3)) * zi + (w / sp.sqrt(3) - C) * zi**2
    H_20 = sigma * sp.sqrt(3 * w / Tv) / (w + C) ** 2 * num / (1 + A1 * zi + A2 * zi**2)
    assert sp.simplify(H_disc - H_20) == 0


def test_cr206937_prewarp_coincides_with_plain_tustin_here():
    # Eq 19: C_BL = (1/tau)*cot(T_v/(2*tau)). At the report's parameters the prewarped
    # constant differs from the plain-Tustin 2/T_v by < 1e-5 relative, so the ledger's
    # plain-Tustin discretization candidate and the report's variant coincide numerically.
    for V in DATA["constants"]["V_cases_ftps"]:
        tau = L_FT / V
        c_bl = (1.0 / tau) / np.tan(T_V / (2.0 * tau))
        assert abs(c_bl - 2.0 / T_V) / (2.0 / T_V) < 1e-5


# ---------------- statistical: reproduce the published run statistics ----------------

def _simulate(model: str, V: float, rng) -> dict:
    """Run the report's protocol for one model at one airspeed.

    Implements the difference equations exactly as printed (Eqs 18/20 for 'tustin',
    Eqs 30-32 for 'mil_std'): zero state at T = 0, first output uses the first noise
    sample, N = 80,001 samples at T_v. Returns per-run sample stats per component.
    """
    tau = L_FT / V
    w = 1.0 / tau
    noise = rng.standard_normal((3, N_RUNS, N_SAMPLES))  # u, v, w streams
    out = np.empty_like(noise)

    if model == "tustin":
        c = w / np.tan(w * T_V / 2.0)                      # Eq 19 (CFIL)
        a1_u = (w - c) / (w + c)                           # FILK1
        g_u = SIGMA * np.sqrt(2.0 * tau / T_V) / (1.0 + c * tau)
        A1 = 2.0 * (w**2 - c**2) / (w + c) ** 2            # FILK4
        A2 = (w - c) ** 2 / (w + c) ** 2                   # FILK5
        g_vw = SIGMA * np.sqrt(3.0 * w / T_V) / (w + c) ** 2   # TWOPIOVTNU*FILK9
        c0, c1, c2 = c + w / np.sqrt(3.0), 2.0 * w / np.sqrt(3.0), w / np.sqrt(3.0) - c
        # u: first order with (v(k) + v(k-1)) input
        x = np.zeros(N_RUNS)
        vm1 = np.zeros(N_RUNS)
        for k in range(N_SAMPLES):
            vk = noise[0, :, k]
            x = -a1_u * x + g_u * (vk + vm1)
            vm1 = vk
            out[0, :, k] = x
        # v, w: second order with 3-tap input
        for i in (1, 2):
            xm1 = np.zeros(N_RUNS)
            xm2 = np.zeros(N_RUNS)
            vm1 = np.zeros(N_RUNS)
            vm2 = np.zeros(N_RUNS)
            for k in range(N_SAMPLES):
                vk = noise[i, :, k]
                x = -A1 * xm1 - A2 * xm2 + g_vw * (c0 * vk + c1 * vm1 + c2 * vm2)
                xm2, xm1 = xm1, x
                vm2, vm1 = vm1, vk
                out[i, :, k] = x
    elif model == "mil_std":
        params = [(1.0 - T_V / tau, SIGMA * np.sqrt(2.0 * T_V / tau)),        # Eq 30
                  (1.0 - 2.0 * T_V / tau, SIGMA * np.sqrt(4.0 * T_V / tau)),  # Eq 31
                  (1.0 - 2.0 * T_V / tau, SIGMA * np.sqrt(4.0 * T_V / tau))]  # Eq 32
        for i, (a, b) in enumerate(params):
            x = np.zeros(N_RUNS)
            for k in range(N_SAMPLES):
                x = a * x + b * noise[i, :, k]
                out[i, :, k] = x
    else:
        raise ValueError(model)

    stats = {}
    for i, comp in enumerate("uvw"):
        stds = out[i].std(axis=1, ddof=1)   # DRMS uses the N-1 denominator
        means = out[i].mean(axis=1)
        stats[comp] = {"mean_of_std": stds.mean(), "std_of_std": stds.std(ddof=1),
                       "mean_of_mean": means.mean(), "std_of_mean": means.std(ddof=1)}
    return stats


def _table_for(model: str, V: float) -> dict:
    for tbl in DATA["tables"].values():
        if tbl["model"] == model and tbl["V_ftps"] == V:
            return tbl["ensemble"]
    raise KeyError((model, V))


def test_cr206937_statistics_reproduced():
    rng = np.random.default_rng(20260811)
    for model in ("tustin", "mil_std"):
        for V in DATA["constants"]["V_cases_ftps"]:
            ours = _simulate(model, V, rng)
            ref = _table_for(model, V)
            for comp in "uvw":
                m_ref = ref["mean_of_std"][comp]
                s_ref = ref["std_of_std"][comp]
                ctx = f"{model} V={V} {comp}"
                # calibration: both the published and our mean-of-stds sit on sigma
                # within the 10-run standard error (a sqrt(2) convention slip is ~14 SE
                # away; the driving-noise-scaling bug is ~66 SE away)
                se = s_ref / np.sqrt(N_RUNS)
                assert abs(m_ref - SIGMA) < 4 * se, ctx
                assert abs(ours[comp]["mean_of_std"] - SIGMA) < 4 * se, \
                    (ctx, ours[comp]["mean_of_std"])
                # agreement with the published ensemble (two independent 10-run means)
                assert abs(ours[comp]["mean_of_std"] - m_ref) < 4 * np.sqrt(2) * se, \
                    (ctx, ours[comp]["mean_of_std"], m_ref)
                # spreads: chi-square with 9 dof on both sides — wide bands
                assert 1 / 3 < ours[comp]["std_of_std"] / s_ref < 3, ctx
                assert 1 / 3 < ours[comp]["std_of_mean"] / ref["std_of_mean"][comp] < 3, ctx
                # run-means center on zero at the published spread
                assert abs(ours[comp]["mean_of_mean"]) < 2 * ref["std_of_mean"][comp], ctx
                assert abs(ref["mean_of_mean"][comp]) < 2 * ref["std_of_mean"][comp], ctx


def test_cr206937_theoretical_sampling_spread_formulas():
    # Report Eqs 62/63 (std of the 1000-s sample mean) and 65/66 (std of the sample std)
    # are closed-form predictions of the tables' ensemble spread rows, derived from the
    # Dryden autocorrelations. 10-run estimates of a spread are chi-square-noisy
    # (relative std ~33%), and 65/66 are stated as approximations — hence the bands.
    T = DATA["constants"]["run_length_s"]
    for V in DATA["constants"]["V_cases_ftps"]:
        tau = L_FT / V
        eq62_u = SIGMA * np.sqrt(2 * (tau / T - tau**2 / T**2))
        eq63_vw = SIGMA * np.sqrt(tau / T)
        eq65_u = SIGMA * np.sqrt(tau / (2 * T) - 9 * tau**2 / (4 * T**2)
                                 + 4 * tau**3 / T**3 - 2 * tau**4 / T**4)
        eq66_vw = SIGMA / 2 * np.sqrt(13 * tau / (4 * T) - 83 * tau**2 / (8 * T**2))
        for model in ("continuous", "tustin", "mil_std"):
            ref = _table_for(model, V)
            assert 1 / 2 < ref["std_of_mean"]["u"] / eq62_u < 2, (model, V)
            for comp in "vw":
                assert 1 / 2 < ref["std_of_mean"][comp] / eq63_vw < 2, (model, V, comp)
            assert 1 / 2.5 < ref["std_of_std"]["u"] / eq65_u < 2.5, (model, V)
            for comp in "vw":
                assert 1 / 2.5 < ref["std_of_std"][comp] / eq66_vw < 2.5, (model, V, comp)
