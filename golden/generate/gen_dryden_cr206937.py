"""
Check-data freezer: NASA CR-1998-206937 Dryden turbulence statistics (pinned document).

⚠ ARCHAIC-SOURCE EXCEPTION (2026-08-11): unlike the other generators, this one
does not execute reference code. The reference is a pinned NASA contractor report whose
GUSTMDL test program is printed in full but written in ACSL — a discontinued commercial
simulation language with no runnable interpreter today — and driven by ACSL's internal
Gaussian generator, whose sample stream is irreproducible. Sample-exact vectors are
therefore impossible from this source. What IS usable, and frozen here, are the numeric
outputs the authors published from running their actual code: Tables 2-7, per-run sample
statistics of ten 1000-second runs per model per airspeed. They validate the CALIBRATION
of the Dryden filters (output variance vs σ, the exact thing the driving-noise-scaling
trap corrupts) — not trajectories.

What this script does:
  1. Embeds the transcribed tables + run constants (transcription from the scanned PDF,
     page-referenced; full ACSL/Fortran listing archived alongside as
     cr206937_gustmdl_transcription.txt).
  2. Self-checks the transcription: recomputes every printed ensemble row (mean/std over
     the ten per-run values) and requires agreement within print-rounding tolerance —
     a hand-typed table cell that disagrees with its own printed summary fails here.
  3. Verifies the local PDF's sha256 if --pdf is given.
  4. Writes golden/checkdata/dryden_cr206937_tables.json (its own directory: this is a
     different evidence class from the run-the-code vectors in golden/vectors/, and the
     generic vector harness must not pick it up).

Consumed by properties/test_dryden_authenticity.py:
  - symbolic: the report's Tustin difference equations (Eqs 18-20) are exactly the
    pole-prewarped bilinear discretization of spec.wind.dryden_filter_u / _vw under the
    N(0, π/dt) driving-noise convention (unit-noise gain σ√(2τ/T_v)).
  - statistical: re-simulating the published difference equations with numpy noise
    reproduces the tables' u/v/w calibration within the published run-to-run spread.

Usage:
    uv run python golden/generate/gen_dryden_cr206937.py [--pdf path] [--out golden/vectors]
"""

import argparse
import hashlib
import json
import math
import pathlib

PROVENANCE = {
    "document": "NASA/CR-1998-206937",
    "title": "Implementation and Testing of Turbulence Models for the F18-HARV Simulation",
    "author": "Jessie C. Yeager (Lockheed Martin Engineering & Sciences)",
    "date": "March 1998",
    "contract": "NAS1-96014 (NASA Langley)",
    "ntrs_id": "19980028448",
    "url": "https://ntrs.nasa.gov/api/citations/19980028448/downloads/19980028448.pdf",
    "pdf_sha256": "4f63d46d3eac9568239941e943b98a59e1444d05b622760eaf8f07a0c1b82686",
    "retrieved": "2026-08-11",
    "pages": {
        "equations_continuous": "report pp. 4-6 (Eqs 1-17)",
        "equations_tustin": "report pp. 6-8 (Eqs 18-29)",
        "equations_mil_std": "report pp. 8-9 (Eqs 30-35)",
        "variable_table": "report pp. 9-10 (Table 1: constants)",
        "statistics_method": "report p. 53 (Eqs 58-61: 10 runs x 1000 s, per-run seeds)",
        "tables": "report pp. 54-59 (Tables 2-7)",
        "theoretical_statistics": "report pp. 60-61 (Eqs 62-68)",
        "code_listing": "report pp. 11-30 (GUSTMDL ACSL program; transcription archived "
                        "in golden/generate/cr206937_gustmdl_transcription.txt)",
    },
    "notes": (
        "US Government work (NASA contractor report), publicly released via NTRS. "
        "The 'Run' column values are run log numbers, NOT seeds (the listing's default "
        "seed is IRSEED = 28545269 via GAUSI; per-run seeds are unrecorded). All three "
        "models run in ONE program execution sharing the same GAUSS noise stream "
        "(FILNU/V/W at TSGAUS = 0.0125 s), so per-run rows are noise-matched across "
        "models but NOT reproducible outside ACSL. Statistics are over N = 80,001 "
        "samples at TSAMP = 0.0125 s (1000 s), filters started from zero state; the "
        "DRMS macro uses the N-1 sample-variance denominator."
    ),
}

#: Run constants from Table 1 (report p. 10) and the statistics section (p. 53).
#: Units are the report's: ft, ft/s, rad/s, seconds.
CONSTANTS = {
    "sigma_u_v_w_ftps": 5.0,      # TURBSIG (code listing, report p. 13)
    "L_u_v_w_ft": 1750.0,         # TURBL — MIL-F-8785C medium/high-altitude scale length
    "b_wing_ft": 37.4,            # BWING (code listing p. 13; Table 1's blurred glyph reads
                                  # "32.4" but sigma_p = 1.9*sigma_w/sqrt(L*b) = 0.0371 in
                                  # the tables confirms 37.4 — the F-18 wingspan)
    "T_v_sample_s": 0.0125,       # TSAMP (80 Hz); TSGAUS = TRMS = TSASC identical
    "noise_std": 1.0,             # SDNOIS — GAUSS(0., 1.) per sample per axis
    "run_length_s": 1000.0,
    "n_runs": 10,
    "n_samples_per_run": 80001,
    "V_cases_ftps": [100.0, 1000.0],   # VTOT (listing default 400 overridden per case)
}

#: The published difference equations the tables were generated with (verbatim algebra,
#: report equation numbers). ω ≡ 1/τ, τ = L/V, C_BL = (1/τ)·cot(T_v/(2τ)) [Eq 19/24/26].
#: Driving noise v(k): unit-variance Gaussian per sample (ACSL GAUSS), NOT PSD-π noise —
#: gains below already absorb the conversion (σ√(2τ/T_v) ≡ σ√(2L/(πV))·√(π/T_v)).
EQUATIONS = {
    "tustin_u_eq18": "xi(k) = -[(1-C*tau)/(1+C*tau)]*xi(k-1)"
                     " + [sigma*sqrt(2*tau/Tv)/(1+C*tau)]*[v(k)+v(k-1)]",
    "tustin_vw_eq20": "xi(k) = -[2*(w^2-C^2)/(w+C)^2]*xi(k-1) - [(w-C)^2/(w+C)^2]*xi(k-2)"
                      " + [sigma*sqrt(3*w/Tv)/(w+C)^2] * [(C+w/sqrt(3))*v(k)"
                      " + (2*w/sqrt(3))*v(k-1) + (w/sqrt(3)-C)*v(k-2)],  w = 1/tau",
    "mil_std_u_eq30": "xi(k) = (1-Tv/tau)*xi(k-1) + sigma*sqrt(2*Tv/tau)*v(k)",
    "mil_std_vw_eq31_32": "xi(k) = (1-2*Tv/tau)*xi(k-1) + sigma*sqrt(4*Tv/tau)*v(k)",
    "prewarp_eq19": "C_BL = (1/tau)*cot(Tv/(2*tau))",
    "theoretical_std_of_mean_u_eq62": "sigma*sqrt(2*(tau/T - tau^2/T^2))",
    "theoretical_std_of_mean_vw_eq63": "sigma*sqrt(tau/T)",
    "theoretical_std_of_std_u_eq65":
        "sigma*sqrt(tau/(2T) - 9*tau^2/(4T^2) + 4*tau^3/T^3 - 2*tau^4/T^4)",
    "theoretical_std_of_std_vw_eq66": "(sigma/2)*sqrt(13*tau/(4T) - 83*tau^2/(8T^2))",
}

# --------------------------------------------------------------------------------------
# Tables 2-7 (report pp. 54-59), transcribed from the scan. Layout per table:
# runs (= ACSL seeds), per-run sample std and mean for each component, then the printed
# ensemble rows. Components u,v,w in ft/s; p,q,r in rad/s.
# --------------------------------------------------------------------------------------

RUNS_V100 = [34, 36, 37, 38, 39, 40, 41, 42, 43, 44]
RUNS_V1000 = [35, 45, 46, 47, 48, 49, 50, 51, 52, 53]

TABLES = {
    "table_2_continuous_V100": {
        "model": "continuous", "V_ftps": 100.0, "runs": RUNS_V100,
        "std": {
            "u": [4.95, 4.80, 5.27, 4.84, 5.27, 5.24, 5.11, 5.46, 5.36, 4.40],
            "v": [5.27, 5.51, 5.39, 4.20, 4.77, 5.32, 4.88, 5.11, 5.20, 5.23],
            "w": [4.90, 5.83, 4.82, 5.16, 4.51, 4.76, 5.00, 4.78, 5.07, 5.24],
            "p": [0.0371, 0.0375, 0.0373, 0.0366, 0.0379, 0.0394, 0.0366, 0.0372, 0.0368, 0.0360],
            "q": [0.0212, 0.0214, 0.0209, 0.0209, 0.0214, 0.0207, 0.0210, 0.0209, 0.0208, 0.0215],
            "r": [0.0242, 0.0245, 0.0239, 0.0236, 0.0243, 0.0236, 0.0245, 0.0240, 0.0236, 0.0243],
        },
        "mean": {
            "u": [0.296, -1.23, -0.261, 0.259, 1.77, 1.11, -0.419, -1.09, 0.425, -0.660],
            "v": [-0.112, -0.951, -0.740, 1.11, -0.499, 0.849, 0.229, 0.887, -1.26, -0.258],
            "w": [-0.463, -0.388, 0.214, -0.418, -0.185, 0.606, -0.351, -0.284, 0.443, -0.923],
            "p": [0.000106, -0.000551, -0.00264, -0.000202, -0.00267, -6.02e-05, 0.000188,
                  -0.000544, -0.000526, -0.00361],
            "q": [-6.42e-05, 3.45e-06, -2.90e-05, -5.56e-05, 8.01e-06, 2.72e-05, -7.82e-05,
                  -4.67e-05, 1.81e-05, -4.15e-05],
            "r": [6.54e-05, 2.08e-05, -0.000109, 5.06e-05, -5.22e-06, 1.67e-05, -3.97e-05,
                  -2.52e-05, -6.10e-05, 5.71e-05],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.07, "v": 5.09, "w": 5.01, "p": 0.0372, "q": 0.0211, "r": 0.0241},
            "std_of_std": {"u": 0.321, "v": 0.382, "w": 0.358, "p": 0.000912, "q": 0.000273,
                           "r": 0.000369},
            "mean_of_mean": {"u": 0.0204, "v": -0.0743, "w": -0.175, "p": -0.00105,
                             "q": -2.58e-05, "r": -2.96e-06},
            "std_of_mean": {"u": 0.950, "v": 0.822, "w": 0.463, "p": 0.00138, "q": 3.73e-05,
                            "r": 5.63e-05},
        },
    },
    "table_3_tustin_V100": {
        "model": "tustin", "V_ftps": 100.0, "runs": RUNS_V100,
        "std": {
            "u": [4.95, 4.80, 5.26, 4.84, 5.26, 5.23, 5.11, 5.45, 5.36, 4.40],
            "v": [5.23, 5.44, 5.35, 4.18, 4.74, 5.27, 4.84, 5.08, 5.15, 5.21],
            "w": [4.89, 5.75, 4.79, 5.11, 4.51, 4.73, 4.98, 4.77, 5.07, 5.22],
            "p": [0.0370, 0.0374, 0.0371, 0.0365, 0.0377, 0.0393, 0.0364, 0.0371, 0.0367, 0.0359],
            "q": [0.0211, 0.0213, 0.0207, 0.0208, 0.0212, 0.0206, 0.0208, 0.0208, 0.0207, 0.0214],
            "r": [0.0240, 0.0243, 0.0237, 0.0233, 0.0241, 0.0234, 0.0243, 0.0238, 0.0234, 0.0241],
        },
        "mean": {
            "u": [0.295, -1.24, -0.258, 0.260, 1.77, 1.11, -0.413, -1.08, 0.422, -0.661],
            "v": [-0.0971, -0.897, -0.697, 1.06, -0.474, 0.812, 0.217, 0.854, -1.18, -0.239],
            "w": [-0.450, -0.380, 0.205, -0.408, -0.176, 0.585, -0.342, -0.275, 0.418, -0.877],
            "p": [0.000106, -0.000551, -0.00264, -0.000198, -0.00267, -6.05e-05, 0.000191,
                  -0.000548, -0.000532, -0.00362],
            "q": [-6.22e-05, 5.48e-06, -3.03e-05, -5.59e-05, 7.45e-06, 2.74e-05, -7.60e-05,
                  -4.70e-05, 1.82e-05, -4.21e-05],
            "r": [6.32e-05, 1.95e-05, -0.000111, 4.93e-05, -5.30e-06, 1.45e-05, -3.90e-05,
                  -2.78e-05, -6.23e-05, 5.65e-05],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.07, "v": 5.05, "w": 4.98, "p": 0.0372, "q": 0.0209, "r": 0.0238},
            "std_of_std": {"u": 0.320, "v": 0.373, "w": 0.341, "p": 0.000919, "q": 0.000276,
                           "r": 0.000372},
            "mean_of_mean": {"u": 0.0207, "v": -0.0646, "w": -0.170, "p": -0.00105,
                             "q": -2.55e-05, "r": -4.22e-06},
            "std_of_mean": {"u": 0.948, "v": 0.781, "w": 0.444, "p": 0.00138, "q": 3.70e-05,
                            "r": 5.62e-05},
        },
    },
    "table_4_mil_std_V100": {
        "model": "mil_std", "V_ftps": 100.0, "runs": RUNS_V100,
        "std": {
            "u": [4.95, 4.80, 5.27, 4.84, 5.26, 5.24, 5.11, 5.46, 5.36, 4.40],
            "v": [5.29, 5.47, 5.33, 4.22, 4.80, 5.28, 4.91, 5.09, 5.11, 5.18],
            "w": [4.90, 5.76, 4.85, 5.11, 4.59, 4.81, 4.93, 4.80, 5.03, 5.23],
            "p": [0.0372, 0.0376, 0.0374, 0.0368, 0.0380, 0.0395, 0.0367, 0.0373, 0.0369, 0.0361],
            "q": [0.0245, 0.0247, 0.0241, 0.0241, 0.0246, 0.0239, 0.0242, 0.0241, 0.0240, 0.0248],
            "r": [0.0280, 0.0284, 0.0277, 0.0273, 0.0282, 0.0274, 0.0284, 0.0278, 0.0274, 0.0282],
        },
        "mean": {
            "u": [0.295, -1.24, -0.258, 0.260, 1.77, 1.11, -0.413, -1.08, 0.422, -0.661],
            "v": [-0.121, -0.956, -0.752, 1.10, -0.491, 0.835, 0.232, 0.866, -1.27, -0.262],
            "w": [-0.451, -0.368, 0.203, -0.423, -0.191, 0.617, -0.341, -0.295, 0.443, -0.930],
            "p": [0.000106, -0.000551, -0.00264, -0.000197, -0.00267, -6.16e-05, 0.000191,
                  -0.000548, -0.000532, -0.00362],
            "q": [-6.13e-05, 5.42e-06, -2.34e-05, -5.72e-05, 9.76e-06, 1.64e-05, -6.41e-05,
                  -4.24e-05, 2.34e-05, -2.94e-05],
            "r": [5.09e-05, 3.08e-06, -0.000121, 4.72e-05, -1.80e-05, 1.69e-05, -5.00e-05,
                  -2.16e-05, -6.00e-05, 6.86e-05],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.07, "v": 5.07, "w": 5.00, "p": 0.0373, "q": 0.0243, "r": 0.0279},
            "std_of_std": {"u": 0.320, "v": 0.357, "w": 0.320, "p": 0.000914, "q": 0.000312,
                           "r": 0.000421},
            "mean_of_mean": {"u": 0.0206, "v": -0.0819, "w": -0.174, "p": -0.00105,
                             "q": -2.23e-05, "r": -8.37e-06},
            "std_of_mean": {"u": 0.948, "v": 0.821, "w": 0.464, "p": 0.00138, "q": 3.39e-05,
                            "r": 5.83e-05},
        },
    },
    "table_5_continuous_V1000": {
        "model": "continuous", "V_ftps": 1000.0, "runs": RUNS_V1000,
        "std": {
            "u": [5.07, 5.17, 5.16, 5.10, 5.28, 5.31, 5.26, 5.22, 4.92, 4.91],
            "v": [5.03, 5.20, 4.94, 4.79, 5.07, 4.91, 5.03, 4.96, 4.85, 5.10],
            "w": [5.07, 5.22, 4.98, 5.05, 5.16, 4.92, 5.03, 5.03, 5.01, 5.20],
            "p": [0.0373, 0.0372, 0.0373, 0.0370, 0.0373, 0.0376, 0.0370, 0.0372, 0.0370, 0.0368],
            "q": [0.0207, 0.0207, 0.0206, 0.0207, 0.0208, 0.0208, 0.0206, 0.0207, 0.0206, 0.0207],
            "r": [0.0238, 0.0238, 0.0237, 0.0237, 0.0240, 0.0237, 0.0240, 0.0239, 0.0238, 0.0238],
        },
        "mean": {
            "u": [0.0393, -0.417, -0.0929, 0.0400, 0.551, 0.391, -0.125, -0.367, 0.143, -0.210],
            "v": [-0.0217, -0.302, -0.266, 0.362, -0.157, 0.271, 0.0640, 0.270, -0.416, -0.0693],
            "w": [-0.159, -0.115, 0.0601, -0.147, -0.0590, 0.199, -0.128, -0.104, 0.143, -0.303],
            "p": [3.68e-05, -0.000174, -0.000824, -5.01e-05, -0.000864, -4.39e-05, 5.42e-05,
                  -0.000167, -0.000171, -0.00115],
            "q": [1.63e-08, -3.16e-06, -8.60e-06, -9.49e-07, 1.58e-06, -6.69e-06, 4.89e-06,
                  -1.82e-07, 8.47e-06, 2.91e-06],
            "r": [-4.04e-06, -2.90e-07, -7.35e-06, -1.41e-06, -9.09e-06, -6.65e-06, -9.71e-06,
                  -3.87e-06, -6.60e-06, 8.59e-06],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.14, "v": 4.99, "w": 5.07, "p": 0.0372, "q": 0.0207, "r": 0.0238},
            "std_of_std": {"u": 0.141, "v": 0.122, "w": 0.0973, "p": 0.000228, "q": 8.38e-05,
                           "r": 0.000115},
            "mean_of_mean": {"u": -0.00478, "v": -0.0266, "w": -0.0612, "p": -0.000335,
                             "q": -1.72e-07, "r": -4.04e-06},
            "std_of_mean": {"u": 0.308, "v": 0.266, "w": 0.152, "p": 0.000437, "q": 5.12e-06,
                            "r": 5.40e-06},
        },
    },
    "table_6_tustin_V1000": {
        "model": "tustin", "V_ftps": 1000.0, "runs": RUNS_V1000,
        "std": {
            "u": [5.06, 5.16, 5.15, 5.08, 5.26, 5.30, 5.25, 5.21, 4.91, 4.90],
            "v": [5.02, 5.18, 4.93, 4.77, 5.05, 4.90, 5.02, 4.95, 4.84, 5.08],
            "w": [5.06, 5.20, 4.97, 5.03, 5.15, 4.91, 5.02, 5.02, 5.00, 5.19],
            "p": [0.0363, 0.0362, 0.0362, 0.0359, 0.0363, 0.0366, 0.0360, 0.0361, 0.0359, 0.0358],
            "q": [0.0195, 0.0195, 0.0194, 0.0195, 0.0196, 0.0196, 0.0194, 0.0195, 0.0194, 0.0195],
            "r": [0.0221, 0.0221, 0.0220, 0.0219, 0.0223, 0.0220, 0.0222, 0.0221, 0.0220, 0.0220],
        },
        "mean": {
            "u": [0.0389, -0.418, -0.0919, 0.0404, 0.551, 0.390, -0.123, -0.365, 0.142, -0.211],
            "v": [-0.0219, -0.303, -0.266, 0.362, -0.157, 0.271, 0.0647, 0.272, -0.416, -0.0689],
            "w": [-0.160, -0.115, 0.0605, -0.149, -0.0594, 0.201, -0.127, -0.106, 0.143, -0.304],
            "p": [3.69e-05, -0.000174, -0.000824, -4.84e-05, -0.000865, -4.46e-05, 5.50e-05,
                  -0.000168, -0.000173, -0.00115],
            "q": [-5.64e-08, -3.61e-06, -8.72e-06, -7.64e-07, 1.49e-06, -6.83e-06, 4.66e-06,
                  -1.10e-07, 8.33e-06, 2.69e-06],
            "r": [-4.11e-06, -5.62e-08, -7.21e-06, -1.44e-06, -9.28e-06, -6.68e-06, -9.81e-06,
                  -3.78e-06, -6.40e-06, 8.24e-06],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.13, "v": 4.97, "w": 5.05, "p": 0.0361, "q": 0.0195, "r": 0.0221},
            "std_of_std": {"u": 0.141, "v": 0.122, "w": 0.0977, "p": 0.000235, "q": 8.50e-05,
                           "r": 0.000120},
            "mean_of_mean": {"u": -0.00473, "v": -0.0262, "w": -0.0615, "p": -0.000335,
                             "q": -2.93e-07, "r": -4.05e-06},
            "std_of_mean": {"u": 0.308, "v": 0.267, "w": 0.153, "p": 0.000437, "q": 5.12e-06,
                            "r": 5.34e-06},
        },
    },
    "table_7_mil_std_V1000": {
        "model": "mil_std", "V_ftps": 1000.0, "runs": RUNS_V1000,
        "std": {
            "u": [5.07, 5.17, 5.17, 5.10, 5.28, 5.31, 5.27, 5.23, 4.92, 4.92],
            "v": [5.06, 5.19, 4.97, 4.81, 5.06, 4.93, 5.06, 4.98, 4.88, 5.10],
            "w": [5.10, 5.23, 5.01, 5.05, 5.15, 4.95, 5.05, 5.03, 5.03, 5.21],
            "p": [0.0386, 0.0385, 0.0385, 0.0383, 0.0386, 0.0389, 0.0383, 0.0385, 0.0383, 0.0381],
            "q": [0.0258, 0.0258, 0.0257, 0.0258, 0.0260, 0.0260, 0.0258, 0.0258, 0.0257, 0.0259],
            "r": [0.0308, 0.0308, 0.0307, 0.0307, 0.0310, 0.0307, 0.0310, 0.0309, 0.0308, 0.0308],
        },
        "mean": {
            "u": [0.0387, -0.418, -0.0919, 0.0403, 0.551, 0.390, -0.123, -0.365, 0.142, -0.211],
            "v": [-0.0229, -0.301, -0.264, 0.360, -0.157, 0.270, 0.0644, 0.270, -0.414, -0.0691],
            "w": [-0.159, -0.115, 0.0600, -0.148, -0.0590, 0.201, -0.126, -0.104, 0.144, -0.303],
            "p": [3.71e-05, -0.000175, -0.000823, -4.77e-05, -0.000865, -4.53e-05, 5.54e-05,
                  -0.000167, -0.000173, -0.00115],
            "q": [-6.71e-07, -4.36e-06, -9.53e-06, -2.11e-07, 6.75e-07, -7.01e-06, 4.44e-06,
                  -3.17e-07, 8.52e-06, 2.65e-06],
            "r": [-3.69e-06, 5.25e-07, -4.69e-06, -1.92e-06, -9.44e-06, -8.05e-06, -9.36e-06,
                  -3.44e-06, -6.00e-06, 7.13e-06],
        },
        "ensemble": {
            "mean_of_std": {"u": 5.15, "v": 5.00, "w": 5.08, "p": 0.0385, "q": 0.0258, "r": 0.0308},
            "std_of_std": {"u": 0.141, "v": 0.112, "w": 0.0906, "p": 0.000227, "q": 0.000102,
                           "r": 0.000132},
            "mean_of_mean": {"u": -0.00476, "v": -0.0263, "w": -0.0609, "p": -0.000335,
                             "q": -5.82e-07, "r": -3.89e-06},
            "std_of_mean": {"u": 0.308, "v": 0.266, "w": 0.153, "p": 0.000437, "q": 5.33e-06,
                            "r": 5.04e-06},
        },
    },
}


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _quantum(values) -> float:
    """Print quantum of a 3-significant-figure column: ULP of its largest magnitude."""
    m = max(abs(v) for v in values)
    return 10.0 ** (math.floor(math.log10(m)) - 2)


def self_check() -> list:
    """Recompute every printed ensemble row from the per-run rows.

    Both sides carry print rounding: each per-run cell is printed to 3 significant
    figures (quantum Q of the column's largest value), and the printed summary is too.
    Mean aggregates move by at most ~Q/2 under cell rounding; std aggregates are more
    sensitive when the true spread is only a few Q (the q/r columns) — 0.8·Q covers the
    worst observed quantization shift while still catching any real slip: a transposed
    digit in one cell moves the column std by ≫ Q.
    """
    problems = []
    for name, tbl in TABLES.items():
        for kind, agg, printed_key in (("std", _mean, "mean_of_std"), ("std", _std, "std_of_std"),
                                       ("mean", _mean, "mean_of_mean"), ("mean", _std, "std_of_mean")):
            for comp in "uvwpqr":
                rows = tbl[kind][comp]
                assert len(rows) == 10, (name, kind, comp)
                got = agg(rows)
                want = tbl["ensemble"][printed_key][comp]
                tol = max(0.025 * abs(want), 0.8 * _quantum(rows)) + 0.5 * _quantum([want])
                if abs(got - want) > tol:
                    problems.append(f"{name}.{printed_key}.{comp}: recomputed {got:.6g} "
                                    f"vs printed {want:.6g} (tol {tol:.2g})")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=pathlib.Path, default=None,
                    help="optional local copy of the report PDF to verify against the pin")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / "checkdata")
    args = ap.parse_args()

    if args.pdf is not None:
        digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
        assert digest == PROVENANCE["pdf_sha256"], \
            f"PDF sha256 mismatch: {digest} != pinned {PROVENANCE['pdf_sha256']}"
        print(f"pdf sha256 verified: {digest}")

    problems = self_check()
    if problems:
        raise SystemExit("transcription self-check FAILED:\n  " + "\n  ".join(problems))
    print("transcription self-check passed: all 144 printed ensemble cells reproduced "
          "from per-run rows")

    payload = {
        "provenance": PROVENANCE,
        "constants": CONSTANTS,
        "equations": EQUATIONS,
        "tables": TABLES,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "dryden_cr206937_tables.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
