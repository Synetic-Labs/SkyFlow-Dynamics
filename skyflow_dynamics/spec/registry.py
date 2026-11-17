"""
The term registry: every physics term in the spec, with tier, provenance, and tests — and
the ledger of what is missing or unfinished (INTAKE.md).

Tiers:
  verified  — reproduces an independent reference (see EVIDENCE); covered by golden vectors
              or exact property tests in this repo.
  candidate — credible published model, symbolically checked and cited, in spec/; awaiting
              independent numeric validation.
  proposed  — a known effect with provenance but no spec expression yet (expression "").

Ledger fields, per INTAKE.md: `evidence` (how the term was checked), `use` (math only, or
emitted by a backend), and — for every term not yet final (verified AND in a backend) —
`decision` on the next step, the `revisit` condition, and the `effect` size at the INTAKE
operating points. Harness terms are exempt from the ledger fields.

Domains: rigid_body · actuator · rotor_aero · frame_aero · sensor · disturbance ·
discretization · differentiation · environment · harness (not physics — timing/stateful
machinery, listed so nothing is lost).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    citation: str
    url: str = ""


TIERS = ("proposed", "candidate", "verified")

#: Evidence kinds (INTAKE.md step 7). All but 'derived' are independent of the spec and can
#: verify a term; 'derived' (computed from other spec terms) informs values and triage only.
EVIDENCE = {
    "executed": "golden vectors from running the source's own code",
    "published": "reference tables or run statistics published by the source, pinned by hash",
    "measured": "beats the backend model on held-out measured data (fixed dataset + split)",
    "analytic": "exact closed-form identity or hand-computed values, derived independently",
    "derived": "computed from other spec terms — not independent, does not verify",
}
INDEPENDENT_EVIDENCE = tuple(k for k in EVIDENCE if k != "derived")

USES = ("spec", "backend")          # math only | emitted by backends/ (checked by tracing)
DECISIONS = ("open", "pursue", "defer", "hold")


@dataclass(frozen=True)
class Term:
    key: str
    tier: str            # TIERS
    domain: str
    summary: str
    expression: str      # dotted path to the defining spec function(s); "" when proposed
    sources: tuple
    parameters: tuple = ()
    tests: tuple = ()
    notes: str = ""
    evidence: tuple = ()     # EVIDENCE kinds
    use: str = "spec"        # USES
    decision: str = "open"   # next step while not final: DECISIONS ('open' = not triaged)
    revisit: str = ""        # observable condition that reopens a 'defer' (or 'hold')
    effect: str = ""         # size at the INTAKE operating points; required once triaged

    @property
    def final(self) -> bool:
        """Nothing left to do: verified and emitted by a backend."""
        return self.tier == "verified" and self.use == "backend"

    @property
    def ledger(self) -> bool:
        """Carries a next-step decision (not final, not harness)."""
        return not self.final and self.domain != "harness"


SOURCES = {s.key: s for s in [
    Source("rotorpy", "Folk, Paulos, Kumar — RotorPy: a Python-based Multirotor Simulator "
           "with Aerodynamics for Education and Research (arXiv:2306.04485); reference "
           "implementation github.com/spencerfolk/rotorpy",
           "https://github.com/spencerfolk/rotorpy"),
    Source("mahony2012", "Mahony, Kumar, Corke — Multirotor Aerial Vehicles: Modeling, "
           "Estimation, and Control of Quadrotor, IEEE RAM 2012"),
    Source("crazyflow", "Crazyflow first-principles dynamics "
           "(crazyflow/dynamics/first_principles/dynamics.py) + identified params.toml",
           "https://github.com/learnsyslab/crazyflow"),
    Source("skydreamer", "SkyDreamer (arXiv:2510.14783) + reference implementation "
           "embodied/envs/skydreamer.py", "https://github.com/The-Real-Thisas/dreamerv3"),
    Source("flightning", "Heeg, Song, Scaramuzza — Learning Quadrotor Control From Visual "
           "Features Using Differentiable Simulation, ICRA 2025; reference implementation "
           "executed at commit a5d5619 (gen_flightning.py, era jax 0.4.30 per finding F-28)",
           "https://github.com/uzh-rpg/rpg_flightning"),
    Source("forster2015", "Förster — System Identification of the Crazyflie 2.0 Nano "
           "Quadrocopter, ETH Zürich, 2015"),
    Source("graf", "Graf — Quaternions and Dynamics (quaternion kinematics)"),
    Source("eschmann2024", "Eschmann et al. — Data-Driven System Identification of Quadrotors "
           "Subject to Motor Delays, 2024 (arXiv:2404.07837)"),
    Source("faessler2018", "Faessler, Franchi, Scaramuzza — Differential Flatness of Quadrotor "
           "Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories, "
           "IEEE RA-L 2018"),
    Source("jsbsim", "JSBSim flight dynamics engine (FGPropeller, FGRotor, "
           "FGBrushLessDCMotor, FGStandardAtmosphere, FGWinds); evaluated at commit "
           "9a0b028; golden vectors from the EXECUTED official PyPI wheel v1.3.1 "
           "(golden/generate/gen_jsbsim.py)", "https://github.com/JSBSim-Team/jsbsim"),
    Source("mccormick", "McCormick — Aerodynamics, Aeronautics, and Flight Mechanics, 1st ed. "
           "(momentum-theory induced velocity, Eq. 6.15)"),
    Source("sh79", "Shaughnessy, Deaux, Yenni — Development and Validation of a Piloted "
           "Simulation of a Helicopter and External Sling Load, NASA TP-1285, 1979 (JSBSim "
           "FGRotor's model basis); body-rate flap terms per Amer, NACA TN-2136, 1950"),
    Source("bramwell", "Bramwell — Helicopter Dynamics, 2nd ed., eqns 3.43-3.44 (rotor torque "
           "decomposition: profile + induced/climb components)"),
    Source("talbot1977", "Talbot & Corliss — A Mathematical Force and Moment Model of a UH-1H "
           "Helicopter for Flight Dynamics Simulations, NASA TM-73,254, 1977 (eqn 10a "
           "ground-effect inflow factor)"),
    Source("sanchez2017", "Sanchez-Cuevas, Heredia, Ollero — Characterization of the Aerodynamic "
           "Ground Effect and Its Influence in Multirotor Control, Int. J. Aerospace Eng. 2017, "
           "doi 10.1155/2017/1823056"),
    Source("cheeseman1955", "Cheeseman & Bennett — The Effect of the Ground on a Helicopter "
           "Rotor in Forward Flight, ARC R&M 3021, 1955. Read from the report (Cranfield "
           "AERADE archive) on 2026-11-16: Eqs. (1)-(4) at constant power, Eqs. (5)-(6) "
           "for a fixed-pitch rotor at fixed speed",
           "https://reports.aerade.cranfield.ac.uk/handle/1826.2/3590"),
    Source("he2020", "He & Leang — Quasi-Steady In-Ground-Effect Model for Single and "
           "Multirotor Aerial Vehicles, AIAA Journal 58(12), 2020, doi 10.2514/1.J059223. "
           "Eqs. (3)-(16) and Table 2 read from the author-hosted paper: eleven fixed-pitch "
           "propellers on a thrust stand at constant rotor speed, 13 heights",
           "https://doi.org/10.2514/1.J059223"),
    Source("carter2021", "Carter, Bouchard, Quinn — Influence of the Ground, Ceiling, and "
           "Sidewall on Micro-Quadrotors, AIAA Journal, 2021, doi 10.2514/1.J059787: a tethered Crazyflie 2.0 "
           "on a load cell (Fig. 2a read from the figure by the literature search)",
           "https://doi.org/10.2514/1.J059787"),
    Source("kan2019", "Kan, Thomas, Teng, Tanner, Kumar, Karydis — Analysis of Ground Effect "
           "for Small-scale UAVs in Forward Flight, IEEE Robotics and Automation Letters, "
           "2019 (accepted-version preprint read): an AscTec Hummingbird in free flight, "
           "hover fit Eq. (12), forward-flight fit Eq. (11)",
           "https://par.nsf.gov/servlets/purl/10181219"),
    Source("dumutel2024", "David Du Mutel, Scanavino, Conyers et al. — Ground, ceiling and "
           "wall effect evaluation of small quadcopters in pressure-controlled environments, "
           "J. Intell. Robot. Syst. 110:125, 2024, doi 10.1007/s10846-024-02155-7. Setup "
           "(§2.1.1, §3.1, Table 1) read; data from Zenodo 13166292 (CC-BY 4.0), pinned by "
           "sha256 in tools/ground_data.py (F-36): a quadcopter on a load cell, 10×4.7 and "
           "12×5 propellers, ground and ceiling panel at 5–100 cm, two rotor speeds logged",
           "https://zenodo.org/records/13166292"),
    Source("hsiao2019", "Hsiao & Chirarattananon — Ceiling Effects for Hybrid Aerial-Surface "
           "Locomotion of Small Rotorcraft, IEEE/ASME Trans. Mechatronics, 2019 (preprint "
           "read): momentum-theory ceiling coefficient γ(δ = R/D), Eqs. (6), (10), (14); the "
           "thrust coefficient needs three blade-profile constants c0, c1, c2",
           "https://arxiv.org/abs/1905.04632"),
    Source("kiran2025", "Kiran, Ayanian, Breuer — Influence of Static and Dynamic Downwash "
           "Interactions on Multi-Quadrotor Systems, RSS 2025, doi 10.15607/RSS.2025.XXI.102. "
           "Sec. II-A, III-A/B, Figs. 5-9 and Table I read from the proceedings PDF (scanned). "
           "Data: Hugging Face ajkiran18/QuadrotorDownwash (MIT), pinned by sha256 in "
           "tools/downwash_data.py (F-38)",
           "https://doi.org/10.15607/RSS.2025.XXI.102"),
    Source("mtb2", "Conley, Russell, Kallstrom, Shirazi, Pereyra, Wright — Multirotor Test Bed "
           "Second Wind Tunnel Entry Data Report, NASA/TM-20250004795, May 2025 (geometry, "
           "Tables 7.2.1-7.2.2 and Appendix A read), and the data tables "
           "mtbii_data_tables_v2.xlsx, pinned by sha256 in tools/rotor_interference_data.py "
           "(F-37): 24.5 in two-blade rotors, loads and speed per rotor, single, tandem, "
           "side-by-side and four-rotor layouts",
           "https://rotorcraft.arc.nasa.gov/Research/Programs/MTB2.html"),
    Source("johnson2005", "Johnson — Model for Vortex Ring State Influence on Rotorcraft "
           "Flight Dynamics, NASA/TP-2005-213477, 2005. Text pp. 14-16, Table 3 (algorithm), "
           "Table 4 (constants) and Figs. 36-39 read from the report",
           "https://ntrs.nasa.gov/citations/20060024029"),
    Source("shetty2011", "Shetty & Selig — Small-Scale Propellers Operating in the Vortex "
           "Ring State, AIAA 2011-1254: APC 9-11 in propellers at 4000 rpm, thrust against "
           "negative advance ratio (Figs. 4 and 11 checked; no torque was measured)",
           "https://m-selig.ae.illinois.edu/pubs/ShettySelig-2011-AIAA-2011-1254-LRN-VSR-Props.pdf"),
    Source("shetty2011thesis", "Shetty — Measurements of Small-Scale Propellers Operating in "
           "the Vortex Ring State, M.S. thesis, UIUC 2011: Appendix B tabulates C_T(J) at "
           "4000 rpm for 24 APC 9-11 in propellers down to J = −0.8, Appendix A their chord "
           "and twist (read here; pinned by sha256 in tools/descent_data.py, F-35)",
           "https://hdl.handle.net/2142/18400"),
    Source("uiucprop", "Selig et al. — UIUC Propeller Data Site, Volume 1: static C_T(rpm) "
           "of the same APC propellers, same lab and tunnel (the hover normaliser for F-35)",
           "https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html"),
    Source("veismann2022", "Veismann, Yos, Gharib — Parametric study of small-scale rotors in "
           "axial descent, Physics of Fluids 34, 035124, 2022: 6 in printed rotors of varied "
           "pitch, chord and taper at constant rotor speed (read by the literature search)",
           "https://doi.org/10.1063/5.0083761"),
    Source("foster2017", "Foster & Hartman — High-Fidelity Multi-Rotor Unmanned Aircraft "
           "System Simulation Development for Trajectory Prediction Under Off-Nominal Flight "
           "Dynamics, AIAA conference paper, 2017 (NASA NTRS 20170006073). Figs. 8-9 (isolated 14 in "
           "rotor: mean and fluctuating thrust over V_x/v_h, V_z/v_h) and Eq. (8) read",
           "https://ntrs.nasa.gov/citations/20170006073"),
    Source("bogaerts2023", "Bogaerts — Towards a Better Understanding of Attitude "
           "Oscillations Experienced by Quadcopters in Axial Descent, MSc thesis, TU Delft, "
           "July 2023: a 5 in Betaflight quadrotor in free-flight descents and on a gimbal "
           "in a wind tunnel, isolated-rotor thrust maps, an 11-state closed-loop model "
           "(conclusions read; figures read by the literature search)",
           "https://repository.tudelft.nl/record/uuid:e868d941-42e5-46c0-b706-c21a6bece0da"),
    Source("russell2018", "Russell, Willink, Theodore, Jung, Glasner — Wind Tunnel and Hover "
           "Performance Test Results for Multicopter UAS Vehicles, NASA/TM-2018-219758, "
           "February 2018: "
           "forces and moments of five whole vehicles, bare airframes and isolated rotors, "
           "tabulated in Appendix C (Tables C68a, C69a, C84a read)",
           "https://rotorcraft.arc.nasa.gov/Publications/files/Russell_1180_Final_TM_022218.pdf"),
    Source("nguyen2018", "Nguyen, Liu, Mori — Experimental Study for Aerodynamic Performance "
           "of Quadrotor Helicopter, Trans. Japan Soc. Aero. Space Sci. 61(1):29-39, 2018. "
           "Eqs. (33)-(40) and Figs. 6, 18 read from the paper",
           "https://doi.org/10.2322/tjsass.61.29"),
    Source("chen1989", "R.T.N. Chen — A Survey of Nonuniform Inflow Models for Rotorcraft "
           "Flight Dynamics and Control Applications, NASA TM-102219, 1989: the Pitt-Peters "
           "dynamic inflow matrices, Eqs. (13)-(21) (read by the literature search)",
           "https://ntrs.nasa.gov/citations/19900006622"),
    Source("pybullet_drones", "utiasDSL/gym-pybullet-drones BaseAviary (_groundEffect, "
           "_downwash) + cf2x.urdf identified constants",
           "https://github.com/utiasDSL/gym-pybullet-drones"),
    Source("jain2019", "Jain, Fortmuller, Byun, Makiharju, Mueller — Modeling of aerodynamic "
           "disturbances for proximity flight of multirotors, ICUAS 2019, Eqs. (1)-(8)"),
    Source("bangura", "Bangura & Mahony (ACRA 2012, Eqs. 6-11); Bangura, Lim, Kim, Mahony "
           "(ICRA 2014, Eqs. 3-15); Bangura et al. — Aerodynamics of Rotor Blades for "
           "Quadrotors (arXiv:1601.00733)"),
    Source("kai2017", "Kai, Allibert, Hua, Hamel — Nonlinear feedback control of quadrotors "
           "exploiting first-order drag effects, IFAC World Congress 2017, Eqs. (6)-(13)"),
    Source("rotors_px4", "ethz-asl/rotors_simulator gazebo_motor_model.cpp + "
           "PX4/PX4-SITL_gazebo-classic variant (signed rolling moment); common.h "
           "FirstOrderFilter (exact-exp motor lag, asymmetric up/down time constants); "
           "gazebo_imu_plugin.cpp (Gauss–Markov IMU error model, ADIS16448 defaults)",
           "https://github.com/ethz-asl/rotors_simulator"),
    Source("hamandi2021", "Hamandi, Usai, Sablé, Staub, Tognon, Franchi — Design of "
           "Multirotor Aerial Vehicles: a Taxonomy Based on Input Allocation, IJRR "
           "40(8-9):1015-1044, 2021, doi 10.1177/02783649211025998, Eq. (1) (per-rotor "
           "arbitrary unit thrust axis; equation verified from the open-access HAL PDF "
           "hal-02433405v2)",
           "https://hal.science/hal-02433405"),
    Source("titterton2004", "Titterton & Weston — Strapdown Inertial Navigation Technology, "
           "2nd ed., IEE Radar, Sonar and Navigation Series 17 / AIAA, 2004, "
           "doi 10.1049/PBRA017E — the standard reference for strapdown specific-force "
           "measurement and lever-arm ('size effect') compensation. Book text not directly "
           "consulted: the spec's IMU model is derived from rigid-body kinematics and "
           "proven against an independent closed-form reference (test_sensors.py)"),
    Source("chen2006", "Chen & Rincon-Mora — Accurate Electrical Battery Model Capable of "
           "Predicting Runtime and I-V Performance, IEEE Trans. Energy Conversion 21(2), 2006"),
    Source("crazyflie_fw", "bitcraze/crazyflie-firmware — motors.c "
           "motorsCompensateBatteryVoltage + platform_defaults_cf2.h (master and tag 2022.01)",
           "https://github.com/bitcraze/crazyflie-firmware"),
    Source("gazebo_battery", "gazebosim/gz-sim LinearBatteryPlugin.cc (linear OCV + internal "
           "resistance + current low-pass)", "https://github.com/gazebosim/gz-sim"),
    Source("mil8785c", "MIL-F-8785C / MIL-HDBK-1797 — Flying Qualities of Piloted Aircraft: "
           "Dryden and von Karman continuous turbulence, low-altitude closures, discrete gust"),
    Source("ussa1976", "US Standard Atmosphere 1976 (NASA-TM-X-74335): layered T(h), P(h), "
           "ideal-gas density"),
    Source("neurobem", "Bauersfeld, Kaufmann, Foehn, Sun, Scaramuzza — NeuroBEM: Hybrid "
           "Aerodynamic Quadrotor Model, RSS 2021 (arXiv:2106.08015); BEM component "
           "eqs. (5)-(19), Kingfisher platform §IV-B", "https://arxiv.org/abs/2106.08015"),
    Source("agilicious", "Foehn et al. — Agilicious (Science Robotics 2022) agilib simulator, "
           "GPLv3; evaluated via public mirror alibabasomeone/agilicious_internal_mine at "
           "commit ba8caa7 — BEM/model sources byte-identical to the RPG init commit 2d78b81 "
           "(Foehn, 2022-06-22); golden vectors from the EXECUTED compiled agilib "
           "(golden/generate/gen_agilicious.py)",
           "https://github.com/alibabasomeone/agilicious_internal_mine"),
    Source("hoffmann2007", "Hoffmann, Huang, Waslander, Tomlin — Quadrotor Helicopter Flight "
           "Dynamics and Control: Theory and Experiment, AIAA GNC 2007 (VRS empirical quartic; "
           "hinged-blade spring model)"),
    Source("gill2017", "Gill & D'Andrea — Propeller Thrust and Drag in Forward Flight, IEEE "
           "CCTA 2017 (sinusoidal high-incidence lift/drag polars; with Ducard & Hua, CCA 2014)"),
    Source("mathworks_aeroblks", "MathWorks Aerospace Blockset documentation — block-equation "
           "pages only (implementations are proprietary and were not consulted); each block "
           "cites the public standard it implements",
           "https://www.mathworks.com/help/aeroblks/"),
    Source("mujoco_fluid", "MuJoCo documentation, Computation > Fluid forces, inertia-based "
           "model (equivalent box from the body inertia; quadratic and Stokes force and "
           "torque); active in google-deepmind/mujoco_menagerie skydio_x2/x2.xml and "
           "bitcraze_crazyflie_2/cf2.xml via <option density viscosity>",
           "https://mujoco.readthedocs.io/en/stable/computation/fluid.html"),
    Source("cr206937", "Yeager — Implementation and Testing of Turbulence Models for the "
           "F18-HARV Simulation, NASA CR-1998-206937, 1998. Pinned-document golden source "
           "(sha256 4f63d46d…, NTRS 19980028448): GUSTMDL ACSL listing + Tables 2-7 run "
           "statistics", "https://ntrs.nasa.gov/citations/19980028448"),
    Source("simitl", "SimITL (AJ92/SimITL) src/sim/physics.cpp at commit c881d55, as vendored "
           "and patched by hammerill/libfpv at 08d140a (patches/simitl/0011-0017) — a "
           "Betaflight-in-the-loop FPV game flight model. Constants are hand-tuned by feel; "
           "no measured data. Evaluated by reading, not executed",
           "https://github.com/hammerill/libfpv"),
    Source("veismann2023", "Veismann, Dougherty, Gharib — Effects of rotor separation on the "
           "axial descent performance of dual-rotor configurations, Flow 3, E7, 2023, "
           "doi 10.1017/flo.2022.31 (open access; §4 and figs. 6-7 read from the PDF): "
           "mean and standard deviation of thrust for 4, 5 and 6 in fixed-pitch rotor "
           "pairs in simulated axial descent, Re 3.0e4-8.1e4, 2 kHz load cell, 50 Hz "
           "low-pass", "https://doi.org/10.1017/flo.2022.31"),
    Source("ghalamchi2020", "Ghalamchi, Jia, Mueller — Real-Time Vibration-Based Propeller "
           "Fault Diagnosis for Multicopters, IEEE/ASME Trans. Mechatronics 25(1):395-405, "
           "2020, doi 10.1109/TMECH.2019.2947250. Equations (1)-(10), Table I and Fig. 3 "
           "read from the author-hosted version (UC Berkeley HiPeRLab); the journal "
           "reference is from the publisher listing, not printed on that version",
           "https://hiperlab.berkeley.edu/wp-content/uploads/2019/10/"
           "2019-Real-time-vibration-based-propeller-fault-diagnosis-for-multicopters.pdf"),
    Source("l2f", "rl-tools L2F multirotor environment (include/rl_tools/rl/environments/l2f) "
           "as pinned by rl-tools/hyperdrone 01eee73 (rl-tools ad2e56e, 2026-10-30; Eschmann "
           "& Loianno — HyperDrone, arXiv:2610.28950) and rl-tools/raptor 2c789df (rl-tools "
           "e43ae4b, 2025-04-22; RAPTOR foundation policy, Science Robotics, "
           "doi 10.1126/scirobotics.aec1481). Golden vectors from the EXECUTED header-only "
           "library at ad2e56e (golden/generate/gen_l2f.py)",
           "https://github.com/rl-tools/rl-tools"),
]}


TERMS = (
    # ---------------- rigid body ----------------
    Term("newton_euler", "verified", "rigid_body",
         "Rigid-body translational + rotational EOM with full inertia matrix",
         "spec.rigid_body.translational, spec.rigid_body.rotational",
         ("rotorpy", "mahony2012"), ("mass", "grav", "inertia"),
         ("properties/test_rigid_body.py", "properties/test_golden.py"),
         evidence=("executed",),
         use="backend"),
    Term("quaternion_kinematics", "verified", "rigid_body",
         "q̇ = ½ q ⊗ (0, ω); wxyz scalar-first Hamilton, body→world",
         "spec.quaternion.kinematics", ("graf", "rotorpy"),
         (), ("properties/test_quaternion.py",),
         "Unit norm preserved exactly by the continuous equation; discrete integrators "
         "renormalize post-step (harness).",
         evidence=("analytic",),
         use="backend"),

    # ---------------- actuator ----------------
    Term("motor_first_order_lag", "verified", "actuator",
         "Ω̇ = (Ω_c − Ω)/τ_m", "spec.motor.first_order_lag",
         ("rotorpy", "forster2015"), ("tau_m",),
         ("properties/test_motor.py", "properties/test_golden.py"),
         evidence=("executed",),
         use="backend"),
    Term("motor_asymmetric_lag", "verified", "actuator",
         "Separate spin-up/spin-down linear+quadratic rates",
         "spec.motor.asymmetric_lag", ("crazyflow", "rotors_px4", "l2f"),
         ("ka1", "ka2", "kd1", "kd2"),
         ("properties/test_motor.py", "properties/test_golden.py",
          "properties/test_golden_l2f.py"),
         "Crazyflow coefficients are RPM-based; because Ω̇ rescales with Ω, ka1/kd1 carry "
         "over unchanged and ka2/kd2 convert by ×60/2π — i.e. (60/2π)^(Ω-power − 1). Reduces "
         "to first-order at (1/τ, 0, 1/τ, 0). RotorS/PX4's FirstOrderFilter is the "
         "pure-exponential special case (ka2 = kd2 = 0), as is L2F's rising/falling "
         "time-constant pair (executed 2026-11-15: its RK4-integrated rotor state "
         "reproduced through spec RK4 at 1e-12).",
         evidence=("executed",),
         use="backend"),
    Term("motor_exact_exp_discretization", "verified", "discretization",
         "Closed-form Ω(dt) = Ω_c + (Ω₀−Ω_c)e^(−dt/τ); operator-split from the RK stages",
         "spec.motor.exact_exp_step", ("flightning", "rotors_px4", "l2f"), ("tau_m",),
         ("properties/test_motor.py", "properties/test_golden_flightning.py",
          "properties/test_golden_l2f.py"),
         "Unconditionally stable; per-step gradient factor e^(−dt/τ) ∈ (0,1). Linear lag "
         "only. Exactness proven symbolically against the lag ODE (test_motor.py); "
         "verified 2026-08-19 against the EXECUTED flightning quadrotor_obj.py "
         "((Ω−Ω_c)e^(−dt/τ)+Ω_c, post-step clip to [Ω_min, Ω_max] as harness detail); also "
         "RotorS' FirstOrderFilter. Executed 2026-11-15 as L2F's CLOSED_FORM rotor step "
         "with the time constant picked by direction (rising / falling) — exact under a "
         "held command, since the approach never crosses its setpoint; the reference "
         "feeds that path the raw action instead of the mapped command (finding F-31).",
         evidence=("executed", "analytic"),
         use="backend"),
    Term("throttle_curve", "verified", "actuator",
         "Ω_c = (Ω_max−Ω_min)√(k·u² + (1−k)u) + Ω_min",
         "spec.motor.throttle_to_speed", ("skydreamer",), (),
         ("properties/test_motor.py",),
         "Identified k = 0.5 for a 5-inch racer (Ω_min 341.75, Ω_max 3100 rad/s).",
         evidence=("analytic",),
         use="backend"),
    Term("pwm_quantization", "verified", "actuator",
         "Throttle snapped to the integer PWM grid",
         "spec.motor.pwm_quantize", ("crazyflow",), (),
         ("properties/test_motor.py",),
         "Piecewise-constant — exclude from differentiable paths.",
         evidence=("analytic",)),
    Term("battery_voltage_speed_cap", "verified", "actuator",
         "Supply voltage → achievable Ω_max (linear map); slow drift over discharge",
         "spec.motor.voltage_to_rpm", ("crazyflow", "skydreamer"), (),
         ("properties/test_motor.py",),
         "Battery state evolution (SoC, sag dynamics) is harness-side.",
         evidence=("analytic",)),
    Term("rotor_thrust_polynomial", "verified", "rotor_aero",
         "T_i = ct0 + ct1·Ω + ct2·Ω² per rotor (per-rotor coefficient asymmetry supported)",
         "spec.rotor_aero.thrust_magnitude", ("rotorpy", "crazyflow", "skydreamer"),
         ("ct0", "ct1", "ct2"), ("properties/test_wrench.py", "properties/test_golden.py"),
         "Crazyflow identifies RPM-unit polynomials; SkyDreamer's k_w is mass-normalized "
         "(multiply by m; finding F-4).",
         evidence=("executed",),
         use="backend"),
    Term("rotor_torque_polynomial", "verified", "rotor_aero",
         "Q_i = cq0 + cq1·Ω + cq2·Ω²; yaw torque on airframe = −s_i·Q_i·ê_i (opposes spin)",
         "spec.rotor_aero.torque_magnitude", ("rotorpy", "crazyflow"),
         ("cq0", "cq1", "cq2", "spin"), ("properties/test_wrench.py", "properties/test_golden.py"),
         "⚠ RotorPy's rotor_directions = torque sign = −spin (finding F-6).",
         evidence=("executed",),
         use="backend"),
    Term("thrust_axis_misalignment", "verified", "rotor_aero",
         "Per-rotor unit thrust axis ê_i ≠ ẑ from assembly tolerance → parasitic forces/moments",
         "spec.wrench.body_wrench", ("hamandi2021",), ("axis",),
         ("properties/test_wrench.py",),
         "General per-rotor axis form per Hamandi Eq. (1). No executed reference models "
         "assembly tilt directly; verified by exact hand-computed wrench tests "
         "(test_wrench.py) — pure geometry, zero numerical slack.",
         evidence=("analytic",),
         use="backend"),
    Term("rotor_inertia_moments", "verified", "rotor_aero",
         "Gyroscopic precession −ω×h and yaw reaction −I_rot·Σ s_i Ω̇_i·ẑ",
         "spec.wrench.rotor_inertia_moment", ("crazyflow", "skydreamer", "flightning"),
         ("I_rot", "spin"), ("properties/test_wrench.py", "properties/test_golden.py",
                             "properties/test_golden_flightning.py"),
         "Signs re-derived from τ = −d/dt(h). Crazyflow's gyro roll-row sign was flipped "
         "(finding F-3, confirmed against their running code); fixed upstream by "
         "learnsyslab/crazyflow PR #86 (merged 2026-07-13) — post-fix Crazyflow golden "
         "vectors now cross-validate this term. flightning implements ONLY the yaw-reaction "
         "half (+I_m·Σ dir_i·Ω̇_i·ẑ with dir_i = −s_i ≡ this term at ω = 0, executed-code "
         "verified) with the continuous rate Ω̇ = (Ω_c−Ω)/τ; it omits the −ω×h precession.",
         evidence=("executed",),
         use="backend"),

    # ---------------- aerodynamics ----------------
    Term("rotor_drag_hforce", "verified", "rotor_aero",
         "H_i = −Ω_i·diag(k_d,k_d,k_z)·v_i at each hub, v_i incl. ω×r_i lever arm",
         "spec.rotor_aero.rotor_drag_force", ("rotorpy", "mahony2012", "skydreamer"),
         ("k_d", "k_z"), ("properties/test_energy.py", "properties/test_golden.py",
                          "properties/test_findings.py"),
         "SkyDreamer's lumped −k_x·v·ΣΩ is this summed over rotors (k_d = m·k_x; F-4). "
         "Nondimensional form κ = k_d/√(ρ·A·c_T/2), the H-force over the hover momentum "
         "drag: 0.40 on RACER_5IN (measured k_d), 0.22 on CRAZYFLIE (placeholder k_d). "
         "libfpv's datasheet prior κ = 0.6 overstates the racer's k_d by 51 % "
         "(tools/sim_source_triage.py) — a starting value before identification, no more.",
         evidence=("executed",),
         use="backend"),
    Term("blade_flapping_moment", "verified", "rotor_aero",
         "M_flap,i = −k_flap·Ω_i·(v_i × ẑ); +M_y (nose-down in FLU) for v_x > 0, k_flap > 0",
         "spec.rotor_aero.flapping_moment", ("rotorpy", "mahony2012", "russell2018"),
         ("k_flap",),
         ("properties/test_wrench.py", "properties/test_golden.py", "properties/test_wake.py"),
         "Sign and size from measurement (finding F-34, 2026-11-16): an isolated rigid 10 in "
         "rotor in edgewise flow carries a NOSE-UP hub moment of 0.13–0.22·T·R at advance "
         "ratio 0.08–0.16 (NASA/TM-2018-219758, Tables C68a/C69a; the range is with and "
         "without the hover offset). In this expression that is k_flap < 0 with "
         "−k_flap/c_T = 1.2–2.2; CFD and one more wind-tunnel source put it at about 0.9. "
         "On RACER_5IN at 10 m/s: 0.06–0.15 N·m nose-up, 2–5 % of pitch authority "
         "(tools/wake_triage.py). Both reference vehicles ship k_flap = 0: the racer's "
         "flight data cannot identify moments (F-30) and the measured rotor is not its "
         "propeller. The moment falls with nose-down tilt (about half at 40°) and the linear "
         "form does not carry that.",
         evidence=("executed",),
         use="backend"),
    Term("translational_lift", "verified", "rotor_aero",
         "ΔT_i = k_h·(v_i,x² + v_i,y²)",
         "spec.rotor_aero.translational_lift", ("rotorpy", "agilicious"), ("k_h",),
         ("properties/test_golden.py", "properties/test_golden_agilicious.py"),
         "Also executed as agilib ModelLinCubDrag's induced_lift_coeff (2026-08-19). "
         "Small-airspeed linearization of the AoA/advance-ratio model — mutually exclusive "
         "with k_angle/k_hor (validation rule).",
         evidence=("executed",),
         use="backend"),
    Term("aoa_advance_ratio_thrust", "verified", "rotor_aero",
         "T × (1 + k_angle·atan2(v_az, rΩ̄) + k_hor·atan2(‖v_axy‖, rΩ̄))",
         "spec.rotor_aero.aoa_thrust_factor", ("skydreamer",),
         ("k_angle", "k_hor", "r_prop"), ("properties/test_golden.py",),
         "Identified to racing speeds: k_angle 3.145, k_hor 7.245 (mass-normalized k_w; F-4). "
         "Spec follows the runnable reference (ENU, mean Ω̄, hypot); the paper's printed "
         "equations differ (NED, ΣΩ, squared numerator) — see the function docstring.",
         evidence=("executed",),
         use="backend"),
    Term("vertical_climb_drag", "verified", "rotor_aero",
         "−k_v2·v_az·|v_az|·ẑ collective at CoM",
         "spec.rotor_aero.vertical_climb_drag", ("skydreamer",), ("k_v2",),
         ("properties/test_golden.py",),
         evidence=("executed",),
         use="backend"),
    Term("linear_drag", "verified", "frame_aero",
         "Lumped linear body-frame drag F = −diag(c_L)·v_a (Faessler differential-flatness form)",
         "spec.rotor_aero.linear_drag", ("faessler2018", "crazyflow", "agilicious"),
         ("c_L",),
         ("properties/test_energy.py", "properties/test_golden.py",
          "properties/test_golden_agilicious.py"),
         "Ω-independent lumping of the per-rotor H-force; identify against c_L OR k_d, "
         "not both. Crazyflow stores the negated diagonal (drag_matrix = −diag(c_L)).",
         evidence=("executed",),
         use="backend"),
    Term("parasitic_drag", "verified", "frame_aero",
         "D = −‖v_a‖·diag(c_D)·v_a at CoM",
         "spec.rotor_aero.parasitic_drag", ("rotorpy",), ("c_D",),
         ("properties/test_energy.py", "properties/test_golden.py"),
         "⚠ ‖v‖-scaled, NOT per-axis |v_k|·v_k (SkyDreamer's form) — structurally different; "
         "don't transplant coefficients between the two.",
         evidence=("executed",),
         use="backend"),

    # ---------------- disturbances / inputs ----------------
    Term("external_wrench_inputs", "verified", "disturbance",
         "Exogenous F_ext (world) and τ_ext (body) enter the EOM directly",
         "spec.rigid_body.translational, spec.rigid_body.rotational",
         ("skydreamer", "crazyflow"), (), ("properties/test_golden.py",),
         "The two-band resample-and-hold schedule that drives these in training (SkyDreamer "
         "Table III: ±3 m/s²+±3 rad/s² @1 Hz, ±125 rad/s² @90 Hz, ε_u ±0.2) is harness-side.",
         evidence=("executed",),
         use="backend"),

    # ---------------- sensors ----------------
    Term("imu_measurement", "verified", "sensor",
         "Specific force + body rate at offset/rotated mount, lever-arm terms in body frame",
         "spec.sensors.imu", ("titterton2004",), (),
         ("properties/test_sensors.py",),
         "Derived from rigid-body kinematics: a_S = a_B + α×r + ω×(ω×r) with all lever-arm "
         "quantities in the body frame, rotated once; proven against an independent "
         "closed-form model in test_sensors.py. Upstream rotorpy's IMU mixes frames "
         "(findings F-1/F-2, see REFERENCES.md) and is not usable as a reference.",
         evidence=("analytic",),
         use="backend"),

    # ---------------- differentiable simulation ----------------
    Term("point_mass_surrogate", "verified", "differentiation",
         "Point mass + kinematic attitude; surrogate Jacobian for BPTT via straight-through",
         "spec.simplified.step, spec.simplified.dynamics", ("flightning",), (),
         ("properties/test_simplified.py", "properties/test_golden_flightning.py"),
         "Verified 2026-08-19 against the EXECUTED flightning quadrotor_dyn: primal steps, "
         "jax.jvp tangents, AND the step()-level custom_jvp wiring (c = f_d/m, dt-tangent 0) "
         "— executed-code confirmation of the surrogate-gradient scheme. Their attitude step "
         "is a biased-angle Rodrigues, not the exp map (finding F-25, deviation bounded in "
         "the golden test); their custom_jvp is broken on JAX ≥ 0.11 (finding F-28).",
         evidence=("executed",)),
    Term("rk4_fixed_step", "verified", "discretization",
         "Classical RK4; the differentiable reference integrator (adaptive solvers are not "
         "cleanly differentiable)", "spec.discretization.rk4_step", ("rotorpy",),
         (), ("properties/test_motor.py", "properties/test_golden.py"),
         "flightning attribution removed 2026-08-19: its executed integrator is explicit "
         "Euler at 1 kHz (exact attitude/motor substeps), no RK4 anywhere in the repo.",
         evidence=("executed",),
         use="backend"),

    # ---------------- candidates: ground effect / downwash ----------------
    Term("ground_effect_cheeseman_bennett", "candidate", "rotor_aero",
         "Single-rotor IGE thrust ratio 1/(1−(R/4z)²); forward-flight washout variant",
         "spec.ground_effect.cheeseman_bennett, spec.ground_effect.cheeseman_bennett_forward",
         ("cheeseman1955", "sanchez2017"), (),
         ("properties/test_candidates.py", "properties/test_wake.py"),
         "Valid 0.5 ≤ z/R ≤ 2; singular at z = R/4 (clamp). Under-predicts for multirotors. "
         "Both forms confirmed against the report (2026-11-16): they hold at CONSTANT POWER. "
         "To first order that is an inflow sensitivity of 1; a fixed-pitch rotor at fixed "
         "speed has 0.27–0.63 (ground_effect_fixed_rpm), so it gains less. The report's own "
         "flight tests agree above Z/R = 0.6. F-36 (whole quadcopter, rotor speed measured): "
         "within 0.03 of the data at z/R ≥ 0.98, but 2.39 against 1.26 at z/R = 0.33 — "
         "below its envelope it is worse than no ground effect at all.",
         decision="hold",
         effect=("RACER_5IN geometry: +6.7 % at z = R (0.5 rotor diameters), +0.4 % at 4R "
                 "(2 diameters). Measured at constant speed near z = R: +6 to +9 % for single "
                 "and paired rotors, +10 to +18 % for whole quadrotors (F-33)")),
    Term("ground_effect_fixed_rpm", "verified", "rotor_aero",
         "Ground effect at constant rotor speed: T/T_∞ = 1 + s·δ(z), δ the fraction of the "
         "induced velocity that the ground cancels — exp(−C_b·z/R), or the image-source "
         "(R/4z)²/(1 + (V/v_i)²)",
         "spec.ground_effect.he_leang_exponential, "
         "spec.ground_effect.cheeseman_bennett_fixed_pitch, "
         "spec.ground_effect.ground_inflow_fraction_exponential, "
         "spec.ground_effect.ground_inflow_fraction_image",
         ("he2020", "cheeseman1955", "carter2021", "kan2019", "dumutel2024"), (),
         ("properties/test_wake.py",),
         "The form for an ESC-driven fixed-pitch propeller (the other thrust-ratio "
         "candidates are at constant power or fitted to one vehicle). s is the blade's "
         "inflow sensitivity (inflow_thrust_law; He & Leang's C_a), so ground effect, the "
         "k_z inflow term and the descent thrust loss share one parameter. Exponential "
         "form: finite at the ground (1 + s), measured on eleven propellers — C_a 0.27–0.63, "
         "C_b 1.18–2.87, thrust monotonic down to the ground; three-blade 5×3: 0.53 / 1.80. "
         "A rotor inside a quadrotor reaches higher (C_b up to 30 % lower). Apply per rotor "
         "with each hub's own height: that gives the levelling torque over a tilted or "
         "partial ground with no extra term. Not covered: the body lift of a real vehicle "
         "(ground_effect_sanchez_cuevas), a thrust-LOSS region for bare rotor pairs at tip "
         "gaps of 1.2–2.8 R and heights of 1.1–2.3 R (up to 8 %, He & Leang §IV.D — the "
         "Crazyflie sits in it and still gains, so it is disputed), unsteady fountain "
         "re-ingestion below one diameter (CFD only), torque and power. Forward-flight "
         "washout 1/(1 + (V/v_i)²) is Cheeseman & Bennett's; flight data on a small "
         "quadrotor show the height term dominating to V/v_h ≈ 1.2 (Kan et al.). "
         "VERIFIED 2026-11-16 on measured data (F-36, tools/ground_data.py, Zenodo "
         "13166292 pinned): a whole quadcopter at matched rotor speed, 10 in and 12 in "
         "propellers, z/R 0.33–2.8. He & Leang's single-propeller 5×3 constants, not fitted "
         "here, give 0.020 / 0.024 rms against 0.14 / 0.12 for the backend (no ground "
         "effect); the fixed-pitch image form with the F-35 descent s = 0.44 gives 0.035 / "
         "0.033. Fitted freely: C_a 0.47 / 0.53, C_b 1.74 / 2.08 — single-propeller values. "
         "Envelope of the check: hover, four coplanar rotors at tip gaps 0.32 R and 0.78 R, "
         "frame upstream of the rotors (no body in the fountain), a shared 100 cm reference "
         "whose arm download may bias the ratios by +1 to +3 %. Next (backend): a height-"
         "above-ground input per rotor and C_a, C_b parameters.",
         evidence=("analytic", "measured"),
         decision="pursue",
         effect=("thrust gain at z/R = 0.5 / 1 / 2 / 4 with the 5×3 values: +21.5 / +8.8 / "
                 "+1.4 / +0.0 % — material below one rotor diameter. Whole quadrotors "
                 "measure more: Hummingbird +31.6 / +13.6 / +6.4 / +3.1 %, Crazyflie +18 % "
                 "at 1.14R and +7 % at 2.85R (tools/wake_triage.py)")),
    Term("ground_effect_sanchez_cuevas", "candidate", "rotor_aero",
         "Quadrotor IGE with mirrored-rotor images + fountain body-lift (K_b ≈ 2)",
         "spec.ground_effect.sanchez_cuevas", ("sanchez2017",), (),
         ("properties/test_candidates.py", "properties/test_wake.py"),
         "Significant to z ≈ 5R. Reduces to Cheeseman-Bennett as d, b → ∞, K_b → 0. "
         "Constant-power form. Independent support: it fits a tethered Crazyflie's lift for "
         "1 < z/R < 3 (R² = 0.956, Carter et al. 2021), and whole quadrotors measure well "
         "above a bare rotor (F-33). The body term is what a bare-rotor form lacks. F-36: "
         "with no body in the fountain (frame upstream) and rotor speed held, a whole "
         "quadcopter gains like a single propeller; this form overshoots even with K_b = 0 "
         "(2.55 against 1.26 at z/R = 0.33, +6 % against +2 % at 1.6R) — its image terms "
         "are at constant power. The body term itself remains untested.",
         decision="defer",
         revisit=("a consumer hovers, takes off or lands a vehicle whose body sits under the "
                  "rotor plane within 5 rotor radii of the ground, and ground_effect_fixed_rpm "
                  "alone leaves a thrust residual"),
         effect=("RACER_5IN geometry, K_b = 2: +15.8 % at z = R and +3.7 % at 4R, against "
                 "+8.8 % and +0.0 % for the bare 5×3 rotor at constant speed")),
    Term("ground_effect_pybullet", "candidate", "rotor_aero",
         "Per-rotor additive increment ΔT = T·G·(R/4z)² (linearized CB, identified G)",
         "spec.ground_effect.pybullet_ground_effect", ("pybullet_drones",), (),
         ("properties/test_candidates.py", "properties/test_wake.py"),
         "G = 11.37 identified for CF2 (fountain amplification folded in); needs height clip. "
         "F-33: G is a far-field value. A tethered Crazyflie's measured gain needs G = 3.7 "
         "at z/R = 1.14 and 9.1 at 2.85 (Carter et al. 2021, Fig. 2a): a 1/z² law falls "
         "faster than the data, so no single G fits. Do not use below z/R ≈ 2.5. F-36: a "
         "10–12 in quadcopter needs G = 0.4–1.5 at z/R ≤ 2 — G is vehicle-specific.",
         decision="hold",
         effect=("+71 % at z = R and +4.4 % at 4R with G = 11.37; nothing measured at "
                 "constant rotor speed exceeds +18 % near z = R (tools/wake_triage.py)")),
    Term("downwash_pybullet", "candidate", "disturbance",
         "Inter-vehicle Gaussian downwash force fit (DSL/SiQi Zhou)",
         "spec.ground_effect.pybullet_downwash_force", ("pybullet_drones", "kiran2025"), (),
         ("properties/test_candidates.py", "properties/test_wake.py"),
         "CF2-specific fit, thrust-independent, untrustworthy below Δz ≈ 0.7 m. F-38: "
         "contradicted in its own range by load-cell maps of two Crazyflies (Kiran et al. "
         "2025): at 0.7–1.1 m it gives 24–58 % of the weight on the centreline, measured "
         "1.8–3.6 % (12–16×; 6–8× against the measured near-field trend, which steps down "
         "between 0.6 and 0.7 m). Its core (half-max 2–78 mm) is far narrower than measured "
         "(0.11–0.14 m), so rescaling c1 does not fix it.",
         decision="hold",
         effect=("measured on a Crazyflie: 1.8–3.6 % of weight lost at 0.7–1.1 m directly "
                 "below another, 13–37 % at 0.13–0.60 m; the fit predicts 6–16× more")),
    Term("downwash_jain_jet", "candidate", "disturbance",
         "Turbulent-jet wake velocity field + frame drag + per-rotor thrust loss",
         "spec.ground_effect.jain_wake_velocity", ("jain2019", "kiran2025"), (),
         ("properties/test_candidates.py", "properties/test_wake.py"),
         "Thrust-scaled, gives force AND moment; ZEF only (z > 3L). Route wake velocity "
         "through ONE rotor-inflow path to avoid double-counting with AoA thrust terms. "
         "F-38: the form fits a Crazyflie's PIV wake (z/l 6.5–24, 0.023 U_i rms) with a "
         "virtual origin 0.37 m ABOVE the rotor plane, c_rad = 91 (round-jet spread 0.087) "
         "and c_ax ≈ 11.6 (L = 2l, reading the field with U_i ≈ 2·v_h, which closes the "
         "momentum flux); the published z0 (0.03–0.11 m below) misses the centreline by up "
         "to 70 %. Constants are per vehicle: refit, do not copy.",
         decision="defer",
         revisit=("a consumer flies two vehicles within about 1 m vertically; fit z0, c_ax "
                  "and c_rad to that vehicle's wake first"),
         effect=("Crazyflie wake centreline 0.56–0.33 of 2·v_h from 0.21 to 0.76 m below; "
                 "the lower vehicle loses 37 % of its weight at 0.13 m and 13 % at 0.6 m "
                 "(F-38) — material for stacked flight, absent for one vehicle")),
    Term("ceiling_effect", "proposed", "rotor_aero",
         "A surface ABOVE the rotors raises thrust at fixed rotor speed: weaker than the "
         "ground near it, but reaching further",
         "", ("dumutel2024", "hsiao2019"), (),
         ("properties/test_wake.py",),
         "No spec term. Measured (F-36, 12×5 whole quadcopter, rotor speed matched): +14.6 / "
         "+4.7 / +3.8 / +1.9 % at z/R = 0.33 / 0.98 / 1.64 / 2.30, against +26 / +10 / +2 / "
         "−1 % for the ground at the same heights. Hsiao & Chirarattananon give a momentum-"
         "theory form, γ = ½ + ½·√(1 + δ²/8) with δ = R/D, but thrust at fixed speed needs "
         "three blade-profile constants (their Eq. 14) and two empirical multirotor factors, "
         "which these data cannot separate. A route through inflow_thrust_law (the ceiling "
         "lowers the induced velocity) is the obvious candidate form; not derived yet.",
         decision="defer",
         revisit=("a consumer flies within about 3 rotor radii under a ceiling, a bridge or "
                  "an overhang for longer than a pass, or perches"),
         effect=("whole 12 in quadcopter: +14.6 % at z/R = 0.33, +4.7 % at 1, +1.9 % at 2.3; "
                 "on RACER_5IN that is about 2 cm and 6 cm under a surface — a pass under a "
                 "gate bar, not a sustained condition in racing")),

    # ---------------- candidates: inflow / propeller / atmosphere ----------------
    Term("momentum_induced_velocity", "verified", "rotor_aero",
         "Actuator-disk induced velocity: hover v_h = √(T/2ρA); sign-safe axial closed form",
         "spec.inflow.hover_induced_velocity, spec.inflow.induced_velocity_axial",
         ("mccormick", "jsbsim", "bangura"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "The physical input behind ground effect / downwash / climb corrections. "
         "Verified 2026-08-18 against executed JSBSim FGPropeller "
         "(golden/vectors/jsbsim_prop_bldc.json): hover form at V_a = 0 and axial form "
         "at V_a up to 18 m/s. V_a < 0 (descent/reverse-flow branch) unexercised.",
         evidence=("executed",)),
    Term("oblique_momentum_thrust", "verified", "rotor_aero",
         "Nonlinear T(airspeed): T = 2ρA·v_i·U, U = √(Vx²+Vy²+(v_i−Vz)²) (implicit v_i)",
         "spec.inflow.oblique_momentum_thrust",
         ("bangura", "neurobem", "agilicious"), (),
         ("properties/test_candidates.py", "properties/test_golden_agilicious.py"),
         "Principled model the identified k_v2/k_angle/k_hor terms linearize. VRS validity "
         "band excluded (descent 0.5–2 v_h; spec.inflow VRS constants). Verified "
         "2026-08-19: executed verbatim as agilib's ThrustFunction momentum side "
         "(NeuroBEM eq. 5) and pinned via the BEM closure vectors — the implicit-v_i "
         "root regime; explicit T(v_i) evaluation is the same expression.",
         evidence=("executed",)),
    Term("dynamic_inflow_lag", "verified", "rotor_aero",
         "First-order induced-inflow lag to Glauert equilibrium, τ ≈ 16/(γΩ); exact-exp step",
         "spec.inflow.dynamic_inflow_lag", ("jsbsim",), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Same operator-split pattern as the verified motor exact-exp discretization. "
         "Verified 2026-08-18 against executed JSBSim FGRotor "
         "(golden/vectors/jsbsim_rotor_inflow.json): the recorded ν sequences satisfy "
         "the exact-exp step to 1e-10 across hover/axial/edgewise/oblique conditions; "
         "ν_eq is the Glauert equilibrium with the reference's Bailey C_T (transcribed, "
         "self-checked at 1e-12 in the generator — the Bailey closed form itself is not "
         "a spec term). Time constants (2026-11-16): the Pitt–Peters uniform state about "
         "hover is τ = (2/(3π))·R/v_h (apparent mass 8/(3π), Carpenter & Fridovich's value); "
         "no thrust overshoot or inflow constant has been measured on a small fixed-pitch "
         "rotor. The slow lag that matters is the build-up of the vortex ring state "
         "(vrs_johnson_inflow, 4.4·R/v_h): this state would carry it.",
         evidence=("executed",),
         decision="hold",
         effect=("τ at hover: 1.8 ms on RACER_5IN, 1.1 ms on CRAZYFLIE, against motor lags "
                 "of 33 and 72 ms — second order as a thrust lag (tools/wake_triage.py). The "
                 "VRS lag by Johnson's scaling is 37 and 23 ms")),
    Term("advance_ratio_tables", "verified", "rotor_aero",
         "T = C_T(J)·ρ·n²·D⁴, P = C_P(J)·ρ·n³·D⁵ with measured tables; windmilling via sign",
         "spec.atmosphere.advance_ratio, spec.atmosphere.propeller_thrust",
         ("jsbsim", "neurobem"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Generalizes the polynomial T(Ω) (a fixed-J slice). J uses axial inflow only; "
         "UIUC/APC databases supply tables for small UAV props. NeuroBEM's BEM model is the "
         "higher-fidelity per-element variant. Verified 2026-08-18 against executed "
         "JSBSim FGPropeller with the wheel's APC 9x4.5E tables "
         "(golden/vectors/jsbsim_prop_bldc.json): J and T pinned per step over J up to "
         "0.42; the P form is exercised through the shaft-ODE load. Windmilling "
         "(J < 0 / C_T < 0) unexercised — the shipped table domain is J ≥ 0.",
         evidence=("executed",)),
    Term("isa_atmosphere", "verified", "environment",
         "USSA-1976 layered T(h), P(h); ρ = P/RT; thrust/torque scale linearly with ρ",
         "spec.atmosphere.temperature_troposphere, spec.atmosphere.pressure_gradient_layer, "
         "spec.atmosphere.density, spec.atmosphere.speed_of_sound",
         ("ussa1976", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "Verified-tier coefficients absorb ρ at identification altitude — scale by ρ/ρ_ident. "
         "Verified 2026-08-18 against executed JSBSim FGStandardAtmosphere "
         "(golden/vectors/jsbsim_isa_atmosphere.json): T/P/ρ/a at 12 altitudes to 35 kft, "
         "5e-4 relative (imperial-vs-ICAO constant sets); inputs are geopotential "
         "altitude. Stratosphere (isothermal layer) unexercised.",
         evidence=("executed",)),

    # ---------------- candidates: rotor aero extensions ----------------
    Term("rolling_moment", "candidate", "rotor_aero",
         "Per-rotor rolling moment −Ω·s·μ_R·v_⊥ (advancing/retreating dissymmetry)",
         "spec.rotor_aero.rolling_moment", ("rotors_px4", "kai2017"), (),
         ("properties/test_candidates.py",),
         "RotorS omits the spin sign (bug); PX4 variant adopted. Cancels for balanced pairs."),
    Term("flapping_force_body_rate", "candidate", "rotor_aero",
         "Kai Eq. (10) flapping force incl. spin-signed lateral and body-rate damping terms",
         "spec.rotor_aero.flapping_force_kai", ("kai2017", "faessler2018"), (),
         ("properties/test_candidates.py",),
         "The published basis for the lumped linear drag; adds rotor-plane damping (B·ω) "
         "absent from the verified tier."),
    Term("flapping_moment_body_rate", "candidate", "rotor_aero",
         "Rotor roll/pitch damping moment −k_flap_w·Ω·Π_ẑ·ω (tip-path plane lags body rate)",
         "spec.rotor_aero.flapping_moment_body_rate", ("sh79", "jsbsim", "kai2017"), (),
         ("properties/test_candidates.py",),
         "Spin-sign-free: adds (not cancels) pairwise — a net damping derivative. Kai "
         "Eq. (7) carries the same hub moment with √T scaling; JSBSim derives it from flap "
         "angles + hinge-offset hub moments.",
         decision="defer",
         revisit=("a measured or executed reference separates the hub flapping moment from the "
                  "lever-arm thrust differential (F-29 shows BEM cannot)"),
         effect=("F-29: BEM's roll damping is all lever-arm thrust differential (k_z channel); "
                 "the hub moment adds nothing in roll and +0.00083 N·m·s/rad in pitch, only "
                 "through agilib's vehicle-specific flapping fits (rejected, REFERENCES). No "
                 "defensible k_flap_w; adding one on top of a k_z set to total damping "
                 "double-counts")),
    Term("bramwell_rotor_torque", "verified", "rotor_aero",
         "Q = ρbcδ(ΩR)²R²(1+4.5μ²)/8 − (Tλ+Hμ)R: profile + induced/climb torque vs flight "
         "state",
         "spec.rotor_aero.bramwell_torque, spec.rotor_aero.blade_profile_drag",
         ("bramwell", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "The flight-condition dependence (yaw authority and power rise with μ, fall in "
         "descent) that the verified torque polynomial — its fixed-condition slice — lacks. "
         "Needs λ, μ: adopt together with dynamic_inflow_lag. Verified 2026-08-18 against "
         "executed JSBSim FGRotor (golden/vectors/jsbsim_rotor_inflow.json): torque "
         "identity with the δ = 0.009 + 0.3(6C_T/(aσ))² polar at 1e-7 incl. edgewise "
         "μ ≈ 0.07 (the (1+4.5μ²) term is live). H from zero-body-rate flapping; "
         "body-rate flapping contributions to H unexercised (rig holds rates at 0).",
         evidence=("executed",)),
    Term("ground_effect_talbot_inflow", "candidate", "rotor_aero",
         "IGE inflow factor v_i ← (1 − load·e^{−k_ge(h+h₀)})·v_i, exponential in height",
         "spec.ground_effect.talbot_inflow_factor", ("talbot1977", "jsbsim"), (),
         ("properties/test_candidates.py",),
         "Acts on induced velocity (composes with dynamic_inflow_lag); the thrust-ratio "
         "family (cheeseman_bennett, sanchez_cuevas, pybullet) acts on T directly — use "
         "one route, never both. F-33: the decay rate is dimensional. JSBSim's "
         "0.13–0.33 1/m is a full-scale value; on a small rotor it must be C_b/R with C_b "
         "1.2–2.9 (28 1/m on RACER_5IN) — which is ground_inflow_fraction_exponential of "
         "ground_effect_fixed_rpm, the same route with a measured, scale-free constant.",
         decision="hold",
         effect=("with k_ge = 0.33 1/m the factor is 0.021 per rotor radius on RACER_5IN: "
                 "the inflow would be almost cancelled several metres up. Superseded at small "
                 "scale by ground_effect_fixed_rpm")),

    # ---------------- candidates: blade-element-momentum (NeuroBEM / agilicious) ----------------
    Term("bem_blade_element_loads", "verified", "rotor_aero",
         "BEM disk-load integrands dT/dQ/dH: sinusoidal stall-capable polars "
         "cl=cl0·(sinα·cosα+ε_c), cd=cd0·sin²α over U_T=Ωr+v_hor·sinψ, U_P=(v_ver−v_i)−…",
         "spec.bem.blade_element_integrands, spec.bem.blade_section_velocities, "
         "spec.bem.inflow_angle, spec.bem.section_aoa, spec.bem.lift_coefficient, "
         "spec.bem.drag_coefficient, spec.bem.chord",
         ("neurobem", "agilicious", "gill2017"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib "
         "(golden/vectors/agilicious_bem.json): T/Q/H disk integrals reproduced at the "
         "recorded induced velocity across hover, forward flight to 18 m/s, climb, and "
         "descent; the pure-spec exact-atan2 form deviates <=2.3% from the reference's "
         "float32 atan2 (finding F-22, bounded in the test). Nonzero flapping inside the "
         "integrands unexercised (the reference zeroes it while integrating). "
         "Valid at any incidence/advance ratio (unlike the small-angle verified-tier rotor "
         "terms); smooth throughout. Camber offset ε_c = 0.07 and the H-force correction 3.0 "
         "are executed-code identifications absent from the paper. ⚠ paper eq. (7) prints "
         "+v_ver·β·cosψ where the code has −v_ver·β·cosψ (inert: flapping zeroed during "
         "integration; code form adopted). Reference quadrature: single 15-point "
         "Gauss-Kronrod per axis (generator/consumer detail, not spec).",
         evidence=("executed",),
         decision="defer",
         revisit=("a consumer task needs body-rate damping above ~50 rad/s (over ~1 % of training "
                  "or flight steps), or sustained axial descent through the vortex-ring band "
                  "(0.5–2 v_h)"),
         effect="see bem_tpp_wrench (the BEM family moves together)"),

    Term("bem_momentum_inflow_closure", "verified", "rotor_aero",
         "Induced velocity as root of T_BEM(v_i) = 2ρA·v_i·√(v_hor²+(v_ver−v_i)²)",
         "spec.bem.momentum_closure_residual", ("neurobem", "agilicious"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: the recorded Brent roots "
         "satisfy the residual within solver tolerance in 11/12 cases; in deep descent "
         "(25 m/s) the residual provably has NO root in the solver range and the "
         "reference silently returns range-max 30 m/s (finding F-21) — pinned as such. "
         "Momentum side IS spec.inflow.oblique_momentum_thrust with V=(v_hor,0,v_ver). "
         "Reference solves by warm-started vectorized Brent (tol 1e-3); differentiable "
         "backends: fixed smooth iterations, or v_i from dynamic_inflow_lag state (in-ODE).",
         evidence=("executed",),
         decision="defer",
         revisit=("a consumer task needs body-rate damping above ~50 rad/s (over ~1 % of training "
                  "or flight steps), or sustained axial descent through the vortex-ring band "
                  "(0.5–2 v_h)"),
         effect="see bem_tpp_wrench"),

    Term("vrs_empirical_inflow", "verified", "rotor_aero",
         "Vortex-ring-state induced velocity: ṽ_i = v_h·(1+1.125x−1.372x²+1.718x³−0.655x⁴), "
         "x = v_ver/v_h, gated on v_ver/v_i ∈ (0.01, 2)",
         "spec.bem.vrs_induced_velocity", ("hoffmann2007", "neurobem", "agilicious"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: quartic + executed blend replayed "
         "exactly on 6 gated cases (shallow/deep/oblique/fast descent, mixed-regime ANY-gate) "
         "and 6 ungated. Fills the descent band where momentum theory fails (spec.inflow VRS constants). "
         "Blend variants differ: paper max(ṽ_i, v_h); executed agilib max(v_i^mom, ṽ_i) then "
         "clamp ≤ 2·v_h — and its gate fires on ANY-rotor predicates (finding F-20). "
         "Non-smooth (gate + max/min): document surrogate before differentiating through.",
         evidence=("executed",),
         decision="defer",
         revisit=("a consumer task needs body-rate damping above ~50 rad/s (over ~1 % of training "
                  "or flight steps), or sustained axial descent through the vortex-ring band "
                  "(0.5–2 v_h)"),
         effect=("see bem_tpp_wrench; in the F-29 roll sweep the descending-side rotors sit in "
                 "this window at every rate, so it shapes BEM's rate-damping curve. F-32: "
                 "with this quartic the net flow through the disk only falls in descent, so "
                 "BEM gives 1.18× hover thrust at constant speed and a descent ratio of 1.2 "
                 "where 4–6 in rotors measure 0.71–0.78× — use vrs_johnson_inflow for the "
                 "mean thrust in descent")),

    # ---------------- candidates: a rotor in its own wake (2026-11-16) ----------------
    Term("inflow_thrust_law", "candidate", "rotor_aero",
         "Fixed-pitch rotor at fixed speed: T/T_static = 1 − s·(λ/λ_h − 1), "
         "s = 3λ_h/(2θ₀ − 3λ_h) — thrust is linear in the flow through the disk",
         "spec.inflow.inflow_thrust_ratio, spec.inflow.inflow_thrust_sensitivity",
         ("he2020", "cheeseman1955", "veismann2023", "nguyen2018"), (),
         ("properties/test_wake.py",),
         "Blade-element theory with constant pitch and chord, printed in four of the "
         "sources read (He & Leang Eq. 15, Cheeseman & Bennett Eq. 5–6, Veismann Eq. 2.4, "
         "Nguyen Eq. 40). It makes ONE inflow path out of effects the registry so far "
         "carried apart: ground effect (the ground cancels part of the induced velocity), "
         "descent and the vortex ring state (vrs_johnson_inflow), rear-rotor interference "
         "(rotor_wake_interference), another vehicle's downwash and vertical gusts (added "
         "to the axial flow). They add in the inflow; thrust ratios are never multiplied. "
         "The verified k_z channel of rotor_drag_hforce is this law with the induced "
         "velocity frozen: k_z = s·√(2ρA·c_T)/κ. s is identifiable four ways — blade pitch "
         "and solidity, a thrust stand with a ground board (C_a), the slope of a C_T(J) "
         "table at J = 0, or k_z from flight data. Measured C_a: 0.27–0.63. Valid while the "
         "blade is not stalled: with the inflow REDUCED (ground effect, the first part of a "
         "descent) the measured gain is about half the law's, and past a descent ratio of "
         "2 (flow up through the disk) the law overshoots thrust by 0.2–0.4 of hover. F-35: "
         "fitted to 18 tabulated descent curves (with vrs_johnson_inflow) s has median 0.44, "
         "IQR 0.36–0.67 — inside the ground-effect C_a range, so one s serves both; s from "
         "geometric pitch does not (0.4–3.3 on cambered low-pitch APC propellers, which have "
         "a negative zero-lift angle). Identify s from data, not from the pitch. F-36: a "
         "whole quadcopter's ground effect fits s = 0.46 (12×5) and 0.58 (10×4.7) — the "
         "ground and descent values agree on another lab's propellers. F-37: NASA MTB2 rear "
         "rotors in a front wake fit s = 0.50 (0.37–0.56 by layout). To verify: a test "
         "that varies the inflow at constant speed by a known amount (a descent test, or "
         "a measured external wake).",
         evidence=("analytic",),
         decision="pursue",
         effect=("RACER_5IN: the backend's k_z is s = 0.17 frozen, 0.41 with momentum inflow; "
                 "BEM's own slope gives 0.76; the 5×3 propeller measured 0.53. With s = 0.5: "
                 "+8.8 % thrust at z = R, and −9 % (Johnson's constants) to −22 % (measured "
                 "5×4) at a descent ratio of 1.2–1.4 where the backend gives +21 %")),
    Term("vrs_johnson_inflow", "candidate", "rotor_aero",
         "Mean induced velocity through the vortex ring state: momentum theory with the "
         "autorotation singularity bridged (baseline) plus an increment that puts a minimum "
         "of the net inflow at V_z = −0.45·v_h and a maximum at −1.5·v_h, closing at "
         "V_x = 0.95·v_h; first-order lag 4.4·R/v_h",
         "spec.descent.johnson_vrs_inflow_axial, spec.descent.johnson_vrs_increment, "
         "spec.descent.johnson_baseline_cubic, spec.descent.johnson_transition_points, "
         "spec.descent.momentum_inflow_residual, spec.descent.momentum_inflow_slope, "
         "spec.descent.momentum_inflow_axial, spec.descent.johnson_inflow_time_constant",
         ("johnson2005", "veismann2023", "veismann2022", "shetty2011", "shetty2011thesis",
          "uiucprop", "foster2017", "neurobem"), (),
         ("properties/test_wake.py",),
         "Johnson's parametric model, implemented from Table 3 with the constants of Table "
         "4; reproduces the report's Fig. 37 (V_x = 0.75: maximum 0.85 near −1.1, minimum "
         "0.58 near −0.55). Two misprints in Table 3 are read as the body text has them "
         "(the E end point; the 3·d·V_z² slope term). Finding F-32 is why it is here: "
         "between N and X the net inflow RISES with descent rate, so through "
         "inflow_thrust_law a fixed-pitch rotor loses thrust as it sinks faster — negative "
         "heave damping. The quartic (vrs_empirical_inflow) cannot do that; use ONE of the "
         "two. Against measured thrust at constant speed (tools/wake_triage.py): with "
         "Table 4 and s from blade geometry the dip has the right sign and place and about "
         "half the depth (0.85–0.91 against 0.70–0.78); with (V+v)_X raised from 1.25 to "
         "1.54–1.57 it matches two HQProp rotors to 0.04 rms up to a descent ratio of 1.75, "
         "minimum at 1.26–1.32 against 1.2 measured. Not every propeller follows: an APC "
         "10×5 dips earlier (1.07), has no initial rise and recovers faster; high-pitch "
         "propellers barely dip. So (V+v)_X — or f — is a per-propeller parameter, like "
         "c_T; Table 4's 1.25 comes from helicopter flight dynamics. The forward-speed "
         "dependence (band closed at V_x = 0.95·v_h) agrees with an isolated 14 in rotor "
         "(Foster & Hartman Figs. 8–9: loss and fluctuation gone by V_x/v_h ≈ 1). v_h is "
         "that of the CURRENT thrust, so the band moves with throttle. Non-smooth: "
         "piecewise cubics, C¹ at D, N, X, value-only at E and B, and a step of up to "
         "0.15·v_h where V_x crosses 0.95·v_h. The lag τ = 0.7/λ_h revolutions is a "
         "helicopter identification extrapolated by R/v_h; nothing at small scale tests "
         "it. Free-vortex and vorticity-transport simulations put full VRS development at "
         "20–25 revolutions, so a short pass through the band meets less than the "
         "quasi-steady loss. F-35 (tools/descent_data.py, Shetty's thesis tables, 18 APC "
         "propellers): the measured minimum sits at a descent ratio of 0.87–1.24 and Table 4 "
         "puts it 0.15–0.42 later on every propeller that dips. With three parameters per propeller — "
         "s, a scale k on the points D, N, X, E, and (V+v)_X — every curve fits to ≤ 0.09 "
         "rms (median 0.03). Held out (one propeller left out, (k, peak) from the others, s "
         "fitted): median 0.055 rms against 0.118 for the backend's linear law with its slope "
         "fitted; better on 16 of 18 (the two exceptions are high-pitch propellers that barely "
         "dip). But k does not transfer between labs: 0.69–0.83 for the UIUC set, 0.93–0.97 "
         "for Veismann's FPV rotors, and the UIUC constants on those do no better than the "
         "backend. So k and (V+v)_X are propeller-family parameters, like c_T; for 5 in FPV "
         "propellers the FPV set gives k ≈ 0.95, (V+v)_X ≈ 1.4–1.8. NeuroBEM flight data "
         "reach a descent ratio of 0.8 near-axially and show a 0.04 dip at 0.4–0.5 — the band "
         "onset, not decisive for k. To verify for a vehicle: a thrust curve of ITS propeller "
         "family at constant speed (the UIUC tables verify the form for the APC family only).",
         evidence=("analytic", "derived", "published"),
         decision="pursue",
         effect=("axial descent at 0.5 / 2 v_h, constant speed, s = 0.5: T/T_static 1.06 / "
                 "about 1.2 (the law overshoots past 2), and 0.91 at the minimum near 1.4 — "
                 "0.78 measured on a 5×4 — where the backend gives 1.21: an error of 30–45 % "
                 "of thrust and the wrong sign of heave damping from 0.45 to 1.5 v_h. The "
                 "band is 3.5–11.6 m/s of descent along the thrust axis on RACER_5IN at hover "
                 "thrust, 7.4–24.5 m/s at full thrust, 1.0–3.2 m/s at idle")),

    Term("bem_tpp_wrench", "verified", "rotor_aero",
         "Per-rotor force/torque from tip-path-plane tilt: f = Rz(χ)·(−(H+T·sin a1), "
         "s·T·sin b1, T·cos a0); τ = Rz(χ)·(−s·k_β·b1, −k_β·a1, −s·Q) + r×f",
         "spec.bem.tpp_rotor_force, spec.bem.tpp_rotor_torque",
         ("neurobem", "agilicious", "hoffmann2007"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib: composition replayed from "
         "recorded (T, Q, H, a0, a1, b1) through to the executed acceleration/omega-dot "
         "contributions (live flapping angles in all 12 cases, spring moments, 0.9575 "
         "z-obstruction). χ = atan2(v_y, v_x) aligns H with the in-plane hub velocity (drag, rearward); "
         "hinge-spring moments k_β per Hoffmann's hinged-blade model. Flapping angles "
         "(a0, a1, b1) are INPUTS: the reference's machine-generated vehicle-specific "
         "rational fits are rejected (REFERENCES.md) — general closures per Prouty pp. 463 "
         "remain future work. Reduces to (0,0,T) / −s·Q·ẑ at zero flapping and H. The "
         "executed reference also scales the collective z-force by 0.9575 (frame "
         "obstruction) — an assembly-level identified constant.",
         evidence=("executed",),
         decision="defer",
         revisit=("a consumer task needs body-rate damping above ~50 rad/s (over ~1 % of training "
                  "or flight steps), or sustained axial descent through the vortex-ring band "
                  "(0.5–2 v_h)"),
         effect=("F-29, 5-inch racer at hover (tools/bem_rate_damping.py): roll damping 0.00086 "
                 "/ 0.00132 / 0.00220 / 0.00402 N·m·s/rad at 1 / 10 / 50 / 200 rad/s; the "
                 "backend's linear k_z lever-arm channel, set to BEM at 13 rad/s, gives 1.69× / "
                 "1.10× / 0.66× / 0.36× of that. Cost: a Brent inflow solve and disk quadrature "
                 "per rotor per step")),

    Term("per_axis_quadratic_drag", "verified", "frame_aero",
         "F_k = −k_Q,k·v_a,k·|v_a,k| per body axis (k_Q = ½ρ·c_k·A_k physical packing)",
         "spec.rotor_aero.per_axis_quadratic_drag", ("agilicious", "skydreamer",
                                                     "flightning"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py",
          "properties/test_golden_flightning.py"),
         "Verified 2026-08-19 against the EXECUTED agilib ModelBodyDrag "
         "(golden/vectors/agilicious_simple_models.json) and the EXECUTED flightning "
         "compute_drag_force (same ½ρ·c_k·A_k packing, correctly divided by mass — no F-19 "
         "analog; its ±50% coefficient randomization is harness-side, replayed as effective "
         "params). Per-axis |v|·v form (SkyDreamer convention), NOT parasitic_drag's ‖v‖·v — "
         "don't mix coefficients. vertical_climb_drag is its z-restriction: enable one, not "
         "both. ⚠ agilib's ModelBodyDrag adds the force to the acceleration slot without "
         "dividing by mass (finding F-19); vectors pin the force expression.",
         evidence=("executed",)),

    Term("cubic_axis_drag", "verified", "frame_aero",
         "F_k = −k_C,k·v_a,k³ per body axis — cubic companion of linear_drag (PolyFit model)",
         "spec.rotor_aero.cubic_drag", ("agilicious", "neurobem"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib ModelLinCubDrag. "
         "The NeuroBEM 'PolyFit' baseline is linear_drag + this + translational_lift "
         "(agilib ModelLinCubDragIndLift, induced_lift_coeff ≡ k_h). Smooth odd polynomial.",
         evidence=("executed",)),

    # ---------------- candidates: motor / battery electrical ----------------
    Term("dc_motor_quasistatic", "verified", "actuator",
         "J_r·Ω̇ = (K_q/R_a)(V_m − K_e·Ω) − k_m·Ω² − b·Ω; τ_m bridge via linearization",
         "spec.motor_electrical.dc_motor_speed_dynamics", ("bangura", "jsbsim"), (),
         ("properties/test_candidates.py", "properties/test_golden_jsbsim.py"),
         "τ_m = J_r/(K_qK_e/R_a + b + 2k_mΩ₀) connects to the verified first-order lag. "
         "Verified 2026-08-18 against executed JSBSim FGBrushLessDCMotor + FGPropeller "
         "(golden/vectors/jsbsim_prop_bldc.json): static spin-ups at three throttles, "
         "Euler-replayed through the reference's own discretization at 1e-7, with "
         "K_q = K_e = 60/(2πKv) (Drela) and k_m = C_P(0)ρD⁵/(8π³) from the APC table. "
         "The b·Ω viscous term and the I0 friction deadband are unexercised (b = 0, "
         "I0 = 0 in the rig; the deadband is not a spec term).",
         evidence=("executed",)),
    Term("esc_battery_coupling", "candidate", "actuator",
         "V_m = u·V_batt; battery-coupled steady-state speed (√-like in u·V_batt)",
         "spec.motor_electrical.esc_mean_voltage, spec.motor_electrical.steady_state_speed",
         ("bangura", "crazyflie_fw"), (),
         ("properties/test_candidates.py",),
         "The physical origin of the verified throttle curve's √(k·u²+(1−k)·u) shape."),
    Term("crazyflie_battery_compensation", "candidate", "actuator",
         "Firmware cubic T(v_m) with Cardano inversion (+ legacy quadratic), PWM/V_batt scaling",
         "spec.motor_electrical.crazyflie_thrust_from_voltage", ("crazyflie_fw",), (),
         ("properties/test_candidates.py",),
         "Identified C0–C3 per prop variant; thrust held constant under sag by design."),
    Term("thevenin_battery", "candidate", "actuator",
         "OCV(SoC) + series R(SoC) + two RC branches; Gazebo linear model as special case",
         "spec.motor_electrical.thevenin_battery, spec.motor_electrical.chen_ocv",
         ("chen2006", "gazebo_battery"), (),
         ("properties/test_candidates.py",),
         "Load coupling i_batt = Σ u·(u·V_batt − K_e·Ω)/R_a (duty-reflected motor current, "
         "ideal-inverter power balance) closes the sag loop physically."),

    # ---------------- candidates: wind / turbulence ----------------
    Term("dryden_turbulence", "verified", "environment",
         "Dryden forming filters H_u/H_v/H_w + low-altitude scale/intensity closures",
         "spec.wind.dryden_filter_u, spec.wind.dryden_filter_vw, "
         "spec.wind.dryden_low_altitude_scales", ("mil8785c", "cr206937"), (),
         ("properties/test_candidates.py", "properties/test_dryden_authenticity.py",
          "properties/test_golden_jsbsim.py"),
         "⚠ discrete driving noise must be N(0, π/dt) for the published gains; 8785C vs "
         "1797 length-scale factor-of-2 trap. Low-altitude fit is in FEET. Verified via "
         "the ARCHAIC-SOURCE EXCEPTION (2026-08-11): the golden data are NASA "
         "CR-1998-206937's published run statistics (pinned PDF, transcribed tables + "
         "listing), not vectors from executed code — the reference is ACSL (no runnable "
         "interpreter) with an irreproducible RNG. Proven: the report's Tustin difference "
         "equations are exactly the prewarped bilinear of these filters, and re-simulating "
         "them reproduces Tables 2-7 calibration (σ, u/v/w) within the published spread at "
         "L = 1750 ft, V ∈ {100, 1000} ft/s. UPGRADE 2026-08-18: the low-altitude "
         "closures (L(h), σ(h, W20), h < 1000 ft) are now ALSO pinned by executed-code "
         "vectors — JSBSim FGWinds ttTustin implements the same CR-206937 difference "
         "equations with the closures active, and its seeded runs are reproduced "
         "sample-exactly via recovered driving noise "
         "(golden/vectors/jsbsim_dryden_lowalt.json).",
         evidence=("executed", "published")),
    Term("von_karman_turbulence", "candidate", "environment",
         "von Kármán spectra (5/6, 11/6 exponents) + standard rational filter approximations",
         "spec.wind.von_karman_psd_u", ("mil8785c",), (),
         ("properties/test_candidates.py",),
         "Measurement-preferred; no exact finite filter."),
    Term("discrete_gust", "candidate", "environment",
         "1-cosine discrete gust ramp per axis",
         "spec.wind.one_minus_cosine_gust", ("mil8785c",), (),
         ("properties/test_candidates.py",)),
    Term("wind_shear_log", "candidate", "environment",
         "MIL-F-8785C mean-wind log profile u_w = W20·ln(h/z0)/ln(20/z0) (h, z0 in ft)",
         "spec.wind.log_wind_shear", ("mil8785c", "mathworks_aeroblks"), (),
         ("properties/test_candidates.py",),
         "The deterministic member of the 8785C wind triad (shear + turbulence + gust); "
         "superposes onto v_wind. Valid 3–1000 ft AGL; z0 = 0.15 ft (Category C landing) "
         "or 2.0 ft (otherwise). Anchored to the same W20 as the Dryden closures, so mean "
         "wind and turbulence intensity stay mutually calibrated."),

    # ---------------- candidates: discretization / integration ----------------
    Term("semi_implicit_euler", "verified", "discretization",
         "Symplectic (semi-implicit) Euler: velocities first with f(s), then positions with "
         "f evaluated at the velocity-updated state",
         "spec.discretization.semi_implicit_euler_step", ("agilicious", "neurobem"), (),
         ("properties/test_bem.py", "properties/test_golden_agilicious.py"),
         "Verified 2026-08-19 against the EXECUTED agilib IntegratorSymplecticEuler. "
         "NeuroBEM's evaluation integrator (1 ms steps, §IV-D): first-order like explicit "
         "Euler but symplectic on the mechanical part — bounded energy error instead of "
         "drift. agilib groups (v, ω, Ω) as velocities and (x, q) as positions; the "
         "quaternion advances with the NEW ω against the OLD q. Single smooth composition, "
         "cleanly differentiable.",
         evidence=("executed",)),

    Term("quaternion_norm_correction", "candidate", "discretization",
         "q̇ = ½ q ⊗ (0, ω) + K·(1−‖q‖²)·q — smooth Lagrange-style norm stabilization",
         "spec.quaternion.kinematics_norm_corrected", ("mathworks_aeroblks",), (),
         ("properties/test_candidates.py",),
         "Differentiable alternative to the harness's post-step renormalization: the "
         "correction lives inside the ODE (single smooth vector field for backends that "
         "differentiate through the integrator). d‖q‖²/dt = 2K·ε·‖q‖², ε = 1−‖q‖² → norm "
         "error decays at rate ≈ 2K; exactly quaternion.kinematics on the unit manifold. "
         "Textbook basis: Stevens & Lewis; Zipfel. Choose K·dt ≪ 1."),

    # ---------------- candidates: sensors ----------------
    Term("imu_stochastic_errors", "verified", "sensor",
         "Per-axis IMU corruption: exact-discretized Gauss–Markov bias + white noise + turn-on bias",
         "spec.sensors.imu_bias_gauss_markov_step, spec.sensors.imu_corrupt",
         ("rotors_px4", "l2f"), (),
         ("properties/test_candidates.py", "properties/test_golden_l2f.py",
          "properties/test_backend_jax.py"),
         "b⁺ = e^(−dt/τ)·b + σ_b·√(τ/2·(1−e^(−2dt/τ)))·w; y = y_true + b + b_on + (σ_nd/√dt)·n. "
         "Exact OU discretization (Maybeck Vol. 1 Eq. 4-114, as cited in gazebo_imu_plugin.cpp; "
         "book not directly consulted — exactness proven symbolically instead). Unit-normal "
         "draws and the power-up b_on draw are harness-side, mirroring the Dryden noise split. "
         "RotorS ships 2× the ADIS16448 datasheet values as defaults and initializes b at zero "
         "rather than the stationary σ_b²τ/2 (their own TODO). Verified 2026-11-15 "
         "against the EXECUTED rl-tools L2F (golden/vectors/l2f_imu.json): "
         "bias_process_step reproduced at 1e-12 for τ = 0.05 / 2 / 300 s, the τ ≤ 0 "
         "random-walk branch (this step's τ → ∞ limit, proven symbolically) and the held "
         "bias; gyro and accelerometer rows of full rl_tools::step calls reproduced as "
         "imu_corrupt over the strapdown truth with σ_nd = std·√dt and b_on = the "
         "reference's constant offset. Driving noise is the reference's own, replayed on "
         "a copy of its generator — not recovered from the outputs. Size: with the RotorS "
         "ADIS16448 numbers the gyro bias moves 1.2e-4 rad/s over a 10 s episode, about "
         "1 % of the per-sample white noise at 1 kHz — minor, cheap and identifiable "
         "(Allan variance), so it is carried; it grows with episode length and with "
         "low-grade MEMS thermal drift. The backend emits 1 − e^(−2dt/τ) as "
         "−expm1(−2dt/τ): the literal form loses the driving noise in float32 once "
         "τ ≫ dt (test_backend_jax.py).",
         evidence=("executed",),
         use="backend"),

    # ---------------- proposed (known effects, no spec expression yet) ----------------
    Term("rotor_rate_damping_nonlinear", "proposed", "rotor_aero",
         "Rotor body-rate damping that stiffens with rate: D(p) from the rotor inflow "
         "response, in place of the constant lever-arm derivative 4·Ω·k_z·a² of the "
         "verified H-force",
         "", ("neurobem", "agilicious", "hoffmann2007"), (),
         ("properties/test_findings.py",),
         "Finding F-29 (REFERENCES.md). Routes: (a) put the verified BEM family in the "
         "backend (exact, costly); (b) a reduced term fitted to tools/bem_rate_damping.py, "
         "e.g. D(p) = D₀ + D₁·|p|. Either REPLACES the k_z lever-arm share of the damping — "
         "never add it on top. Shape caveat: across the sweep the descending-side rotors "
         "sit in the empirical VRS window (vrs_empirical_inflow) and at 200 rad/s the "
         "climbing-side rotors windmill (−0.33 N each), so the curve leans on the least "
         "certain part of BEM. Roll and pitch agree once the hub spring moment (fed by the "
         "vehicle-specific flapping fits) is zeroed (test_findings.py).",
         evidence=("derived",),
         decision="defer",
         revisit=("a consumer task needs body-rate damping above ~50 rad/s (over ~1 % of "
                  "training or flight steps), or a flip / freestyle task"),
         effect=("F-29, 5-inch racer at hover: BEM roll damping 0.00086 / 0.00132 / 0.00220 / "
                 "0.00402 N·m·s/rad at 1 / 10 / 50 / 200 rad/s. A linear term set to BEM at "
                 "13 rad/s (the NeuroBEM flight-data maximum) is 1.69× / 1.10× / 0.66× / "
                 "0.36× of BEM: within 10 % only near 10–13 rad/s, a third of BEM at 200")),
    Term("angular_body_drag", "proposed", "frame_aero",
         "Frame angular drag −diag(c_R)·(|ω|∘ω) − diag(c_r)·ω (MuJoCo inertia-based fluid "
         "model torque terms)",
         "", ("mujoco_fluid",), (),
         (),
         "REFERENCES.md listed this as 'ADOPTED (candidate tier)' but no spec function or "
         "registry term ever landed; recorded here as proposed (2026-10-04). MuJoCo's box "
         "formula g_D,i = −½ρ·r_i·(r_j⁴+r_k⁴)·|ω_i|ω_i, g_V,i = −8βπ·r_eq³·ω_i is not "
         "physical for an open frame: most rotational damping comes from the rotors, "
         "already in rotor_drag_hforce. Treat c_R, c_r as fitted residuals, not box values.",
         decision="defer",
         revisit=("a fit of the backend model to measured flight data leaves a rate-"
                  "proportional residual torque after the rotor terms"),
         effect=("box formula on the 5-inch racer (ρ 1.225, β 1.8e-5, J of the NeuroBEM "
                 "platform): 3 % / 10 % / 22 % of BEM's roll damping at 10 / 50 / 200 rad/s "
                 "— an upper bound, since the box overstates an open frame")),

    Term("propwash_thrust_unsteadiness", "candidate", "rotor_aero",
         "Unsteady per-rotor thrust when a rotor descends into its own wake (propwash): the "
         "VRS inflow increment fluctuates, Δ(V_z+v)·(1 + c_n·n(t)), independently on each "
         "rotor; published stand-in T·(1 + A·cos(ω·t + δ(t)))",
         "spec.descent.vrs_fluctuating_increment, spec.descent.vrs_thrust_modulation",
         ("veismann2023", "veismann2022", "shetty2011", "foster2017", "bogaerts2023",
          "johnson2005", "simitl"), (),
         ("properties/test_wake.py", "properties/test_findings.py"),
         "Measured, isolated rotors at constant speed: the relative rms of low-passed "
         "thrust rises near-linearly from < 0.02 at hover to 0.12–0.20 at a descent ratio "
         "of 1.2–1.3 and is back to 0.06–0.08 at 2 (Veismann 2023, 4–6 in); 0.18 at the "
         "minimum of an APC 10×5 (Shetty & Selig); 0.16–0.18 on a 14 in rotor at "
         "V_x/v_h ≈ 0.4–0.5, V_z/v_h ≈ −0.9 to −1.2, under 0.04 beyond V_x/v_h ≈ 1 (Foster "
         "& Hartman Fig. 9). Rotors with a larger mean loss fluctuate more: rms over mean "
         "loss 0.33–0.46. Hence the closure — noise on the VRS increment of "
         "vrs_johnson_inflow — which gets the footprint in (V_x, V_z), the scaling with "
         "the blade's s and the switch-off with forward speed from the mean model; it "
         "is DERIVED, and it under-states the measured rms at descent ratios of 0.5 and 2. "
         "Spectrum: isolated rotors show f·R/v_h ≈ 0.02–0.1 (1–3 Hz at 14 in; periods of "
         "20–50 revolutions in a water tank) — 2–12 Hz on RACER_5IN. On a flying 5 in "
         "Betaflight quadrotor the gyro spectrum centres at 13–22 Hz, the forcing is flat "
         "to 80 Hz and 2–8× the isolated-rotor level, and a closed-loop model driven by "
         "isolated-rotor forcing gets the frequencies but 'orders of magnitude' too "
         "little power (Bogaerts 2023) — so rotor-rotor or rotor-frame interaction adds "
         "forcing that no model here carries. Betaflight's documentation puts propwash at "
         "20–60 Hz. Rotor-to-rotor correlation is unmeasured (a free-flying quadrotor's "
         "attitude oscillation is 'explained by asynchronous thrust fluctuations', "
         "Veismann et al. 2021); torque fluctuation is small on large rotors (ΔQ/Q < 2.5 % "
         "against ΔT/T > 15 %) and unmeasured at small scale. Control authority is the "
         "other half of FPV propwash (low rotor speed: small thrust slope, slow "
         "deceleration, filter delay) and comes from the motor and controller models, not "
         "from this term. SimITL's stand-in (loss ∝ reverse-flow speed, 45 Hz noise, "
         "physics.cpp:516-552) grows without limit where the measurement peaks and "
         "recovers. ONE descent path: this multiplies nothing — it perturbs the inflow of "
         "the mean model. To verify: blackbox logs with rotor-speed telemetry through "
         "descents (10–80 Hz gyro power against descent ratio), or a load cell above "
         "500 Hz under a rotor in a vertical tunnel.",
         evidence=("derived",),
         decision="pursue",
         effect=("σ(T)/T̄ 0.13 at the worst descent rate (5×4): with independent rotors a roll "
                 "disturbance of 14 rad/s² rms on RACER_5IN at hover thrust and 63 rad/s² at "
                 "full thrust — 2 % and 10 % of roll authority — on top of the mean loss; "
                 "1 rad/s² at idle, where the band sits at 1.0–3.2 m/s (tools/wake_triage.py). "
                 "On-vehicle forcing was measured 2–8× the isolated-rotor level")),
    Term("rotor_wake_interference", "candidate", "rotor_aero",
         "Rear rotors in the front rotors' wake: extra downwash k·v_i at the rear disk, k "
         "from a horseshoe-vortex wake as a function of spacing, incidence and wake angle",
         "spec.inflow.tandem_interference_factor", ("nguyen2018", "russell2018", "mtb2"), (),
         ("properties/test_wake.py",),
         "Every rotor in the spec is aerodynamically isolated. Measured and computed "
         "rear-rotor thrust deficits near level flight: 11 % (wind tunnel, l/R = 2.5, "
         "6 m/s, 5° — this factor gives 0.86 against 0.89 measured at l/R = 2.83), 5–12 % "
         "(CFD, two in-line rotors, 20–60 kt), 13–18 % below the isolated expectation "
         "(CFD of a trimmed quadrotor at 5–10 m/s); the rear hub moment also falls by "
         "about half in CFD, which this factor does not model. Whole-vehicle evidence: on "
         "a 3DR SOLO at 12 m/s and 0 to −5° pitch a nose-up moment of 0.3–0.5 N·m remains "
         "after four isolated rotors and the airframe are subtracted, and nothing remains "
         "at −20 to −40° (search-derived from NASA/TM-2018-219758 tables). No interference "
         "in axial climb. Side-by-side rotors see a small upwash (favourable). The thrust "
         "change goes through inflow_thrust_law (ONE inflow path). Installation loss is a "
         "separate constant: full vehicle over four isolated rotors is 0.85–0.95 in hover "
         "(5–15 % download; a c_T identified in flight already contains it). F-37 (NASA "
         "MTB2 tables, 24.5 in rotors, hub spacing 2.69R): back rotor in line −6.5 / −28 % "
         "at 20 / 40 ft/s with the rotors at one height, −2 / −11 % raised 0.33R, 0 / −9 % "
         "raised 0.57R; the front rotor is unchanged (±1.5 %); the back rotor's torque "
         "rises 5–17 % as its thrust falls (not modelled). This factor (hub line as "
         "reference) with the descent s = 0.44 held out gives 0.051 rms over 48 conditions "
         "against 0.144 for no interference; fitted, s = 0.50 — no new constant. NOT "
         "verified: it gets the speed trend wrong (local s 0.04–0.29 at V/v_h ≈ 1.4, "
         "0.42–0.63 at 2.8), over-predicting slow near-level flight when the back rotor "
         "sits above a steep front wake. Blade phase moves mean thrust under 1 %.",
         evidence=("analytic",),
         decision="defer",
         revisit=("a consumer flies slow, near-level forward flight (5–12 m/s, under 15° of "
                  "tilt) and sees a pitch trim offset or a fore/aft rotor-speed split that "
                  "the hub moment (blade_flapping_moment) and the centre of gravity do not "
                  "explain"),
         effect=("RACER_5IN rear rotors, s = 0.5: −10 % thrust at 6 m/s and 5° nose-down, "
                 "−12 % at 10 m/s and 8°, −2.5 % at 20 m/s and 30°, −0.2 % at 35 m/s and "
                 "60° — material in slow level flight, gone at racing attitude "
                 "(tools/wake_triage.py). F-37 suggests the slowest cases are over-stated "
                 "(measured local s 0.04–0.29 at V/v_h ≈ 1.4 on MTB2)")),
    Term("rotor_gust_gradient", "proposed", "disturbance",
         "Per-rotor wind: each rotor's local airspeed uses the air velocity at its own hub, "
         "so a gust that differs across the airframe makes a torque",
         "", ("mil8785c", "simitl"), (),
         ("properties/test_findings.py",),
         "The spec takes ONE wind vector for the vehicle (v_a = Rᵀ·(v − v_wind)); gusts "
         "therefore make forces but no torques. The multirotor form of MIL-F-8785C's "
         "angular gusts p_g, q_g, r_g (skipped as wingspan-based, REFERENCES.md) is a "
         "per-rotor wind input v_wind,i feeding local_airspeed — no new coefficient, the "
         "existing k_d / k_z / flapping terms do the rest. libfpv implements it (patch "
         "0017). The spec change is small; the cost is harness-side: a gust generator "
         "correlated across the hub positions (the Dryden forming filters are one-point). "
         "Equivalent shortcut: subtract a gust rotation ω_g from ω in the ω×r_i term.",
         decision="defer",
         revisit=("a consumer flies within a few metres of the ground in wind above the MIL "
                  "'moderate' level (W20 ≥ 15 m/s), flies a vehicle several times the "
                  "racer's size, or adds a spatially correlated turbulence driver"),
         effect=("von Kármán transverse correlation at the MIL low-altitude scales, through "
                 "the k_z thrust channel (tools/sim_source_triage.py): rms roll disturbance "
                 "on RACER_5IN 0.6 / 0.9 / 1.8 / 2.7 rad/s² at 10 m and 1.0 / 1.5 / 3.1 / "
                 "4.6 rad/s² at 2 m for W20 = 5 m/s / light / moderate / severe; CRAZYFLIE "
                 "0.8–6.0 rad/s². Under 1 % of roll authority (≈ 620 rad/s²) throughout")),
    Term("rotor_imbalance_vibration", "candidate", "disturbance",
         "Rotor mass imbalance: a hub force U_i·Ω_i² turning with each rotor — the "
         "rotation-frequency lines in the accelerometer and gyro",
         "spec.wrench.rotor_imbalance_wrench", ("ghalamchi2020", "simitl"), (),
         ("properties/test_candidates.py", "properties/test_findings.py"),
         "F_i = U_i·Ω_i²·(cos θ_i, sin θ_i, 0) = (U_i/c_T)·T_i in the rotor plane "
         "(Ghalamchi eq. 4, read from the paper); accelerometer line U·Ω²/m (their eq. 5, "
         "measured in flight on 39 g, 690 g and 1170 g vehicles). The r_i × F_i moment is "
         "rigid-body mechanics added here — the paper models no moment and no gyro: a yaw "
         "line of rate amplitude |r|·U·Ω/J_zz, roll and pitch only through hub height. "
         "U = m_u·r_u comes from a prop balancer or from the hover accelerometer line. "
         "Not covered: frame flexibility and soft mounts (their estimate of one damaged "
         "prop doubled between two airframes), the 2× and 3× lines flight controllers "
         "also notch, blade-pass aerodynamic loads. SimITL instead adds gyro-side "
         "sinusoids with a hand-tuned Ω² law (physics.cpp:388-498; 0.18 rad/s per motor "
         "at full throttle, 36× its white noise) — against a rigid-body Ω law here. "
         "Needs the rotor phase θ_i as harness state and a step well below 1/Ω; "
         "SkyFlow's ROADMAP.md holds the matching firmware-side design (RPM filter in "
         "the loop).",
         decision="defer",
         revisit=("a consumer policy or estimator reads the raw gyro or accelerometer "
                  "instead of the flight controller's notch-filtered output, the RPM "
                  "filter enters the firmware loop, or a sim-to-real gap is traced to "
                  "rotor-harmonic noise"),
         effect=("RACER_5IN, 10 mg at one blade tip (U = 6.5e-7 kg·m; the paper's damaged "
                 "props carry 5 mg at 32 mm and 40 mg at 100 mm): accelerometer line "
                 "1.0 m/s² at hover and 4.5 m/s² at full thrust (10 % and 46 % of g), yaw "
                 "gyro line 0.021 rad/s at hover, at 174 Hz — above the 5 % threshold on "
                 "the accelerometer; zero mean, so no effect on the mean flight path")),

    # ---------------- harness (tracked, not physics) ----------------
    Term("command_transport_delay", "verified", "harness",
         "u_applied(t) = u_cmd(t − t_d); ring buffer of round(t_d/dt) steps",
         "(harness — no symbolic form)", ("skydreamer", "eschmann2024"), (),
         (), "SkyDreamer trains with t_d = 11 ms."),
    Term("control_rate_zoh", "verified", "harness",
         "Controller decides at a lower rate than physics; command zero-order-held between",
         "(harness — no symbolic form)", ("crazyflow",), ()),
    Term("ground_contact_heuristic", "candidate", "harness",
         "Normal-force cancellation + velocity clamps at z ≤ 0 — bookkeeping, not contact physics",
         "(harness — no symbolic form)", ("rotorpy",), (),
         (), "Do not port as physics; a real contact model would be a new candidate."),
)


#: Sources reviewed and rejected, so intake doesn't re-litigate them.
EXCLUSIONS = (
    ("genesis", ("Genesis (Genesis-Embodied-AI): props are fixed joints, KF·rpm² force + "
                 "KM·rpm² yaw torque only, no gyroscopic effects — strict subset of this spec.")),
)


def by_key(key: str) -> Term:
    for t in TERMS:
        if t.key == key:
            return t
    raise KeyError(key)


def validate_registry() -> None:
    """Structural invariants checked by properties/test_registry.py."""
    from skyflow_dynamics.spec.parameters import SCHEMA
    keys = [t.key for t in TERMS]
    assert len(keys) == len(set(keys)), "duplicate term keys"
    domains = ("rigid_body", "actuator", "rotor_aero", "frame_aero", "sensor", "disturbance",
               "discretization", "differentiation", "environment", "harness")
    for t in TERMS:
        assert t.tier in TIERS, t.key
        assert (t.expression == "") == (t.tier == "proposed"), \
            f"{t.key}: proposed terms, and only they, have no expression"
        assert set(t.evidence) <= set(EVIDENCE), f"{t.key}: unknown evidence {t.evidence}"
        assert t.use in USES, f"{t.key}: unknown use {t.use}"
        assert t.use == "spec" or t.tier == "verified", \
            f"{t.key}: only verified terms go into a backend"
        if t.tier == "verified" and t.domain != "harness":
            assert set(t.evidence) & set(INDEPENDENT_EVIDENCE), \
                f"{t.key} is verified without independent evidence"
        if "derived" in t.evidence:
            assert t.tests, f"{t.key}: derived evidence needs a test that reproduces it"
        if t.ledger:
            assert t.decision in DECISIONS, f"{t.key}: unknown decision {t.decision}"
            if t.decision != "open":
                assert t.effect, f"{t.key}: a triaged term records its effect size"
            if t.decision == "defer":
                assert t.revisit, f"{t.key}: a deferred term needs a revisit condition"
        else:
            assert t.decision == "open" and not (t.revisit or t.effect), \
                f"{t.key}: final and harness terms carry no next-step decision"
        assert t.domain in domains, f"{t.key}: unknown domain {t.domain}"
        assert t.sources, f"{t.key} has no sources"
        for s in t.sources:
            assert s in SOURCES, f"{t.key}: unknown source {s}"
        for p in t.parameters:
            assert p in SCHEMA, f"{t.key}: unknown parameter {p}"
        if t.tier == "verified" and t.domain not in ("harness",):
            assert t.tests or t.expression.startswith("(harness"), \
                f"{t.key} is verified but lists no tests"
