"""
Golden-vector generator: agilicious agilib (NeuroBEM BEM rotor + simple models), EXECUTED.

Builds a small C++ driver against the UNMODIFIED agilib sources (GPLv3), runs the actual
ModelPropellerBEM / ModelMotor / ModelThrustTorqueSimple / ModelRigidBody / ModelBodyDrag /
ModelLinCubDrag models and the Euler / symplectic-Euler / RK4 integrators, and freezes every
input, internal, and derivative at full precision. Before writing anything, a float-exact
Python replica of every executed code path (15-point Gauss-Kronrod disk quadrature, the
vectorized Brent solver, the float32 fast-atan2, the VRS branch with its ANY-rotor gate, the
machine-generated flapping fits, and the force/torque composition) is asserted against the
executed outputs at ≤1e-10 relative — the transcription self-check demanded by INTAKE.md
step 6. The replica also reconstructs quantities the reference overwrites (the pre-VRS
momentum-closure root), which are stored with that provenance note.

Reference provenance: public GPLv3 mirror of the RPG agilicious repository. The BEM/model
sources are byte-identical to the RPG init commit 2d78b81 (Philipp Foehn, 2022-06-22); the
mirror owner's later commits never touch them. Every compiled physics-bearing file is
pinned by sha256 below and re-verified before each build.

Usage (from the repo root):

    uv run python golden/generate/gen_agilicious.py --out golden/vectors \
        [--repo /path/to/agilicious_internal_mine] [--eigen /path/to/eigen-3.4.0]

Without --repo the mirror is cloned to a temp dir and checked out at the pinned commit;
without --eigen the Eigen 3.4.0 headers (build-time dependency of agilib's types) are
downloaded. Requires g++ (C++17). Nothing from agilib is redistributed: the vectors hold
only numeric inputs/outputs plus provenance.
"""

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

import numpy as np

MIRROR_URL = "https://github.com/alibabasomeone/agilicious_internal_mine"
PINNED_COMMIT = "ba8caa708a570954dcb906fc676ae6b9d99be1a2"
BEM_UPSTREAM_COMMIT = "2d78b81 (RPG init commit, Philipp Foehn, 2022-06-22)"
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
CXX_FLAGS = ["-O2", "-std=c++17", "-ffp-contract=off", "-Wno-unknown-pragmas"]

#: sha256 (first 16 hex chars) of every physics-bearing file that gets compiled — verified
#: against the checkout before building; a mismatch aborts generation.
PINNED_SHA256_16 = {
    "agilib/src/simulator/model_propeller_bem.cpp": "aff3b58e2a68caa0",
    "agilib/src/simulator/model_propeller_bem_params.cpp": "d779f1c0e4207e45",
    "agilib/src/simulator/bem/functions.cpp": "af0fa773e0c6f577",
    "agilib/src/simulator/bem/propeller_state.cpp": "766c81861f3bced9",
    "agilib/include/agilib/simulator/bem/brent.hpp": "b5c2518764eeff52",
    "agilib/include/agilib/simulator/bem/gauss_kronrod.hpp": "2b2dda8927d994ad",
    "agilib/include/agilib/math/fast_atan2.hpp": "8f5e0b0def23431f",
    "agilib/src/simulator/model_motor.cpp": "5be803ed86a8d6f4",
    "agilib/src/simulator/model_rigid_body.cpp": "3c898f456ccaf6da",
    "agilib/src/simulator/model_thrust_torque_simple.cpp": "b01c44e9aa1f5d49",
    "agilib/src/simulator/model_body_drag.cpp": "d7a7b2594eaeb9fe",
    "agilib/src/simulator/model_lin_cub_drag.cpp": "63c368fab7fbfda0",
    "agilib/src/math/integrator_symplectic_euler.cpp": "6f01b68acf7f4654",
    "agilib/src/math/integrator_euler.cpp": "5e18ca4ed532120b",
    "agilib/src/math/integrator_rk4.cpp": "0ce527c149242e4f",
    "agilib/src/math/math.cpp": "a19046103c292c0d",
    "agilib/src/types/quadrotor.cpp": "7ed7dd98a0a0db7b",
}

#: agilib translation units compiled into the driver (support files not sha-pinned above
#: are structural: base classes, loggers, state layout).
AGILIB_TUS = [
    "src/simulator/model_base.cpp",
    "src/simulator/model_propeller_bem.cpp",
    "src/simulator/model_propeller_bem_params.cpp",
    "src/simulator/bem/functions.cpp",
    "src/simulator/bem/propeller_state.cpp",
    "src/simulator/model_motor.cpp",
    "src/simulator/model_rigid_body.cpp",
    "src/simulator/model_thrust_torque_simple.cpp",
    "src/simulator/model_body_drag.cpp",
    "src/simulator/model_body_drag_params.cpp",
    "src/simulator/model_lin_cub_drag.cpp",
    "src/simulator/model_lin_cub_drag_params.cpp",
    "src/simulator/model_init.cpp",
    "src/math/math.cpp",
    "src/math/integrator_base.cpp",
    "src/math/integrator_euler.cpp",
    "src/math/integrator_rk4.cpp",
    "src/math/integrator_symplectic_euler.cpp",
    "src/types/quadrotor.cpp",
    "src/types/quad_state.cpp",
    "src/utils/logger.cpp",
    "src/utils/timer.cpp",
    "src/base/parameter_base.cpp",
]

#: Canonical physical spin signs about +z_B (FLU) for the driver's rotor order
#: (fr, bl, br, fl): the reference's clockwise flag is cw = (i >= 2 ? +1 : −1) = −spin.
SPIN = [1, 1, -1, -1]

# state-vector block offsets (agilib QuadState::IDX)
POS, ATT, VEL, OME, MOT, MOTDES, SIZE = 0, 3, 7, 10, 31, 35, 39

DRIVER_CPP = r'''
// SkyFlow-Dynamics golden-vector driver for agilicious agilib (GPLv3 reference).
// Executes the UNMODIFIED agilib simulator models and prints every input, internal, and
// derivative at full precision as JSON. The private→public define is a driver-local access
// trick (layout-identical); the compiled agilib translation units are untouched.

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Eigen>

#define private public
#include "agilib/math/integrator_euler.hpp"
#include "agilib/math/integrator_rk4.hpp"
#include "agilib/math/integrator_symplectic_euler.hpp"
#include "agilib/math/math.hpp"
#include "agilib/simulator/model_body_drag.hpp"
#include "agilib/simulator/model_init.hpp"
#include "agilib/simulator/model_lin_cub_drag.hpp"
#include "agilib/simulator/model_motor.hpp"
#include "agilib/simulator/model_propeller_bem.hpp"
#include "agilib/simulator/model_rigid_body.hpp"
#include "agilib/simulator/model_thrust_torque_simple.hpp"
#include "agilib/types/quad_state.hpp"
#include "agilib/types/quadrotor.hpp"
#undef private

using namespace agi;

static void pd(const char* key, double v, bool comma = true) {
  std::printf("\"%s\": %.17g%s", key, v, comma ? ", " : "");
}

template <typename V>
static void pvec(const char* key, const V& v, bool comma = true) {
  std::printf("\"%s\": [", key);
  for (int i = 0; i < v.size(); ++i)
    std::printf("%.17g%s", (double)v(i), i + 1 < v.size() ? ", " : "");
  std::printf("]%s", comma ? ", " : "");
}

// 3x4 per-rotor matrix as list of 4 column triples
static void pmat34(const char* key, const Matrix<3, 4>& m, bool comma = true) {
  std::printf("\"%s\": [", key);
  for (int c = 0; c < 4; ++c)
    std::printf("[%.17g, %.17g, %.17g]%s", m(0, c), m(1, c), m(2, c),
                c < 3 ? ", " : "");
  std::printf("]%s", comma ? ", " : "");
}

static Vector<QS::SIZE> make_state(const Vector<3>& v, const Vector<4>& q,
                                   const Vector<3>& w, const Vector<4>& mot,
                                   const Vector<4>& motdes) {
  Vector<QS::SIZE> s = Vector<QS::SIZE>::Zero();
  s.segment<QS::NATT>(QS::ATT) = q.normalized();
  s.segment<QS::NVEL>(QS::VEL) = v;
  s.segment<QS::NOME>(QS::OME) = w;
  s.segment<QS::NMOT>(QS::MOT) = mot;
  s.segment<QS::NMOTDES>(QS::MOTDES) = motdes;
  return s;
}

struct Case {
  Vector<3> v, w;
  Vector<4> q, mot, motdes;
};

int main() {
  // Quadrotor: Kingfisher-like inputs (recorded verbatim in the vectors — the physics
  // pinning does not depend on their provenance, plausibility keeps regimes physical).
  Quadrotor quad(0.752, 0.15);
  quad.J_ = Vector<3>(2.5e-3, 2.1e-3, 4.3e-3).asDiagonal();
  quad.J_inv_ = quad.J_.inverse();
  quad.motor_omega_min_ = 150.0;
  quad.motor_omega_max_ = 4000.0;
  quad.motor_tau_inv_ = 1.0 / 0.033;
  quad.thrust_max_ = 4000.0 * 4000.0 * quad.thrust_map_(0);

  BEMParameters bem;  // defaults == sim_kingfisher.yaml values

  BodyDragParameters bdrag;
  bdrag.cxy_ = 1.04;
  bdrag.cz_ = 1.04;
  bdrag.ax_ = 1.5e-2;
  bdrag.ay_ = 1.5e-2;
  bdrag.az_ = 3.0e-2;
  bdrag.rho_ = 1.204;

  LinCubDragParameters lcd;
  lcd.lin_drag_coeff_ = Vector<3>(0.30, 0.35, 0.40);
  lcd.cub_drag_coeff_ = Vector<3>(0.010, 0.012, 0.014);
  lcd.induced_lift_coeff_ = 0.05;

  std::vector<Case> bem_cases = {
    // hover
    {{0, 0, 0}, {0, 0, 0}, {1, 0, 0, 0}, {1800, 1850, 1900, 1950}, {1800, 1850, 1900, 1950}},
    // hover, strongly asymmetric motor speeds
    {{0, 0, 0}, {0, 0, 0}, {1, 0, 0, 0}, {1200, 2400, 1600, 2000}, {1200, 2400, 1600, 2000}},
    // slow forward
    {{3, 0, 0}, {0, 0, 0}, {1, 0, 0, 0}, {1800, 1800, 1800, 1800}, {1800, 1800, 1800, 1800}},
    // fast forward
    {{15, 0, 0.5}, {0, 0, 0}, {1, 0, 0, 0}, {1700, 1750, 1800, 1850}, {1700, 1750, 1800, 1850}},
    // racing speed, pitched attitude, body rates (flapping-fit inputs live)
    {{18, -2, 1}, {0.5, -0.3, 0.2}, {0.98481, 0, -0.17365, 0}, {2000, 2100, 2200, 2300},
     {2000, 2100, 2200, 2300}},
    // climb
    {{0, 0, 6}, {0, 0, 0}, {1, 0, 0, 0}, {2000, 2000, 2000, 2000}, {2000, 2000, 2000, 2000}},
    // shallow descent (VRS gate active at small v_ver/v_i)
    {{0, 0, -1}, {0, 0, 0}, {1, 0, 0, 0}, {1800, 1800, 1800, 1800}, {1800, 1800, 1800, 1800}},
    // deep descent inside the VRS band
    {{0, 0, -8}, {0, 0, 0}, {1, 0, 0, 0}, {1700, 1750, 1800, 1850}, {1700, 1750, 1800, 1850}},
    // oblique descent
    {{4, 0, -5}, {0, 0, 0}, {1, 0, 0, 0}, {1800, 1800, 1800, 1800}, {1800, 1800, 1800, 1800}},
    // fast descent at the upper edge of the VRS band (post-correction ratios ~1.0-1.1)
    {{0, 0, -25}, {0, 0, 0}, {1, 0, 0, 0}, {1600, 1650, 1700, 1750}, {1600, 1650, 1700, 1750}},
    // mixed per-rotor regimes: roll rate + lateral/descending flow + tilt (ANY-gate case)
    {{0, 3, -3}, {3, 0, 0}, {0.96593, 0.25882, 0, 0}, {1500, 2500, 1800, 2200},
     {1500, 2500, 1800, 2200}},
    // lateral flight with yaw rate
    {{0, 10, 0}, {0, 0, 2}, {1, 0, 0, 0}, {1900, 1700, 2100, 1500}, {1900, 1700, 2100, 1500}},
  };

  std::printf("{\n\"quad\": {");
  pd("mass", quad.m_);
  pvec("J_diag", Vector<3>(quad.J_.diagonal()));
  pmat34("t_BM", quad.t_BM_);
  pd("motor_tau", 1.0 / quad.motor_tau_inv_);
  pvec("thrust_map", quad.thrust_map_);
  pvec("torque_map", quad.torque_map_);
  pd("kappa", quad.kappa_);
  pd("motor_omega_min", quad.motor_omega_min_);
  pd("motor_omega_max", quad.motor_omega_max_);
  pd("thrust_min", quad.thrust_min_);
  pd("thrust_max", quad.thrust_max_);
  pd("G", G, false);
  std::printf("},\n\"bem_params\": {");
  pd("rho", bem.rho_);
  pd("r_prop", bem.r_prop_);
  pd("prop_area", bem.prop_area_);
  pd("theta0", bem.theta0_);
  pd("theta1", bem.theta1_);
  pd("chord_inner", bem.ci_);
  pd("chord_outer", bem.co_);
  pd("num_blades", bem.b_);
  pd("cl0", bem.cl_);
  pd("cd0", bem.cd_);
  pd("k_spring", bem.k_spring_, false);
  std::printf("},\n\"body_drag_params\": {");
  pd("cxy", bdrag.cxy_);
  pd("cz", bdrag.cz_);
  pd("ax", bdrag.ax_);
  pd("ay", bdrag.ay_);
  pd("az", bdrag.az_);
  pd("rho", bdrag.rho_, false);
  std::printf("},\n\"lin_cub_params\": {");
  pvec("lin_drag_coeff", lcd.lin_drag_coeff_);
  pvec("cub_drag_coeff", lcd.cub_drag_coeff_);
  pd("induced_lift_coeff", lcd.induced_lift_coeff_, false);
  std::printf("},\n");

  // ---------------- BEM: one persistent model, cases run in sequence ----------------
  ModelPropellerBEM bem_model(quad, bem);
  std::printf("\"bem_cases\": [\n");
  for (size_t k = 0; k < bem_cases.size(); ++k) {
    const Case& c = bem_cases[k];
    Vector<QS::SIZE> s = make_state(c.v, c.q, c.w, c.mot, c.motdes);
    Vector<QS::SIZE> d = Vector<QS::SIZE>::Zero();

    std::printf("{");
    pvec("v_W", c.v);
    pvec("q_wxyz", Vector<4>(s.segment<QS::NATT>(QS::ATT)));
    pvec("w_B", c.w);
    pvec("mot", c.mot);
    pvec("vind_warmstart", Vector<4>(bem_model.vind_.matrix()));
    pvec("vind_h_warmstart", Vector<4>(bem_model.vind_h_.matrix()));

    bem_model.run(s, d);

    const PropellerState& ps = *bem_model.prop_state_;
    pmat34("hub_velocity_frd", ps.velocity_);
    pvec("v_hor", ps.vhor_);
    pvec("v_ver", ps.vver_);
    pvec("v_tot", ps.vtot_);
    pvec("alpha_s", Vector<4>(ps.alpha_s_.matrix()));
    pvec("mu", ps.mu_);
    pvec("omega_mot", Vector<4>(ps.omega_mot_.matrix()));
    pvec("vind", ps.vind_);
    pvec("vind_h", Vector<4>(bem_model.vind_h_.matrix()));
    pvec("a0", ps.a0_);
    pvec("a1s", ps.a1s_);
    pvec("b1s", ps.b1s_);
    pvec("thrust", Vector<4>(ps.thrust_.matrix()));
    pvec("torque", Vector<4>(ps.torque_.matrix()));
    pvec("hforce", Vector<4>(ps.hforce_.matrix()));
    pvec("dvel", Vector<3>(d.segment<QS::NVEL>(QS::VEL)));
    pvec("dome", Vector<3>(d.segment<QS::NOME>(QS::OME)), false);
    std::printf("}%s\n", k + 1 < bem_cases.size() ? "," : "");
  }
  std::printf("],\n");

  // ---------------- simple models + integrators ----------------
  ModelInit m_init(quad);
  ModelMotor m_motor(quad);
  ModelThrustTorqueSimple m_tts(quad);
  ModelRigidBody m_rb(quad);
  ModelBodyDrag m_bd(quad, bdrag);
  ModelLinCubDrag m_lcd(quad, lcd);

  std::vector<Case> simple_cases = {
    {{0, 0, 0}, {0, 0, 0}, {1, 0, 0, 0}, {1800, 1850, 1900, 1950}, {2000, 2000, 2000, 2000}},
    {{5, -3, 2}, {0.4, -0.2, 0.9}, {0.9, 0.1, -0.3, 0.2}, {1500, 2100, 1800, 1650},
     {1400, 2300, 1900, 1500}},
    {{-8, 1, -4}, {-1.2, 0.7, -0.4}, {0.7, -0.4, 0.4, -0.3}, {2500, 2450, 2400, 2350},
     {2600, 2600, 2600, 2600}},
    {{12, 6, -1}, {2.0, -1.5, 3.0}, {0.6, 0.6, -0.3, 0.4}, {1000, 3000, 2000, 1500},
     {900, 3200, 2100, 1400}},
    {{0.3, -0.2, 0.1}, {0.05, 0.02, -0.04}, {1, 0.01, -0.02, 0.005},
     {1810, 1790, 1805, 1795}, {1800, 1800, 1800, 1800}},
    {{-2, 9, 5}, {0.9, 1.1, -2.2}, {0.5, -0.5, 0.5, -0.5}, {3500, 1200, 2800, 1900},
     {3400, 1300, 2900, 1800}},
  };

  auto pipeline = [&](const Ref<const Vector<QS::SIZE>> st, Ref<Vector<QS::SIZE>> dd) {
    m_init.run(st, dd);
    m_motor.run(st, dd);
    m_tts.run(st, dd);
    m_rb.run(st, dd);
    return true;
  };
  DynamicsFunction dyn = pipeline;
  const double dt = 0.001;
  IntegratorEuler int_euler(dyn, dt);
  IntegratorSymplecticEuler int_sym(dyn, dt);
  IntegratorRK4 int_rk4(dyn, dt);

  std::printf("\"simple_cases\": [\n");
  for (size_t k = 0; k < simple_cases.size(); ++k) {
    const Case& c = simple_cases[k];
    Vector<QS::SIZE> s = make_state(c.v, c.q, c.w, c.mot, c.motdes);

    std::printf("{");
    pvec("v_W", c.v);
    pvec("q_wxyz", Vector<4>(s.segment<QS::NATT>(QS::ATT)));
    pvec("w_B", c.w);
    pvec("mot", c.mot);
    pvec("motdes", c.motdes);
    pd("dt", dt);

    Vector<QS::SIZE> d;
    d.setConstant(0.0);
    m_motor.run(s, d);
    pvec("motor_dmot", Vector<4>(d.segment<QS::NMOT>(QS::MOT)));

    d.setConstant(0.0);
    m_tts.run(s, d);
    pvec("tts_dvel", Vector<3>(d.segment<QS::NVEL>(QS::VEL)));
    pvec("tts_dome", Vector<3>(d.segment<QS::NOME>(QS::OME)));

    d.setConstant(0.0);
    m_rb.run(s, d);
    pvec("rb_dpos", Vector<3>(d.segment<QS::NPOS>(QS::POS)));
    pvec("rb_datt", Vector<4>(d.segment<QS::NATT>(QS::ATT)));
    pvec("rb_dome", Vector<3>(d.segment<QS::NOME>(QS::OME)));

    d.setConstant(0.0);
    m_bd.run(s, d);
    pvec("bodydrag_dvel", Vector<3>(d.segment<QS::NVEL>(QS::VEL)));

    d.setConstant(0.0);
    m_lcd.run(s, d);
    pvec("lincub_dvel", Vector<3>(d.segment<QS::NVEL>(QS::VEL)));

    d.setConstant(0.0);
    pipeline(s, d);
    pvec("pipeline_dvel", Vector<3>(d.segment<QS::NVEL>(QS::VEL)));
    pvec("pipeline_dome", Vector<3>(d.segment<QS::NOME>(QS::OME)));
    pvec("pipeline_dmot", Vector<4>(d.segment<QS::NMOT>(QS::MOT)));

    Vector<QS::SIZE> nxt = Vector<QS::SIZE>::Zero();
    int_euler.step(s, dt, nxt);
    pvec("euler_pos", Vector<3>(nxt.segment<QS::NPOS>(QS::POS)));
    pvec("euler_att", Vector<4>(nxt.segment<QS::NATT>(QS::ATT)));
    pvec("euler_vel", Vector<3>(nxt.segment<QS::NVEL>(QS::VEL)));
    pvec("euler_ome", Vector<3>(nxt.segment<QS::NOME>(QS::OME)));
    pvec("euler_mot", Vector<4>(nxt.segment<QS::NMOT>(QS::MOT)));

    nxt.setZero();
    int_sym.step(s, dt, nxt);
    pvec("sym_pos", Vector<3>(nxt.segment<QS::NPOS>(QS::POS)));
    pvec("sym_att", Vector<4>(nxt.segment<QS::NATT>(QS::ATT)));
    pvec("sym_vel", Vector<3>(nxt.segment<QS::NVEL>(QS::VEL)));
    pvec("sym_ome", Vector<3>(nxt.segment<QS::NOME>(QS::OME)));
    pvec("sym_mot", Vector<4>(nxt.segment<QS::NMOT>(QS::MOT)));

    nxt.setZero();
    int_rk4.step(s, dt, nxt);
    pvec("rk4_pos", Vector<3>(nxt.segment<QS::NPOS>(QS::POS)));
    pvec("rk4_att", Vector<4>(nxt.segment<QS::NATT>(QS::ATT)));
    pvec("rk4_vel", Vector<3>(nxt.segment<QS::NVEL>(QS::VEL)));
    pvec("rk4_ome", Vector<3>(nxt.segment<QS::NOME>(QS::OME)));
    pvec("rk4_mot", Vector<4>(nxt.segment<QS::NMOT>(QS::MOT)), false);
    std::printf("}%s\n", k + 1 < simple_cases.size() ? "," : "");
  }
  std::printf("]\n}\n");
  return 0;
}
'''


FLU = np.array([1.0, -1.0, -1.0])

# ---- 15-point Gauss-Kronrod (agilib bem/gauss_kronrod.hpp, verbatim) ----
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


def gk_integrate(fcn, lo, hi, param=None):
    """Single 15-point Kronrod application; fcn(nodes(15), param) -> (4,15)."""
    scale = 0.5 * (hi - lo)
    offset = 0.5 * (hi + lo)
    vals = fcn(GK_X * scale + offset, param)          # (4,15)
    # sequential dot to mirror Eigen's coefficient loop
    acc = np.zeros(vals.shape[0])
    for j in range(15):
        acc = acc + vals[:, j] * GK_W[j]
    return acc * scale


F32 = np.float32
N1 = F32(0.97239411)
N2 = F32(-0.19194795)
PI_F = F32(np.float64(math.pi))
PI2_F = F32(np.float64(math.pi / 2))


def approx_atan2(y, x):
    """agilib fast_atan2.hpp ApproxAtan2, float32 ops replicated exactly."""
    xf, yf = F32(x), F32(y)
    if xf != F32(0.0):
        if abs(xf) >= abs(yf):
            offset = F32(math.copysign(float(PI_F), float(yf))) if xf < 0 else F32(0.0)
            z = F32(yf / xf)
            t = F32(F32(N2 * z) * z)
            return float(F32(offset + F32(F32(N1 + t) * z)))
        offset = F32(math.copysign(float(PI2_F), float(yf)))
        z = F32(xf / yf)
        t = F32(F32(N2 * z) * z)
        return float(F32(offset - F32(F32(N1 + t) * z)))
    if yf > 0:
        return float(PI2_F)
    if yf < 0:
        return float(-PI2_F)
    return 0.0


def quat_rotmat(q):
    """Eigen Quaternion::toRotationMatrix (Hamilton, wxyz, body->world)."""
    w, x, y, z = q
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    return np.array([[1 - (tyy + tzz), txy - twz, txz + twy],
                     [txy + twz, 1 - (txx + tzz), tyz - twx],
                     [txz - twy, tyz + twx, 1 - (txx + tyy)]])


class PState:
    """PropellerState replica (agilib bem/propeller_state.cpp)."""

    def __init__(self, bp):
        self.bp = bp

    def update(self, v_W, q, w_B, mot, t_BM):
        bp = self.bp
        self.omega_mot = np.maximum(np.asarray(mot, float), 10.0)
        self.rot = quat_rotmat(q)
        w_frd = np.asarray(w_B, float) * FLU
        v_body_frd = FLU * (self.rot.T @ np.asarray(v_W, float))
        # velocity_ = v_body_frd (replicated) - t_BM_col x w_frd   [code's frame mixing kept]
        self.velocity = np.empty((3, 4))
        for i in range(4):
            self.velocity[:, i] = v_body_frd - np.cross(t_BM[:, i], w_frd)
        self.vhor = np.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2) + 1e-6
        self.vver = self.velocity[2].copy()
        self.vtot = np.sqrt((self.velocity ** 2).sum(axis=0)) + 1e-6
        self.alpha_s = np.array([approx_atan2(self.vver[i], self.vhor[i]) for i in range(4)])
        self.mu = self.vhor / (self.omega_mot * bp["r_prop"])
        self.a0 = np.zeros(4)
        self.a1s = np.zeros(4)
        self.b1s = np.zeros(4)
        self.vind = np.zeros(4)
        self.w_frd = w_frd


def integrand_psi(ps, kind, radius, vind, psis):
    """IntegrandPsi::evaluate — (4, 15) over azimuth nodes; flapping zeroed as executed."""
    bp = ps.bp
    s_psi, c_psi = np.sin(psis), np.cos(psis)
    beta = ps.a0[:, None] - (np.outer(ps.a1s, c_psi) + np.outer(ps.b1s, s_psi))
    u_t = (radius + bp["r_prop"] * np.outer(ps.mu, s_psi)) * ps.omega_mot[:, None]
    u_p = (ps.vver - vind)[:, None] - (
        np.outer(ps.vver, c_psi) * beta
        + (np.outer(ps.a1s, s_psi) + np.outer(ps.b1s, c_psi)) * ps.omega_mot[:, None] * radius)
    phi = np.empty_like(u_t)
    for i in range(4):
        for j in range(15):
            phi[i, j] = approx_atan2(u_p[i, j], u_t[i, j])
    alpha = bp["theta0"] + bp["theta1"] * radius / bp["r_prop"] + phi
    s_a, c_a = np.sin(alpha), np.cos(alpha)
    cl = bp["cl0"] * (s_a * c_a + 0.07)
    cd = bp["cd0"] * s_a ** 2
    c = bp["chord_inner"] + radius / bp["r_prop"] * (bp["chord_outer"] - bp["chord_inner"])
    u_sqr = u_t ** 2 + u_p ** 2
    d_lift = u_sqr * cl * c
    d_drag = u_sqr * cd * c
    if kind == "T":
        return d_lift * np.cos(phi) + d_drag * np.sin(phi)
    if kind == "Q":
        return radius * (-d_lift * np.sin(phi) + d_drag * np.cos(phi))
    return (-d_lift * np.sin(phi) + d_drag * np.cos(phi)) * s_psi[None, :]


def integrand_r(ps, kind):
    def f(radii, param):
        vind = ps.vind if param is None else param
        out = np.empty((4, 15))
        for j in range(15):
            out[:, j] = gk_integrate(
                lambda psis, _p, rj=radii[j], vv=vind: integrand_psi(ps, kind, rj, vv, psis),
                0.0, 2.0 * math.pi)
        return ps.bp["num_blades"] * ps.bp["rho"] / (4.0 * math.pi) * out
    return f


def thrust_residual(ps, vind):
    """ThrustFunction::evaluate — T_BEM(vind) − T_mom(vind), all four rotors."""
    t_bem = gk_integrate(integrand_r(ps, "T"), 0.0, ps.bp["r_prop"], vind)
    tmp = ps.vver - vind
    t_mom = 2.0 * ps.bp["rho"] * ps.bp["prop_area"] * vind * np.sqrt(
        ps.vhor * ps.vhor + tmp * tmp)
    return t_bem - t_mom


def brent(fcn, lo, hi, guess, warmstart, tol=1e-3, max_iter=100, ws_delta=0.02):
    """agilib bem/brent.hpp verbatim (vectorized over the 4 rotors)."""
    n = len(guess)
    scale = hi - lo
    err_flag = False
    if warmstart:
        x0 = guess - scale * ws_delta
        x1 = guess + scale * ws_delta
        f_x0 = fcn(x0)
        f_x1 = fcn(x1)
    if (not warmstart) or (f_x0 * f_x1 > 0).any():
        x0 = np.full(n, lo)
        x1 = np.full(n, hi)
        f_x0 = fcn(x0)
        f_x1 = fcn(x1)
    if (f_x0 * f_x1 > 0).any():
        err_flag = True
    x2 = x0.copy()
    f_x2 = f_x0.copy()
    x3 = x1 - x0
    x4 = x3.copy()

    ret = np.zeros(n)
    it = 0
    while not (ret == 1).all() and not (ret == 2).any() and it < max_iter and not err_flag:
        for i in range(n):
            ret[i] = _brent_step(i, f_x0, f_x1, f_x2, x0, x1, x2, x3, x4, tol)
        f_x1 = fcn(x1)
        it += 1
    return x1.copy()


def _brent_step(i, f_x0, f_x1, f_x2, x0, x1, x2, x3, x4, tol):
    if f_x1[i] * f_x2[i] > 0:
        x2[i] = x0[i]
        f_x2[i] = f_x0[i]
        x3[i] = x1[i] - x0[i]
        x4[i] = x3[i]
    if abs(f_x2[i]) < abs(f_x1[i]):
        x0[i] = x1[i]
        x1[i] = x2[i]
        x2[i] = x0[i]
        f_x0[i] = f_x1[i]
        f_x1[i] = f_x2[i]
        f_x2[i] = f_x0[i]
    m = 0.5 * (x2[i] - x1[i])
    if abs(m) > tol and abs(f_x1[i]) > 0:
        if abs(x4[i]) < tol or abs(f_x0[i]) <= abs(f_x1[i]):
            x3[i] = m
            x4[i] = m
        else:
            s = f_x1[i] / f_x0[i]
            if x0[i] == x2[i]:
                p = 2 * m * s
                q = 1 - s
            else:
                q = f_x0[i] / f_x2[i]
                r = f_x1[i] / f_x2[i]
                p = s * (2 * m * q * (q - r) - (x1[i] - x0[i]) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            s = x4[i]
            x4[i] = x3[i]
            if (2 * p < 3 * m * q - abs(tol * q)) and (p < abs(s * q / 2)):
                x3[i] = p / q
            else:
                x3[i] = m
                x4[i] = m
        x0[i] = x1[i]
        f_x0[i] = f_x1[i]
        if abs(x3[i]) > tol:
            x1[i] += x3[i]
        else:
            x1[i] += tol if m > 0 else -tol
    else:
        return 1
    return 0


def run_bem(bp, quad, case, vind_ws, vind_h_ws):
    """ModelPropellerBEM::run replica. Returns dict of internals + derivative contribs."""
    t_BM = np.array(quad["t_BM"]).T  # stored column-major as list of columns
    ps = PState(bp)
    ps.update(case["v_W"], case["q_wxyz"], case["w_B"], case["mot"], t_BM)

    vind = brent(lambda v: thrust_residual(ps, v), -20.0, 30.0, vind_ws.copy(), True)
    ps.vind = vind.copy()
    vind_momentum = vind.copy()  # pre-VRS momentum-closure root
    vind_h = vind_h_ws.copy()
    vind2 = np.zeros(4)

    tmp = ps.vver / ps.vind
    in_window = (tmp >= 0.01) & (tmp <= 2)
    any_gate = bool((tmp >= 0.01).any() and (tmp <= 2).any())
    if any_gate:
        k0, k1, k2, k3, k4 = 1.0, -1.125, -1.372, -1.718, -0.655
        vz = -ps.vver.copy()
        vtot_old = ps.vtot.copy()
        vver_old = ps.vver.copy()
        ps.vtot = ps.vhor.copy()
        ps.vver = np.zeros(4)
        vind_h = brent(lambda v: thrust_residual(ps, v), -20.0, 30.0, vind_h_ws.copy(), True)
        vzvh = vz / vind_h
        vind2 = vind_h * (k0 + k1 * vzvh + k2 * vzvh ** 2 + k3 * vzvh ** 3 + k4 * vzvh ** 4)
        ps.vind = np.maximum(ps.vind, vind2)
        ps.vver = vver_old  # code: vver_ = -vz
        ps.vtot = vtot_old
        for i in range(4):
            if in_window[i]:
                ps.vind[i] = min(ps.vind[i], 2 * vind_h[i])

    thrust = gk_integrate(integrand_r(ps, "T"), 0.0, bp["r_prop"], None)
    torque = gk_integrate(integrand_r(ps, "Q"), 0.0, bp["r_prop"], None)
    hforce = gk_integrate(integrand_r(ps, "H"), 0.0, bp["r_prop"], None) * 3.0

    a0, a1s, b1s = flapping_fits(ps)

    force = np.zeros(3)
    torque_b = np.zeros(3)
    for i in range(4):
        chi = math.atan2(ps.velocity[1, i], ps.velocity[0, i])
        cc, sc = math.cos(chi), math.sin(chi)
        Rz = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
        cw = 1.0 if i >= 2 else -1.0
        f_frd = Rz @ np.array([-(hforce[i] + math.sin(a1s[i]) * thrust[i]),
                               cw * math.sin(b1s[i]) * thrust[i],
                               -math.cos(a0[i]) * thrust[i]])
        f_flu = f_frd * FLU
        force += f_flu
        t_frd = Rz @ np.array([cw * bp["k_spring"] * b1s[i],
                               bp["k_spring"] * a1s[i],
                               -cw * torque[i]])
        torque_b += t_frd * FLU + np.cross(t_BM[:, i], f_flu)
    force[2] *= 0.9575

    J_inv = np.diag(1.0 / np.array(quad["J_diag"]))
    dvel = ps.rot @ force / quad["mass"] + np.array([0.0, 0.0, -quad["G"]])
    dome = J_inv @ torque_b
    return {"vhor": ps.vhor, "vver": ps.vver, "vtot": ps.vtot, "alpha_s": ps.alpha_s,
            "mu": ps.mu, "omega_mot": ps.omega_mot, "velocity": ps.velocity,
            "vind": ps.vind, "vind_h": vind_h, "vind_momentum": vind_momentum,
            "vind2": vind2, "vrs_any_gate": any_gate, "vrs_in_window": in_window,
            "a0": a0, "a1s": a1s, "b1s": b1s, "thrust": thrust, "torque": torque,
            "hforce": hforce, "dvel": dvel, "dome": dome}


def flapping_fits(ps):
    """PropellerState::calculateFlapping replica (machine-generated rational fits,
    agilib bem/propeller_state.cpp:75-166, constants transcribed verbatim)."""
    om = [np.ones(4)]
    mu = [np.ones(4)]
    for i in range(1, 7):
        om.append(om[i - 1] * ps.omega_mot)
        mu.append(mu[i - 1] * ps.mu)
    p = ps.w_frd[0]
    q = ps.w_frd[1]
    al = ps.alpha_s
    vi = ps.vind

    den = (-4.377059028e8 - 1.147609987e-7 * om[4] * mu[2]
           + 7.650732730e-8 * om[4] * mu[4] + 1.135865755e-14 * om[6] * mu[4]
           - 2.999395522e-14 * om[6] - 2.020271598e-7 * om[4] - 64.98399099 * om[2])
    a0 = (17860.69768 - 9.085245727e-8 * om[3] * mu[1] * q
          + 5.075429983e-14 * om[5] * mu[2] * vi
          - 1.201149295e-15 * om[5] * mu[3] * p
          - 3.287355978e-15 * om[6] * al * mu[3]
          - 8.012930184e-15 * om[6] * al * mu[1]
          + 1.011494156e-15 * om[6] * al * mu[5]
          - 8.680285403 * om[1] * mu[1] * p
          - 17.36057081 * om[2] * al * mu[1]
          + 1.517241217e-15 * om[5] * mu[5] * p
          - 5.958332683e-15 * om[5] * mu[1] * p
          - 1.561670740e-14 * om[5] * mu[4] * vi
          - 3.121900413e-12 * om[4] * mu[4]
          + 1.237136043e-13 * om[5] * vi
          - 3.157428176e-15 * om[6] * mu[2]
          + 6.274197178e-16 * om[6] * mu[6]
          - 2.031302206e-16 * om[6] * mu[4]
          + 268.0341334 * om[1] * vi
          - 3.589529587 * om[2] * mu[2]
          - 1.902162223e-15 * om[6]
          + 8.243768249e-12 * om[4] - 4.121166794 * om[2]) / den

    den_b = (-4.377059028e8 - 1.147609987e-7 * om[4] * mu[2]
             + 7.650732720e-8 * om[4] * mu[4] + 1.135865756e-14 * om[6] * mu[4]
             - 2.999395520e-14 * om[6] - 2.020271601e-7 * om[4] - 64.98399099 * om[2])
    b1s = 3.034482349e-15 * om[1] * (
        14.44639117 * om[4] * q
        + 0.6202900305 * om[5] * mu[5]
        + 1.177725817e17 * mu[1] * vi
        - 1.208793256 * om[5] * mu[3]
        - 1.157259683 * om[5] * mu[1]
        + 9.730505319e7 * om[2] * q
        - 3.482171723e15 * om[1] * mu[1]
        - 3086.424696 * om[3] * mu[3]
        - 5.169754829e8 * om[3] * mu[1]
        - 6.724278493e8 * om[2] * p
        - 5.988003267e7 * om[2] * mu[2] * q
        + 1.500000041 * om[4] * mu[4] * p
        - 8.890086873 * om[4] * mu[2] * q
        - 7.628130118e15 * om[1] * al * mu[2]
        - 4.529202255e15 * p
        - 15.43924687 * om[4] * mu[3] * vi
        + 1.748510208e10 * om[2] * mu[1] * vi
        + 75.26632912 * om[4] * mu[1] * vi
        + 0.9999999999 * om[5] * al * mu[4]
        - 4.875000134 * om[5] * al * mu[2]
        - 3.625000097 * om[4] * mu[2] * p
        - 1.132510062e9 * om[3] * al * mu[2]) / den_b

    den_a = (-4.377059028e8 - 1.147609987e-7 * om[4] * mu[2]
             + 7.650732720e-8 * om[4] * mu[4] + 1.135865755e-14 * om[6] * mu[4]
             - 2.999395520e-14 * om[6] - 2.020271610e-7 * om[4] - 64.98399099 * om[2])
    a1s = 4.543463022e-14 * (
        0.4564908655 * om[5] * mu[3]
        + 0.7417976561 * om[5] * mu[1]
        + 4.491002632e7 * om[2] * q
        + 0.9648437489 * om[4] * p
        + 1.559168274e10 * om[1] * mu[1]
        - 58779.37391 * om[3] * mu[3]
        + 1.398833618e6 * om[3] * mu[1]
        + 6.498797591e6 * om[2] * p + 3.024957887e14 * q
        - 15.43924657 * om[4] * mu[3] * vi
        - 3.578285830e6 * om[2] * mu[2] * p
        + 6.499534432e7 * om[2] * mu[1] * vi
        - 25.08877564 * om[4] * mu[1] * vi
        - 1.039925259e8 * om[2] * mu[3] * vi
        + om[5] * al * mu[4]
        + 1.625000001 * om[5] * al * mu[2]
        + 0.5937500002 * om[4] * mu[2] * p
        + 6.735595891e6 * om[3] * al * mu[4]
        - 4.209748462e6 * om[3] * al * mu[2]) * om[1] / den_a

    return a0, a1s, b1s


# ---------------------------------------------------------------------------
# transcription self-checks (hard asserts — nothing is written unless these pass)
# ---------------------------------------------------------------------------

def _chk(name, got, want, tol):
    got, want = np.asarray(got, float), np.asarray(want, float)
    # absolute floor 1e-6: near-zero entries (e.g. symmetric-case H ~ 1e-8 N) compare
    # against float noise, not against themselves
    scale = np.maximum(1e-6, np.abs(want))
    rel = float(np.max(np.abs(got - want) / scale))
    assert rel < tol, f"transcription self-check failed: {name} rel={rel:.3e}"
    return rel


def check_bem_cases(doc):
    """Replay every BEM case through the replica; return per-case replica extras
    (pre-VRS momentum root, VRS gate details) for storage."""
    quad, b = doc["quad"], doc["bem_params"]
    bp = {"rho": b["rho"], "r_prop": b["r_prop"], "prop_area": b["prop_area"],
          "theta0": b["theta0"], "theta1": b["theta1"], "chord_inner": b["chord_inner"],
          "chord_outer": b["chord_outer"], "num_blades": b["num_blades"],
          "cl0": b["cl0"], "cd0": b["cd0"], "k_spring": b["k_spring"]}
    extras, worst = [], 0.0
    for c in doc["bem_cases"]:
        out = run_bem(bp, quad, c, np.array(c["vind_warmstart"]),
                      np.array(c["vind_h_warmstart"]))
        for key, rec, tol in [
                ("vhor", "v_hor", 1e-12), ("vver", "v_ver", 1e-12),
                ("alpha_s", "alpha_s", 1e-12), ("mu", "mu", 1e-12),
                ("vind", "vind", 1e-12), ("vind_h", "vind_h", 1e-12),
                ("thrust", "thrust", 1e-12), ("torque", "torque", 1e-12),
                ("hforce", "hforce", 1e-10), ("a0", "a0", 1e-12),
                ("a1s", "a1s", 1e-12), ("b1s", "b1s", 1e-12),
                ("dvel", "dvel", 1e-10), ("dome", "dome", 1e-10)]:
            worst = max(worst, _chk(key, out[key], c[rec], tol))
        worst = max(worst, _chk("velocity", out["velocity"],
                                np.array(c["hub_velocity_frd"]).T, 1e-12))
        extras.append({
            "vind_momentum": out["vind_momentum"].tolist(),
            "vind_vrs_candidate": out["vind2"].tolist(),
            "vrs_any_gate": out["vrs_any_gate"],
            "vrs_in_window": [bool(x) for x in out["vrs_in_window"]]})
    print(f"BEM self-check: {len(doc['bem_cases'])} cases, worst rel {worst:.2e}")
    return extras


def _quat_mult(q, p):
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array([qw * pw - qx * px - qy * py - qz * pz,
                     qw * px + qx * pw + qy * pz - qz * py,
                     qw * py - qx * pz + qy * pw + qz * px,
                     qw * pz + qx * py - qy * px + qz * pw])


def check_simple_cases(doc):
    """Replay motor / thrust-torque / rigid-body / drag models and the three integrators."""
    quad = doc["quad"]
    t_BM = np.array(quad["t_BM"]).T
    m, G_, tau = quad["mass"], quad["G"], quad["motor_tau"]
    Jd = np.array(quad["J_diag"])
    tm, kappa = np.array(quad["thrust_map"]), quad["kappa"]
    bd, lc = doc["body_drag_params"], doc["lin_cub_params"]
    alloc = np.vstack([np.ones(4), t_BM[1], -t_BM[0],
                       kappa * np.array([-1.0, -1.0, 1.0, 1.0])])

    def f_pipeline(s):
        d = np.zeros(SIZE)
        q = s[ATT:ATT + 4]
        w = s[OME:OME + 3]
        mot = s[MOT:MOT + 4]
        d[POS:POS + 3] = s[VEL:VEL + 3]
        d[ATT:ATT + 4] = 0.5 * _quat_mult(q, np.array([0.0, *w]))
        thr = tm[0] * mot**2 + tm[1] * mot + tm[2]
        ft = alloc @ thr
        d[VEL:VEL + 3] = quat_rotmat(q) @ np.array([0.0, 0.0, ft[0]]) / m \
            + np.array([0.0, 0.0, -G_])
        d[OME:OME + 3] = (ft[1:4] - np.cross(w, Jd * w)) / Jd
        d[MOT:MOT + 4] = (s[MOTDES:MOTDES + 4] - mot) / tau
        return d

    worst = 0.0
    for c in doc["simple_cases"]:
        v = np.array(c["v_W"])
        q = np.array(c["q_wxyz"])
        w = np.array(c["w_B"])
        mot = np.array(c["mot"])
        R = quat_rotmat(q)
        vb = R.T @ v

        worst = max(worst, _chk("motor_dmot", (np.array(c["motdes"]) - mot) / tau,
                                c["motor_dmot"], 1e-12))
        thr = tm[0] * mot**2 + tm[1] * mot + tm[2]
        ft = alloc @ thr
        worst = max(worst, _chk(
            "tts_dvel", R @ np.array([0.0, 0.0, ft[0]]) / m + [0, 0, -G_],
            c["tts_dvel"], 1e-12))
        worst = max(worst, _chk("tts_dome", ft[1:4] / Jd, c["tts_dome"], 1e-12))
        worst = max(worst, _chk("rb_dpos", v, c["rb_dpos"], 1e-12))
        worst = max(worst, _chk("rb_datt", 0.5 * _quat_mult(q, np.array([0.0, *w])),
                                c["rb_datt"], 1e-12))
        worst = max(worst, _chk("rb_dome", -np.cross(w, Jd * w) / Jd, c["rb_dome"], 1e-12))
        coeff = 0.5 * bd["rho"] * np.array(
            [bd["cxy"] * bd["ax"], bd["cxy"] * bd["ay"], bd["cz"] * bd["az"]])
        # NOTE: agilib adds this FORCE to the acceleration slot without /m (finding F-19)
        worst = max(worst, _chk("bodydrag_dvel", R @ (-vb * np.abs(vb) * coeff),
                                c["bodydrag_dvel"], 1e-12))
        F = -vb * np.array(lc["lin_drag_coeff"]) - vb**3 * np.array(lc["cub_drag_coeff"])
        ind = np.array([0.0, 0.0, lc["induced_lift_coeff"] * (vb[0]**2 + vb[1]**2)])
        worst = max(worst, _chk("lincub_dvel", R @ (F + ind) / m, c["lincub_dvel"], 1e-12))

        s = np.zeros(SIZE)
        s[ATT:ATT + 4] = q
        s[VEL:VEL + 3] = v
        s[OME:OME + 3] = w
        s[MOT:MOT + 4] = mot
        s[MOTDES:MOTDES + 4] = c["motdes"]
        d = f_pipeline(s)
        worst = max(worst, _chk("pipeline_dvel", d[VEL:VEL + 3], c["pipeline_dvel"], 1e-12))
        worst = max(worst, _chk("pipeline_dome", d[OME:OME + 3], c["pipeline_dome"], 1e-12))
        worst = max(worst, _chk("pipeline_dmot", d[MOT:MOT + 4], c["pipeline_dmot"], 1e-12))

        dt = c["dt"]
        e = s + dt * d
        for key, sl in [("euler_pos", slice(POS, POS + 3)), ("euler_att", slice(ATT, ATT + 4)),
                        ("euler_vel", slice(VEL, VEL + 3)), ("euler_ome", slice(OME, OME + 3)),
                        ("euler_mot", slice(MOT, MOT + 4))]:
            worst = max(worst, _chk(key, e[sl], c[key], 1e-12))
        # symplectic: velocities with f(s), then positions with f at the velocity-updated state
        s1 = s.copy()
        for sl in (slice(VEL, VEL + 3), slice(OME, OME + 3), slice(MOT, MOT + 4)):
            s1[sl] = s[sl] + dt * d[sl]
        d1 = f_pipeline(s1)
        s2 = s1.copy()
        for sl in (slice(POS, POS + 3), slice(ATT, ATT + 4)):
            s2[sl] = s[sl] + dt * d1[sl]
        for key, sl in [("sym_pos", slice(POS, POS + 3)), ("sym_att", slice(ATT, ATT + 4)),
                        ("sym_vel", slice(VEL, VEL + 3)), ("sym_ome", slice(OME, OME + 3)),
                        ("sym_mot", slice(MOT, MOT + 4))]:
            worst = max(worst, _chk(key, s2[sl], c[key], 1e-12))
        k1 = f_pipeline(s)
        k2 = f_pipeline(s + dt / 2 * k1)
        k3 = f_pipeline(s + dt / 2 * k2)
        k4 = f_pipeline(s + dt * k3)
        rk = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        for key, sl in [("rk4_pos", slice(POS, POS + 3)), ("rk4_att", slice(ATT, ATT + 4)),
                        ("rk4_vel", slice(VEL, VEL + 3)), ("rk4_ome", slice(OME, OME + 3)),
                        ("rk4_mot", slice(MOT, MOT + 4))]:
            worst = max(worst, _chk(key, rk[sl], c[key], 1e-12))
    print(f"simple-model self-check: {len(doc['simple_cases'])} cases, worst rel {worst:.2e}")


# ---------------------------------------------------------------------------
# build / run / write
# ---------------------------------------------------------------------------

def _sha16(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def ensure_repo(repo_arg, workdir: pathlib.Path) -> pathlib.Path:
    if repo_arg:
        repo = pathlib.Path(repo_arg).resolve()
    else:
        repo = workdir / "agilicious_internal_mine"
        subprocess.run(["git", "clone", "--quiet", MIRROR_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", PINNED_COMMIT],
                       check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_COMMIT, f"repo at {head}, expected {PINNED_COMMIT}"
    for rel, want in PINNED_SHA256_16.items():
        got = _sha16(repo / rel)
        assert got == want, f"sha mismatch {rel}: {got} != {want}"
    return repo


def ensure_eigen(eigen_arg, workdir: pathlib.Path) -> pathlib.Path:
    if eigen_arg:
        return pathlib.Path(eigen_arg).resolve()
    tgz = workdir / "eigen.tar.gz"
    urllib.request.urlretrieve(EIGEN_URL, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(workdir)
    return workdir / "eigen-3.4.0"


def build_and_run(repo: pathlib.Path, eigen: pathlib.Path, workdir: pathlib.Path) -> dict:
    driver = workdir / "driver.cpp"
    driver.write_text(DRIVER_CPP)
    exe = workdir / "driver"
    ag = repo / "agilib"
    cmd = (["g++", *CXX_FLAGS, "-I", str(ag / "include"), "-I", str(eigen),
            str(driver)] + [str(ag / tu) for tu in AGILIB_TUS] + ["-o", str(exe)])
    subprocess.run(cmd, check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def provenance() -> dict:
    return {
        "source": "agilicious agilib (uzh-rpg, GPLv3) — EXECUTED via compiled driver",
        "mirror_url": MIRROR_URL,
        "commit": PINNED_COMMIT,
        "bem_upstream": f"BEM/model sources byte-identical to {BEM_UPSTREAM_COMMIT}",
        "file_sha256_16": PINNED_SHA256_16,
        "compiler": "g++ " + " ".join(CXX_FLAGS),
        "eigen": "3.4.0 (build-time header dependency only)",
        "driver": "unmodified agilib TUs; driver reads internals via a layout-identical "
                  "private->public accessor trick (gen_agilicious.py DRIVER_CPP)",
        "self_check": "float-exact Python replica of every executed path (GK15, Brent, "
                      "float32 atan2, VRS any-gate, flapping fits, composition) asserted "
                      "<=1e-10 rel before writing; vind_momentum / vind_vrs_candidate / "
                      "vrs_* fields are reconstructed by that validated replica",
        "paper": "NeuroBEM (arXiv:2106.08015) eqs. (5)-(19)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--repo", help="existing mirror checkout (skips cloning)")
    ap.add_argument("--eigen", help="existing Eigen source dir (skips download)")
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gen_agilicious_"))
    try:
        repo = ensure_repo(args.repo, tmp)
        eigen = ensure_eigen(args.eigen, tmp)
        doc = build_and_run(repo, eigen, tmp)

        extras = check_bem_cases(doc)
        check_simple_cases(doc)

        args.out.mkdir(parents=True, exist_ok=True)
        bem_doc = {
            "kind": "agilicious_terms",
            "family": "bem_rotor",
            "provenance": provenance(),
            "quad": doc["quad"],
            "bem_params": doc["bem_params"],
            "spin": SPIN,
            "identified": {"camber_offset": 0.07, "h_force_correction": 3.0,
                           "z_obstruction_factor": 0.9575},
            "cases": [{**c, **e} for c, e in zip(doc["bem_cases"], extras)],
        }
        (args.out / "agilicious_bem.json").write_text(json.dumps(bem_doc, indent=1))
        simple_doc = {
            "kind": "agilicious_terms",
            "family": "simple_models",
            "provenance": provenance(),
            "quad": doc["quad"],
            "body_drag_params": doc["body_drag_params"],
            "lin_cub_params": doc["lin_cub_params"],
            "spin": SPIN,
            "cases": doc["simple_cases"],
        }
        (args.out / "agilicious_simple_models.json").write_text(
            json.dumps(simple_doc, indent=1))
        print(f"wrote {args.out / 'agilicious_bem.json'}")
        print(f"wrote {args.out / 'agilicious_simple_models.json'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
