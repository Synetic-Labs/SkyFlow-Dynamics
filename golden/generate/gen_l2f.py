"""
Golden-vector generator: rl-tools L2F IMU error model and motor steps, EXECUTED.

Builds a small C++ driver against the UNMODIFIED header-only rl-tools library and runs
  A. `l2f::bias_process_step` (operations_generic/70_post_integration.h) — the sensor bias
     process: Ornstein–Uhlenbeck for a finite correlation time, pure random walk for
     tau <= 0, held constant for sigma = 0;
  B. the full `rl_tools::step` of an L2F environment whose state carries the IMU
     (StateIMU = StateAccelerometer<StateGyro<...>>): RK4 rigid body, gyro and
     accelerometer bias processes, white noise with a constant offset, finite-difference
     specific force — once with the integrated motor lag and once with CLOSED_FORM.

The driving noise is NOT recovered from the outputs. Before every reference call the driver
copies the generator state (std::mt19937) and replays the reference's own sampler
(`random::normal_distribution::sample`) on the copy, in the reference's draw order. A
plain-Python replica of every executed path is then asserted against the executed outputs
before anything is written (INTAKE.md step 7) — a wrong draw order cannot pass it.

Reference provenance: rl-tools/rl-tools at the commit pinned by rl-tools/hyperdrone 01eee73.
Every physics-bearing header is pinned by sha256 below and re-verified before the build.
The older rl-tools pin of rl-tools/raptor (e43ae4b) predates the IMU error model.

Usage (from the repo root):

    uv run python golden/generate/gen_l2f.py --out golden/vectors [--repo /path/to/rl-tools]

Without --repo the repository is cloned to a temp dir at the pinned commit. Requires g++
(C++17). Nothing from rl-tools is redistributed: the vectors hold numeric inputs/outputs
plus provenance.
"""

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import tempfile

REPO_URL = "https://github.com/rl-tools/rl-tools"
PINNED_COMMIT = "ad2e56eb3e66a5b093a91661db81df39ff19be6c"
CXX_FLAGS = ["-O1", "-std=c++17", "-ffp-contract=off"]

#: sha256 (first 16 hex chars) of every physics-bearing header the driver instantiates —
#: verified against the checkout before building; a mismatch aborts generation.
PINNED_SHA256_16 = {
    "include/rl_tools/rl/environments/l2f/operations_generic/70_post_integration.h": "cf154482c2663a1a",
    "include/rl_tools/rl/environments/l2f/operations_generic/60_dynamics.h": "e212261f1bd1aeeb",
    "include/rl_tools/rl/environments/l2f/operations_generic/00_actuation.h": "abebab6d6185207f",
    "include/rl_tools/rl/environments/l2f/operations_generic/20_initial_state.h": "b1e395c76b9acee4",
    "include/rl_tools/rl/environments/l2f/operations_generic.h": "ae52899cc2e6c79c",
    "include/rl_tools/rl/environments/l2f/multirotor.h": "98948f843f35f6dd",
    "include/rl_tools/rl/environments/l2f/quaternion_helper.h": "63b347a6d7397fe3",
    "include/rl_tools/rl/environments/l2f/parameters/default.h": "1c17610024f61481",
    "include/rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h": "e95f0d373829eefd",
    "include/rl_tools/random/operations_cpu.h": "393241e777dba8c3",
    "include/rl_tools/utils/generic/integrators.h": "f4a7a6a49d6d31c6",
}

DRIVER_CPP = r"""// Golden-vector driver for rl-tools L2F (compiled against the UNMODIFIED headers).
#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <cmath>
#include <cstdio>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;

constexpr TI ACTION_HISTORY_LENGTH = 4;

template <bool T_CLOSED_FORM>
struct STATIC_PARAMETERS{
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr bool CLOSED_FORM = T_CLOSED_FORM;
    using STATE_BASE_INNER = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_IMU        = l2f::StateIMU<T, TI, STATE_BASE_INNER>;
    using STATE_TYPE       = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_IMU>>>>;
    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
        l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
        l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using BASE = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;
    using PARAMETERS = typename BASE::PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = BASE::nominal_parameters;
    static constexpr T STATE_LIMIT_POSITION_X = 100000;
    static constexpr T STATE_LIMIT_POSITION_Y = 100000;
    static constexpr T STATE_LIMIT_POSITION_Z = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
};

template <bool CF> using ENV_SPEC = l2f::Specification<T, TI, STATIC_PARAMETERS<CF>>;
template <bool CF> using ENV = rlt::rl::environments::Multirotor<ENV_SPEC<CF>>;

static void vec(const char* name, const T* v, int n, bool comma = true){
    std::printf("\"%s\": [", name);
    for(int i = 0; i < n; i++) std::printf("%s%.17g", i ? ", " : "", v[i]);
    std::printf("]%s", comma ? ", " : "");
}
static void num(const char* name, T v, bool comma = true){
    std::printf("\"%s\": %.17g%s", name, v, comma ? ", " : "");
}

struct Process{ T init_max; T tau; T sigma; };

// A: the bias process alone. The unit-normal draws are obtained by replaying the
// reference's own sampler on a copy of the generator state taken before the call.
static void bias_sequences(){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng);
    typename DEVICE::SPEC::RANDOM random_dev;
    const Process processes[] = {{0, 0.05, 0.5}, {0, 2.0, 0.02}, {0, 300.0, 1.9e-5}, {0, 0.0, 0.05}, {0, -1.0, 0.003}, {0, 0.7, 0.0}};
    const T dts[] = {0.01, 0.001, 0.0025, 0.01, 0.002, 0.01};
    std::printf("\"bias_process\": [");
    for(int c = 0; c < 6; c++){
        rlt::init(device, rng, (TI)(100 + c));
        T bias[3] = {0.011 * (c + 1), -0.007 * (c + 2), 0.003 * (c - 2)};
        std::printf("%s{", c ? ", " : "");
        num("tau", processes[c].tau); num("sigma", processes[c].sigma); num("dt", dts[c]);
        std::printf("\"seed\": %d, \"steps\": [", 100 + c);
        for(int s = 0; s < 8; s++){
            RNG replay = rng;
            T w[3] = {0, 0, 0};
            if(processes[c].sigma != 0){
                for(int i = 0; i < 3; i++) w[i] = rlt::random::normal_distribution::sample(random_dev, (T)0, (T)1, replay);
            }
            T next[3];
            l2f::bias_process_step(device, processes[c], dts[c], bias, next, rng);
            std::printf("%s{", s ? ", " : "");
            vec("b", bias, 3); vec("w", w, 3); vec("b_next", next, 3, false);
            std::printf("}");
            for(int i = 0; i < 3; i++) bias[i] = next[i];
        }
        std::printf("]}");
    }
    std::printf("]");
}

// B: full rlt::step with the IMU state. Draw order inside one step (action noise std = 0
// consumes nothing): gyro bias (3, only if sigma != 0), gyro white (3, only if std != 0),
// accelerometer bias, accelerometer white. Replayed on a copy; the generator asserts it.
template <bool CF>
static void full_steps(const char* name, int seed, T gyro_tau, T accel_tau){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)seed);
    typename DEVICE::SPEC::RANDOM random_dev;
    ENV<CF> env; typename ENV<CF>::Parameters params; typename ENV<CF>::State state, next;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.disturbances.random_force = {0, 0};
    params.disturbances.random_torque = {0, 0};
    params.mdp.action_noise.normalized_rpm = 0;
    params.imu.gyro.error.noise = {-0.004, 0.012};
    params.imu.gyro.error.bias = {0, gyro_tau, 0.03};
    params.imu.accelerometer.error.noise = {0.05, 0.25};
    params.imu.accelerometer.error.bias = {0, accel_tau, 0.4};
    rlt::initial_state(device, env, params, state);
    state.linear_velocity[0] = 0.4; state.linear_velocity[1] = -0.3; state.linear_velocity[2] = 0.2;
    state.angular_velocity[0] = 0.9; state.angular_velocity[1] = -1.3; state.angular_velocity[2] = 0.6;
    const T h = 0.5 * 0.35;
    state.orientation[0] = std::cos(h); state.orientation[1] = std::sin(h) * 0.6; state.orientation[2] = std::sin(h) * 0.0; state.orientation[3] = std::sin(h) * 0.8;
    for(int i = 0; i < 3; i++){ state.gyro_bias[i] = 0.01 * (i + 1); state.accelerometer_bias[i] = -0.08 * (i + 1); }
    for(int i = 0; i < 4; i++) state.rpm[i] = 0.55 + 0.08 * i;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV<CF>::ACTION_DIM>> action;
    rlt::malloc(device, action);
    std::printf("\"%s\": {", name);
    num("dt", params.integration.dt); vec("gravity", params.dynamics.gravity, 3);
    num("gyro_noise_mean", params.imu.gyro.error.noise.mean); num("gyro_noise_std", params.imu.gyro.error.noise.std);
    num("gyro_bias_tau", params.imu.gyro.error.bias.tau); num("gyro_bias_sigma", params.imu.gyro.error.bias.sigma);
    num("accel_noise_mean", params.imu.accelerometer.error.noise.mean); num("accel_noise_std", params.imu.accelerometer.error.noise.std);
    num("accel_bias_tau", params.imu.accelerometer.error.bias.tau); num("accel_bias_sigma", params.imu.accelerometer.error.bias.sigma);
    vec("tau_rising", params.dynamics.rotor_time_constants_rising, 4); vec("tau_falling", params.dynamics.rotor_time_constants_falling, 4);
    num("action_min", params.dynamics.action_limit.min); num("action_max", params.dynamics.action_limit.max);
    std::printf("\"closed_form\": %s, \"seed\": %d, \"steps\": [", CF ? "true" : "false", seed);
    for(int s = 0; s < 10; s++){
        T a[4];
        for(int i = 0; i < 4; i++){ a[i] = std::sin(1.7 * s + 2.1 * i) * 0.9; rlt::set(action, 0, i, a[i]); }
        RNG replay = rng;
        T gw[3], gn[3], aw[3], an[3];
        for(int i = 0; i < 3; i++) gw[i] = rlt::random::normal_distribution::sample(random_dev, (T)0, (T)1, replay);
        for(int i = 0; i < 3; i++) gn[i] = rlt::random::normal_distribution::sample(random_dev, params.imu.gyro.error.noise.mean, params.imu.gyro.error.noise.std, replay);
        for(int i = 0; i < 3; i++) aw[i] = rlt::random::normal_distribution::sample(random_dev, (T)0, (T)1, replay);
        for(int i = 0; i < 3; i++) an[i] = rlt::random::normal_distribution::sample(random_dev, params.imu.accelerometer.error.noise.mean, params.imu.accelerometer.error.noise.std, replay);
        rlt::step(device, env, params, state, action, next, rng);
        std::printf("%s{", s ? ", " : "");
        vec("action", a, 4);
        vec("v", state.linear_velocity, 3); vec("rpm", state.rpm, 4);
        vec("gyro_bias", state.gyro_bias, 3); vec("accel_bias", state.accelerometer_bias, 3);
        vec("gyro_w", gw, 3); vec("gyro_draw", gn, 3); vec("accel_w", aw, 3); vec("accel_draw", an, 3);
        vec("q_next", next.orientation, 4); vec("v_next", next.linear_velocity, 3); vec("omega_next", next.angular_velocity, 3);
        vec("rpm_next", next.rpm, 4);
        vec("gyro_bias_next", next.gyro_bias, 3); vec("accel_bias_next", next.accelerometer_bias, 3);
        vec("gyro", next.gyro, 3); vec("accel", next.accelerometer, 3, false);
        std::printf("}");
        state = next;
    }
    std::printf("]}");
    rlt::free(device, action);
}

int main(){
    std::printf("{");
    bias_sequences();
    std::printf(", ");
    full_steps<false>("imu_step_ou", 7, 0.8, 0.2);
    std::printf(", ");
    full_steps<false>("imu_step_random_walk", 8, 0.0, -1.0);
    std::printf(", ");
    full_steps<true>("imu_step_closed_form_motor", 9, 0.8, 0.2);
    std::printf("}\n");
    return 0;
}
"""


# ---------------------------------------------------------------------------------------
# plain-Python replica of the executed paths (the transcription self-check)
# ---------------------------------------------------------------------------------------

def bias_step(b, w, tau, sigma, dt):
    """70_post_integration.h:66-90."""
    if sigma == 0:
        return list(b)
    if tau > 0:
        alpha = math.exp(-dt / tau)
        sigma_d = sigma * math.sqrt(tau * (1 - alpha * alpha) / 2)
    else:
        alpha = 1.0
        sigma_d = sigma * math.sqrt(dt)
    return [alpha * bi + sigma_d * wi for bi, wi in zip(b, w)]


def rotation_matrix(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def specific_force_fd(q_next, v, v_next, dt, gravity):
    """70_post_integration.h:110-120: R(q⁺)ᵀ·((v⁺ − v)/dt − g) — a finite difference."""
    R = rotation_matrix(q_next)
    a_w = [(v_next[i] - v[i]) / dt - gravity[i] for i in range(3)]
    return [sum(R[j][i] * a_w[j] for j in range(3)) for i in range(3)]


def motor_setpoint(action, case):
    """The setpoint each path uses, AS EXECUTED. The integrated path maps the action,
    c = (clamp(a, −1, 1) + 1)/2·(max − min) + min (operations_generic.h:120-130); the
    CLOSED_FORM path takes the raw action clamped to the action limits
    (70_post_integration.h:307) — the two differ (finding F-31)."""
    lo, hi = case["action_min"], case["action_max"]
    if case["closed_form"]:
        return min(hi, max(lo, action))
    return (min(1.0, max(-1.0, action)) + 1) / 2 * (hi - lo) + lo


def motor_step(u, setpoint, case, i):
    """Asymmetric first-order lag over one step: exact exponential with CLOSED_FORM, the
    RK4 polynomial of the same linear ODE otherwise (the branch cannot switch inside a
    step); then the post-step clamp to the action limits."""
    tau = case["tau_rising"][i] if setpoint >= u else case["tau_falling"][i]
    z = case["dt"] / tau
    decay = math.exp(-z) if case["closed_form"] else 1 - z + z * z / 2 - z**3 / 6 + z**4 / 24
    return min(case["action_max"], max(case["action_min"], setpoint + (u - setpoint) * decay))


def _chk(name, got, want, tol):
    err = max(abs(g - w) for g, w in zip(got, want))
    if err > tol:
        raise AssertionError(f"self-check failed: {name}: max abs error {err:.3e} > {tol}")


def check(doc):
    for ci, case in enumerate(doc["bias_process"]):
        for si, step in enumerate(case["steps"]):
            _chk(f"bias_process[{ci}][{si}]",
                 bias_step(step["b"], step["w"], case["tau"], case["sigma"], case["dt"]),
                 step["b_next"], 1e-15)
    for name in STEP_FAMILIES:
        case = doc[name]
        for si, s in enumerate(case["steps"]):
            tag = f"{name}[{si}]"
            _chk(tag + " gyro bias", bias_step(s["gyro_bias"], s["gyro_w"],
                 case["gyro_bias_tau"], case["gyro_bias_sigma"], case["dt"]),
                 s["gyro_bias_next"], 1e-15)
            _chk(tag + " accel bias", bias_step(s["accel_bias"], s["accel_w"],
                 case["accel_bias_tau"], case["accel_bias_sigma"], case["dt"]),
                 s["accel_bias_next"], 1e-15)
            _chk(tag + " gyro", [s["omega_next"][i] + s["gyro_bias_next"][i] + s["gyro_draw"][i]
                                 for i in range(3)], s["gyro"], 1e-14)
            f = specific_force_fd(s["q_next"], s["v"], s["v_next"], case["dt"], case["gravity"])
            _chk(tag + " accel", [f[i] + s["accel_bias_next"][i] + s["accel_draw"][i]
                                  for i in range(3)], s["accel"], 1e-13)
            s["motor_setpoint"] = [motor_setpoint(a, case) for a in s["action"]]
            _chk(tag + " motor", [motor_step(s["rpm"][i], s["motor_setpoint"][i], case, i)
                                  for i in range(4)], s["rpm_next"], 1e-14)


STEP_FAMILIES = ("imu_step_ou", "imu_step_random_walk", "imu_step_closed_form_motor")


# ---------------------------------------------------------------------------------------
# build + run
# ---------------------------------------------------------------------------------------

def _sha16(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def ensure_repo(repo_arg, workdir: pathlib.Path) -> pathlib.Path:
    if repo_arg:
        repo = pathlib.Path(repo_arg)
    else:
        repo = workdir / "rl-tools"
        repo.mkdir()
        for cmd in (["init", "--quiet"], ["remote", "add", "origin", REPO_URL],
                    ["fetch", "--quiet", "--depth", "1", "origin", PINNED_COMMIT],
                    ["checkout", "--quiet", "FETCH_HEAD"]):
            subprocess.run(["git", "-C", str(repo), *cmd], check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != PINNED_COMMIT:
        raise SystemExit(f"rl-tools checkout is at {head}, expected {PINNED_COMMIT}")
    for rel, want in PINNED_SHA256_16.items():
        got = _sha16(repo / rel)
        if got != want:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")
    return repo


def build_and_run(repo: pathlib.Path, workdir: pathlib.Path) -> dict:
    src = workdir / "driver.cpp"
    exe = workdir / "driver"
    src.write_text(DRIVER_CPP)
    subprocess.run(["g++", *CXX_FLAGS, "-I", str(repo / "include"), str(src), "-o", str(exe)],
                   check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def provenance() -> dict:
    return {
        "source": "rl-tools L2F multirotor environment — EXECUTED via compiled driver",
        "repo_url": REPO_URL,
        "commit": PINNED_COMMIT,
        "pinned_by": "rl-tools/hyperdrone 01eee73 (submodule rl_tools)",
        "file_sha256_16": PINNED_SHA256_16,
        "compiler": "g++ " + " ".join(CXX_FLAGS),
        "vehicle": "L2F default model: crazyflie (parameters/dynamics/crazyflie.h)",
        "noise": "unit-normal and white draws replayed with the reference's own sampler "
                 "(std::normal_distribution<float> on std::mt19937, cast to double) on a "
                 "copy of the generator state taken before each call; not recovered from "
                 "the outputs",
        "self_check": "plain-Python replica of every executed path (bias process, IMU "
                      "composition, finite-difference specific force, motor step) asserted "
                      "<= 1e-13 abs before writing; motor_setpoint fields are produced by "
                      "that validated replica",
        "units_note": "the L2F rotor state `rpm` is in normalized command units "
                      "(action limit 0..1), not a rotor speed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--repo", help="existing rl-tools checkout at the pinned commit")
    args = ap.parse_args()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gen_l2f_"))
    try:
        repo = ensure_repo(args.repo, tmp)
        doc = build_and_run(repo, tmp)
        check(doc)
        args.out.mkdir(parents=True, exist_ok=True)
        out = {"schema": 1, "kind": "l2f_terms", "name": "l2f_imu",
               "provenance": provenance(), **doc}
        path = args.out / "l2f_imu.json"
        path.write_text(json.dumps(out, indent=1))
        print(f"wrote {path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
