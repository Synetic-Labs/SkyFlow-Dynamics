"""Registry integrity: every term well-formed, every expression path resolves to real code,
every `use` field matches what the backend actually calls, reference parameter sets valid
against the schema."""

import importlib
import pathlib
import sys

import pytest

from skyflow_dynamics.spec.parameters import CRAZYFLIE, RACER_5IN, SCHEMA, validate
from skyflow_dynamics.spec.registry import (
    EXCLUSIONS,
    SOURCES,
    TERMS,
    by_key,
    validate_registry,
)


def test_registry_validates():
    validate_registry()


def test_expression_paths_resolve():
    for term in TERMS:
        if term.tier == "proposed" or term.expression.startswith("(harness"):
            continue
        for path in term.expression.split(", "):
            # Registry paths stay in the spec's own namespace ("spec.motor.first_order_lag");
            # they resolve inside the installed package.
            module_path, attr = path.rsplit(".", 1)
            mod = importlib.import_module(f"skyflow_dynamics.{module_path}")
            assert hasattr(mod, attr), f"{term.key}: {path} does not resolve"


def test_sources_have_citations():
    for src in SOURCES.values():
        assert len(src.citation) > 20


def test_exclusions_documented():
    assert len(EXCLUSIONS) >= 1
    for key, reason in EXCLUSIONS:
        assert len(reason) > 20


@pytest.mark.parametrize("name, vehicle", [("CRAZYFLIE", CRAZYFLIE), ("RACER_5IN", RACER_5IN)])
def test_reference_vehicles_valid(name, vehicle):
    validate(vehicle)
    for key in vehicle:
        if key == "limits":
            continue
        assert key in SCHEMA, f"{name} key {key} not in SCHEMA"
    assert set(SCHEMA) <= set(vehicle), f"{name} misses {set(SCHEMA) - set(vehicle)}"


def test_by_key():
    assert by_key("newton_euler").tier == "verified"


def test_listed_tests_exist():
    root = pathlib.Path(__file__).resolve().parent.parent
    for term in TERMS:
        for t in term.tests:
            assert (root / t).is_file(), f"{term.key}: {t} does not exist"


#: Backend functions that implement a term in JAX directly instead of calling its spec
#: function, each golden-tested in test_backend_jax.py.
BACKEND_EQUIVALENTS = {"motor_exact_exp_discretization": "exact_exp_step_fn"}


def test_use_matches_backend():
    """A term is use='backend' iff the JAX backend calls one of its spec functions (traced
    while building and stepping every backend function) or implements it directly."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    import skyflow_dynamics.spec as spec_pkg
    from skyflow_dynamics.backends import jax as B

    spec_dir = str(pathlib.Path(spec_pkg.__file__).resolve().parent)
    called = set()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.startswith(spec_dir):
            module = pathlib.Path(frame.f_code.co_filename).stem
            called.add(f"spec.{module}.{frame.f_code.co_name}")

    factories = (B.statedot_fn, B.rk4_step_fn, B.exact_exp_step_fn, B.imu_fn,
                 B.throttle_to_speed_fn)
    for f in factories:
        f.cache_clear()
    p = B.pack_params(CRAZYFLIE)
    s = B.pack_state([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                     [0.1, 0.0, 0.0], [2000.0] * 4)
    u = B.pack_inputs([2100.0] * 4)
    sys.setprofile(profile)
    try:
        B.statedot_fn(4, "asymmetric")(s, u, p)
        B.rk4_step_fn(4, "first_order")(s, u, p, 1e-3)
        B.exact_exp_step_fn(4)(s, u, p, 1e-3)
        B.imu_fn(4, "first_order")(s, u, p, jnp.zeros(3), jnp.eye(3).reshape(-1))
        B.throttle_to_speed_fn()(0.5, 100.0, 2000.0, 0.5)
    finally:
        sys.setprofile(None)

    for term in TERMS:
        if term.domain == "harness":
            continue
        uses = any(path in called for path in term.expression.split(", ") if path)
        uses = uses or term.key in BACKEND_EQUIVALENTS
        assert (term.use == "backend") == uses, \
            f"{term.key}: use={term.use!r} but the backend {'calls' if uses else 'never calls'} it"
