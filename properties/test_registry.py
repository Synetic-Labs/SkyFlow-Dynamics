"""Registry integrity: every term well-formed, every expression path resolves to real code,
reference parameter sets valid against the schema."""

import importlib

from skyflow_dynamics.spec.parameters import CRAZYFLIE, SCHEMA, validate
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
        if term.expression.startswith("(harness"):
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


def test_crazyflie_reference_valid():
    validate(CRAZYFLIE)
    for key in CRAZYFLIE:
        if key == "limits":
            continue
        assert key in SCHEMA, f"CRAZYFLIE key {key} not in SCHEMA"


def test_by_key():
    assert by_key("newton_euler").tier == "verified"
