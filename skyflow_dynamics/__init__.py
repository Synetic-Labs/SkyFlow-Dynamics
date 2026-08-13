"""
skyflow_dynamics — the installable surface of SkyFlow-Dynamics.

Subpackages:
    skyflow_dynamics.spec      — the SymPy source of truth (symbolic, framework-free).
    skyflow_dynamics.backends  — generated adapters (backends.jax requires the `jax` extra).

The validation machinery (properties/, golden/) stays repo-level and is not shipped:
consumers get the math and the generated backends; the proofs live with the repo.
"""
