"""
Generated backend adapters — the spec (spec/) rendered into target frameworks.

Backends are a code-generation step, not a rewrite: each module emits numeric functions from
the same SymPy expressions via sympy.lambdify / printers, and is correct iff it reproduces the
same golden vectors (golden/vectors/) and passes the same property suite (properties/).

Available: backends.jax (requires the `jax` extra: `uv sync --extra jax`).
"""
