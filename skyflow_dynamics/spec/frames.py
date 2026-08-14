"""
Reference frames and basic vector algebra.

World frame W: right-handed, ẑ_W up; gravity acts along −ẑ_W.
Body frame B: right-handed, x̂_B forward, ŷ_B left, ẑ_B up through the rotor plane.
A quantity's frame is part of its definition and is noted at every definition site.
"""

import sympy as sp

#: Body-frame unit vectors.
EX = sp.Matrix([1, 0, 0])
EY = sp.Matrix([0, 1, 0])
EZ = sp.Matrix([0, 0, 1])


def hat(v: sp.Matrix) -> sp.Matrix:
    """Skew-symmetric (cross-product) matrix: hat(v) @ u == v × u."""
    x, y, z = v.flat()
    return sp.Matrix([
        [0,  -z,  y],
        [z,   0, -x],
        [-y,  x,  0],
    ])


def cross(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    """Cross product a × b of two 3-vectors (column Matrix form)."""
    return hat(a) * b


def gravity_world(g: sp.Expr | float) -> sp.Matrix:
    """Gravitational acceleration in the world frame: (0, 0, −g), g > 0 in m/s²."""
    return sp.Matrix([0, 0, -g])
