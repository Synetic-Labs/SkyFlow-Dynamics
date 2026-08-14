"""The symbolic Jacobian of the full dynamics must match finite differences of the numeric
evaluation — the strongest single check that the symbolic model and its lambdified form agree,
and the artifact a differentiable backend will consume."""

import numpy as np
import sympy as sp

from properties.helpers import (
    P,
    S,
    U,
    flat_params,
    make_inputs,
    params_dict,
    random_state,
    statedot_expr,
    statedot_fn,
)


def test_jacobian_matches_finite_differences():
    F = statedot_expr()
    diff_vars = list(S.flat()) + U.W_c.flat() + U.v_wind.flat()
    J = F.jacobian(sp.Matrix(diff_vars))
    Jfn = sp.lambdify((S.flat(), U.flat(), P.flat()), J, modules="numpy", cse=True)
    f = statedot_fn()

    vals = params_dict(c_D=[0.02, 0.02, 0.04], k_flap=1e-7, I_rot=3.452e-8)
    p = flat_params(vals)
    rng = np.random.default_rng(123)

    for _ in range(3):
        s = random_state(rng)
        u = make_inputs(W_c=rng.uniform(900, 2300, 4), v_wind=rng.uniform(-2, 2, 3))
        J_sym = np.asarray(Jfn(s, u, p), dtype=float)

        n_s = len(s)
        J_fd = np.zeros_like(J_sym)
        base = np.concatenate([s, u[:4], u[4:7]])
        for k in range(len(base)):
            eps = 1e-6 * max(1.0, abs(base[k]))

            def eval_at(delta, base=base, k=k, n_s=n_s, u=u):
                z = base.copy()
                z[k] += delta
                s_k = z[:n_s]
                u_k = u.copy()
                u_k[:4] = z[n_s:n_s + 4]
                u_k[4:7] = z[n_s + 4:n_s + 7]
                return f(s_k, u_k, p)
            J_fd[:, k] = (eval_at(eps) - eval_at(-eps)) / (2 * eps)

        scale = np.abs(J_sym).max()
        np.testing.assert_allclose(J_sym, J_fd, atol=1e-5 * scale, rtol=1e-5)
