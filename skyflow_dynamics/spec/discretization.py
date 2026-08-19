"""
Discretization schemes — how the continuous model ṡ = f(t, s) is advanced in time.

These are part of the spec because a differentiable simulator's gradient quality depends on
the discretization as much as on the physics:

- **Fixed-step RK4** (below) is the reference integrator. It is a single, static composition
  of RHS evaluations, hence cleanly differentiable. Adaptive solvers (RK45/dopri5) are fine
  for non-differentiable forward simulation but must NOT be differentiated through: their
  step-size controller branches on local error estimates, so the unrolled computation graph is
  data-dependent and reverse-mode AD through it is ill-defined/ill-conditioned.
- **Exact-exponential motor step** (spec.motor.exact_exp_step) removes the stiffest mode from
  the numerical integrator via operator splitting: inside the RK stages the rotor speed is the
  analytic Ω(t) = Ω_c + (Ω₀−Ω_c)e^(−t/τ) (and Ω̇(t) = (Ω_c−Ω(t))/τ feeds the rotor-inertia
  moment); after the step Ω is set to Ω(dt) in closed form. Per-step gradient factor
  e^(−dt/τ) ∈ (0,1) — a contraction — versus explicit Euler's (1 − dt/τ).
- **Surrogate gradients** for BPTT are specified in spec.simplified.

Source: rpg_flightning (ICRA 2025) for the exact-exp/surrogate machinery; classical RK4.
"""


def rk4_step(f, s, h, t0=0):
    """
    One classical fixed-step RK4 step for ṡ = f(t, s), generic over the numeric/symbolic type
    of s (works for numpy arrays, sympy Matrices, jax arrays…):

        k₁ = f(t₀, s)            k₂ = f(t₀+h/2, s + h/2·k₁)
        k₃ = f(t₀+h/2, s + h/2·k₂)   k₄ = f(t₀+h, s + h·k₃)
        s⁺ = s + h/6 · (k₁ + 2k₂ + 2k₃ + k₄)

    f receives the stage time so operator-split terms (exact-exp motor) can evaluate their
    closed forms at intra-step times.
    """
    k1 = f(t0, s)
    k2 = f(t0 + h / 2, s + (h / 2) * k1)
    k3 = f(t0 + h / 2, s + (h / 2) * k2)
    k4 = f(t0 + h, s + h * k3)
    return s + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def semi_implicit_euler_step(f, s, h, vel_idx, pos_idx, t0=0):
    """
    One semi-implicit (symplectic) Euler step, split into a velocity group (v, ω, Ω — the
    slots whose derivatives are forces/accelerations) and a position group (x, q — the slots
    whose derivatives are velocities):

        s¹ = s  with  s¹[vel] = s[vel] + h·f(t₀, s)[vel]        (velocities first)
        s⁺ = s¹ with  s⁺[pos] = s[pos] + h·f(t₀+h, s¹)[pos]      (positions with NEW velocities)

    First-order accurate like explicit Euler, but the position update uses the updated
    velocities, which preserves the symplectic structure on the mechanical (x, v) part —
    the energy of the discrete trajectory stays bounded instead of drifting. This is the
    integration scheme NeuroBEM's evaluation uses (1 ms steps) and agilib implements;
    note the quaternion slot is advanced with the NEW ω but the OLD q (q̇ = ½·q ⊗ (0, ω⁺)).

    `vel_idx`/`pos_idx` are index arrays (or slices) into the flat state. Generic over the
    numeric type of s (numpy, jax). Source: agilib integrator_symplectic_euler.cpp
    (executed); NeuroBEM (arXiv:2106.08015) §IV-D.
    """
    s1 = s.copy()
    s1[vel_idx] = s[vel_idx] + h * f(t0, s)[vel_idx]
    s2 = s1.copy()
    s2[pos_idx] = s[pos_idx] + h * f(t0 + h, s1)[pos_idx]
    return s2
