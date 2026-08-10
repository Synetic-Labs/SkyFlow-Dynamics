# Intake protocol — evaluating a new source

When a new paper, repository, or dataset lands ("does this contribute anything to our known
dynamics?"), follow this protocol. Its purpose is to make source evaluation repeatable and to
guarantee that anything adopted arrives with provenance, in canonical conventions, and tested.

## 1. Inventory the source

List every dynamics-relevant model in the source: forces, torques, actuator dynamics,
sensor models, disturbance models. For each, record the exact location — paper equation
number/section, or repo file path + line numbers at a specific commit.

## 2. Check coverage against the registry

For each model, search `spec/registry.py` for an existing term. A model is *covered* only if
the math is equivalent **after conversion to canonical conventions** — verify, don't
pattern-match on names. Known traps (each burned us once; see the finding IDs in
`golden/README.md`):

- [ ] **Units**: RPM vs rad/s (Crazyflow polynomials are in RPM — coefficients scale by
      `60/2π` per power of Ω); degrees vs rad.
- [ ] **Normalization**: coefficients may be mass-normalized (accelerations, no `1/m`) or
      inertia-normalized (no `I⁻¹`) — SkyDreamer's are (finding F-4). Multiply back before comparing.
- [ ] **Frames**: NED vs ENU world; FRD vs FLU body; where gravity's sign lives.
- [ ] **Quaternion**: wxyz vs xyzw; Hamilton vs JPL; body→world vs world→body.
- [ ] **Rotor direction convention**: spin sign vs yaw-torque sign (opposite! finding F-6).
- [ ] **Norm conventions**: e.g. quadratic drag with `‖v‖·v` vs per-axis `|vᵢ|·vᵢ` — these are
      structurally different models (found between RotorPy and SkyDreamer).
- [ ] **Sign derivations**: gyroscopic/precession terms — re-derive `−ω × h`, don't trust the
      source's signs (Crazyflow's gyro-x sign is wrong, finding F-3).
- [ ] **Lumped vs structural**: identified lumped coefficients (e.g. per-rotor `k_p·Ω²` moments)
      may be the same physics as a structural `r × F` model — don't double-adopt.

## 3. Extract and convert

Write the model in canonical conventions (README): wxyz quaternion, spin-sign rotors, SI,
world-ENU/body-FLU, forces (not accelerations). Define every symbol. Note the validity
envelope (airspeed range, incidence angles) if the source states one.

## 4. Symbolic checks

Before any code lands: dimensional consistency; limiting behavior (reduces to an existing term
when the new effect is switched off); required symmetries (e.g. yaw invariance, mirrored-rotor
antisymmetry); equilibria still solvable. Add these as `properties/` tests.

## 5. Land as candidate

Add the expression to the right `spec/` module and a registry entry with `tier='candidate'`,
full citation, parameter values (with units) if the source identifies them, and the notes from
steps 2–4.

## 6. Promote to verified

If the source is runnable, write a generator under `golden/generate/` that executes the
source's *actual code* to freeze reference vectors (record repo commit + params). The term is
`verified` once the spec reproduces those vectors and its property tests pass.

## 7. Record the outcome

Add the source to `REFERENCES.md` — including models **rejected** and why (duplicate, unphysical,
out of scope). A source that contributes nothing still gets an entry; that's what makes the next
evaluation fast.
