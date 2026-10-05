# Intake protocol — adding, deferring, and tracking dynamics

This protocol decides what physics enters the spec, what stays out, and what waits. It also
keeps the list of open items. Its purpose is to make source evaluation repeatable, to keep
one ledger of everything missing or unfinished, and to guarantee that anything adopted
arrives with provenance, in canonical conventions, and tested.

## Two ways in

- **A source arrives** — a paper, repository, or dataset ("does this contribute anything to
  our known dynamics?"). Follow steps 1–9.
- **A finding arrives** — a consumer's experiment, a test, or an analysis shows a missing or
  wrong effect (example: F-29, found while building SkyFlow's 5-inch racer). Record it in
  [REFERENCES.md](REFERENCES.md), "Findings from use", with the next F-number and a script
  in this repo that reproduces it (`tools/` or `golden/generate/`). A finding that only a
  private or external script reproduces is a lead, not evidence. Then follow steps 2–9
  with the finding in place of the source.

## The ledger

`spec/registry.py` is the ledger: every term, and every known effect not yet a term, is one
`Term`. `properties/test_registry.py` enforces the fields; `tools/render_docs.py` renders
them, with a **Backlog** section, into [docs/equations.md](docs/equations.md).

| field | values | meaning |
|---|---|---|
| `tier` | `proposed` → `candidate` → `verified` | proposed: known effect with provenance, no spec expression yet. candidate: expression in `spec/`, symbolically checked, cited. verified: reproduces an independent reference (step 7). |
| `evidence` | `executed`, `published`, `measured`, `analytic`, `derived` | how the term was checked (step 7). |
| `use` | `spec`, `backend` | math only, or emitted by `skyflow_dynamics/backends/` (step 8). Checked by tracing what the backend calls. |
| `decision` | `open`, `pursue`, `defer`, `hold` | the next step, for every term not yet final (verified and `backend`). |
| `revisit` | text | the observable condition that reopens a `defer` (required) or a `hold`. |
| `effect` | text | the effect's size at the operating points (step 3); required once triaged. |

Decisions:

- **open** — not triaged yet. This is the queue.
- **pursue** — worth doing now: the next step (expression, verification, or backend) is
  wanted.
- **defer** — worth doing, but not until `revisit` happens. The condition must be something
  a consumer can observe — a measurement, a task, a fit residual — not a date.
- **hold** — no next step planned; the term stays as it is (for example, spec-only by
  design).
- Rejected models do not enter the registry. They go in REFERENCES.md with the reason, and a
  whole rejected source goes in `EXCLUSIONS`.

## 1. Inventory the source

List every dynamics-relevant model in the source: forces, torques, actuator dynamics,
sensor models, disturbance models. For each, record the exact location — paper equation
number/section, or repo file path + line numbers at a specific commit.

## 2. Check coverage against the registry

For each model, search `spec/registry.py` for an existing term. A model is *covered* only if
the math is equivalent **after conversion to canonical conventions** — verify, don't
pattern-match on names. Known traps (each burned us once; the F-* finding IDs are documented
in [REFERENCES.md](REFERENCES.md), "Prior evaluations"):

- [ ] **Units**: RPM vs rad/s (Crazyflow polynomials are in RPM — coefficients scale by
      `60/2π` per power of Ω); degrees vs rad.
- [ ] **Normalization**: coefficients may be mass-normalized (accelerations, no `1/m`) or
      inertia-normalized (no `I⁻¹`) — SkyDreamer's are (finding F-4). Multiply back before comparing.
- [ ] **Frames**: NED vs ENU world; FRD vs FLU body; where gravity's sign lives. Check mixed
      frames inside one expression too (agilib's lever arm × body rate, finding F-24).
- [ ] **Quaternion**: wxyz vs xyzw; Hamilton vs JPL; body→world vs world→body.
- [ ] **Rotor direction convention**: spin sign vs yaw-torque sign (opposite! finding F-6).
- [ ] **Norm conventions**: e.g. quadratic drag with `‖v‖·v` vs per-axis `|vᵢ|·vᵢ` — these are
      structurally different models (found between RotorPy and SkyDreamer).
- [ ] **Sign derivations**: gyroscopic/precession terms — re-derive `−ω × h`, don't trust the
      source's signs (Crazyflow's gyro-x sign is wrong, finding F-3).
- [ ] **Lumped vs structural**: identified lumped coefficients (e.g. per-rotor `k_p·Ω²` moments)
      may be the same physics as a structural `r × F` model — don't double-adopt. The same
      holds for damping: a coefficient set to a model's *total* damping already contains
      every channel (F-29).

A covered model stops here (record it in step 9).

## 3. Triage

Decide whether the model is worth carrying, before any conversion work. Answer five
questions and record the answers in the entry's `effect` and `notes`:

1. **How big is it?** Evaluate the effect at the operating points below, on every reference
   vehicle in `spec/parameters.py` (plus the vehicle the finding came from, if it is not one
   of them). State it as a fraction of the dominant existing term on the same axis: thrust
   for F_z, rotor drag for F_xy, the existing damping derivative for body rates.
2. **What is the evidence?** Executed code, published reference data, measured data, or only
   a paper. A model with no parameter values for any vehicle cannot be used yet.
3. **Can users get the parameters** for their own vehicle — from a datasheet, a thrust stand,
   flight data? A coefficient that the available data cannot separate from another one is
   not identifiable (F-30); say which data would separate it.
4. **What does it cost?** Smoothness and differentiability, stiffness (the time step it
   needs), per-step compute, and double counting against existing terms.
5. **Where does it apply?** The validity envelope the source states.

Operating points (body frame, still air unless stated):

| point | state |
|---|---|
| hover | v = 0, ω = 0, Ω = hover speed |
| cruise | 10 m/s level forward flight (5 m/s for vehicles under 100 g) |
| axial | climb and descent at 0.5 and 2 × the hover induced velocity v_h (covers the vortex-ring band) |
| rate | body rate 1, 10, 50, 200 rad/s about roll, pitch, and yaw, at hover thrust |
| ground | hover at 0.5 and 2 rotor diameters above the ground |

Default size thresholds: **≥ 5 %** at any point of a vehicle's envelope is material; **1–5 %**
is minor — carry it only if it is cheap and identifiable; **< 1 %** everywhere is negligible —
reject it, unless it changes a qualitative behavior (stability, a sign, a limit cycle).

Then decide: land it (steps 4–6), record it as `proposed` with `decision="defer"` and a
`revisit` condition, or reject it (step 9). The bar for the **spec** is credibility and
material size somewhere; the bar for the **backend** is higher (step 8).

## 4. Extract and convert

Write the model in canonical conventions (README): wxyz quaternion, spin-sign rotors, SI,
world-ENU/body-FLU, forces (not accelerations). Define every symbol. Note the validity
envelope (airspeed range, incidence angles) if the source states one.

## 5. Symbolic checks

Before any code lands: dimensional consistency; limiting behavior (reduces to an existing term
when the new effect is switched off); required symmetries (e.g. yaw invariance, mirrored-rotor
antisymmetry); equilibria still solvable. Add these as `properties/` tests.

## 6. Land

Add the expression to the right `spec/` module and a registry entry with `tier="candidate"`,
full citation, parameter values (with units) if the source identifies them, the notes from
steps 2–5, and the triage decision. An effect that passed triage but has no expression yet
lands as `tier="proposed"` with `expression=""`.

## 7. Verify (promote to verified)

A term is `verified` once the spec reproduces an **independent** reference and its property
tests pass. Record the kind in `evidence`:

- **executed** — a generator under `golden/generate/` runs the source's *actual code* to
  freeze reference vectors (record repo commit + params).
- **published** — reference tables or run statistics that the source published, pinned by
  document hash (the archaic-source exception; Dryden via NASA CR-1998-206937).
- **measured** — measured data (flight or test stand) at a fixed dataset version, split, and
  error metric: the term must beat the backend model's held-out error. Check input noise
  first — coefficients that multiply noisy inputs are biased toward zero (F-30).
- **analytic** — an exact closed-form identity or hand-computed values, derived independently
  of the spec function (pure geometry and kinematics).
- **derived** — computed from other spec terms, e.g. a fit to the verified BEM model. It is
  not independent, so it gives values and triage numbers but **never verifies**. It needs a
  test that reproduces it (`properties/test_findings.py`).

## 8. Backend

A term goes into a backend (`use="backend"`) only when it is verified, its effect is material
(step 3) inside an envelope the backend supports, and at least one reference vehicle has its
parameters. `test_registry.py` traces the backend and fails if `use` disagrees with what the
backend actually calls.

## 9. Record the outcome

Add the source or finding to `REFERENCES.md` — including models **rejected** and why
(duplicate, unphysical, negligible, out of scope). A source that contributes nothing still gets
an entry; that's what makes the next evaluation fast. Re-render the catalog:
`uv run python tools/render_docs.py`.

## Review

- A `revisit` condition is checked when it happens: a consumer who observes it reports it as
  a finding, and the decision is reopened.
- At every release, go through the Backlog in docs/equations.md: triage some `open` entries,
  and confirm each `defer` condition still holds.
