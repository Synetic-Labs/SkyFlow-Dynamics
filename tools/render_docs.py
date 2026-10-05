"""
Render docs/equations.md — the human-readable catalog — from the registry and the canonical
symbolic model. Run after any spec change:

    uv run python tools/render_docs.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from skyflow_dynamics.spec.registry import DECISIONS, EXCLUSIONS, SOURCES, TERMS

DOMAIN_TITLES = {
    "rigid_body": "Rigid body",
    "actuator": "Actuators (motor / ESC / battery)",
    "rotor_aero": "Rotor aerodynamics",
    "frame_aero": "Frame aerodynamics",
    "disturbance": "Disturbances & interaction",
    "sensor": "Sensors",
    "discretization": "Discretization",
    "differentiation": "Differentiable simulation",
    "environment": "Environment (atmosphere / turbulence)",
    "harness": "Harness (timing & stateful machinery — not physics)",
}

HEADER = """# SkyFlow-Dynamics — equation catalog

*Generated from `spec/registry.py` by `tools/render_docs.py` — do not edit by hand.*

The authoritative math lives in the `spec/` modules (each function's docstring carries the
full equations, unit statements, and pitfalls); this catalog is the index. Conventions are
stated in [README.md](../README.md).

## The canonical model

State s = (x, v, q, ω, Ω); input Ω_c; exogenous (v_wind, F_ext, τ_ext):

```
v_a  = R(q)ᵀ (v − v_wind)                        body airspeed
v_i  = v_a + ω × r_i                              local airspeed at rotor hub i
T_i  = (ct0 + ct1·Ω_i + ct2·Ω_i²)·(1 + k_angle·α + k_hor·μ) + k_h·(v_i,x² + v_i,y²)
H_i  = −Ω_i · diag(k_d, k_d, k_z) · v_i
Q_i  = cq0 + cq1·Ω_i + cq2·Ω_i²
F_B  = Σᵢ (T_i·ê_i + H_i) − ‖v_a‖·diag(c_D)·v_a − diag(c_L)·v_a − k_v2·v_az|v_az|·ẑ
M_B  = Σᵢ r_i×(T_i·ê_i + H_i) − Σᵢ s_i·Q_i·ê_i − Σᵢ k_flap·Ω_i·(v_i × ẑ)
       − ω×h + (−I_rot·Σᵢ s_i·Ω̇_i)·ẑ,          h = I_rot·(Σᵢ s_i·Ω_i)·ẑ

ẋ = v                    v̇ = (0,0,−g) + (R(q)·F_B + F_ext)/m
q̇ = ½·q ⊗ (0, ω)        ω̇ = I⁻¹(M_B + τ_ext − ω×(I·ω))
Ω̇ = motor model          (first-order lag or asymmetric spin-up/down)
```

Tier legend: **verified** = reproduces an independent reference (evidence: executed code,
published reference data, measured data, or an exact analytic check); *candidate* = published
model, symbolically checked and cited, awaiting numeric validation; *proposed* = known effect
with provenance, no spec expression yet. **Use**: `backend` = emitted by
`skyflow_dynamics/backends/jax.py`; `spec` = math only. Terms not yet verified-and-in-backend
carry a next-step decision — see [Backlog](#backlog) and [INTAKE.md](../INTAKE.md).
"""

BADGE = {"verified": "**verified**", "candidate": "*candidate*", "proposed": "*proposed*"}


def main():
    lines = [HEADER]
    domains = {}
    for t in TERMS:
        domains.setdefault(t.domain, []).append(t)

    for domain, title in DOMAIN_TITLES.items():
        if domain not in domains:
            continue
        lines.append(f"\n## {title}\n")
        for t in domains[domain]:
            lines.append(f"### `{t.key}` — {BADGE[t.tier]}\n")
            lines.append(f"{t.summary}.\n")
            if t.expression:
                lines.append(f"- **Defined in:** `{t.expression}`")
            else:
                lines.append("- **Defined in:** — (proposed: no spec expression yet)")
            if t.parameters:
                lines.append(f"- **Parameters:** {', '.join(f'`{p}`' for p in t.parameters)}")
            cites = "; ".join(SOURCES[s].citation for s in t.sources)
            lines.append(f"- **Sources:** {cites}")
            if t.tests:
                lines.append(f"- **Tests:** {', '.join(f'`{x}`' for x in t.tests)}")
            if t.domain != "harness":
                ev = ", ".join(t.evidence) or "none yet"
                lines.append(f"- **Evidence:** {ev} · **Use:** `{t.use}`")
            if t.ledger:
                lines.append(f"- **Decision:** {t.decision}")
                if t.revisit:
                    lines.append(f"- **Revisit when:** {t.revisit}")
                if t.effect:
                    lines.append(f"- **Effect:** {t.effect}")
            if t.notes:
                lines.append(f"- **Notes:** {t.notes}")
            lines.append("")

    lines.append("\n## Backlog\n")
    lines.append("Every term short of verified-and-in-backend, by decision (INTAKE.md: "
                 "*pursue* = do now, *defer* = wait for the revisit condition, *hold* = no "
                 "next step planned, *open* = not triaged yet).\n")
    for decision in ("pursue", "defer", "hold", "open"):
        rows = [t for t in TERMS if t.ledger and t.decision == decision]
        if not rows:
            continue
        lines.append(f"\n### {decision} ({len(rows)})\n")
        lines.append("| term | tier | use | revisit when |")
        lines.append("|---|---|---|---|")
        for t in rows:
            lines.append(f"| `{t.key}` | {t.tier} | {t.use} | {t.revisit or '—'} |")
    assert set(DECISIONS) == {"pursue", "defer", "hold", "open"}

    lines.append("\n## Sources\n")
    for s in SOURCES.values():
        url = f" — <{s.url}>" if s.url else ""
        lines.append(f"- **{s.key}**: {s.citation}{url}")

    lines.append("\n## Reviewed and excluded\n")
    for key, reason in EXCLUSIONS:
        lines.append(f"- **{key}**: {reason}")
    lines.append("\nSee [REFERENCES.md](../REFERENCES.md) for the full per-source "
                 "evaluation ledger including skipped models.\n")

    out = pathlib.Path(__file__).resolve().parent.parent / "docs" / "equations.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    n = {tier: sum(1 for t in TERMS if t.tier == tier) for tier in BADGE}
    n_b = sum(1 for t in TERMS if t.use == "backend")
    print(f"wrote {out} ({n['verified']} verified, {n['candidate']} candidate, "
          f"{n['proposed']} proposed terms; {n_b} in the backend)")


if __name__ == "__main__":
    main()
