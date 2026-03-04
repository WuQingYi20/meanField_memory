# Dual-Memory Agent-Based Model — Reference

Agents play a symmetric coordination game {A, B}, equipped with experiential memory (FIFO) and normative memory (DDM-based norm crystallisation).

---

## Formula Table

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 1 | Experiential belief | `b_exp^A = (#A in last w entries) / w` | Frequency-based estimate of strategy A from recent observations |
| 2a | Confidence update (correct) | `C ← C + α(1 − C)` | Additive increase on correct prediction |
| 2b | Confidence update (wrong) | `C ← C(1 − β)` | Multiplicative decay on wrong prediction |
| 3 | Memory window | `w = w_base + ⌊C · (w_max − w_base)⌋` | Maps confidence to observation window size |
| 4 | DDM evidence accumulation | `e ← e + (1 − C) · (2·b_exp^A − 1) + s_push` | Drift-diffusion accumulator, gated by uncertainty |
| 5 | Crystallisation trigger | `\|e\| ≥ θ_crystal → r = sign(e), σ = σ₀` | Norm forms when evidence exceeds threshold |
| 6a | Norm strengthening | `σ ← σ + α_σ(1 − σ)` | Slow additive increase on conforming observation |
| 6b | Crisis decay | `σ ← λ_crisis · σ` (when `a ≥ θ_crisis`) | Sharp multiplicative decay when anomalies accumulate |
| 6c | Dissolution | `r → none, e → 0, σ → 0` (when `σ < σ_min`) | Norm dissolves, agent re-enters DDM |
| 7 | Compliance | `compliance = σ^k` | Nonlinear mapping from norm strength to behavioural commitment |
| 8 | Effective belief | `b_eff^A = compliance · b_norm^A + (1 − compliance) · b_exp^A` | Blends norm direction with experiential estimate |
| 9 | Action selection | `P(play A) = b_eff^A` | Probability matching |
| 10 | Enforcement signal | `s_push = Φ · (1 − C_partner) · γ_signal · direction` | Crystallised agents push violators' DDM toward their norm |

---

## Variable & Parameter Table

| Symbol | Name | Default | Description & Rationale |
|--------|------|---------|------------------------|
| N | Population size | 100 | Baseline; tested N = 4–20,000 |
| T | Max ticks | 3000 | Sufficient for convergence at all tested N |
| α | Confidence increase rate | 0.1 | Slow build; α < β ensures "fast collapse, slow build" (Behrens et al. 2007) |
| β | Confidence decrease rate | 0.3 | Fast collapse on prediction error; asymmetric with α for volatility response |
| C₀ | Initial confidence | 0.5 | Neutral starting point, gives initial w = 4 |
| w_base | Min memory window | 2 | Minimum under high uncertainty; matches small-sample learning (Hertwig & Pleskac 2010) |
| w_max | Max memory window | 6 | Upper bound matches 5–6 trial effective window in human experiments |
| θ_crystal | Crystallisation threshold | 3.0 | DDM evidence threshold for norm formation; robust in [3, 5] |
| σ₀ | Initial norm strength | 0.8 | High enough for immediate compliance (0.8² = 0.64) at crystallisation |
| θ_crisis | Crisis threshold | 10 | Anomaly count before crisis; minority agent ~25 ticks to trigger |
| λ_crisis | Crisis decay factor | 0.3 | First crisis: 0.8→0.24; second crisis: 0.24→0.072 < σ_min → dissolution |
| σ_min | Dissolution threshold | 0.1 | Survives one crisis (0.24 > 0.1), dissolves after two (0.072 < 0.1) |
| α_σ | Norm strengthening rate | 0.005 | Slow recovery (~50 ticks from 0.24 to 0.45); ensures minority norms erode faster than rebuild |
| k | Compliance exponent | 2.0 | Convex: low σ → low compliance, high σ → strong compliance |
| θ_enforce | Enforcement strength threshold | 0.7 | Only well-established norms (σ > 0.7) send enforcement signals |
| γ_signal | Signal amplification | 2.0 | Scales enforcement push relative to organic drift |
| Φ | Enforcement gain | 0.0 | Global switch; disabled by default. Tested Φ ∈ {0, 0.1, …, 5.0} |
| V | Extra observations per tick | 0 | Minimal-information baseline: agent observes only its own partner |

---

*Reference derived from cascade_report_advisor.tex and source code (src/params.jl, src/types.jl, src/stages.jl, src/init.jl).*
