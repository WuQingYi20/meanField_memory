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
| 4 | DDM evidence accumulation | `e ← e + (1 − C) · f_diff + s_push` | Drift-diffusion accumulator; `f_diff = b_exp^A − b_exp^B` |
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

| Symbol | Name | Default | Description & Rationale | Sweep Status |
|--------|------|---------|------------------------|--------------|
| N | Population size | 100 | Baseline (Behrens et al. 2007) | Swept {20–20,000}. O(1) scaling (N^0.028) with norms; A fails at N≥200 |
| T | Max ticks | 3000 | Sufficient for convergence at all tested N | — |
| α | Confidence increase rate | 0.1 | Slow build; α < β for "fast collapse, slow build" (Behrens et al. 2007) | Swept β with α=0.1 fixed. C/D insensitive; only B benefits from higher β |
| β | Confidence decrease rate | 0.3 | Fast collapse on prediction error; asymmetric with α | Swept {0.15–0.7}. Normative layer makes β irrelevant |
| C₀ | Initial confidence | 0.5 | Neutral starting point, gives initial w = 4 | — |
| w_base | Min memory window | 2 | Minimum under high uncertainty (Hertwig & Pleskac 2010) | **No sweep.** [2,6] from literature but sensitivity untested |
| w_max | Max memory window | 6 | Upper bound matches 5–6 trial effective window | **No sweep.** Same as above |
| θ_crystal | Crystallisation threshold | 3.0 | DDM evidence threshold for norm formation | Swept {0.5–20} + θ×N grid. **Sweet spot [2,3]**; phase transition at θ≈7 |
| σ₀ | Initial norm strength | 0.8 | Immediate compliance (0.8²=0.64) at crystallisation | **No sweep.** Interacts with λ_crisis, σ_min |
| θ_crisis | Crisis threshold | 10 | Anomaly count before crisis; ~25 ticks in 60-40 minority | **No sweep.** Hand-tuned for "survive 1, dissolve after 2" |
| λ_crisis | Crisis decay factor | 0.3 | 0.8→0.24→0.072 < σ_min → dissolution | **No sweep.** Same as above |
| σ_min | Dissolution threshold | 0.1 | Survives one crisis (0.24>0.1), dissolves after two (0.072<0.1) | **No sweep.** Same as above |
| α_σ | Norm strengthening rate | 0.005 | Slow recovery (~50 ticks 0.24→0.45); cascade asymmetry | **No sweep.** Assumed critical but untested |
| k | Compliance exponent | 2.0 | Convex: low σ→low compliance, high σ→strong compliance | **No sweep.** k=1 vs k=3 could change cascade dynamics |
| θ_enforce | Enforcement strength threshold | 0.7 | Only strong norms (σ>0.7) enforce | **No sweep.** Only tested indirectly via Φ |
| γ_signal | Signal amplification | 2.0 | Scales enforcement push relative to organic drift | **No sweep.** Same as above |
| Φ | Enforcement gain | 0.0 | Global switch; disabled by default | Swept {0–5} + θ×Φ phase boundary. **Blind enforcement trap**: V=0+Φ>0 catastrophic |
| V | Extra observations per tick | 0 | Minimal-information: agent observes only own partner | Swept {0–10}×Φ. V≥1 rescues enforcement; V>3 diminishing returns |

---

*Reference derived from cascade_report_advisor.tex and source code (src/params.jl, src/types.jl, src/stages.jl, src/init.jl).*
