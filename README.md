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

## Parameter Sweep Summary

### Completed Sweeps

| Sweep | Dimensions | Range | Trials | Data File |
|-------|-----------|-------|--------|-----------|
| Ablation 3×2 | exp_level × norm_on × N | 3×2×{20–20,000} | 30 each | `ablation_3x2_summary.csv` |
| N scaling | N | {20, 50, 100, 200, 500} + {1k–20k} | 30 each | `sweep_N.csv`, `large_N/sweep_N_extended.csv` |
| θ_crystal | θ | {0.5, 1, 2, 3, 5, 8, 12} + fine grid | 30 each | `sweep_theta_crystal.csv`, `theta_crystal_fine.csv` |
| θ × N interaction | θ × N | {1,3,5,8,12} × {50,100,200,500} | 30 each | `theta_crystal_x_N.csv` |
| α/β asymmetry | β | {0.15, 0.2, 0.3, 0.5, 0.7}, α=0.1 fixed | 30 each | `sweep_alpha_beta.csv` |
| V × Φ | V × Φ | {0,1,3,5,10} × {0,0.5,1,2} | 30 each | `sweep_V_Phi.csv` |
| Phase boundary | θ × Φ | 2D grid | 30 each | `cascade_phase_boundary.csv` |
| Φ non-monotonicity | Φ | fine grid at multiple θ | 30 each | `cascade_phi_nonmonotonic.csv` |
| Network topology | topology × degree × rewire | complete/ring/small-world | 30 each | `network_topology_sweep.csv` |
| Cascade pathways | θ × Φ | detailed per-agent metrics | 30 each | `cascade_pathways.csv` |
| Causal analysis | Granger causality | N={100–20,000} | — | `causal_analysis_summary.csv` |

### Key Conclusions

1. **Mechanism hierarchy**: Normative memory is the primary driver (23x speedup); lock-in adds marginal benefit (30x total). Without normative memory, convergence never occurs (0%).
2. **θ_crystal sweet spot [2, 3]**: Too low → premature 50/50 split slows convergence; too high → confidence-DDM paradox (C→1 kills drift→0). Phase transition at θ ≈ 7.
3. **O(1) scaling**: Convergence tick scales as N^0.028 ≈ O(1) with normative memory. Without it, condition A fails at N ≥ 200.
4. **α/β insensitive**: Normative layer makes confidence asymmetry irrelevant; only lock-in-only (B) benefits from higher β.
5. **Blind enforcement trap**: V=0 + Φ>0 is catastrophic (convergence drops to 10–27%). V ≥ 1 completely rescues. Information must precede enforcement.
6. **Norm churn**: Full model (D) exhibits a turbulent phase (tick 30–50) where norms form/dissolve before stabilising — emergent, not designed.

### Gaps & Missing Sweeps

- **θ_crisis, λ_crisis, σ_min**: No systematic sweep. Currently set by hand to produce "survive one crisis, dissolve after two". Need to verify robustness range.
- **α_σ (norm strengthening rate)**: No sweep. Slow recovery is assumed critical for cascade asymmetry but untested.
- **k (compliance exponent)**: No sweep. k=2 is assumed; k=1 (linear) vs k=3 (sharper) could change cascade dynamics.
- **σ₀ (initial norm strength)**: No sweep. Interacts with λ_crisis and σ_min in determining crisis survival count.
- **w_base × w_max interaction**: No sweep beyond the fixed [2, 6]. Literature supports this range but model sensitivity is untested.
- **θ_enforce × γ_signal**: No independent sweep. These two jointly control enforcement strength but are only tested indirectly via Φ.
- **Heterogeneous agents**: All sweeps use homogeneous populations. Distributions of θ_crystal or θ_crisis across agents are unexplored.
- **Large-N phase boundary**: θ × Φ boundary only computed at N={100, 1000, 5000}. Missing N={10,000, 20,000}.

---

*Reference derived from cascade_report_advisor.tex and source code (src/params.jl, src/types.jl, src/stages.jl, src/init.jl).*
