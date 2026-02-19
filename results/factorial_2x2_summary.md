# 2x2 Factorial Experiment Results

**Date**: 2026-02-19
**Script**: `scripts/run_factorial.jl`
**Raw data**: `results/factorial_2x2.csv`

## Design

2x2 factorial: Memory Type (fixed w=5 vs dynamic [2,6]) x Normative Layer (OFF vs ON)

| Condition | w_base | w_max | enable_normative |
|-----------|--------|-------|-----------------|
| A: Baseline | 5 | 5 | false |
| B: Lock-in Only | 2 | 6 | false |
| C: Normative Only | 5 | 5 | true |
| D: Full Model | 2 | 6 | true |

**Fixed parameters**: N=100, T=3000, V=0, Phi=0.0, alpha=0.1, beta=0.3, C0=0.5, base_seed=1000
**Trials**: 50 per condition (200 total)
**End condition**: Norm level 5 (institutional) or T reached

## Results

| Metric | A: Baseline | B: Lock-in | C: Normative | D: Full Model |
|--------|-------------|------------|--------------|---------------|
| Max norm level (median) | 3 | 3 | 5 | 5 |
| Max norm level (range) | [0, 3] | [3, 3] | [5, 5] | [5, 5] |
| L5 convergence rate | 0% | 0% | 100% | 100% |
| Mean convergence tick | — | — | 289.8 | 279.6 |
| Median convergence tick | — | — | 286.5 | 277.5 |
| Convergence tick range | — | — | [269, 327] | [263, 326] |
| Final majority (mean) | 0.974 | 1.000 | 1.000 | 1.000 |
| Total ticks (mean) | 3000 | 3000 | 289.8 | 279.6 |

## Key Findings

1. **Normative layer is necessary and sufficient for Level 5.** Conditions A and B (normative OFF) never reach L5 regardless of memory type. Conditions C and D (normative ON) reach L5 in 100% of trials.

2. **Dynamic memory improves behavioral convergence without norms.** B always reaches Level 3 (shared beliefs); A sometimes fails (range [0, 3]). Dynamic memory's confidence-driven window creates stronger lock-in.

3. **Dynamic memory provides modest speed-up with norms.** D converges ~10 ticks faster than C on average (280 vs 290). The effect is small at V=0.

4. **All conditions achieve behavioral consensus.** Even A reaches 97.4% majority on average. The difference is in norm *depth* — whether the consensus is backed by internalized rules (L4-5) or just statistical drift (L1-3).

5. **V=0 limits the memory type effect.** With no additional observations, the normative layer dominates. The dynamic memory advantage may be more pronounced at higher V or under environmental shocks (Phi > 0).
