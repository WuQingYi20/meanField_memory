# Large-N Scaling Comparison Report

Generated: 2026-03-02 17:21:47

---

## 1. Convergence Rate & Speed vs N (Sweep)

### Convergence Rate by Condition

| N | A_baseline | B_lockin | C_norm_only | D_full |
|---|-----------|---------|------------|--------|
| 1000 | 0% | 0% | 100% | 100% |
| 2000 | 0% | 0% | 100% | 100% |
| 5000 | 0% | 0% | 100% | 100% |
| 10000 | 0% | 0% | 100% | 100% |
| 20000 | 0% | 0% | 100% | 100% |

### Mean Convergence Tick (converged trials)

| N | A_baseline | B_lockin | C_norm_only | D_full |
|---|-----------|---------|------------|--------|
| 1000 | — | — | 110 | 107 |
| 2000 | — | — | 118 | 107 |
| 5000 | — | — | 123 | 113 |
| 10000 | — | — | 125 | 115 |
| 20000 | — | — | 131 | 115 |

### Speed Ratio D_full/A_baseline (behavioral layer)

| N | D/A ratio | A mean tick | D mean tick |
|---|-----------|------------|------------|
| 1000 | 61.5x | 2890 | 47 |
| 2000 | — | — | — |
| 5000 | 177.1x | 9722 | 55 |
| 10000 | — | — | — |
| 20000 | — | — | — |

### Scaling Analysis: convergence_tick ∝ N^α

**D_full_model:** α ≈ 0.028 (log-log slope)
  → **Sub-linear growth** — cascade time largely independent of N

## 2. Ablation Effect Sizes vs N

| N | Dyn_Norm conv% | Dyn_noNorm conv% | None_noNorm conv% | Norm speedup |
|---|----------------|------------------|-------------------|-------------|
| 1000 | 100% | 100% | 0% | 16.6x |
| 2000 | 100% | 100% | 0% | 18.4x |
| 5000 | 100% | 100% | 0% | 29.5x |
| 10000 | 100% | 93% | 0% | 40.0x |
| 20000 | 100% | 77% | 0% | 76.0x |

## 3. Cascade Pathway Metrics vs N

### A2 Pathway: θ=3.0, Φ=0.0

| N | conv% | mean_diss | mean_enf | mean_churn | mean_flip | mean_tick |
|---|-------|-----------|----------|------------|-----------|-----------|
| 1000 | 100% | 312.0 | 0.0 | 27.6 | 254.1 | 108 |
| 5000 | 100% | 1770.9 | 0.0 | 35.8 | 1466.6 | 111 |
| 10000 | 100% | 3891.2 | 0.0 | 39.6 | 3141.3 | 113 |
| 20000 | 100% | 8164.4 | 0.0 | 43.5 | 6735.6 | 114 |

## 4. Granger Causality Stability Across N

Key test: anomaly → sigma (dissolution proxy)

| N | Lag1 F | Lag1 p | Lag2 F | Lag2 p |
|---|--------|--------|--------|--------|
| 1000 | 2.31 | 0.1320 | 9.96 | 0.0001 |
| 5000 | 0.00 | 0.9658 | 4.62 | 0.0122 |
| 10000 | 0.05 | 0.8210 | 6.13 | 0.0031 |
| 20000 | 4.64 | 0.0337 | 3.72 | 0.0278 |

## 5. CUSUM Changepoint Timing Across N

| N | Belief CP | Sigma CP | Confidence CP |
|---|-----------|----------|---------------|
| 1000 | 39 | 47 | 48 |
| 5000 | 41 | 49 | 32 |
| 10000 | 44 | 52 | 37 |
| 20000 | 59 | 25 | 56 |

## 6. Key Findings

*(Auto-populated after all experiments complete)*

1. **Cascade convergence time is sub-linear in N** (α≈0.028)
   - Supports the paper's claim that cascade timing is largely N-independent

