# Comparison Experiment Specification: DualMemory vs EWA

## Overview

This document specifies a systematic comparison between the Dual-Memory (DM) norm emergence model and Experience-Weighted Attraction (EWA, Camerer & Ho 1999) as a baseline competitor. EWA nests reinforcement learning (delta=0) and belief learning (delta=1) as special cases, making it the strongest single-model baseline from behavioral game theory.

**Core hypothesis**: The dual-memory architecture (experiential + normative) outperforms EWA's single-attraction mechanism, particularly in perturbation robustness and small-population convergence.

---

## Models Under Comparison

| Label | Model | Key Parameters |
|-------|-------|---------------|
| `EWA_RL` | EWA, delta=0 | Pure reinforcement learning |
| `EWA_MIX` | EWA, delta=0.5 | Mixed imagination/reinforcement |
| `EWA_BL` | EWA, delta=1 | Pure belief learning |
| `DM_base` | DualMemory, normative OFF | Experience-only (no norms) |
| `DM_full` | DualMemory, normative ON | Full dual-memory with enforcement |

---

## Experiment C1: Steady-State Convergence

**Question**: How do convergence speed and final coordination compare across models?

### Design
- **Models**: EWA_RL, EWA_MIX, EWA_BL, DM_base, DM_full
- **Population**: N=100
- **Duration**: T=3000
- **Trials**: 50 per model condition
- **Seeds**: `seed = hash(("C1", model_label, trial_idx)) % typemax(Int)`

### EWA Parameters
- phi=0.9, rho=0.9, lambda=1.0 (or best from C4)
- convergence_window=50, thresh_majority=0.95, thresh_belief_error=0.10, thresh_belief_var=0.05

### DM Parameters
- Default SimulationParams for DM_base (enable_normative=false)
- DM_full: enable_normative=true, V=3, Phi=1.0

### Metrics
1. **Convergence rate**: fraction of trials reaching convergence
2. **Convergence speed**: median tick at convergence (among converged trials)
3. **Final coordination rate**: mean coordination_rate at last tick
4. **Belief accuracy**: mean belief_error at last tick

### Prediction
DM_full >= DM_base ~ EWA_BL > EWA_MIX > EWA_RL

---

## Experiment C2: Perturbation Robustness

**Question**: After convergence, how do models recover from an exogenous shock?

This is the **core experiment** demonstrating the value of normative memory. Crystallized norms should resist perturbation while EWA's attractions are fragile.

### Design
- **Models**: EWA_BL (best EWA config from C4), DM_full
- **Protocol**:
  1. Run simulation for 500 ticks (both models should converge by then)
  2. At tick 500, inject perturbation: randomly select 20% of agents, reset their state:
     - EWA: set attract_A = attract_B = 0, prob_A = 0.5, N_exp = 1.0
     - DM: set b_exp_A = 0.5, C = C0, w = w_base+floor(C0*(w_max-w_base)), r = NO_NORM, sigma = 0.0, e = 0.0, a = 0
  3. Continue simulation for another 500 ticks (total T=1000)
- **Population**: N=100
- **Trials**: 50 per model

### Metrics
1. **Coordination drop**: coordination_rate at tick 501 minus tick 500
2. **Recovery speed**: first tick after 500 where coordination_rate > 0.90 again
3. **Recovery rate**: fraction of trials that recover within 500 post-shock ticks
4. **Steady-state coordination**: mean coordination_rate over ticks 900-1000

### Prediction
- DM_full: small drop, fast recovery (crystallized agents pull shocked agents back)
- EWA_BL: larger drop, slower recovery (no memory anchor beyond attractions)

---

## Experiment C3: Small Population Scaling

**Question**: How does population size affect convergence reliability?

### Design
- **Models**: EWA_BL (best config), DM_full
- **Population**: N in [20, 50, 100]
- **Duration**: T=3000
- **Trials**: 50 per (model, N) combination

### Metrics
1. **Convergence rate** by N
2. **Convergence speed** by N (median among converged)
3. **Coordination rate variability**: std of final coordination_rate across trials

### Prediction
- DM_full maintains high convergence even at N=20 (norms crystallize and stabilize)
- EWA_BL degrades more at small N (noisier frequency estimates)

---

## Experiment C4: EWA Parameter Sensitivity

**Question**: What is EWA's best-case configuration for pure coordination?

This experiment should run **first** to find EWA's optimal parameters for use in C1-C3.

### Design
- **Model**: EWA only
- **Population**: N=100, T=3000
- **Parameter grid**:
  - delta in [0.0, 0.25, 0.5, 0.75, 1.0]
  - lambda in [0.5, 1.0, 2.0, 5.0]
  - phi = 0.9, rho = 0.9 (fixed)
- **Trials**: 20 per (delta, lambda) combination
- **Total runs**: 5 x 4 x 20 = 400

### Metrics
1. **Convergence rate** per (delta, lambda)
2. **Mean convergence speed** per (delta, lambda)
3. **Best config**: (delta, lambda) with highest convergence rate, tie-break by speed

### Expected Result
delta=1.0 (belief learning) with lambda in [1.0, 2.0] should dominate, confirming that frequency tracking is the strongest EWA strategy for coordination games.

---

## Output Format

All experiments produce CSV files with columns:
```
experiment, model, trial, N, T, [model-specific params], convergence_tick, converged,
final_fraction_A, final_coordination_rate, final_belief_error, total_ticks
```

For C2 (perturbation), additional columns:
```
pre_shock_coord, post_shock_coord, coord_drop, recovery_tick, recovered
```

---

## Implementation Notes

1. **Shared output**: Both models produce `SimulationResult` with `TickMetrics` history, so `summarize()` and CSV export work identically
2. **C2 requires custom run loop**: The perturbation experiment needs to interrupt the simulation at tick 500, modify agents, then continue. This will be a separate script, not part of the core `ewa_run!` or `run!` functions
3. **Execution order**: Run C4 first, extract best EWA config, then run C1, C2, C3
4. **Reproducibility**: All experiments use deterministic seeding via `hash((experiment, model, trial))`
