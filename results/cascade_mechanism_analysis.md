# Normative Cascade Mechanism: Deep Analysis

**Data source**: Single trial diagnostic (N=100, seed=42, Dynamic+Norm, V=0, Phi=0.0)
**Script**: `scripts/diagnose_cascade.jl`
**Raw data**: `results/cascade_diagnostic.csv`

## Overview

The normative system achieves ~7-39x speedup over experiential learning alone. This analysis traces the exact mechanism tick-by-tick, revealing a **five-phase process** with a critical phase transition (cascade) that accounts for the speed advantage.

## Five Phases

### Phase 1: Symmetric Random (tick 1-6)

All 100 agents are uncrystallized. `b_exp_A ~ 0.5`, actions are pure random.

| tick | frac_A | uncr | crA | crB | b_exp_all | C_all |
|------|--------|------|-----|-----|-----------|-------|
| 1 | 0.49 | 100 | 0 | 0 | 0.490 | 0.458 |
| 3 | 0.50 | 100 | 0 | 0 | 0.478 | 0.379 |
| 6 | 0.52 | 100 | 0 | 0 | 0.520 | 0.324 |

- DDM accumulates evidence from random partner actions (symmetric random walk)
- Confidence C drops from 0.46 to 0.32 (predictions are coin-flips, ~50% accuracy)
- No signal, no learning, pure noise

### Phase 2: Initial Crystallization -- Symmetric Split (tick 7-30)

DDM random walks cross threshold (theta_crystal = 3.0). Agents crystallize roughly evenly to A and B because the environment is still ~50/50.

| tick | frac_A | uncr | crA | crB | b_exp_all | sigma_A | sigma_B | anom_A | anom_B |
|------|--------|------|-----|-----|-----------|---------|---------|--------|--------|
| 10 | 0.48 | 82 | 11 | 7 | 0.485 | 0.80 | 0.80 | 0.8 | 0.4 |
| 16 | 0.49 | 60 | 20 | 20 | 0.493 | 0.80 | 0.80 | 2.3 | 2.3 |
| 24 | 0.49 | 38 | 30 | 32 | 0.486 | 0.75 | 0.79 | 4.1 | 3.9 |
| 30 | 0.50 | 23 | 39 | 38 | 0.495 | 0.66 | 0.69 | 4.2 | 4.0 |

Key observations:
- **crA ~ crB**: the split is nearly symmetric (39 vs 38 at tick 30)
- **sigma decays for both sides**: conflicting norms generate mutual anomalies
- **b_exp_all ~ 0.49**: experiential memory has barely broken symmetry yet
- The crystallized agents' `b_eff` creates a small bimodal structure (b_eff_crA ~ 0.86, b_eff_crB ~ 0.12) but they cancel out at the population level

### Phase 3: Tipping -- Experiential Memory Breaks Symmetry (tick 30-55)

This is the critical phase. Experiential memory slowly shifts `b_exp` in one direction. Meanwhile, the minority norm accumulates anomalies faster and begins dissolving.

| tick | frac_A | uncr | crA | crB | b_exp | sigma_A | sigma_B | anom_A | anom_B | diss |
|------|--------|------|-----|-----|-------|---------|---------|--------|--------|------|
| 35 | 0.50 | 21 | 40 | 39 | 0.500 | 0.56 | 0.59 | 4.7 | 4.7 | 0 |
| 40 | 0.51 | 13 | 46 | 41 | 0.546 | 0.52 | 0.44 | 4.5 | 4.4 | 1 |
| 43 | 0.53 | 12 | 48 | 40 | 0.528 | 0.49 | 0.44 | 4.3 | 5.0 | **3** |
| 48 | 0.53 | 20 | 45 | 35 | 0.534 | 0.48 | 0.35 | 4.7 | 4.1 | **4** |
| 50 | 0.61 | 22 | 44 | 34 | 0.588 | 0.46 | 0.34 | 5.0 | 4.7 | 1 |
| 55 | 0.66 | 33 | 40 | 27 | 0.661 | 0.51 | 0.32 | 4.2 | 6.8 | **5** |

The feedback loop at work:
1. **b_exp drifts** from 0.50 to 0.58 (experiential memory picks up a slight A-majority)
2. **B-norms dissolve faster** (sigma_B: 0.59 -> 0.32, while sigma_A: 0.56 -> 0.51)
3. **Dissolutions appear**: tick 43 onwards, 1-5 dissolutions per tick, mostly B-norms
4. **Dissolved agents re-enter the pool** as uncrystallized (uncr rises: 12 -> 33)
5. **anom_B diverges from anom_A**: B-norms face 6.8 anomalies vs A-norms' 4.2 at tick 55
6. **crB falls**: 41 -> 27, while crA holds at ~40-44

The asymmetry is subtle but self-reinforcing:
- Each B dissolution removes one agent from the B-side
- The dissolved agent sees an A-leaning environment -> re-crystallizes to A
- This makes the environment even more A-leaning -> more B anomalies -> more B dissolutions

### Phase 4: Cascade -- Avalanche (tick 55-72)

Once the asymmetry reaches a critical threshold (~60% A-crystallized), the cascade becomes self-sustaining and explosive.

| tick | frac_A | uncr | crA | crB | new_crA | diss | b_exp | sigma_A | anom_B |
|------|--------|------|-----|-----|---------|------|-------|---------|--------|
| 55 | 0.66 | 33 | 40 | 27 | 1 | 5 | 0.661 | 0.51 | 6.8 |
| 58 | 0.64 | 40 | 42 | 18 | 2 | 4 | 0.684 | 0.55 | 7.4 |
| 60 | 0.70 | 46 | 44 | **10** | 3 | 4 | 0.716 | 0.57 | 5.8 |
| 63 | 0.80 | 43 | **52** | 5 | 4 | 0 | 0.784 | 0.60 | 5.4 |
| 65 | 0.84 | 38 | **58** | 4 | 4 | 2 | 0.825 | 0.64 | 6.5 |
| 67 | 0.91 | 29 | **69** | 2 | **6** | 1 | 0.867 | 0.67 | 6.5 |
| 69 | 0.95 | 17 | **81** | 2 | 5 | 0 | 0.916 | 0.68 | 8.5 |
| 71 | 0.92 | 12 | **88** | **0** | 3 | 1 | 0.945 | 0.69 | -- |
| 72 | 0.96 | 8 | **92** | 0 | 4 | 0 | 0.952 | 0.70 | -- |

The cascade anatomy:
- **B-norms collapse**: 27 -> 10 -> 5 -> 2 -> 0 in ~15 ticks
- **anom_B explodes**: reaches 7-9 (near theta_crisis=10), triggering rapid crisis/dissolution
- **A re-crystallizations surge**: dissolved agents see 70-95% A environment -> DDM quickly crystallizes to A
- **sigma_A starts recovering**: with most agents on the same side, fewer anomalies for A-norms (0.51 -> 0.70)
- **b_exp surges**: 0.66 -> 0.95 as the FIFO fills with A observations

**Cascade speed**: crB goes from 27 to 0 in just **16 ticks** (tick 55-71). This is the source of the speedup.

### Phase 5: Lock-in and Strengthening (tick 72+)

| tick | frac_A | coord | crA | b_exp | C_all | sigma_A |
|------|--------|-------|-----|-------|-------|---------|
| 75 | 1.00 | 1.00 | 97 | 0.987 | 0.718 | 0.71 |
| 80 | 1.00 | 1.00 | 98 | 1.000 | 0.832 | 0.72 |
| 100 | 1.00 | 1.00 | 100 | 1.000 | 0.980 | 0.75 |
| 150 | 1.00 | 1.00 | 100 | 1.000 | 1.000 | 0.80 |
| 200 | 1.00 | 1.00 | 100 | 1.000 | 1.000 | 0.85 |

- Perfect behavioral convergence (frac_A = 1.0, coordination = 1.0)
- Confidence C recovers to 1.0 (predictions now always correct)
- sigma slowly increases (0.71 -> 0.85): the norm strengthens through continued conformity
- The norm is fully institutionalized -- both experiential and normative memory agree

## The Dual-Memory Feedback Loop

```
Experiential Memory (slow, continuous)          Normative Memory (fast, discrete)
┌──────────────────────────────────┐           ┌──────────────────────────────────────┐
│ FIFO accumulates partner actions │           │ DDM accumulates observation evidence  │
│ b_exp_A drifts toward majority   │──(1)──>   │ Crystallization to dominant norm      │
│ Positive feedback: action->obs   │           │ sigma^k compliance amplifies signal   │
│ Speed: ~200 ticks for N=100      │           │                                      │
└──────────────────────────────────┘           │ Minority norm: anomaly accumulation  │
        ^                                      │ -> crisis -> dissolution             │
        │                                      │ -> re-enter DDM -> re-crystallize    │
        │                              (2)     │    to majority norm                  │
        └──────────────────────────────────────│ = ONE-WAY RATCHET                   │
                                               │                                      │
                                               │ Cascade: 15 ticks to sweep N=100     │
                                               └──────────────────────────────────────┘

(1) Experiential breaks symmetry: provides the initial ~55/45 signal
(2) Normative amplifies: crystallization cascade converts the signal to 100/0 consensus
```

## Why the Cascade is Fast: Quantitative Argument

**Without normative memory** (Exp_only):
- Each agent updates b_exp from FIFO window (~5 observations)
- Drift rate per tick: ~1/w * (majority - 0.5), very small when majority is near 0.50
- Convergence is a slow diffusion process: O(N) ticks because each agent independently drifts

**With normative memory** (Both):
- Phase 1-3 (symmetry breaking): ~50 ticks -- same slow experiential drift
- Phase 4 (cascade): ~15 ticks -- explosive phase transition
  - Each B-dissolution flips one agent, increasing the A-majority
  - This makes the environment more A-biased for ALL remaining B-agents simultaneously
  - The cascade is a **collective** process, not agent-by-agent diffusion
  - Speed determined by anomaly accumulation rate to theta_crisis, not by N

**The key insight**: Experiential learning is a **diffusion** process (slow, O(N)). The normative cascade is a **phase transition** (fast, weakly dependent on N). The normative system converts a local, gradual process into a global, abrupt one.

This explains the scaling results:
- Exp_only: 76t (N=20) -> 231t (N=100) -> 467t (N=500) -- linear in N
- Both: 25t (N=20) -> 37t (N=100) -> 43t (N=500) -- sublinear, because the cascade phase is almost N-independent

## Per-Group Effective Belief (b_eff_A) Trajectory

This table shows how the three subgroups' actual action probabilities evolve:

| tick | b_eff uncr | b_eff crA | b_eff crB | Phase |
|------|-----------|-----------|-----------|-------|
| 1 | 0.500 | -- | -- | Random |
| 10 | 0.512 | **0.886** | **0.026** | Crystallization: bimodal split |
| 20 | 0.480 | **0.860** | **0.125** | Symmetric conflict |
| 30 | 0.478 | 0.686 | 0.224 | sigma decaying, signals weakening |
| 40 | 0.449 | 0.709 | 0.431 | B-compliance collapsing |
| 50 | 0.572 | 0.711 | 0.402 | Tipping: b_exp breaks symmetry |
| 60 | 0.570 | **0.872** | 0.673 | Cascade starting: A recovering |
| 70 | 0.853 | **0.964** | 0.940 | Cascade complete: near-unanimity |
| 80 | 1.000 | **0.999** | -- | Lock-in |

Crystallized-A agents consistently act A with 70-96% probability, providing a strong directional signal. As B-norms dissolve and re-crystallize to A, this signal overwhelms the system.
