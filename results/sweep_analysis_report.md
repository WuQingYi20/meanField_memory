# Dual-Memory Model: Parameter Sweep Analysis Report

> Generated 2026-02-19. Based on Julia implementation (`DualMemoryABM` module, spec v5.1).

---

## 1. Experimental Design

### 1.1 Model Overview

The model features two memory systems for norm emergence in a coordination game:
- **Experience memory**: FIFO buffer with confidence-gated dynamic window (w_base to w_max)
- **Normative memory**: DDM-based crystallisation, norm strengthening/crisis/dissolution, partner-directed enforcement

Key parameters: N=100 agents, T=3000 ticks, binary coordination game, probability matching action selection.

### 1.2 Convergence Definition (3-layer)

Convergence requires all three layers to be simultaneously met for `convergence_window` (50) consecutive ticks:
- **Behavioral**: majority fraction >= 0.95
- **Belief**: belief_error < 0.10 AND belief_variance < 0.05
- **Crystallisation**: frac_crystallised >= 0.80 AND frac_dominant_norm >= 0.90

### 1.3 Four Experimental Conditions (2x2 Factorial)

| Condition | Window | Normative | What it tests |
|-----------|--------|-----------|---------------|
| A: Baseline | Fixed (w=5) | OFF | Pure random walk (control) |
| B: Lock-in | Dynamic (2-6) | OFF | Experience memory lock-in alone |
| C: Normative | Fixed (w=5) | ON | Normative memory alone |
| D: Full | Dynamic (2-6) | ON | Both mechanisms combined |

---

## 2. Baseline Factorial Results (N=100, 50 trials each)

### 2.1 Layer Milestones

| Condition | Behavioral (mean tick) | Belief | Crystal | All met | Converged |
|-----------|----------------------|--------|---------|---------|-----------|
| A | 897 (50/50) | 815 | — | — | **0/50** |
| B | **101** (50/50) | 103 | — | — | **0/50** |
| C | 40 (50/50) | 40 | 47 | 47 | **50/50** μ=97 |
| D | **30** (50/50) | 31 | 38 | 39 | **50/50** μ=88 |

### 2.2 Key Finding: Layers are Nearly Simultaneous

- Belief tracks behavioral almost instantly (gap ≈ 0-2 ticks)
- Crystal lags behavioral by only ~7-8 ticks
- The true bottleneck is the 50-tick convergence_window waiting period
- Behavioral convergence is the rate-limiting step; once it happens, everything else follows

### 2.3 Speed Ratios

| Comparison | Ratio | Interpretation |
|------------|-------|----------------|
| B/A | **8.9x** | Lock-in mechanism effect |
| C/A | **22.5x** | Normative mechanism effect |
| D/A | **29.9x** | Combined effect |
| D/C | **1.3x** | Marginal benefit of lock-in on top of normative |

---

## 3. Tipping Point Analysis

### 3.1 Phase Transition Structure

All conditions exhibit three phases:
1. **Noise phase** — fraction_A oscillates around 0.5
2. **Takeoff** — deviation becomes sustained (tipping point)
3. **Avalanche** — positive feedback drives rapid convergence

| Condition | Takeoff tick | Mechanism |
|-----------|-------------|-----------|
| A | 2864 | Slow random walk, can reverse |
| B | 66 | Short window amplifies fluctuations |
| C | 13 | DDM crystallises early random bias |
| D | 50 | Norm churn delays initial crystallisation |

### 3.2 Condition A: Why It Fails

With fixed w=5, the system has three fatal weaknesses:
1. **No amplification**: Fixed window gives constant, tiny feedback gain
2. **No ratchet**: The system reversed from fA=0.86 (t=1000) back to fA=0.42 (t=1500)
3. **Confidence is irrelevant**: w = 5 + floor(C × 0) = 5 always; confidence computes but has zero behavioral effect

### 3.3 Condition D: Norm Churn Phenomenon

The full model shows an unexpected "norm churn" phase (ticks 30-50):
- Agents crystallise rapidly (90 agents by tick 40) but with weak norms (σ ≈ 0.48)
- Norms dissolve under anomaly pressure (n_cryst drops from 90 → 55)
- Only stabilises once behavioral majority becomes clear (~tick 60)

This makes D's takeoff *later* than C's for individual seeds, though D is faster on average.

---

## 4. Population Size (N) Sweep

N ∈ {20, 50, 100, 200, 500}, 30 trials each, all 4 conditions.

### 4.1 Behavioral Layer Speed

| N | A | B | C | D |
|---|---|---|---|---|
| 20 | 95 | 42 | 26 | 23 |
| 50 | 464 | 81 | 37 | 30 |
| 100 | 731 | 96 | 43 | 35 |
| 200 | 1569 (73%) | 116 | 49 | 37 |
| 500 | 1858 (40%) | 138 | 53 | 45 |

### 4.2 Key Findings

1. **Condition A breaks at scale**: Only 40% converge at N=500. Random fluctuations scale as ~1/√N, too weak to break symmetry.
2. **B, C, D: 100% convergence at all N**. Active mechanisms compensate for weaker fluctuations.
3. **Speed ratios grow with N**: B/A goes from 2.3x (N=20) to 13.5x (N=500); C/A from 3.7x to 35x.
4. **C and D scale weakly**: convergence time ~82→110 from N=20 to N=500. Normative mechanism is nearly population-invariant.

---

## 5. Crystallisation Threshold (θ_crystal) Sweep

θ ∈ {0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0}, conditions A/C/D, 30 trials each.

### 5.1 Overview

| θ | C behavioral | C converged | D behavioral | D converged | Norm quality |
|---|-------------|-------------|-------------|-------------|-------------|
| 0.5 | 74 | 100% μ=128 | 58 | 100% μ=111 | 1.00 |
| 1.0 | 53 | 100% μ=107 | 41 | 100% μ=94 | 1.00 |
| **2.0** | **40** | **100% μ=97** | **34** | **100% μ=93** | **1.00** |
| **3.0** | **38** | **100% μ=95** | **30** | **100% μ=88** | **0.98** |
| 5.0 | 52 | 100% μ=110 | 37 | 100% μ=98 | 0.97 |
| 8.0 | 75 | 100% μ=202 | 49 | 100% μ=329 | 0.93 |
| 12.0 | 97 | 73% μ=667 | 65 | 63% μ=1023 | 0.90 |

### 5.2 Three Regimes

**Too low (θ ≤ 1.0): Premature crystallisation**
- All agents crystallise within 1-5 ticks, before any behavioral pattern forms
- Norm direction is essentially random → ~50/50 split between A-norms and B-norms
- System must correct through slow dissolution/re-crystallisation cycle
- Convergence is *slower* despite faster crystallisation (μ=128 vs μ=95 at θ=3)
- However, norm quality is perfect (1.0) — wrong norms eventually dissolve

**Sweet spot (θ = 2-3): Calibrated crystallisation**
- Agents wait 10-20 ticks to accumulate evidence before crystallising
- Initial norm split is less severe → faster alignment
- Optimal convergence speed AND high norm quality

**Too high (θ ≥ 8): Crystallisation bottleneck**
- DDM cannot accumulate enough evidence because drift = (1-C) × f_diff
- Once behavior converges, C → 1.0, drift → 0, DDM stalls
- At θ=12, agents' mean |e| ≈ 6-9, oscillating but never reaching threshold
- 27-37% of trials fail to converge within T=3000

### 5.3 Fine-Grid Failure Boundary

| θ | Conv. rate | Crystal tick | Behavioral tick | Gap (bottleneck) |
|---|-----------|-------------|----------------|-----------------|
| 3.0 | 100% | 39 | 31 | 8 |
| 5.0 | 100% | 42 | 33 | 9 |
| **7.0** | **100%** | **139** | **42** | **97** ← crystal becomes bottleneck |
| 9.0 | 97% | 534 | 51 | 483 |
| 11.0 | 73% | 1016 | 58 | 958 |
| 15.0 | 40% | 1341 | 68 | 1273 |
| 20.0 | 0% | — | 80 | ∞ |

**Phase transition at θ ≈ 7**: below this, behavioral layer is the bottleneck; above, crystallisation layer is the bottleneck.

### 5.4 Confidence-DDM Paradox

At high θ, a fundamental contradiction emerges:
- DDM drift = `(1 - C) × f_diff`
- Once behavior converges, predictions are easy → C → 1.0 → drift → 0
- With drift ≈ 0 and noise = 0.1, DDM becomes a pure random walk
- Expected time to reach θ from 0 scales as θ²/σ² ≈ θ²/0.01
- At θ=12: expected ~14400 ticks (far exceeding T=3000)

### 5.5 θ × N Interaction

| N \ θ | 1.0 | 3.0 | 5.0 | 8.0 | 12.0 |
|-------|-----|-----|-----|-----|------|
| 50 | 100% μ=98 | 100% μ=81 | 100% μ=88 | 100% μ=332 | 70% |
| 100 | 100% μ=94 | 100% μ=93 | 100% μ=98 | 100% μ=326 | 67% |
| 200 | 100% μ=105 | 100% μ=96 | 100% μ=99 | 100% μ=315 | 53% |
| 500 | 100% μ=103 | 100% μ=105 | 100% μ=113 | 100% μ=521 | 43% |

The sweet spot (θ ≈ 2-3) does not shift with N. The failure rate at θ=12 worsens with N (70% → 43%).

---

## 6. Confidence Asymmetry (α/β) Sweep

Fixed α=0.1, β ∈ {0.15, 0.2, 0.3, 0.5, 0.7} (ratio 1.5x to 7x), all 4 conditions, 30 trials each.

### 6.1 Behavioral Layer Speed

| β (ratio) | A | B | C | D |
|-----------|---|---|---|---|
| 0.15 (1.5x) | 733 | **183** | 42 | 39 |
| 0.20 (2.0x) | 1030 | 147 | 42 | 37 |
| 0.30 (3.0x) | 699 | **94** | 37 | 34 |
| 0.50 (5.0x) | 853 | 102 | 40 | 40 |
| 0.70 (7.0x) | 942 | **71** | 40 | 34 |

### 6.2 Key Findings

1. **Only B is sensitive to β**: from 183 ticks (1.5x) to 71 ticks (7x), a 2.6x speedup. Higher β → faster confidence drop → faster window shrinkage → stronger lock-in.
2. **C and D are insensitive** (37-42 ticks across all β). The normative mechanism dominates.
3. **Convergence times for C/D are stable** (93-100 ticks). The normative layer provides robustness to confidence parameter choices.
4. **A is noisy and insensitive** — fixed window means β changes have no amplification mechanism.

---

## 7. Summary of Conclusions

### 7.1 Mechanism Hierarchy

| Mechanism | Effect on behavioral speed | Effect on convergence | Robustness |
|-----------|--------------------------|----------------------|------------|
| None (A) | Baseline (slow, unreliable) | Never converges | Breaks at large N |
| Lock-in only (B) | 9x faster | Never converges (no crystallisation) | Robust to N, sensitive to β |
| Normative only (C) | 23x faster | 100% convergence | Robust to N, β; sensitive to θ |
| Both (D) | 30x faster | 100% convergence, ~10% faster | Most robust overall |

### 7.2 Critical Parameters

- **θ_crystal (most sensitive)**: Sweet spot at 2-3. Only parameter that can break convergence. Phase transition at θ ≈ 7.
- **N (matters for A only)**: Active mechanisms are population-invariant. Without them, convergence fails at N ≥ 200.
- **β/α (matters for B only)**: Higher asymmetry strengthens lock-in. Normative layer makes this irrelevant.

### 7.3 Emergent Phenomena

1. **Norm churn** (condition D): Dynamic window + normative creates a turbulent early phase where norms form and dissolve before stabilising.
2. **Confidence-DDM paradox**: High confidence closes the DDM drift channel, preventing late crystallisation. Constrains viable θ range.
3. **Premature crystallisation penalty**: Very low θ causes 50-50 norm splits that require slow correction, making convergence *slower* despite faster crystallisation.

---

## 8. Visibility (V) × Enforcement (Φ) Sweep — PRELIMINARY

> **Caveat**: These results are for reference only. The exact mechanisms by which V and Φ interact with the dual-memory system are not yet fully theorised. This section provides empirical patterns to guide subsequent mechanism analysis.

V ∈ {0, 1, 3, 5, 10}, Φ ∈ {0.0, 0.5, 1.0, 2.0}, conditions C and D, 30 trials each.

- **V** (visibility): number of additional pair interactions an agent observes per tick (beyond their own partner). Feeds into both experience memory (FIFO) and normative DDM evidence.
- **Φ** (enforcement strength): scales the signal push in DDM (`Φ × (1-C) × γ × direction`) and gates whether enforcement is triggered at all (`Φ > 0` required).

### 8.1 Behavioral Layer Speed

**Condition C (normative only)**:

| V \ Φ | 0.0 | 0.5 | 1.0 | 2.0 |
|-------|-----|-----|-----|-----|
| 0 | 40 | 100 (16/30) | 48 (12/30) | 117 (10/30) |
| 1 | 34 | 30 | 31 | 28 |
| 3 | 38 | 32 | 32 | 31 |
| 5 | 45 | 38 | 35 | 31 |
| 10 | 60 | 50 | 43 | 42 |

**Condition D (full model)**:

| V \ Φ | 0.0 | 0.5 | 1.0 | 2.0 |
|-------|-----|-----|-----|-----|
| 0 | 33 | 67 (13/30) | 112 (21/30) | 24 (13/30) |
| 1 | 28 | 29 | 24 | 26 |
| 3 | 30 | 26 | 25 | 25 |
| 5 | 29 | 29 | 29 | 30 |
| 10 | 35 | 28 | 34 | 32 |

### 8.2 Convergence

**Condition C**:

| V \ Φ | 0.0 | 0.5 | 1.0 | 2.0 |
|-------|-----|-----|-----|-----|
| 0 | 100% μ=97 | **10%** μ=156 | **10%** μ=206 | **13%** μ=90 |
| 1 | 100% μ=84 | 100% μ=80 | 100% μ=82 | 100% μ=81 |
| 3 | 100% μ=89 | 100% μ=83 | 100% μ=82 | 100% μ=81 |
| 5 | 100% μ=96 | 100% μ=89 | 100% μ=85 | 100% μ=82 |
| 10 | 100% μ=114 | 100% μ=102 | 100% μ=94 | 100% μ=96 |

**Condition D**:

| V \ Φ | 0.0 | 0.5 | 1.0 | 2.0 |
|-------|-----|-----|-----|-----|
| 0 | 100% μ=93 | **10%** μ=220 | **17%** μ=209 | **27%** μ=182 |
| 1 | 100% μ=79 | 100% μ=79 | 100% μ=75 | 100% μ=78 |
| 3 | 100% μ=80 | 100% μ=76 | 100% μ=75 | 100% μ=75 |
| 5 | 100% μ=80 | 100% μ=78 | 100% μ=78 | 100% μ=79 |
| 10 | 100% μ=85 | 100% μ=78 | 100% μ=84 | 100% μ=82 |

### 8.3 Norm Quality (frac_dominant_norm)

**Condition C**:

| V \ Φ | 0.0 | 0.5 | 1.0 | 2.0 |
|-------|-----|-----|-----|-----|
| 0 | 0.982 | **0.842** | **0.784** | **0.765** |
| 1 | 0.999 | 1.000 | 1.000 | 0.999 |
| 3 | 0.999 | 0.999 | 1.000 | 0.999 |
| 5 | 0.998 | 0.999 | 0.999 | 0.998 |
| 10 | 0.993 | 0.996 | 0.997 | 0.996 |

### 8.4 Key Findings

#### Finding 1: V=0 + Φ>0 is catastrophic ("Blind Enforcement Trap")

The most striking result: **enabling enforcement without visibility destroys convergence**. At V=0 with any Φ>0, convergence rate collapses from 100% to 10-27%.

Hypothesised mechanism:
- With V=0, each agent sees only their partner's action (1 observation per tick)
- Enforcement signals push the DDM in the enforcer's norm direction
- But with limited observations, different agents crystallise different norms early
- Opposing enforcement signals create a tug-of-war that prevents norm consensus
- Norm quality drops to 0.76-0.84, confirming fragmented norm landscape

This is analogous to a real-world scenario where people enforce rules without understanding what others are doing — enforcement without information creates conflict rather than order.

#### Finding 2: V≥1 completely rescues the system

Even a single additional observation (V=1) restores 100% convergence at all Φ values. The information from one extra pair interaction is sufficient to disambiguate the norm direction and align enforcement.

#### Finding 3: Higher V slightly *slows* behavioral convergence (without Φ)

At Φ=0: V=0 converges at tick 40, V=10 at tick 60 (condition C). More observations create a stronger averaging effect, reducing the stochastic fluctuations that drive initial symmetry breaking. The DDM still accumulates evidence, but the experience memory becomes more conservative.

#### Finding 4: Φ compensates for the V-induced slowdown

At V=10: increasing Φ from 0.0 to 2.0 speeds convergence from μ=114 to μ=96 (condition C). Enforcement provides an additional coordination channel that offsets the averaging effect of high visibility.

#### Finding 5: Norm strength increases monotonically with V

Final mean norm strength (σ) in condition D: 0.834 (V=0) → 0.978 (V=10) at Φ=0. More observations mean more conformity signals, strengthening established norms through the α_σ mechanism.

#### Finding 6: The optimal operating point

The best convergence speed (μ≈75-80) occurs at moderate V (1-3) with any Φ>0. Beyond V=3, diminishing returns set in. The system is remarkably insensitive to Φ once V≥1.

### 8.5 Interpretation Caveats

These patterns raise several questions requiring deeper theoretical analysis:

1. **Why does V=0 + Φ>0 fail?** The enforcement signals likely create conflicting DDM pushes. A detailed per-agent trace analysis would clarify the norm fragmentation dynamics.
2. **Is the V=1 threshold robust?** This boundary should be tested with different N values and θ_crystal settings.
3. **How does V interact with the FIFO window?** V observations are added to the FIFO but not used for window size calculation. The experience memory vs. normative evidence channels may interact differently at different V.
4. **Real-world mapping**: V maps loosely to "social transparency" (how much agents know about others' behavior) and Φ to "normative pressure" (willingness/ability to sanction deviants). The finding that pressure without information backfires has clear social implications.

---

## 9. Updated Summary of Conclusions

### 9.1 Mechanism Hierarchy (revised)

| Mechanism | Effect on behavioral speed | Effect on convergence | Robustness |
|-----------|--------------------------|----------------------|------------|
| None (A) | Baseline (slow, unreliable) | Never converges | Breaks at large N |
| Lock-in only (B) | 9x faster | Never converges (no crystallisation) | Robust to N, sensitive to β |
| Normative only (C) | 23x faster | 100% convergence | Robust to N, β; sensitive to θ, **fragile at V=0+Φ>0** |
| Both (D) | 30x faster | 100% convergence, ~10% faster | Most robust overall, **same V=0+Φ>0 fragility** |

### 9.2 Critical Parameters (revised)

- **θ_crystal (most sensitive)**: Sweet spot at 2-3. Phase transition at θ ≈ 7. Only parameter that can break convergence within pure normative conditions.
- **V × Φ interaction (newly discovered)**: Enforcement without visibility is catastrophic (10% convergence). V≥1 restores full convergence regardless of Φ.
- **N (matters for A only)**: Active mechanisms are population-invariant.
- **β/α (matters for B only)**: Normative layer makes this irrelevant.

### 9.3 Emergent Phenomena (revised)

1. **Norm churn** (condition D): Dynamic window + normative creates a turbulent early phase.
2. **Confidence-DDM paradox**: High confidence closes DDM drift, preventing late crystallisation.
3. **Premature crystallisation penalty**: Very low θ causes 50-50 norm splits that slow convergence.
4. **Blind enforcement trap** (NEW): Enforcement without visibility (V=0, Φ>0) fragments norms and collapses convergence. Information must precede enforcement for norms to emerge.
