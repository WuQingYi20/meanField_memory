# DM_full Tipping Point Parameter Exploration — Analysis

## Overview

From the baseline tipping point sweep, DM_full's critical forcing fraction is ~35-40% (default: w_max=6, V=3, Phi=1.0, K=10), while DM_base tips at ~15%. This exploration varies one parameter at a time to isolate which mechanisms drive the ~20pp resilience gap.

**Setup**: 4 sweeps × 5 values × 9 fractions (0.10–0.50) × 30 trials = 5400 runs.

---

## 1. Parameter Influence Ranking

| Parameter | Critical fraction range | Total shift | Mechanism |
|-----------|------------------------|-------------|-----------|
| **K** (forcing duration) | >0.50 → 0.14 | **>36pp** | FIFO overwrite depth |
| **V** (visibility) | >0.50 → 0.21 | **>29pp** | Signal cascade amplification |
| **Phi** (enforcement) | 0.29 → 0.42 | **13pp** | Compliance push-back |
| **w_max** (FIFO size) | 0.37 → 0.38 | **~0pp** | No effect on tipping point |

K and V are the dominant drivers. Phi provides moderate protection. w_max is irrelevant to whether a flip occurs.

---

## 2. Per-Parameter Analysis

### 2.1 K (Forcing Duration) — Strongest Driver

| K | Critical fraction | S-curve width (10%→90%) |
|---|-------------------|------------------------|
| 3 | >0.50 (unflippable) | N/A |
| 5 | >0.50 (unflippable) | N/A |
| 10 | 0.382 | 0.10 |
| 20 | 0.214 | 0.05 |
| 30 | 0.139 | 0.05 |

**Key findings**:

- **K ≤ 5 makes DM_full completely unflippable** — even 50% forcing for 5 ticks cannot flip the norm. This is because 5 ticks of forcing cannot overwrite the FIFO (w_max=6), leaving the majority of experiential memory intact.
- **K=30 drops the critical fraction to 0.14, matching DM_base** — 30 ticks of forcing completely erases the experiential memory advantage, rendering the normative layer irrelevant.
- **The S-curve sharpens with larger K** (width shrinks from 0.10 to 0.05), indicating a sharper phase transition when forcing is prolonged.

**Mechanistic interpretation**: DM_full's resilience is fundamentally about **incomplete FIFO overwrite**. The normative layer does not actively resist; rather, the experiential memory retains pre-forcing information that guides recovery. Once K exceeds w_max, this protection vanishes.

### 2.2 V (Visibility / Signal Range) — Destabilizing, Not Stabilizing

| V | Critical fraction | Convergence rate | S-curve width |
|---|-------------------|-----------------|---------------|
| 0 | >0.50 | 17–27% (poor) | N/A |
| 1 | >0.50 (unflippable) | 100% | N/A |
| 3 | 0.376 (default) | 100% | 0.10 |
| 5 | 0.266 | 100% | 0.05 |
| 8 | 0.213 | 100% | 0.05 |

**Key findings**:

- **V=0 barely converges** (only 5–8/30 trials reach steady state in 600 ticks) because agents lack social signals and rely solely on direct experience.
- **V=1 is a "resilience sweet spot"** — sufficient visibility for norm emergence, but signal range too small for the forced minority's behavior to cascade to non-forced agents. Result: unflippable at all forcing fractions.
- **V ≥ 3 is progressively destabilizing**. V=8 tips at just ~21%, nearly as fragile as DM_base.
- **S-curves sharpen dramatically**: V=8 transitions from 0% to 97% flip rate between frac=0.15 and frac=0.25.

**Mechanistic interpretation**: V amplifies signal propagation symmetrically — it strengthens both the incumbent norm and any perturbation. During steady state, high V reinforces the majority. But during forcing, high V also broadcasts the forced minority's behavior to non-forced agents, creating a cascade nucleation effect. The **net effect is destabilizing** because forcing creates a concentrated behavioral cluster whose influence radiates outward proportionally to V.

This mirrors real-world observations that increased information connectivity accelerates social change in both directions.

### 2.3 Phi (Enforcement Strength) — Moderate, Diminishing Returns

| Phi | Critical fraction | Shift from Phi=0 |
|-----|-------------------|------------------|
| 0.0 | 0.288 | baseline |
| 0.5 | 0.372 | +8.4pp |
| 1.0 | 0.377 | +8.9pp |
| 2.0 | 0.392 | +10.4pp |
| 3.0 | 0.422 | +13.4pp |

**Key findings**:

- **Phi=0 ≠ DM_base**: even without enforcement, the critical fraction is 0.288, not ~0.15 as in DM_base. The 14pp gap is attributable to **crystallisation alone** — crystallised agents have lower sigma (noise), making their behavior more stable under perturbation.
- **The largest marginal gain is Phi 0→0.5 (+8pp)**; subsequent increases from 0.5→3.0 add only 5pp total. Enforcement's primary value is binary (present/absent), not proportional to strength.
- Perturbation depth at frac=0.3 is nearly identical across all Phi values (~0.53–0.55 for non-flip trials). Phi does not prevent behavioral disruption — it only affects whether recovery succeeds.

**Mechanistic interpretation**: Enforcement acts as a compliance pressure that nudges agents back toward the established norm during recovery. But it cannot prevent the initial behavioral shock. The crystallisation mechanism itself is the larger contributor: by reducing sigma, crystallised agents are less responsive to noisy signals, providing passive resistance.

### 2.4 w_max (FIFO Memory Size) — No Effect on Tipping Point

| w_max | Critical fraction | Recovery at frac=0.2 | Recovery at frac=0.3 |
|-------|-------------------|---------------------|---------------------|
| 4 | 0.375 | 2 ticks | 16 ticks |
| 6 | 0.381 | 2 ticks | 19 ticks |
| 10 | 0.372 | 5 ticks | 28 ticks |
| 15 | 0.381 | 16 ticks | 29 ticks |
| 20 | 0.379 | 38 ticks | 30 ticks |

**Key findings**:

- **Critical fraction is flat at ~0.38 regardless of w_max** (range: 0.372–0.381). The initial hypothesis — that larger FIFO retains more pre-forcing memory — was wrong.
- **Why**: with K=10, the forcing duration already exceeds w_max=6 (default) and is comparable to w_max=10. For w_max=4, the FIFO is completely overwritten. For w_max=20, 10 minority memories in a 20-slot FIFO still constitute 50% of recent experience, which is sufficient to destabilize beliefs.
- **w_max does affect recovery speed**: at frac=0.2, w_max=4 recovers in 2 ticks while w_max=20 takes 38 ticks. Larger FIFOs dilute the forcing memories more slowly because each new experience only displaces 1/w_max of the buffer.

**Mechanistic interpretation**: w_max does not create a "deeper memory bank" that protects against forcing. Instead, it creates a **longer memory tail** that slows both disruption and recovery. The tipping point is determined by whether forcing creates a self-sustaining behavioral cascade (driven by K and V), not by how much pre-forcing memory remains.

---

## 3. Cross-Cutting Findings

### 3.1 Norm Survival Is Always ~1.0

Across all 5400 trials, norm_survival (fraction of crystallised agents retaining their norm) never drops below 0.9. **Norms do not dissolve under any forcing condition tested**. This confirms that the normative layer operates purely as an **inertial anchor** — it maintains crystallisation state throughout the perturbation, but does not actively counteract behavioral change.

The implication is stark: norms persist even when behavior flips. Agents can be behaviorally forced into a new strategy while still holding their old crystallised norm. The flip succeeds not because norms dissolve, but because the experiential layer overwhelms the normative signal.

### 3.2 S-Curves Indicate True Phase Transitions

Most parameter configurations show a transition from 10% to 90% flip rate within 5% forcing fraction (one step in our grid). This sharp threshold behavior is characteristic of a phase transition, not gradual degradation. The system is either resilient (recovers fully) or collapses (flips completely), with very little middle ground.

### 3.3 Predicted Interaction: K × V

K determines "memory overwrite depth" and V determines "signal cascade range." Their combination (high K + high V) likely produces super-additive destabilization. A 2D sweep of K × V would map the joint fragility surface and is a natural next step.

---

## 4. Implications for the Paper Narrative

1. **Core thesis**: DM_full's resilience comes from **incomplete experiential memory overwrite** (K < w_max protection), not from normative layer resistance. The normative layer provides "behavioral inertia" through crystallisation (reduced sigma), not active defense.

2. **Crystallisation alone adds ~14pp** of resilience over DM_base (Phi=0 critical fraction at 0.288 vs DM_base at ~0.15). This can be framed as the contribution of the "passive normative mechanism" — the act of crystallising reduces noise and stabilizes behavior, independent of enforcement.

3. **V's destabilizing effect** is the most counter-intuitive finding: greater social visibility makes norms *easier* to overturn. This parallels real-world dynamics where information connectivity accelerates both norm maintenance and norm change, with the net effect favoring disruption when a coordinated perturbation is present.

4. **V=1 as the resilience optimum** suggests a "small-world" insight: minimal but nonzero social connectivity maximizes norm stability. This is testable and could be a standalone discussion point.

5. **K's dominance** provides a clear policy-relevant interpretation: the duration of a behavioral disruption matters far more than its intensity. A brief, intense shock (K=3, 50% forcing) has zero effect, while a prolonged, moderate shock (K=30, 15% forcing) is devastating.
