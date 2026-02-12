# Tick Pipeline Fairness Evidence (V5)

## Purpose
This note evaluates the V5 tick pipeline from a neutral perspective.  
Goal: determine whether the chosen execution order is mechanism-faithful, not merely rhetorically convenient.

## Question
Is the synchronous staged pipeline (`pair/action -> M^E update -> C update -> normative update -> enforcement broadcast -> metrics`) a fair modeling choice?

## Neutral Assessment
### What the synchronous staged pipeline improves
1. Reduces implementation-order artifacts.
If pair processing order changes, aggregate outcomes should not drift purely because of loop order.

2. Preserves causal layering.
Behavior generates evidence first; epistemic confidence updates second; normative inference/enforcement follows.

3. Improves reproducibility and cross-implementation comparability.
A fixed stage boundary clarifies what information is available at each update.

### What it may lose
1. Less micro-temporal realism.
Real social interaction can be asynchronous and event-driven.

2. Potential underestimation of fast cascades.
Immediate same-tick reactions are damped by stage boundaries.

3. Adds a discrete-clock assumption.
This is a modeling choice, not a universal truth.

## Why this is still justified for the current paper
The current research target is mechanism identification and reproducible effect estimation (norm coverage, coordination, crystallisation timing, shock robustness), not high-frequency event timing realism.

Given that target, minimizing artificial order bias is more important than simulating fully asynchronous micro-timing.

## Falsifiable Checks (Not Just Narrative)
1. Order invariance test (required):
Randomly permute pair processing order; summary distributions should remain statistically stable.

2. Sync-vs-async robustness test (recommended):
Run an asynchronous variant and compare effect direction/sign for the four core outputs.

3. Signal timing sensitivity (required):
Compare enforcement effect when applied next tick vs same tick; report differences explicitly.

If these checks fail materially, the pipeline claim must be revised.

## Practical Position
For V5 main results, synchronous staged update is a fair and defensible default.
For external validity, asynchronous robustness should be reported as a sensitivity appendix.

