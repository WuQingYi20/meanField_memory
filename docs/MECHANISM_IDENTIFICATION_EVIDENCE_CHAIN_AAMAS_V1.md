# Mechanism Identification Evidence Chain (AAMAS V1)

## Scope
- Goal: establish that V5 effects are caused by the intended dual-memory mechanism, not by incidental implementation choices.
- Target mechanism: `predictive confidence -> adaptive experience window + normative uptake gate -> norm crystallisation + resilience`.
- Unit: individual-level cognition and population-level emergence in repeated coordination games.

## Assumptions
- A1: Current codebase and docs reflect V5 semantics (`SimulationEnvironment` tick pipeline, normative memory on/off switches, reproducible seeds).
- A2: Existing evidence files are accepted as mechanism-direction support, but not as full causal identification.
- A3: AAMAS-level claim needs explicit falsification tests and competing hypotheses.

## Evidence Chain Summary

| Chain ID | Claim | Evidence | Mechanism Hypothesis | Model Mapping | Confidence | Gap |
|---|---|---|---|---|---|---|
| M1 | Social information can be represented as evidence accumulation before norm crystallisation | Germar et al. 2014/2019, Duderstadt et al. 2022; see `/Users/yifan/Documents/New project/meanField_memory/docs/INDIVIDUAL_NORM_EVIDENCE_CHAIN_V5.md` | Agents integrate social consistency as noisy drift until threshold crossing | `normative_ddm_or_anomaly`, `crystal_threshold`, `ddm_noise` | Medium-High | Functional form not uniquely identified |
| M2 | Lower predictive confidence increases norm uptake susceptibility | Rendell 2010, Wood 2016, Behrens 2007; see `/Users/yifan/Documents/New project/meanField_memory/docs/INDIVIDUAL_NORM_EVIDENCE_CHAIN_V5.md` | Uncertain agents overweight external regularities and internalise faster | Drift gate term using confidence (currently linear in V5 text) | Medium | Gate shape not empirically pinned |
| M3 | Fixed staged update is fair for mechanism estimation | `/Users/yifan/Documents/New project/meanField_memory/docs/TICK_PIPELINE_FAIRNESS_EVIDENCE_V5.md`; reproducibility tests in `/Users/yifan/Documents/New project/meanField_memory/tests/test_reproducible_tick_pipeline.py` | Stage boundaries reduce order artifacts and keep causal layer separation | `get_tick_update_order`, synchronized passes | Medium-High | Async external validity still untested |
| M4 | Normative memory adds explanatory power beyond lock-in-only | Matched ON/OFF analysis scaffold exists in `/Users/yifan/Documents/New project/meanField_memory/experiments/run_normative_delta_comparison.py` | Distinct memory channel contributes incremental effect on convergence and adoption | `enable_normative=True/False` matched seeds | Medium | Needs pre-registered primary estimands and robustness |
| M5 | Core outputs are reproducible under fixed seeds | Reproducible pipeline script exists: `/Users/yifan/Documents/New project/meanField_memory/experiments/run_v5_core_metrics_reproducible.py` | Deterministic seed policy allows stable estimation and comparison | `master_seed`, per-trial seeds, dataframe fingerprint | High | Reproducibility alone is not identification |

## Chain Records (Model-Ready)

### Chain M1
- Chain ID: M1
- Norm topic: individual norm crystallisation
- Claim: pre-crystallisation dynamics are evidence-accumulation based.
- Evidence source(s): Germar et al. (2014, 2019), Duderstadt et al. (2022), plus local synthesis in `/Users/yifan/Documents/New project/meanField_memory/docs/INDIVIDUAL_NORM_EVIDENCE_CHAIN_V5.md`.
- Evidence type: experimental + theory-linked modeling.
- Population/context fit: lab perceptual/social influence tasks; partial transfer to ABM norm setting.
- Mechanism hypothesis: repeated consistent observations shift internal evidence toward a norm hypothesis until threshold crossing.
- Model mapping:
  - Agent attribute: `evidence`, `norm`, `strength`
  - Decision rule: crystallise when `evidence >= crystal_threshold`
  - Parameter(s): `ddm_noise`, `crystal_threshold`
  - Network/institution element: optional observation channel `observation_k`
- Expected directional effect: consistency increase -> earlier crystallisation.
- Confidence: medium-high.
- Gap or uncertainty: drift function and noise form are underspecified empirically.
- Planned test: compare `linear`, `logistic`, `threshold` drift gates with equal parameter budget; rank via out-of-sample predictive fit and effect stability.

### Chain M2
- Chain ID: M2
- Norm topic: uncertainty-conditioned social uptake
- Claim: lower confidence should increase internalisation speed.
- Evidence source(s): Rendell et al. (2010), Wood et al. (2016), Behrens et al. (2007), local synthesis file above.
- Evidence type: empirical/experimental.
- Population/context fit: high on direction, moderate on exact coefficient form.
- Mechanism hypothesis: confidence acts as a gating variable controlling social-evidence impact.
- Model mapping:
  - Agent attribute: `predictive_confidence`
  - Decision rule: drift multiplier decreases with confidence
  - Parameter(s): gate shape parameters
  - Network/institution element: exposure frequency via observation and pairing
- Expected directional effect: confidence down -> hazard of crystallisation up.
- Confidence: medium.
- Gap or uncertainty: linear gate `(1-C)` is a convenience assumption.
- Planned test: survival/hazard analysis of first crystallisation tick, controlling for exposure and seed.

### Chain M3
- Chain ID: M3
- Norm topic: fairness of update timing
- Claim: synchronous staged order is justified for mechanism identification.
- Evidence source(s): `/Users/yifan/Documents/New project/meanField_memory/docs/TICK_PIPELINE_FAIRNESS_EVIDENCE_V5.md`; `/Users/yifan/Documents/New project/meanField_memory/tests/test_reproducible_tick_pipeline.py`.
- Evidence type: methodological rationale + reproducibility tests.
- Population/context fit: high for internal validity, unknown for micro-timing realism.
- Mechanism hypothesis: separating action, memory, confidence, norm update, enforcement, metrics reduces artificial within-tick causality leakage.
- Model mapping:
  - Agent attribute: all state updates happen in staged passes
  - Decision rule: no same-pass feedback from later phases
  - Parameter(s): none (schedule-level mechanism)
  - Network/institution element: global stage barrier per tick
- Expected directional effect: permutation of pair processing should not change aggregate outputs.
- Confidence: medium-high.
- Gap or uncertainty: async variant may produce different cascade speeds.
- Planned test: required triple-check set:
  1. pair-order permutation invariance,
  2. sync vs async direction-consistency,
  3. enforcement same-tick vs next-tick sensitivity.

### Chain M4
- Chain ID: M4
- Norm topic: incremental value of normative memory
- Claim: normative memory contributes beyond dynamic experience memory.
- Evidence source(s): existing matched-grid analysis script `/Users/yifan/Documents/New project/meanField_memory/experiments/run_normative_delta_comparison.py`.
- Evidence type: simulation-comparative.
- Population/context fit: direct for current ABM.
- Mechanism hypothesis: explicit rule-state (`norm`, `strength`, anomalies) improves stability and post-shock recovery relative to lock-in-only.
- Model mapping:
  - Agent attribute: normative state on/off
  - Decision rule: compliance blending and enforcement broadcast
  - Parameter(s): `enable_normative`, `compliance_exponent`, `enforce_threshold`
  - Network/institution element: signal broadcast phase
- Expected directional effect: ON condition yields higher adoption and stronger persistence.
- Confidence: medium.
- Gap or uncertainty: potential confounding from schedule and parameter interactions.
- Planned test: pre-registered 2x2 factorial + matched seeds + interaction effect estimates.

### Chain M5
- Chain ID: M5
- Norm topic: reproducibility foundation
- Claim: core trajectories are reproducible under fixed seed policy.
- Evidence source(s): `/Users/yifan/Documents/New project/meanField_memory/experiments/run_v5_core_metrics_reproducible.py`.
- Evidence type: computational reproducibility.
- Population/context fit: direct.
- Mechanism hypothesis: deterministic seed and pipeline constraints eliminate accidental randomness between repeated runs.
- Model mapping:
  - Agent attribute: deterministic RNG path per run
  - Decision rule: same seeds -> same trajectories
  - Parameter(s): `master_seed`, trial seeds
  - Network/institution element: none
- Expected directional effect: identical fingerprints across reruns.
- Confidence: high.
- Gap or uncertainty: reproducible wrong model is still possible.
- Planned test: continuous integration check on hash fingerprints and summary stats.

## Falsification-Oriented Experiment Set (Most Critical for AAMAS)

### E1. Mechanism Isolation (required)
- Design: 2x2 factorial
  - Factor A: adaptive window OFF/ON
  - Factor B: normative memory OFF/ON
- Estimands:
  - main effect A on convergence and belief stability
  - main effect B on norm adoption and resilience
  - interaction `A x B` on institutional-level persistence
- Reject claim if:
  - B has near-zero effect across metrics, or
  - observed gains are fully explained by A alone.

### E2. Gate-Shape Competition (required)
- Design: replace confidence gate with three alternatives under matched complexity:
  - linear, logistic, threshold.
- Estimands:
  - first-crystallisation hazard fit,
  - final adoption calibration error,
  - shock-recovery error.
- Reject current formulation if:
  - linear gate is dominated on most metrics or unstable across seeds/topologies.

### E3. Update-Schedule External Validity (required)
- Design:
  - synchronous staged (current) vs asynchronous event-driven variant.
  - enforcement timing ablation (same tick vs next tick).
- Estimands:
  - sign consistency of key effects,
  - effect-size drift bounds.
- Reject strong fairness claim if:
  - effect direction flips or large uncontrolled drift appears.

## Minimal Statistical Protocol
- Primary metrics (pre-registered):
  - `time_to_convergence`,
  - `norm_adoption_rate@t`,
  - `collapse_probability`,
  - `recovery_time`.
- Estimation:
  - bootstrap CIs for means/medians,
  - nonparametric paired tests for matched-seed contrasts,
  - report effect sizes, not only p-values.
- Seed policy:
  - fixed master seed and published derived seed list.
- Reproducibility artifact:
  - scripts, configs, raw CSV, and figure-generation commands.

## Decision Use
- If E1-E3 all pass: claim can be upgraded from "structurally plausible" to "mechanistically identified (within model class)".
- If any fail: downgrade corresponding claim and keep as alternative hypothesis in appendix/main text sensitivity section.

## Why This Is the Key Chain
- It directly links your strongest contribution (dual-memory mechanism) to falsifiable causal evidence.
- It addresses the most likely AAMAS reviewer concern: "interesting dynamics, but is the mechanism truly identified?"
- It converts existing assets in this repo into a publication-grade evidence package with minimal extra assumptions.
