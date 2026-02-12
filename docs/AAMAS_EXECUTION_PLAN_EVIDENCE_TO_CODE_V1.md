# AAMAS Execution Plan: Evidence -> Conceptual Model -> Code (V1)

## 1. Objective
- Primary objective: upgrade current work from "mechanism-consistent" to "mechanism-identified (within model class)" with reproducible evidence.
- Submission target: AAMAS main track standards on soundness and reproducibility.
- Scope: documentation claims, falsification-oriented experiments (`E1/E2/E3`), and code pipeline updates.

## 2. Why This Order
- Reason 1: avoid claim-overreach first. We must align conceptual claims to existing evidence before adding new results.
- Reason 2: maximize reviewer impact per unit effort. `E1` (2x2 isolation) gives the fastest causal signal about core contribution.
- Reason 3: prevent rework. `E2/E3` depend on stable estimands and reporting protocol defined up front.
- Reason 4: keep artifacts submission-ready. Each phase outputs rerunnable scripts + tables + figures, not only narrative text.

## 3. Current Gaps (Blocking AAMAS-Level Claim)
- G1: no completed `E1` main-effect/interaction table for dual-memory mechanism isolation.
- G2: no completed `E2` function-form competition for confidence gate/compliance mapping.
- G3: no completed `E3` sync-vs-async and enforcement timing robustness.
- G4: no single pre-registered analysis protocol enforced by scripts.
- G5: conceptual text still contains places where readers may interpret candidate equations as fixed laws.

## 4. How To Modify Conceptual Model Documents

### 4.1 Core principle
- Keep mechanism-direction claims strong.
- Downgrade exact functional forms to "candidate implementations" unless validated by `E2`.
- Attach each key equation to explicit falsifier language.

### 4.2 Required edits in `/Users/yifan/Documents/New project/meanField_memory/docs/conceptual_model_v5.tex`
- Edit A (Abstract and Intro):
  - Current risk area: strong wording around unique mechanism certainty.
  - Change to: "we model as" / "candidate implementation" where equation-level claims appear.
- Edit B (DDM section around `eq:drift`, line markers near 214-234 by current grep):
  - Keep evidence-grade paragraph.
  - Add one explicit sentence: "Linear gate is a baseline hypothesis to be compared against logistic/threshold alternatives."
- Edit C (Compliance section around `sigma^k`, markers near 283-305):
  - Keep monotonicity claim.
  - Add explicit alternative-family statement and reference to `E2`.
- Edit D (Crisis operator section around `eq:crisis`, markers near 257-269):
  - Keep nonlinear-collapse rationale.
  - Mark operator as exploratory until stress-tested.
- Edit E (Enforcement threshold section around `theta_enforce`, markers near 315-340):
  - Keep ambiguity-sensitive boundary support.
  - Clarify threshold value is calibrated, not literature-fixed.
- Edit F (Prediction section around falsifiable predictions, markers near 512-533):
  - Convert into direct experiment references: `E1/E2/E3` and rejection criteria.

### 4.3 Supporting document sync
- Update `/Users/yifan/Documents/New project/meanField_memory/docs/MECHANISM_IDENTIFICATION_ONEPAGER_AAMAS.md`:
  - add final links to generated result tables/figures after each experiment phase.
- Keep `/Users/yifan/Documents/New project/meanField_memory/docs/MECHANISM_IDENTIFICATION_EVIDENCE_CHAIN_AAMAS_V2_WEBVERIFIED.md` as source-of-truth for claim tiers.

## 5. How To Use Existing Evidence Chains

## 5.1 Evidence role split
- Directional mechanism support:
  - from web-verified chain (`V2`), supports whether mechanism direction is plausible.
- Functional-form support:
  - not pre-granted by literature; must be won by model competition (`E2`).
- Implementation-fairness support:
  - local fairness note + reproducibility tests, then strengthened by `E3`.

## 5.2 Claim governance rule (for paper writing)
- Allowed now:
  - "evidence-grounded mechanism direction"
  - "candidate equation family"
- Allowed after `E1`:
  - "incremental contribution of normative memory observed"
- Allowed after `E1+E2+E3`:
  - "mechanism identified within tested model class and update schedules"

## 6. Code Modification Plan

### Phase P0: Analysis Protocol Lock (must do first)
- New file: `/Users/yifan/Documents/New project/meanField_memory/experiments/protocols/aamas_mechanism_protocol_v1.json`
  - fields: primary metrics, estimands, CI method, seeds, run counts, exclusion rules.
- New helper: `/Users/yifan/Documents/New project/meanField_memory/experiments/utils/analysis_protocol.py`
  - validates protocol and injects defaults into experiment runners.

### Phase P1: E1 Mechanism Isolation (2x2)
- New runner: `/Users/yifan/Documents/New project/meanField_memory/experiments/run_e1_mechanism_isolation.py`
  - factors:
    - `adaptive_window`: off/on
    - `normative_memory`: off/on
  - matched seed grid.
- New analysis: `/Users/yifan/Documents/New project/meanField_memory/experiments/analyze_e1_mechanism_isolation.py`
  - outputs:
    - `data/experiments/e1_effect_table.csv`
    - `data/experiments/e1_interaction_plot.png`
    - `data/experiments/e1_summary.md`
- Minimal tests:
  - `/Users/yifan/Documents/New project/meanField_memory/tests/test_e1_pipeline.py`
  - seed determinism and required output columns.

### Phase P2: E2 Functional-Form Competition
- Code extension in `/Users/yifan/Documents/New project/meanField_memory/src/memory/normative.py`
  - add configurable gate families: `linear`, `logistic`, `threshold`.
- Config extension in `/Users/yifan/Documents/New project/meanField_memory/config.py`
  - `confidence_gate_family`, gate parameters.
- Runner: `/Users/yifan/Documents/New project/meanField_memory/experiments/run_e2_gate_competition.py`
- Analysis:
  - rank by calibration error + stability across seeds.
  - output comparison table and dominance summary.

### Phase P3: E3 Update-Schedule Robustness
- Add async variant implementation:
  - new file `/Users/yifan/Documents/New project/meanField_memory/src/environment_async.py`
  - same interface as `SimulationEnvironment`.
- Timing ablation in `/Users/yifan/Documents/New project/meanField_memory/src/environment.py`
  - switch `enforcement_timing = same_tick | next_tick`.
- Runner: `/Users/yifan/Documents/New project/meanField_memory/experiments/run_e3_schedule_robustness.py`
- Analysis output:
  - sign-consistency table,
  - effect-drift bounds table,
  - risk note for external validity.

### Phase P4: Artifact Packaging
- New script: `/Users/yifan/Documents/New project/meanField_memory/experiments/package_aamas_artifacts.py`
  - copies protocol, configs, raw CSV, figure scripts, and checksums into one bundle directory.
- New manifest: `/Users/yifan/Documents/New project/meanField_memory/data/experiments/AAMAS_ARTIFACT_MANIFEST.md`

## 7. Concrete Writing Updates After Each Phase
- After P1:
  - update claim C3 status in one-pager from "pending" to "supported/not supported".
- After P2:
  - update sections around `eq:drift` and `eq:compliance` with winning/non-dominated function family.
- After P3:
  - update tick-pipeline fairness statement with measured sync-vs-async drift bounds.
- After P4:
  - add reproducibility appendix paragraph linking artifact manifest.

## 8. Definition of Done
- DoD-1: `E1/E2/E3` each has script, raw outputs, summary markdown, and at least one figure.
- DoD-2: conceptual model text uses calibrated claim strength; no equation is presented as uniquely identified without test evidence.
- DoD-3: one command sequence can rerun all core analyses from protocol file.
- DoD-4: evidence chain docs and conceptual model citations are cross-linked and internally consistent.

## 9. Immediate Next Step
- Execute P0 + P1 first.
- Deliverable for next checkpoint:
  - `e1_effect_table.csv`
  - `e1_interaction_plot.png`
  - short interpretation note in `e1_summary.md`.
