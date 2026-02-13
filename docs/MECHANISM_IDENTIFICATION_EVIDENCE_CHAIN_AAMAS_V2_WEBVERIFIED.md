# Mechanism Identification Evidence Chain (AAMAS V2, Web-Verified)

## Verification Scope
- Verification date: 2026-02-12
- Goal: upgrade V1 chain with externally verified primary sources and explicit support tiers.
- Previous draft: `/Users/yifan/Documents/New project/meanField_memory/docs/MECHANISM_IDENTIFICATION_EVIDENCE_CHAIN_AAMAS_V1.md`

## Web-Verified Source Register (Primary / Official)

| ID | Source | Verified link | What it supports | Support tier |
|---|---|---|---|---|
| S1 | Germar et al., 2014, *PSPB*, DOI 10.1177/0146167213508985 | https://pubmed.ncbi.nlm.nih.gov/24154917/ | Social influence represented with diffusion-model decomposition | Mechanism-level |
| S2 | Germar & Mojzisch, 2019, *JESP*, DOI 10.1016/j.jesp.2019.03.012 | https://www.sciencedirect.com/science/article/abs/pii/S0022103118304347 | Persistent perceptual bias after social norm learning | Mechanism-level |
| S3 | Hohmann et al., 2023, *Cognition*, DOI 10.1016/j.cognition.2023.105611 | https://www.sciencedirect.com/science/article/abs/pii/S0010027723002457 | Replication/extension: norm learning can alter visual appearance and decision dynamics | Mechanism-level (reinforcement) |
| S4 | Rendell et al., 2010, *Science*, DOI 10.1126/science.1184719 | https://pubmed.ncbi.nlm.nih.gov/20378813/ | Why and when social learning is adaptive in uncertain environments | Direction-level gate support |
| S5 | Wood et al., 2016, *JECP*, DOI 10.1016/j.jecp.2016.06.005 | https://pubmed.ncbi.nlm.nih.gov/27371768/ | Copy-when-uncertain direction in social learning behavior | Direction-level gate support |
| S6 | Behrens et al., 2007, *Nat Neurosci*, DOI 10.1038/nn1954 | https://www.nature.com/articles/nn1954 | Volatility-sensitive learning-rate adaptation under uncertainty | Direction-level gate support |
| S7 | Andrighetto et al., 2015, *Frontiers*, DOI 10.3389/fpsyg.2015.01413 | https://pubmed.ncbi.nlm.nih.gov/26500568/ | Norm compliance can persist from perceived legitimacy even without monitoring | Mechanism-level compliance rationale |
| S8 | Toribio-Florez et al., 2023, *PSPB*, DOI 10.1177/01461672211067675 | https://pubmed.ncbi.nlm.nih.gov/35100898/ | Third-party punishment decreases under ambiguity | Boundary-condition support |
| S9 | AAMAS 2026 main-track submission instructions (official) | https://cyprusconferences.org/aamas2026/submission-instructions/ | Review criteria explicitly include soundness and reproducibility | Venue requirement |

## Evidence Chain (Updated with Support Tier)

### C1. DDM-style pre-crystallisation channel
- Claim: representing pre-norm social influence as evidence accumulation is justified.
- Evidence: S1, S2, S3.
- Model mapping: `evidence`, `ddm_noise`, `crystal_threshold`.
- Support tier:
  - Strong: existence/direction of accumulation-style mechanism.
  - Weak: exact drift functional form in your equation.
- Required falsifier: if alternative update families outperform DDM-style gate across all core outputs, downgrade C1.

### C2. Confidence/uncertainty gate for norm uptake
- Claim: lower predictive confidence should increase social uptake/internalisation speed.
- Evidence: S4, S5, S6.
- Model mapping: confidence gate term over drift.
- Support tier:
  - Strong: directional relation (uncertainty -> more social reliance).
  - Weak: exact linear gate `(1-C)` and coefficient scale.
- Required falsifier: if hazard of first crystallisation is not monotone in confidence after exposure controls, reject gate claim.

### C3. Internal normative state beyond pure payoff adaptation
- Claim: persistent internal rule state is needed, not only short-term conformity.
- Evidence: S2, S7.
- Model mapping: `norm`, `strength`, anomaly handling.
- Support tier:
  - Moderate-Strong: persistence rationale for internal state.
  - Weak: specific compliance curve (e.g., `sigma^k` form).
- Required falsifier: if persistence and post-shock recovery are fully matched by lock-in-only model, reject incremental claim.

### C4. Enforcement must be ambiguity-sensitive
- Claim: ambiguity should reduce enforcement intensity.
- Evidence: S8.
- Model mapping: enforcement gate and confidence in violation detection.
- Support tier:
  - Strong: boundary direction.
  - Weak: any fixed numerical threshold.
- Required falsifier: if ambiguity manipulations do not reduce enforcement in model/data, revise enforcement gate.

### C5. Update schedule fairness and reproducibility standard
- Claim: staged synchronous pipeline is acceptable for identification, but must be stress-tested.
- Evidence:
  - local methodological note: `/Users/yifan/Documents/New project/meanField_memory/docs/TICK_PIPELINE_FAIRNESS_EVIDENCE_V5.md`
  - local reproducibility tests: `/Users/yifan/Documents/New project/meanField_memory/tests/test_reproducible_tick_pipeline.py`
  - venue criteria requiring reproducibility: S9.
- Support tier:
  - Moderate for internal validity.
  - Pending for external validity.
- Required falsifier: async comparison flips key effect signs or causes large uncontrolled effect drift.

## Most Critical AAMAS-Ready Test Bundle

1. Mechanism isolation (2x2 factorial, required)
- `adaptive_window`: off/on
- `normative_memory`: off/on
- Primary estimands:
  - main(A): convergence and belief stability,
  - main(B): adoption and recovery,
  - interaction(AxB): institutional persistence.

2. Functional-form competition for gate/compliance (required)
- Compare `linear`, `logistic`, `threshold` gate families under matched parameter budgets.
- Report:
  - out-of-sample fit on crystallisation timing,
  - effect stability across seeds/topology variants,
  - calibration error for adoption/recovery.

3. Schedule robustness (required)
- Sync staged vs async event-driven.
- Same-tick vs next-tick enforcement timing.
- Report sign consistency and bounded effect-size drift.

## Minimal Reporting Standard (Aligned with AAMAS Criteria)
- As per official AAMAS 2026 instructions (S9), review includes originality, significance, soundness, reproducibility, clarity.
- Therefore, report at minimum:
  - fixed seed policy + released seed list,
  - pre-declared primary metrics,
  - effect sizes with confidence intervals,
  - full scripts/configs for rerun.

## Bottom Line
- After web verification, your strongest defensible statement is:
  - "Mechanism direction is evidence-grounded; exact functional forms remain hypothesis-level and must be selected by falsification-oriented model comparison."
- This framing is safer and more AAMAS-consistent than claiming direct empirical calibration of current equations.
