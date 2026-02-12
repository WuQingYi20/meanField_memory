# Mechanism Identification One-Pager (AAMAS Draft)

## Purpose
Provide a compact, publication-ready evidence map for the V5 dual-memory mechanism with explicit falsifiers.

## Claim-Evidence-Gap-Falsifier

| Claim | Evidence | Gap | Falsifier |
|---|---|---|---|
| C1. Pre-crystallisation social influence is well modeled as evidence accumulation. | Germar et al. 2014 (PSPB), Germar & Mojzisch 2019 (JESP), Hohmann et al. 2023 (Cognition). | Exact drift equation is not directly estimated from these studies. | If non-accumulation alternatives consistently outperform DDM-family models on crystallisation timing and adoption trajectories, reject C1 implementation form. |
| C2. Lower predictive confidence should increase norm uptake speed (directional claim). | Rendell et al. 2010 (Science), Wood et al. 2016 (JECP), Behrens et al. 2007 (Nat Neurosci). | Direction is supported; linear gate `(1-C)` is a modeling choice. | If first-crystallisation hazard is not monotone with lower confidence after controlling for exposure, reject gate claim. |
| C3. Internal normative state adds explanatory power beyond lock-in-only learning. | Persistent norm-learning effects (Germar & Mojzisch 2019), legitimacy-driven compliance rationale (Andrighetto et al. 2015), local ON/OFF analysis scaffold. | Incremental contribution not yet identified with pre-registered estimands. | If 2x2 isolation shows near-zero main/interaction effects for normative memory on adoption/persistence/recovery, reject incremental claim. |
| C4. Enforcement should be ambiguity-sensitive, not fixed-intensity. | Toribio-Florez et al. 2023 (PSPB): punishment decreases under ambiguity. | No empirical basis for a universal numeric threshold. | If ambiguity manipulations do not reduce enforcement in model outputs, reject enforcement gate design. |
| C5. Synchronous staged pipeline is acceptable for internal identification. | Local fairness note + deterministic reproducibility tests; AAMAS requires soundness/reproducibility. | External validity to async micro-timing remains open. | If sync-vs-async comparison flips key effect signs or causes large uncontrolled drift, downgrade pipeline claim. |

## Minimal Experiment Bundle (Required for Strong AAMAS Position)

1. 2x2 mechanism isolation: `adaptive_window {off,on}` x `normative_memory {off,on}` with matched seeds.
2. Functional-form competition: `linear/logistic/threshold` for confidence gate (and compliance curve variants).
3. Schedule robustness: sync staged vs async event-driven; enforcement same-tick vs next-tick.

## Minimal Reporting Standard

- Pre-declare primary estimands: convergence time, adoption at fixed horizons, collapse probability, recovery time.
- Publish seed policy and seed list.
- Report effect sizes with confidence intervals (not p-values only).
- Release scripts/configs/raw CSV for end-to-end rerun.

## Source Pointers

- Web-verified full chain: `/Users/yifan/Documents/New project/meanField_memory/docs/MECHANISM_IDENTIFICATION_EVIDENCE_CHAIN_AAMAS_V2_WEBVERIFIED.md`
- Extended chain with implementation mapping: `/Users/yifan/Documents/New project/meanField_memory/docs/MECHANISM_IDENTIFICATION_EVIDENCE_CHAIN_AAMAS_V1.md`
