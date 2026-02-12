# Individual Norm Evidence Chain for Conceptual Model V5

## Scope
- Question: Is the individual-level norm mechanism in `docs/conceptual_model_v5.tex` evidence-grounded?
- Focus: micro/individual mechanisms only (formation, compliance, enforcement gating).
- Rule: separate Evidence from Inference and report confidence.

## Sources Used
- Germar et al. (2013/2014), Social Influence and Perceptual Decision Making (PSPB), DOI: `10.1177/0146167213508985`
- Germar and Mojzisch (2019), Learning of social norms can lead to a persistent perceptual bias (JESP), DOI: `10.1016/j.jesp.2019.03.012`
- Duderstadt et al. (2022), Social norm learning from non-human agents can induce a persistent perceptual bias (Acta Psychologica), DOI: `10.1016/j.actpsy.2022.103691`
- Rendell et al. (2010), Why Copy Others? (Science), DOI: `10.1126/science.1184719`
- Wood et al. (2016), Model age-based and copy when uncertain biases (JECP), DOI: `10.1016/j.jecp.2016.06.005`
- Andrighetto et al. (2015), Perceived legitimacy of normative expectations... (Frontiers), DOI: `10.3389/fpsyg.2015.01413`
- Toribio-Florez et al. (2022/2023), Proof Under Reasonable Doubt (PSPB), DOI: `10.1177/01461672211067675`
- Behrens et al. (2007), Learning the value of information in an uncertain world (Nature Neuroscience), DOI: `10.1038/nn1954`

## Evidence Chain (Individual Level)

### Claim 1: Social information can shift perceptual evidence accumulation (DDM-compatible)
- V5 claim: social norm-related input enters an evidence accumulation process before crystallization.
- Evidence:
  - Germar et al. uses diffusion-model decomposition for social influence in perceptual decision making.
- Inference:
  - Modeling social influence as a DDM-like evidence channel is theoretically consistent.
- Confidence: High for direction, Medium for exact equation form.

### Claim 2: Norm learning can create persistent perceptual bias (not only short-term conformity)
- V5 claim: internalized norm has persistence.
- Evidence:
  - Germar and Mojzisch (2019) states persistent perceptual bias after social norm learning.
  - Duderstadt et al. (2022) reports similar persistence with non-human agents.
- Inference:
  - A persistent normative-memory state is justified.
- Confidence: Medium-High.

### Claim 3: Uncertainty increases reliance on social learning (copy when uncertain)
- V5 claim: lower predictive confidence should speed norm uptake.
- Evidence:
  - Rendell et al. (2010): social-learning strategies perform strongly in changing/uncertain environments.
  - Wood et al. (2016): experimental evidence for copy-when-uncertain behavior.
  - Behrens et al. (2007): volatility-sensitive information weighting in learning.
- Inference:
  - Using uncertainty/confidence as gate on social evidence uptake is supported.
- Confidence: Medium (construct-level support is good; exact linear form `(1-C)*consistency` is not directly estimated).

### Claim 4: Compliance can persist without monitoring via normative expectations
- V5 claim: behavior can be norm-constrained beyond pure payoff/punishment pressure.
- Evidence:
  - Andrighetto et al. (2015): perceived legitimacy of normative expectations can motivate compliance when unobserved.
- Inference:
  - An internal normative constraint in decision policy is well motivated.
- Confidence: High for qualitative mechanism, Medium for exact function `sigma^k`.

### Claim 5: Enforcement should be boundary-conditioned by violation certainty
- V5 claim: ambiguity should reduce punishment/enforcement.
- Evidence:
  - Toribio-Florez et al. (6 studies): third-party punishment decreases under ambiguity of norm violation.
- Inference:
  - Enforcement should be certainty-sensitive; this supports gating logic.
- Confidence: High for boundary condition, Low-Medium for any specific numeric threshold.

### Claim 6: Anomaly accumulation and crisis-style collapse
- V5 claim: violations accumulate and can trigger nonlinear collapse.
- Evidence:
  - Current paper set does not directly identify this exact update law.
- Inference:
  - Plausible modeling choice, but weakly identified empirically.
- Confidence: Low-Medium.

## Overall Assessment (Individual Norm)
- Verdict: Partially validated and structurally plausible.
- Strongly grounded parts:
  - Social influence represented in evidence-accumulation terms.
  - Persistence of norm-driven bias.
  - Uncertainty-sensitive social learning direction.
  - Ambiguity-sensitive enforcement boundary.
- Weakly grounded parts:
  - Exact drift functional form.
  - Exact compliance curve (`sigma^k`) and exponent value.
  - Exact anomaly-to-crisis operator and thresholds.

## Recommended Revisions to `docs/conceptual_model_v5.tex`

### R1. Reframe exact equations as testable hypotheses
- Replace hard wording like "is" with "we model as" or "we hypothesize" for:
  - drift equation,
  - compliance function,
  - crisis update function.

### R2. Separate mechanism support from parametrization choice
- After each key equation, add one sentence:
  - mechanism has empirical support,
  - chosen functional form/parameters are provisional for falsification and simulation tractability.

### R3. Add explicit evidence-grade labels
- Use labels in prose: strong / moderate / exploratory support.
- Minimum insertion points:
  - DDM formation section,
  - compliance section,
  - enforcement section,
  - crisis section.

### R4. Tighten uncertainty statement
- Keep directional claim: lower confidence -> higher social uptake.
- Avoid implying direct empirical calibration of the exact linear coefficient unless fitted evidence is provided.

### R5. Clarify enforcement threshold interpretation
- Treat `theta_enforce` as a model parameter to calibrate/sweep.
- Keep Toribio citation as qualitative boundary support, not numeric calibration.

### R6. Add "What evidence would change this model" subsection
- Suggested falsifiable checks:
  1. If low-confidence agents do not adopt earlier, revise drift gating.
  2. If compliance does not track internalization-strength proxy, revise `sigma^k`.
  3. If violation ambiguity does not reduce enforcement in your experiments, revise enforcement gate.

## Suggested Next Validation Steps
1. Parameter recovery for `sigma^k` versus linear/logistic alternatives.
2. Competing drift-gate models: linear, logistic, thresholded.
3. Explicit ambiguity manipulation in simulation/behavior to validate enforcement gate.
4. Report fitted intervals before asserting specific threshold values.

## Local Evidence Artifacts
- All downloaded files for this review are in:
  - `data/papers/individual_norm_review`
- Includes source pages and metadata snapshots (PMC/PubMed/OpenAlex/Crossref/Semantic Scholar).
