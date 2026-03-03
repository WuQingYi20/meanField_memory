# Variables, Formulas, and Parameter Reference

## Model Overview

This document provides a complete reference for all variables, formulas, and parameter values in the Dual-Memory Agent-Based Model for norm emergence. The model features agents playing a symmetric pure coordination game with two strategies {A, B}, equipped with two cognitive subsystems: experiential memory (FIFO-based belief tracking) and normative memory (DDM-based norm crystallisation).

---

## 1. Agent State Variables

Each agent maintains two memory systems, a confidence tracker, and derived action variables:

| Variable | Symbol | Type | Initial Value | Description |
|----------|--------|------|---------------|-------------|
| FIFO buffer | `fifo` | RingBuffer(w_max) | empty | Circular buffer storing observed partner actions |
| Experiential belief | b_exp^A | Float in [0,1] | 0.5 | Fraction of A in last w entries of FIFO |
| Confidence | C | Float in [0,1] | C_0 = 0.5 | Prediction accuracy tracker |
| Memory window | w | Int in [w_base, w_max] | w_base + floor(C_0 * (w_max - w_base)) = 4 | Number of recent FIFO entries used |
| Norm rule | r | {none, A, B} | none | Crystallised norm direction |
| Norm strength | sigma | Float in [0,1] | 0.0 (set to sigma_0 at crystallisation) | Strength of held norm |
| Anomaly counter | a | Int >= 0 | 0 | Count of observed violations since last crisis |
| DDM evidence | e | Float | 0.0 | Drift-diffusion evidence accumulator |
| Pending signal | pending_signal | {none, A, B} | none | Enforcement signal received (consumed next tick) |
| Compliance | compliance | Float in [0,1] | 0.0 | Derived: sigma^k |
| Effective belief | b_eff^A | Float in [0,1] | 0.5 | Derived: blended action probability for strategy A |

---

## 2. Model Parameters

### 2.1 Experiential Layer

| Symbol | Parameter | Default | Justification |
|--------|-----------|---------|---------------|
| N | Population size | 100 | Baseline; tested across N = 4 to 20,000 |
| T | Max simulation ticks | 3000 | Sufficient for convergence at all tested N |
| alpha | Confidence increase rate | 0.1 | **Calibration.** Additive increase on correct prediction. Asymmetric with beta to implement "fast collapse, slow build" of confidence, consistent with Behrens et al. (2007) finding that learning rate increases sharply under volatility. The alpha < beta constraint ensures confidence drops faster than it rises. |
| beta | Confidence decrease rate | 0.3 | **Calibration.** Multiplicative decay on wrong prediction. beta > alpha ensures asymmetric response: confidence collapses quickly (responding to volatility) but builds slowly (requiring sustained accuracy). Grounded in Behrens et al. (2007): the brain adjusts learning rate faster upward (high volatility) than downward (stable environment). |
| C_0 | Initial confidence | 0.5 | **Neutral starting point.** Agents begin with moderate confidence, reflecting no prior experience. Produces initial window w = 4 (midpoint of [2, 6]). |
| w_base | Minimum memory window | 2 | **Hertwig & Pleskac (2010); Nevo & Erev (2012).** Human decision-makers in decisions-from-experience rely on small, recent samples. Computational models fitting human data in repeated games find effective windows of 5-6 trials. w_base = 2 represents the minimum under high uncertainty (low C). |
| w_max | Maximum memory window | 6 | **Hertwig & Pleskac (2010); Nevo & Erev (2012).** Upper bound matches the 5-6 trial effective window found in human decision-making experiments. The [2, 6] range implements the volatility-adaptive mechanism from Behrens et al. (2007): uncertain agents use short windows (recent data only); confident agents use longer windows (more stable estimates). |

### 2.2 Normative Layer

| Symbol | Parameter | Default | Justification |
|--------|-----------|---------|---------------|
| theta_crystal | Crystallisation threshold | 3.0 | **Calibration.** The DDM evidence threshold |e| that triggers norm formation. Structurally analogous to Granovetter's (1978) adoption threshold but operates at the individual cognitive level through internal evidence integration. The phase boundary analysis (Section 4.1) shows robust cascade behaviour for theta in [3, 5]; convergence degrades above theta = 7. Lower values produce earlier crystallisation with more symmetric initial splits; higher values require stronger directional signal before committing. |
| sigma_0 | Initial norm strength | 0.8 | **Calibration.** Strength assigned at the moment of crystallisation. Set high enough that newly crystallised agents immediately exhibit strong compliance (compliance = 0.8^2 = 0.64), creating a noticeable behavioural commitment. Must be > sigma_min to prevent immediate dissolution. |
| theta_crisis | Crisis threshold (anomaly count) | 10 | **Calibration.** Number of accumulated norm violations before a crisis event triggers sigma decay. Controls how resilient norms are to transient perturbations. At 10, an agent in a 60-40 minority position takes roughly 25 ticks to reach crisis (violation rate ~ 0.4 per tick). |
| lambda_crisis | Crisis decay factor | 0.3 | **Calibration.** Multiplicative decay applied to sigma during crisis: sigma <- lambda_crisis * sigma. With sigma_0 = 0.8 and lambda_crisis = 0.3, first crisis reduces sigma to 0.24, second crisis to 0.072 (below sigma_min = 0.1, triggering dissolution). This means a norm typically survives at most one crisis before dissolving. |
| sigma_min | Dissolution threshold | 0.1 | **Calibration.** If sigma falls below this after a crisis, the norm dissolves and the agent re-enters the DDM. Set low enough that a single crisis from sigma_0 does not dissolve (0.8 * 0.3 = 0.24 > 0.1), but two crises do (0.24 * 0.3 = 0.072 < 0.1). |
| alpha_sigma | Norm strengthening rate | 0.005 | **Calibration.** Additive increase to sigma per conforming observation: sigma <- sigma + alpha_sigma * (1 - sigma). Deliberately slow — an agent seeing 100% conformity takes ~50 ticks to recover from sigma = 0.24 to sigma = 0.45. This asymmetry (fast decay via crisis, slow recovery) ensures that minority norms erode faster than they can rebuild. |
| k | Compliance exponent | 2.0 | **Calibration.** Controls the nonlinearity of the compliance function: compliance = sigma^k. With k = 2, compliance is a convex function of sigma, meaning agents with moderate sigma (e.g., 0.5) show low compliance (0.25), while agents with high sigma (e.g., 0.9) show strong compliance (0.81). This creates a sharp distinction between weakly and strongly committed agents. |
| theta_enforce | Enforcement strength threshold | 0.7 | **Calibration.** Minimum norm strength required for an agent to send enforcement signals. Ensures only agents with well-established norms enforce, preventing premature or inconsistent enforcement from recently crystallised agents. |
| gamma_signal | Signal amplification | 2.0 | **Calibration.** Amplification factor for enforcement signals entering the DDM: push = Phi * (1 - C) * gamma_signal * direction. Scales the enforcement push relative to the organic drift from experiential belief. |

### 2.3 Enforcement and Environment

| Symbol | Parameter | Default | Justification |
|--------|-----------|---------|---------------|
| Phi | Enforcement gain | 0.0 | **Disabled by default.** Global switch for enforcement; when Phi = 0, no enforcement signals are sent regardless of agent state. Tested at Phi in {0, 0.1, 0.2, ..., 5.0} in enforcement sweep experiments. The base model demonstrates the cascade mechanism without enforcement. |
| V | Additional observations per tick | 0 | **Minimal information baseline.** Number of additional strategy observations sampled from other pairs beyond the agent's own partner. V = 0 means each agent observes only its partner — the minimal-information condition. |

### 2.4 Convergence Criteria

| Symbol | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| thresh_majority | Behavioural majority threshold | 0.95 | Fraction playing dominant strategy >= 0.95 |
| thresh_belief_error | Belief error threshold | 0.10 | Mean |b_exp^A - fraction_A| < 0.10 |
| thresh_belief_var | Belief variance threshold | 0.05 | Cross-agent variance of b_exp < 0.05 |
| thresh_crystallised | Crystallised fraction threshold | 0.80 | Fraction of agents with a norm >= 0.80 |
| thresh_dominant_norm | Dominant norm fraction threshold | 0.90 | Fraction holding the majority norm >= 0.90 |
| convergence_window | Sustained convergence window | 50 ticks | All three layers must be met simultaneously for 50 consecutive ticks |

---

## 3. Formulas

### 3.1 Experiential Belief (Eq. 1)

```
b_exp^A = (#A in last w entries of FIFO) / w
```

- **What it does:** Computes a local, frequency-based estimate of how often strategy A is played in the population, using only the agent's recent interaction history.
- **Statistical property:** Unbiased estimator of the true population frequency f_A, with variance = f_A(1 - f_A) / w. With w in [2, 6], this variance is substantial, producing noisy estimates.
- **Source:** Hertwig & Pleskac (2010) — decisions-from-experience paradigm; agents rely on small, recent samples.

### 3.2 Confidence Update (Eq. 2)

```
If prediction correct:   C <- C + alpha * (1 - C)     [additive increase]
If prediction wrong:     C <- C * (1 - beta)           [multiplicative decay]
```

- **What it does:** Tracks prediction accuracy using an AIMD (Additive Increase, Multiplicative Decrease) rule. Correct predictions raise confidence slowly; incorrect predictions crash it quickly.
- **With defaults (alpha = 0.1, beta = 0.3):**
  - Correct: C goes 0.5 -> 0.55 -> 0.595 -> ...
  - Wrong: C goes 0.5 -> 0.35 -> 0.245 -> ...
- **Source:** Behrens et al. (2007) — the brain continuously tracks environmental volatility and adjusts learning rate. Volatile environments produce faster learning (shorter window); stable environments produce slower learning (longer window). The asymmetry (fast collapse, slow build) ensures agents respond quickly to environmental change.

### 3.3 Memory Window

```
w = w_base + floor(C * (w_max - w_base))
```

- **What it does:** Maps confidence to window size. High confidence -> long memory (stable environment -> trust more history); low confidence -> short memory (volatile environment -> rely on recent data).
- **With defaults:** w = 2 + floor(C * 4), so w ranges from 2 (at C = 0) to 6 (at C = 1).
- **Source:** Implements the volatility-adaptive learning rate from Behrens et al. (2007), where confidence C serves as the agent's estimate of environmental stability (inverse of perceived volatility).

### 3.4 DDM Evidence Accumulation (Eq. 3) — Pre-crystallisation

```
e <- e + (1 - C) * f_diff + s_push
```

where:
- `f_diff = b_exp^A - b_exp^B = 2 * b_exp^A - 1` (signed belief difference, in [-1, 1])
- `s_push = Phi * (1 - C) * gamma_signal * direction` (enforcement signal, if received)

**Crystallisation trigger:** When |e| >= theta_crystal:
- If e > 0: crystallise norm r = A
- If e < 0: crystallise norm r = B
- Norm strength set to sigma_0

- **What it does:** Accumulates evidence for one strategy over the other, gated by uncertainty. Low-confidence agents (uncertain about the environment) accumulate evidence faster and crystallise sooner.
- **Key property:** At symmetric 50-50 population (f_diff ~ 0), the drift is near zero and crystallisation does not occur. The DDM waits for experiential learning to break symmetry.
- **No additive noise:** Sampling noise is already captured by b_exp's finite-window variance; adding Gaussian noise would double-count uncertainty.
- **Source:** Germar et al. (2014) — majority opinion exposure alters drift rate in perceptual decision-making. Germar & Mojzisch (2019) — this shift persists after social influence removal. The confidence gating (1 - C) implements "copy when uncertain" (Rendell et al., 2010; Boyd & Richerson, 1985).

### 3.5 Post-Crystallisation Norm Maintenance

```
Conformity (observation matches norm r):
    sigma <- sigma + alpha_sigma * (1 - sigma)

Violation (observation != r):
    a <- a + 1                                    [anomaly accumulation]

Crisis (when a >= theta_crisis):
    sigma <- lambda_crisis * sigma                 [multiplicative decay]
    a <- 0                                         [reset counter]

Dissolution (when sigma < sigma_min after crisis):
    r <- none, e <- 0, sigma <- 0, a <- 0         [return to DDM]
```

- **What it does:** Implements a state machine: conformity strengthens the norm slowly; violations accumulate anomalies; enough anomalies trigger a crisis that sharply weakens the norm; if the norm is too weak after crisis, it dissolves and the agent re-enters the DDM.
- **Typical lifecycle:** With sigma_0 = 0.8, a minority-norm agent in a 60-40 environment accumulates ~0.4 anomalies/tick, reaching theta_crisis = 10 in ~25 ticks. First crisis: sigma -> 0.24. Second crisis: sigma -> 0.072 < sigma_min -> dissolution.
- **Source:** The crisis-dissolution pathway is the core cascade mechanism. No single empirical source; calibrated to produce the cascade dynamics.

### 3.6 Effective Belief / Blending Equation (Eq. 4)

```
If crystallised (r != none):
    compliance = sigma^k
    b_eff^A = compliance * b_norm^A + (1 - compliance) * b_exp^A

If uncrystallised (r = none):
    compliance = 0
    b_eff^A = b_exp^A
```

where `b_norm^A = 1 if r = A, 0 if r = B`.

- **What it does:** Blends the agent's norm direction with its experiential estimate. Strongly crystallised agents mostly follow their norm; weakly crystallised or uncrystallised agents rely on experience.
- **Example:** A-crystallised agent with sigma = 0.8, k = 2: compliance = 0.64. Even if b_exp^A = 0.5 (uncertain), b_eff^A = 0.64 * 1.0 + 0.36 * 0.5 = 0.82. The agent plays A with 82% probability despite uncertain experience.
- **This is the signal amplification mechanism:** Crystallised agents convert weak experiential signals into strong behavioural commitments, biasing the environment for other agents.

### 3.7 Action Selection

```
P(play A) = b_eff^A      [probability matching]
```

- **What it does:** Agents sample actions stochastically according to their effective belief. This is probability matching, not best-response.

### 3.8 MAP Prediction

```
prediction = A  if b_eff^A > 0.5
           = B  if b_eff^A < 0.5
           = random tie-break  if b_eff^A = 0.5
```

- **What it does:** Each agent predicts its partner's action using its own effective belief as a proxy. The prediction outcome drives the confidence update.

### 3.9 Enforcement Signal (Stage 5)

```
Condition: Phi > 0 AND sigma > theta_enforce AND partner violated norm
Signal delivered to partner's DDM (one-tick delay):
    s_push = Phi * (1 - C_partner) * gamma_signal * direction
```

- **What it does:** Crystallised agents with strong norms push violating partners' DDM evidence toward the enforcer's norm. Gated by the partner's uncertainty: confident partners are less affected.
- **Two channels:** (1) Beneficial: accelerates DDM crystallisation in pre-crystallised agents toward majority. (2) Harmful: triggers mutual violations between opposing crystallised agents, destabilising both.

---

## 4. Feedback Loop Structure

### Loop 1: Differential Erosion (slow)

```
n_A > n_B among crystallised agents
-> B-norm agents see more violations (partner likely plays A)
-> B-norms accumulate anomalies faster
-> B-norms reach crisis sooner, sigma_B decreases
-> B-agents' compliance drops, they act less distinctly B
-> Environment becomes more A-biased
-> (reinforces)
```

### Loop 2: Dissolution Cascade (fast)

```
sigma_B < sigma_min after crisis
-> B-norm agent dissolves, re-enters DDM
-> In A-biased environment, recrystallises to A (one-way ratchet)
-> n_A increases, n_B decreases
-> Accelerates Loop 1
-> (reinforces, exponential acceleration)
```

**Key result:** The cascade completes in O(1) time regardless of N because the dissolution-recrystallisation pathway operates locally at every minority-norm agent simultaneously (inherently parallel).

---

## 5. Scaling Properties

| Metric | Scaling | Evidence |
|--------|---------|----------|
| Convergence tick (with norms) | O(N^0.028) ~ O(1) | Log-log regression across N = 100 to 20,000 |
| Convergence tick (exp-only) | O(N) | Grows linearly; fails at large N |
| Dissolution count | O(N) | ~40% of agents undergo dissolution regardless of N |
| Cascade duration | ~O(log N) | 4 ticks at N=100, 44 ticks at N=20,000 |
| Normative speedup | Grows with N | 3x at N=20, 76x at N=20,000 |

---

## 6. Parameter Sensitivity Summary

| Parameter | Robust Range | Critical Boundary | Effect of Exceeding |
|-----------|-------------|-------------------|---------------------|
| theta_crystal | [3, 5] | ~7 (at N=100) | Convergence rate drops; sharper at large N |
| Phi | [0, 2] at low theta | Phi >= 5 at low theta | More turbulence (dissolutions), slower convergence |
| Phi | [0.3, 3] at high theta | --- | Helps convergence above phase boundary |
| w_base, w_max | [2, 6] | --- | Matches empirical window from Hertwig (2010) |
| alpha, beta | beta > alpha | --- | Asymmetry required for volatility-adaptive behaviour |

---

## 7. Key References for Parameter Choices

| Mechanism | Parameters | Key Reference | Finding Used |
|-----------|-----------|---------------|--------------|
| Small-sample learning | w_base = 2, w_max = 6 | Hertwig & Pleskac (2010); Nevo & Erev (2012) | Human decision-makers use effective windows of 5-6 trials in repeated games |
| Volatility-adaptive learning rate | alpha, beta, C, w(C) | Behrens et al. (2007) | Brain tracks environmental volatility; learning rate increases under volatility (= shorter window) |
| Social influence on drift rate | DDM with (1-C) gating | Germar et al. (2014) | Majority opinion exposure alters drift rate in perceptual decision-making |
| Persistent perceptual shift | Crystallisation + maintenance | Germar & Mojzisch (2019) | Social influence produces lasting cognitive change, not just transient compliance |
| Copy when uncertain | Drift gated by (1-C) | Rendell et al. (2010); Boyd & Richerson (1985) | "Copy when uncertain" strategies dominate in social learning tournaments |
| Adoption threshold | theta_crystal (endogenous) | Granovetter (1978) | Structural analogy; our threshold is endogenous (DDM accumulator) rather than exogenous |

---

*Document generated from cascade_report_advisor.tex and source code (src/params.jl, src/types.jl, src/stages.jl, src/init.jl).*
