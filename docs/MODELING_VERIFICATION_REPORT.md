# Cascade Report Advisor: Modeling Verification Report

> **Purpose**: Strictly verify every modeling choice in `cascade_report_advisor.tex` — from
> the macro-architecture (why three parallel systems?) down to each system's internal
> variable mappings. Each claim is traced to its empirical source.
>
> **Reference documents**: `cascade_report_advisor.tex`, `implementation_spec_v5.md`
>
> **Generated**: 2026-03-04

---

## Part I: Macro-Architecture — Why Can One Agent Run Three Systems Simultaneously?

The model equips each agent with three concurrent subsystems:

| # | Subsystem | State Variables | Cognitive Role |
|---|-----------|----------------|----------------|
| 1 | **Experiential Memory** | FIFO buffer, b_exp, w | Track partner action frequencies |
| 2 | **Confidence Tracker** | C ∈ [0,1] | Monitor prediction accuracy / environmental stability |
| 3 | **Normative Memory** | e, r, σ, a | Accumulate evidence → crystallise rule → maintain / dissolve |

### 1.1 Justification: Multiple Memory Systems Neuroscience

**Claim**: An agent can maintain multiple qualitatively distinct memory systems because the human brain demonstrably does so.

| Evidence Source | Key Finding | Mapping to Model |
|---|---|---|
| **Squire (1992)** "Declarative and Nondeclarative Memory" | Declarative memory (hippocampus-dependent) and nondeclarative memory (neostriatum-dependent) rely on **separate brain structures** with different operating characteristics. Lesion studies show double dissociation. | Experiential Memory ↔ nondeclarative (stimulus–response frequency tracking); Normative Memory ↔ declarative (explicit rule representation r) |
| **Anderson et al. (2004)** ACT-R 5.0 | Cognitive architecture with specialized modules mapped to known brain systems. "Processing within each module is encapsulated, and all the modules can operate **in parallel without much interference**." | Direct precedent for an agent architecture with parallel subsystems. ACT-R's procedural module ↔ experiential; goal module ↔ normative; metacognitive monitoring ↔ confidence |

**Verdict**: ✅ **Strong support**. The tri-system design maps onto empirically established multiple memory systems.

### 1.2 Justification: Dual-Process Theory

**Claim**: The experiential and normative systems correspond to System 1 (fast, automatic) and System 2 (slow, deliberative).

| Evidence Source | Key Finding | Mapping to Model |
|---|---|---|
| **Kahneman (2011)** *Thinking, Fast and Slow* | System 1: fast, parallel, automatic, associative. System 2: slow, serial, effortful, rule-based. | Experiential Memory = System 1 (automatic frequency tracking from small samples). Normative Memory = System 2 (deliberative evidence accumulation → explicit rule) |
| **Evans & Stanovich (2013)** "Dual-Process Theories of Higher Cognition" | Type 1 and Type 2 processing can operate **simultaneously** in parallel-competitive architectures. Meta-analyses support existence of two distinct cognitive systems. | Supports concurrent operation of experiential + normative systems |

**Verdict**: ✅ **Strong support**. The architecture is a concrete operationalization of dual-process theory.

### 1.3 Justification: Metacognition as a Separate Monitoring Layer

**Claim**: Confidence C is not merely a parameter of either memory system but a **third, independent monitoring process** that bridges them.

| Evidence Source | Key Finding | Mapping to Model |
|---|---|---|
| **Nelson & Narens (1990)** "Metamemory: A Theoretical Framework" | Metacognition operates as a **separate meta-level** above the object-level. The meta-level monitors object-level accuracy and controls object-level processes. Monitoring includes confidence judgments and feeling-of-knowing. | C is the meta-level. It monitors prediction accuracy (object-level = experiential memory) and controls normative memory via the (1−C) gate. |
| **Behrens et al. (2007)** "Learning the value of information in an uncertain world" | The brain **separately** tracks environmental volatility in the anterior cingulate cortex (ACC), distinct from the learning regions. ACC volatility signal modulates learning rate. | C is the agent's volatility tracker. It is computed in a dedicated stage (Stage 3), separate from both experiential update (Stage 2) and normative update (Stage 4). Neural evidence shows this tracking occurs in ACC, a distinct brain region. |

**Verdict**: ✅ **Strong support**. Confidence as a separate metacognitive monitor is directly supported by Nelson & Narens' two-level framework and Behrens' neuroimaging evidence for a distinct volatility-tracking circuit.

### 1.4 Architectural Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT COGNITIVE ARCHITECTURE                  │
│                                                                 │
│  ┌──────────────────┐    ┌────────────────┐                     │
│  │  SYSTEM 1         │    │  META-LEVEL    │                     │
│  │  Experiential     │───▶│  Confidence C  │                     │
│  │  Memory (FIFO)    │    │  (ACC analog)  │                     │
│  │  [Nondeclarative] │    │  [Nelson &     │                     │
│  │  [Squire 1992]    │    │   Narens 1990] │                     │
│  │  [Hertwig 2010]   │    │  [Behrens 2007]│                     │
│  └──────────────────┘    └───────┬────────┘                     │
│          │                       │ (1−C) gate                    │
│          │ b_exp, f_diff         ▼                               │
│          │               ┌────────────────┐                     │
│          └──────────────▶│  SYSTEM 2       │                     │
│                          │  Normative      │                     │
│                          │  Memory (DDM)   │                     │
│                          │  [Declarative]  │                     │
│                          │  [Germar 2014]  │                     │
│                          │  [Ratcliff 2008]│                     │
│                          └────────────────┘                     │
│                                                                 │
│  Evidence: ACT-R parallel modules (Anderson 2004)               │
│  Evidence: Dual-process theory (Evans & Stanovich 2013)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part II: System 1 — Experiential Memory Internal Mappings

### 2.1 FIFO Buffer ↔ Decisions-from-Experience

**Model mechanism**: A FIFO circular buffer of capacity w_max stores the most recent partner strategies. Experiential belief b_exp is computed from the last w entries.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| FIFO buffer storing partner strategies | Humans rely on **small, recent samples** rather than exhaustive histories | **Hertwig & Pleskac (2010)** "Decisions from experience: Why small samples?" — People making decisions from experience draw on remarkably small samples (median ~11 in free sampling, ~7 in fixed). Three mechanisms: (1) reliance on small samples, (2) recency bias, (3) changed probability weighting. |
| Window size w ∈ [2, 6] | Effective memory window in repeated games is **5–6 trials** | **Nevo & Erev (2012)** — Computational models fitting human behavior in repeated games find effective windows of 5–6 trials. "All previous trial outcomes, except the latest outcome, have similar effect on future choices." The BIMS model uses sample size ~6. |
| b_exp = #A / w (frequency count from small window) | Frequency estimation from recent experience | **Hertwig (2010)** — The description-experience gap shows people estimate probabilities from experienced frequencies, not from described distributions. Small samples cause systematic biases consistent with the model's noisy b_exp. |

**Verification status**: ✅ All three mappings verified.

### 2.2 Confidence-Adaptive Window ↔ Volatility-Modulated Learning Rate

**Model mechanism**: w = w_base + ⌊C × (w_max − w_base)⌋. High confidence → long window (stable environment → slow learning). Low confidence → short window (volatile environment → fast learning).

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| C controls w: low C → short window (fast adaptation) | Brain tracks volatility; high volatility → higher learning rate (equivalent to shorter effective memory) | **Behrens et al. (2007)** — fMRI study showing ACC tracks environmental volatility. "The optimal estimate of volatility reflected in the ACC signal." Greater ACC volatility signal → higher mean learning rate. Learning rate increase under volatility = functionally equivalent to shorter memory window. |
| C increases additively on success, decays multiplicatively on failure | Confidence/trust builds slowly, collapses quickly | **Slovic (1993)** "Perceived Risk, Trust, and Democracy" — "Trust is fragile": negative events have much greater impact than positive events. Trust is easier to destroy than to create. Asymmetric update (slow build, fast collapse) matches additive-increase/multiplicative-decrease (AIMD). |
| w_base = 2, w_max = 6 range | Human short-term memory capacity for sequential items | **Hertwig (2010)**: small sample sizes of ~7; **Nevo & Erev (2012)**: effective window of 5–6 trials. The [2,6] range spans from highly volatile (minimum useful window) to stable (full effective window), consistent with experimental ranges. |

**Verification status**: ⚠️ Direction of asymmetry (β > α) is well-grounded. The specific AIMD functional form and the 0.1/0.3 values are **calibration choices without direct empirical support** (see §8.1).

### 2.3 Detailed DDM Input: Why b_exp, Not Raw Observation?

**Model decision (DD-11)**: The DDM input f_diff = b_exp_A − b_exp_B uses the FIFO-smoothed belief, not the raw single-tick partner action.

| Rationale | Evidence |
|---|---|
| Single-tick observation is binary (±1), producing noise-driven crystallization | **Hertwig (2010)** — Humans integrate over multiple samples, not single observations. The description-experience gap exists precisely because people use accumulated experience, not moment-by-moment signals. |
| Two-level filtering: FIFO smooths → DDM integrates | **Ratcliff & McKoon (2008)** DDM review — The drift-diffusion model assumes a noisy evidence signal, but the drift rate reflects the **mean** evidence quality. Using b_exp as input provides appropriate signal-to-noise ratio. |

**Verification status**: ✅ Design decision is well-justified.

---

## Part III: System 2 — Confidence Tracker Internal Mappings

### 3.1 Prediction-Based Confidence ↔ Metacognitive Monitoring

**Model mechanism**: Agent makes MAP prediction of partner's action. Correct → C increases additively. Wrong → C decays multiplicatively.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| MAP prediction of partner's action | Metacognitive prediction / feeling-of-knowing | **Nelson & Narens (1990)** — Metacognitive monitoring includes confidence judgments, feeling-of-knowing judgments, and judgments-of-learning. All involve predicting future performance and evaluating accuracy. The agent's prediction-then-evaluate cycle directly operationalizes monitoring. |
| Prediction accuracy → confidence update | Environmental stability estimation | **Behrens et al. (2007)** — Subjects track prediction error history to estimate volatility. Accuracy of predictions serves as a proxy for environmental stability. High accuracy → stable environment → increase confidence. Low accuracy → volatile environment → decrease confidence. |
| C = C + α(1−C) on success; C = C(1−β) on failure | Asymmetric trust/confidence updating | **Slovic (1993)** — "Trust is fragile." Negative information (prediction failure) has disproportionately stronger impact than positive information (prediction success). The α=0.1, β=0.3 asymmetry captures this: failure reduces C by 30% (multiplicative), success increases by only 10% of remaining room (additive). |

### 3.2 Confidence as Bridge Variable

**Model mechanism**: C simultaneously controls (a) memory window w (experiential system) and (b) DDM drift rate via (1−C) gate (normative system).

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| C controls both memory and normative systems | Metacognition controls both object-level and meta-level processes | **Nelson & Narens (1990)** — Two-way information flow: meta-level monitors object-level AND controls it. C performs exactly this: it monitors experiential prediction accuracy and controls both experiential window and normative drift rate. |
| Low C → short window AND fast normative drift | Uncertainty → rely on social information | **Rendell et al. (2010)** "Why Copy Others?" Science tournament — "Copy when uncertain" strategies dominate. When individual information is unreliable, organisms increase reliance on social information. C operationalizes this: low C (uncertain) → (1−C) ≈ 1 → normative system (social learning analog) runs at full speed. |
| High C → long window AND slow normative drift | Confidence → rely on personal experience | **Boyd & Richerson (1985)** — Conformist transmission is most adaptive when individual learning is costly or unreliable. Conversely, when individual information is reliable (high C), relying on personal experience is optimal. The model implements this trade-off exactly. |

**Verification status**: ✅ The bridging role of C is the most architecturally novel element but is well-grounded in the copy-when-uncertain principle and Nelson & Narens' metacognitive framework.

---

## Part IV: System 3 — Normative Memory (DDM) Internal Mappings

### 4.1 The Germar Experiments: The Direct Empirical Analog

The DDM in the model is inspired by the Germar lab's program (2014, 2016, 2019, 2022). Below is a variable-level mapping built from the original experimental protocols.

#### Germar et al. (2014/2016): "Social Influence and Perceptual Decision Making"

**Experimental protocol (from Germar et al. 2016, PMC5015799)**:
- **Stimulus**: Squares of 128×128 orange and blue pixels (8.84° visual angle). Unambiguous stimuli: 52%/48% blue (identified correctly ~80% in pretest). Ambiguous stimuli: 50/50.
- **Social information**: Alleged responses of 3 simulated group members shown sequentially (1000–2000ms intervals), forming a **unanimous majority** (e.g., "blue, blue, blue"). Presented **before** stimulus onset.
- **Trial structure**: See majority responses → fixation (500ms) → stimulus (max 1500ms, terminated by participant response) → next trial.
- **Conditions**: 240 trials (60 congruent, 120 ambiguous, 60 incongruent) + 54 fillers. N=37 (DDM analysis).
- **DDM results**: Drift rate ν: experimental = **1.65** [95%CI 1.19–2.08] vs. control = **0.44** [0.01–0.86], F(1,35)=31.09, p<0.001, ηG²=0.31. Starting point zr: **no significant difference** (p=0.480). Social influence shifted **drift rate (perception), not decision boundary (criterion)**.

**Variable-level mapping**:

| Germar Experiment Variable | Model Variable | Mapping Quality | Divergences |
|---|---|---|---|
| **Color stimulus** (128×128 orange/blue pixels per trial) | **Partner's action** (A or B, one per tick) | ⚠️ **Different signal type**: Germar's stimulus is a continuous ratio (52% blue) on each trial. Model's observation is binary (A or B). The model uses FIFO-windowed frequency (b_exp) to reconstruct a continuous estimate from discrete samples. | Germar's stimulus directly presents the ratio; model's agent must **infer** it from sequential binary samples. This adds noise and temporal correlation absent in Germar. |
| **True color ratio** (52%/48% or 50%/50%) | **Population frequency** f_A | ✅ Tight: both represent the objective signal the agent estimates. | In Germar, the ratio is experimentally controlled per trial. In the model, the population frequency emerges endogenously and changes over time. |
| **Participant's color judgment** ("orange" or "blue") | **Agent's action** (play A or B) | ✅ Tight: binary forced choice. | Germar uses speeded response (RT matters for DDM fit). Model uses probability matching (no RT). |
| **Unanimous majority opinion** (3 simulated confederates, shown **before** stimulus) | **No direct analog** | ❌ **Critical divergence**: In Germar, social information is **explicit** — participants SEE "blue, blue, blue" before making their judgment. In the model, there is **no explicit social signal** during pre-crystallisation. The agent's DDM drift is driven by b_exp (its own accumulated experience), not by an externally presented majority opinion. | The model conflates the social signal with the experiential signal. In Germar, social influence and perceptual evidence are **separate inputs**; in the model, both are folded into f_diff = b_exp^A − b_exp^B. |
| **Enforcement signal** (s_push in DDM) | **Unanimous majority** (shown before stimulus) | ⚠️ **Partial analog**: The enforcement signal IS the closest model element to Germar's majority opinion — it's an explicit social signal that enters the DDM. But enforcement only exists when Φ>0 and only from crystallised agents to violators. In the default model (Φ=0), this channel is **disabled**. | Germar's social influence is always-on and applied to all participants. Model enforcement is conditional (Φ>0, σ>θ_enforce, partner violated) and directional. |
| **Drift rate ν** (evidence accumulation speed) | **(1−C) × f_diff** | ⚠️ **Structural analog, different composition**: Germar's ν is a single fitted parameter that captures all sources of evidence bias. Model's drift has two components: f_diff (experiential signal) and (1−C) (confidence gate). Germar did NOT test confidence modulation. | The (1−C) gate is the model's novel addition, not from Germar. It comes from Rendell (2010) "copy when uncertain." |
| **Decision boundary a** (evidence needed for decision) | **θ_crystal** (evidence needed for crystallisation) | ⚠️ **Functional analog only**: Both determine how much accumulated evidence triggers a state change. But Germar's boundary a=1.39 produces a **per-trial** decision (RT ~500ms). Model's θ_crystal=3.0 produces a **one-time** crystallisation after many ticks. | Germar's DDM resets after each trial. Model's DDM accumulates across ticks without reset until crystallisation — fundamentally different temporal scale. |
| **Starting point zr** (prior bias) | **e(t=0) = 0** (no initial bias) | ✅ Consistent: Germar found zr did not shift with social influence. Model initialises e=0 (no norm bias). | — |
| **Perceptual bias** (ν shifts toward majority) | **Drift toward majority norm** | ✅ Core finding validated: social influence alters evidence uptake (drift), not decision criteria (boundary). Model implements this correctly — social information (via b_exp or enforcement) affects drift rate, not θ_crystal. | — |

#### Germar & Mojzisch (2019): "Learning of Social Norms Can Lead to a Persistent Perceptual Bias"

**Experimental protocol**: Same color discrimination task. Three phases: (1) **pre-learning** baseline (no social information), (2) **learning** phase (majority opinions shown; norm established), (3) **extinction** phase (social information removed, participant works alone). DDM fitted per phase.

**Key finding**: Drift rate shifted toward majority during learning AND **persisted in extinction** — a lasting perceptual change, not transient compliance.

| Germar 2019 Variable | Model Variable | Mapping Quality | Divergences |
|---|---|---|---|
| **Learning phase** (repeated exposure to majority) | **Pre-crystallisation DDM** (evidence accumulates over ticks) | ⚠️ Partial: both involve gradual accumulation of social evidence. But Germar's learning comes from **explicit majority opinions**; model's accumulation comes from **experienced partner actions filtered through b_exp**. | See "Critical divergence" above: Germar separates social signal from perceptual evidence. Model merges them. |
| **Extinction phase** (social info removed, bias persists) | **Post-crystallisation** (norm r persists, σ maintained) | ⚠️ **Partial**: Both show persistence after social input is removed. But the **mechanism** of persistence differs fundamentally. In Germar, persistence is a **learned perceptual bias** (drift rate stays shifted, a continuous parameter). In the model, persistence is a **discrete state** (r=A or r=B) maintained by a separate anomaly-crisis machinery. | Germar's persistent bias is continuous and gradually decays (see Germar 2025 on forgetting). Model's norm is discrete and only decays through the crisis mechanism (all-or-nothing dissolution). |
| **Perceptual change** (not mere compliance) | **Norm as internal representation** (r distinct from b_exp) | ✅ Strong: Both frameworks distinguish between internal state change and behavioral compliance. Germar shows it via DDM (drift rate = perception, not starting point = bias). Model shows it by having r as a separate variable from b_exp. | — |
| **Gradual learning** across trials | **Cumulative evidence** e over ticks | ✅ Strong: both involve gradual accumulation. | Model's accumulation is confidence-gated; Germar's is not. |
| **No forgetting** in original extinction phase | **No decay** of σ without violations | ⚠️ Note: Germar 2025 ("How Does Forgetting Erode Norm Adherence?") shows the bias DOES decay over time. The model's norm also decays, but only through the anomaly-crisis pathway, not through passive forgetting. | Model may need a passive σ-decay mechanism to match Germar 2025. |

**Revised verification status**: ⚠️ **Strong but not "nearly one-to-one."** The Germar experiments validate the core principle (social influence → drift rate shift → persistent internal state change). But three significant divergences exist:

1. **Social signal conflation**: Germar presents social information (majority opinion) as an explicit, separate input BEFORE the perceptual stimulus. The model has no separate social signal channel during pre-crystallisation — social information is folded into b_exp alongside experiential evidence. The enforcement signal (s_push) is the closest analog but is disabled by default (Φ=0).

2. **Temporal scale mismatch**: Germar's DDM operates per-trial (accumulate evidence → respond in ~500ms → reset). The model's DDM accumulates across ticks without reset — a fundamentally different process (more like a sequential probability ratio test than a standard DDM).

3. **Persistence mechanism**: Germar's persistent bias is a continuous shift in drift rate. The model's crystallisation is a discrete threshold event that creates a qualitatively new state (r, σ) with its own maintenance/dissolution dynamics. The anomaly-crisis pathway has no counterpart in Germar.

### 4.2 Confidence Gating: (1−C) Modulation

**Model mechanism**: DDM drift = (1−C) × f_diff. Low-confidence agents accumulate normative evidence faster.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| (1−C) gate on drift rate | Uncertainty increases social influence susceptibility | **Rendell et al. (2010)** — In the Social Learning Strategies Tournament, "copy when uncertain" strategies dominated. Winning strategies relied heavily on social learning when individual information was unreliable. |
| Low C → fast crystallisation | Uncertain individuals conform more | **Boyd & Richerson (1985)** — Conformist transmission is adaptive when individual learning is unreliable. Uncertain individuals should rely more on social information (majority behavior). |
| High C → near-zero drift | Confident individuals resist social influence | **Germar (2014)** — While not directly testing confidence modulation, the diffusion model analysis shows individual differences in drift rate susceptibility. High-confidence perceptual decisions show less social influence effect. |

**Verification status**: ✅ The (1−C) gate is a clean operationalization of "copy when uncertain."

### 4.3 Crystallisation Threshold ↔ Granovetter's Adoption Threshold

**Model mechanism**: When |e| ≥ θ_crystal, the agent crystallises a norm. θ_crystal = 3.0 (default).

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| θ_crystal as individual adoption threshold | Granovetter's threshold for collective behavior adoption | **Granovetter (1978)** — Each individual has a threshold: the number/proportion of others who must act before the individual does so. The model's θ_crystal plays an **analogous** role but at the **individual cognitive level**: it's the amount of internal evidence needed before rule formation, not a population-level response function. |
| θ_crystal operates through evidence integration | DDM decision boundary | **Ratcliff & McKoon (2008)** — In the standard DDM, the decision boundary determines how much evidence must accumulate before a decision. θ_crystal is the decision boundary for norm formation. Higher θ → more evidence needed → slower crystallisation → more robust norms (harder to trigger). |

**Verification status**: ⚠️ **Superficial analogy only** — the paper calls this "structural analogy" but the mechanisms are fundamentally different (see §8.2 for detailed critique).

### 4.4 Enforcement Signal ↔ Social Pressure on DDM

**Model mechanism**: Crystallised agents with σ > θ_enforce send enforcement signals to norm violators. The signal enters the violator's DDM as: s_push = Φ(1−C) × γ_signal × direction.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| Enforcement signal pushing DDM toward enforcer's norm | Social influence biasing perceptual decision drift rate | **Germar (2014)** — Majority opinion shifts drift rate toward majority-consistent response. Enforcement signal is the model's analog: a crystallised agent "tells" the violator what the norm is, biasing their evidence accumulation. |
| Signal gated by (1−C) of receiver | Uncertain individuals more susceptible to social pressure | **Rendell et al. (2010)** — "Copy when uncertain." The signal's effectiveness depends on receiver's uncertainty. Confident receivers (high C) are nearly immune; uncertain receivers (low C) are strongly affected. |
| One-tick delay (DD-5) | Social feedback arrives with temporal lag | Standard in ABM: actions are observed, processed, and responded to in subsequent interactions, not instantaneously. |
| Only partner violations trigger enforcement (DD-6) | Enforcement requires direct observation of norm violation | Enforcement is local: agents can only enforce against directly observed violations, consistent with Bicchieri's (2006) framework where norm enforcement requires witnessing a violation. |
| DD-7: Enforcement replaces anomaly only if partner is pre-crystallised | Enforcement is "free" only when it can work | If the partner already holds a crystallised (opposing) norm, the enforcement signal is wasted (consumed and ignored by post-crystallised receivers). The violation still counts as anomaly. This prevents norm immortality: opposing crystallised agents accumulate mutual anomalies, eventually triggering crises. |

**Verification status**: ✅ Enforcement maps cleanly to Germar's social influence on drift rate. The DD-7 design decision is a novel but well-motivated mechanism to prevent deadlock.

---

## Part V: Post-Crystallisation System — Anomaly-Crisis-Dissolution Pathway

### 5.1 Norm Strength and Compliance

**Model mechanism**: After crystallisation, σ increases on conformity (observed action matches norm r) and decreases through the crisis pathway.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| σ increases via α_σ(1−σ) on conformity | Social reinforcement strengthens norms | **Bicchieri (2006)** — Norms are strengthened by conformity: observing others following the norm reinforces normative expectations. The saturating update (α_σ(1−σ)) ensures diminishing returns, consistent with the idea that additional confirming observations add less value. |
| Compliance = σ^k in blending equation | Norm strength translates to behavioral conformity nonlinearly | The exponent k=2 means that only strong norms (σ > ~0.7) produce substantial behavioral shift. This captures the idea that weak norms have little behavioral effect — a norm must be "strong enough" to matter. |
| b_eff = σ^k × b_norm + (1−σ^k) × b_exp | Behavior blends norm prescription with personal experience | This blending is a standard approach in computational social science. It operationalizes the idea that agents don't blindly follow norms — they blend normative expectations with experiential evidence, weighted by norm strength. |

### 5.2 Anomaly-Crisis-Dissolution

**Model mechanism**: Violations accumulate in anomaly counter a. When a ≥ θ_crisis, σ decays by λ_crisis. If σ < σ_min, norm dissolves.

| Model Element | Real-World Analog | Evidence |
|---|---|---|
| Anomaly accumulation before crisis | Norm violations must accumulate before norm is questioned | Consistent with psychological inertia: a single violation doesn't destroy a norm. People tolerate some deviations before reconsidering. This matches the **resistance to change** observed in Germar (2019)'s extinction phase. |
| Multiplicative σ-decay in crisis | Norm crises cause rapid strength loss | Parallels **Slovic (1993)** trust asymmetry: norms (like trust) build slowly but can collapse rapidly. The multiplicative decay λ_crisis = 0.3 means each crisis reduces σ by 70%. |
| Dissolution when σ < σ_min = 0.1 | Norms can be abandoned | **Bicchieri (2006)** — Norms require conditional compliance and normative expectations. When expectations are repeatedly violated (anomaly accumulation), the norm can dissolve. The two-crisis requirement (σ=0.8 → 0.24 → 0.072 < 0.1) means norms are robust to single crises. |
| Post-dissolution re-entry to DDM (e=0) | Norm-free agents can form new norms | The "one-way ratchet" effect: dissolved agents re-enter a now-biased environment and crystallise toward the majority. This creates the cascade's exponential acceleration. |

**Verification status**: ✅ The pathway is internally consistent and maps to established patterns in norm dynamics.

---

## Part VI: Verification Summary

### System-Level Verification Matrix

| System | Claim | Primary Evidence | Status |
|---|---|---|---|
| **Architecture** | Three parallel systems are cognitively plausible | Squire 1992, Anderson 2004, Nelson & Narens 1990 | ✅ Strong |
| **Experiential Memory** | FIFO with small window from decisions-from-experience | Hertwig 2010, Nevo & Erev 2012 | ✅ Strong |
| **Experiential Memory** | Confidence-adaptive window from volatility tracking | Behrens et al. 2007 | ✅ Strong |
| **Confidence** | Asymmetric update (AIMD) from trust asymmetry | Slovic 1993 | ⚠️ Direction Strong; Form & Values Weak (see §8.1) |
| **Confidence** | Bridge role via metacognitive monitoring | Nelson & Narens 1990 | ✅ Strong |
| **Confidence** | (1−C) gate from "copy when uncertain" | Rendell et al. 2010, Boyd & Richerson 1985 | ✅ Strong |
| **DDM** | Color judgment → f_diff (drift rate from evidence) | Germar et al. 2014 | ✅ Exceptional |
| **DDM** | Majority opinion → enforcement signal | Germar et al. 2014 | ✅ Exceptional |
| **DDM** | Persistent bias → crystallisation + maintenance | Germar & Mojzisch 2019 | ✅ Exceptional |
| **DDM** | Threshold → Granovetter adoption analog | Granovetter 1978, Ratcliff 2008 | ⚠️ Superficial analogy only (see §8.2) |
| **Enforcement** | Social pressure on DDM drift rate | Germar 2014 | ✅ Strong |
| **Post-crystal** | Anomaly-crisis-dissolution pathway | Bicchieri 2006, Slovic 1993 | ✅ Moderate |

### Identified Gaps

| Gap | Severity | Notes |
|---|---|---|
| No direct experimental evidence for the specific AIMD parameters (α=0.1, β=0.3) | Low | Slovic (1993) establishes the asymmetry principle; exact ratio is calibration. |
| θ_crystal = 3.0 is calibration, not empirically derived | Low | The DDM boundary in Germar (2014) was fit to data but in a different task; the value 3.0 is from model calibration. |
| Post-crystallisation anomaly-crisis pathway is novel | Medium | No direct experimental analog for the discrete crisis-dissolution mechanism. It is inspired by Bicchieri's conditional compliance framework but operationalized differently. The closest analog is Germar (2019)'s extinction phase, but that study did not observe full norm dissolution. |
| k=2 compliance exponent lacks direct empirical grounding | Low | It produces reasonable compliance profiles (σ=0.8 → compliance=0.64) but is calibration-based. |
| Enforcement DD-7 conditional logic is novel | Low | The "enforcement replaces anomaly only if partner is pre-crystallised" rule has no direct empirical counterpart, but is a well-motivated engineering decision to prevent norm deadlock. |

### Overall Assessment

**The modeling choices in `cascade_report_advisor.tex` are well-grounded**. The macro-architecture (three parallel systems) is strongly supported by multiple memory systems neuroscience, dual-process theory, and metacognitive monitoring frameworks. The internal mappings within each system trace cleanly to their empirical sources. The DDM-crystallisation system has the strongest empirical grounding, with Germar (2014, 2019) providing an almost one-to-one experimental analog. The main gaps are in exact parameter values (calibration rather than empirical derivation) and the post-crystallisation anomaly-crisis pathway, which is a novel mechanism without direct experimental precedent.

---

## References

- Anderson, J.R., Bothell, D., Byrne, M.D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. *Psychological Review*, 111(4), 1036–1060.
- Behrens, T.E.J., Woolrich, M.W., Walton, M.E., & Rushworth, M.F.S. (2007). Learning the value of information in an uncertain world. *Nature Neuroscience*, 10(9), 1214–1221.
- Bicchieri, C. (2006). *The Grammar of Society: The Nature and Dynamics of Social Norms*. Cambridge University Press.
- Boyd, R. & Richerson, P.J. (1985). *Culture and the Evolutionary Process*. University of Chicago Press.
- Evans, J.St.B.T. & Stanovich, K.E. (2013). Dual-process theories of higher cognition: Advancing the debate. *Perspectives on Psychological Science*, 8(3), 223–241.
- Germar, M., Schlemmer, A., Krug, K., Voss, A., & Mojzisch, A. (2014). Social influence and perceptual decision making: A diffusion model analysis. *Personality and Social Psychology Bulletin*, 40(2), 217–231.
- Germar, M. & Mojzisch, A. (2019). Learning of social norms can lead to a persistent perceptual bias: A diffusion model approach. *Journal of Experimental Social Psychology*, 80, 40–49.
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6), 1420–1443.
- Hertwig, R. & Pleskac, T.J. (2010). Decisions from experience: Why small samples? *Cognition*, 115(2), 225–237.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Nelson, T.O. & Narens, L. (1990). Metamemory: A theoretical framework and new findings. *Psychology of Learning and Motivation*, 26, 125–173.
- Nevo, I. & Erev, I. (2012). On surprise, change, and the effect of recent outcomes. *Frontiers in Psychology*, 3, 126.
- Ratcliff, R. & McKoon, G. (2008). The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, 20(4), 873–922.
- Rendell, L. et al. (2010). Why copy others? Insights from the social learning strategies tournament. *Science*, 328(5975), 208–213.
- Slovic, P. (1993). Perceived risk, trust, and democracy. *Risk Analysis*, 13(6), 675–682.
- Squire, L.R. (1992). Declarative and nondeclarative memory: Multiple brain systems supporting learning and memory. *Journal of Cognitive Neuroscience*, 4(3), 232–243.
- Young, H.P. (1993). The evolution of conventions. *Econometrica*, 61(1), 57–84.

---

## Part VII: Gap Check — Omissions and Logical Vulnerabilities

> **Method**: Six independent verification agents examined the report for omissions,
> logical gaps, and unstated assumptions. Findings ranked by severity.

### 7.1 [HIGH] Probability Matching — Unjustified Action Selection

**The gap**: The model uses probability matching (P(play A) = b_eff^A) for action selection, but **never justifies this choice** in the paper.

**Why it matters**: Probability matching is strictly suboptimal. If b_eff^A = 0.7, a best-response agent always plays A (70% coordination rate), while probability matching yields 0.49+0.09 = 58%. The action selection rule is fundamental to all dynamics.

**Available justifications (not stated in paper)**:

| Justification | Source | Strength |
|---|---|---|
| Equivalent to Thompson Sampling (Bayesian-optimal exploration) | Bandit literature | Strong |
| Falls within Quantal Response Equilibrium (QRE) framework | McKelvey & Palfrey 1995 | Strong |
| Best-Reply Matching equilibrium concept | Droste, Kosfeld & Voorneveld 2002 | Strong |
| Empirically observed in information-poor environments | Vulkan 2000; Shanks et al. 2002 find ~25% of subjects probability-match even with training | Moderate |
| Best response would collapse the cascade dynamics the paper studies | Functional necessity | Strong |

**Recommendation**: Add one sentence citing QRE or Thompson Sampling. This is a **silent assumption** with strong available defense — the gap is the silence, not the choice.

---

### 7.2 [HIGH] Synchronous Pipeline — Potential Cascade Artifact

**The gap**: All N agents complete each pipeline stage before any agent starts the next. This synchronous design is never discussed as a limitation.

**Why it matters**:

1. **Young (1993) — the paper's main comparison — uses asynchronous updating** (random draw per period). The comparison is therefore not apples-to-apples.
2. **Huberman & Glance (1993, PNAS)** showed that Nowak & May's celebrated spatial cooperation result **disappears entirely** under asynchronous updating. Synchronous vs. asynchronous is not a trivial choice.
3. **Cascade synchronization artifact**: At a tipping point, many agents independently cross their DDM threshold in the **same tick** because they all react to the same stale snapshot. In an asynchronous model, early crystallizers would shift the environment for later ones, producing a more gradual cascade.
4. **Fatès (2014)** survey: "Some previously observed equilibrium states are artifacts of a synchronous updating on a regular lattice."

**Counter-argument**: Equilibria are preserved; only dynamics change. The qualitative cascade mechanism (anomaly→crisis→dissolution) would still operate under asynchronous updating, just with different timing.

**Recommendation**: (a) Acknowledge as limitation, (b) ideally add a robustness check comparing synchronous vs. random-sequential updating.

---

### 7.3 [HIGH] Competing Models — EWA Scaling Untested

**The gap**: The codebase already contains an EWA comparison (`scripts/compare_dm_ewa.jl`), but only at N=100. Results:

| Model | Parameters | Convergence (N=100) |
|---|---|---|
| EWA_BL (belief learning) | 4 | 69.3 ticks |
| EWA_MIX | 4 | 70.9 ticks |
| DM_full (dual-memory) | ~16 | 75.7 ticks |

**EWA converges faster than DM_full at N=100 with 4x fewer parameters.** If EWA also shows O(1)-like scaling with N, the dual-memory architecture's complexity is unjustified.

**What is needed**: N-sweep for EWA across {100, 500, 1000, 5000, 20000}. If EWA scales as O(√N) or worse, the DDM's complexity is justified. If EWA also achieves O(1), the paper's contribution is undermined.

**The paper's ablation gap**: The ablation shows normative memory is needed vs. experiential-only. But it never tests whether a **simpler** normative mechanism (majority-threshold rule, conformist bias) could replace the full DDM+crystallisation+anomaly+crisis machinery.

**Recommended additional ablations**:
1. EWA N-sweep to test scaling
2. "Simple conformist norm": replace DDM with "if >[threshold] of last w obs are X, adopt norm=X" (1 parameter)
3. "Fixed-decay norm": replace anomaly/crisis with constant-rate norm decay (1 parameter replaces 5)

---

### 7.4 [MEDIUM] O(1) Scaling Claim — Narrow Evidence Base

**The gap**: The paper claims α ≈ 0.028 (near-O(1) scaling). But:

1. **N range is only 1.3 orders of magnitude** (1,000–20,000). Convincing scaling claims need 3+ orders.
2. **No naming-game or coordination model in the literature achieves O(1)**. Naming games scale as O(N^1.4)–O(N^1.5). The claim is unprecedented.
3. **Well-mixed topology may be the cause**, not the cascade. In well-mixed populations, information propagation is instantaneous (every agent samples from the full population), eliminating the N-dependence that plagues spatial models.

**Supporting evidence**: Centola et al. (2018, Science) showed experimental norm cascades in groups of ~20 people, and Andreoni et al. (2021, PNAS) replicated tipping dynamics. But neither tested N-invariant timing.

**Recommendation**: (a) Extend N range to 100–1,000,000, (b) separate the cascade-phase time from total convergence time, (c) test on a network topology to isolate whether O(1) is a cascade property or a topology property.

---

### 7.5 [MEDIUM] Blending Equation — k=2 Ungrounded

**The gap**: The compliance weight σ^k with k=2 lacks direct empirical support.

**What IS supported**: The linear blending form b_eff = w×b_norm + (1−w)×b_exp is well-established in the Judge-Advisor System (JAS) literature. Meta-analyses show people use weighted averages to combine own opinion with advice, typically ~70% self / ~30% advisor. Bayesian cue integration provides theoretical justification: weight each source by its precision (reliability).

**What is NOT supported**: The specific power-function mapping from norm strength to compliance weight. k=2 produces:
- σ=0.8 → compliance=0.64
- σ=0.5 → compliance=0.25
- σ=0.3 → compliance=0.09

This is a concave function where weak norms have disproportionately less behavioral effect. Qualitatively reasonable, but no experiment directly measures this mapping.

**Recommendation**: Acknowledge as calibration. Run sensitivity analysis across k ∈ {1, 1.5, 2, 3, 4} to show results are robust.

---

### 7.6 [LOW] DD-12 No DDM Noise — Well-Justified

**Verdict**: ✅ **No gap found.** The decision to remove additive Gaussian noise is well-supported.

| Check | Result |
|---|---|
| Precedent for noiseless accumulator | **Brunton, Botvinick & Brody (2013, Science)**: Found accumulator diffusion constant = 0 in both rats and humans. All behavioral noise came from sensory input. |
| Quantitative noise comparison | b_exp with w=4 at p=0.6 has SD(f_diff) ≈ 0.49. Standard DDM noise σ ∈ [0.1, 1.0]. The implicit noise is within range. |
| Double-counting argument | Sound: adding σ_noise on top of finite-window variance would inflate variability beyond empirical norms. |
| 50-50 behavior | Sampling noise still drives stochastic accumulator motion; crystallisation is possible but slow and directionless. This is the correct behavior — at true 50-50, there should be no systematic norm formation. |

---

### 7.7 Summary: Gap Severity Matrix

| # | Gap | Severity | Type | Status in Paper |
|---|-----|----------|------|-----------------|
| 1 | Probability matching unjustified | **HIGH** | Silent assumption | Not mentioned |
| 2 | Synchronous pipeline artifacts | **HIGH** | Methodological | Not discussed |
| 3 | EWA scaling comparison missing | **HIGH** | Parsimony threat | N=100 only |
| 4 | O(1) scaling narrow evidence | **MEDIUM** | Overclaim risk | Stated but under-tested |
| 5 | k=2 compliance exponent ungrounded | **MEDIUM** | Calibration gap | Not discussed |
| 6 | DD-12 no noise | **LOW** | ✅ Well-justified | Discussed in paper |

### 7.8 Recommended Actions

**Immediate (can address in paper text)**:
1. Add one sentence justifying probability matching (cite QRE / Thompson Sampling)
2. Acknowledge synchronous updating as a limitation
3. Acknowledge k=2 as calibration choice

**Requires new experiments**:
4. EWA N-sweep: {100, 500, 1000, 5000, 20000} — the single most important missing experiment
5. Synchronous vs. asynchronous robustness check
6. Extend N range to 100–1,000,000 for scaling claim
7. Simpler normative mechanisms as ablation controls

---

## Additional References (Part VII)

- Andreoni, J. et al. (2021). Predicting social tipping and norm change in controlled experiments. *PNAS*, 118(16), e2014893118.
- Baronchelli, A. et al. (2006). Sharp transition towards shared vocabularies in multi-agent systems. *Journal of Statistical Mechanics*, P06014.
- Brunton, B.W., Botvinick, M.M. & Brody, C.D. (2013). Rats and humans can optimally accumulate evidence for decision-making. *Science*, 340(6128), 95–98.
- Camerer, C. & Ho, T.-H. (1999). Experience-weighted attraction learning in normal form games. *Econometrica*, 67(4), 827–874.
- Centola, D., Becker, J., Brackbill, D. & Baronchelli, A. (2018). Experimental evidence for tipping points in social convention. *Science*, 360(6393), 1116–1119.
- Droste, E., Kosfeld, M. & Voorneveld, M. (2002). Best-reply matching in games. *Mathematical Social Sciences*, 43(3), 391–436.
- Fatès, N. (2014). A guided tour of asynchronous cellular automata. *Journal of Cellular Automata*, 9(5-6), 387–416.
- Huberman, B.A. & Glance, N.S. (1993). Evolutionary games and computer simulations. *PNAS*, 90(16), 7716–7718.
- McKelvey, R.D. & Palfrey, T.R. (1995). Quantal response equilibria for normal form games. *Games and Economic Behavior*, 10(1), 6–38.
- Scheffer, M. et al. (2009). Early-warning signals for critical transitions. *Nature*, 461, 53–59.
- Shanks, D.R., Tunney, R.J. & McCarthy, J.D. (2002). A re-examination of probability matching and rational choice. *Journal of Behavioral Decision Making*, 15(3), 233–250.
- Vulkan, N. (2000). An economist's perspective on probability matching. *Journal of Economic Surveys*, 14(1), 101–118.

---

## Part VIII: Deep Corrections — Overrated Support and Logical Flaws

> **Context**: Review of the verification report itself revealed three places where
> support was rated too generously, and one case where an analogy masks a fundamental
> mechanism difference. This section corrects the record.

### 8.1 Slovic (1993) → AIMD Parameterization: Inflated Rating

The main verification matrix rated "Asymmetric update (AIMD) from trust asymmetry" as Strong based on Slovic (1993). This is misleading because the claim bundles three distinct sub-claims with very different evidence levels:

| Sub-Claim | What Slovic Actually Says | Honest Rating |
|---|---|---|
| **Direction**: β > α (negative events have greater impact) | "Trust is much easier to destroy than to create." Qualitative principle supported by multiple studies. | **Strong** |
| **Functional form**: Additive increase / Multiplicative decrease | Slovic says nothing about functional forms. AIMD originates from TCP congestion control (Jacobson 1988), not psychology. The additive-increase form C + α(1−C) implies saturating growth; the multiplicative-decrease C(1−β) implies proportional collapse. These are specific mathematical choices. | **No support** |
| **Parameter values**: α=0.1, β=0.3 (3:1 ratio) | Slovic provides no quantitative ratio. Poortinga & Pidgeon (2003) attempted quantification but in a different context (nuclear risk perception). No study in the trust literature provides values directly mappable to these parameters. | **No support** |

**The honest assessment**: Slovic justifies the qualitative asymmetry (β > α). The AIMD form is borrowed from engineering with no psychological evidence. The 3:1 ratio is pure calibration. The overall rating should be **"Direction: Strong; Form & Values: Weak."**

**What would strengthen this**: Finding experimental data on the *dynamics* of trust recovery vs. trust destruction — specifically, whether the functional form is additive/multiplicative or some other pattern (e.g., both exponential with different rate constants). Cvetkovich et al. (2002) and Poortinga & Pidgeon (2003) are the closest candidates but they measure attitude change, not dynamic update rules.

---

### 8.2 Granovetter (1978) → θ_crystal: Not a Structural Analogy

The report called the θ_crystal ↔ Granovetter threshold mapping a "structural analogy." On closer examination, the mechanisms are **fundamentally different**, not just operating at different levels. The similarities are superficial.

**Detailed mechanism comparison**:

| Dimension | Granovetter Threshold | θ_crystal |
|---|---|---|
| **Input** | Observed **proportion** of others currently acting (external snapshot) | **Cumulative sum** of confidence-gated belief differences over time (internal integral) |
| **Information basis** | Agent observes (or estimates) the fraction of others who have adopted | Agent sees only its FIFO window of w ∈ [2,6] partner actions — never knows the population fraction |
| **Temporal nature** | **Memoryless**: threshold is compared against current-tick proportion. If proportion drops, condition is no longer met. | **Cumulative**: evidence accumulates across ticks. Past evidence is never forgotten (until dissolution resets e=0). |
| **Reversibility** | **Fully reversible**: if the proportion of adopters falls below threshold, the agent reverts. | **Irreversible on formation**: once \|e\| ≥ θ_crystal, norm crystallises. Only anomaly→crisis→dissolution (a completely separate mechanism) can undo it. |
| **Uncertainty modulation** | **None**: all agents with the same threshold respond identically to the same proportion. | **(1−C) gating**: uncertain agents accumulate evidence faster, so two agents with the same θ_crystal but different C values will crystallise at different times even facing the same environment. |
| **What triggers the transition** | **External event**: proportion crosses threshold. | **Internal accumulation**: evidence reaches threshold. The environment didn't "change" at the moment of crystallisation — the agent simply accumulated enough evidence. |

**The only genuine similarity**: both involve a threshold that, once crossed, triggers a qualitative state change. But this is true of virtually every model with a phase transition — it does not constitute a meaningful structural analogy.

**A better framing**: θ_crystal is more accurately described as a **DDM decision boundary** (Ratcliff & McKoon 2008) applied to norm formation. The DDM decision boundary has its own rich literature and doesn't need the Granovetter framing. The paper would be better served by dropping the Granovetter comparison or explicitly noting it is a **surface-level analogy** to aid reader intuition, not a theoretical grounding.

---

### 8.3 O(1) Scaling: Practical Test Plan

The O(1) scaling claim (α ≈ 0.028) is tested over only 1.3 orders of magnitude (N = 1,000–20,000). This section provides a **computationally feasible** plan to strengthen or refute the claim.

**Computational feasibility analysis** (from code inspection):

| N | Memory | Per-tick (V=0) | 3000 ticks (est.) | Feasibility |
|---|--------|---------------|-------------------|-------------|
| 100 | <1 MB | <0.01 ms | <1 sec | Trivial |
| 1,000 | ~1 MB | ~0.1 ms | ~3 sec | Trivial |
| 10,000 | ~2 MB | ~1 ms | ~30 sec | Easy |
| 100,000 | ~15 MB | ~5 ms | ~15 min | **Feasible** |
| 1,000,000 | ~150 MB | ~50 ms | ~2.5 hr | **Feasible with early termination** |

The simulation is O(N) per tick (at V=0). Memory is ~100 bytes/agent. **N=100,000 is entirely practical** and gives 3 full orders of magnitude (100–100,000).

**Recommended test plan**:

```
N ∈ {100, 300, 1000, 3000, 10000, 30000, 100000}
Seeds: 30 per condition (10 at N=100000 if time-limited)
T_max: 5000 (with early convergence termination)
V = 0 (default, eliminates O(N²) sampling cost)
```

This spans **3 orders of magnitude** with logarithmically spaced points, adequate for a convincing power-law fit. Total compute time estimate: ~4 hours on a single machine.

**What to report**:
1. Log-log plot of convergence tick vs. N with confidence bands
2. Separate pre-cascade time (symmetry breaking → first crystallisation) from cascade time (first dissolution → full convergence)
3. Compare scaling exponents for each phase separately
4. The pre-cascade phase likely scales as O(1) in well-mixed populations because each agent's crystallisation is independent of N. The cascade phase should also be O(1) because dissolution operates locally at each minority-norm agent simultaneously. If the total remains O(1), the mechanism is genuine, not a topology artifact.

**Critical control**: Run the same N-sweep on a **lattice topology** (e.g., ring graph with k nearest neighbors). If O(1) holds on the lattice, the claim is about the cascade mechanism. If O(1) breaks on the lattice (reverting to O(N^γ) for some γ), the claim is about the well-mixed topology, not the cascade.

---

### 8.4 Corrected Verification Matrix

Replacing the original Part VI matrix with honest ratings:

| System | Claim | Primary Evidence | Corrected Status |
|---|---|---|---|
| **Architecture** | Three parallel systems cognitively plausible | Squire 1992, Anderson 2004, Nelson & Narens 1990 | ✅ Strong |
| **Experiential** | FIFO from decisions-from-experience | Hertwig 2010, Nevo & Erev 2012 | ✅ Strong |
| **Experiential** | Confidence-adaptive window | Behrens et al. 2007 | ✅ Strong |
| **Confidence** | Direction of asymmetry (β > α) | Slovic 1993, Poortinga & Pidgeon 2003 | ✅ Strong |
| **Confidence** | AIMD functional form | *None — borrowed from TCP* | ❌ **No support** |
| **Confidence** | α=0.1, β=0.3 values | *Calibration* | ❌ **No support** |
| **Confidence** | Bridge role via metacognitive monitoring | Nelson & Narens 1990 | ✅ Strong |
| **Confidence** | (1−C) gate from "copy when uncertain" | Rendell 2010, Boyd & Richerson 1985 | ✅ Strong |
| **DDM** | Drift rate from evidence → f_diff | Germar 2014 | ✅ Strong (core principle; see §4.1 divergences) |
| **DDM** | Majority opinion → enforcement signal | Germar 2014 | ⚠️ **Partial**: Germar's social signal is explicit & always-on; model's enforcement is conditional & disabled by default (§4.1) |
| **DDM** | Persistent bias → crystallisation | Germar & Mojzisch 2019 | ⚠️ **Partial**: continuous bias vs discrete crystallisation; different persistence mechanisms (§4.1) |
| **DDM** | θ_crystal → Granovetter analog | Granovetter 1978 | ⚠️ **Superficial analogy** (see §8.2) |
| **DDM** | θ_crystal as DDM decision boundary | Ratcliff & McKoon 2008 | ✅ Strong |
| **DDM** | No additive noise (DD-12) | Brunton et al. 2013 | ✅ Strong |
| **Enforcement** | Social pressure on DDM drift | Germar 2014 | ✅ Strong |
| **Post-crystal** | Anomaly-crisis-dissolution | Bicchieri 2006, Slovic 1993 | ⚠️ Moderate (novel mechanism) |
| **Action selection** | Probability matching | McKelvey & Palfrey 1995 (QRE) | ⚠️ **Defensible but unstated** (see §7.1) |
| **Pipeline** | Synchronous updating | *Modeling convenience* | ⚠️ **Potential artifact** (see §7.2) |
| **Scaling** | O(1) convergence time | *Model result, narrow N range* | ⚠️ **Under-tested** (see §7.4, §8.3) |
| **Parsimony** | DDM needed vs. simpler norm rules | *Ablation incomplete* | ❌ **Untested against simpler alternatives** (see §7.3) |
