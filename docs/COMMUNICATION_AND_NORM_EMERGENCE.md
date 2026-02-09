# Communication, Memory, and Norm Emergence

## How Agents Think, When Norms Appear, and How Communication Integrates with Memory

---

## 1. The Model's Core Question

Starting from a **cold start** (50-50 random strategy distribution, no shared history), how does randomness become a **social norm** — not just behavioral convergence, but a self-enforcing pattern sustained by mutual expectations?

The answer lies in **feedback loops modulated by trust**, where communication channels amplify or dampen the dynamics.

---

## 2. How an Agent "Thinks": Decision Architecture

### 2.1 Information Sources

Each agent has access to four types of information, ordered by directness:

| Source | What It Tells You | Persistence | Trust in Your Model |
|--------|-------------------|-------------|---------------------|
| **Direct memory** | What happened TO ME | Stored in Memory (window-limited) | High weight when trust high |
| **Observations** | What I SAW others do | Currently ephemeral (cleared each tick) | Weight increases when trust low |
| **Normative messages** | What others think I SHOULD do | Separate expectation array | Weight increases when trust low |
| **Pre-play signals** | What my partner WILL do (this round) | Single-use | Immediate coordination device |

### 2.2 The Integration Problem

Currently, `choose_action()` only uses raw memory:

```python
belief = self._memory.get_strategy_distribution()  # Only direct experience!
```

But a theoretically grounded agent should combine all sources. The literature converges on a key principle:

> **"Copy when uncertain"** — The winning strategy in the social learning strategies tournament (Rendell et al. 2010, *Science*). Agents should use social information when individual learning is unreliable, and rely on personal experience when confident.

In your model, **trust IS the reliability index**. This gives us the integration rule:

### 2.3 Trust-Modulated Belief Integration

**Theoretical basis**:
- Rendell et al. (2010): Copy when uncertain (trust = confidence in own learning)
- Toelch & Dolan (2015): Informational and normative influences are parallel pathways
- Behrens et al. (2007, 2008): Learning rate adjusts to environmental volatility (trust tracks volatility)

```
EMPIRICAL PATHWAY (what others DO):
    direct_belief = Memory.get_strategy_distribution()
    obs_belief = aggregate(observations)

    memory_weight = 0.5 + 0.4 × trust     # [0.5, 0.9]
    obs_weight = 1 - memory_weight          # [0.1, 0.5]

    empirical_belief = memory_weight × direct_belief + obs_weight × obs_belief

NORMATIVE PATHWAY (what others think I SHOULD do):
    normative_belief = normative_expectation
    consensus = max(normative_belief)

    # Low trust → seek social guidance; strong consensus → stronger influence
    normative_weight = normative_base_weight × (1 - trust) × consensus

COMBINED BELIEF:
    combined = (1 - normative_weight) × empirical_belief + normative_weight × normative_belief

SIGNAL OVERRIDE (immediate coordination):
    if signal received:
        signal_weight = confidence × (1 - trust × 0.5)
        final_belief = (1 - signal_weight) × combined + signal_weight × signal_belief
    else:
        final_belief = combined
```

**Why trust modulates**: This implements Behrens et al.'s (2007) volatility-adjusted learning rate. In their Bayesian framework, the optimal learning rate increases with environmental volatility. In your model:
- **High trust** = "environment is stable" = low volatility → slow learning rate → rely on memory (long window, high memory weight)
- **Low trust** = "environment is volatile" = high volatility → fast learning rate → rely on recent social information (short window, high observation weight)

### 2.4 How Communication Feeds Back Into Trust

Communication doesn't just provide information — it **modifies the feedback loops**:

```
TRUST UPDATE (extended with communication feedback):

    base_delta = α if prediction_correct else -β      # Original individual feedback

    # Observation feedback: social alignment amplifies trust dynamics
    if observations available:
        obs_alignment = fraction_of_observations_matching_prediction × 2 - 1  # [-1, 1]
        obs_multiplier = 1 + γ_obs × obs_alignment    # Amplification factor
        base_delta *= obs_multiplier

    # Normative feedback: deviating from norm costs trust
    if normative_consensus > threshold:
        if my_strategy ≠ norm_strategy:
            norm_cost = δ × consensus²                 # Nonlinear threshold
            base_delta -= norm_cost

    trust_new = clip(trust + base_delta, 0, 1)
```

**Parameters**:
- `γ_obs` (observation amplification): How much social alignment amplifies trust dynamics (default: 0.3)
- `δ` (normative pressure): Cost of deviating from established norms (default: 0.2)
- `consensus²`: Nonlinear threshold — weak norms have little effect, strong norms have powerful effect

---

## 3. When Does a Norm Appear?

### 3.1 Theoretical Framework: Bicchieri (2006)

A **social norm** requires three conditions (Bicchieri 2006, *The Grammar of Society*):

1. **Empirical expectations (E)**: Agent believes a sufficiently large subset of their reference network conforms to the behavior
2. **Normative expectations (N)**: Agent believes a sufficiently large subset of their reference network believes they *ought to* conform
3. **Conditional preference (CP)**: Agent prefers to conform *conditional on* both (E) and (N) being satisfied

**Key distinction**:
- **Convention** (Young 1993): Only requires (E) — "I do X because others do X"
- **Social norm** (Bicchieri 2006): Requires (E) + (N) + (CP) — "I do X because others do X AND others expect me to do X AND I prefer to do X given these expectations"

### 3.2 Detection Levels in Your Model

Mapping Bicchieri's framework to measurable quantities:

| Level | Name | Condition | Measurement | Threshold |
|-------|------|-----------|-------------|-----------|
| 0 | **NONE** | No regularity | `majority_fraction` | < 0.70 |
| 1 | **BEHAVIORAL** | Behavioral regularity | `majority_fraction ≥ threshold` for `stability_window` ticks | ≥ 0.95 for 50 ticks |
| 2 | **EMPIRICAL** | + Accurate empirical expectations | `mean_belief_error < ε` | < 0.10 |
| 3 | **SHARED** | + Common knowledge of behavior | `belief_variance < σ²` | < 0.05 |
| 4 | **NORMATIVE** | + Normative alignment | `normative_alignment ≥ threshold` | ≥ 0.80 |
| 5 | **INSTITUTIONAL** | + Temporal stability | Maintained for extended period | 200+ ticks |

**Level 4 (NORMATIVE) is the critical threshold** — this is where a convention becomes a norm.

### 3.3 The Emergence Path from Cold Start

Based on the feedback loop dynamics, norm emergence follows a characteristic trajectory:

#### Phase 1: Random Drift (ticks 0–30)
- **State**: 50-50 distribution, no consensus
- **Trust**: Declining from 0.5 (predictions are random → ~50% accuracy → trust falls)
- **Memory**: Shrinking windows (low trust → short memory)
- **Communication**: Conflicting observations and messages, no consensus
- **Feedback**: Negative loop dominates (failure → shorter memory → more exploration)
- **Norm level**: NONE

#### Phase 2: Symmetry Breaking (ticks 30–80)
- **State**: Random drift creates small majority (55-65%)
- **Trust**: Stabilizing (majority agents predict better → accuracy improves slightly)
- **Memory**: Beginning to differentiate (majority agents: trust rises → windows grow)
- **Communication**: Observations start showing bias toward majority
- **Feedback**: **Tipping point** — positive loop starts competing with negative loop
- **Norm level**: NONE → approaching BEHAVIORAL

**Critical moment**: Centola et al. (2018, *Science*) showed that when a committed minority reaches ~25% of the group, it can tip the entire population. In your model, this maps to the moment where one strategy's positive feedback becomes self-sustaining.

#### Phase 3: Cascade (ticks 80–150)
- **State**: Rapid convergence (65% → 90%+)
- **Trust**: Rising sharply for majority agents
- **Memory**: Majority agents have long windows (locked in), minority agents have short windows (adaptive → switch faster)
- **Communication**:
  - Observations strongly favor majority → amplifies trust dynamics
  - Normative messages increasingly unified → creates social pressure on minority
- **Feedback**: **Positive feedback dominates** — success → trust↑ → memory↑ → beliefs stabilize → more success
- **Norm level**: BEHAVIORAL achieved (95%+ adoption)

#### Phase 4: Belief Alignment (ticks 150–250)
- **State**: 95%+ behavioral convergence
- **Trust**: High and stable
- **Memory**: Long windows, stable beliefs
- **Communication**:
  - Observations uniformly confirm majority strategy
  - Normative messages reach consensus
- **Feedback**: Both loops positive, self-reinforcing
- **Norm level**: BEHAVIORAL → EMPIRICAL → SHARED (beliefs accurate and aligned)

#### Phase 5: Norm Crystallization (ticks 250+)
- **State**: Behavioral + belief + normative convergence
- **Trust**: Near maximum
- **Memory**: Maximum windows
- **Communication**: Strong normative consensus, agents express commitment
- **Feedback**: Locked in — perturbations are quickly corrected
- **Norm level**: NORMATIVE → INSTITUTIONAL (self-sustaining)

### 3.4 Normative Alignment Detection

To detect Level 4 (NORMATIVE), measure whether agents have developed normative expectations:

```python
def compute_normative_alignment(normative_expectations, majority_strategy):
    """
    Fraction of agents whose normative expectation aligns with majority.

    A NORM exists when most agents believe most agents think
    they SHOULD follow the majority strategy.
    """
    aligned = sum(
        1 for norm_exp in normative_expectations
        if norm_exp[majority_strategy] > 0.6
    )
    return aligned / len(normative_expectations)
```

---

## 4. The Three Feedback Loops

### Loop 1: Individual Learning (Original Model)

```
Prediction correct → Trust += α(1-T)   → Memory window↑ → Beliefs stabilize → Better predictions
Prediction wrong  → Trust *= (1-β)     → Memory window↓ → Beliefs flexible  → Try alternatives
```

**Steady state** (from cognitive_lockin.py):
```
T* = pα / (pα + (1-p)β)
```
where p = prediction accuracy. At p=0.5: T*=0.25. At p=0.9: T*=0.75.

**References**: Slovic (1993) for asymmetric trust (α < β); Behrens et al. (2007) for volatility-adjusted learning rate.

### Loop 2: Social Amplification (Observations)

```
Agent succeeds → Visible to observers → Others shift beliefs toward agent's strategy
→ More partners match → Agent succeeds MORE → Trust rises FASTER
```

**Effect**: Amplifies BOTH positive and negative feedback.
- Aligned with observations → trust dynamics multiplied by (1 + γ_obs)
- Misaligned → trust dynamics multiplied by (1 - γ_obs)

**Reference**: Toyokawa et al. (2019) — conformist social learning model where action probability combines individual Q-values with social frequency:
```
P(action) = (1-σ) × softmax(Q) + σ × f(action)^θ / Σf(j)^θ
```
In your model, σ (copying weight) maps to `(1 - trust)` and θ (conformity exponent) > 1 creates nonlinear majority amplification.

### Loop 3: Normative Pressure (Signaling)

```
Majority forms → Agents broadcast "we SHOULD do X" → Consensus grows
→ Minority faces normative pressure → Switches faster → Majority strengthens
→ Even stronger normative consensus → LOCK-IN
```

**Effect**: Creates additional positive feedback with **threshold dynamics**.
- Below consensus threshold: Little effect (weak norms don't propagate)
- Above consensus threshold: Powerful cascade (strong norms are self-enforcing)
- Uses consensus² to create nonlinear threshold (Centola et al. 2018: ~25% tipping point)

**Reference**: Bicchieri (2006) — norms require normative expectations, not just behavioral regularity. Without this loop, your model produces conventions, not norms.

---

## 5. How Communication Integrates with Memory

### 5.1 Current Architecture and Its Limitation

**Current**: Observations are ephemeral (cleared each tick), never enter Memory. This means:
- Social information doesn't accumulate
- No learning about reliability of social sources
- Memory only reflects personal experience

### 5.2 Proposed Integration: Communication as Trust Feedback Modulator

Rather than storing communication in memory, **communication modifies the trust dynamics that control memory**:

```
DIRECT EXPERIENCE → Memory → Beliefs (empirical, what works)
                           ↕
TRUST ← prediction accuracy × observation alignment × normative conformity
  ↓
MEMORY WINDOW = base + trust × (max - base)
  ↓
How much of direct experience you remember

COMMUNICATION → Modifies trust update → Indirectly controls memory window
```

**Why this design**:
1. **Preserves clean memory**: Memory stores only what you experienced directly
2. **Communication acts through trust**: Social information modulates HOW you learn, not WHAT you remember
3. **Matches neuroscience**: Behrens et al. (2008) showed that social and reward information produce separate prediction errors in ACC, but both modulate the same learning rate
4. **Theoretically principled**: Toelch & Dolan (2015) show informational and normative influences are parallel pathways that converge at decision time

### 5.3 The Feedback Loop Integration

```
┌─────────────────────────────────────────────────────┐
│                    TRUST                             │
│   Updated by: prediction error × social alignment   │
│              × normative conformity                  │
└──────────┬──────────────────────────┬───────────────┘
           │                          │
           ▼                          ▼
    MEMORY WINDOW              INFORMATION WEIGHTING
    base + T×(max-base)        memory_w = 0.5 + 0.4×T
           │                   obs_w = 0.5 - 0.4×T
           ▼                          │
    DIRECT EXPERIENCE                 ▼
    (last W interactions)      INTEGRATED BELIEF
           │                   (memory + obs + norms)
           ▼                          │
    MEMORY BELIEF                     ▼
           │                      DECISION
           └──────────────────►  choose action
                                      │
                                      ▼
                               INTERACTION OUTCOME
                                      │
                      ┌───────────────┼──────────────┐
                      ▼               ▼              ▼
               Direct Memory    Trust Update    Observable
               (add interaction) (pred error     (others see
                                 + social)       your action)
```

---

## 6. Theoretical Justification Summary

### Trust as Integration Mechanism

| Literature | Key Finding | Mapping to Your Model |
|-----------|-------------|----------------------|
| **Rendell et al. (2010)** *Science* | "Copy when uncertain" wins tournament | Trust = confidence → low trust = copy more |
| **Slovic (1993)** *Risk Analysis* | Trust is hard to build, easy to break | α < β asymmetry in trust update |
| **Behrens et al. (2007)** *Nature Neuroscience* | Volatility modulates learning rate via ACC | Trust = inverse volatility → controls memory window |
| **Behrens et al. (2008)** *Nature* | Separate prediction errors for reward and social value | Prediction error + social alignment both update trust |
| **Toelch & Dolan (2015)** *Trends Cogn Sci* | Informational vs normative are parallel pathways | Empirical pathway (memory+obs) parallel to normative pathway |
| **Toyokawa et al. (2019)** *eLife* | Conformist copying: P = (1-σ)softmax(Q) + σf^θ | σ maps to (1-trust); θ > 1 creates conformist bias |
| **Bicchieri (2006)** *Grammar of Society* | Norm = empirical + normative expectations + conditional preference | Norm detection requires behavioral + belief + normative alignment |
| **Centola et al. (2018)** *Science* | 25% committed minority tips social conventions | Nonlinear consensus threshold (consensus²) in normative pressure |

### Communication as Feedback Modulator

| Mechanism | Feedback Role | Parameter |
|-----------|---------------|-----------|
| **Observations** | Amplifies both positive and negative feedback | γ_obs = 0.3 |
| **Normative signals** | Creates additional positive feedback with threshold | δ = 0.2, consensus² |
| **Pre-play signals** | Immediate coordination (bypass feedback loops) | signal_confidence |

---

## 7. Testable Predictions

### H1: Communication accelerates convergence
- **Mechanism**: Observation amplifies positive feedback loop
- **Test**: Compare convergence time with/without observations
- **Expected**: Convergence time decreases with observation_k

### H2: Normative signaling creates true norms (not just conventions)
- **Mechanism**: Normative messages create Level 4 (NORMATIVE) alignment
- **Test**: Compare norm levels with/without normative signaling
- **Expected**: Without normative → stuck at SHARED; with normative → reaches NORMATIVE

### H3: Normative convergence can LEAD behavioral convergence
- **Mechanism**: Normative pressure drives minority to switch before they would otherwise
- **Test**: Track normative alignment vs behavioral convergence timing
- **Expected**: Normative alignment above 80% before behavioral convergence in some runs

### H4: 25% tipping point for norm cascade
- **Mechanism**: Nonlinear consensus threshold (Centola 2018)
- **Test**: Vary initial bias (55-45, 60-40, 70-30, 75-25)
- **Expected**: Sharp transition in convergence speed around 75-25 split

### H5: Low-trust agents are norm followers, high-trust agents are norm setters
- **Mechanism**: Trust modulates weight of social vs personal information
- **Test**: Track which agents broadcast normative messages vs follow them
- **Expected**: High-trust agents broadcast more; low-trust agents conform more

### H6: Dynamic memory + communication > either alone
- **Mechanism**: Dynamic memory provides positive feedback; communication amplifies it
- **Test**: 2×2 comparison (dynamic/fixed × communication/no communication)
- **Expected**: Dynamic + communication converges fastest and reaches highest norm level

---

## 8. Key Parameters

| Parameter | Symbol | Default | Role | Source |
|-----------|--------|---------|------|--------|
| Trust increase rate | α | 0.1 | Slow trust building | Slovic (1993) |
| Trust decrease rate | β | 0.3 | Fast trust breaking | Slovic (1993) |
| Memory base | base | 2 | Min window at trust=0 | Miller (1956) |
| Memory max | max | 6 | Max window at trust=1 | Miller (1956) |
| Observation amplification | γ_obs | 0.3 | Social feedback strength | New parameter |
| Normative pressure | δ | 0.2 | Cost of norm deviation | New parameter |
| Normative base weight | w_norm | 0.3 | Baseline normative influence | Bicchieri (2006) |
| Consensus exponent | exp | 2 | Threshold nonlinearity | Centola (2018) |
| Observation weight | w_obs | 0.5 | Reliability of observations | Toyokawa (2019) |

---

## References

- Behrens, T. E., Woolrich, M. W., Walton, M. E., & Rushworth, M. F. (2007). Learning the value of information in an uncertain world. *Nature Neuroscience*, 10(9), 1214-1221.
- Behrens, T. E., Hunt, L. T., Woolrich, M. W., & Rushworth, M. F. (2008). Associative learning of social value. *Nature*, 456(7219), 245-249.
- Bicchieri, C. (2006). *The Grammar of Society: The Nature and Dynamics of Social Norms*. Cambridge University Press.
- Centola, D., Becker, J., Brackbill, D., & Baronchelli, A. (2018). Experimental evidence for tipping points in social convention. *Science*, 360(6393), 1116-1119.
- Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.
- Rendell, L., Boyd, R., Cownden, D., et al. (2010). Why copy others? Insights from the social learning strategies tournament. *Science*, 328(5975), 208-213.
- Skyrms, B. (2010). *Signals: Evolution, Learning, and Information*. Oxford University Press.
- Slovic, P. (1993). Perceived risk, trust, and democracy. *Risk Analysis*, 13(6), 675-682.
- Toelch, U., & Dolan, R. J. (2015). Informational and normative influences in conformity from a neurocomputational perspective. *Trends in Cognitive Sciences*, 19(10), 579-589.
- Toyokawa, W., Whalen, A., & Laland, K. N. (2019). Conformist social learning leads to self-organised prevention against adverse bias in risky decision making. *eLife*, 8, e75308.
- Young, H. P. (1993). The evolution of conventions. *Econometrica*, 61(1), 57-84.
