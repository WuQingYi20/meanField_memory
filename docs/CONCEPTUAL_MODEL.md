# Conceptual Model: Adaptive Memory in Norm Formation

## 1. Research Motivation

### The Puzzle

Social norms emerge from repeated interactions, but individuals have **limited memory**. How does the way people remember past interactions affect:
- The **speed** at which norms emerge?
- The **stability** of established norms?
- The **flexibility** to adapt when environments change?

### The Gap in Literature

Existing models typically assume:
- Fixed memory capacity (e.g., "remember last k interactions")
- Memory independent of interaction outcomes
- No feedback between success/failure and cognitive processes

**But in reality**: When we successfully coordinate with others, we become more confident and may rely more on past experience. When we fail, we become uncertain and may discount old information.

---

## 2. Core Concept: Prediction-Error-Driven Adaptive Memory

### The Key Insight

Memory is not a passive storage system—it is **actively modulated by how well we predict our social environment**.

```
Traditional View:    Memory → Belief → Decision
                     (fixed capacity)

Our View:           Memory ⟷ Trust ⟷ Prediction Accuracy
                    (adaptive capacity)
```

### The Mechanism

When I interact with someone:
1. I **predict** what they will do (based on my memory of past interactions)
2. I **observe** what they actually do
3. I compute **prediction error** (was I right or wrong?)
4. This error updates my **confidence/anxiety** level
5. My confidence determines **how much history I rely on**

---

## 3. Agent Mental State

An agent's internal state consists of four interconnected components:

### 3.1 Memory (M)

**What it is**: A record of past social interactions

**What it stores**: Partner identities, their choices, outcomes

**Key question**: How much of this history should influence current decisions?

### 3.2 Belief (b)

**What it is**: Mental model of what others are likely to do

**Formally**: b = [P(strategy A), P(strategy B)]

**How it forms**: Weighted aggregation of memory

### 3.3 Temperature (τ) — "Anxiety Level"

**What it is**: Degree of uncertainty in decision-making

**Interpretation**:
| τ Level | Mental State | Behavioral Pattern |
|---------|--------------|-------------------|
| Low τ | Calm, confident | Consistent, predictable |
| High τ | Anxious, uncertain | Variable, exploratory |

**Key property**: τ is not fixed—it responds to prediction accuracy

### 3.4 Trust

**What it is**: Confidence that the social environment is stable and predictable

**Relationship to τ**: Trust = inverse function of anxiety
- Low anxiety (τ↓) → High trust
- High anxiety (τ↑) → Low trust

**Key role**: Trust determines how much past experience to consider

---

## 4. The Feedback Loops

### 4.1 Positive Feedback Loop (Reinforcing)

```
Coordination Success
        ↓
Prediction was Correct
        ↓
Anxiety Decreases (τ↓)
        ↓
Trust Increases
        ↓
Rely on Longer History
        ↓
More Stable Beliefs
        ↓
More Consistent Behavior
        ↓
Higher Chance of Coordination Success ←──┘
```

**Implication**: Success breeds success. Once coordination begins, it tends to strengthen.

### 4.2 Negative Feedback Loop (Balancing)

```
Coordination Failure
        ↓
Prediction was Wrong
        ↓
Anxiety Increases (τ↑)
        ↓
Trust Decreases
        ↓
Discount Old History
        ↓
Focus on Recent Events
        ↓
More Flexible/Exploratory Behavior
        ↓
Opportunity to Find New Equilibrium
```

**Implication**: Failure triggers adaptation. The system doesn't get permanently stuck.

### 4.3 The Balance

These two loops create a **self-regulating system**:
- Positive loop drives **norm crystallization**
- Negative loop enables **norm revision**

The relative strength determines system behavior:
- Strong positive loop → Fast convergence, but potentially brittle
- Strong negative loop → Slow convergence, but highly adaptive

---

## 5. Memory Types as Theoretical Constructs

We consider three idealized memory mechanisms:

### 5.1 Fixed Memory

**Assumption**: Agent remembers exactly k most recent interactions, all weighted equally

**Cognitive interpretation**: Simple rule-based recall with hard capacity limit

**Properties**:
- Deterministic forgetting (oldest drops out)
- No outcome-dependent modulation
- Baseline model for comparison

### 5.2 Decay Memory

**Assumption**: All interactions retained, but older ones weighted less

**Weight function**: w(age) = λ^age, where λ ∈ (0, 1)

**Cognitive interpretation**: Natural memory fading over time

**Properties**:
- Smooth forgetting (gradual weight reduction)
- Recent events always dominate
- Still no outcome-dependent modulation

### 5.3 Dynamic Memory (Our Innovation)

**Assumption**: Effective memory window adapts based on trust

**Mechanism**:
```
window_size = f(trust)

High trust → Large window → Consider long history
Low trust  → Small window → Focus on recent events
```

**Cognitive interpretation**: Confidence modulates reliance on experience

**Properties**:
- Outcome-dependent modulation
- Creates feedback loop with prediction accuracy
- Endogenous memory dynamics

---

## 6. The Decision Process

### 6.1 Belief Formation

Agent aggregates memory into a belief about others' behavior:

```
b = weighted_average(observed_strategies, weights_from_memory)
```

The weights depend on memory type:
- Fixed: equal weights
- Decay: exponential weights
- Dynamic: equal weights over adaptive window

### 6.2 Action Selection

Agent chooses action probabilistically based on belief and anxiety:

**Core idea**: Higher anxiety → More random choice

```
Low τ (confident)  → Choose best response to belief (exploit)
High τ (anxious)   → Choose more randomly (explore)
```

This naturally implements **exploration-exploitation tradeoff** driven by prediction accuracy.

### 6.3 Prediction

Before observing outcome, agent forms expectation:

```
prediction = most_likely_strategy_according_to_belief
```

This prediction is then compared to actual observation.

---

## 7. Learning Dynamics

### 7.1 The Update Rule

After each interaction:

```
If prediction correct:
    Anxiety decreases gradually
    → Agent becomes more confident
    → (In dynamic memory) Window expands

If prediction wrong:
    Anxiety increases sharply
    → Agent becomes more uncertain
    → (In dynamic memory) Window shrinks
```

### 7.2 Asymmetry in Learning

**Key design choice**: Confidence builds slowly, but breaks quickly

**Rationale**:
- Trust requires **consistent positive evidence** to build
- A **single surprising failure** can shake confidence

This mirrors psychological findings on trust formation and destruction.

### 7.3 Bounded Adaptation

Both anxiety and memory window have bounds:

```
τ ∈ [τ_min, τ_max]           // Cannot be infinitely confident or anxious
window ∈ [base, max_limit]   // Cognitive capacity constraints
```

The upper bound on memory (≈6) reflects human working memory limits (Miller's Law).

---

## 8. Theoretical Predictions

### Prediction 1: Dynamic Memory Accelerates Convergence

**Logic**:
- Positive feedback is stronger with dynamic memory
- Early successes quickly expand memory window
- Larger window → more stable beliefs → more consistent behavior → more success

**Expected pattern**: Dynamic memory should show fastest norm emergence

### Prediction 2: Dynamic Memory Produces More Resilient Norms

**Logic**:
- Established norms come with high trust and large memory windows
- Large windows are "inertial"—resistant to random fluctuations
- Takes sustained failure to shrink window and destabilize

**Expected pattern**: Once formed, norms under dynamic memory should be more robust

### Prediction 3: Dynamic Memory Enables Better Adaptation

**Logic**:
- When environment changes, initial failures trigger window shrinkage
- Smaller window means faster belief updating
- System can track changing conditions better than fixed memory

**Expected pattern**: Dynamic memory should re-equilibrate faster after shocks

---

## 9. Relationship to Existing Theories

### Bounded Rationality (Simon)

Our model embodies bounded rationality:
- Finite memory (not perfect recall)
- Satisficing via probabilistic choice (not optimization)
- Adaptive heuristics (memory window adjustment)

### Reinforcement Learning

The prediction error mechanism resembles:
- Temporal difference learning (prediction vs. outcome)
- Temperature-based exploration (softmax action selection)
- But learning modifies **cognitive parameters**, not just values

### Cultural Evolution

Connects to norm emergence literature:
- Young (1993): Conventions from adaptive play
- Boyd & Richerson: Cultural transmission with cognitive constraints
- Our addition: **Endogenous memory dynamics**

### Trust Dynamics

Reflects empirical findings:
- Trust builds slowly through positive interactions
- Trust breaks quickly from negative experiences
- Trust influences information processing

---

## 10. Key Conceptual Questions for Discussion

1. **Is the trust-memory link psychologically plausible?**
   - Do people actually rely more on history when confident?
   - Evidence from cognitive psychology?

2. **What determines the feedback loop strengths?**
   - How fast should confidence build vs. break?
   - Are there individual differences?

3. **What are the boundary conditions?**
   - When would fixed memory outperform dynamic?
   - What if the environment is truly unstable?

4. **How does this scale to larger groups?**
   - Mean-field assumption limitations
   - Role of network structure

5. **Normative implications?**
   - Is faster convergence always better?
   - Trade-off between stability and flexibility

---

## 11. Visual Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT i                                   │
│                                                                  │
│   ┌──────────┐      ┌──────────┐      ┌──────────────────────┐ │
│   │  Memory  │ ──→  │  Belief  │ ──→  │  Decision            │ │
│   │   M(t)   │      │   b(t)   │      │  (modulated by τ)    │ │
│   └────▲─────┘      └──────────┘      └──────────┬───────────┘ │
│        │                                          │             │
│        │    ┌─────────────────┐                  │             │
│        │    │  Trust = f(τ)   │                  │ Strategy    │
│        │    └────────▲────────┘                  ↓             │
│        │             │                                          │
│   window size   ┌────┴─────┐                                   │
│   (dynamic)     │   τ(t)   │ ←── Prediction Error              │
│                 │ Anxiety  │                                    │
│                 └──────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓ action
                    ┌─────────────────────┐
                    │   ENVIRONMENT       │
                    │   (other agents)    │
                    │                     │
                    │   Partner's choice  │
                    └─────────┬───────────┘
                              ↓
                    Compare: prediction vs. observation
                              ↓
                    Update τ (anxiety/confidence)
```

---

## 12. Summary

### The Model in One Paragraph

Agents in a coordination game form beliefs about others based on **memory of past interactions**. They make decisions with a degree of randomness controlled by their **anxiety level (τ)**. After each interaction, they compare their **prediction** against the **actual outcome**. Correct predictions reduce anxiety (building confidence), while wrong predictions increase anxiety (triggering uncertainty). Crucially, anxiety determines **trust**, which in dynamic memory modulates **how much history the agent considers**. This creates dual feedback loops: **success reinforces stability** (positive loop), while **failure enables flexibility** (negative loop). The interplay of these loops determines how quickly and how stably social norms emerge.

### The Contribution

We propose that **memory is not merely a storage constraint but an active, outcome-dependent cognitive process** that fundamentally shapes social dynamics. By making memory endogenous to the coordination process, we offer a richer account of norm formation that bridges cognitive psychology and social theory.

---

*This document describes the conceptual model. Implementation details and simulation results are documented separately.*
