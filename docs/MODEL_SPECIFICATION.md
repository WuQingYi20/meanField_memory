# Model Specification: Adaptive Memory in Norm Formation

## 1. Introduction

### 1.1 Research Question

How do different memory mechanisms affect the emergence and stability of social norms in coordination games?

### 1.2 Core Innovation

We introduce a **prediction-error-driven feedback loop** that links:
- Decision-making confidence (temperature τ)
- Trust in environmental stability
- Adaptive memory window size

This creates endogenous dynamics where **successful coordination reinforces behavioral stability**, while **failed coordination promotes adaptive flexibility**.

---

## 2. Model Components

### 2.1 Game Structure

**Pure Coordination Game** with two strategies {A, B}:

|       | A     | B     |
|-------|-------|-------|
| **A** | (1,1) | (0,0) |
| **B** | (0,0) | (1,1) |

- Agents receive positive payoff **only when they coordinate** on the same strategy
- No payoff asymmetry between strategies (pure coordination, not Battle of Sexes)
- This models situations where **agreement matters more than which option is chosen**

**Examples**: Driving conventions, language standards, technology adoption, meeting time conventions

### 2.2 Population Structure

- **N agents**: Configurable from 2 to 200
- **Network**: Fully connected (mean-field approximation)
- **Matching**: Random pairing each time step (well-mixed population)
- Each tick: N/2 pairs interact simultaneously

---

## 3. Agent Architecture

Each agent i maintains the following state variables:

### 3.1 Memory System M_i(t)

Stores recent interaction history as tuples:
```
M_i(t) = {(partner_j, strategy_j, outcome, t_k) | k ∈ recent interactions}
```

**Three memory types implemented:**

| Type | Mechanism | Window Size | Weight Function |
|------|-----------|-------------|-----------------|
| **Fixed** | Sliding window | Constant k | w(age) = 1/k |
| **Decay** | Exponential decay | Soft (effective) | w(age) = λ^age |
| **Dynamic** | Trust-linked | Variable [base, max] | w(age) = 1/window |

### 3.2 Belief Formation b_i(t)

Agent's estimate of the population strategy distribution:

```
b_i(t) = [P(A), P(B)]

where P(s) = Σ w(age) × I(strategy = s) / Σ w(age)
```

- Computed from weighted memory history
- Represents agent's **mental model** of what others are likely to play
- Updated after each interaction

### 3.3 Temperature τ_i(t)

Represents the agent's **decision uncertainty** or "anxiety level":

```
τ_i(t) ∈ [τ_min, τ_max]

Default: τ_min = 0.1, τ_max = 2.0, τ_initial = 1.0
```

**Interpretation:**
- Low τ → Confident, deterministic choices (exploitation)
- High τ → Uncertain, random choices (exploration)

### 3.4 Trust Level

Derived quantity representing confidence in environment stability:

```
trust_i(t) = 1 - (τ_i(t) - τ_min) / (τ_max - τ_min)

trust ∈ [0, 1]
```

| τ value | Trust | Behavioral interpretation |
|---------|-------|---------------------------|
| τ_min (0.1) | 1.0 | Highly confident, committed |
| τ_mid (1.0) | 0.5 | Moderate uncertainty |
| τ_max (2.0) | 0.0 | Very uncertain, exploring |

---

## 4. Decision Mechanism

### 4.1 Softmax Action Selection

Agent chooses strategy using temperature-modulated softmax:

```
P(choose A) = exp(b_A / τ) / [exp(b_A / τ) + exp(b_B / τ)]
```

**Properties:**
- τ → 0: Approaches deterministic choice of argmax(b)
- τ → ∞: Approaches uniform random (0.5, 0.5)
- Smoothly interpolates between exploitation and exploration

### 4.2 Prediction Generation

Before observing outcome, agent predicts partner's strategy:

```
prediction_i = argmax(b_i)
```

This prediction is compared against actual observation to drive learning.

---

## 5. Learning Dynamics

### 5.1 Prediction Error

After interaction, agent computes prediction error:

```
ε_i = I(prediction_i ≠ observed_strategy)

ε ∈ {0, 1}  (binary: correct or incorrect)
```

### 5.2 Temperature Update Rule

**Core mechanism**: Temperature adjusts based on prediction accuracy

```python
if prediction == observed:    # Correct prediction
    τ ← max(τ_min, τ × (1 - cooling_rate))    # Multiplicative cooling

else:                         # Wrong prediction
    τ ← min(τ_max, τ + heating_penalty)       # Additive heating
```

**Default parameters:**
- cooling_rate = 0.1 (smooth decrease)
- heating_penalty = 0.3 (sharp increase)

**Rationale for asymmetry:**
- Cooling is **gradual** (multiplicative): Trust builds slowly through consistent success
- Heating is **sudden** (additive): Single failures can significantly disrupt confidence
- This mirrors psychological findings on trust formation/destruction

### 5.3 Memory Update

After each interaction:
```
M_i(t+1) = M_i(t) ∪ {new_interaction}

# For fixed/decay: oldest removed if over capacity
# For dynamic: capacity itself changes with trust
```

---

## 6. Trust-Memory Feedback Loop

### 6.1 Dynamic Memory Window

For dynamic memory type only:

```
window_i(t) = base + floor(trust_i(t) × (max - base))

Default: base = 2, max = 6
```

| Trust | Window Size | Interpretation |
|-------|-------------|----------------|
| 0.0 | 2 | Minimal memory, highly adaptive |
| 0.5 | 4 | Moderate history consideration |
| 1.0 | 6 | Maximum memory (cognitive limit) |

**Cognitive limit justification**: Miller's Law suggests humans can hold 7±2 items in working memory. We use 6 as a reasonable upper bound for social interaction tracking.

### 6.2 Complete Feedback Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    POSITIVE FEEDBACK                         │
│                                                              │
│  Coordination Success                                        │
│       ↓                                                      │
│  Prediction Correct (ε = 0)                                  │
│       ↓                                                      │
│  τ decreases (cooling)                                       │
│       ↓                                                      │
│  Trust increases                                             │
│       ↓                                                      │
│  Memory window expands (dynamic only)                        │
│       ↓                                                      │
│  More stable belief estimation                               │
│       ↓                                                      │
│  More deterministic choice (low τ)                           │
│       ↓                                                      │
│  Higher probability of coordination ←───────────────────────┘
│                                                              │
│  → REINFORCING LOOP: Success breeds success                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    NEGATIVE FEEDBACK                         │
│                                                              │
│  Coordination Failure                                        │
│       ↓                                                      │
│  Prediction Wrong (ε = 1)                                    │
│       ↓                                                      │
│  τ increases (heating)                                       │
│       ↓                                                      │
│  Trust decreases                                             │
│       ↓                                                      │
│  Memory window shrinks (dynamic only)                        │
│       ↓                                                      │
│  Focus on most recent interactions                           │
│       ↓                                                      │
│  More exploratory choice (high τ)                            │
│       ↓                                                      │
│  Ability to escape suboptimal conventions                    │
│                                                              │
│  → BALANCING LOOP: Failure enables adaptation                │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Simulation Protocol

### 7.1 Initialization

```python
for each agent i:
    strategy_i ~ Bernoulli(0.5)     # Random initial strategy
    τ_i = τ_initial                  # Moderate temperature
    M_i = ∅                          # Empty memory
    b_i = [0.5, 0.5]                 # Uniform prior belief
```

### 7.2 Each Time Step t

```python
# 1. Random matching
pairs = random_permutation(agents).reshape(N/2, 2)

# 2. Simultaneous decision
for each agent i:
    prediction_i = argmax(b_i)
    action_i ~ Softmax(b_i, τ_i)

# 3. Interaction and payoff
for each pair (i, j):
    success = (action_i == action_j)
    payoff = 1 if success else 0

# 4. Learning update
for each agent i with partner j:
    # Update memory
    M_i.add(action_j, success, t)

    # Update temperature
    if prediction_i == action_j:
        τ_i *= (1 - cooling_rate)
    else:
        τ_i += heating_penalty
    τ_i = clip(τ_i, τ_min, τ_max)

    # Update belief from memory
    b_i = compute_belief(M_i)
```

### 7.3 Convergence Criterion

```python
converged = (majority_fraction >= threshold) for (window consecutive ticks)

Default: threshold = 0.95, window = 50
```

---

## 8. Key Parameters

| Parameter | Symbol | Default | Range | Description |
|-----------|--------|---------|-------|-------------|
| Population size | N | 100 | [2, 200] | Number of agents |
| Memory size | k | 5 | [2, 10] | Fixed memory capacity |
| Decay rate | λ | 0.9 | (0, 1] | Exponential decay factor |
| Dynamic base | - | 2 | [1, 4] | Minimum memory window |
| Dynamic max | - | 6 | [4, 10] | Maximum memory window |
| Initial temperature | τ₀ | 1.0 | [τ_min, τ_max] | Starting uncertainty |
| Min temperature | τ_min | 0.1 | (0, 1) | Maximum commitment |
| Max temperature | τ_max | 2.0 | [1, 5] | Maximum exploration |
| Cooling rate | α | 0.1 | [0, 0.5] | Success learning rate |
| Heating penalty | β | 0.3 | [0, 1] | Failure learning rate |
| Convergence threshold | - | 0.95 | [0.8, 1.0] | Consensus criterion |

---

## 9. Outcome Metrics

### 9.1 Primary Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Convergence time** | First tick where threshold maintained | Speed of norm emergence |
| **Convergence rate** | Fraction of simulations that converge | Reliability of norm formation |
| **Consensus strength** | Final majority fraction | Degree of agreement |

### 9.2 Secondary Metrics

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Mean τ** | Average temperature across agents | Population uncertainty |
| **Mean trust** | Average trust level | Population confidence |
| **Strategy switches** | Per-tick strategy changes | Behavioral volatility |
| **Coordination rate** | Fraction of successful pairings | Current efficiency |

---

## 10. Theoretical Predictions

### 10.1 Hypothesis 1: Dynamic Memory Accelerates Convergence

**Mechanism**: Trust-linked memory creates stronger positive feedback
- Success → Higher trust → Longer memory → More stable beliefs → More success

**Prediction**: Dynamic memory should show faster convergence than fixed/decay

### 10.2 Hypothesis 2: Asymmetric Learning Creates Resilient Norms

**Mechanism**: Slow cooling + fast heating means:
- Norms form gradually through accumulated success
- But single shocks don't immediately destabilize (requires sustained failure)

**Prediction**: Established norms should resist transient perturbations

### 10.3 Hypothesis 3: Critical Population Size Effect

**Mechanism**: Larger populations have:
- More diverse initial conditions
- Lower probability of random coordination
- Longer expected convergence time

**Prediction**: Convergence time scales with population size, potentially non-linearly

---

## 11. Relationship to Existing Literature

### 11.1 Connections

| Literature | Connection |
|------------|------------|
| **Bounded rationality** (Simon) | Finite memory, satisficing via softmax |
| **Reinforcement learning** | Prediction error drives parameter updates |
| **Cultural evolution** | Norm emergence through repeated interaction |
| **Trust dynamics** | Slow build, fast destruction pattern |
| **Working memory limits** (Miller) | Cognitive cap on memory window |

### 11.2 Novel Contributions

1. **Endogenous memory adaptation**: Memory capacity responds to interaction outcomes
2. **Trust-temperature linkage**: Unified framework connecting confidence and exploration
3. **Dual feedback structure**: Explicit reinforcing and balancing loops

---

## 12. Implementation Notes

### 12.1 Code Structure

```
src/
├── memory/
│   ├── base.py      # Abstract memory interface
│   ├── fixed.py     # Fixed sliding window
│   ├── decay.py     # Exponential decay weights
│   └── dynamic.py   # Trust-linked adaptive
├── decision.py      # Softmax + temperature update
├── trust.py         # τ ↔ trust ↔ window mapping
├── agent.py         # Integrated agent class
├── game.py          # Coordination game logic
└── environment.py   # Simulation orchestration
```

### 12.2 Numerical Stability

- Softmax uses log-sum-exp trick to prevent overflow
- Temperature clamped to [τ_min, τ_max] after each update
- Beliefs normalized after each memory update

---

## 13. Future Extensions

1. **Network structure**: Move beyond mean-field to lattice, small-world, scale-free
2. **Strategy space**: Extend to N > 2 strategies
3. **Heterogeneous agents**: Vary learning rates, memory capacities across population
4. **Environmental shocks**: Introduce sudden payoff changes to test norm resilience
5. **Multi-level selection**: Add group competition alongside individual learning

---

## References

- Miller, G.A. (1956). The magical number seven, plus or minus two.
- Simon, H.A. (1955). A behavioral model of rational choice.
- Young, H.P. (1993). The evolution of conventions.
- Binmore, K. (2010). Game theory and the social contract.
- Camerer, C. (2003). Behavioral game theory.

---

*Document version: 1.0*
*Last updated: 2024*
