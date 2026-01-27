# Model Summary: Adaptive Memory in Norm Formation

## One-Sentence Summary

An agent-based model where **prediction accuracy drives trust**, which in turn **modulates memory capacity**, creating a feedback loop that affects **norm emergence speed and stability**.

---

## The Core Idea

```
Traditional models:  Memory → Belief → Action → Outcome
                     (fixed)

Our model:          Memory ⟷ Belief → Action → Outcome → Prediction Error
                       ↑                                        ↓
                       └──────── Trust ←──── Temperature ←──────┘
```

**Key insight**: Memory is not fixed—it adapts based on how well the agent predicts its environment.

---

## Agent State Variables

| Variable | Range | Meaning |
|----------|-------|---------|
| **Memory M** | list of interactions | What I remember about others |
| **Belief b** | [P(A), P(B)] | What I think others will do |
| **Temperature τ** | [0.1, 2.0] | My uncertainty/anxiety level |
| **Trust** | [0, 1] | My confidence in environment stability |
| **Strategy s** | {A, B} | What I choose to do |

---

## Key Equations

### Decision: Softmax Choice
```
P(choose A) = exp(b_A / τ) / Σ exp(b_i / τ)
```

### Learning: Temperature Update
```
Correct prediction:  τ_new = τ × (1 - 0.1)     // gradual cooling
Wrong prediction:    τ_new = τ + 0.3           // sudden heating
```

### Trust-Memory Link (Dynamic Memory Only)
```
trust = 1 - (τ - τ_min) / (τ_max - τ_min)
window = 2 + floor(trust × 4)                  // range [2, 6]
```

---

## The Two Feedback Loops

### Loop 1: Positive Feedback (Reinforcing)

```
Success → Correct prediction → τ↓ → Trust↑ → Longer memory → Stable beliefs → More success
```

**Effect**: Once coordination starts working, it becomes **self-reinforcing**.

### Loop 2: Negative Feedback (Balancing)

```
Failure → Wrong prediction → τ↑ → Trust↓ → Shorter memory → Adaptive beliefs → Try new things
```

**Effect**: When stuck in bad equilibrium, system **becomes flexible** to escape.

---

## Memory Type Comparison

| Type | Window | Weights | Trust Effect |
|------|--------|---------|--------------|
| **Fixed** | k (constant) | Equal | None |
| **Decay** | Soft (λ-dependent) | Exponential | None |
| **Dynamic** | [2, 6] (variable) | Equal | Window size adapts |

**Hypothesis**: Dynamic memory converges faster due to stronger positive feedback.

---

## Experimental Results Preview

| Memory Type | Mean Convergence Tick | Convergence Rate |
|-------------|----------------------|------------------|
| Fixed | 143.7 | 95% |
| Decay | 91.6 | 100% |
| **Dynamic** | **64.0** | **100%** |

Dynamic memory shows **2.2× faster convergence** than fixed memory.

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `cooling_rate` | 0.1 | How fast trust builds |
| `heating_penalty` | 0.3 | How fast trust breaks |
| `τ_min` | 0.1 | Maximum commitment level |
| `τ_max` | 2.0 | Maximum exploration level |
| `dynamic_base` | 2 | Minimum memory when trust=0 |
| `dynamic_max` | 6 | Maximum memory when trust=1 |

---

## Research Questions

1. **Speed**: Does adaptive memory accelerate norm convergence?
2. **Stability**: Are norms more resilient with dynamic memory?
3. **Scaling**: How does convergence time scale with population size?
4. **Robustness**: How do different cooling/heating rates affect outcomes?

---

## Theoretical Contributions

1. **Endogenous memory**: Memory capacity is outcome-dependent, not fixed
2. **Unified framework**: Temperature links decision-making, trust, and memory
3. **Dual dynamics**: Explicit reinforcing loop (stability) + balancing loop (flexibility)

---

## File Locations

- Conceptual diagram: `data/conceptual_model_v2.png`
- Full specification: `docs/MODEL_SPECIFICATION.md`
- Source code: `src/`
- Example results: `data/feedback_loop_demo.png`
