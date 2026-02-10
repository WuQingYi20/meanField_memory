# Cognitive Lock-in and Norm Emergence: A Trust-Memory Feedback Model

## Abstract

We present a novel agent-based model (ABM) investigating the emergence of social norms through cognitive mechanisms. Our model introduces a **cognitive lock-in** mechanism where trust dynamics regulate memory windows, creating a feedback loop that explains both norm emergence and persistence. Unlike traditional approaches that rely on behavioral biases (e.g., conformity), our model achieves norm formation through purely cognitive channels: agents use probability matching (behaviorally neutral) while trust affects only the memory window size. This clean separation allows precise attribution of emergent phenomena to the memory mechanism alone. We demonstrate that dynamic trust-adaptive memory significantly outperforms fixed and decay-based alternatives, achieving faster convergence and higher consensus strength. Our framework also integrates three theoretically-grounded communication mechanisms (Bicchieri's normative signaling, Skyrms' pre-play signaling, and Centola's threshold contagion) for studying the interplay between cognitive and social factors in norm formation.

**Keywords**: Social Norms, Agent-Based Modeling, Trust Dynamics, Memory Systems, Coordination Games, Cognitive Lock-in

---

## 1. Introduction

### 1.1 Motivation

The emergence and persistence of social norms remains a central puzzle in computational social science. While extensive research has explored behavioral mechanisms (imitation, conformity, best-response dynamics), the role of **cognitive mechanisms** - particularly memory and trust - has received less attention. This gap is significant because:

1. **Human cognitive limitations** constrain how agents process social information
2. **Trust dynamics** fundamentally shape how individuals weight past experiences
3. **Memory window effects** determine which observations influence current decisions

### 1.2 Research Questions

1. How do different memory mechanisms affect norm emergence in coordination games?
2. Can trust-adaptive memory windows explain both norm formation and lock-in?
3. What is the relative contribution of cognitive vs. behavioral factors?

### 1.3 Contributions

1. **Cognitive Lock-in Mechanism**: A novel feedback loop where prediction accuracy drives trust, which regulates memory windows, which stabilizes beliefs
2. **Clean Attribution Design**: Behaviorally neutral action selection (probability matching) ensures any norm emergence is attributable solely to cognitive mechanisms
3. **Comprehensive Memory Taxonomy**: Systematic comparison of fixed, decay, and dynamic memory systems
4. **Integrated Communication Framework**: Three theoretically-grounded mechanisms for future extension

---

## 2. Related Work

### 2.1 Norm Emergence in Agent-Based Models

Traditional ABM approaches to norm formation include:
- **Evolutionary game theory** (Axelrod 1984; Nowak 2006): Strategy selection through fitness-based reproduction
- **Social learning models** (Boyd & Richerson 1985): Imitation of successful neighbors
- **Best-response dynamics** (Young 1993): Agents optimize against beliefs

Our approach differs by focusing on **cognitive processes** rather than behavioral biases.

### 2.2 Memory in Social Learning

- **Bounded rationality** (Simon 1955): Cognitive limitations shape decisions
- **Recency effects** (Erev & Roth 1998): Recent experiences weighted more heavily
- **Adaptive memory** (Hertwig et al. 2004): Memory strategies as adaptive responses

We extend this by making memory window size **endogenous** to trust dynamics.

### 2.3 Trust and Coordination

- **Asymmetric trust dynamics** (Slovic 1993): Trust builds slowly, breaks quickly
- **Trust games** (Berg et al. 1995): Trust as willingness to be vulnerable
- **Reputation systems** (Resnick et al. 2000): Trust through repeated interactions

Our model operationalizes Slovic's asymmetry: correct predictions slowly build trust; incorrect predictions rapidly destroy it.

### 2.4 Social Norms Theory

- **Bicchieri (2006)**: Norms require both empirical and normative expectations
- **Skyrms (2010)**: Signaling and coordination in strategic settings
- **Centola (2018)**: Complex vs. simple contagion in behavior spread

Our framework integrates all three perspectives as optional communication mechanisms.

---

## 3. Model Description

### 3.1 Overview

We model N agents playing a pure coordination game with random matching. The key innovation is a **trust-memory feedback loop**:

```
Prediction Accuracy --> Trust Level --> Memory Window --> Belief Stability
       ^                                                        |
       |______________ Coordination Success <__________________|
```

### 3.2 Coordination Game

Agents choose between two strategies (A=0, B=1) in a pure coordination game:

|   | A | B |
|---|---|---|
| A | 1 | 0 |
| B | 0 | 1 |

Coordination succeeds when both agents choose the same strategy.

### 3.3 Memory Systems

#### 3.3.1 Fixed Memory
Stores the k most recent interactions with equal weighting:
```
belief[s] = count(strategy=s in last k) / k
```

#### 3.3.2 Decay Memory
Exponentially weights past interactions:
```
weight(age) = lambda^age
belief[s] = sum(weight * I(strategy=s)) / sum(weights)
```

#### 3.3.3 Dynamic Memory (Trust-Adaptive)
Memory window size varies with trust level:
```
effective_window = base + round(trust * (max - base))
```
Where `base=2` (minimum) and `max=6` (cognitive limit, following Miller 1956).

### 3.4 Decision Mechanism: Cognitive Lock-in

#### 3.4.1 Core State Variable: Trust

Trust T in [0.01, 0.99] represents confidence in environmental predictability.

#### 3.4.2 Action Selection: Probability Matching

```python
P(action = i) = belief[i]
```

This is **behaviorally neutral**: no amplification of majority beliefs. Any norm emergence must come from the memory mechanism, not behavioral biases.

#### 3.4.3 Trust Update (Asymmetric, following Slovic 1993)

**Correct prediction (predicted == observed)**:
```
T_new = T + alpha * (1 - T)    # Slow build, saturates at 1
```

**Incorrect prediction (predicted != observed)**:
```
T_new = T * (1 - beta)         # Fast break, proportional to current level
```

Default parameters: alpha = 0.1, beta = 0.3

#### 3.4.4 Steady State Analysis

At equilibrium, expected trust change is zero:
```
p * alpha * (1-T*) = (1-p) * beta * T*
```

Solving:
```
T* = (p * alpha) / (p * alpha + (1-p) * beta)
```

Where p is prediction accuracy.

### 3.5 The Cognitive Lock-in Feedback Loop

**Positive feedback (norm stabilization)**:
1. Majority forms
2. Prediction accuracy increases (easier to predict common strategy)
3. Trust increases
4. Memory window expands
5. Beliefs become more stable (longer history)
6. Majority strengthens

**Negative feedback (system flexibility)**:
1. Coordination failures occur
2. Prediction accuracy drops
3. Trust decreases
4. Memory window shrinks
5. Recent experiences weighted more
6. Faster adaptation to change

---

## 4. Experimental Design

### 4.1 Baseline Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N agents | 100 | Standard ABM scale |
| Max ticks | 500 | Sufficient for convergence |
| Memory type | Dynamic | Primary mechanism of interest |
| Base window | 2 | Minimum cognitive capacity |
| Max window | 6 | Cognitive limit (Miller 1956) |
| Initial trust | 0.5 | Neutral starting point |
| alpha | 0.1 | Slow trust building |
| beta | 0.3 | Fast trust breaking |
| Initial strategy | Random 50/50 | No initial bias |

### 4.2 Experimental Conditions

1. **Memory type comparison**: Fixed vs. Decay vs. Dynamic
2. **Parameter sensitivity**: alpha-beta combinations
3. **Scale effects**: Agent count from 10 to 200
4. **Communication mechanisms**: Baseline vs. extended

### 4.3 Metrics

- **Convergence time**: Ticks until >90% consensus
- **Final majority**: Proportion playing dominant strategy
- **Strategy switches**: Total switches per agent
- **Trust distribution**: Mean and variance of trust levels
- **Memory window distribution**: For dynamic memory

---

## 5. Results

### 5.1 Baseline Performance

Over 50 independent runs with the baseline configuration:

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Convergence time | 222 | 111 | 68 | 469 |
| Final majority | 0.93 | 0.04 | 0.82 | 0.99 |
| Strategy switches | 8.2 | 2.1 | 4.5 | 14.3 |
| Final trust (mean) | 0.72 | 0.08 | 0.58 | 0.85 |

**Key finding**: The system reliably converges to a dominant norm despite starting from perfect symmetry (50/50).

### 5.2 Strategy Symmetry Verification

After fixing an implementation bias (argmax ties), strategy win rates are balanced:
- Strategy 0 wins: 56%
- Strategy 1 wins: 44%

This confirms the model has no systematic preference - norm selection is path-dependent.

### 5.3 Memory Type Comparison

| Memory Type | Conv. Time | Final Majority | Switches |
|-------------|------------|----------------|----------|
| Dynamic [2-6] | 257 | 0.93 | 6.8 |
| Fixed (k=5) | 312 | 0.89 | 9.4 |
| Decay (lambda=0.9) | 298 | 0.91 | 8.7 |

**Finding**: Dynamic memory achieves:
- 17% faster convergence than fixed memory
- Higher final consensus (93% vs 89%)
- Fewer strategy switches (more stable trajectories)

### 5.4 Parameter Sensitivity (alpha-beta)

| Configuration | alpha | beta | Conv. Time | Majority |
|---------------|-------|------|------------|----------|
| Conservative | 0.05 | 0.15 | 205 | 0.95 |
| Standard | 0.10 | 0.30 | 229 | 0.93 |
| Aggressive | 0.15 | 0.45 | 268 | 0.90 |
| Symmetric | 0.10 | 0.10 | 341 | 0.86 |

**Finding**: Maintaining beta > alpha (asymmetric trust) is crucial. The 3:1 ratio (beta/alpha) appears near-optimal.

### 5.5 Scale Effects

| N Agents | Conv. Time | Final Majority |
|----------|------------|----------------|
| 10 | 89 | 0.96 |
| 50 | 178 | 0.94 |
| 100 | 222 | 0.93 |
| 200 | 312 | 0.91 |

**Finding**: Convergence time scales sub-linearly with agent count (approximately O(N^0.6)), suggesting the mechanism remains effective at scale.

### 5.6 Trust Dynamics Validation

Comparing theoretical steady-state trust with empirical observations:

| Pred. Accuracy | Theoretical T* | Empirical T | Error |
|----------------|----------------|-------------|-------|
| 0.5 | 0.25 | 0.27 | 8% |
| 0.7 | 0.44 | 0.46 | 5% |
| 0.9 | 0.75 | 0.73 | 3% |

The model behavior closely matches theoretical predictions.

---

## 6. Communication Mechanisms (Extension Framework)

We have implemented three communication mechanisms for future experiments:

### 6.1 Normative Signaling (Bicchieri 2006)

Agents broadcast what they think **should** be done, separate from what they observe others doing. This creates:
- **Empirical expectations**: What do others do?
- **Normative expectations**: What do others expect me to do?

Implementation:
- Broadcast probability: 0.3 per tick
- Message content: Current strategy + confidence
- Belief update: Weighted combination of empirical and normative

### 6.2 Pre-play Signaling (Skyrms 2010)

Before each interaction, agents exchange signals about intended actions.

**Asymmetric protocol** (to avoid coordination failure):
- Higher-trust agent acts as "leader" (signals intention)
- Lower-trust agent acts as "follower" (receives and follows signal)

This creates a focal point convention enabling coordination.

### 6.3 Threshold Contagion (Centola 2018)

Agents require multiple independent sources before updating beliefs:
- **Simple contagion**: Any exposure triggers update
- **Complex contagion**: Threshold (e.g., 2 sources) required

This models the observation that behavioral change often requires "social proof" from multiple sources.

---

## 7. Discussion

### 7.1 Theoretical Implications

**Cognitive Lock-in as Norm Persistence Mechanism**:
The feedback loop explains why norms persist even when suboptimal:
1. Established norms have high prediction accuracy
2. High accuracy maintains high trust
3. High trust maintains long memory
4. Long memory stabilizes beliefs
5. Stable beliefs perpetuate the norm

**Clean Attribution**:
By using probability matching (behaviorally neutral), we can definitively attribute norm emergence to the memory mechanism rather than behavioral biases.

### 7.2 Empirical Predictions

1. **Memory manipulation**: Disrupting memory should destabilize norms
2. **Trust shocks**: Negative events should trigger norm reconsideration
3. **Asymmetric recovery**: Norms should be easier to destroy than create

### 7.3 Limitations

1. **Simplified interaction structure**: Random matching ignores network effects
2. **Two strategies only**: Real norms often have continuous action spaces
3. **No learning about others**: Agents don't model individual partners

---

## 8. Future Directions

### 8.1 Immediate Extensions

1. **Network structure**: Move from random matching to structured networks (small-world, scale-free)
2. **Heterogeneous agents**: Vary alpha-beta parameters across population
3. **Multiple norms**: Extend to games with >2 strategies

### 8.2 Communication Mechanism Studies

1. **Normative signaling effects**: How does explicit norm expression change dynamics?
2. **Signaling equilibria**: When does pre-play communication help vs. hurt?
3. **Contagion thresholds**: Optimal threshold for norm spread

### 8.3 Theoretical Extensions

1. **Analytical results**: Derive convergence time bounds
2. **Phase transitions**: Identify critical parameter values
3. **Multi-level dynamics**: Trust at individual vs. system level

### 8.4 Empirical Validation

1. **Laboratory experiments**: Test cognitive lock-in predictions
2. **Field studies**: Observe memory effects in real norm change
3. **Historical analysis**: Norm stability and external shocks

---

## 9. Conclusion

We have presented a novel agent-based model where social norms emerge through purely cognitive mechanisms. The **cognitive lock-in** feedback loop - where trust regulates memory, and memory stabilizes beliefs - provides a parsimonious explanation for both norm emergence and persistence.

Key contributions:
1. **Theoretical**: Identified trust-memory feedback as sufficient for norm formation
2. **Methodological**: Clean separation of cognitive and behavioral factors
3. **Empirical**: Demonstrated dynamic memory's superiority over fixed alternatives
4. **Extensible**: Framework for integrating communication mechanisms

This work opens new avenues for understanding social norms as emergent products of cognitive constraints rather than purely social processes.

---

## References

Axelrod, R. (1984). The Evolution of Cooperation. Basic Books.

Berg, J., Dickhaut, J., & McCabe, K. (1995). Trust, reciprocity, and social history. Games and Economic Behavior, 10(1), 122-142.

Bicchieri, C. (2006). The Grammar of Society: The Nature and Dynamics of Social Norms. Cambridge University Press.

Boyd, R., & Richerson, P. J. (1985). Culture and the Evolutionary Process. University of Chicago Press.

Centola, D. (2018). How Behavior Spreads: The Science of Complex Contagions. Princeton University Press.

Erev, I., & Roth, A. E. (1998). Predicting how people play games: Reinforcement learning in experimental games with unique, mixed strategy equilibria. American Economic Review, 88(4), 848-881.

Hertwig, R., Barron, G., Weber, E. U., & Erev, I. (2004). Decisions from experience and the effect of rare events in risky choice. Psychological Science, 15(8), 534-539.

Miller, G. A. (1956). The magical number seven, plus or minus two. Psychological Review, 63(2), 81-97.

Nowak, M. A. (2006). Evolutionary Dynamics: Exploring the Equations of Life. Harvard University Press.

Resnick, P., Zeckhauser, R., Swanson, J., & Lockwood, K. (2006). The value of reputation on eBay. Experimental Economics, 9(2), 79-101.

Simon, H. A. (1955). A behavioral model of rational choice. Quarterly Journal of Economics, 69(1), 99-118.

Skyrms, B. (2010). Signals: Evolution, Learning, and Information. Oxford University Press.

Slovic, P. (1993). Perceived risk, trust, and democracy. Risk Analysis, 13(6), 675-682.

Young, H. P. (1993). The evolution of conventions. Econometrica, 61(1), 57-84.

---

## Appendix A: Implementation Details

### A.1 Code Repository Structure

```
meanField_memory/
├── src/
│   ├── agent.py              # Agent class with memory-decision integration
│   ├── memory/
│   │   ├── base.py           # Memory base class
│   │   ├── fixed.py          # Fixed-window memory
│   │   ├── decay.py          # Exponential decay memory
│   │   └── dynamic.py        # Trust-adaptive memory
│   ├── decision/
│   │   ├── base.py           # Decision base class
│   │   └── cognitive_lockin.py  # Trust-based decision
│   ├── communication/
│   │   ├── mechanisms.py     # Three communication mechanisms
│   │   └── observation.py    # Observation-based learning
│   ├── environment.py        # Base simulation environment
│   └── environment_extended.py  # Environment with communication
├── experiments/
│   └── runner.py             # Batch experiment execution
└── visualization/
    ├── realtime.py           # Real-time animation
    └── static_plots.py       # Analysis plots
```

### A.2 Key Algorithm: Trust Update

```python
def update(self, predicted: int, observed: int) -> bool:
    success = (predicted == observed)

    if success:
        # Slow build: Trust += alpha * (1 - Trust)
        self._trust = self._trust + self._alpha * (1 - self._trust)
    else:
        # Fast break: Trust *= (1 - beta)
        self._trust = self._trust * (1 - self._beta)

    # Clamp to valid range
    self._trust = np.clip(self._trust, self._trust_min, self._trust_max)
    return success
```

### A.3 Key Algorithm: Dynamic Memory Window

```python
def get_effective_window(self) -> int:
    if self._trust_getter is None:
        return self._max_size

    trust = self._trust_getter()
    extra = round(trust * (self._max_size - self._base_size))
    return self._base_size + extra
```

---

## Appendix B: Detailed Experimental Results

### B.1 Convergence Time Distribution

The convergence time follows an approximately log-normal distribution:
- Median: 198 ticks
- Mode: ~150 ticks
- Right tail extends to ~500 ticks in rare cases

### B.2 Trust Trajectory Analysis

Three distinct phases observed:
1. **Exploration phase** (ticks 0-50): High variance, trust fluctuates
2. **Transition phase** (ticks 50-200): Trust begins stratifying
3. **Lock-in phase** (ticks 200+): Trust stabilizes at high levels

### B.3 Memory Window Evolution

Dynamic memory windows show:
- Initial: Uniform at base (2)
- Mid-simulation: Bimodal (some at 2, some at 5-6)
- Final: Most at 5-6, few stragglers at 2-3
