# Theoretical Framework: Memory, Communication, and Norm Emergence

## 1. What is a Norm? Theoretical Definitions

### 1.1 Bicchieri (2006) - Social Norms

A **social norm** exists in population P when a sufficiently large subset:

1. **Empirical expectations**: Believes that most others follow rule R
2. **Normative expectations**: Believes that most others expect them to follow R
3. **Conditional preference**: Prefers to follow R given (1) and (2)

**Key insight**: Norms are not just behavioral regularities, but require *shared beliefs about beliefs*.

### 1.2 Lewis (1969) - Conventions

A **convention** is a behavioral regularity R where:
- Everyone conforms to R
- Everyone expects others to conform
- Everyone prefers to conform *given* others conform
- There exists an alternative R' with the same properties

**Key insight**: Conventions require *common knowledge* - I know that you know that I know...

### 1.3 Aoki (2001) - Institutions as Cognitive Constructs

An **institution** is:
> "A self-sustaining system of shared beliefs about how the game is played"

**Key insight**: Institutions are cognitive, not just behavioral. They are *shared mental models*.

### 1.4 Young (1993, 1998) - Stochastic Evolutionary Approach

Norms emerge through:
- Bounded rationality (agents sample from history)
- Stochastic best response
- Long-run selection among conventions

**Key insight**: Memory length and sampling affect which conventions are stochastically stable.

---

## 2. Our Model's Position

### 2.1 Current State

Our model captures:
- ✓ Behavioral regularity (strategy distribution)
- ✓ Individual beliefs (from memory)
- ✗ Shared beliefs (no communication)
- ✗ Normative expectations (no "ought" dimension)
- ✗ Common knowledge (only private observations)

### 2.2 The Gap

**Problem**: Without communication, each agent has only *private* beliefs based on personal experience. There is no mechanism for:
- Knowing what others believe
- Forming expectations about expectations
- Achieving common knowledge

**Consequence**: What we observe is *behavioral convergence*, not necessarily *norm emergence* in the Bicchieri/Lewis sense.

---

## 3. Proposed Extension: Communication Mechanisms

### 3.1 Theoretical Basis

| Communication Type | Theoretical Source | Real-world Analogy |
|-------------------|-------------------|-------------------|
| No communication | Baseline ABM | Isolated interactions |
| Local observation | Henrich & Boyd (1998) | Watching neighbors |
| Gossip | Dunbar (1996) | Social information sharing |
| Signaling | Skyrms (2010) | Pre-play communication |
| Global broadcast | Kuran (1995) | Media, public announcements |

### 3.2 How Communication Changes the Model

```
WITHOUT COMMUNICATION:
Agent's belief = f(personal interaction history)

WITH COMMUNICATION:
Agent's belief = f(personal history, observed behaviors, received messages)
```

This creates qualitatively different dynamics:
- **Information aggregation**: Beliefs can incorporate population-level information
- **Expectation alignment**: Agents can form beliefs about others' beliefs
- **Common knowledge emergence**: Through public signals

### 3.3 Communication Modes

**Mode 0: NONE (Current baseline)**
- Agent observes only direct interaction partner
- Information spreads through random matching
- Slowest convergence, no shared expectations

**Mode 1: OBSERVATION**
- Agent observes k random interactions per tick (not just own)
- Models "social learning" (Bandura, 1977)
- Enables Henrich & Boyd's conformist transmission

**Mode 2: GOSSIP**
- After interaction, agent tells m others about partner's strategy
- Creates indirect information flow
- Models reputation systems (Nowak & Sigmund, 1998)

**Mode 3: SIGNALING**
- Before interaction, agents exchange signals about intended play
- Can be cheap talk or costly
- Models pre-play communication (Farrell, 1987)

**Mode 4: BROADCAST**
- Central mechanism broadcasts population statistics
- All agents receive same public signal
- Models media/institutional communication (Kuran, 1995)

---

## 4. Defining Norm Emergence

### 4.1 Operational Definition

We propose a multi-level definition:

**Level 1: Behavioral Norm (Descriptive)**
```
Condition: majority_fraction > θ_behavior (e.g., 0.95)
           AND stable for t_stable ticks
```

**Level 2: Belief Norm (Cognitive)**
```
Condition: Level 1 satisfied
           AND mean(|agent_belief - true_distribution|) < θ_belief
           (agents' beliefs match reality)
```

**Level 3: Shared Expectation Norm (Social)**
```
Condition: Level 2 satisfied
           AND variance(agent_beliefs) < θ_variance
           (agents have similar beliefs)
```

**Level 4: Common Knowledge Norm (Institutional)**
```
Condition: Level 3 satisfied
           AND agents know that others know
           (requires explicit tracking)
```

### 4.2 Measurement

| Level | Metric | Interpretation |
|-------|--------|----------------|
| 1 | Majority fraction | Behavioral regularity |
| 2 | Belief accuracy | Cognitive alignment with reality |
| 3 | Belief variance | Cognitive alignment with each other |
| 4 | Meta-belief accuracy | Expectation about expectations |

---

## 5. Research Questions

With this framework, we can ask:

1. **Memory × Communication interaction**:
   - Does longer memory substitute for or complement communication?
   - How does trust-linked memory window interact with information flow?

2. **Norm levels**:
   - Can behavioral norms exist without belief norms?
   - Under what conditions do shared expectations emerge?

3. **Lock-in dynamics**:
   - Does communication accelerate or hinder norm change?
   - How does information structure affect norm stability?

4. **Pluralistic ignorance** (Kuran, 1995):
   - When do agents' beliefs diverge from reality?
   - Can private-public belief gaps persist?

---

## 6. Implementation Priorities

### Phase 1: Norm Detection (Current focus)
- Implement multi-level norm detection
- Track belief accuracy and variance
- No new communication yet

### Phase 2: Observation Mechanism
- Add local observation (k observed interactions)
- Test conformist vs. payoff-biased learning

### Phase 3: Gossip/Reputation
- Add gossip network
- Test information cascade effects

### Phase 4: Signaling
- Add pre-play signaling
- Test cheap talk vs. costly signals

---

## References

- Aoki, M. (2001). Toward a Comparative Institutional Analysis. MIT Press.
- Bicchieri, C. (2006). The Grammar of Society. Cambridge University Press.
- Dunbar, R. (1996). Grooming, Gossip, and the Evolution of Language.
- Henrich, J., & Boyd, R. (1998). The evolution of conformist transmission.
- Kuran, T. (1995). Private Truths, Public Lies. Harvard University Press.
- Lewis, D. (1969). Convention. Harvard University Press.
- Nowak, M., & Sigmund, K. (1998). Evolution of indirect reciprocity.
- Skyrms, B. (2010). Signals: Evolution, Learning, and Information. Oxford.
- Young, H.P. (1993). The evolution of conventions. Econometrica.
- Young, H.P. (1998). Individual Strategy and Social Structure. Princeton.
