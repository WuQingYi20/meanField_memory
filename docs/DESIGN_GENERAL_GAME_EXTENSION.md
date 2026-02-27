# Design Note: Extending the Dual-Memory Model to General-Form Games

> **Status**: Design draft. Records the reasoning behind extending the
> dual-memory norm emergence model from a pure coordination game to
> general 2×2 symmetric games (with Stag Hunt as the motivating case).

---

## 1. Problem Statement

The current model has **no payoff structure**. Action selection is probability
matching: `P(A) = b_eff_A`. This implicitly assumes agents want to imitate the
majority — a pure coordination / conformity motive. The belief-to-action mapping
is an identity function.

In a general 2×2 game, actions have asymmetric consequences. The Stag Hunt
`[4,0; 3,3]` is the canonical example: Stag is payoff-dominant but risky;
Hare is risk-dominant but suboptimal. The mapping from beliefs to actions must
pass through a **payoff matrix**, and the same belief can produce different
actions depending on the game structure.

**Goal**: Introduce a payoff matrix with minimal disruption to the existing
architecture, preserving the dual-memory and normative machinery.

---

## 2. Core Insight: Only Stage 1 Changes

The entire model can be decomposed into two concerns:

| Concern | Modules | Question answered |
|---------|---------|-------------------|
| **Epistemology** | FIFO, b_exp, C, w, DDM, r, σ, enforcement | "What is happening around me?" |
| **Decision** | Stage 1 action selection | "What should I do about it?" |

The payoff matrix only enters the **decision** concern. Everything in the
epistemology concern — experience memory, confidence dynamics, normative
crystallisation, enforcement — is about estimating and responding to the
behavioral environment. None of it requires knowledge of payoffs.

```
Current:    b_eff_A  ──────────────────────→  P(A) = b_eff_A
General:    b_eff_A  ──→  EU(A), EU(B)  ──→  P(A) = softmax(λ, EU_A, EU_B)
```

---

## 3. The Certainty–Risk Nexus

### 3.1 Why risk preference need not be exogenous

In the Stag Hunt with payoffs `π = [4,0; 3,3]`:

```
EU(Stag) = b × 4 + (1−b) × 0 = 4b
EU(Hare) = b × 3 + (1−b) × 3 = 3
```

Stag is preferred when `4b > 3`, i.e., `b > 0.75`.

The belief `b = b_eff_A` is not exogenous — it is learned through the FIFO and
shaped by confidence C (via window size) and norms (via compliance blending).
The key observation:

- **Low C** → small window → volatile b_exp → b rarely sustains above 0.75
  → agent *behaves* risk-averse (chooses Hare)
- **High C** → large window → stable b_exp → if the population plays Stag,
  b stabilises above 0.75 → agent *behaves* risk-tolerant (chooses Stag)

**Risk preference is an emergent property of the agent's epistemic state, not
a personality trait.** An agent's willingness to take the risky-but-rewarding
action is a function of how certain it is about the environment. This matches
empirical findings: people take more risks in familiar environments (Weber &
Hsee 1998; Hertwig et al. 2004).

### 3.2 The softmax temperature problem

The softmax decision rule introduces a temperature parameter λ:

```
P(A) = exp(λ × EU_A) / (exp(λ × EU_A) + exp(λ × EU_B))
```

If λ is a fixed exogenous parameter, it acts as a uniform "rationality level"
across all agents and all time steps. This is unsatisfying because:

1. It introduces a free parameter with no clear empirical anchor.
2. It severs the link between certainty and decisiveness — an agent can be
   uncertain (low C) but still respond sharply to EU differences.
3. It misses the learning dynamics: agents *should* become more decisive as
   they learn.

### 3.3 The solution: λ = λ_base × C

Tying the temperature to confidence:

```julia
λ_i(t) = λ_base × C_i(t)
```

This makes C control **three** aspects of agent behaviour, all from the same
intuition ("how well do I understand my environment"):

| Aspect | Mechanism | Low C | High C |
|--------|-----------|-------|--------|
| Memory depth | w = w_base + ⌊C × (w_max − w_base)⌋ | Short window, volatile beliefs | Long window, stable beliefs |
| Norm susceptibility | drift = (1−C) × f_diff | Fast crystallisation | Slow crystallisation |
| Decision sharpness | λ = λ_base × C | Random exploration | Decisive exploitation |

**Boundary behaviour** with `λ_base × C`:
- `C → 0`: `λ → 0` → `P(A) → 0.5` (uniform random). The agent explores
  because it has no reliable information to exploit.
- `C → 1`: `λ → λ_base` → approaches best response. The agent exploits
  its well-calibrated beliefs.

**λ_base** is the only new exogenous parameter. Its semantics shift from
"risk preference" to **"cognitive capacity ceiling"** — the maximum rationality
an agent can achieve when perfectly confident. All individual variation in
decision sharpness is endogenously driven by C.

---

## 4. The Role of Norms Becomes Richer

In the pure coordination game, norms accelerate convergence — they push agents
toward the emerging majority faster. Useful but not qualitatively different from
what experience memory alone achieves.

In the Stag Hunt, norms do something **qualitatively new**: they enable agents
to cross risk thresholds.

### Example

Agent state: `b_exp_A = 0.5` (uncertain), `C = 0.5`, Stag norm with `σ = 0.8, k = 2`.

```
compliance = 0.8^2 = 0.64
b_eff_A = 0.64 × 1.0 + 0.36 × 0.5 = 0.82

EU(Stag) = 0.82 × 4 = 3.28
EU(Hare) = 3.0

→ Stag preferred (b_eff_A = 0.82 > 0.75 threshold)
```

Without the norm: `b_eff_A = 0.5 → EU(Stag) = 2.0 < 3.0 → Hare preferred`.

**The norm manufactures subjective certainty that doesn't exist in the data.**
It tells the agent: "act *as if* others will play Stag, even though your
experience is ambiguous." This is precisely the function of social norms
in real coordination problems under risk — they provide the mutual assurance
needed to sustain payoff-dominant equilibria (Bicchieri 2006, Skyrms 2004).

### The bootstrapping problem

For a Stag norm to form, agents must first observe enough Stag-playing to
drive the DDM past θ_crystal. But without the norm, rational agents in a
Stag Hunt gravitate toward Hare. This creates a chicken-and-egg problem:

1. **Without norms**: low C → random play → ~50-50 split → some agents
   happen to encounter Stag-heavy sequences → crystallise Stag norms
2. **Early norm holders**: their b_eff is pulled toward Stag → they play
   Stag more → enforcement pushes neighbours toward Stag
3. **Cascade**: if enough early adopters emerge, the population can tip
   from the Hare basin to the Stag basin

This is exactly the **norm emergence under risk** story that the model should
tell. The dual-memory system provides the mechanism: experience memory tracks
what *is*; normative memory tracks what *should be*; the gap between them is
what makes norms non-trivial.

---

## 5. Specification of Changes

### 5.1 New parameters

| Parameter | Symbol | Default | Type | Constraint | Notes |
|-----------|--------|---------|------|------------|-------|
| Payoff matrix | π | [1,0; 0,1] | Matrix{Float64} | 2×2 | Default = pure coordination (backward compatible) |
| Softmax base temperature | λ_base | 5.0 | Float64 | > 0 | Cognitive capacity ceiling. Calibrate so that at C=1, P(best_response) ≈ 0.95 for typical EU gaps. |

The `game_type` parameter is intentionally **not** added. The payoff matrix
fully specifies the game. Named presets (coordination, stag_hunt,
prisoners_dilemma) can be provided as convenience constructors, not as a
parameter.

### 5.2 Stage 1 modification

Replace probability matching with expected-utility softmax:

```julia
function select_action!(agent, params, rng)
    b = agent.b_eff_A
    π = params.payoff_matrix

    EU_A = b * π[1,1] + (1 - b) * π[1,2]
    EU_B = b * π[2,1] + (1 - b) * π[2,2]

    λ = params.lambda_base * agent.C
    p_A = exp(λ * EU_A) / (exp(λ * EU_A) + exp(λ * EU_B))

    return rand(rng) < p_A ? STRATEGY_A : STRATEGY_B
end
```

**Numerical safety**: when `λ × EU` is large, `exp()` overflows. Use the
log-sum-exp trick:

```julia
function softmax_p(λ, EU_A, EU_B)
    x_A = λ * EU_A
    x_B = λ * EU_B
    m = max(x_A, x_B)
    return exp(x_A - m) / (exp(x_A - m) + exp(x_B - m))
end
```

**Backward compatibility**: with `π = [1,0; 0,1]` and moderate `λ_base`:
- `EU_A = b, EU_B = 1−b`
- `P(A) = softmax(λ_base × C, b, 1−b)`

This is *not* identical to the current `P(A) = b` (probability matching), but
it converges to the same qualitative behaviour: when `b > 0.5`, `P(A) > 0.5`,
with sharpness controlled by `λ_base × C`. To recover exact backward
compatibility for the coordination case, a flag or special-casing could be
added, but this would complicate the code for marginal benefit.

### 5.3 Prediction (MAP) — unchanged

MAP prediction remains `argmax(b_eff)`. The prediction is about *what the
partner will do*, not *what I should do*. Payoffs don't enter.

### 5.4 Metrics additions

```julia
# New fields in TickMetrics
mean_payoff::Float64         # Mean payoff across all agents this tick
frac_payoff_dominant::Float64  # Fraction of pairs at (A,A) — the payoff-dominant equilibrium
frac_risk_dominant::Float64    # Fraction of pairs at (B,B) — the risk-dominant equilibrium
```

### 5.5 Convergence criteria — unchanged in structure

The 3-layer system (behavioral / belief / crystallisation) still applies.
Convergence to (Stag, Stag) vs (Hare, Hare) is captured by `fraction_A`
in the behavioral layer. No structural changes needed, though researchers may
want to inspect *which* equilibrium was reached.

---

## 6. What Does NOT Change

For clarity, an explicit list of modules that require **zero modification**:

- **RingBuffer / FIFO** — still stores partner strategies
- **b_exp computation** — still frequency-based from FIFO window
- **Confidence update** — still prediction-accuracy-driven (α/β)
- **Window dynamics** — still w = f(C)
- **DDM evidence accumulation** — still `drift = (1−C) × f_diff`
- **Crystallisation / dissolution** — still threshold-based on |e|
- **Anomaly / strengthening / crisis** — still observation-based
- **Enforcement** — still partner-violation-triggered, one-tick delay
- **b_eff blending** — still `compliance × b_norm + (1−compliance) × b_exp`

---

## 7. Calibration Considerations

### 7.1 λ_base

The key calibration target: at `C = 1` (maximum confidence), how close to
best-response should the agent be?

For the Stag Hunt `[4,0; 3,3]` at `b = 0.8`:
- `EU_Stag = 3.2, EU_Hare = 3.0`, gap = 0.2
- `λ_base = 5`: `P(Stag) = softmax(5, 3.2, 3.0) = exp(16)/(exp(16)+exp(15)) ≈ 0.73`
- `λ_base = 10`: `P(Stag) ≈ 0.88`
- `λ_base = 20`: `P(Stag) ≈ 0.98`

A `λ_base` in the range **5–15** produces agents that are responsive to payoff
differences but not perfectly rational. This aligns with bounded rationality
assumptions in the ABM literature (Camerer 2003).

### 7.2 Payoff normalisation

The softmax is sensitive to the *scale* of payoffs, not just their ratios.
Multiplying all payoffs by 10 is equivalent to multiplying λ_base by 10.
Recommend normalising payoffs so that the maximum payoff = 1:

```
Stag Hunt [4,0; 3,3] → normalised [1.0, 0.0; 0.75, 0.75]
```

This makes `λ_base` interpretable across different games.

---

## 8. Research Questions Enabled

1. **Can norms overcome risk dominance?**
   Under what (V, Φ) combinations does a population escape the Hare-Hare
   basin and converge to Stag-Stag?

2. **The confidence paradox in Stag Hunt.**
   High C makes agents more decisive (λ ↑) but also more resistant to norms
   (DDM drift ↓). Which effect dominates? Is there an optimal C for
   collective welfare?

3. **Norm emergence as a phase transition.**
   In the Stag Hunt, is there a critical fraction of Stag-norm holders above
   which the population tips? How does this critical mass depend on σ₀, Φ, V?

4. **Endogenous risk sensitivity across games.**
   Compare the same agent population across different payoff matrices
   (coordination, Stag Hunt, Prisoner's Dilemma). Does the dual-memory system
   produce game-appropriate behaviour without game-specific tuning?

5. **The role of enforcement in risk-dominant environments.**
   In pure coordination, enforcement accelerates convergence. In Stag Hunt,
   enforcement may be *necessary* (not just accelerating) for reaching the
   payoff-dominant equilibrium. Testable prediction: there exists a critical
   Φ below which convergence to Stag-Stag is impossible.

---

## 9. Summary

| Dimension | Current model | General game extension |
|-----------|--------------|----------------------|
| Payoff structure | None (implicit coordination) | Explicit 2×2 matrix |
| Action selection | Probability matching: P(A) = b_eff_A | EU + softmax: P(A) = softmax(λ_base × C, EU_A, EU_B) |
| Risk preference | Not applicable | Endogenous via C (certainty → risk tolerance) |
| Softmax temperature | Not applicable | Endogenous: λ = λ_base × C |
| New exogenous params | — | π (payoff matrix), λ_base (cognitive ceiling) |
| Belief system | Unchanged | Unchanged |
| Normative layer | Unchanged | Unchanged (but richer functional role) |
| Code changes | — | Stage 1 action selection + metrics |
