# Implementation Specification v5.1: Dual-Memory Norm Emergence Model

> **Purpose**: This document specifies every data structure, algorithm, timing
> detail, and edge case needed to implement the dual-memory model described in
> `conceptual_model_v5.tex`. A developer should be able to implement the
> complete simulation from this spec alone.
>
> **Status**: Authoritative. Where code deviates from this spec, the spec is correct.

---

## 0. Resolved Design Decisions

| ID | Question | Decision | Rationale |
|----|----------|----------|-----------|
| DD-1 | Does experience FIFO store V additional observations? | **No.** FIFO stores only the direct partner's strategy (1 entry/tick). V observations feed normative memory only. | Preserves dynamic window semantics: window of 2–6 means 2–6 ticks of history, not 2–6 observations. |
| DD-2 | σ\_min (norm dissolution threshold)? | **0.1** | σ=0.8 → 1 crisis → 0.24 (survives) → 2nd crisis → 0.072 < 0.1 → dissolves. Requires 2 crises. |
| DD-3 | Post-crystallisation observation processing order? | **Batch**: count all violations/conformities first, apply strengthening and anomaly, check crisis once at end. | Consistent with synchronous pipeline ("all outcomes collected before updates"). |
| DD-4 | What does each V observation contain? | One randomly selected participant's strategy from a random interaction (not involving the observer). | Matches \|O\_i(t)\| = 1 + V. |
| DD-5 | Enforcement signal timing? | **One-tick delay.** Written in Stage 5 of tick t, consumed in Stage 4 of tick t+1. | Pipeline order: Stage 4 (DDM) before Stage 5 (enforcement). |
| DD-6 | What triggers enforcement? | **Partner violation only.** V additional observations never trigger enforcement (only anomaly). | Eq. `violation_response` specifies "partner s\_j ≠ r\_i". |
| DD-7 | Does enforcement replace anomaly? | **Yes**, for the partner observation only. If enforcement is triggered, that partner violation is NOT counted as anomaly. | Eq. `violation_response`: enforce OR accumulate, not both. |
| DD-8 | Signal push computation timing? | `pending_signal` stores the enforced **strategy direction** only. Magnitude (including receiver's C\_j) is computed at **consumption time** (Stage 4 of next tick). | Clean separation: enforcer sends intent; receiver's susceptibility uses current state. |
| DD-9 | FIFO capacity vs. window? | FIFO capacity = w\_max (always). Belief computation uses last w\_i entries (w\_i ≤ w\_max). Shrinking window doesn't delete old entries. | Entries can become "visible" again if confidence recovers. |
| DD-10 | MAP prediction tie-breaking? | **Random uniform** when b\_A = b\_B. | Avoids systematic bias at 50-50. |

---

## 1. Parameters

### 1.1 Experience Layer

| Parameter | Symbol | Default | Type | Constraint | Source |
|-----------|--------|---------|------|------------|--------|
| Number of agents | N | 100 | int | even, ≥ 2 | — |
| Number of ticks | T | 1000 | int | ≥ 1 | — |
| Random seed | seed | None | int? | — | For reproducibility |
| Confidence increase | α | 0.1 | float | (0, 1) | Slovic 1993 |
| Confidence decrease | β | 0.3 | float | (0, 1), β > α | Slovic 1993 |
| Initial confidence | C₀ | 0.5 | float | [0, 1] | — |
| Memory base window | w\_base | 2 | int | ≥ 1 | Hertwig 2010 |
| Memory max window | w\_max | 6 | int | ≥ w\_base | Nevo & Erev 2012 |

### 1.2 Normative Layer

| Parameter | Symbol | Default | Type | Constraint | Source |
|-----------|--------|---------|------|------------|--------|
| DDM noise (std dev) | σ\_noise | 0.1 | float | ≥ 0 | Germar 2014 |
| Crystallisation threshold | θ\_crystal | 3.0 | float | > 0 | calibration |
| Initial norm strength | σ₀ | 0.8 | float | (0, 1] | calibration |
| Crisis threshold | θ\_crisis | 10 | int | ≥ 1 | calibration |
| Crisis decay | λ\_crisis | 0.3 | float | (0, 1) | calibration |
| Dissolution threshold | σ\_min | 0.1 | float | (0, σ₀) | DD-2 |
| Strengthen rate | α\_σ | 0.005 | float | (0, 1) | Will 2023 |
| Enforce threshold | θ\_enforce | 0.7 | float | (0, 1) | Toribio 2023 |
| Compliance exponent | k | 2.0 | float | > 0 | calibration |
| Signal amplification | γ\_signal | 2.0 | float | > 0 | calibration |

### 1.3 Environment

| Parameter | Symbol | Default | Type | Constraint | Notes |
|-----------|--------|---------|------|------------|-------|
| Visibility | V | 0 | int | ≥ 0 | Additional observations/tick. Default 0 = isolated learning. |
| Social pressure | Φ | 0.0 | float | ≥ 0 | Enforcement gain. 0 = disabled, 1 = baseline. |

### 1.4 Convergence

| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| convergence\_threshold | 0.95 | float | Fraction for majority |
| convergence\_window | 50 | int | Ticks maintaining threshold |

---

## 2. Data Structures

### 2.1 Strategy Encoding

```
Strategy = int   # 0 = A, 1 = B
```

### 2.2 AgentState

```python
class AgentState:
    # ── Experience memory ──
    fifo: Deque[Strategy]     # maxlen = w_max; stores partner strategies only
    b_exp: float[2]           # [b_A, b_B]; sums to 1.0; default [0.5, 0.5]

    # ── Confidence ──
    C: float                  # ∈ [0, 1]; init = C₀
    w: int                    # Current window = f(C); init = w_base + floor(C₀ * (w_max - w_base))

    # ── Normative memory ──
    r: Strategy | None        # Norm rule; None = no norm
    sigma: float              # Norm strength ∈ [0, 1]; 0.0 when no norm
    a: int                    # Anomaly counter; 0 when no norm
    e: float                  # DDM evidence accumulator; 0.0 initial

    # ── Enforcement buffer ──
    pending_signal: Strategy | None   # Enforced strategy from prev tick, or None

    # ── Derived (recomputed in Stage 1) ──
    compliance: float         # sigma^k; 0.0 if no norm
    b_eff: float[2]           # Effective belief after normative constraint
```

### 2.3 InteractionRecord

```python
class InteractionRecord:
    i: int                    # Agent index
    j: int                    # Partner index
    action_i: Strategy        # i's chosen strategy
    action_j: Strategy        # j's chosen strategy
    pred_i: Strategy          # i's MAP prediction of j
    pred_j: Strategy          # j's MAP prediction of i
    coordinated: bool         # action_i == action_j
```

### 2.4 TickState (transient, per-tick working data)

```python
class TickState:
    pairs: list[tuple[int, int]]          # N/2 pairs for this tick
    interactions: list[InteractionRecord]  # Outcomes
    observations: dict[int, list[Strategy]]  # agent_id → O_i(t), len = 1 + V
    enforcement_intents: dict[int, int]    # enforcer_id → partner_id
```

### 2.5 TickMetrics

```python
class TickMetrics:
    tick: int
    fraction_A: float                 # Fraction of agents who played A
    mean_confidence: float            # Mean C across all agents
    coordination_rate: float          # Fraction of pairs that coordinated
    num_crystallised: int             # Agents with r ≠ None
    mean_norm_strength: float         # Mean σ among crystallised agents (0 if none)
    num_enforcements: int             # Enforcement events this tick
    norm_level: int                   # 0–5 (Section 5)
    belief_error: float               # Mean |b_A_eff - fraction_A| across agents
    belief_variance: float            # Var(b_A_eff) across agents
    convergence_counter: int          # Consecutive ticks at ≥ convergence_threshold
```

---

## 3. Initialization

```python
def initialize(params) -> list[AgentState]:
    rng = RandomState(params.seed)
    agents = []
    for i in range(params.N):
        a = AgentState()
        a.fifo = Deque(maxlen=params.w_max)      # empty
        a.b_exp = [0.5, 0.5]
        a.C = params.C0
        a.w = params.w_base + floor(params.C0 * (params.w_max - params.w_base))
        a.r = None
        a.sigma = 0.0
        a.a = 0
        a.e = 0.0
        a.pending_signal = None
        a.compliance = 0.0
        a.b_eff = [0.5, 0.5]
        agents.append(a)
    return agents
```

**Invariants at t=0:**
- All agents identical (homogeneous parameters)
- All beliefs uniform → 50-50 strategy split in expectation
- All DDM accumulators at 0 → no norm formation pressure
- All pending signals None → no enforcement

---

## 4. Per-Tick Pipeline

```
function run_tick(t, agents, params) -> TickMetrics:
    ts = TickState()

    stage_1_pair_and_act(agents, ts, params)       # Write: actions, predictions
    stage_2_observe_and_memory(agents, ts, params)  # Write: fifo, b_exp, observations
    stage_3_confidence(agents, ts, params)           # Write: C, w
    stage_4_normative(agents, ts, params)            # Write: e/r/sigma/a; consume pending_signal
    stage_5_enforce(agents, ts, params)              # Write: pending_signal on receivers
    metrics = stage_6_metrics(agents, ts, params)    # Read-only

    return metrics
```

**Synchronisation guarantee**: Each stage processes **all** agents before the next
stage begins. Within a stage, agent processing order is irrelevant because no
agent reads another agent's state that was modified in the same stage.

Exception: Stage 5 writes to other agents' `pending_signal`, but this field is
only read in Stage 4 of the **next** tick, so no within-tick conflict exists.

---

### 4.1 Stage 1: Pair and Action

**Purpose**: Form random pairs, compute effective beliefs, select actions, make predictions.

```python
def stage_1_pair_and_act(agents, ts, params):
    # 1a. Form pairs via random permutation
    indices = rng.permutation(params.N)
    ts.pairs = [(indices[2*k], indices[2*k+1]) for k in range(params.N // 2)]

    ts.interactions = []
    for (i, j) in ts.pairs:
        # 1b. Compute effective belief (uses state from end of previous tick)
        compute_effective_belief(agents[i])
        compute_effective_belief(agents[j])

        # 1c. Action selection: probability matching
        action_i = 0 if rng.random() < agents[i].b_eff[0] else 1
        action_j = 0 if rng.random() < agents[j].b_eff[0] else 1

        # 1d. MAP prediction (deterministic, tie-break random)
        pred_i = map_predict(agents[i].b_eff)   # i predicts j's action
        pred_j = map_predict(agents[j].b_eff)   # j predicts i's action

        # 1e. Record
        ts.interactions.append(InteractionRecord(
            i=i, j=j,
            action_i=action_i, action_j=action_j,
            pred_i=pred_i, pred_j=pred_j,
            coordinated=(action_i == action_j)
        ))

def compute_effective_belief(agent):
    if agent.r is not None:
        agent.compliance = agent.sigma ** k
        b_norm = [1.0, 0.0] if agent.r == 0 else [0.0, 1.0]
        c = agent.compliance
        agent.b_eff = [c * b_norm[0] + (1 - c) * agent.b_exp[0],
                       c * b_norm[1] + (1 - c) * agent.b_exp[1]]
    else:
        agent.compliance = 0.0
        agent.b_eff = agent.b_exp[:]    # copy

def map_predict(b_eff) -> Strategy:
    if b_eff[0] > b_eff[1]:
        return 0    # predict A
    elif b_eff[1] > b_eff[0]:
        return 1    # predict B
    else:
        return rng.choice([0, 1])       # tie-break: random (DD-10)
```

**Notes:**
- Action selection is stochastic (probability matching); prediction is deterministic (MAP).
- `b_eff` is computed from the state at the **end of the previous tick**.
  At t=0, this is the initialisation state: b\_eff = [0.5, 0.5].

---

### 4.2 Stage 2: Observe and Update Experience Memory

**Purpose**: Add partner's strategy to FIFO, sample V additional observations,
recompute experience belief.

```python
def stage_2_observe_and_memory(agents, ts, params):
    # Build lookup: agent_id → (partner_id, partner_action)
    partner_map = {}
    for rec in ts.interactions:
        partner_map[rec.i] = (rec.j, rec.action_j)
        partner_map[rec.j] = (rec.i, rec.action_i)

    # Collect all interactions for observation sampling
    all_interactions = ts.interactions  # len = N/2

    for i in range(params.N):
        partner_j, partner_action = partner_map[i]

        # 2a. Add partner's strategy to FIFO (DD-1: partner only)
        agents[i].fifo.append(partner_action)

        # 2b. Build observation set O_i(t) for normative memory
        obs = [partner_action]   # Always includes partner

        if params.V > 0:
            # Eligible interactions: those not involving agent i
            eligible = [rec for rec in all_interactions
                        if rec.i != i and rec.j != i]
            n_sample = min(params.V, len(eligible))
            sampled = rng.choice(eligible, size=n_sample, replace=False)

            for rec in sampled:
                # Observe one random participant's strategy (DD-4)
                if rng.random() < 0.5:
                    obs.append(rec.action_i)
                else:
                    obs.append(rec.action_j)

        ts.observations[i] = obs   # len = 1 + actual_V (may be < 1+V if N small)

        # 2c. Recompute experience belief from FIFO
        agents[i].b_exp = compute_b_exp(agents[i])

def compute_b_exp(agent) -> float[2]:
    # Use the last w entries from FIFO (DD-9)
    window = min(agent.w, len(agent.fifo))
    if window == 0:
        return [0.5, 0.5]
    # fifo[-window:] = most recent `window` entries
    recent = list(agent.fifo)[-window:]
    n_A = sum(1 for s in recent if s == 0)
    b_A = n_A / window
    return [b_A, 1.0 - b_A]
```

**Observation semantics:**
- `obs[0]` is always the direct partner's strategy.
- `obs[1:]` are from V additional random interactions (if V > 0).
- Total `|obs| = 1 + min(V, N/2 - 1)`. When N=2, no additional observations possible.

**FIFO semantics:**
- Max capacity = w\_max (constant). When full, oldest entry is dropped on append.
- Belief uses last `w` entries (w ≤ w\_max). If FIFO has fewer than `w` entries
  (early ticks), use all available entries.

---

### 4.3 Stage 3: Confidence Update

**Purpose**: Update predictive confidence based on prediction accuracy, recompute window size.

```python
def stage_3_confidence(agents, ts, params):
    partner_map = {}   # same as Stage 2
    for rec in ts.interactions:
        partner_map[rec.i] = (rec.j, rec.action_j, rec.pred_i)
        partner_map[rec.j] = (rec.i, rec.action_i, rec.pred_j)

    for i in range(params.N):
        _, partner_action, my_prediction = partner_map[i]
        correct = (my_prediction == partner_action)

        if correct:
            agents[i].C = agents[i].C + params.alpha * (1.0 - agents[i].C)
        else:
            agents[i].C = agents[i].C * (1.0 - params.beta)

        # Clamp (should be redundant but defensive)
        agents[i].C = max(0.0, min(1.0, agents[i].C))

        # Recompute window size
        agents[i].w = params.w_base + int(agents[i].C * (params.w_max - params.w_base))
```

**Dynamics:**
- Correct prediction: additive increase, saturating at 1. Slow build.
- Wrong prediction: multiplicative decay. Fast collapse.
- Steady state: C\* = max(b) · α / (max(b) · α + min(b) · β)

**Window semantics:**
- w = w\_base + ⌊C × (w\_max − w\_base)⌋
- At C=0: w = w\_base (shortest memory)
- At C=1: w = w\_max (longest memory)
- New w takes effect in **next tick's** Stage 2 (when b\_exp is recomputed).

---

### 4.4 Stage 4: Normative Update

**Purpose**: Update normative memory — DDM evidence accumulation (pre-crystallisation)
or anomaly tracking / strengthening / crisis (post-crystallisation). Consume pending
enforcement signals.

```python
def stage_4_normative(agents, ts, params):
    for i in range(params.N):
        obs = ts.observations[i]  # O_i(t) from Stage 2

        if agents[i].r is None:
            # ── PRE-CRYSTALLISATION: DDM ──
            ddm_update(agents[i], obs, params)
        else:
            # ── POST-CRYSTALLISATION: anomaly / strengthening / crisis ──
            post_crystal_update(agents[i], i, obs, ts, params)
```

#### 4.4.1 Pre-Crystallisation: DDM Update

```python
def ddm_update(agent, obs, params):
    # a) Signed consistency from full observation set
    n_A = sum(1 for s in obs if s == 0)
    n_B = len(obs) - n_A
    f_diff = (n_A - n_B) / len(obs)      # ∈ [-1, 1]

    # b) Drift: confidence-gated
    drift = (1.0 - agent.C) * f_diff

    # c) Signal push from previous tick's enforcement (DD-5, DD-8)
    signal_push = 0.0
    if agent.pending_signal is not None:
        direction = +1.0 if agent.pending_signal == 0 else -1.0   # dir(A)=+1, dir(B)=-1
        signal_push = params.Phi * (1.0 - agent.C) * params.gamma_signal * direction
        agent.pending_signal = None       # consumed

    # d) Noise
    noise = rng.normal(0, params.sigma_noise)

    # e) Evidence accumulation
    agent.e += drift + signal_push + noise

    # f) Crystallisation check
    if abs(agent.e) >= params.theta_crystal:
        agent.r = 0 if agent.e > 0 else 1   # A if positive, B if negative
        agent.sigma = params.sigma_0
        agent.a = 0
        # e is NOT reset; it's irrelevant post-crystallisation
        # (will be reset to 0 only on dissolution)
```

**Key properties:**
- At 50-50 (f\_diff = 0): drift = 0. Accumulator does a random walk. Expected time
  to crystallise ≈ θ² / σ\_noise² = 900 ticks.
- At 70-30 with C=0.3: drift = 0.7 × 0.4 = 0.28/tick. Expected time ≈ 11 ticks.
- Low-C agents crystallise faster (drift ∝ (1−C)). This is H2.
- Signal push is independent of drift — it always pushes toward the enforced strategy.

#### 4.4.2 Post-Crystallisation: Anomaly, Strengthening, Crisis

```python
def post_crystal_update(agent, agent_id, obs, ts, params):
    norm = agent.r  # A (0) or B (1)

    # ── Separate partner observation from V observations ──
    partner_action = obs[0]
    v_observations = obs[1:]       # may be empty if V=0

    # ── Count violations and conformities from V observations ──
    v_violations = sum(1 for s in v_observations if s != norm)
    v_conform = sum(1 for s in v_observations if s == norm)

    # ── Handle partner observation ──
    partner_conform = 0
    partner_violation = 0
    enforcement_triggered = False

    if partner_action == norm:
        partner_conform = 1
    else:
        # Partner violated — enforce or accumulate? (DD-6, DD-7)
        can_enforce = (params.Phi > 0) and (agent.sigma > params.theta_enforce)
        if can_enforce:
            enforcement_triggered = True
            # Partner violation NOT counted as anomaly
        else:
            partner_violation = 1

    # ── Totals ──
    total_violations = v_violations + partner_violation
    total_conform = v_conform + partner_conform

    # ── Batch strengthening (DD-3) ──
    for _ in range(total_conform):
        agent.sigma = min(1.0, agent.sigma + params.alpha_sigma * (1.0 - agent.sigma))

    # ── Batch anomaly accumulation (DD-3) ──
    agent.a += total_violations

    # ── Crisis check (once, after all updates) (DD-3) ──
    if agent.a >= params.theta_crisis:
        agent.sigma *= params.lambda_crisis
        agent.a = 0

        # Dissolution check
        if agent.sigma < params.sigma_min:
            agent.r = None
            agent.e = 0.0           # Reset DDM for re-crystallisation
            agent.sigma = 0.0
            agent.a = 0

    # ── Record enforcement intent for Stage 5 ──
    if enforcement_triggered:
        ts.enforcement_intents[agent_id] = get_partner_id(agent_id, ts)

    # ── Consume wasted pending signal (DD-8) ──
    # Post-crystallised agents ignore incoming signals (DDM inactive)
    agent.pending_signal = None
```

**Batch strengthening detail:**
- Formula applied `total_conform` times sequentially.
- With α\_σ = 0.005 and max 6 conforming observations, the sequential vs. single-step
  difference is negligible (<0.01%).
- Equivalent closed form: σ\_new = 1 − (1 − σ\_old) × (1 − α\_σ)^n\_conform

**Crisis detail:**
- Crisis fires when accumulated anomalies ≥ θ\_crisis (across **all ticks**, not per-tick).
- After crisis: anomaly counter resets to 0, σ drops by factor λ\_crisis.
- If σ drops below σ\_min = 0.1: norm dissolves entirely (r = None, e = 0).
- After dissolution, agent re-enters pre-crystallisation and can form a new norm
  (possibly for the other strategy).

---

### 4.5 Stage 5: Partner-Directed Enforcement

**Purpose**: Agents flagged for enforcement in Stage 4 write a `pending_signal`
to their matched partner. Signal takes effect in the partner's DDM in the **next tick**.

```python
def stage_5_enforce(agents, ts, params):
    ts_enforcements_count = 0

    for enforcer_id, partner_id in ts.enforcement_intents.items():
        enforced_strategy = agents[enforcer_id].r

        # Write pending signal to partner (DD-8: strategy direction only)
        # If partner already has a pending_signal (from a different source),
        # overwrite — at most one signal per agent per tick.
        agents[partner_id].pending_signal = enforced_strategy
        ts_enforcements_count += 1

    # Store count for metrics
    ts.num_enforcements = ts_enforcements_count
```

**Signal lifecycle:**
1. **Tick t, Stage 4**: Agent i detects partner j violating; flags enforcement intent.
2. **Tick t, Stage 5**: `agents[j].pending_signal = agents[i].r`
3. **Tick t+1, Stage 4**: If j is pre-crystallised, signal\_push is computed:
   `Φ × (1 − C_j) × γ × dir(pending_signal)` and added to DDM. Signal cleared.
   If j is post-crystallised, signal is consumed and wasted.

**At most one signal per agent per tick**: Because each agent has exactly one partner,
and only the partner can send an enforcement signal, an agent receives at most one
signal per tick. No aggregation needed.

**Mutual enforcement**: If both agents in a pair have norms and observe mutual
violations, both may send signals. But since both are post-crystallised, both
signals are wasted (received by agents with inactive DDMs).

---

### 4.6 Stage 6: Metrics and Norm Detection

**Purpose**: Compute per-tick metrics. Read-only (no state modification).

```python
def stage_6_metrics(agents, ts, params) -> TickMetrics:
    N = params.N
    actions = collect_all_actions(ts)  # list of Strategy for all agents this tick

    fraction_A = sum(1 for a in actions if a == 0) / N
    mean_C = sum(ag.C for ag in agents) / N
    coord_rate = sum(1 for rec in ts.interactions if rec.coordinated) / len(ts.interactions)

    crystallised = [ag for ag in agents if ag.r is not None]
    num_cryst = len(crystallised)
    mean_sigma = (sum(ag.sigma for ag in crystallised) / num_cryst) if num_cryst > 0 else 0.0

    # Belief accuracy and consensus
    b_A_values = [ag.b_eff[0] for ag in agents]
    belief_error = sum(abs(b - fraction_A) for b in b_A_values) / N
    mean_b = sum(b_A_values) / N
    belief_var = sum((b - mean_b) ** 2 for b in b_A_values) / N

    # Norm detection (Section 5)
    norm_level = detect_norm_level(agents, fraction_A, belief_error,
                                    belief_var, num_cryst, history)

    return TickMetrics(
        tick=t,
        fraction_A=fraction_A,
        mean_confidence=mean_C,
        coordination_rate=coord_rate,
        num_crystallised=num_cryst,
        mean_norm_strength=mean_sigma,
        num_enforcements=ts.num_enforcements,
        norm_level=norm_level,
        belief_error=belief_error,
        belief_variance=belief_var,
        convergence_counter=update_convergence_counter(fraction_A, params)
    )
```

---

## 5. Norm Detection: 6-Level Hierarchy

Each level subsumes all lower levels. The norm level is the **highest** level whose
conditions are currently met.

```python
def detect_norm_level(agents, fraction_A, belief_error, belief_var,
                       num_crystallised, history, params) -> int:
    N = params.N
    majority_frac = max(fraction_A, 1.0 - fraction_A)

    # Level 0: NONE (default)
    level = 0

    # Level 1: BEHAVIORAL — behavioural regularity
    #   Condition: ≥ 95% play the same strategy, stable for 50 ticks
    if majority_frac >= 0.95:
        ticks_at_majority = count_consecutive_ticks(history, lambda m: max(m.fraction_A, 1 - m.fraction_A) >= 0.95)
        if ticks_at_majority >= 50:
            level = 1

    # Level 2: EMPIRICAL — + accurate beliefs
    #   Condition: Level 1 AND mean |b_A_eff - actual_fraction_A| < 0.10
    if level >= 1 and belief_error < 0.10:
        level = 2

    # Level 3: SHARED — + belief consensus
    #   Condition: Level 2 AND Var(b_A_eff) < 0.05
    if level >= 2 and belief_var < 0.05:
        level = 3

    # Level 4: NORMATIVE — + norm internalisation
    #   Condition: Level 3 AND ≥ 80% agents have crystallised norms
    if level >= 3 and num_crystallised / N >= 0.80:
        level = 4

    # Level 5: INSTITUTIONAL — + self-enforcing stability
    #   Condition: Level 4 for ≥ 200 consecutive ticks
    if level >= 4:
        ticks_at_level4 = count_consecutive_ticks(history, lambda m: m.norm_level >= 4)
        if ticks_at_level4 >= 200:
            level = 5

    return level
```

**Important**: Levels 4–5 require `enable_normative = True`. Without normative memory,
max achievable is Level 3.

---

## 6. Convergence and Termination

```python
def check_convergence(history, params) -> bool:
    if len(history) < params.convergence_window:
        return False
    recent = history[-params.convergence_window:]
    return all(
        max(m.fraction_A, 1.0 - m.fraction_A) >= params.convergence_threshold
        for m in recent
    )
```

**Termination conditions** (any one triggers stop):
1. `tick >= T` (max ticks reached)
2. `check_convergence() == True` (stable consensus achieved)

When termination is by convergence, record the convergence tick (first tick of
the convergence window).

---

## 7. Invariants and Edge Cases

### 7.1 Invariants (must hold at all times)

| Invariant | Check |
|-----------|-------|
| C ∈ [0, 1] | After every confidence update |
| σ ∈ [0, 1] | After every strengthening/crisis |
| b\_exp sums to 1 | After every belief recomputation |
| b\_eff sums to 1 | After every effective belief computation |
| w ∈ [w\_base, w\_max] | After every window recomputation |
| a ≥ 0 | Anomaly counter non-negative |
| If r is None: σ = 0 and a = 0 | No norm → no strength, no anomalies |
| len(fifo) ≤ w\_max | FIFO capacity respected |

### 7.2 Edge Cases

| Case | Handling |
|------|----------|
| N is odd | Adjust to N−1 (drop one agent from pairing, still participates in observations) or reject input |
| V ≥ N/2 | Cap at N/2 − 1 (can't observe more interactions than exist, excluding own) |
| N = 2 | Only 1 pair; V observations = 0 (no other interactions) |
| FIFO empty (t=0) | b\_exp = [0.5, 0.5] |
| FIFO shorter than w | Use all entries in FIFO |
| b\_eff = [0.5, 0.5] (tie) | Action: 50-50 random. Prediction: random tie-break (DD-10). |
| Both agents enforce each other | Both signals sent; both wasted (both post-crystallised) |
| Signal to post-crystallised agent | Consumed and ignored (DDM inactive) |
| Signal to agent who dissolves same tick | Dissolution in Stage 4 resets e=0; signal from Stage 5 is written to pending\_signal; consumed in next tick's Stage 4 where agent is pre-crystallised → signal takes effect |
| Crisis reduces σ but not below σ\_min | Norm survives weakened; anomaly counter resets; recovery via strengthening possible |
| Agent crystallises and gets V observation violations in same tick | Not possible: crystallisation happens in DDM (pre-crystallised path); V observations are processed in the same path as drift. Post-crystallisation branch is not entered until next tick. |
| All agents crystallised (no DDM active) | Enforcement signals have no effect; model is in steady state unless crises occur |
| Φ = 0 with V > 0 | Observations feed DDM and anomaly, but no enforcement. Loop 2 active, Loop 3 incomplete. |
| V = 0 with Φ > 0 | Each agent sees only partner. f\_diff = ±1 (noisy). Enforcement triggers from partner violations. Loop 3 partially active but slow. |

### 7.3 State Transition Diagram: Normative Memory

```
                          ┌──────────────┐
         init             │  NO NORM     │
        ─────────────────→│  r = None    │
                          │  e ∈ ℝ       │
                          └──────┬───────┘
                                 │ |e| ≥ θ_crystal
                                 ▼
                          ┌──────────────┐
                          │  HAS NORM    │ ←── strengthening
                          │  r = A or B  │     (σ ↑ per conform)
                          │  σ ∈ (0, 1]  │
                          │  a ∈ ℕ       │
                          └──┬───────┬───┘
                             │       │
                a ≥ θ_crisis │       │ a < θ_crisis
                             ▼       │ (normal operation)
                    ┌────────────┐   │
                    │  CRISIS    │   │
                    │  σ *= λ    │   │
                    │  a = 0     │   │
                    └──┬─────┬───┘   │
                       │     │       │
            σ < σ_min  │     │ σ ≥ σ_min
                       ▼     └───────→ back to HAS NORM
              ┌──────────────┐
              │  DISSOLVED   │
              │  r = None    │
              │  e = 0       │
              │  σ = 0       │
              └──────┬───────┘
                     │ re-enters DDM
                     ▼
              (back to NO NORM)
```

---

## 8. Execution Order Summary (Single Tick)

For maximum clarity, here is the complete tick as a flat sequence:

```
TICK t:
│
├─ STAGE 1: Pair and Action
│   For all agents (via pairs):
│     1. Compute b_eff from {b_exp, r, sigma} [end-of-tick-(t-1) values]
│     2. Sample action from b_eff (probability matching)
│     3. Compute MAP prediction from b_eff
│     4. Record (action, prediction, partner)
│
├─ STAGE 2: Observe and Experience Memory
│   For all agents:
│     1. Append partner_action to FIFO (1 entry)
│     2. Sample V additional strategies from other interactions
│     3. Store observation set O_i = [partner_action, ...V strategies]
│     4. Recompute b_exp from FIFO (last w entries; w from tick t-1)
│
├─ STAGE 3: Confidence Update
│   For all agents:
│     1. Compare prediction (Stage 1) with partner_action
│     2. Update C: additive increase or multiplicative decay
│     3. Recompute w = f(C)
│     [New w takes effect in NEXT tick's Stage 2]
│
├─ STAGE 4: Normative Update
│   For all agents:
│     IF pre-crystallised (r = None):
│       1. Compute f_A - f_B from O_i
│       2. drift = (1 - C) × (f_A - f_B)
│       3. Read & clear pending_signal → compute signal_push
│       4. e += drift + signal_push + noise
│       5. If |e| ≥ θ_crystal: crystallise (set r, σ₀, a=0)
│     IF post-crystallised (r ≠ None):
│       1. Count violations (V obs → anomaly; partner → enforce or anomaly)
│       2. Count conformities → strengthening
│       3. Apply batch strengthening (n_conform times)
│       4. Apply batch anomaly (a += n_violations)
│       5. Check crisis: if a ≥ θ_crisis → σ *= λ, a=0
│       6. Check dissolution: if σ < σ_min → dissolve (r=None, e=0)
│       7. Clear pending_signal (wasted on post-crystallised)
│       8. Record enforcement intent if triggered
│
├─ STAGE 5: Partner-Directed Enforcement
│   For all enforcement intents from Stage 4:
│     1. Write enforcer's norm strategy to partner's pending_signal
│     [Partner reads this in NEXT tick's Stage 4]
│
└─ STAGE 6: Metrics (read-only)
      Compute fraction_A, mean_C, coordination_rate, num_crystallised,
      mean_sigma, belief_error, belief_var, norm_level, convergence
```

---

## 9. Variable Read/Write Map

This table shows exactly when each variable is written and when it is read, to
prevent stale-value bugs.

| Variable | Written in | Read in | Latency |
|----------|-----------|---------|---------|
| action\_i | Stage 1 | Stage 2 (partner obs), Stage 6 | Same tick |
| prediction\_i | Stage 1 | Stage 3 (accuracy check) | Same tick |
| fifo | Stage 2 (append) | Stage 2 (b\_exp computation) | Same tick (after append) |
| b\_exp | Stage 2 | Stage 1 of **next tick** (via b\_eff) | +1 tick |
| C | Stage 3 | Stage 4 (DDM drift, signal push), Stage 1 of **next tick** (via w → b\_exp) | Same tick (Stage 4); +1 tick (Stage 1) |
| w | Stage 3 | Stage 2 of **next tick** (b\_exp window) | +1 tick |
| e | Stage 4 | Stage 4 (same: accumulate) | Same tick (self-read-write) |
| r | Stage 4 (crystallise/dissolve) | Stage 1 of **next tick** (b\_eff), Stage 4 of **next tick** (branch), Stage 5 (enforcement) | Same tick (Stage 5); +1 tick (Stages 1, 4) |
| σ | Stage 4 (strengthen/crisis) | Stage 1 of **next tick** (compliance), Stage 4 (enforce gate) | Same tick (enforce gate); +1 tick (compliance) |
| a | Stage 4 (accumulate/reset) | Stage 4 (crisis check) | Same tick (self-read-write) |
| pending\_signal | Stage 5 | Stage 4 of **next tick** | +1 tick |

---

## 10. Testing Checklist

These are minimal correctness checks. Each corresponds to a mechanism or hypothesis.

| # | Test | Expected |
|---|------|----------|
| 1 | V=0, Φ=0, fixed memory: run 1000 ticks | Slow convergence; max Level 1 |
| 2 | Dynamic memory vs. fixed: compare convergence speed | Dynamic faster |
| 3 | DDM at 50-50: track crystallisation times | Mean ≈ θ²/σ\_noise² = 900; high variance |
| 4 | DDM at 70-30: track crystallisation times | Mean ≈ θ / ((1-C) × 0.4) ≈ 11 ticks at C=0.3 |
| 5 | Low-C agents crystallise before high-C (H2) | Negative correlation: C vs. crystallisation order |
| 6 | Enforcement signal pushes DDM (H4) | Compare crystallisation speed with/without Φ |
| 7 | Crisis after 10 anomalies: σ drops by ×0.3 | σ: 0.8 → 0.24 |
| 8 | Two crises: norm dissolves | σ: 0.8 → 0.24 → 0.072 < 0.1 → dissolved |
| 9 | Strengthening recovery | After crisis (σ=0.24), 100 conforming obs → σ ≈ 0.59 |
| 10 | Partner-only enforcement | Enforce signal only to matched partner, not V observations |
| 11 | Φ=0 blocks all enforcement | No pending\_signals written; norm fragile |
| 12 | V=0: DDM noisy (f\_diff = ±1) | Crystallisation slow and unreliable |
| 13 | Full model (dynamic, normative, V>0, Φ>0) | Reaches Level 5 |
| 14 | Backward compatibility: no normative, fixed memory | Equivalent to V1 model |
