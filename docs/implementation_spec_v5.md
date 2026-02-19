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
| Number of agents | N | 100 | Int | even, ≥ 2 | — |
| Number of ticks | T | 3000 | Int | ≥ 1 | — |
| Random seed | seed | nothing | Union{Int, Nothing} | — | For reproducibility |
| Confidence increase | α | 0.1 | Float64 | (0, 1) | Slovic 1993 |
| Confidence decrease | β | 0.3 | Float64 | (0, 1), β > α | Slovic 1993 |
| Initial confidence | C₀ | 0.5 | Float64 | [0, 1] | — |
| Memory base window | w\_base | 2 | Int | ≥ 1 | Hertwig 2010 |
| Memory max window | w\_max | 6 | Int | ≥ w\_base | Nevo & Erev 2012 |

### 1.2 Normative Layer

| Parameter | Symbol | Default | Type | Constraint | Source |
|-----------|--------|---------|------|------------|--------|
| Enable normative memory | enable\_normative | false | Bool | — | Gates entire normative subsystem. When false, Stages 4–5 are skipped and max norm level is 3. |
| DDM noise (std dev) | σ\_noise | 0.1 | Float64 | ≥ 0 | Germar 2014 |
| Crystallisation threshold | θ\_crystal | 3.0 | Float64 | > 0 | calibration |
| Initial norm strength | σ₀ | 0.8 | Float64 | (0, 1] | calibration |
| Crisis threshold | θ\_crisis | 10 | Int | ≥ 1 | calibration |
| Crisis decay | λ\_crisis | 0.3 | Float64 | (0, 1) | calibration |
| Dissolution threshold | σ\_min | 0.1 | Float64 | (0, σ₀) | DD-2 |
| Strengthen rate | α\_σ | 0.005 | Float64 | (0, 1) | Will 2023 |
| Enforce threshold | θ\_enforce | 0.7 | Float64 | (0, 1) | Toribio 2023 |
| Compliance exponent | k | 2.0 | Float64 | > 0 | calibration |
| Signal amplification | γ\_signal | 2.0 | Float64 | > 0 | calibration |

### 1.3 Environment

| Parameter | Symbol | Default | Type | Constraint | Notes |
|-----------|--------|---------|------|------------|-------|
| Visibility | V | 0 | Int | ≥ 0 | Additional observations/tick. Default 0 = isolated learning. |
| Social pressure | Φ | 0.0 | Float64 | ≥ 0 | Enforcement gain. 0 = disabled, 1 = baseline. |

### 1.4 Convergence

| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| convergence\_threshold | 0.95 | Float64 | Fraction for majority |
| convergence\_window | 50 | Int | Ticks maintaining threshold |

---

## 2. Data Structures

### 2.1 Strategy Encoding and RNG

```julia
const Strategy = Int   # 0 = A, 1 = B
```

**RNG ownership**: A single `MersenneTwister(seed)` instance (from `Random`) is
created during initialisation and passed explicitly to every function that
requires randomness. All pseudocode in this spec uses the bare name `rng` for
brevity; in implementation, `rng` is always the same object, threaded through as
a parameter (`rng` argument on `run_tick!` and every helper/stage that uses
randomness, such as `stage_1_pair_and_act!`, `stage_2_observe_and_memory!`,
`stage_4_normative!`, `map_predict`, and `ddm_update!`). This guarantees:
- **Reproducibility**: identical seed → identical trajectory.
- **No hidden state**: RNG is never module-global or thread-local.
- **Parallelism-safe**: independent runs use independent RNG instances.

### 2.2 AgentState

```julia
mutable struct AgentState
    # ── Experience memory ──
    fifo::CircularBuffer{Int}     # capacity = w_max; stores partner strategies only
    b_exp::Vector{Float64}        # [b_A, b_B]; sums to 1.0; default [0.5, 0.5]

    # ── Confidence ──
    C::Float64                    # ∈ [0, 1]; init = C₀
    w::Int                        # Current window = f(C); init = w_base + floor(C₀ * (w_max - w_base))

    # ── Normative memory ──
    r::Union{Int, Nothing}        # Norm rule; nothing = no norm
    sigma::Float64                # Norm strength ∈ [0, 1]; 0.0 when no norm
    a::Int                        # Anomaly counter; 0 when no norm
    e::Float64                    # DDM evidence accumulator; 0.0 initial

    # ── Enforcement buffer ──
    pending_signal::Union{Int, Nothing}   # Enforced strategy from prev tick, or nothing

    # ── Derived (recomputed in Stage 1) ──
    compliance::Float64           # sigma^k; 0.0 if no norm
    b_eff::Vector{Float64}        # Effective belief after normative constraint
end
```

### 2.3 InteractionRecord

```julia
struct InteractionRecord
    i::Int                        # Agent index
    j::Int                        # Partner index
    action_i::Int                 # i's chosen strategy
    action_j::Int                 # j's chosen strategy
    pred_i::Int                   # i's MAP prediction of j
    pred_j::Int                   # j's MAP prediction of i
    coordinated::Bool             # action_i == action_j
end
```

### 2.4 TickState (transient, per-tick working data)

All fields are initialised when `TickState()` is constructed. Stages populate
them progressively; later stages may read fields written by earlier stages.

```julia
mutable struct TickState
    pairs::Vector{Tuple{Int, Int}}                          # Set in Stage 1
    interactions::Vector{InteractionRecord}                  # Set in Stage 1
    observations::Dict{Int, Vector{Int}}                     # Set in Stage 2
    enforcement_intents::Dict{Int, Tuple{Int, Int}}          # enforcer_id => (partner_id, enforced_strategy)
    num_enforcements::Int                                    # Set in Stage 5

    TickState() = new(
        Tuple{Int,Int}[], InteractionRecord[], Dict{Int,Vector{Int}}(),
        Dict{Int,Tuple{Int,Int}}(), 0
    )
end
```

### 2.5 TickMetrics

```julia
struct TickMetrics
    tick::Int
    fraction_A::Float64                 # Fraction of agents who played A
    mean_confidence::Float64            # Mean C across all agents
    coordination_rate::Float64          # Fraction of pairs that coordinated
    num_crystallised::Int               # Agents with r ≠ nothing
    mean_norm_strength::Float64         # Mean σ among crystallised agents (0 if none)
    num_enforcements::Int               # Enforcement events this tick
    norm_level::Int                     # 0–5 (Section 5)
    belief_error::Float64               # Mean |b_A_eff - fraction_A| across agents
    belief_variance::Float64            # Var(b_A_eff) across agents
    convergence_counter::Int            # Consecutive ticks at ≥ convergence_threshold
end
```

---

## 3. Initialization

```julia
function initialize(params)
    rng = MersenneTwister(params.seed)
    agents = AgentState[]
    for i in 1:params.N
        ag = AgentState(
            CircularBuffer{Int}(params.w_max),                                      # fifo (empty)
            [0.5, 0.5],                                                             # b_exp
            params.C0,                                                              # C
            params.w_base + floor(Int, params.C0 * (params.w_max - params.w_base)), # w
            nothing,                                                                # r
            0.0,                                                                    # sigma
            0,                                                                      # a
            0.0,                                                                    # e
            nothing,                                                                # pending_signal
            0.0,                                                                    # compliance
            [0.5, 0.5],                                                             # b_eff
        )
        push!(agents, ag)
    end
    return agents, rng    # rng is passed to run_tick! and all stage functions
end
```

**Invariants at t=0:**
- All agents identical (homogeneous parameters)
- All beliefs uniform → 50-50 strategy split in expectation
- All DDM accumulators at 0 → no norm formation pressure
- All pending signals nothing → no enforcement

---

## 4. Per-Tick Pipeline

```julia
function run_tick!(t, agents, history, params, rng)
    # Execute one tick. `history` is the vector of TickMetrics from ticks 0..t-1.
    ts = TickState()

    stage_1_pair_and_act!(agents, ts, params, rng)       # Write: actions, predictions
    stage_2_observe_and_memory!(agents, ts, params, rng) # Write: fifo, b_exp, observations
    stage_3_confidence!(agents, ts, params)               # Write: C, w
    if params.enable_normative
        stage_4_normative!(agents, ts, params, rng)      # Write: e/r/sigma/a; consume pending_signal
        stage_5_enforce!(agents, ts, params)              # Write: pending_signal on receivers
    else
        ts.num_enforcements = 0
    end
    metrics = stage_6_metrics(agents, ts, t, history, params)  # Read-only

    return metrics
end
```

**Synchronisation guarantee**: Each stage processes **all** agents before the next
stage begins. Within a stage, agent processing order is irrelevant because no
agent reads another agent's state that was modified in the same stage.

Exception: Stage 5 writes to other agents' `pending_signal`, but this field is
only read in Stage 4 of the **next** tick, so no within-tick conflict exists.

---

### 4.1 Stage 1: Pair and Action

**Purpose**: Form random pairs, compute effective beliefs, select actions, make predictions.

```julia
function stage_1_pair_and_act!(agents, ts, params, rng)
    # 1a. Form pairs via random permutation
    indices = randperm(rng, params.N)
    ts.pairs = [(indices[2k-1], indices[2k]) for k in 1:(params.N ÷ 2)]

    empty!(ts.interactions)
    for (i, j) in ts.pairs
        # 1b. Compute effective belief (uses state from end of previous tick)
        compute_effective_belief!(agents[i], params)
        compute_effective_belief!(agents[j], params)

        # 1c. Action selection: probability matching
        action_i = rand(rng) < agents[i].b_eff[1] ? 0 : 1
        action_j = rand(rng) < agents[j].b_eff[1] ? 0 : 1

        # 1d. MAP prediction (deterministic, tie-break random)
        pred_i = map_predict(agents[i].b_eff, rng)   # i predicts j's action
        pred_j = map_predict(agents[j].b_eff, rng)   # j predicts i's action

        # 1e. Record
        push!(ts.interactions, InteractionRecord(
            i, j, action_i, action_j, pred_i, pred_j, action_i == action_j
        ))
    end
end

function compute_effective_belief!(agent, params)
    if agent.r !== nothing
        agent.compliance = agent.sigma ^ params.k
        b_norm = agent.r == 0 ? [1.0, 0.0] : [0.0, 1.0]
        c = agent.compliance
        agent.b_eff = [c * b_norm[1] + (1 - c) * agent.b_exp[1],
                       c * b_norm[2] + (1 - c) * agent.b_exp[2]]
    else
        agent.compliance = 0.0
        agent.b_eff = copy(agent.b_exp)
    end
end

function map_predict(b_eff, rng)::Int
    if b_eff[1] > b_eff[2]
        return 0    # predict A
    elseif b_eff[2] > b_eff[1]
        return 1    # predict B
    else
        return rand(rng, [0, 1])       # tie-break: random (DD-10)
    end
end
```

**Notes:**
- Action selection is stochastic (probability matching); prediction is deterministic (MAP).
- `b_eff` is computed from the state at the **end of the previous tick**.
  At t=0, this is the initialisation state: b\_eff = [0.5, 0.5].

---

### 4.2 Stage 2: Observe and Update Experience Memory

**Purpose**: Add partner's strategy to FIFO, sample V additional observations,
recompute experience belief.

```julia
function stage_2_observe_and_memory!(agents, ts, params, rng)
    # Build lookup: agent_id => (partner_id, partner_action)
    partner_map = Dict{Int, Tuple{Int, Int}}()
    for rec in ts.interactions
        partner_map[rec.i] = (rec.j, rec.action_j)
        partner_map[rec.j] = (rec.i, rec.action_i)
    end

    # Collect all interactions for observation sampling
    all_interactions = ts.interactions  # length = N/2

    for i in 1:params.N
        partner_j, partner_action = partner_map[i]

        # 2a. Add partner's strategy to FIFO (DD-1: partner only)
        push!(agents[i].fifo, partner_action)

        # 2b. Build observation set O_i(t) for normative memory
        obs = [partner_action]   # Always includes partner

        if params.V > 0
            # Eligible interactions: those not involving agent i
            eligible = [rec for rec in all_interactions
                        if rec.i != i && rec.j != i]
            n_sample = min(params.V, length(eligible))
            sampled = sample(rng, eligible, n_sample; replace=false)

            for rec in sampled
                # Observe one random participant's strategy (DD-4)
                if rand(rng) < 0.5
                    push!(obs, rec.action_i)
                else
                    push!(obs, rec.action_j)
                end
            end
        end

        ts.observations[i] = obs   # length = 1 + actual_V (may be < 1+V if N small)

        # 2c. Recompute experience belief from FIFO
        agents[i].b_exp = compute_b_exp(agents[i])
    end
end

function compute_b_exp(agent)::Vector{Float64}
    # Use the last w entries from FIFO (DD-9)
    window = min(agent.w, length(agent.fifo))
    if window == 0
        return [0.5, 0.5]
    end
    # fifo[end-window+1:end] = most recent `window` entries
    recent = collect(agent.fifo)[end-window+1:end]
    n_A = count(s -> s == 0, recent)
    b_A = n_A / window
    return [b_A, 1.0 - b_A]
end
```

**Observation semantics:**
- `obs[1]` is always the direct partner's strategy.
- `obs[2:end]` are from V additional random interactions (if V > 0).
- Total `|obs| = 1 + min(V, N/2 - 1)`. When N=2, no additional observations possible.

**FIFO semantics:**
- Max capacity = w\_max (constant). When full, oldest entry is dropped on push.
- Belief uses last `w` entries (w ≤ w\_max). If FIFO has fewer than `w` entries
  (early ticks), use all available entries.

---

### 4.3 Stage 3: Confidence Update

**Purpose**: Update predictive confidence based on prediction accuracy, recompute window size.

```julia
function stage_3_confidence!(agents, ts, params)
    partner_map = Dict{Int, Tuple{Int, Int, Int}}()
    for rec in ts.interactions
        partner_map[rec.i] = (rec.j, rec.action_j, rec.pred_i)
        partner_map[rec.j] = (rec.i, rec.action_i, rec.pred_j)
    end

    for i in 1:params.N
        _, partner_action, my_prediction = partner_map[i]
        correct = (my_prediction == partner_action)

        if correct
            agents[i].C = agents[i].C + params.alpha * (1.0 - agents[i].C)
        else
            agents[i].C = agents[i].C * (1.0 - params.beta)
        end

        # Clamp (should be redundant but defensive)
        agents[i].C = clamp(agents[i].C, 0.0, 1.0)

        # Recompute window size
        agents[i].w = params.w_base + floor(Int, agents[i].C * (params.w_max - params.w_base))
    end
end
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

```julia
function stage_4_normative!(agents, ts, params, rng)
    for i in 1:params.N
        obs = ts.observations[i]  # O_i(t) from Stage 2

        if agents[i].r === nothing
            # ── PRE-CRYSTALLISATION: DDM ──
            ddm_update!(agents[i], obs, params, rng)
        else
            # ── POST-CRYSTALLISATION: anomaly / strengthening / crisis ──
            post_crystal_update!(agents[i], i, obs, ts, params)
        end
    end
end
```

#### 4.4.1 Pre-Crystallisation: DDM Update

```julia
function ddm_update!(agent, obs, params, rng)
    # a) Signed consistency from full observation set
    n_A = count(s -> s == 0, obs)
    n_B = length(obs) - n_A
    f_diff = (n_A - n_B) / length(obs)      # ∈ [-1, 1]

    # b) Drift: confidence-gated
    drift = (1.0 - agent.C) * f_diff

    # c) Signal push from previous tick's enforcement (DD-5, DD-8)
    signal_push = 0.0
    if agent.pending_signal !== nothing
        direction = agent.pending_signal == 0 ? +1.0 : -1.0   # dir(A)=+1, dir(B)=-1
        signal_push = params.Phi * (1.0 - agent.C) * params.gamma_signal * direction
        agent.pending_signal = nothing       # consumed
    end

    # d) Noise
    noise = randn(rng) * params.sigma_noise

    # e) Evidence accumulation
    agent.e += drift + signal_push + noise

    # f) Crystallisation check
    if abs(agent.e) >= params.theta_crystal
        agent.r = agent.e > 0 ? 0 : 1   # A if positive, B if negative
        agent.sigma = params.sigma_0
        agent.a = 0
        # e is NOT reset; it's irrelevant post-crystallisation
        # (will be reset to 0 only on dissolution)
    end
end
```

**Key properties:**
- At 50-50 (f\_diff = 0): drift = 0. Accumulator does a random walk. Expected time
  to crystallise ≈ θ² / σ\_noise² = 900 ticks.
- At 70-30 with C=0.3: drift = 0.7 × 0.4 = 0.28/tick. Expected time ≈ 11 ticks.
- Low-C agents crystallise faster (drift ∝ (1−C)). This is H2.
- Signal push is independent of drift — it always pushes toward the enforced strategy.

#### 4.4.2 Post-Crystallisation: Anomaly, Strengthening, Crisis

```julia
function post_crystal_update!(agent, agent_id, obs, ts, params)
    norm = agent.r  # A (0) or B (1)

    # ── Separate partner observation from V observations ──
    partner_action = obs[1]
    v_observations = obs[2:end]       # may be empty if V=0

    # ── Count violations and conformities from V observations ──
    v_violations = count(s -> s != norm, v_observations)
    v_conform = count(s -> s == norm, v_observations)

    # ── Handle partner observation ──
    partner_conform = 0
    partner_violation = 0
    enforcement_triggered = false

    if partner_action == norm
        partner_conform = 1
    else
        # Partner violated — enforce or accumulate? (DD-6, DD-7)
        can_enforce = (params.Phi > 0) && (agent.sigma > params.theta_enforce)
        if can_enforce
            enforcement_triggered = true
            # Partner violation NOT counted as anomaly
        else
            partner_violation = 1
        end
    end

    # ── Totals ──
    total_violations = v_violations + partner_violation
    total_conform = v_conform + partner_conform

    # ── Batch strengthening (DD-3) ──
    for _ in 1:total_conform
        agent.sigma = min(1.0, agent.sigma + params.alpha_sigma * (1.0 - agent.sigma))
    end

    # ── Batch anomaly accumulation (DD-3) ──
    agent.a += total_violations

    # ── Crisis check (once, after all updates) (DD-3) ──
    if agent.a >= params.theta_crisis
        agent.sigma *= params.lambda_crisis
        agent.a = 0

        # Dissolution check
        if agent.sigma < params.sigma_min
            agent.r = nothing
            agent.e = 0.0           # Reset DDM for re-crystallisation
            agent.sigma = 0.0
            agent.a = 0
        end
    end

    # ── Record enforcement intent for Stage 5 ──
    if enforcement_triggered
        # Find partner from pairs list (each agent appears in exactly one pair)
        partner_id = 0
        for (a, b) in ts.pairs
            if a == agent_id
                partner_id = b; break
            elseif b == agent_id
                partner_id = a; break
            end
        end
        ts.enforcement_intents[agent_id] = (partner_id, norm)
    end

    # ── Consume wasted pending signal (DD-8) ──
    # Post-crystallised agents ignore incoming signals (DDM inactive)
    agent.pending_signal = nothing
end
```

**Batch strengthening detail:**
- Formula applied `total_conform` times sequentially.
- With α\_σ = 0.005 and max 6 conforming observations, the sequential vs. single-step
  difference is negligible (<0.01%).
- Equivalent closed form: σ\_new = 1 − (1 − σ\_old) × (1 − α\_σ)^n\_conform

**Crisis detail:**
- Crisis fires when accumulated anomalies ≥ θ\_crisis (across **all ticks**, not per-tick).
- After crisis: anomaly counter resets to 0, σ drops by factor λ\_crisis.
- If σ drops below σ\_min = 0.1: norm dissolves entirely (r = nothing, e = 0).
- After dissolution, agent re-enters pre-crystallisation and can form a new norm
  (possibly for the other strategy).

---

### 4.5 Stage 5: Partner-Directed Enforcement

**Purpose**: Agents flagged for enforcement in Stage 4 write a `pending_signal`
to their matched partner. Signal takes effect in the partner's DDM in the **next tick**.

```julia
function stage_5_enforce!(agents, ts, params)
    ts_enforcements_count = 0

    for (enforcer_id, (partner_id, enforced_strategy)) in ts.enforcement_intents

        # Write pending signal to partner (DD-8: strategy direction only)
        # If partner already has a pending_signal (from a different source),
        # overwrite — at most one signal per agent per tick.
        agents[partner_id].pending_signal = enforced_strategy
        ts_enforcements_count += 1
    end

    # Store count for metrics
    ts.num_enforcements = ts_enforcements_count
end
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

`stage_6_metrics` receives the tick index `t` and cumulative `history` (vector of all
prior TickMetrics) from the caller `run_tick!`. It uses `history` for norm detection
(consecutive-tick checks) and convergence tracking.

```julia
function stage_6_metrics(agents, ts, t, history, params)
    N = params.N

    # Extract each agent's action this tick from interaction records.
    # Each InteractionRecord contains both agents' actions; build a full N-vector.
    actions = Vector{Int}(undef, N)
    for rec in ts.interactions
        actions[rec.i] = rec.action_i
        actions[rec.j] = rec.action_j
    end

    fraction_A = count(a -> a == 0, actions) / N
    mean_C = sum(ag.C for ag in agents) / N
    coord_rate = count(rec -> rec.coordinated, ts.interactions) / length(ts.interactions)

    crystallised = [ag for ag in agents if ag.r !== nothing]
    num_cryst = length(crystallised)
    mean_sigma = num_cryst > 0 ? sum(ag.sigma for ag in crystallised) / num_cryst : 0.0

    # Belief accuracy and consensus
    b_A_values = [ag.b_eff[1] for ag in agents]
    belief_error = sum(abs(b - fraction_A) for b in b_A_values) / N
    mean_b = sum(b_A_values) / N
    belief_var = sum((b - mean_b)^2 for b in b_A_values) / N

    # Norm detection (Section 5)
    norm_level = detect_norm_level(agents, fraction_A, belief_error,
                                    belief_var, num_cryst, history, params)

    # Convergence counter: how many consecutive ticks (including this one)
    # the majority fraction has been ≥ convergence_threshold.
    majority_frac = max(fraction_A, 1.0 - fraction_A)
    if majority_frac >= params.convergence_threshold
        prev = length(history) > 0 ? last(history).convergence_counter : 0
        conv_counter = prev + 1
    else
        conv_counter = 0
    end

    return TickMetrics(
        t, fraction_A, mean_C, coord_rate, num_cryst,
        mean_sigma, ts.num_enforcements, norm_level,
        belief_error, belief_var, conv_counter,
    )
end
```

---

## 5. Norm Detection: 6-Level Hierarchy

Each level subsumes all lower levels. The norm level is the **highest** level whose
conditions are currently met.

```julia
function count_consecutive_ticks(history, predicate)
    # Count how many consecutive ticks at the END of history satisfy predicate.
    # Returns 0 if history is empty or the most recent tick fails the predicate.
    n = 0
    for m in Iterators.reverse(history)
        if predicate(m)
            n += 1
        else
            break
        end
    end
    return n
end

function detect_norm_level(agents, fraction_A, belief_error, belief_var,
                           num_crystallised, history, params)
    N = params.N
    majority_frac = max(fraction_A, 1.0 - fraction_A)

    # Level 0: NONE (default)
    level = 0

    # Level 1: BEHAVIORAL — behavioural regularity
    #   Condition: ≥ 95% play the same strategy, stable for 50 ticks
    if majority_frac >= 0.95
        ticks_at_majority = 1 + count_consecutive_ticks(
            history, m -> max(m.fraction_A, 1 - m.fraction_A) >= 0.95
        )
        if ticks_at_majority >= 50
            level = 1
        end
    end

    # Level 2: EMPIRICAL — + accurate beliefs
    #   Condition: Level 1 AND mean |b_A_eff - actual_fraction_A| < 0.10
    if level >= 1 && belief_error < 0.10
        level = 2
    end

    # Level 3: SHARED — + belief consensus
    #   Condition: Level 2 AND Var(b_A_eff) < 0.05
    if level >= 2 && belief_var < 0.05
        level = 3
    end

    # Level 4: NORMATIVE — + norm internalisation
    #   Condition: Level 3 AND ≥ 80% agents have crystallised norms
    if level >= 3 && num_crystallised / N >= 0.80
        level = 4
    end

    # Level 5: INSTITUTIONAL — + self-enforcing stability
    #   Condition: Level 4 for ≥ 200 consecutive ticks
    if level >= 4
        ticks_at_level4 = 1 + count_consecutive_ticks(history, m -> m.norm_level >= 4)
        if ticks_at_level4 >= 200
            level = 5
        end
    end

    return level
end
```

**Important**: Levels 4–5 require `enable_normative = true`. Without normative memory,
max achievable is Level 3.

---

## 6. Convergence and Termination

```julia
function check_convergence(history, tick_count, params)::Bool
    tick_count >= 1 || return false
    return history[tick_count].norm_level >= 5
end
```

**Termination conditions** (any one triggers stop):
1. `tick >= T` (max ticks reached, default T=3000)
2. `check_convergence() == true` (norm level 5 — institutional norm — achieved)

When termination is by convergence, record the tick at which norm level 5 was
first reached.

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
| If r is nothing: σ = 0 and a = 0 | No norm → no strength, no anomalies |
| length(fifo) ≤ w\_max | FIFO capacity respected |

### 7.2 Edge Cases

| Case | Handling |
|------|----------|
| N is odd | **Reject at initialisation** with a clear error message. The caller must supply an even N. This avoids ambiguity about which agent sits out and whether unpaired agents update confidence. |
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
        ─────────────────→│  r = nothing │
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
              │  r = nothing │
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
If `enable_normative=false`, skip Stages 4-5 and keep `num_enforcements=0`.

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
│     IF pre-crystallised (r === nothing):
│       1. Compute f_A - f_B from O_i
│       2. drift = (1 - C) × (f_A - f_B)
│       3. Read & clear pending_signal → compute signal_push
│       4. e += drift + signal_push + noise
│       5. If |e| ≥ θ_crystal: crystallise (set r, σ₀, a=0)
│     IF post-crystallised (r !== nothing):
│       1. Count violations (V obs → anomaly; partner → enforce or anomaly)
│       2. Count conformities → strengthening
│       3. Apply batch strengthening (n_conform times)
│       4. Apply batch anomaly (a += n_violations)
│       5. Check crisis: if a ≥ θ_crisis → σ *= λ, a=0
│       6. Check dissolution: if σ < σ_min → dissolve (r=nothing, e=0)
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
| fifo | Stage 2 (push!) | Stage 2 (b\_exp computation) | Same tick (after push!) |
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

### 10.1 Deterministic Correctness Tests (CI-runnable, single seed)

These verify mechanism logic with exact expected values. Each test constructs a
controlled scenario (often a single agent or small population) and asserts a
precise outcome. **No randomness in expected values** — either seed the RNG or
test the deterministic path directly.

| # | Test | Setup | Expected (exact) |
|---|------|-------|-------------------|
| D1 | Crisis σ decay | Agent with σ=0.8, a=10, θ\_crisis=10 | After crisis: σ = 0.24, a = 0, r unchanged |
| D2 | Double crisis dissolves | Agent with σ=0.8; trigger 2 crises | After 1st: σ=0.24. After 2nd: σ=0.072 < σ\_min → r=nothing, e=0, σ=0, a=0 |
| D3 | Strengthening recovery | Agent with σ=0.24; feed 100 conforming obs | σ\_new = 1 − (1−0.24)×(1−0.005)^100 ≈ 0.5938 |
| D4 | Partner-only enforcement | V=3, Φ>0; partner conforms, V obs violate | No enforcement triggered (DD-6). Anomaly a += count(V violations) |
| D5 | V obs violate, partner violates, enforce eligible | σ > θ\_enforce, Φ > 0 | Enforcement triggered for partner. Partner violation NOT counted as anomaly (DD-7). V violations counted as anomaly. |
| D6 | Φ=0 blocks enforcement | Agent with σ=0.9 > θ\_enforce; partner violates | can\_enforce = false. Violation counted as anomaly. No pending\_signal written. |
| D7 | Effective belief blending | r=0 (A), σ=0.8, k=2, b\_exp=[0.3, 0.7] | compliance = 0.64. b\_eff = [0.64×1.0 + 0.36×0.3, 0.64×0.0 + 0.36×0.7] = [0.748, 0.252] |
| D8 | Window from confidence | C=0.0, w\_base=2, w\_max=6 | w=2 |
| D9 | Window from confidence | C=1.0, w\_base=2, w\_max=6 | w=6 |
| D10 | Confidence update correct | C=0.5, α=0.1 | C\_new = 0.5 + 0.1×0.5 = 0.55 |
| D11 | Confidence update wrong | C=0.5, β=0.3 | C\_new = 0.5 × 0.7 = 0.35 |
| D12 | Backward compat: no normative | enable\_normative=false, fixed memory | Stages 4–5 skipped. Agent state has r=nothing, σ=0, a=0 for all ticks. Max norm level = 3. |
| D13 | Crystallisation direction | e = +3.1 ≥ θ\_crystal=3.0 | r = 0 (A). σ = σ₀. a = 0. |
| D14 | Crystallisation direction | e = −3.1 | r = 1 (B). σ = σ₀. a = 0. |
| D15 | Signal consumed by post-crystallised | Agent with r !== nothing receives pending\_signal | pending\_signal cleared in Stage 4. No DDM update. |

### 10.2 Statistical Hypothesis Tests (require multiple trials)

These validate emergent dynamics and model hypotheses. They require multiple
seeded trials and statistical criteria. Run with `n_trials ≥ 30` unless noted.

| # | Hypothesis | Protocol | Pass criterion |
|---|------------|----------|----------------|
| S1 | Dynamic memory accelerates convergence (H1) | Compare dynamic vs. fixed memory, N=100, 1000 ticks, 30 trials each | Mann-Whitney U test: dynamic convergence\_tick < fixed convergence\_tick, p < 0.05 |
| S2 | DDM crystallisation time at 50-50 | Force b\_exp = [0.5, 0.5] constant; track crystallisation tick per agent, 100 agents × 30 trials | Mean crystallisation tick ∈ [600, 1200] (theoretical: θ²/σ\_noise² = 900). CV > 0.5 (high variance). |
| S3 | DDM crystallisation time at 70-30 | Force b\_exp = [0.7, 0.3] constant, C = 0.3; track crystallisation tick | Mean crystallisation tick ∈ [5, 25] (theoretical: ≈11 at steady drift) |
| S4 | Low-C agents crystallise first (H2) | Full model, N=100, 30 trials; record (C at crystallisation, crystallisation tick) | Spearman ρ(C, crystallisation\_tick) > 0 (positive: higher C → later crystallisation), p < 0.05 |
| S5 | Enforcement accelerates crystallisation (H4) | Compare Φ=1 vs. Φ=0, V=5, N=100, 30 trials | Mean crystallisation tick with Φ=1 < without, Mann-Whitney p < 0.05 |
| S6 | V=0 DDM is noisy | V=0, Φ=0, N=100, 30 trials; track crystallisation times | CV of crystallisation tick > 0.8 (high dispersion from f\_diff = ±1) |
| S7 | Full model reaches Level 5 | Dynamic memory, normative, V=5, Φ=1.0, N=100, 2000 ticks, 30 trials | ≥ 80% of trials reach norm\_level = 5 |
| S8 | No-normative ceiling is Level 3 | enable\_normative=false, dynamic memory, N=100, 1000 ticks, 30 trials | 0% of trials exceed norm\_level = 3 |
