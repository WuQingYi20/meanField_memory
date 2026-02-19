# ──────────────────────────────────────────────────────────────
# Sentinel Constants (eliminate all Union types for type stability)
# ──────────────────────────────────────────────────────────────

const STRATEGY_A = Int8(0)
const STRATEGY_B = Int8(1)
const NO_NORM    = Int8(-1)   # replaces r::Nothing
const NO_SIGNAL  = Int8(-1)   # replaces pending_signal::Nothing

# ──────────────────────────────────────────────────────────────
# RingBuffer — inline circular buffer, no heap allocation per push
# ──────────────────────────────────────────────────────────────

mutable struct RingBuffer
    data::Vector{Int8}   # fixed length = capacity (w_max)
    head::Int            # next write position (1-indexed, wraps)
    count::Int           # current number of entries (≤ capacity)
end

"""
    RingBuffer(capacity::Int)

Create an empty ring buffer with given capacity.
"""
function RingBuffer(capacity::Int)
    return RingBuffer(zeros(Int8, capacity), 1, 0)
end

"""
    capacity(buf::RingBuffer)

Return the maximum number of entries the buffer can hold.
"""
@inline capacity(buf::RingBuffer) = length(buf.data)

"""
    Base.length(buf::RingBuffer)

Return the current number of entries in the buffer.
"""
@inline Base.length(buf::RingBuffer) = buf.count

"""
    push!(buf::RingBuffer, val::Integer)

Push a value into the ring buffer. If full, overwrites the oldest entry.
"""
function Base.push!(buf::RingBuffer, val::Integer)
    buf.data[buf.head] = Int8(val)
    buf.head = mod1(buf.head + 1, capacity(buf))
    buf.count = min(buf.count + 1, capacity(buf))
    return buf
end

"""
    recent(buf::RingBuffer, w::Int)

Return the last `min(w, count)` entries as a view-like iteration.
Returns a Vector{Int8} of the most recent entries (oldest to newest).
"""
function recent(buf::RingBuffer, w::Int)
    n = min(w, buf.count)
    if n == 0
        return Int8[]
    end
    cap = capacity(buf)
    result = Vector{Int8}(undef, n)
    # head-1 is the last written position; go back n entries
    for k in 1:n
        idx = mod1(buf.head - n + k - 1, cap)
        result[k] = buf.data[idx]
    end
    return result
end

"""
    count_strategy_A(buf::RingBuffer, w::Int)

Count STRATEGY_A entries in the last `min(w, count)` entries, without allocating.
"""
function count_strategy_A(buf::RingBuffer, w::Int)
    n = min(w, buf.count)
    if n == 0
        return 0, 0  # (n_A, total)
    end
    cap = capacity(buf)
    n_A = 0
    for k in 1:n
        idx = mod1(buf.head - n + k - 1, cap)
        if buf.data[idx] == STRATEGY_A
            n_A += 1
        end
    end
    return n_A, n
end

# ──────────────────────────────────────────────────────────────
# AgentState — fully concrete, no Union types
# ──────────────────────────────────────────────────────────────

mutable struct AgentState
    # Experience memory
    fifo::RingBuffer          # capacity = w_max
    b_exp_A::Float64          # b_exp[1]; b_exp[2] = 1 - b_exp_A

    # Confidence
    C::Float64                # ∈ [0,1]
    w::Int                    # current window size

    # Normative memory
    r::Int8                   # NO_NORM (-1), STRATEGY_A (0), or STRATEGY_B (1)
    sigma::Float64            # norm strength ∈ [0,1]; 0.0 when no norm
    a::Int                    # anomaly counter
    e::Float64                # DDM evidence accumulator

    # Enforcement buffer
    pending_signal::Int8      # NO_SIGNAL (-1), or STRATEGY_A/B

    # Derived (recomputed each tick Stage 1)
    compliance::Float64
    b_eff_A::Float64          # b_eff[1]; b_eff[2] = 1 - b_eff_A
end

# ──────────────────────────────────────────────────────────────
# TickWorkspace — pre-allocated, reused every tick (zero GC pressure)
# ──────────────────────────────────────────────────────────────

mutable struct TickWorkspace
    # Stage 1: pairs (flat arrays, N/2 pairs)
    pair_i::Vector{Int}              # length N/2, agent i indices
    pair_j::Vector{Int}              # length N/2, agent j indices
    perm::Vector{Int}                # length N, reusable permutation buffer

    # Stage 1: actions & predictions (indexed by agent id 1:N)
    action::Vector{Int8}             # length N
    prediction::Vector{Int8}         # length N
    coordinated::Vector{Bool}        # length N/2 (per pair)

    # Stage 2: partner lookup (indexed by agent id)
    partner_id::Vector{Int}          # length N
    partner_action::Vector{Int8}     # length N

    # Stage 2: V observations (flat pool + per-agent offsets)
    obs_pool::Vector{Int8}           # pre-allocated max size N*(1+V)
    obs_offset::Vector{Int}          # length N+1: agent i's obs = obs_pool[obs_offset[i]:obs_offset[i+1]-1]

    # Stage 4: enforcement intents
    enforce_target::Vector{Int}      # length N (0 = no enforcement from this agent)
    enforce_strategy::Vector{Int8}   # length N

    # Stage 5 counter
    num_enforcements::Int

    # Eligible interaction indices for V sampling (reusable buffer)
    eligible_buf::Vector{Int}        # length N/2
end

# ──────────────────────────────────────────────────────────────
# TickMetrics — immutable record stored in history
# ──────────────────────────────────────────────────────────────

struct TickMetrics
    tick::Int
    fraction_A::Float64
    mean_confidence::Float64
    coordination_rate::Float64
    num_crystallised::Int
    mean_norm_strength::Float64
    num_enforcements::Int
    belief_error::Float64
    belief_variance::Float64
    frac_dominant_norm::Float64
    convergence_counter::Int
end

# ──────────────────────────────────────────────────────────────
# SimulationResult — returned from run!
# ──────────────────────────────────────────────────────────────

struct SimulationResult
    params::Any              # SimulationParams (forward ref resolved by module)
    history::Vector{TickMetrics}
    probes::Any              # ProbeSet (forward ref resolved by module)
    tick_count::Int
end

# ──────────────────────────────────────────────────────────────
# TrialSummary — compact summary for sweep results
# ──────────────────────────────────────────────────────────────

struct TrialSummary
    convergence_tick::Int    # 0 if no convergence
    converged::Bool
    final_fraction_A::Float64
    final_mean_confidence::Float64
    final_num_crystallised::Int
    final_mean_norm_strength::Float64
    final_frac_dominant_norm::Float64
    total_ticks::Int
end
