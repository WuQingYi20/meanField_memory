# ══════════════════════════════════════════════════════════════
# Lifecycle Benchmark — Model-Agnostic Behavioral Perturbation
# ══════════════════════════════════════════════════════════════

# ── Constants ──

const SYMMETRY_BREAK_THRESHOLD = 0.6
const STEADY_MAJORITY_THRESHOLD = 0.95
const STEADY_WINDOW = 50
const BURST_THRESHOLD = 3   # min delta_crystallised for a cascade burst

# ══════════════════════════════════════════════════════════════
# override_actions! — model-agnostic action forcing
# ══════════════════════════════════════════════════════════════

"""
    override_actions!(ws, N, forced_ids, forced_strategy)

Force selected agents to play `forced_strategy`, then recompute pair coordination.
Works on both `TickWorkspace` and `EWAWorkspace` (same field names).
No-op when `forced_ids` is empty.
"""
function override_actions!(ws, N::Int, forced_ids::Vector{Int}, forced_strategy::Int8)
    isempty(forced_ids) && return nothing

    # Override actions
    for i in forced_ids
        ws.action[i] = forced_strategy
    end

    # Recompute coordination for all pairs
    n_pairs = N ÷ 2
    for k in 1:n_pairs
        ws.coordinated[k] = ws.action[ws.pair_i[k]] == ws.action[ws.pair_j[k]]
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# LifecycleMetrics — immutable result record (12 fields)
# ══════════════════════════════════════════════════════════════

struct LifecycleMetrics
    # Emergence
    symmetry_break_tick::Int              # frac_A first leaves [0.4, 0.6]
    agent_divergence_at_break::Float64    # std of beliefs at break tick
    first_mover_mean_C::Float64           # mean C of majority-choosers at break (NaN for EWA)

    # Diffusion
    diffusion_duration::Int               # ticks from 60% to 95% majority
    crystal_cascade_bursts::Int           # ticks with delta_crystallised >= 3 (0 for non-DM_full)

    # Steady state
    steady_state_tick::Int                # first tick of 50-consecutive-tick 95% window
    enforcement_at_steady::Float64        # mean num_enforcements over that 50-tick window

    # Perturbation recovery
    recovery_ticks::Int                   # ticks from end-of-forcing to re-achieve 95%
    perturbation_depth::Float64           # min(majority) during+after forcing
    norm_survival_rate::Float64           # forced crystallised agents still crystallised after recovery (NaN for non-DM_full)
    recrystallisation_rate::Float64       # forced agents who dissolved then re-crystallised (NaN for non-DM_full)

    # Extreme perturbation (50%)
    norm_flip::Bool                       # dominant strategy flipped after 50% shock
end

# ══════════════════════════════════════════════════════════════
# LifecycleTracker — mutable accumulator updated each tick
# ══════════════════════════════════════════════════════════════

mutable struct LifecycleTracker
    # Emergence detection
    symmetry_break_tick::Int
    agent_divergence_at_break::Float64
    first_mover_mean_C::Float64

    # Diffusion tracking
    first_60_tick::Int                    # first tick majority >= 0.6
    first_95_tick::Int                    # first tick majority >= 0.95
    crystal_cascade_bursts::Int
    prev_num_crystallised::Int

    # Steady state tracking
    steady_counter::Int                   # consecutive ticks with majority >= 0.95
    steady_state_tick::Int                # first tick of 50-consecutive window
    enforcement_accumulator::Float64      # sum of enforcements during steady window
    enforcement_window_ticks::Int         # how many ticks counted in steady window

    # Pre-perturbation dominant strategy
    dominant_strategy_at_steady::Int8
end

function LifecycleTracker()
    return LifecycleTracker(
        0, NaN, NaN,             # emergence
        0, 0, 0, 0,             # diffusion
        0, 0, 0.0, 0,           # steady state
        NO_NORM,                 # dominant strategy
    )
end

# ══════════════════════════════════════════════════════════════
# update_tracker! — called each tick during pre-perturbation phase
# ══════════════════════════════════════════════════════════════

"""
    update_tracker!(tracker, metrics, agents, ws, model_type, is_normative)

Update lifecycle tracker with tick-level data. `model_type` is :dm or :ewa.
`is_normative` is true only for DM_full (enable_normative=true).
"""
function update_tracker!(tracker::LifecycleTracker, metrics::TickMetrics,
                          agents, ws, model_type::Symbol, is_normative::Bool)
    N = length(ws.action)
    fraction_A = metrics.fraction_A
    majority = max(fraction_A, 1.0 - fraction_A)

    # ── Symmetry break detection ──
    if tracker.symmetry_break_tick == 0 && majority >= SYMMETRY_BREAK_THRESHOLD
        tracker.symmetry_break_tick = metrics.tick

        # Snapshot std of beliefs at break
        if model_type == :dm
            beliefs = Float64[agents[i].b_exp_A for i in 1:N]
            tracker.agent_divergence_at_break = std(beliefs)

            # Mean C of majority-choosers
            dominant = fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
            sum_C = 0.0
            n_majority = 0
            for i in 1:N
                if ws.action[i] == dominant
                    sum_C += agents[i].C
                    n_majority += 1
                end
            end
            tracker.first_mover_mean_C = n_majority > 0 ? sum_C / n_majority : NaN
        else
            # EWA: use prob_A as belief proxy
            beliefs = Float64[agents[i].prob_A for i in 1:N]
            tracker.agent_divergence_at_break = std(beliefs)
            tracker.first_mover_mean_C = NaN  # no confidence in EWA
        end
    end

    # ── Diffusion tracking ──
    if tracker.first_60_tick == 0 && majority >= 0.6
        tracker.first_60_tick = metrics.tick
    end
    if tracker.first_95_tick == 0 && majority >= STEADY_MAJORITY_THRESHOLD
        tracker.first_95_tick = metrics.tick
    end

    # Crystal cascade bursts (DM_full only)
    if is_normative
        delta_cryst = metrics.num_crystallised - tracker.prev_num_crystallised
        if delta_cryst >= BURST_THRESHOLD
            tracker.crystal_cascade_bursts += 1
        end
        tracker.prev_num_crystallised = metrics.num_crystallised
    end

    # ── Steady-state tracking (behavioral-only: majority >= 0.95) ──
    if majority >= STEADY_MAJORITY_THRESHOLD
        tracker.steady_counter += 1
        if tracker.steady_state_tick == 0 && tracker.steady_counter >= STEADY_WINDOW
            # The window started STEADY_WINDOW ticks ago
            tracker.steady_state_tick = metrics.tick - STEADY_WINDOW + 1
        end
        # Accumulate enforcement for steady window reporting
        if tracker.steady_state_tick > 0 && tracker.enforcement_window_ticks < STEADY_WINDOW
            tracker.enforcement_accumulator += metrics.num_enforcements
            tracker.enforcement_window_ticks += 1
        end
    else
        tracker.steady_counter = 0
    end

    # Track dominant strategy
    if majority >= STEADY_MAJORITY_THRESHOLD
        tracker.dominant_strategy_at_steady = fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════
# Crystallisation snapshot helpers (DM_full only)
# ══════════════════════════════════════════════════════════════

"""
    snapshot_crystallised(agents::Vector{AgentState})

Return a Dict mapping agent_id → norm direction for agents with r != NO_NORM.
"""
function snapshot_crystallised(agents::Vector{AgentState})
    snap = Dict{Int,Int8}()
    for i in eachindex(agents)
        if agents[i].r != NO_NORM
            snap[i] = agents[i].r
        end
    end
    return snap
end

"""
    compute_norm_metrics(agents::Vector{AgentState}, forced_ids::Vector{Int},
                         pre_crystal_snapshot::Dict{Int,Int8})

Compute (survival_rate, recrystallisation_rate) for forced agents after recovery.
- survival: fraction of forced agents who were crystallised before AND still crystallised after
- recrystallisation: among forced agents whose norm dissolved during forcing, fraction who re-crystallised
"""
function compute_norm_metrics(agents::Vector{AgentState}, forced_ids::Vector{Int},
                               pre_crystal_snapshot::Dict{Int,Int8})
    n_were_crystal = 0
    n_still_crystal = 0
    n_dissolved = 0
    n_recrystal = 0

    for i in forced_ids
        if haskey(pre_crystal_snapshot, i)
            # This agent was crystallised before forcing
            n_were_crystal += 1
            if agents[i].r != NO_NORM
                n_still_crystal += 1
            else
                # Dissolved — check if they re-crystallised is impossible here
                # since they're currently dissolved. Mark as dissolved.
                n_dissolved += 1
            end
        end
    end

    # Also check for agents who dissolved at some point but re-crystallised
    # We need a different approach: compare current state vs pre-state
    # Actually, we just look at current state:
    # - survival: were crystallised before, still crystallised now
    # - dissolved: were crystallised before, not crystallised now
    # We can't know if they dissolved and re-crystallised without tick-level tracking.
    # So recrystallisation = agents NOT in pre-snapshot who are NOW crystallised
    # Actually per spec: "among forced agents whose norm dissolved during forcing,
    # fraction who re-crystallised after release"
    # Since we only have before/after snapshots, we approximate:
    # recrystallisation = 0 if all survived, else report what we can.
    # Better: survival_rate is straightforward. For recrystallisation, we need
    # to also track mid-forcing state. Let's handle this in the benchmark script
    # by taking a snapshot at end-of-forcing too.

    survival_rate = n_were_crystal > 0 ? n_still_crystal / n_were_crystal : NaN
    recryst_rate = n_dissolved > 0 ? 0.0 : NaN  # placeholder, refined with mid-snapshot

    return (survival_rate, recryst_rate)
end

"""
    compute_norm_metrics(agents::Vector{AgentState}, forced_ids::Vector{Int},
                         pre_crystal_snapshot::Dict{Int,Int8},
                         mid_crystal_snapshot::Dict{Int,Int8})

Extended version with mid-forcing snapshot for recrystallisation tracking.
- survival: forced agents crystallised before AND still crystallised after recovery
- recrystallisation: forced agents crystallised before, dissolved at mid-snapshot,
  but re-crystallised by recovery
"""
function compute_norm_metrics(agents::Vector{AgentState}, forced_ids::Vector{Int},
                               pre_crystal_snapshot::Dict{Int,Int8},
                               mid_crystal_snapshot::Dict{Int,Int8})
    n_were_crystal = 0
    n_still_crystal = 0
    n_dissolved_mid = 0
    n_recrystal = 0

    for i in forced_ids
        if haskey(pre_crystal_snapshot, i)
            n_were_crystal += 1
            was_dissolved_mid = !haskey(mid_crystal_snapshot, i)

            if agents[i].r != NO_NORM
                n_still_crystal += 1
                if was_dissolved_mid
                    n_recrystal += 1
                end
            end
            if was_dissolved_mid
                n_dissolved_mid += 1
            end
        end
    end

    survival_rate = n_were_crystal > 0 ? n_still_crystal / n_were_crystal : NaN
    recryst_rate = n_dissolved_mid > 0 ? n_recrystal / n_dissolved_mid : NaN

    return (survival_rate, recryst_rate)
end

# ══════════════════════════════════════════════════════════════
# finalize_tracker — produce LifecycleMetrics from tracker
# ══════════════════════════════════════════════════════════════

"""
    finalize_tracker(tracker; recovery_ticks=0, perturbation_depth=1.0,
                     norm_survival_rate=NaN, recrystallisation_rate=NaN, norm_flip=false)

Convert a LifecycleTracker into an immutable LifecycleMetrics.
Perturbation-related fields are passed in from the benchmark script.
"""
function finalize_tracker(tracker::LifecycleTracker;
                           recovery_ticks::Int=0,
                           perturbation_depth::Float64=1.0,
                           norm_survival_rate::Float64=NaN,
                           recrystallisation_rate::Float64=NaN,
                           norm_flip::Bool=false)
    diffusion = if tracker.first_60_tick > 0 && tracker.first_95_tick > 0
        tracker.first_95_tick - tracker.first_60_tick
    else
        0
    end

    enforcement_mean = if tracker.enforcement_window_ticks > 0
        tracker.enforcement_accumulator / tracker.enforcement_window_ticks
    else
        0.0
    end

    return LifecycleMetrics(
        # Emergence
        tracker.symmetry_break_tick,
        tracker.agent_divergence_at_break,
        tracker.first_mover_mean_C,
        # Diffusion
        diffusion,
        tracker.crystal_cascade_bursts,
        # Steady state
        tracker.steady_state_tick,
        enforcement_mean,
        # Perturbation recovery (filled in by caller)
        recovery_ticks,
        perturbation_depth,
        norm_survival_rate,
        recrystallisation_rate,
        # Extreme perturbation
        norm_flip,
    )
end
