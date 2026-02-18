# ──────────────────────────────────────────────────────────────
# NormEvent — crystallization / crisis / dissolution events
# ──────────────────────────────────────────────────────────────

struct NormEvent
    tick::Int
    agent_id::Int
    event_type::Symbol    # :crystallize, :crisis, :dissolve
    details::NTuple{4, Float64}  # (e, sigma_before, sigma_after, norm_direction)
end

# ──────────────────────────────────────────────────────────────
# EventLog — records norm lifecycle events
# ──────────────────────────────────────────────────────────────

mutable struct EventLog
    events::Vector{NormEvent}
    enabled::Bool
end

EventLog(; enabled::Bool=false) = EventLog(NormEvent[], enabled)

# ──────────────────────────────────────────────────────────────
# DDMTracker — evidence walk trajectories
# ──────────────────────────────────────────────────────────────

mutable struct DDMTracker
    tracked_agents::Vector{Int}
    trajectories::Vector{Vector{Float64}}
    enabled::Bool
    n_track::Int
end

DDMTracker(; enabled::Bool=false, n_track::Int=5) =
    DDMTracker(Int[], Vector{Float64}[], enabled, n_track)

# ──────────────────────────────────────────────────────────────
# QuantileTracker — belief distribution snapshots
# ──────────────────────────────────────────────────────────────

mutable struct QuantileTracker
    quantiles::Vector{Float64}
    belief_snapshots::Vector{Vector{Float64}}      # one entry per tick
    confidence_snapshots::Vector{Vector{Float64}}   # one entry per tick
    enabled::Bool
end

QuantileTracker(; enabled::Bool=false,
                  quantiles::Vector{Float64}=[0.1, 0.25, 0.5, 0.75, 0.9]) =
    QuantileTracker(quantiles, Vector{Float64}[], Vector{Float64}[], enabled)

# ──────────────────────────────────────────────────────────────
# ProbeSet — wrapper for all probes
# ──────────────────────────────────────────────────────────────

struct ProbeSet
    event_log::EventLog
    ddm_tracker::DDMTracker
    quantile_tracker::QuantileTracker
end

"""
    ProbeSet(; event_log=false, ddm_tracker=false, ddm_n_track=5, quantile_tracker=false, quantiles=...)

Create a ProbeSet with selectively enabled probes.
"""
function ProbeSet(; event_log::Bool=false, ddm_tracker::Bool=false, ddm_n_track::Int=5,
                    quantile_tracker::Bool=false,
                    quantiles::Vector{Float64}=[0.1, 0.25, 0.5, 0.75, 0.9])
    return ProbeSet(
        EventLog(; enabled=event_log),
        DDMTracker(; enabled=ddm_tracker, n_track=ddm_n_track),
        QuantileTracker(; enabled=quantile_tracker, quantiles=quantiles),
    )
end

# ──────────────────────────────────────────────────────────────
# Probe initialization (called once at start of run!)
# ──────────────────────────────────────────────────────────────

"""
    init_probes!(probes::ProbeSet, params::SimulationParams, rng::AbstractRNG)

Initialize probe state that depends on params (e.g., select tracked agents).
"""
function init_probes!(probes::ProbeSet, params::SimulationParams, rng::AbstractRNG)
    if probes.ddm_tracker.enabled
        n = min(probes.ddm_tracker.n_track, params.N)
        probes.ddm_tracker.tracked_agents = sort(sample(rng, 1:params.N, n; replace=false))
        probes.ddm_tracker.trajectories = [Float64[] for _ in 1:n]
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────
# record_probes! — called at end of each tick
# ──────────────────────────────────────────────────────────────

"""
    record_probes!(probes::ProbeSet, t, agents, ws, metrics)

Record probe data for the current tick. No-ops for disabled probes.
"""
function record_probes!(probes::ProbeSet, t::Int, agents::Vector{AgentState},
                         ws::TickWorkspace, metrics::TickMetrics)
    # DDM Tracker: record evidence values for tracked agents
    if probes.ddm_tracker.enabled
        for (idx, agent_id) in enumerate(probes.ddm_tracker.tracked_agents)
            push!(probes.ddm_tracker.trajectories[idx], agents[agent_id].e)
        end
    end

    # Quantile Tracker: record belief and confidence quantiles
    if probes.quantile_tracker.enabled
        N = length(agents)
        beliefs = Vector{Float64}(undef, N)
        confidences = Vector{Float64}(undef, N)
        for i in 1:N
            beliefs[i] = agents[i].b_eff_A
            confidences[i] = agents[i].C
        end
        push!(probes.quantile_tracker.belief_snapshots,
              [quantile(beliefs, q) for q in probes.quantile_tracker.quantiles])
        push!(probes.quantile_tracker.confidence_snapshots,
              [quantile(confidences, q) for q in probes.quantile_tracker.quantiles])
    end

    return nothing
end

# ──────────────────────────────────────────────────────────────
# Event recording helpers (called from stages.jl via probes)
# ──────────────────────────────────────────────────────────────

"""
    record_event!(log::EventLog, tick, agent_id, event_type, details)

Record a norm lifecycle event if the event log is enabled.
"""
function record_event!(log::EventLog, tick::Int, agent_id::Int,
                        event_type::Symbol, details::NTuple{4,Float64})
    if log.enabled
        push!(log.events, NormEvent(tick, agent_id, event_type, details))
    end
    return nothing
end
