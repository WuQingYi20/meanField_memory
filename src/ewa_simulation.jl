# ──────────────────────────────────────────────────────────────
# EWA Simulation Driver
# ──────────────────────────────────────────────────────────────

"""
    ewa_check_convergence(history, tick_count, params::EWAParams)

Check if EWA convergence has been achieved (behavioral + belief met for convergence_window ticks).
"""
function ewa_check_convergence(history::Vector{TickMetrics}, tick_count::Int,
                                params::EWAParams)::Bool
    tick_count >= 1 || return false
    return history[tick_count].convergence_counter >= params.convergence_window
end

"""
    ewa_run_tick!(t, agents, ws, history, tick_count, params, rng)

Execute one complete EWA tick: pair+act → update attractions → metrics.
"""
function ewa_run_tick!(t::Int, agents::Vector{EWAAgent}, ws::EWAWorkspace,
                       history::Vector{TickMetrics}, tick_count::Int,
                       params::EWAParams, rng::AbstractRNG)
    ewa_pair_and_act!(agents, ws, params, rng)
    ewa_update_attractions!(agents, ws, params)
    metrics = ewa_metrics(agents, ws, t, history, tick_count, params)
    history[tick_count + 1] = metrics
    return metrics
end

"""
    ewa_run!(params::EWAParams)

Run the complete EWA simulation. Returns a SimulationResult.
"""
function ewa_run!(params::EWAParams)
    agents, ws, history, rng = ewa_initialize(params)
    tick_count = 0

    for t in 1:params.T
        ewa_run_tick!(t, agents, ws, history, tick_count, params, rng)
        tick_count += 1

        if ewa_check_convergence(history, tick_count, params)
            break
        end
    end

    return SimulationResult(params, history[1:tick_count], nothing, tick_count)
end

# ──────────────────────────────────────────────────────────────
# Level milestone extraction (EWA version)
# ──────────────────────────────────────────────────────────────

"""
    ewa_first_tick_per_layer(history::Vector{TickMetrics}, N::Int, params::EWAParams)

Scan history and return a NamedTuple with the first tick each layer threshold
was met, plus when all layers were simultaneously met.
crystal is always 0 (EWA has no normative memory).
"""
function ewa_first_tick_per_layer(history::Vector{TickMetrics}, N::Int, params::EWAParams)
    t_behavioral = 0
    t_belief = 0
    t_all_met = 0

    for m in history
        majority = max(m.fraction_A, 1.0 - m.fraction_A)

        if t_behavioral == 0 && majority >= params.thresh_majority
            t_behavioral = m.tick
        end
        if t_belief == 0 && m.belief_error < params.thresh_belief_error && m.belief_variance < params.thresh_belief_var
            t_belief = m.tick
        end
        if t_all_met == 0 && majority >= params.thresh_majority &&
           m.belief_error < params.thresh_belief_error &&
           m.belief_variance < params.thresh_belief_var
            t_all_met = m.tick
        end
    end

    return (behavioral=t_behavioral, belief=t_belief, crystal=0, all_met=t_all_met)
end
