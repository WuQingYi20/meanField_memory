"""
    check_convergence(history, tick_count, params)

Check if convergence has been achieved (all 3 layers met for convergence_window ticks).
"""
function check_convergence(history::Vector{TickMetrics}, tick_count::Int,
                            params::SimulationParams)::Bool
    tick_count >= 1 || return false
    return history[tick_count].convergence_counter >= params.convergence_window
end

"""
    run_tick!(t, agents, ws, history, tick_count, params, rng, probes; network=nothing)

Execute one complete tick of the simulation pipeline.
When network is provided (Vector{Vector{Int}}), uses network-constrained pairing.
"""
function run_tick!(t::Int, agents::Vector{AgentState}, ws::TickWorkspace,
                   history::Vector{TickMetrics}, tick_count::Int,
                   params::SimulationParams, rng::AbstractRNG, probes;
                   network::Union{Nothing, Vector{Vector{Int}}}=nothing)
    use_net = network !== nothing

    if use_net
        stage_1_pair_and_act_network!(agents, ws, params, rng, network)
    else
        stage_1_pair_and_act!(agents, ws, params, rng)
    end

    stage_2_observe_and_memory!(agents, ws, params, rng; use_network=use_net)
    stage_3_confidence!(agents, ws, params; use_network=use_net)

    if params.enable_normative
        stage_4_normative!(agents, ws, params, rng; use_network=use_net)
        stage_5_enforce!(agents, ws, params)
    else
        ws.num_enforcements = 0
    end

    metrics = stage_6_metrics(agents, ws, t, history, tick_count, params; use_network=use_net)
    history[tick_count + 1] = metrics  # 1-indexed

    # Fire probes (no-op if disabled)
    record_probes!(probes, t, agents, ws, metrics)

    return metrics
end

"""
    run!(params::SimulationParams; probes::ProbeSet=ProbeSet(), network=nothing)

Run the complete simulation. Returns a SimulationResult.
When network is provided, uses network-constrained pairing each tick.
"""
function run!(params::SimulationParams; probes::ProbeSet=ProbeSet(),
              network::Union{Nothing, Vector{Vector{Int}}}=nothing)
    agents, ws, history, rng = initialize(params)
    tick_count = 0

    # Initialize DDM tracker agent IDs if enabled
    init_probes!(probes, params, rng)

    for t in 1:params.T
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes; network=network)
        tick_count += 1

        # Early termination on convergence
        if check_convergence(history, tick_count, params)
            break
        end
    end

    return SimulationResult(params, history[1:tick_count], probes, tick_count)
end
