"""
    create_workspace(params::SimulationParams)

Pre-allocate the TickWorkspace with all arrays sized for the given parameters.
"""
function create_workspace(params::SimulationParams)
    N = params.N
    n_pairs = N ÷ 2
    max_obs = N * (1 + params.V)

    return TickWorkspace(
        Vector{Int}(undef, n_pairs),       # pair_i
        Vector{Int}(undef, n_pairs),       # pair_j
        Vector{Int}(undef, N),             # perm

        Vector{Int8}(undef, N),            # action
        Vector{Int8}(undef, N),            # prediction
        Vector{Bool}(undef, n_pairs),      # coordinated

        Vector{Int}(undef, N),             # partner_id
        Vector{Int8}(undef, N),            # partner_action

        Vector{Int8}(undef, max_obs),      # obs_pool
        Vector{Int}(undef, N + 1),         # obs_offset

        zeros(Int, N),                     # enforce_target
        zeros(Int8, N),                    # enforce_strategy

        0,                                 # num_enforcements

        Vector{Int}(undef, n_pairs),       # eligible_buf
    )
end

"""
    initialize(params::SimulationParams)

Create agents, workspace, history, and RNG. Returns (agents, ws, history, rng).
"""
function initialize(params::SimulationParams)
    validate(params)

    rng = isnothing(params.seed) ? MersenneTwister() : MersenneTwister(params.seed)

    w_init = params.w_base + floor(Int, params.C0 * (params.w_max - params.w_base))

    agents = Vector{AgentState}(undef, params.N)
    for i in 1:params.N
        agents[i] = AgentState(
            RingBuffer(params.w_max),   # fifo (empty)
            0.5,                         # b_exp_A
            params.C0,                   # C
            w_init,                      # w
            NO_NORM,                     # r
            0.0,                         # sigma
            0,                           # a
            0.0,                         # e
            NO_SIGNAL,                   # pending_signal
            0.0,                         # compliance
            0.5,                         # b_eff_A
        )
    end

    ws = create_workspace(params)
    history = Vector{TickMetrics}(undef, params.T)

    return agents, ws, history, rng
end
