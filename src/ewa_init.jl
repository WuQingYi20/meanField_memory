# ──────────────────────────────────────────────────────────────
# EWA Initialization
# ──────────────────────────────────────────────────────────────

"""
    ewa_create_workspace(params::EWAParams)

Pre-allocate the EWAWorkspace with all arrays sized for the given parameters.
"""
function ewa_create_workspace(params::EWAParams)
    N = params.N
    n_pairs = N ÷ 2

    return EWAWorkspace(
        Vector{Int}(undef, n_pairs),       # pair_i
        Vector{Int}(undef, n_pairs),       # pair_j
        Vector{Int}(undef, N),             # perm

        Vector{Int8}(undef, N),            # action
        Vector{Bool}(undef, n_pairs),      # coordinated

        Vector{Int}(undef, N),             # partner_id
        Vector{Int8}(undef, N),            # partner_action
    )
end

"""
    ewa_initialize(params::EWAParams)

Create agents, workspace, history, and RNG. Returns (agents, ws, history, rng).
"""
function ewa_initialize(params::EWAParams)
    validate(params)

    rng = isnothing(params.seed) ? MersenneTwister() : MersenneTwister(params.seed)

    agents = Vector{EWAAgent}(undef, params.N)
    for i in 1:params.N
        agents[i] = EWAAgent(
            params.A0,     # attract_A
            params.A0,     # attract_B
            1.0,           # N_exp (initial experience weight)
            0.5,           # prob_A (symmetric with equal attractions)
        )
    end

    ws = ewa_create_workspace(params)
    history = Vector{TickMetrics}(undef, params.T)

    return agents, ws, history, rng
end
