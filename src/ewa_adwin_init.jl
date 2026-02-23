# ──────────────────────────────────────────────────────────────
# EWA-ADWIN Initialization
# ──────────────────────────────────────────────────────────────

"""
    ewa_adwin_initialize(params::EWAADWINParams)

Create agents, workspace, history, and RNG. Returns (agents, ws, history, rng).
Reuses EWAWorkspace since the workspace layout is identical.
"""
function ewa_adwin_initialize(params::EWAADWINParams)
    validate(params)

    rng = isnothing(params.seed) ? MersenneTwister() : MersenneTwister(params.seed)

    agents = Vector{EWAADWINAgent}(undef, params.N)
    for i in 1:params.N
        agents[i] = EWAADWINAgent(
            ADWIN2(M=params.adwin_M, delta=params.adwin_delta),
            ADWIN2(M=params.adwin_M, delta=params.adwin_delta),
            params.A0,    # attract_A
            params.A0,    # attract_B
            0.5,          # prob_A (symmetric with equal attractions)
        )
    end

    # Reuse EWAWorkspace — identical field layout needed
    N = params.N
    n_pairs = N ÷ 2
    ws = EWAWorkspace(
        Vector{Int}(undef, n_pairs),       # pair_i
        Vector{Int}(undef, n_pairs),       # pair_j
        Vector{Int}(undef, N),             # perm
        Vector{Int8}(undef, N),            # action
        Vector{Bool}(undef, n_pairs),      # coordinated
        Vector{Int}(undef, N),             # partner_id
        Vector{Int8}(undef, N),            # partner_action
    )

    history = Vector{TickMetrics}(undef, params.T)

    return agents, ws, history, rng
end
