# ──────────────────────────────────────────────────────────────
# EWAParams — Experience-Weighted Attraction parameters
# ──────────────────────────────────────────────────────────────

Base.@kwdef struct EWAParams
    # Population
    N::Int = 100
    T::Int = 3000
    seed::Union{Int,Nothing} = nothing

    # EWA core (Camerer & Ho 1999)
    delta::Float64 = 0.5      # imagination weight: 0=RL, 1=belief learning
    phi::Float64 = 0.9        # attraction decay
    rho::Float64 = 0.9        # experience weight decay
    lambda::Float64 = 1.0     # logit sensitivity

    # Initial attractions (symmetric)
    A0::Float64 = 0.0         # attract_A = attract_B = A0 → prob_A = 0.5

    # Convergence thresholds (behavioral + belief only, no crystallization)
    thresh_majority::Float64 = 0.95
    thresh_belief_error::Float64 = 0.10
    thresh_belief_var::Float64 = 0.05
    convergence_window::Int = 50
end

"""
    validate(params::EWAParams)

Validate EWA parameter constraints. Throws `ArgumentError` on violation.
"""
function validate(params::EWAParams)
    params.N >= 2 || throw(ArgumentError("N must be >= 2, got $(params.N)"))
    iseven(params.N) || throw(ArgumentError("N must be even, got $(params.N)"))
    params.T >= 1 || throw(ArgumentError("T must be >= 1, got $(params.T)"))

    0 <= params.delta <= 1 || throw(ArgumentError("delta must be in [0,1], got $(params.delta)"))
    0 <= params.phi <= 1 || throw(ArgumentError("phi must be in [0,1], got $(params.phi)"))
    0 <= params.rho <= 1 || throw(ArgumentError("rho must be in [0,1], got $(params.rho)"))
    params.lambda > 0 || throw(ArgumentError("lambda must be > 0, got $(params.lambda)"))

    0 < params.thresh_majority <= 1 || throw(ArgumentError("thresh_majority must be in (0,1], got $(params.thresh_majority)"))
    0 < params.thresh_belief_error <= 1 || throw(ArgumentError("thresh_belief_error must be in (0,1], got $(params.thresh_belief_error)"))
    0 < params.thresh_belief_var <= 1 || throw(ArgumentError("thresh_belief_var must be in (0,1], got $(params.thresh_belief_var)"))
    params.convergence_window >= 1 || throw(ArgumentError("convergence_window must be >= 1, got $(params.convergence_window)"))

    return nothing
end

"""
    to_namedtuple(params::EWAParams)

Convert EWAParams to a NamedTuple for reconstruction with overrides.
"""
function to_namedtuple(params::EWAParams)
    return (
        N = params.N, T = params.T, seed = params.seed,
        delta = params.delta, phi = params.phi, rho = params.rho,
        lambda = params.lambda, A0 = params.A0,
        thresh_majority = params.thresh_majority,
        thresh_belief_error = params.thresh_belief_error,
        thresh_belief_var = params.thresh_belief_var,
        convergence_window = params.convergence_window,
    )
end

# ──────────────────────────────────────────────────────────────
# EWAAgent — single agent state (4 fields)
# ──────────────────────────────────────────────────────────────

mutable struct EWAAgent
    attract_A::Float64     # attraction for strategy A
    attract_B::Float64     # attraction for strategy B
    N_exp::Float64         # experience weight
    prob_A::Float64        # logit probability of choosing A
end

# ──────────────────────────────────────────────────────────────
# EWAWorkspace — pre-allocated arrays reused every tick
# ──────────────────────────────────────────────────────────────

mutable struct EWAWorkspace
    # Pairing (same layout as TickWorkspace)
    pair_i::Vector{Int}
    pair_j::Vector{Int}
    perm::Vector{Int}

    # Actions & coordination
    action::Vector{Int8}
    coordinated::Vector{Bool}

    # Partner lookup
    partner_id::Vector{Int}
    partner_action::Vector{Int8}
end
