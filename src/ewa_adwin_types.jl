# ──────────────────────────────────────────────────────────────
# EWAADWINParams — EWA with ADWIN2 adaptive windowing
# ──────────────────────────────────────────────────────────────

Base.@kwdef struct EWAADWINParams
    # Population
    N::Int = 100
    T::Int = 3000
    seed::Union{Int,Nothing} = nothing

    # ADWIN parameters
    adwin_delta::Float64 = 0.01   # confidence (lower = less sensitive to change)
    adwin_M::Int = 5              # max buckets per level

    # EWA core (phi/rho removed — ADWIN replaces exponential decay)
    delta::Float64 = 0.5          # imagination weight: 0=RL, 1=belief learning
    lambda::Float64 = 1.0         # logit sensitivity
    A0::Float64 = 0.0             # initial attraction

    # Convergence thresholds (behavioral + belief only, no crystallization)
    thresh_majority::Float64 = 0.95
    thresh_belief_error::Float64 = 0.10
    thresh_belief_var::Float64 = 0.05
    convergence_window::Int = 50
end

"""
    validate(params::EWAADWINParams)

Validate EWA-ADWIN parameter constraints. Throws `ArgumentError` on violation.
"""
function validate(params::EWAADWINParams)
    params.N >= 2 || throw(ArgumentError("N must be >= 2, got $(params.N)"))
    iseven(params.N) || throw(ArgumentError("N must be even, got $(params.N)"))
    params.T >= 1 || throw(ArgumentError("T must be >= 1, got $(params.T)"))

    0 < params.adwin_delta <= 1 || throw(ArgumentError("adwin_delta must be in (0,1], got $(params.adwin_delta)"))
    params.adwin_M >= 1 || throw(ArgumentError("adwin_M must be >= 1, got $(params.adwin_M)"))

    0 <= params.delta <= 1 || throw(ArgumentError("delta must be in [0,1], got $(params.delta)"))
    params.lambda > 0 || throw(ArgumentError("lambda must be > 0, got $(params.lambda)"))

    0 < params.thresh_majority <= 1 || throw(ArgumentError("thresh_majority must be in (0,1], got $(params.thresh_majority)"))
    0 < params.thresh_belief_error <= 1 || throw(ArgumentError("thresh_belief_error must be in (0,1], got $(params.thresh_belief_error)"))
    0 < params.thresh_belief_var <= 1 || throw(ArgumentError("thresh_belief_var must be in (0,1], got $(params.thresh_belief_var)"))
    params.convergence_window >= 1 || throw(ArgumentError("convergence_window must be >= 1, got $(params.convergence_window)"))

    return nothing
end

"""
    to_namedtuple(params::EWAADWINParams)

Convert EWAADWINParams to a NamedTuple for reconstruction with overrides.
"""
function to_namedtuple(params::EWAADWINParams)
    return (
        N = params.N, T = params.T, seed = params.seed,
        adwin_delta = params.adwin_delta, adwin_M = params.adwin_M,
        delta = params.delta, lambda = params.lambda, A0 = params.A0,
        thresh_majority = params.thresh_majority,
        thresh_belief_error = params.thresh_belief_error,
        thresh_belief_var = params.thresh_belief_var,
        convergence_window = params.convergence_window,
    )
end

# ──────────────────────────────────────────────────────────────
# EWAADWINAgent — single agent state with ADWIN2 instances
# ──────────────────────────────────────────────────────────────

mutable struct EWAADWINAgent
    adwin_A::ADWIN2       # adaptive window for strategy A reinforcements
    adwin_B::ADWIN2       # adaptive window for strategy B reinforcements
    attract_A::Float64    # = adwin_mean(adwin_A)
    attract_B::Float64    # = adwin_mean(adwin_B)
    prob_A::Float64       # logit probability
end
