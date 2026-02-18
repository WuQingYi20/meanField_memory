# ──────────────────────────────────────────────────────────────
# SimulationParams — all model parameters with defaults
# ──────────────────────────────────────────────────────────────

Base.@kwdef struct SimulationParams
    # Experience layer
    N::Int = 100
    T::Int = 1000
    seed::Union{Int,Nothing} = nothing
    alpha::Float64 = 0.1
    beta::Float64 = 0.3
    C0::Float64 = 0.5
    w_base::Int = 2
    w_max::Int = 6

    # Normative layer
    enable_normative::Bool = false
    sigma_noise::Float64 = 0.1
    theta_crystal::Float64 = 3.0
    sigma_0::Float64 = 0.8
    theta_crisis::Int = 10
    lambda_crisis::Float64 = 0.3
    sigma_min::Float64 = 0.1
    alpha_sigma::Float64 = 0.005
    theta_enforce::Float64 = 0.7
    k::Float64 = 2.0
    gamma_signal::Float64 = 2.0

    # Environment
    V::Int = 0
    Phi::Float64 = 0.0

    # Convergence
    convergence_threshold::Float64 = 0.95
    convergence_window::Int = 50
end

"""
    validate(params::SimulationParams)

Validate parameter constraints. Throws `ArgumentError` on violation.
"""
function validate(params::SimulationParams)
    params.N >= 2 || throw(ArgumentError("N must be ≥ 2, got $(params.N)"))
    iseven(params.N) || throw(ArgumentError("N must be even, got $(params.N)"))
    params.T >= 1 || throw(ArgumentError("T must be ≥ 1, got $(params.T)"))

    0 < params.alpha < 1 || throw(ArgumentError("alpha must be in (0,1), got $(params.alpha)"))
    0 < params.beta < 1 || throw(ArgumentError("beta must be in (0,1), got $(params.beta)"))
    params.beta > params.alpha || throw(ArgumentError("beta must be > alpha: beta=$(params.beta), alpha=$(params.alpha)"))

    0 <= params.C0 <= 1 || throw(ArgumentError("C0 must be in [0,1], got $(params.C0)"))

    params.w_base >= 1 || throw(ArgumentError("w_base must be ≥ 1, got $(params.w_base)"))
    params.w_max >= params.w_base || throw(ArgumentError("w_max must be ≥ w_base: w_max=$(params.w_max), w_base=$(params.w_base)"))

    params.sigma_noise >= 0 || throw(ArgumentError("sigma_noise must be ≥ 0, got $(params.sigma_noise)"))
    params.theta_crystal > 0 || throw(ArgumentError("theta_crystal must be > 0, got $(params.theta_crystal)"))
    0 < params.sigma_0 <= 1 || throw(ArgumentError("sigma_0 must be in (0,1], got $(params.sigma_0)"))
    params.theta_crisis >= 1 || throw(ArgumentError("theta_crisis must be ≥ 1, got $(params.theta_crisis)"))
    0 < params.lambda_crisis < 1 || throw(ArgumentError("lambda_crisis must be in (0,1), got $(params.lambda_crisis)"))
    0 < params.sigma_min < params.sigma_0 || throw(ArgumentError("sigma_min must be in (0, sigma_0): sigma_min=$(params.sigma_min), sigma_0=$(params.sigma_0)"))
    0 < params.alpha_sigma < 1 || throw(ArgumentError("alpha_sigma must be in (0,1), got $(params.alpha_sigma)"))
    0 < params.theta_enforce < 1 || throw(ArgumentError("theta_enforce must be in (0,1), got $(params.theta_enforce)"))
    params.k > 0 || throw(ArgumentError("k must be > 0, got $(params.k)"))
    params.gamma_signal > 0 || throw(ArgumentError("gamma_signal must be > 0, got $(params.gamma_signal)"))

    params.V >= 0 || throw(ArgumentError("V must be ≥ 0, got $(params.V)"))
    params.Phi >= 0 || throw(ArgumentError("Phi must be ≥ 0, got $(params.Phi)"))

    0 < params.convergence_threshold <= 1 || throw(ArgumentError("convergence_threshold must be in (0,1], got $(params.convergence_threshold)"))
    params.convergence_window >= 1 || throw(ArgumentError("convergence_window must be ≥ 1, got $(params.convergence_window)"))

    return nothing
end

"""
    to_namedtuple(params::SimulationParams)

Convert SimulationParams to a NamedTuple for reconstruction with overrides.
"""
function to_namedtuple(params::SimulationParams)
    return (
        N = params.N, T = params.T, seed = params.seed,
        alpha = params.alpha, beta = params.beta, C0 = params.C0,
        w_base = params.w_base, w_max = params.w_max,
        enable_normative = params.enable_normative,
        sigma_noise = params.sigma_noise, theta_crystal = params.theta_crystal,
        sigma_0 = params.sigma_0, theta_crisis = params.theta_crisis,
        lambda_crisis = params.lambda_crisis, sigma_min = params.sigma_min,
        alpha_sigma = params.alpha_sigma, theta_enforce = params.theta_enforce,
        k = params.k, gamma_signal = params.gamma_signal,
        V = params.V, Phi = params.Phi,
        convergence_threshold = params.convergence_threshold,
        convergence_window = params.convergence_window,
    )
end
