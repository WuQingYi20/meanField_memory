# ──────────────────────────────────────────────────────────────
# Level milestone extraction
# ──────────────────────────────────────────────────────────────

"""
    first_tick_per_layer(history::Vector{TickMetrics}, N::Int, params::SimulationParams)

Scan history and return a NamedTuple with the first tick each layer threshold
was met, plus when all layers were simultaneously met. 0 = never reached.
"""
function first_tick_per_layer(history::Vector{TickMetrics}, N::Int, params::SimulationParams)
    t_behavioral = 0
    t_belief = 0
    t_crystal = 0
    t_all_met = 0

    for m in history
        majority = max(m.fraction_A, 1.0 - m.fraction_A)
        frac_cryst = m.num_crystallised / N

        if t_behavioral == 0 && majority >= params.thresh_majority
            t_behavioral = m.tick
        end
        if t_belief == 0 && m.belief_error < params.thresh_belief_error && m.belief_variance < params.thresh_belief_var
            t_belief = m.tick
        end
        if t_crystal == 0 && frac_cryst >= params.thresh_crystallised && m.frac_dominant_norm >= params.thresh_dominant_norm
            t_crystal = m.tick
        end
        if t_all_met == 0 && all_layers_met(m.fraction_A, m.belief_error, m.belief_variance,
                                             frac_cryst, m.frac_dominant_norm, params)
            t_all_met = m.tick
        end
    end

    return (behavioral=t_behavioral, belief=t_belief, crystal=t_crystal, all_met=t_all_met)
end

# ──────────────────────────────────────────────────────────────
# Sweep Configuration
# ──────────────────────────────────────────────────────────────

struct SweepConfig
    base_params::SimulationParams
    V_range::Vector{Int}
    Phi_range::Vector{Float64}
    n_trials::Int
    base_seed::Int
end

# ──────────────────────────────────────────────────────────────
# Sweep Result
# ──────────────────────────────────────────────────────────────

struct SweepResult
    V_vals::Vector{Int}
    Phi_vals::Vector{Float64}
    results::Array{TrialSummary, 3}  # [V_idx, Phi_idx, trial_idx]
end

# ──────────────────────────────────────────────────────────────
# Summarize a single simulation result
# ──────────────────────────────────────────────────────────────

"""
    summarize(result::SimulationResult)

Extract a compact TrialSummary from a full simulation result.
"""
function summarize(result::SimulationResult)
    history = result.history
    tc = result.tick_count
    params = result.params

    if tc == 0
        return TrialSummary(0, false, 0.5, 0.5, 0, 0.0, 0.0, 0)
    end

    last_m = history[tc]

    # Check convergence: convergence_counter >= convergence_window
    converged = last_m.convergence_counter >= params.convergence_window

    # Find convergence tick (first tick convergence_counter reached window)
    conv_tick = 0
    if converged
        for i in 1:tc
            if history[i].convergence_counter >= params.convergence_window
                conv_tick = history[i].tick
                break
            end
        end
    end

    return TrialSummary(
        conv_tick,
        converged,
        last_m.fraction_A,
        last_m.mean_confidence,
        last_m.num_crystallised,
        last_m.mean_norm_strength,
        last_m.frac_dominant_norm,
        tc,
    )
end

# ──────────────────────────────────────────────────────────────
# Sweep execution
# ──────────────────────────────────────────────────────────────

"""
    sweep(config::SweepConfig; use_threads=true)

Run a parameter grid search over V × Φ × trials.
Each (V, Φ, trial) combination runs with an independent seed.
Thread-safe: each run! creates its own agents, workspace, history, rng.
"""
function sweep(config::SweepConfig; use_threads::Bool=true)
    nV = length(config.V_range)
    nP = length(config.Phi_range)
    nT = config.n_trials

    results = Array{TrialSummary}(undef, nV, nP, nT)

    # Build flat task list
    tasks = [(iv, ip, it) for iv in 1:nV for ip in 1:nP for it in 1:nT]

    base_nt = to_namedtuple(config.base_params)

    if use_threads
        Threads.@threads for idx in eachindex(tasks)
            iv, ip, it = tasks[idx]
            seed = hash((config.base_seed, iv, ip, it)) % typemax(Int)
            p = SimulationParams(; base_nt...,
                V = config.V_range[iv],
                Phi = config.Phi_range[ip],
                seed = Int(seed),
            )
            result = run!(p)
            results[iv, ip, it] = summarize(result)
        end
    else
        for idx in eachindex(tasks)
            iv, ip, it = tasks[idx]
            seed = hash((config.base_seed, iv, ip, it)) % typemax(Int)
            p = SimulationParams(; base_nt...,
                V = config.V_range[iv],
                Phi = config.Phi_range[ip],
                seed = Int(seed),
            )
            result = run!(p)
            results[iv, ip, it] = summarize(result)
        end
    end

    return SweepResult(config.V_range, config.Phi_range, results)
end

# ──────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────

"""
    save_sweep_csv(filepath::String, sr::SweepResult)

Save sweep results to CSV. Each row is one trial.
"""
function save_sweep_csv(filepath::String, sr::SweepResult)
    rows = []
    for iv in eachindex(sr.V_vals)
        for ip in eachindex(sr.Phi_vals)
            for it in axes(sr.results, 3)
                s = sr.results[iv, ip, it]
                push!(rows, (
                    V = sr.V_vals[iv],
                    Phi = sr.Phi_vals[ip],
                    trial = it,
                    convergence_tick = s.convergence_tick,
                    converged = s.converged,
                    final_fraction_A = s.final_fraction_A,
                    final_mean_confidence = s.final_mean_confidence,
                    final_num_crystallised = s.final_num_crystallised,
                    final_mean_norm_strength = s.final_mean_norm_strength,
                    final_frac_dominant_norm = s.final_frac_dominant_norm,
                    total_ticks = s.total_ticks,
                ))
            end
        end
    end

    CSV.write(filepath, rows)
    return nothing
end

"""
    save_sweep_jld2(filepath::String, sr::SweepResult)

Save full sweep results to JLD2 for later Julia analysis.
"""
function save_sweep_jld2(filepath::String, sr::SweepResult)
    JLD2.@save filepath sr
    return nothing
end
