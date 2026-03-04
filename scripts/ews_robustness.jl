#!/usr/bin/env julia
#
# Early Warning Signal Robustness Test
# =====================================
# Runs 30 seeds at N=100 (Dynamic+Norm) and checks whether the
# belief-shift CUSUM changepoint consistently precedes the
# crystallisation-count changepoint (the "19-tick early warning window").

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Statistics, Printf, Random
using DualMemoryABM

# ── Configuration ──
const N_SEEDS = 30
const N_POP   = 100
const T_MAX   = 3000

# ── Custom run that records mean_beff_A per tick ──
function run_with_beff(params::SimulationParams)
    agents, ws, history, rng = DualMemoryABM.initialize(params)
    tick_count = 0

    mean_beff_A = Float64[]
    mean_sigma  = Float64[]
    num_cryst   = Int[]

    for t in 1:params.T
        DualMemoryABM.run_tick!(t, agents, ws, history, tick_count, params, rng,
                                DualMemoryABM.ProbeSet())
        tick_count += 1

        # Compute mean b_eff_A across all agents
        s_beff = 0.0
        s_sigma = 0.0
        nc = 0
        for i in 1:params.N
            s_beff += agents[i].b_eff_A
            if agents[i].r != DualMemoryABM.NO_NORM
                nc += 1
                s_sigma += agents[i].sigma
            end
        end
        push!(mean_beff_A, s_beff / params.N)
        push!(mean_sigma, nc > 0 ? s_sigma / nc : 0.0)
        push!(num_cryst, nc)

        if DualMemoryABM.check_convergence(history, tick_count, params)
            break
        end
    end

    return (mean_beff_A=mean_beff_A, mean_sigma=mean_sigma,
            num_cryst=num_cryst, tick_count=tick_count)
end

# ── CUSUM changepoint (same method as paper) ──
function cusum_changepoint(vals::Vector{Float64})
    mu = mean(vals)
    cusum = cumsum(vals .- mu)
    cp_idx = argmax(abs.(cusum))
    return cp_idx  # tick index (1-based)
end

# ── Main ──
println("=" ^ 70)
println("Early Warning Signal Robustness: $N_SEEDS seeds, N=$N_POP")
println("=" ^ 70)

results = []

for seed in 1:N_SEEDS
    params = SimulationParams(
        N = N_POP,
        T = T_MAX,
        seed = seed,
        enable_normative = true,
    )

    data = run_with_beff(params)
    T_actual = data.tick_count

    if T_actual < 20
        println(@sprintf("Seed %2d: too short (%d ticks), skipping", seed, T_actual))
        continue
    end

    # CUSUM on each series
    cp_belief = cusum_changepoint(data.mean_beff_A)
    cp_sigma  = cusum_changepoint(data.mean_sigma)
    cp_cryst  = cusum_changepoint(Float64.(data.num_cryst))

    lead_time = cp_cryst - cp_belief  # positive = belief leads

    push!(results, (seed=seed, T=T_actual,
                    cp_belief=cp_belief, cp_sigma=cp_sigma, cp_cryst=cp_cryst,
                    lead_time=lead_time))

    println(@sprintf("Seed %2d: T=%4d | belief_cp=%4d  sigma_cp=%4d  cryst_cp=%4d | lead=%+3d",
                     seed, T_actual, cp_belief, cp_sigma, cp_cryst, lead_time))
end

# ── Summary statistics ──
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

lead_times = [r.lead_time for r in results]
positive   = count(x -> x > 0, lead_times)
zero_      = count(x -> x == 0, lead_times)
negative   = count(x -> x < 0, lead_times)

println(@sprintf("Seeds analysed:      %d / %d", length(results), N_SEEDS))
println(@sprintf("Lead time (belief before cryst):"))
println(@sprintf("  Mean:    %+.1f ticks", mean(lead_times)))
println(@sprintf("  Median:  %+.1f ticks", median(lead_times)))
println(@sprintf("  Std:      %.1f ticks", std(lead_times)))
println(@sprintf("  Min:     %+d ticks", minimum(lead_times)))
println(@sprintf("  Max:     %+d ticks", maximum(lead_times)))
println(@sprintf("  Positive (belief leads): %d / %d  (%.0f%%)",
                 positive, length(results), 100 * positive / length(results)))
println(@sprintf("  Zero:                    %d / %d", zero_, length(results)))
println(@sprintf("  Negative (cryst leads):  %d / %d", negative, length(results)))
