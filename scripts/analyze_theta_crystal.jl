#!/usr/bin/env julia
#
# Deep analysis of theta_crystal:
# 1. Crystallisation dynamics trajectories (norm A vs norm B vs no norm over time)
# 2. Fine grid around the failure boundary
# 3. θ × N interaction
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ═══════════════════════════════════════════════════════════════
# Part 1: Crystallisation dynamics at key θ values
# ═══════════════════════════════════════════════════════════════

println("="^90)
println("PART 1: Crystallisation dynamics (condition D, seed=42)")
println("="^90)

THETA_KEYS = [0.5, 1.0, 3.0, 8.0, 12.0]

traj_rows = []

for theta in THETA_KEYS
    p = SimulationParams(
        N=100, T=3000, seed=42,
        w_base=2, w_max=6, enable_normative=true,
        theta_crystal=theta,
    )
    agents, ws, history, rng = initialize(p)
    probes = ProbeSet()
    init_probes!(probes, p, rng)

    tc = 0
    for t in 1:p.T
        run_tick!(t, agents, ws, history, tc, p, rng, probes)
        tc += 1

        m = history[tc]

        # Count norm directions among crystallised agents
        n_norm_A = count(a -> a.r == STRATEGY_A, agents)
        n_norm_B = count(a -> a.r == STRATEGY_B, agents)
        n_no_norm = p.N - n_norm_A - n_norm_B

        # Mean sigma among crystallised
        cryst_agents = [a for a in agents if a.r != NO_NORM]
        mean_sigma = length(cryst_agents) > 0 ? mean(a.sigma for a in cryst_agents) : 0.0

        # Mean |e| among non-crystallised (DDM progress)
        non_cryst = [a for a in agents if a.r == NO_NORM]
        mean_abs_e = length(non_cryst) > 0 ? mean(abs(a.e) for a in non_cryst) : 0.0

        push!(traj_rows, (
            theta_crystal = theta,
            tick          = t,
            fraction_A    = m.fraction_A,
            mean_confidence = m.mean_confidence,
            n_norm_A      = n_norm_A,
            n_norm_B      = n_norm_B,
            n_no_norm     = n_no_norm,
            mean_sigma    = round(mean_sigma, digits=4),
            mean_abs_e    = round(mean_abs_e, digits=4),
            frac_dominant_norm = m.frac_dominant_norm,
            convergence_counter = m.convergence_counter,
        ))

        if check_convergence(history, tc, p)
            break
        end
    end

    # Print summary
    last_tick = traj_rows[end].tick
    println("\nθ=$theta → ended at tick $last_tick")

    # Show key moments
    println("  Tick | frac_A | norm_A | norm_B | no_norm | mean_σ | mean|e| | dom_norm")
    println("  " * "-"^80)
    all_for_theta = [r for r in traj_rows if r.theta_crystal == theta]
    # Show snapshots at key ticks
    for snap_t in [1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 500, last_tick]
        idx = findfirst(r -> r.tick == snap_t, all_for_theta)
        if idx !== nothing
            r = all_for_theta[idx]
            println("  $(lpad(r.tick, 4)) |  $(lpad(round(r.fraction_A, digits=2), 4)) |" *
                    "   $(lpad(r.n_norm_A, 3)) |   $(lpad(r.n_norm_B, 3)) |" *
                    "    $(lpad(r.n_no_norm, 3)) |  $(lpad(r.mean_sigma, 5)) |" *
                    "   $(lpad(r.mean_abs_e, 5)) | $(lpad(round(r.frac_dominant_norm, digits=2), 5))")
        end
    end
end

# Save trajectories
outdir = joinpath(@__DIR__, "..", "results")
mkpath(outdir)
CSV.write(joinpath(outdir, "theta_crystal_trajectories.csv"), traj_rows)
println("\nTrajectories saved.")

# ═══════════════════════════════════════════════════════════════
# Part 2: Fine grid around failure boundary (θ = 5..20)
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^90)
println("PART 2: Fine grid around failure boundary (condition D, N=100)")
println("="^90)

THETA_FINE = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 18.0, 20.0]
N_TRIALS = 30

fine_rows = []
for theta in THETA_FINE
    Threads.@threads for trial in 1:N_TRIALS
        seed = hash((5000, theta, trial)) % typemax(Int)
        p = SimulationParams(
            N=100, T=3000, seed=Int(seed),
            w_base=2, w_max=6, enable_normative=true,
            theta_crystal=theta,
        )
        result = run!(p)
        s = summarize(result)
        layers = first_tick_per_layer(result.history, p.N, p)

        push!(fine_rows, (
            theta_crystal         = theta,
            trial                 = trial,
            converged             = s.converged,
            convergence_tick      = s.convergence_tick,
            first_tick_behavioral = layers.behavioral,
            first_tick_crystal    = layers.crystal,
            first_tick_all_met    = layers.all_met,
            final_num_crystallised = s.final_num_crystallised,
            final_frac_dominant_norm = s.final_frac_dominant_norm,
            final_mean_norm_strength = s.final_mean_norm_strength,
            total_ticks           = s.total_ticks,
        ))
    end
end

CSV.write(joinpath(outdir, "theta_crystal_fine.csv"), fine_rows)

println(rpad("θ", 8), rpad("conv_rate", 12), rpad("conv_tick", 14),
        rpad("behav_tick", 14), rpad("cryst_tick", 14), rpad("dom_norm", 12))
println("-"^74)
for theta in THETA_FINE
    sub = [r for r in fine_rows if r.theta_crystal == theta]
    n_conv = count(r -> r.converged, sub)
    conv_t = [r.convergence_tick for r in sub if r.converged]
    beh_t = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
    cry_t = [r.first_tick_crystal for r in sub if r.first_tick_crystal > 0]
    dn = mean([r.final_frac_dominant_norm for r in sub])

    println(rpad(theta, 8),
            rpad("$(n_conv)/$(length(sub))", 12),
            rpad(length(conv_t) > 0 ? "μ=$(round(Int, mean(conv_t)))" : "—", 14),
            rpad(length(beh_t) > 0 ? "μ=$(round(Int, mean(beh_t)))" : "—", 14),
            rpad(length(cry_t) > 0 ? "μ=$(round(Int, mean(cry_t)))" : "—", 14),
            rpad(round(dn, digits=3), 12))
end

# ═══════════════════════════════════════════════════════════════
# Part 3: θ × N interaction — does the sweet spot shift?
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^90)
println("PART 3: θ × N interaction (condition D)")
println("="^90)

THETA_GRID = [1.0, 3.0, 5.0, 8.0, 12.0]
N_GRID = [50, 100, 200, 500]

interaction_rows = []
for N in N_GRID
    for theta in THETA_GRID
        Threads.@threads for trial in 1:N_TRIALS
            seed = hash((6000, N, theta, trial)) % typemax(Int)
            p = SimulationParams(
                N=N, T=3000, seed=Int(seed),
                w_base=2, w_max=6, enable_normative=true,
                theta_crystal=theta,
            )
            result = run!(p)
            s = summarize(result)
            layers = first_tick_per_layer(result.history, p.N, p)

            push!(interaction_rows, (
                N = N, theta_crystal = theta, trial = trial,
                converged = s.converged,
                convergence_tick = s.convergence_tick,
                first_tick_behavioral = layers.behavioral,
                final_frac_dominant_norm = s.final_frac_dominant_norm,
            ))
        end
    end
end

CSV.write(joinpath(outdir, "theta_crystal_x_N.csv"), interaction_rows)

# Print as table: rows=N, cols=θ, cell = conv_rate (mean_tick)
println("\nConvergence rate (mean tick if converged):")
println(rpad("N", 8), [rpad("θ=$t", 20) for t in THETA_GRID]...)
println("-"^108)
for N in N_GRID
    parts = []
    for theta in THETA_GRID
        sub = [r for r in interaction_rows if r.N == N && r.theta_crystal == theta]
        n_conv = count(r -> r.converged, sub)
        conv_t = [r.convergence_tick for r in sub if r.converged]
        if n_conv > 0
            push!(parts, "$(n_conv)/$(length(sub)) μ=$(round(Int, mean(conv_t)))")
        else
            push!(parts, "0/$(length(sub))")
        end
    end
    println(rpad(N, 8), [rpad(p, 20) for p in parts]...)
end

println("\nBehavioral layer speed:")
println(rpad("N", 8), [rpad("θ=$t", 20) for t in THETA_GRID]...)
println("-"^108)
for N in N_GRID
    parts = []
    for theta in THETA_GRID
        sub = [r for r in interaction_rows if r.N == N && r.theta_crystal == theta]
        vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
        if length(vals) > 0
            push!(parts, "$(length(vals))/$(length(sub)) μ=$(round(Int, mean(vals)))")
        else
            push!(parts, "0/$(length(sub))")
        end
    end
    println(rpad(N, 8), [rpad(p, 20) for p in parts]...)
end

println("\nNorm quality (mean frac_dominant_norm):")
println(rpad("N", 8), [rpad("θ=$t", 16) for t in THETA_GRID]...)
println("-"^88)
for N in N_GRID
    parts = []
    for theta in THETA_GRID
        sub = [r for r in interaction_rows if r.N == N && r.theta_crystal == theta]
        dn = mean([r.final_frac_dominant_norm for r in sub])
        push!(parts, round(dn, digits=3))
    end
    println(rpad(N, 8), [rpad(p, 16) for p in parts]...)
end
