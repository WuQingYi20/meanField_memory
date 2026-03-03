#!/usr/bin/env julia
#
# Large-N Cascade Phase Boundary & Pathway Analysis
# Extension of sweep_convergence_cascade.jl to large N.
#
# Strategy:
#   N_FULL = [1000, 5000]             — run A1 + A2 + A3 (full grid)
#   N_PATHWAYS_ONLY = [10000, 20000]  — run A2 only (pathway classification)
#
# A1: Phase Boundary    — 6θ × 5Φ × 30 trials = 900 runs/N
# A2: Pathway Class.    — 2θ × 5Φ × 30 trials = 300 runs/N
# A3: Non-Monotonic Φ   — 3θ × 11Φ × 30 trials = 990 runs/N
#
# Output per N:
#   results/large_N/cascade_phase_boundary_N{N}.csv   (N_FULL only)
#   results/large_N/cascade_pathways_N{N}.csv          (all N)
#   results/large_N/cascade_phi_nonmonotonic_N{N}.csv  (N_FULL only)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

include(joinpath(@__DIR__, "large_N_common.jl"))

# ── Configuration ──

const N_FULL           = [1000, 5000]
const N_PATHWAYS_ONLY  = [10000, 20000]
const N_ALL            = vcat(N_FULL, N_PATHWAYS_ONLY)

const N_TRIALS  = 30
const BASE_SEED = 9500

const OUTDIR = joinpath(@__DIR__, "..", "results", "large_N")

# Grid parameters (identical to sweep_convergence_cascade.jl)
const A1_THETA = [3.0, 5.0, 7.0, 9.0, 12.0, 15.0]
const A1_PHI   = [0.0, 0.5, 1.0, 2.0, 5.0]

const A2_THETA = [3.0, 7.0]
const A2_PHI   = [0.0, 0.5, 1.0, 2.0, 5.0]

const A3_THETA = [3.0, 5.0, 7.0]
const A3_PHI   = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

# ══════════════════════════════════════════════════════════════
# run_one_trial() — manual tick loop with pathway tracking
# (from sweep_convergence_cascade.jl, parameterized by N)
# ══════════════════════════════════════════════════════════════

function run_one_trial(; N::Int, theta_crystal::Float64, Phi::Float64, seed::Int)
    T_MAX = select_T_MAX(N)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = Phi,
        theta_crystal = theta_crystal,
    )

    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    # ── Transition tracking state ──
    prev_norms = fill(NO_NORM, N)
    last_held_norm = fill(NO_NORM, N)

    # ── Accumulators ──
    total_dissolutions = 0
    total_enforcements = 0
    recryst_flip = 0
    recryst_same = 0
    peak_norm_split = 0.0
    time_in_churn = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        m = history[tick_count]

        # ── Accumulate enforcements ──
        total_enforcements += m.num_enforcements

        # ── Detect transitions this tick ──
        tick_dissolutions = 0
        tick_new_crystals = 0

        for i in 1:N
            prev = prev_norms[i]
            curr = agents[i].r

            if prev != NO_NORM && curr == NO_NORM
                tick_dissolutions += 1
                last_held_norm[i] = prev
            elseif prev == NO_NORM && curr != NO_NORM
                tick_new_crystals += 1
                if last_held_norm[i] != NO_NORM
                    if curr == last_held_norm[i]
                        recryst_same += 1
                    else
                        recryst_flip += 1
                    end
                    last_held_norm[i] = NO_NORM
                end
            end

            prev_norms[i] = curr
        end

        total_dissolutions += tick_dissolutions

        if tick_dissolutions > 0 && tick_new_crystals > 0
            time_in_churn += 1
        end

        # ── Peak norm split among crystallized agents ──
        n_norm_A = 0
        n_norm_B = 0
        for i in 1:N
            if agents[i].r == STRATEGY_A
                n_norm_A += 1
            elseif agents[i].r == STRATEGY_B
                n_norm_B += 1
            end
        end
        n_cryst_total = n_norm_A + n_norm_B
        if n_cryst_total > 0
            split = max(n_norm_A, n_norm_B) / n_cryst_total
            peak_norm_split = max(peak_norm_split, split)
        end

        if check_convergence(history, tick_count, params)
            break
        end
    end

    # ── Standard summary ──
    result = SimulationResult(params, history[1:tick_count], probes, tick_count)
    s = summarize(result)
    layers = first_tick_per_layer(result.history, params.N, params)

    return (
        converged               = s.converged,
        convergence_tick        = s.convergence_tick,
        total_ticks             = s.total_ticks,
        final_fraction_A        = s.final_fraction_A,
        final_mean_confidence   = s.final_mean_confidence,
        final_num_crystallised  = s.final_num_crystallised,
        final_mean_norm_strength = s.final_mean_norm_strength,
        final_frac_dominant_norm = s.final_frac_dominant_norm,
        first_tick_behavioral   = layers.behavioral,
        first_tick_belief       = layers.belief,
        first_tick_crystal      = layers.crystal,
        first_tick_all_met      = layers.all_met,
        total_dissolutions      = total_dissolutions,
        total_enforcements      = total_enforcements,
        recryst_flip            = recryst_flip,
        recryst_same            = recryst_same,
        peak_norm_split         = round(peak_norm_split, digits=4),
        time_in_churn           = time_in_churn,
    )
end

# ══════════════════════════════════════════════════════════════
# Generic grid runner with checkpoint/resume
# ══════════════════════════════════════════════════════════════

function run_grid(N::Int, thetas, phis, outpath::String, analysis_tag::Symbol)
    done = load_completed_trials(outpath, [:theta_crystal, :Phi, :trial])

    tasks = [(tc, phi, trial)
             for tc in thetas
             for phi in phis
             for trial in 1:N_TRIALS]

    needed = [(tc, phi, trial) for (tc, phi, trial) in tasks
              if (tc, phi, trial) ∉ done]

    if isempty(needed)
        progress_log("  $(analysis_tag) N=$N — all $(length(tasks)) runs done, skipping")
        return
    end

    progress_log("  $(analysis_tag) N=$N: $(length(needed))/$(length(tasks)) runs")
    t_start = time()

    rows = Vector{Any}()
    lk = ReentrantLock()

    Threads.@threads for idx in eachindex(needed)
        tc, phi, trial = needed[idx]
        seed = Int(hash((BASE_SEED, analysis_tag, N, tc, phi, trial)) % typemax(Int))
        r = run_one_trial(N=N, theta_crystal=tc, Phi=phi, seed=seed)

        row = (theta_crystal=tc, Phi=phi, trial=trial, N=N, r...)

        lock(lk) do
            push!(rows, row)
        end
    end

    incremental_csv_write(outpath, rows)

    elapsed = round(time() - t_start, digits=1)
    n_conv = count(r -> r.converged, rows)
    progress_log("  → $(length(rows)) runs saved ($(n_conv) converged) in $(elapsed)s")
end

# ══════════════════════════════════════════════════════════════
# Console summaries
# ══════════════════════════════════════════════════════════════

function print_a2_summary_for_N(N::Int)
    outpath = joinpath(OUTDIR, "cascade_pathways_N$(N).csv")
    if !isfile(outpath)
        return
    end
    df = CSV.read(outpath, DataFrame)
    rows = [NamedTuple(row) for row in eachrow(df)]

    println("\n── N=$N: Pathway Classification ──")
    for tc in A2_THETA
        println("\n  θ_c = $tc")
        println("  ", rpad("Φ", 8), rpad("Conv%", 8), rpad("μ_diss", 10),
                rpad("μ_enf", 10), rpad("μ_churn", 10), rpad("μ_flip", 10), rpad("μ_tick", 10))
        println("  ", "-" ^ 66)

        for phi in A2_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi]
            if isempty(sub)
                continue
            end
            n_conv = count(r -> r.converged, sub)
            conv_pct = round(n_conv / length(sub) * 100, digits=0)
            μ_diss  = round(mean(r.total_dissolutions for r in sub), digits=1)
            μ_enf   = round(mean(r.total_enforcements for r in sub), digits=1)
            μ_churn = round(mean(r.time_in_churn for r in sub), digits=1)
            μ_flip  = round(mean(r.recryst_flip for r in sub), digits=1)

            conv_sub = [r for r in sub if r.converged]
            μ_tick = isempty(conv_sub) ? "--" : string(round(Int, mean(r.convergence_tick for r in conv_sub)))

            println("  ", rpad(phi, 8), rpad("$(Int(conv_pct))%", 8),
                    rpad(μ_diss, 10), rpad(μ_enf, 10), rpad(μ_churn, 10),
                    rpad(μ_flip, 10), rpad(μ_tick, 10))
        end
    end
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(OUTDIR)

    total_t = time()

    for N in N_ALL
        progress_log("=" ^ 60)
        progress_log("CASCADE ANALYSIS — N = $N  (T_MAX=$(select_T_MAX(N)))")
        progress_log("=" ^ 60)

        run_full = N in N_FULL

        if run_full
            # A1: Phase Boundary
            a1_path = joinpath(OUTDIR, "cascade_phase_boundary_N$(N).csv")
            run_grid(N, A1_THETA, A1_PHI, a1_path, :A1)
        end

        # A2: Pathway Classification (always)
        a2_path = joinpath(OUTDIR, "cascade_pathways_N$(N).csv")
        run_grid(N, A2_THETA, A2_PHI, a2_path, :A2)

        if run_full
            # A3: Non-Monotonic Phi
            a3_path = joinpath(OUTDIR, "cascade_phi_nonmonotonic_N$(N).csv")
            run_grid(N, A3_THETA, A3_PHI, a3_path, :A3)
        end
    end

    # Console summaries
    println("\n" * "=" ^ 100)
    println("PATHWAY COMPARISON ACROSS N")
    println("=" ^ 100)
    for N in N_ALL
        print_a2_summary_for_N(N)
    end

    elapsed = round(time() - total_t, digits=1)
    progress_log("Total time: $(elapsed)s")
    println("\nDone.")
end

main()
