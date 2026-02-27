#!/usr/bin/env julia
#
# Cascade Phase Boundary & Pathway Analysis
#
# Three mechanistic questions about Phi's effect on convergence cascades:
#   A1: Does Phi shift the phase transition boundary? (known at θ≈7 for Phi=0)
#   A2: Does Phi create fundamentally different convergence pathways?
#   A3: Is there a non-monotonic Phi effect? (inverted-U)
#
# Output:
#   results/cascade_phase_boundary.csv    — 900 rows (A1)
#   results/cascade_pathways.csv          — 300 rows (A2)
#   results/cascade_phi_nonmonotonic.csv  — 990 rows (A3)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const N_AGENTS  = 100
const T_MAX     = 3000
const N_TRIALS  = 30
const BASE_SEED = 9500

# ══════════════════════════════════════════════════════════════════════════════
# Shared: run_one_trial() — manual tick loop with pathway tracking
# ══════════════════════════════════════════════════════════════════════════════

"""
    run_one_trial(; theta_crystal, Phi, seed) -> NamedTuple

Run one simulation with manual tick loop, tracking novel pathway metrics:
- total_dissolutions, total_enforcements
- recryst_flip, recryst_same (recrystallization direction)
- peak_norm_split (worst polarization among crystallized)
- time_in_churn (ticks with both dissolutions AND new crystallizations)

Returns a rich NamedTuple combining standard metrics (via summarize/first_tick_per_layer)
and novel pathway metrics.
"""
function run_one_trial(; theta_crystal::Float64, Phi::Float64, seed::Int)
    params = SimulationParams(
        N = N_AGENTS, T = T_MAX, seed = seed,
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
    prev_norms = fill(NO_NORM, N_AGENTS)           # norm at end of previous tick
    last_held_norm = fill(NO_NORM, N_AGENTS)        # remembers pre-dissolution norm

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

        for i in 1:N_AGENTS
            prev = prev_norms[i]
            curr = agents[i].r

            if prev != NO_NORM && curr == NO_NORM
                # Dissolution
                tick_dissolutions += 1
                last_held_norm[i] = prev  # remember what they held
            elseif prev == NO_NORM && curr != NO_NORM
                # Crystallization (new or re-)
                tick_new_crystals += 1
                if last_held_norm[i] != NO_NORM
                    # Recrystallization — check direction
                    if curr == last_held_norm[i]
                        recryst_same += 1
                    else
                        recryst_flip += 1
                    end
                    last_held_norm[i] = NO_NORM  # consumed
                end
            end

            prev_norms[i] = curr
        end

        total_dissolutions += tick_dissolutions

        # ── Churn: ticks with BOTH dissolutions and new crystallizations ──
        if tick_dissolutions > 0 && tick_new_crystals > 0
            time_in_churn += 1
        end

        # ── Peak norm split among crystallized agents ──
        n_norm_A = 0
        n_norm_B = 0
        for i in 1:N_AGENTS
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

        # ── Early termination ──
        if check_convergence(history, tick_count, params)
            break
        end
    end

    # ── Standard summary via existing helpers ──
    result = SimulationResult(params, history[1:tick_count], probes, tick_count)
    s = summarize(result)
    layers = first_tick_per_layer(result.history, params.N, params)

    return (
        # Standard
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
        # Novel pathway metrics
        total_dissolutions      = total_dissolutions,
        total_enforcements      = total_enforcements,
        recryst_flip            = recryst_flip,
        recryst_same            = recryst_same,
        peak_norm_split         = round(peak_norm_split, digits=4),
        time_in_churn           = time_in_churn,
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1: Phase Boundary — θ_c × Phi grid
# ══════════════════════════════════════════════════════════════════════════════

const A1_THETA = [3.0, 5.0, 7.0, 9.0, 12.0, 15.0]
const A1_PHI   = [0.0, 0.5, 1.0, 2.0, 5.0]

function run_phase_boundary()
    rows = []
    lk = ReentrantLock()

    tasks = [(tc, phi, trial)
             for tc in A1_THETA
             for phi in A1_PHI
             for trial in 1:N_TRIALS]

    println("  A1: $(length(tasks)) runs ($(length(A1_THETA)) θ × $(length(A1_PHI)) Φ × $N_TRIALS trials)")

    Threads.@threads for idx in eachindex(tasks)
        tc, phi, trial = tasks[idx]
        seed = Int(hash((BASE_SEED, :A1, tc, phi, trial)) % typemax(Int))
        r = run_one_trial(theta_crystal=tc, Phi=phi, seed=seed)

        row = (theta_crystal=tc, Phi=phi, trial=trial, r...)

        lock(lk) do
            push!(rows, row)
        end
    end

    return rows
end

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2: Pathway Classification — θ_c ∈ {3,7} × Phi
# ══════════════════════════════════════════════════════════════════════════════

const A2_THETA = [3.0, 7.0]
const A2_PHI   = [0.0, 0.5, 1.0, 2.0, 5.0]

function run_pathways()
    rows = []
    lk = ReentrantLock()

    tasks = [(tc, phi, trial)
             for tc in A2_THETA
             for phi in A2_PHI
             for trial in 1:N_TRIALS]

    println("  A2: $(length(tasks)) runs ($(length(A2_THETA)) θ × $(length(A2_PHI)) Φ × $N_TRIALS trials)")

    Threads.@threads for idx in eachindex(tasks)
        tc, phi, trial = tasks[idx]
        seed = Int(hash((BASE_SEED, :A2, tc, phi, trial)) % typemax(Int))
        r = run_one_trial(theta_crystal=tc, Phi=phi, seed=seed)

        row = (theta_crystal=tc, Phi=phi, trial=trial, r...)

        lock(lk) do
            push!(rows, row)
        end
    end

    return rows
end

# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3: Non-Monotonic Phi — fine Phi grid at 3 θ values
# ══════════════════════════════════════════════════════════════════════════════

const A3_THETA = [3.0, 5.0, 7.0]
const A3_PHI   = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

function run_nonmonotonic()
    rows = []
    lk = ReentrantLock()

    tasks = [(tc, phi, trial)
             for tc in A3_THETA
             for phi in A3_PHI
             for trial in 1:N_TRIALS]

    println("  A3: $(length(tasks)) runs ($(length(A3_THETA)) θ × $(length(A3_PHI)) Φ × $N_TRIALS trials)")

    Threads.@threads for idx in eachindex(tasks)
        tc, phi, trial = tasks[idx]
        seed = Int(hash((BASE_SEED, :A3, tc, phi, trial)) % typemax(Int))
        r = run_one_trial(theta_crystal=tc, Phi=phi, seed=seed)

        row = (theta_crystal=tc, Phi=phi, trial=trial, r...)

        lock(lk) do
            push!(rows, row)
        end
    end

    return rows
end

# ══════════════════════════════════════════════════════════════════════════════
# Console Output
# ══════════════════════════════════════════════════════════════════════════════

function print_a1_summary(rows)
    println("\n" * "=" ^ 100)
    println("A1: PHASE BOUNDARY — Convergence rate heatmap (θ_c × Φ)")
    println("=" ^ 100)

    # Convergence rate heatmap
    println("\n── Convergence rate ──")
    println(rpad("θ_c \\ Φ", 10), [rpad("Φ=$(phi)", 16) for phi in A1_PHI]...)
    println("-" ^ (10 + 16 * length(A1_PHI)))

    for tc in A1_THETA
        parts = String[]
        for phi in A1_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi]
            n_conv = count(r -> r.converged, sub)
            rate = round(n_conv / length(sub) * 100, digits=0)
            push!(parts, "$(Int(rate))% ($n_conv/$N_TRIALS)")
        end
        println(rpad("θ=$tc", 10), [rpad(p, 16) for p in parts]...)
    end

    # Mean dissolutions heatmap
    println("\n── Mean dissolutions ──")
    println(rpad("θ_c \\ Φ", 10), [rpad("Φ=$(phi)", 16) for phi in A1_PHI]...)
    println("-" ^ (10 + 16 * length(A1_PHI)))

    for tc in A1_THETA
        parts = String[]
        for phi in A1_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi]
            μ_diss = round(mean(r.total_dissolutions for r in sub), digits=1)
            push!(parts, string(μ_diss))
        end
        println(rpad("θ=$tc", 10), [rpad(p, 16) for p in parts]...)
    end

    # Mean convergence tick (converged only)
    println("\n── Mean convergence tick (converged trials only) ──")
    println(rpad("θ_c \\ Φ", 10), [rpad("Φ=$(phi)", 16) for phi in A1_PHI]...)
    println("-" ^ (10 + 16 * length(A1_PHI)))

    for tc in A1_THETA
        parts = String[]
        for phi in A1_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi && r.converged]
            if isempty(sub)
                push!(parts, "--")
            else
                push!(parts, string(round(Int, mean(r.convergence_tick for r in sub))))
            end
        end
        println(rpad("θ=$tc", 10), [rpad(p, 16) for p in parts]...)
    end
end

function print_a2_summary(rows)
    println("\n" * "=" ^ 100)
    println("A2: PATHWAY CLASSIFICATION — Metrics per θ_c × Φ")
    println("=" ^ 100)

    for tc in A2_THETA
        println("\n── θ_c = $tc ──")
        println(rpad("Φ", 8),
                rpad("Conv%", 8),
                rpad("μ_diss", 10),
                rpad("μ_enf", 10),
                rpad("μ_churn", 10),
                rpad("μ_split", 10),
                rpad("μ_flip", 10),
                rpad("μ_same", 10),
                rpad("μ_tick", 10))
        println("-" ^ 86)

        for phi in A2_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi]
            n_conv = count(r -> r.converged, sub)
            conv_pct = round(n_conv / length(sub) * 100, digits=0)

            μ_diss  = round(mean(r.total_dissolutions for r in sub), digits=1)
            μ_enf   = round(mean(r.total_enforcements for r in sub), digits=1)
            μ_churn = round(mean(r.time_in_churn for r in sub), digits=1)
            μ_split = round(mean(r.peak_norm_split for r in sub), digits=3)
            μ_flip  = round(mean(r.recryst_flip for r in sub), digits=1)
            μ_same  = round(mean(r.recryst_same for r in sub), digits=1)

            conv_sub = [r for r in sub if r.converged]
            μ_tick = isempty(conv_sub) ? "--" : string(round(Int, mean(r.convergence_tick for r in conv_sub)))

            println(rpad(phi, 8),
                    rpad("$(Int(conv_pct))%", 8),
                    rpad(μ_diss, 10),
                    rpad(μ_enf, 10),
                    rpad(μ_churn, 10),
                    rpad(μ_split, 10),
                    rpad(μ_flip, 10),
                    rpad(μ_same, 10),
                    rpad(μ_tick, 10))
        end
    end
end

function print_a3_summary(rows)
    println("\n" * "=" ^ 100)
    println("A3: NON-MONOTONIC Φ — Convergence rate & dissolutions across fine Φ grid")
    println("=" ^ 100)

    for tc in A3_THETA
        println("\n── θ_c = $tc ──")
        println(rpad("Φ", 8),
                rpad("Conv%", 10),
                rpad("μ_diss", 10),
                rpad("μ_enf", 10),
                rpad("μ_tick", 10),
                rpad("μ_churn", 10),
                rpad("μ_dom_norm", 12))
        println("-" ^ 70)

        for phi in A3_PHI
            sub = [r for r in rows if r.theta_crystal == tc && r.Phi == phi]
            n_conv = count(r -> r.converged, sub)
            conv_pct = round(n_conv / length(sub) * 100, digits=0)

            μ_diss  = round(mean(r.total_dissolutions for r in sub), digits=1)
            μ_enf   = round(mean(r.total_enforcements for r in sub), digits=1)
            μ_churn = round(mean(r.time_in_churn for r in sub), digits=1)
            μ_dn    = round(mean(r.final_frac_dominant_norm for r in sub), digits=3)

            conv_sub = [r for r in sub if r.converged]
            μ_tick = isempty(conv_sub) ? "--" : string(round(Int, mean(r.convergence_tick for r in conv_sub)))

            println(rpad(phi, 8),
                    rpad("$(Int(conv_pct))%", 10),
                    rpad(μ_diss, 10),
                    rpad(μ_enf, 10),
                    rpad(μ_tick, 10),
                    rpad(μ_churn, 10),
                    rpad(μ_dn, 12))
        end
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

function main()
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)

    total_runs = length(A1_THETA) * length(A1_PHI) * N_TRIALS +
                 length(A2_THETA) * length(A2_PHI) * N_TRIALS +
                 length(A3_THETA) * length(A3_PHI) * N_TRIALS

    println("=" ^ 100)
    println("CASCADE PHASE BOUNDARY & PATHWAY ANALYSIS")
    println("D_full_model: w_base=2, w_max=6, enable_normative=true, V=0")
    println("Total: $total_runs runs across 3 analyses")
    println("=" ^ 100)

    # ── A1: Phase Boundary ──
    println("\n── Analysis 1: Phase Boundary ──")
    t1 = time()
    a1_rows = run_phase_boundary()
    CSV.write(joinpath(outdir, "cascade_phase_boundary.csv"), a1_rows)
    println("  Saved $(length(a1_rows)) rows → cascade_phase_boundary.csv ($(round(time()-t1, digits=1))s)")

    # ── A2: Pathway Classification ──
    println("\n── Analysis 2: Pathway Classification ──")
    t2 = time()
    a2_rows = run_pathways()
    CSV.write(joinpath(outdir, "cascade_pathways.csv"), a2_rows)
    println("  Saved $(length(a2_rows)) rows → cascade_pathways.csv ($(round(time()-t2, digits=1))s)")

    # ── A3: Non-Monotonic Phi ──
    println("\n── Analysis 3: Non-Monotonic Φ ──")
    t3 = time()
    a3_rows = run_nonmonotonic()
    CSV.write(joinpath(outdir, "cascade_phi_nonmonotonic.csv"), a3_rows)
    println("  Saved $(length(a3_rows)) rows → cascade_phi_nonmonotonic.csv ($(round(time()-t3, digits=1))s)")

    # ── Console summaries ──
    print_a1_summary(a1_rows)
    print_a2_summary(a2_rows)
    print_a3_summary(a3_rows)

    println("\n" * "=" ^ 100)
    println("Total time: $(round(time()-t1, digits=1))s")
    println("Done.")
end

main()
