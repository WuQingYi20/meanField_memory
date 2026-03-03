#!/usr/bin/env julia
#
# Small-N Speedup Ratio Experiment
# Tests normative speedup at very small populations: N ∈ {4, 6, 8, 10, 15, 20}
#
# Hypothesis: speedup → 1 as N shrinks (experiential memory suffices),
# then rises sharply — revealing a transition point.
#
# Two conditions only (Dynamic × Norm ON/OFF):
#   Dyn_noNorm (w_base=2, w_max=6, norm=false)
#   Dyn_Norm   (w_base=2, w_max=6, norm=true)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using DataFrames
using Statistics
using Random
using Plots

include(joinpath(@__DIR__, "large_N_common.jl"))

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

const N_VALUES  = [4, 6, 8, 10, 16, 20]
const N_TRIALS  = 100
const T_MAX     = 3000
const BASE_SEED = 9000

const BEH_THRESH  = 0.95
const BEH_WINDOW  = 50

const OUTDIR    = joinpath(@__DIR__, "..", "results", "small_N")
const FIGDIR    = joinpath(@__DIR__, "..", "figures")
const SUMM_PATH = joinpath(@__DIR__, "..", "results", "small_N_speedup.csv")

# ══════════════════════════════════════════════════════════════
# Condition definition
# ══════════════════════════════════════════════════════════════

struct Condition
    label::String
    enable_normative::Bool
    w_base::Int
    w_max::Int
end

const CONDITIONS = [
    Condition("Dyn_noNorm", false, 2, 6),
    Condition("Dyn_Norm",   true,  2, 6),
]

# ══════════════════════════════════════════════════════════════
# Run loop (Dynamic experiential only, norm ON/OFF)
# ══════════════════════════════════════════════════════════════

function run_ablation(N::Int, T_MAX::Int, cond::Condition, seed::Int)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = cond.w_base, w_max = cond.w_max,
        enable_normative = cond.enable_normative,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    beh_counter = 0
    beh_converged = false
    beh_conv_tick = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Behavioral convergence check ──
        m = history[tick_count]
        majority = max(m.fraction_A, 1.0 - m.fraction_A)
        if majority >= BEH_THRESH
            beh_counter += 1
            if !beh_converged && beh_counter >= BEH_WINDOW
                beh_converged = true
                beh_conv_tick = t - BEH_WINDOW + 1
            end
        else
            beh_counter = 0
        end

        if beh_converged && t >= beh_conv_tick + 200
            break
        end
    end

    return (
        history    = history[1:tick_count],
        params     = params,
        converged  = beh_converged,
        conv_tick  = beh_conv_tick,
        tick_count = tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Post-hoc metrics
# ══════════════════════════════════════════════════════════════

function compute_metrics(result, N::Int)
    hist = result.history
    n = length(hist)
    final = hist[end]

    first_beh_tick = 0
    for m in hist
        if max(m.fraction_A, 1.0 - m.fraction_A) >= BEH_THRESH
            first_beh_tick = m.tick
            break
        end
    end

    first_crystal_50 = 0
    for m in hist
        if m.num_crystallised >= N ÷ 2
            first_crystal_50 = m.tick
            break
        end
    end

    mean_enforce = 0.0
    if result.converged
        ct = result.conv_tick
        start_idx = 0
        for idx in 1:n
            if hist[idx].tick >= ct
                start_idx = idx
                break
            end
        end
        if start_idx > 0
            end_idx = min(start_idx + 99, n)
            mean_enforce = mean(hist[idx].num_enforcements for idx in start_idx:end_idx)
        end
    end

    return (
        converged         = result.converged,
        conv_tick         = result.conv_tick,
        first_beh_tick    = first_beh_tick,
        first_crystal_50  = first_crystal_50,
        final_coord       = final.coordination_rate,
        final_majority    = max(final.fraction_A, 1.0 - final.fraction_A),
        final_cryst       = final.num_crystallised,
        final_norm_str    = final.mean_norm_strength,
        final_belief_err  = final.belief_error,
        mean_enforce      = mean_enforce,
        total_ticks       = result.tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Run sweep
# ══════════════════════════════════════════════════════════════

function run_sweep()
    mkpath(OUTDIR)

    for N in N_VALUES
        outpath = joinpath(OUTDIR, "ablation_small_N_$(N).csv")

        # Checkpoint
        done = load_completed_trials(outpath, [:condition, :trial])
        if length(done) >= length(CONDITIONS) * N_TRIALS
            progress_log("N=$N — all $(length(CONDITIONS) * N_TRIALS) trials done, skipping")
            continue
        end
        if !isempty(done)
            progress_log("N=$N — resuming ($(length(done)) trials already completed)")
        end

        progress_log("N=$N  T_MAX=$T_MAX  ($N_TRIALS trials × $(length(CONDITIONS)) conditions)")
        t_start = time()

        for cond in CONDITIONS
            needed = [t for t in 1:N_TRIALS if (cond.label, t) ∉ done]
            if isempty(needed)
                progress_log("  $(cond.label) — all done, skipping")
                continue
            end

            progress_log("  $(cond.label) × $(length(needed)) trials...")

            block_rows = Vector{Any}()
            lk = ReentrantLock()

            Threads.@threads for trial in needed
                seed = hash((BASE_SEED, N, cond.label, trial)) % typemax(Int)
                result = run_ablation(N, T_MAX, cond, Int(seed))
                met = compute_metrics(result, N)

                row = (
                    N              = N,
                    condition      = cond.label,
                    norm_on        = cond.enable_normative,
                    trial          = trial,
                    converged      = met.converged,
                    conv_tick      = met.conv_tick,
                    first_beh_tick = met.first_beh_tick,
                    first_cryst_50 = met.first_crystal_50,
                    final_coord    = met.final_coord,
                    final_majority = met.final_majority,
                    final_cryst    = met.final_cryst,
                    final_norm_str = met.final_norm_str,
                    final_belief_err = met.final_belief_err,
                    mean_enforce   = met.mean_enforce,
                    total_ticks    = met.total_ticks,
                )

                lock(lk) do
                    push!(block_rows, row)
                end
            end

            incremental_csv_write(outpath, block_rows)
        end

        elapsed = round(time() - t_start, digits=1)
        progress_log("  N=$N complete in $(elapsed)s")
    end
end

# ══════════════════════════════════════════════════════════════
# Compute summary & write CSV
# ══════════════════════════════════════════════════════════════

function compute_summary()
    summary_rows = []

    for N in N_VALUES
        outpath = joinpath(OUTDIR, "ablation_small_N_$(N).csv")
        if !isfile(outpath)
            progress_log("WARNING: missing $outpath")
            continue
        end
        df = CSV.read(outpath, DataFrame)

        off = filter(r -> r.condition == "Dyn_noNorm", df)
        on  = filter(r -> r.condition == "Dyn_Norm",   df)

        n_off = nrow(off)
        n_on  = nrow(on)

        conv_off = filter(r -> r.converged, off)
        conv_on  = filter(r -> r.converged, on)

        rate_off = nrow(conv_off) / n_off
        rate_on  = nrow(conv_on)  / n_on

        ticks_off = conv_off.conv_tick
        ticks_on  = conv_on.conv_tick

        mean_off = isempty(ticks_off) ? NaN : mean(ticks_off)
        mean_on  = isempty(ticks_on)  ? NaN : mean(ticks_on)

        se_off = isempty(ticks_off) ? NaN : std(ticks_off) / sqrt(length(ticks_off))
        se_on  = isempty(ticks_on)  ? NaN : std(ticks_on)  / sqrt(length(ticks_on))

        speedup = (isnan(mean_off) || isnan(mean_on) || mean_on == 0) ? NaN : mean_off / mean_on

        push!(summary_rows, (
            N           = N,
            n_trials    = N_TRIALS,
            conv_rate_off = round(rate_off, digits=3),
            conv_rate_on  = round(rate_on,  digits=3),
            mean_tick_off = round(mean_off, digits=1),
            mean_tick_on  = round(mean_on,  digits=1),
            speedup       = round(speedup, digits=2),
            se_off        = round(se_off, digits=1),
            se_on         = round(se_on,  digits=1),
        ))
    end

    CSV.write(SUMM_PATH, summary_rows)
    progress_log("Summary written to $SUMM_PATH")
    return summary_rows
end

# ══════════════════════════════════════════════════════════════
# Plot: speedup vs N
# ══════════════════════════════════════════════════════════════

function plot_speedup(summary_rows)
    mkpath(FIGDIR)

    Ns = [r.N for r in summary_rows]
    speedups = [r.speedup for r in summary_rows]

    # Compute SE of the speedup ratio via delta method:
    # speedup = μ_off / μ_on
    # SE(speedup) ≈ speedup * sqrt( (se_off/mean_off)^2 + (se_on/mean_on)^2 )
    se_speedup = [
        let s = r.speedup, cv_off = r.se_off / r.mean_tick_off, cv_on = r.se_on / r.mean_tick_on
            s * sqrt(cv_off^2 + cv_on^2)
        end
        for r in summary_rows
    ]

    p = plot(Ns, speedups,
        seriestype = :scatter,
        yerror = se_speedup,
        label = "Normative speedup",
        color = :royalblue,
        markerstrokecolor = :royalblue,
        markersize = 6,
        lw = 0,
        xlabel = "Population size N",
        ylabel = "Speedup ratio (Dyn-only / Dyn+Norm)",
        title = "Normative Speedup at Small Population Sizes",
        legend = :topleft,
        size = (700, 450),
        grid = true, gridalpha = 0.3,
        fontfamily = "sans-serif", titlefontsize = 12,
        xlims = (2, 22), xticks = N_VALUES,
    )

    # Connecting line
    plot!(p, Ns, speedups,
        label = "", lw = 2.5, color = :royalblue, alpha = 0.6)

    # Horizontal dashed line at y=1 (no-speedup baseline)
    hline!(p, [1.0], color = :gray, ls = :dash, lw = 1.5,
        label = "No speedup (ratio = 1)")

    figpath = joinpath(FIGDIR, "fig_small_N_speedup.png")
    savefig(p, figpath)
    progress_log("Figure saved to $figpath")
    return p
end

# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════

function print_summary(summary_rows)
    println("\n" * "=" ^ 80)
    println("Small-N Speedup Summary")
    println("=" ^ 80)
    println()
    println(rpad("N", 6), rpad("Conv%_off", 12), rpad("Conv%_on", 12),
            rpad("Mean_off", 12), rpad("Mean_on", 12), rpad("Speedup", 10),
            rpad("SE_off", 10), rpad("SE_on", 10))
    println("-" ^ 80)
    for r in summary_rows
        println(rpad(r.N, 6),
                rpad("$(round(r.conv_rate_off*100, digits=0))%", 12),
                rpad("$(round(r.conv_rate_on*100, digits=0))%", 12),
                rpad(r.mean_tick_off, 12),
                rpad(r.mean_tick_on, 12),
                rpad("$(r.speedup)×", 10),
                rpad(r.se_off, 10),
                rpad(r.se_on, 10))
    end
    println()
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    progress_log("Small-N Speedup Experiment: N ∈ $(N_VALUES)")
    t0 = time()

    run_sweep()
    summary = compute_summary()
    print_summary(summary)
    plot_speedup(summary)

    elapsed = round(time() - t0, digits=1)
    progress_log("Total time: $(elapsed)s")
    println("\nDone.")
end

main()
