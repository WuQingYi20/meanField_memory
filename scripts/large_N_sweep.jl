#!/usr/bin/env julia
#
# Large-N Sweep: Population size N × 4 conditions (2×2 factorial)
# Extension of sweep_N.jl to N ∈ {1000, 2000, 5000, 10000, 20000}
#
# Features over the original:
#   - Adaptive T_MAX via select_T_MAX(N)
#   - Incremental CSV output (crash-safe)
#   - Checkpoint/resume: skips already-completed (N, condition, trial) combos
#
# Output: results/large_N/sweep_N_extended.csv
#         (schema identical to results/sweep_N.csv)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

include(joinpath(@__DIR__, "large_N_common.jl"))

# ── Configuration ──

const N_RANGE   = [1000, 2000, 5000, 10000, 20000]
const N_TRIALS  = 30
const BASE_SEED = 2000

const CONDITIONS = [
    # (label, w_base, w_max, enable_normative)
    ("A_baseline",       5, 5, false),
    ("B_lockin_only",    2, 6, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

const OUTDIR  = joinpath(@__DIR__, "..", "results", "large_N")
const OUTPATH = joinpath(OUTDIR, "sweep_N_extended.csv")

# ── Run ──

function run_sweep()
    mkpath(OUTDIR)

    # Load checkpoint
    done = load_completed_trials(OUTPATH, [:N, :condition, :trial])
    if !isempty(done)
        progress_log("Resuming: $(length(done)) trials already completed")
    end

    all_rows = []   # for summary printing

    for N in N_RANGE
        T = select_T_MAX(N)
        for (label, wb, wm, enorm) in CONDITIONS
            # Check how many trials need running
            needed = [t for t in 1:N_TRIALS if (N, label, t) ∉ done]
            if isempty(needed)
                progress_log("N=$N  $label  — all $N_TRIALS trials done, skipping")
                continue
            end

            progress_log("N=$N  $label  ($(length(needed))/$N_TRIALS trials, T=$T)")
            t_start = time()

            block_rows = Vector{Any}()
            lk = ReentrantLock()

            Threads.@threads for trial in needed
                seed = hash((BASE_SEED, N, label, trial)) % typemax(Int)
                p = SimulationParams(
                    N       = N,
                    T       = T,
                    seed    = Int(seed),
                    w_base  = wb,
                    w_max   = wm,
                    enable_normative = enorm,
                )
                result = run!(p)
                s = summarize(result)
                layers = first_tick_per_layer(result.history, p.N, p)

                row = (
                    N                       = N,
                    condition               = label,
                    w_base                  = wb,
                    w_max                   = wm,
                    enable_normative        = enorm,
                    trial                   = trial,
                    convergence_tick        = s.convergence_tick,
                    converged               = s.converged,
                    first_tick_behavioral   = layers.behavioral,
                    first_tick_belief       = layers.belief,
                    first_tick_crystal      = layers.crystal,
                    first_tick_all_met      = layers.all_met,
                    final_fraction_A        = s.final_fraction_A,
                    final_mean_confidence   = s.final_mean_confidence,
                    final_num_crystallised  = s.final_num_crystallised,
                    final_mean_norm_strength = s.final_mean_norm_strength,
                    final_frac_dominant_norm = s.final_frac_dominant_norm,
                    total_ticks             = s.total_ticks,
                )

                lock(lk) do
                    push!(block_rows, row)
                end
            end

            # Write this block incrementally
            incremental_csv_write(OUTPATH, block_rows)
            append!(all_rows, block_rows)

            elapsed = round(time() - t_start, digits=1)
            n_conv = count(r -> r.converged, block_rows)
            progress_log("  → $(length(block_rows)) trials saved ($(n_conv) converged) in $(elapsed)s")
        end
    end

    # Also load previously completed rows for summary
    if isfile(OUTPATH)
        all_rows = CSV.read(OUTPATH, DataFrame)
        all_rows = [NamedTuple(row) for row in eachrow(all_rows)]
    end

    return all_rows
end

# ── Summary tables (reused from sweep_N.jl) ──

function print_summary(rows)
    println("\n" * "="^100)
    println("SUMMARY: behavioral layer (first tick ≥95% majority)")
    println("="^100)
    println(rpad("N", 8),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^100)

    for N in N_RANGE
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            if length(vals) > 0
                s = "$(length(vals))/$(length(sub)) μ=$(round(Int, mean(vals)))"
            else
                s = "0/$(length(sub))"
            end
            push!(parts, s)
        end
        println(rpad(N, 8), rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    println("\n" * "="^100)
    println("SUMMARY: convergence (all layers met + stable 50 ticks)")
    println("="^100)
    println(rpad("N", 8),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^100)

    for N in N_RANGE
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            n_conv = count(r -> r.converged, sub)
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            if n_conv > 0
                s = "$(n_conv)/$(length(sub)) μ=$(round(Int, mean(conv_ticks)))"
            else
                s = "0/$(length(sub))"
            end
            push!(parts, s)
        end
        println(rpad(N, 8), rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    # Speed ratios
    println("\n" * "="^100)
    println("SPEED RATIOS (behavioral layer, mean tick)")
    println("="^100)
    println(rpad("N", 8), rpad("B/A", 12), rpad("C/A", 12), rpad("D/A", 12), rpad("D/C", 12))
    println("-"^60)

    for N in N_RANGE
        means = Dict{String, Float64}()
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            means[label] = length(vals) > 0 ? mean(vals) : NaN
        end
        a = means["A_baseline"]
        b = means["B_lockin_only"]
        c = means["C_normative_only"]
        d = means["D_full_model"]
        println(rpad(N, 8),
                rpad(isnan(a) || isnan(b) ? "N/A" : string(round(a/b, digits=1), "x"), 12),
                rpad(isnan(a) || isnan(c) ? "N/A" : string(round(a/c, digits=1), "x"), 12),
                rpad(isnan(a) || isnan(d) ? "N/A" : string(round(a/d, digits=1), "x"), 12),
                rpad(isnan(c) || isnan(d) ? "N/A" : string(round(c/d, digits=1), "x"), 12))
    end
end

# ── Main ──

function main()
    progress_log("Large-N Sweep: N ∈ $(N_RANGE), $N_TRIALS trials × $(length(CONDITIONS)) conditions")
    progress_log("Output: $OUTPATH")
    t0 = time()

    rows = run_sweep()
    print_summary(rows)

    elapsed = round(time() - t0, digits=1)
    progress_log("Total time: $(elapsed)s")
    println("\nDone.")
end

main()
